"""Paired timing of the k-space step: the engine's closure, the planner's probe.

Why a script and not a test: on a laptop GPU the round-to-round scatter of a
single timed block is tens of percent (the RTX 5050 in this tree drifts its
clocks within one process and, under sustained load, drops to a third of its
cold rate and stays there), so an absolute millisecond figure is a sample and
not a value. Every number here is therefore a RATIO taken inside one round,
between two things timed back to back in one process with the fields reseeded
before each timed block, and the reported figure is the median across rounds
with the round-to-round spread beside it. Cross-process comparison of absolute
milliseconds at this scale is not measurement.

Two modes:

``closures``
    Times the engine's own per-step closure (captured out of a real solve),
    the closure the calibration probe builds, and a reconstruction of the
    two-pass replica the probe carried before the damping fusion. Answers
    "does the probe drive the same arithmetic as the engine".

``acceptance``
    Times what ``measure_step_time`` actually RETURNS against a real solve's
    marginal wall time per step, taken from the progress callback's span
    between period boundaries. Answers the question the T0.9 brief asks, and
    it is a different question: the probe deliberately omits the source
    injection, the convergence reduction and the harmonic accumulation, so
    this ratio carries that omission and the closure ratio does not.

Usage (in-tree interpreter, GPU):

    ./.venv/Scripts/python.exe scripts/dev_step_timing.py closures --shape 128
    ./.venv/Scripts/python.exe scripts/dev_step_timing.py acceptance --shape 192
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caustica.core.backend import get_backend  # noqa: E402
from caustica.core.grid import Grid  # noqa: E402
from caustica.materials import water  # noqa: E402
from caustica.medium import Medium  # noqa: E402
from caustica.planner import calibration as C  # noqa: E402
from caustica.planner.model import fft_sizes  # noqa: E402
from caustica.solvers import CWRunSpec, get  # noqa: E402
from caustica.solvers.kspace import engine as E  # noqa: E402
from caustica.solvers.kspace import operators as ops  # noqa: E402
from caustica.sources import CWSource  # noqa: E402

DX = 0.25e-3
F0 = 1e6


def _sync(backend_name: str) -> None:
    if backend_name == "cupy":
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()


def _source(shape: tuple[int, ...]) -> CWSource:
    c = shape[0] // 2
    return CWSource(
        indices=np.array([[c, c, shape[2] // 6]], dtype=np.int32),
        phases=np.zeros(1, dtype=np.float32),
        amplitude=1e4,
        f0=F0,
    )


def _run_solve(shape, backend, periods, progress=None, capture=None):
    """One real solve. ``capture`` collects the factory's kwargs and closure."""
    grid = Grid(shape=shape, dx=DX, pml=None)
    med = Medium.homogeneous(grid.shape, water())
    spec = CWRunSpec(
        min_settle_periods=periods,
        max_settle_periods=periods,
        convergence_tol=0.0,
        n_record_periods=1,
    )
    real = E.make_propagation_step
    if capture is not None:

        def counted(**kw):
            closure = real(**kw)
            capture.append((kw, closure))
            return closure

        E.make_propagation_step = counted
    try:
        get("linear")().run(grid, med, _source(shape), spec, backend=backend, progress=progress)
    finally:
        E.make_propagation_step = real


# ------------------------------------------------------------------ closures


def _probe_arrays(padded, xp, nonlinear=False):
    """The synthetic state ``measure_step_time`` builds, kept in one place."""
    nd = len(padded)
    rng = np.random.default_rng(0)
    p = xp.asarray(rng.standard_normal(padded, dtype=np.float32))
    u = [xp.zeros(padded, dtype=xp.float32) for _ in range(nd)]
    coef = 1e-4
    maps = {
        "dt_over_rho": xp.full(padded, coef, dtype=xp.float32),
        "rhoc2_dt": xp.full(padded, coef, dtype=xp.float32),
        "beta2_dt": xp.full(padded, coef, dtype=xp.float32) if nonlinear else None,
    }
    ks = ops.k_vectors(padded, 1e-3, xp)
    kappa = ops.kappa_sinc(ks, c_ref=1500.0, dt=1e-7, xp=xp)
    deriv = ops.spectral_derivative_factors(ks, kappa, padded, xp)
    del ks, kappa
    return p, u, deriv, maps


def _two_pass_replica(*, fft, padded, p, u, deriv, absorb, sponge, dt_over_rho, rhoc2_dt):
    """The step the probe carried BEFORE the damping fusion: two multiplies."""
    nd = len(padded)
    axes = tuple(range(nd))

    def propagate() -> None:
        pk = fft.rfftn(p)
        for i in range(nd):
            grad_i = fft.irfftn(deriv[i] * pk, s=padded, axes=axes)
            u[i] -= dt_over_rho * grad_i
            u[i] *= absorb
            u[i] *= sponge
        acc = None
        for i in range(nd):
            term = deriv[i] * fft.rfftn(u[i])
            acc = term if acc is None else acc + term
        divu = fft.irfftn(acc, s=padded, axes=axes)
        p_local = p
        p_local -= rhoc2_dt * divu
        p_local *= absorb
        p_local *= sponge

    return propagate


def _quartiles(v):
    """Lower and upper quartile of a sorted list, by the median of halves."""
    n = len(v)
    if n < 2:
        return v[0], v[0]
    lo, hi = v[: n // 2], v[(n + 1) // 2 :]
    return statistics.median(lo), statistics.median(hi)


def _timed(step, state, seed, n, backend_name):
    """One timed block: reseed the fields, run ``n`` steps, return seconds."""
    p, u = state
    p[...] = seed[0]
    for i, c in enumerate(u):
        c[...] = seed[1][i]
    _sync(backend_name)
    t0 = time.perf_counter()
    for _ in range(n):
        step()
    _sync(backend_name)
    return time.perf_counter() - t0


def mode_closures(shape, backend, rounds, steps):
    b = get_backend(backend)
    xp, fft = b.xp, b.fft
    padded, _, _ = fft_sizes(shape)

    capture: list = []
    _run_solve(shape, backend, periods=1, capture=capture)
    kw, engine_step = capture[0]
    engine_state = (kw["p"], kw["u"])

    p2, u2, deriv2, maps2 = _probe_arrays(padded, xp)
    damp2 = xp.full(padded, np.float32(0.999 * 0.999), dtype=xp.float32)
    shared_step = E.make_propagation_step(
        fft=fft, padded=padded, p=p2, u=u2, deriv=deriv2, damp=damp2, **maps2
    )

    p3, u3, deriv3, maps3 = _probe_arrays(padded, xp)
    maps3.pop("beta2_dt")
    old_step = _two_pass_replica(
        fft=fft,
        padded=padded,
        p=p3,
        u=u3,
        deriv=deriv3,
        absorb=xp.full(padded, np.float32(0.999), dtype=xp.float32),
        sponge=xp.full(padded, np.float32(0.999), dtype=xp.float32),
        **maps3,
    )

    entries = [
        ("engine", engine_step, engine_state),
        ("shared", shared_step, (p2, u2)),
        ("replica", old_step, (p3, u3)),
    ]
    seeds = {}
    for name, _, (p, u) in entries:
        seeds[name] = (p.copy(), [c.copy() for c in u])

    ratios: dict[str, list[float]] = {"shared": [], "replica": []}
    absolute: dict[str, list[float]] = {n: [] for n, _, _ in entries}
    for r in range(rounds):
        order = list(entries) if r % 2 == 0 else list(reversed(entries))
        order = order[r % len(order) :] + order[: r % len(order)]
        got = {}
        for name, step, state in order:
            got[name] = _timed(step, state, seeds[name], steps, b.name) / steps
        for name in ratios:
            ratios[name].append(got[name] / got["engine"] - 1.0)
        for name in absolute:
            absolute[name].append(got[name])

    print(f"shape {shape} backend {b.name} rounds {rounds} steps/block {steps}")
    for name in ("engine", "shared", "replica"):
        v = [x * 1e3 for x in absolute[name]]
        print(
            f"  {name:8s} ms/step median {statistics.median(v):7.3f}  "
            f"min {min(v):7.3f}  max {max(v):7.3f}"
        )
    for name in ("shared", "replica"):
        v = sorted(x * 100.0 for x in ratios[name])
        q1, q3 = _quartiles(v)
        print(
            f"  {name:8s} vs engine: median {statistics.median(v):+6.2f} %  "
            f"quartiles {q1:+.2f} % to {q3:+.2f} %  "
            f"round spread {min(v):+.2f} % to {max(v):+.2f} %"
        )
    return ratios, absolute


# ---------------------------------------------------------------- acceptance


def _real_ms_per_step(shape, backend, periods):
    """A real solve's marginal wall time per step, from the progress span."""
    marks: list[tuple[int, float]] = []

    def progress(payload):
        if payload["stage"] == "settle":
            marks.append((payload["step"], payload["elapsed_s"]))

    _run_solve(shape, backend, periods=periods, progress=progress)
    if len(marks) < 3:
        raise SystemExit(
            f"only {len(marks)} settle period boundaries; raise --periods so the "
            f"span is long enough to divide"
        )
    # Drop the first boundary: it carries the plan build and the first-touch
    # allocations, which a per-step figure must not inherit.
    (s0, t0), (s1, t1) = marks[1], marks[-1]
    return (t1 - t0) / (s1 - s0) * 1e3


def mode_acceptance(shape, backend, rounds, periods, first="probe"):
    ratios = []
    probe_ms = []
    run_ms = []
    for r in range(rounds):
        if (r % 2 == 0) == (first == "probe"):
            pr = C.measure_step_time(shape, backend=backend)["t_step_s"] * 1e3
            rn = _real_ms_per_step(shape, backend, periods)
        else:
            rn = _real_ms_per_step(shape, backend, periods)
            pr = C.measure_step_time(shape, backend=backend)["t_step_s"] * 1e3
        probe_ms.append(pr)
        run_ms.append(rn)
        ratios.append(pr / rn - 1.0)
        print(f"  round {r}: probe {pr:7.3f} ms  run {rn:7.3f} ms  {ratios[-1] * 100:+6.2f} %")
    v = sorted(x * 100.0 for x in ratios)
    q1, q3 = _quartiles(v)
    print(f"shape {shape} backend {backend} rounds {rounds} periods/run {periods}")
    print(f"  probe ms/step median {statistics.median(probe_ms):7.3f}")
    print(f"  run   ms/step median {statistics.median(run_ms):7.3f}")
    print(
        f"  probe vs run: median {statistics.median(v):+6.2f} %  "
        f"quartiles {q1:+.2f} % to {q3:+.2f} %  "
        f"round spread {min(v):+.2f} % to {max(v):+.2f} %"
    )
    return ratios


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["closures", "acceptance"])
    ap.add_argument("--shape", type=int, default=128)
    ap.add_argument("--backend", default="cupy")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--steps", type=int, default=30, help="timed steps per block (closures)")
    ap.add_argument("--periods", type=int, default=12, help="settle periods per run (acceptance)")
    ap.add_argument(
        "--first",
        choices=["probe", "run"],
        default="probe",
        help="which side round 0 times first (acceptance)",
    )
    a = ap.parse_args()
    shape = (a.shape,) * 3
    if a.mode == "closures":
        mode_closures(shape, a.backend, a.rounds, a.steps)
    else:
        mode_acceptance(shape, a.backend, a.rounds, a.periods, a.first)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
