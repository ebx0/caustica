"""On-device calibration for the planner (~20 timed engine steps).

:func:`measure_step_time` replays ``run_cw_kspace_pstd``'s step composition
op-for-op with synthetic data (timing/memory only, no physics claim), so the
measured per-step cost pays exactly the FFT mix and elementwise passes of a
real solve. :func:`calibrate` measures >= 2 grid sizes and persists them to
``~/.caustica/calibration.json`` keyed by device name; estimates targeting a
matching device are then labeled ``"calibrated"`` (the ±25% gate applies
to this path, on-device).

BOTH halves of the wall-time model — the per-step cost and the one-time
warmup — are predicted by INTERPOLATING their measured samples
(:func:`predict_step_time`, :func:`calibrated_warmup`) rather than by forcing
a global law through them. The fits (``t_step = a*P*log2(P) + b*P``, and
``warmup = context + per_elem*P``) are still computed and stored — older
stores carry nothing else, and they remain the fallback and a diagnostic —
but on a real device neither is a law. On an RTX PRO 6000 Blackwell
(2026-08-23):

* ``t_step/P`` RISES 1.70x from 216^3 to 432^3 (small grids are
  disproportionately fast), and the two-term fit missed its own three
  saturating probes by 70%;
* warmup is SUPERLINEAR in P (0.069 -> 0.096 -> 0.124 us/element over
  400^3/512^3/640^3), so the linear refit lands on a negative context,
  clamps it to zero, and misses its own smallest sample by 108% — and
  over-predicted a real 192^3 run's warmup by 2x.

The time gate failed that session on the warmup half, not the per-step half.

Works on the numpy backend too (device key ``"cpu"``) — that is how the
mechanics are tested locally before the Colab session.
"""

from __future__ import annotations

import bisect
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np

from caustica.core.backend import get_backend
from caustica.planner.model import fft_sizes, step_time
from caustica.solvers.kspace import operators as ops


def default_calibration_path() -> Path:
    return Path.home() / ".caustica" / "calibration.json"


def device_name(backend: str = "auto") -> str:
    """Stable key for the executing device: GPU product name, or ``"cpu"``."""
    b = get_backend(backend)
    if b.name == "cupy":
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        return props["name"].decode()
    return "cpu"


# Even 2,3,5-smooth sides: the engine pads every axis to the next smooth
# length, so probing a non-smooth side would time a grid that no solve runs.
_SMOOTH_SIDES = (96, 120, 128, 144, 160, 180, 192, 216, 240, 256, 288, 320, 360, 384, 432, 480)

#: Fractions of free VRAM the GPU probe shapes are sized to occupy.
PROBE_VRAM_FRACTIONS = (0.01, 0.04, 0.10)

#: A GPU probe below this many voxels measures kernel-launch latency, not
#: throughput. The first calibration on an A100 timed 48^3 and 72^3 -- 110k
#: and 373k voxels, both ~1.0 ms/step because neither saturates the device.
#: Fitting a*P*log2(P) + b*P through two flat points collapsed to a=0 and a
#: b that could not reproduce its own samples, and extrapolating it to 16.8M
#: voxels over-predicted the step cost by 6.8x (measured, 2026-08-22).
MIN_GPU_PROBE_ELEMS = 2_000_000


def free_device_bytes(backend: str = "auto") -> int | None:
    """Free VRAM on the executing device, or None off the GPU."""
    b = get_backend(backend)
    if b.name != "cupy":
        return None
    import cupy as cp

    free, _total = cp.cuda.Device().mem_info
    return int(free)


def probe_shapes_for_budget(
    free_bytes: int,
    *,
    nonlinear: bool = False,
    ndim: int = 3,
) -> tuple[tuple[int, ...], ...]:
    """GPU probe shapes for a device with ``free_bytes`` of VRAM. Pure.

    Split out from :func:`default_probe_shapes` so the sizing rule can be
    checked for every card the library claims to support without owning one
    -- the same seam the GPU gate suite uses for its ladder.
    """
    from caustica.planner import model  # noqa: PLC0415 (import cycle at module load)

    chosen: list[int] = []
    for frac in PROBE_VRAM_FRACTIONS:
        target = free_bytes * frac
        fit = [
            n
            for n in _SMOOTH_SIDES
            if model.kspace_memory((n,) * ndim, nonlinear, 1, rec_elems=n**ndim).total_bytes
            <= target
        ]
        if fit and fit[-1] not in chosen:
            chosen.append(fit[-1])
    # Never fewer than two sizes (the fit needs leverage), and never a
    # largest probe so small that it measures launch latency instead of
    # throughput -- which is the whole defect this function exists to fix.
    if len(chosen) < 2 or chosen[-1] ** ndim < MIN_GPU_PROBE_ELEMS:
        chosen = [128, 192]
    return tuple((n,) * ndim for n in chosen)


def default_probe_shapes(
    backend: str = "auto",
    *,
    free_bytes: int | None = None,
    nonlinear: bool = False,
    ndim: int = 3,
) -> tuple[tuple[int, ...], ...]:
    """Probe shapes sized for THIS device.

    On the GPU the shapes are chosen to occupy :data:`PROBE_VRAM_FRACTIONS`
    of free VRAM, so a bigger card is probed with bigger grids and the fit
    is made where the device is throughput-bound rather than latency-bound.
    On numpy the probe stays deliberately small: the CPU path exists to
    exercise the mechanics, and the +/-25% gate is about a GPU.
    """
    b = get_backend(backend)
    if b.name != "cupy":
        return ((48,) * ndim, (72,) * ndim)
    budget = free_bytes if free_bytes is not None else free_device_bytes(backend)
    if not budget:
        return ((128,) * ndim, (192,) * ndim)
    return probe_shapes_for_budget(budget, nonlinear=nonlinear, ndim=ndim)


def measure_step_time(
    active_shape: tuple[int, ...],
    nonlinear: bool = False,
    backend: str = "auto",
    n_steps: int = 20,
    warmup: int = 3,
) -> dict:
    """Median per-step wall time (and mempool footprint) on THIS device.

    Returns ``{"device", "backend", "shape", "p_elems", "t_step_s",
    "warmup_s", "vram_peak_bytes"}`` — ``vram_peak_bytes`` is the cupy
    mempool high-water mark (``total_bytes``, i.e. what nvidia-smi attributes
    to the process) and ``None`` on the numpy backend.

    ``warmup_s`` (fix A2) is the ONE-TIME cost this measurement pays before
    the per-step clock is meaningful: device allocation, CUDA context and
    module load on a cold process, cuFFT plan creation and kernel compilation
    for this op mix. It is measured as *everything up to the end of the
    warmup iterations, minus what those iterations should have cost at the
    steady rate* — so a second call in the same process honestly reports
    almost nothing, and the first one carries the cold start. Without this
    term, ``elapsed / steps`` on a short GPU run reads ~26x the steady cost
    and the planner looks broken when it is merely incomplete (measured,
    first Colab session 2026-08-22).
    """
    t_cold0 = time.perf_counter()
    b = get_backend(backend)
    xp, fft = b.xp, b.fft
    padded, p_elems, _ = fft_sizes(tuple(active_shape))
    nd = len(padded)

    rng = np.random.default_rng(0)
    p = xp.asarray(rng.standard_normal(padded, dtype=np.float32))
    u = [xp.zeros(padded, dtype=xp.float32) for _ in range(nd)]
    # The synthetic map has to be CONTRACTIVE, not merely sub-unity: the
    # spectral derivative multiplies by |k|, which reaches pi/dx = 3.1e3 for
    # the dx below, so a 1e-3 coefficient gives a per-step gain above 3 and
    # p reaches float32 overflow inside the timed loop (RuntimeWarning:
    # overflow encountered in multiply, hit on every nonlinear probe). inf
    # and NaN are not a timing hazard on a GPU but are on a CPU, and a probe
    # that overflows cannot be trusted to be timing the arithmetic a real
    # solve does. 1e-4 puts the gain at ~0.31 per step.
    coef = 1e-4
    dt_over_rho = xp.full(padded, coef, dtype=xp.float32)
    rhoc2_dt = xp.full(padded, coef, dtype=xp.float32)
    absorb = xp.full(padded, 0.999, dtype=xp.float32)
    sponge = xp.full(padded, 0.999, dtype=xp.float32)
    beta2_dt = xp.full(padded, coef, dtype=xp.float32) if nonlinear else None
    ks = ops.k_vectors(padded, 1e-3, xp)
    kappa = ops.kappa_sinc(ks, c_ref=1500.0, dt=1e-7, xp=xp)
    deriv = ops.spectral_derivative_factors(ks, kappa, padded, xp)
    del ks, kappa

    pool = None
    if b.name == "cupy":
        import cupy as cp

        pool = cp.get_default_memory_pool()

    def sync() -> None:
        if b.name == "cupy":
            import cupy as cp

            cp.cuda.get_current_stream().synchronize()

    def one_step() -> None:
        # Mirror of engine.step() — keep in lockstep with engine.py.
        pk = fft.rfftn(p)
        for i in range(nd):
            grad_i = fft.irfftn(deriv[i] * pk, s=padded)
            u[i] -= dt_over_rho * grad_i
            u[i] *= absorb
            u[i] *= sponge
        acc = None
        for i in range(nd):
            term = deriv[i] * fft.rfftn(u[i])
            acc = term if acc is None else acc + term
        divu = fft.irfftn(acc, s=padded)
        p_local = p
        if beta2_dt is None:
            p_local -= rhoc2_dt * divu
        else:
            p_local -= (rhoc2_dt + beta2_dt * p_local) * divu
        p_local *= absorb
        p_local *= sponge

    for _ in range(warmup):
        one_step()
    sync()
    t_cold = time.perf_counter() - t_cold0
    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        one_step()
        sync()
        times.append(time.perf_counter() - t0)

    t_step = float(median(times))
    return {
        "device": device_name(backend),
        "backend": b.name,
        "shape": tuple(active_shape),
        "p_elems": p_elems,
        "t_step_s": t_step,
        "warmup_s": max(0.0, t_cold - warmup * t_step),
        "vram_peak_bytes": int(pool.total_bytes()) if pool is not None else None,
    }


def max_rel_residual(samples: list[tuple[int, float]], predict) -> float:
    """Worst relative miss of ``predict`` against the samples it was fitted to.

    A model that cannot reproduce its own measurements will not survive
    extrapolation, and this is the number that says so out loud.
    """
    worst = 0.0
    for p_elems, t in samples:
        if t > 0.0:
            worst = max(worst, abs(predict(p_elems) - t) / t)
    return float(worst)


def fit_max_rel_residual(samples: list[tuple[int, float]], a: float, b: float) -> float:
    """:func:`max_rel_residual` for the ``a*P*log2(P) + b*P`` time model."""
    return max_rel_residual(samples, lambda p: a * p * np.log2(p) + b * p)


def warmup_fit_max_rel_residual(
    samples: list[tuple[int, float]], context_s: float, per_elem_s: float
) -> float:
    """:func:`max_rel_residual` for the two-term warmup model.

    The number that says the linear model is the wrong shape: on the real
    Blackwell warmup triple (400^3 4.389 s, 512^3 12.838 s, 640^3 32.375 s)
    the refit lands on context 0.0 + 1.4264e-7 s/element and misses the
    smallest of its own three samples by 108% (measured, 2026-08-23).
    """
    return max_rel_residual(samples, lambda p: max(0.0, context_s + per_elem_s * p))


def fit_time_model(samples: list[tuple[int, float]]) -> tuple[float, float]:
    """Nonnegative least-squares ``(a, b)`` for ``t = a*P*log2(P) + b*P``."""
    p_arr = np.array([s[0] for s in samples], dtype=np.float64)
    t_arr = np.array([s[1] for s in samples], dtype=np.float64)
    design = np.stack([p_arr * np.log2(p_arr), p_arr], axis=1)
    coef, *_ = np.linalg.lstsq(design, t_arr, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    if a < 0.0:
        a = 0.0
        b = float(np.linalg.lstsq(design[:, 1:], t_arr, rcond=None)[0][0])
    elif b < 0.0:
        b = 0.0
        a = float(np.linalg.lstsq(design[:, :1], t_arr, rcond=None)[0][0])
    return max(a, 0.0), max(b, 0.0)


#: Above this, the fit is not reproducing its own samples and extrapolating
#: it upward is guesswork; the throughput-anchored fallback takes over.
#: The same limit grades the interpolator, via its leave-one-out error.
FIT_RESIDUAL_LIMIT = 0.10


def _dedupe(samples: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Sorted by ``P``, one time per size — the SLOWER one when a size was
    measured twice. Two timings of one shape disagree only by noise, and the
    planner's job is not to be optimistic."""
    by_p: dict[int, float] = {}
    for p_elems, t in samples:
        key = int(p_elems)
        by_p[key] = max(by_p.get(key, 0.0), float(t))
    return sorted(by_p.items())


def entry_samples(entry: dict) -> list[tuple[int, float]]:
    """The ``(P, t_step)`` measurements an entry carries, sorted by ``P``.

    Tolerant of what is actually on disk: JSON pairs, missing key, empty
    list, junk rows, non-positive or non-finite times.
    """
    clean: list[tuple[int, float]] = []
    for row in entry.get("samples") or ():
        try:
            p_elems, t = int(row[0]), float(row[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if p_elems > 0 and t > 0.0 and math.isfinite(t):
            clean.append((p_elems, t))
    return _dedupe(clean)


def _monotone_knots(samples: list[tuple[int, float]]) -> tuple[list[float], list[float]]:
    """Sorted knots with a running max over ``t``: a bigger grid is never
    predicted cheaper than a smaller one.

    A falling measurement is noise (or a latency-bound probe), not a device
    that speeds up as the problem grows — the same judgment
    :func:`record_warmup_model` makes about a falling warmup. The first A100
    calibration measured exactly that: 1.061 ms at 110,592 elements and
    1.001 ms at 373,248.
    """
    ps: list[float] = []
    ts: list[float] = []
    running = 0.0
    for p_elems, t in _dedupe(samples):
        running = max(running, t)
        ps.append(float(p_elems))
        ts.append(running)
    return ps, ts


def _interp_loglog(samples: list[tuple[int, float]], p_elems: int, *, below: str) -> float:
    """Monotone piecewise-linear interpolation of log(cost) against log(P).

    One machine, two cost curves. Between knots it is a local power law, so
    the measurements are reproduced exactly and there is nothing for a global
    law to miss. Above the largest sample it continues the LAST segment's
    slope, but never predicts a better per-element rate than that sample
    measured: the slope was measured over a finite range, and continued far
    enough it would claim a throughput the device has never demonstrated.
    Extrapolation may assume the card stays as fast as its best measurement.
    It may not assume it gets faster.

    ``below`` is where the two curves genuinely differ, and each rule is a
    measurement rather than a taste:

    * ``"value"`` (per-step time) — hold the smallest sample's TIME. A step
      has a hard floor: the launch and the fixed per-kernel cost. The first
      A100 probe measured 1.061 ms at 110,592 elements and 1.001 ms at
      373,248 — 3.4x the work for the same time. Continuing a log-log slope
      downward through that regime predicts absurdly fast steps on grids that
      are latency-bound in practice.
    * ``"rate"`` (warmup) — hold the smallest sample's PER-ELEMENT rate. Plan
      creation has no such floor; it shrinks with the transform. Checked
      against the one measurement below the sampled range: from warmups at
      400^3/512^3/640^3 this rule predicts 0.485 s for 192^3 where a real run
      paid 0.501 s (-3.1%). Holding the VALUE instead would have said 4.389 s
      (+776%), and the linear two-term model said 1.010 s (+101%).
    """
    if not samples:
        raise ValueError("interpolation needs at least one (P, cost) sample")
    ps, ts = _monotone_knots(samples)
    p = float(p_elems)
    if p <= ps[0] or len(ps) == 1:
        return ts[0] if below == "value" else ts[0] * (p / ps[0])
    if p >= ps[-1]:
        slope = (math.log(ts[-1]) - math.log(ts[-2])) / (math.log(ps[-1]) - math.log(ps[-2]))
        if not math.isfinite(slope):
            slope = 1.0
        extrapolated = ts[-1] * (p / ps[-1]) ** slope
        best_rate_floor = ts[-1] * (p / ps[-1])  # == (cost_max / P_max) * P, exact at the knot
        return max(extrapolated, best_rate_floor)
    i = min(bisect.bisect_right(ps, p) - 1, len(ps) - 2)
    slope = (math.log(ts[i + 1]) - math.log(ts[i])) / (math.log(ps[i + 1]) - math.log(ps[i]))
    return ts[i] * (p / ps[i]) ** slope


def interp_step_time(samples: list[tuple[int, float]], p_elems: int) -> float:
    """Per-step time at ``p_elems``, interpolated through the measurements.

    The calibrated path's predictor, because a device's per-element step cost
    is a *curve*: on an RTX PRO 6000 Blackwell it RISES 1.70x from 216^3
    (0.217 ns/element) to 432^3 (0.370 ns/element) — small grids are
    disproportionately fast — and the global ``a*P*log2(P) + b*P`` fit that
    has to cross all of that missed its own probes by 70% (measured,
    2026-08-23). See :func:`_interp_loglog` for the out-of-range rules.
    """
    return _interp_loglog(samples, p_elems, below="value")


def interp_warmup(samples: list[tuple[int, float]], p_elems: int) -> float:
    """One-time cost at ``p_elems``, interpolated through real runs' warmups.

    Same medicine as :func:`interp_step_time`, for the same reason: the
    Blackwell's measured warmups are SUPERLINEAR in P (0.069, 0.096 and
    0.124 microseconds per element at 400^3, 512^3, 640^3), and the linear
    ``context + per_elem * P`` model refitted on them lands on a negative
    context — clamped to zero — and misses its own smallest sample by 108%.
    """
    return max(0.0, _interp_loglog(samples, p_elems, below="rate"))


def interp_loo_rel_err(samples: list[tuple[int, float]], *, below: str = "value") -> float | None:
    """Worst leave-one-out miss of the interpolator, or None.

    The honest quality number for an interpolator. Its residual against the
    samples it passes through is 0 by construction — reporting THAT would be
    self-flattery — so each interior sample is predicted from the others
    instead, which is exactly what the planner does for a grid size that was
    never probed. None when there is no interior sample (< 3 probes): with
    two, there is nothing to hold out and nothing honest to claim.

    ``below`` never actually applies (a held-out interior sample is inside
    what remains), and is carried only so the two curves cannot drift apart.
    """
    pts = _dedupe(samples)
    if len(pts) < 3:
        return None
    worst = 0.0
    for i in range(1, len(pts) - 1):
        p_elems, t = pts[i]
        pred = _interp_loglog(pts[:i] + pts[i + 1 :], p_elems, below=below)
        worst = max(worst, abs(pred - t) / t)
    return float(worst)


def predict_step_time(entry: dict, p_elems: int) -> float:
    """Per-step time a calibration entry predicts for a ``p_elems`` grid.

    Interpolation wins whenever the entry carries >= 2 samples. Entries that
    carry only ``(a, b)`` — everything written before 2026-08-23 — keep the
    global two-term model, byte for byte.
    """
    samples = entry_samples(entry)
    if len(samples) >= 2:
        return interp_step_time(samples, p_elems)
    return step_time(float(entry["a"]), float(entry["b"]), p_elems)


def time_model_quality(entry: dict) -> tuple[str, float | None]:
    """``(mode, worst_rel_err)`` for whatever drives this entry's prediction.

    ``"interpolated"`` reports leave-one-out error (computed on the spot when
    an older store did not record it); ``"fit"`` reports the global model's
    residual against its own samples. Two different numbers, and pairing the
    wrong one with the mode would either flatter the interpolator (residual
    0 by construction) or condemn it for a defect it does not have.
    """
    samples = entry_samples(entry)
    if len(samples) >= 2:
        loo = entry.get("interp_loo_rel_err")
        if loo is None:
            loo = interp_loo_rel_err(samples)
        return "interpolated", (float(loo) if loo is not None else None)
    residual = entry.get("fit_max_rel_residual")
    return "fit", (float(residual) if residual is not None else None)


def calibrate(
    shapes: tuple[tuple[int, ...], ...] | None = None,
    *,
    nonlinear: bool = False,
    backend: str = "auto",
    n_steps: int = 20,
    path: str | Path | None = None,
) -> dict:
    """Measure ``shapes`` on this device, persist the samples, return entry.

    The samples ARE the model: :func:`predict_step_time` interpolates them.
    ``(a, b)`` is fitted and stored alongside for older readers and as a
    diagnostic — ``fit_max_rel_residual`` says how badly one global law has
    to miss this device's curve, and ``interp_loo_rel_err`` says how well the
    interpolator does on a probe it was not shown.

    ``shapes=None`` (the default) sizes the probe for the executing device
    via :func:`default_probe_shapes` -- a fixed pair was the reason the first
    A100 calibration measured only launch latency.

    The entry lands in ``calibration.json`` under the device name;
    :func:`caustica.planner.estimate` picks it up automatically when the
    ``gpu=`` argument matches the device.
    """
    if shapes is None:
        shapes = default_probe_shapes(backend, nonlinear=nonlinear)
    if len(shapes) < 2:
        raise ValueError("calibration needs >= 2 grid sizes to fit a*N*log2(N) + b*N")
    runs = [
        measure_step_time(s, nonlinear=nonlinear, backend=backend, n_steps=n_steps) for s in shapes
    ]
    samples = [(r["p_elems"], r["t_step_s"]) for r in runs]
    a, b = fit_time_model(samples)
    residual = fit_max_rel_residual(samples, a, b)
    global_fit_mode = "nnls"
    if residual > FIT_RESIDUAL_LIMIT:
        # The nonnegativity clip has left a model that misses its own
        # measurements. Anchor on the LARGEST sample instead: it is the most
        # throughput-bound one, which is the regime every extrapolation from
        # here is heading into.
        p_big, t_big = max(samples, key=lambda s: s[0])
        a, b = 0.0, t_big / p_big
        global_fit_mode = "throughput-anchored"
        residual = fit_max_rel_residual(samples, a, b)
    # (a, b) is kept for backward compatibility and as a diagnostic, but it
    # does not predict any more: with >= 2 samples on file, estimate() reads
    # them through interp_step_time(). fit_max_rel_residual stays the GLOBAL
    # model's number -- the interpolator's is leave-one-out, below.
    loo = interp_loo_rel_err(samples)

    # Warmup is not a constant: it is a per-process cost (CUDA context,
    # module load) plus a per-shape cost (cuFFT plans, kernel specialization)
    # that scales with the padded element count. Only the FIRST probe in this
    # process pays the context, so the later ones isolate the per-element
    # term. Checked against the first gate session: fitting on the 256^3 and
    # 400^3 rungs predicts the 512^3 rung's 20.9 s warmup to +4.3%.
    later = runs[1:]
    per_elem = min(float(r["warmup_s"]) / max(r["p_elems"], 1) for r in later) if later else 0.0
    context_s = max(0.0, float(runs[0]["warmup_s"]) - per_elem * runs[0]["p_elems"])
    entry = {
        "a": a,
        "b": b,
        "fit_mode": "interpolated",
        "global_fit_mode": global_fit_mode,
        "fit_max_rel_residual": residual,
        "interp_loo_rel_err": loo,
        "p_max_probe": max(p for p, _ in samples),
        "warmup_context_s": context_s,
        "warmup_per_elem_s": per_elem,
        # The COLD start, i.e. the largest warmup any of the shapes paid: the
        # first measurement carries the context/module load, the ones after it
        # in the same process do not, and it is the first that a fresh solve
        # looks like (fix A2). record_warmup() replaces this with what a real
        # run actually paid once the validation suite has measured one.
        "warmup_s": max(float(r["warmup_s"]) for r in runs),
        "warmup_source": "probe",
        "backend": runs[0]["backend"],
        "n_steps": n_steps,
        "nonlinear": nonlinear,
        "samples": [[int(p), float(t)] for p, t in samples],
        "shapes": ["x".join(map(str, r["shape"])) for r in runs],
        "vram_peak_bytes_by_shape": {
            "x".join(map(str, r["shape"])): r["vram_peak_bytes"] for r in runs
        },
        "calibrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    target = Path(path) if path is not None else default_calibration_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(target.read_text()) if target.exists() else {"version": 1, "devices": {}}
    data["devices"][runs[0]["device"]] = entry
    target.write_text(json.dumps(data, indent=2))
    return entry


#: A plan this far above the largest probed size is extrapolating; say so.
EXTRAPOLATION_WARN_FACTOR = 8.0


def entry_warmup_samples(entry: dict) -> list[tuple[int, float]]:
    """The ``(P, warmup_s)`` measurements an entry carries, sorted by ``P``.

    Written by :func:`record_warmup_model` from whole runs — the gate suite's
    ladder — not by the probe.
    """
    clean: list[tuple[int, float]] = []
    for row in entry.get("warmup_samples") or ():
        try:
            p_elems, w = int(row[0]), float(row[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if p_elems > 0 and w > 0.0 and math.isfinite(w):
            clean.append((p_elems, w))
    return _dedupe(clean)


def calibrated_warmup(entry: dict, p_elems: int, default: float = 0.0) -> float:
    """One-time cost this entry predicts for a run of ``p_elems`` elements.

    Interpolates the measured warmups when the entry carries >= 2 of them
    (:func:`interp_warmup`) — the same fix, for the same reason, as the
    per-step curve: warmup is superlinear in P on a real card and the linear
    two-term model cannot bend that way. Falls back to that two-term model
    (context + per-element), then to the flat ``warmup_s`` that pre-fix-A2
    entries have, then to ``default``.
    """
    samples = entry_warmup_samples(entry)
    if len(samples) >= 2:
        return interp_warmup(samples, p_elems)
    ctx, per = entry.get("warmup_context_s"), entry.get("warmup_per_elem_s")
    if ctx is not None and per is not None:
        return max(0.0, float(ctx) + float(per) * p_elems)
    flat = entry.get("warmup_s")
    return float(flat) if flat is not None else default


def warmup_model_quality(entry: dict) -> tuple[str, float | None]:
    """``(mode, worst_rel_err)`` for whatever drives this entry's warmup.

    The warmup twin of :func:`time_model_quality`: ``"interpolated"`` reports
    leave-one-out error, ``"fit"`` the two-term model's residual against the
    samples it was fitted to (None when the entry never recorded any).
    """
    samples = entry_warmup_samples(entry)
    if len(samples) >= 2:
        loo = entry.get("warmup_loo_rel_err")
        if loo is None:
            loo = interp_loo_rel_err(samples, below="rate")
        return "interpolated", (float(loo) if loo is not None else None)
    residual = entry.get("warmup_fit_max_rel_residual")
    return "fit", (float(residual) if residual is not None else None)


def record_warmup(
    device: str,
    warmup_s: float,
    *,
    p_elems: int | None = None,
    source: str = "measured",
    path: str | Path | None = None,
) -> dict | None:
    """Teach the calibration what a REAL run's warmup cost on this device.

    The probe in :func:`measure_step_time` replays the step composition, not
    a whole solve: it never builds the medium's property maps, never touches
    the source scatter, and so it under-counts the one-time cost of a real
    run. ``caustica.validation``'s GPU-gate suite measures that number from
    an actual stamped run and calls this to write it back — which is what
    "the warmup is measured on the device and stored in the calibration"
    means in practice (fix A2).

    Returns the updated entry, or None when the device has no calibration
    yet (there is nothing to attach to, and inventing one would let the
    planner label a datasheet guess "calibrated").
    """
    target = Path(path) if path is not None else default_calibration_path()
    data = load_calibration(target)
    entry = data.get("devices", {}).get(device)
    if entry is None:
        return None
    measured = max(0.0, float(warmup_s))
    entry["warmup_s"] = measured
    entry["warmup_source"] = source
    # The caller is asserting ONE number. Older samples would outrank it --
    # calibrated_warmup interpolates them when they exist -- so they go: a
    # stale curve that contradicts a fresh measurement is worse than no curve.
    entry.pop("warmup_loo_rel_err", None)
    if p_elems:
        # A real run at a KNOWN size: keep the fitted per-element slope and
        # move the constant term to match what was actually paid.
        per = float(entry.get("warmup_per_elem_s") or 0.0)
        entry["warmup_context_s"] = max(0.0, measured - per * int(p_elems))
        entry["warmup_per_elem_s"] = per
        entry["warmup_samples"] = [[int(p_elems), measured]]  # one point: not a curve yet
    else:
        # No size context: the caller is asserting a flat number, so make
        # the two-term model reduce to exactly that rather than leaving a
        # stale slope to contradict it.
        entry["warmup_context_s"] = measured
        entry["warmup_per_elem_s"] = 0.0
        entry.pop("warmup_samples", None)
    entry["warmup_recorded_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2))
    return entry


def record_warmup_model(
    device: str,
    samples: list[tuple[int, float]],
    *,
    source: str = "measured",
    path: str | Path | None = None,
) -> dict | None:
    """Store REAL runs' warmups, and refit context + per_elem * P from them.

    Strictly better data than the probe: these are whole solves, at several
    sizes, each in its own process, so every one of them paid the CUDA
    context AND its own plan creation -- which is exactly what the two terms
    mean. With a single sample only the constant can move, and the probe's
    slope is kept.

    The SAMPLES are what predicts (:func:`calibrated_warmup` interpolates
    them); the two-term fit is kept as the fallback for entries that have
    none, and its residual is kept as the evidence that it is the wrong
    shape. On the Blackwell's real triple it misses by 108%.

    Returns the updated entry, or None when the device has no calibration
    (inventing one would let the planner call a datasheet guess
    "calibrated").
    """
    target = Path(path) if path is not None else default_calibration_path()
    data = load_calibration(target)
    entry = data.get("devices", {}).get(device)
    if entry is None or not samples:
        return None

    if len(samples) >= 2:
        p_arr = np.array([float(p) for p, _ in samples], dtype=np.float64)
        w_arr = np.array([float(w) for _, w in samples], dtype=np.float64)
        slope, const = np.polyfit(p_arr, w_arr, 1)
        if slope < 0.0:  # a falling warmup is noise, not a model
            slope, const = 0.0, float(w_arr.mean())
        entry["warmup_per_elem_s"] = max(0.0, float(slope))
        entry["warmup_context_s"] = max(0.0, float(const))
    else:
        p_elems, w = samples[0]
        per = float(entry.get("warmup_per_elem_s") or 0.0)
        entry["warmup_per_elem_s"] = per
        entry["warmup_context_s"] = max(0.0, float(w) - per * int(p_elems))

    entry["warmup_s"] = max(0.0, max(w for _, w in samples))
    entry["warmup_source"] = source
    entry["warmup_samples"] = [[int(p), float(w)] for p, w in samples]
    entry["warmup_mode"] = "interpolated" if len(samples) >= 2 else "fit"
    entry["warmup_fit_max_rel_residual"] = warmup_fit_max_rel_residual(
        [(int(p), float(w)) for p, w in samples],
        float(entry["warmup_context_s"]),
        float(entry["warmup_per_elem_s"]),
    )
    entry["warmup_loo_rel_err"] = interp_loo_rel_err(
        [(int(p), float(w)) for p, w in samples], below="rate"
    )
    entry["warmup_recorded_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2))
    return entry


def load_calibration(path: str | Path | None = None) -> dict:
    target = Path(path) if path is not None else default_calibration_path()
    if not target.exists():
        return {"version": 1, "devices": {}}
    return json.loads(target.read_text())


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_calibration_for(
    gpu_key: str,
    path: str | Path | None = None,
    *,
    device_name: str | None = None,
) -> dict | None:
    """Best calibration entry whose device name matches a gpu_db key.

    Matching is token-based: the key's primary token (e.g. ``a100`` of
    ``A100-40GB``) must appear in the normalized device name
    (``NVIDIA A100-SXM4-40GB``); further tokens break ties. ``cpu`` only
    matches the cpu entry.

    ``device_name`` short-circuits all of that with an exact match on the
    live product string. Without it, a card outside ``gpu_db.json`` cannot
    reach its own calibration: the RTX PRO 6000 Blackwell measured one and
    then planned from the datasheet anyway, because the lookup was hunting
    for "a100" in a name that has never contained it (2026-08-22).
    """
    devices = load_calibration(path).get("devices", {})
    if device_name is not None:
        target = _norm(device_name)
        for name, entry in devices.items():
            if _norm(name) == target:
                return entry
    tokens = [_norm(t) for t in gpu_key.split("-") if t]
    if not tokens:
        return None
    best: tuple[int, dict] | None = None
    for name, entry in devices.items():
        name_n = _norm(name)
        if tokens[0] == "cpu":
            if name_n != "cpu":
                continue
            hits = 1
        else:
            if name_n == "cpu" or tokens[0] not in name_n:
                continue
            hits = sum(1 for t in tokens if t in name_n)
        if best is None or hits > best[0]:
            best = (hits, entry)
    return best[1] if best is not None else None
