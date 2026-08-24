"""Shared CW k-space PSTD engine used by the `linear` and `westervelt` solvers.

One implementation, one set of numerics: the two native solvers differ ONLY
in whether the Westervelt nonlinear term enters the pressure update. Keeping
a single engine guarantees the M5 gate "beta=0 => westervelt == linear"
structurally (identical code path when the medium is linear) and gives the
future GPU work (M7) a single surface to optimize.

Scheme per time step (dt), all states float32 on the chosen backend:

    u_i <- (u_i - dt/rho * IFFT{ i k_i kappa FFT{p} }) * e^{-alpha c dt} * sponge
    p   <- (p - (rho c^2 dt + 2 beta dt p) * IFFT{ kappa sum_i i k_i FFT{u_i} })
           * e^{-alpha c dt} * sponge
    p[src] += (2 c dt / dx) * p0 * ramp(t) * sin(omega t - phase)

The (2 c dt / dx) factor is the k-Wave-style mass-source normalization: the
realized plane amplitude is ~p0 (few-% residual), independent of grid,
CFL, and remote medium content. Recorded phasors follow the library-wide
convention p(t) = Re{P exp(-i omega t)} (outgoing = e^{+ikx}), matching
the analytic references.

The nonlinear term is the notebook's validated form: dp_nl = -2 beta dt p
(div u), i.e. the local Westervelt correction to the equation of state.
Absorption is applied symmetrically to p and u (spatial decay = alpha; see
the M4 devlog entry for the alpha/2 pitfall this avoids).

Steady-state harmonics: the record window accumulates a leakage-free
single-bin DFT at n*f0 for every requested harmonic n (exact-period dt makes
all integer harmonics leakage-free in one pass).
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable
from itertools import product
from math import ceil, floor, isfinite

import numpy as np

from caustica.core.backend import CausticaWarning, get_backend
from caustica.core.grid import Grid
from caustica.io.checkpoint import (
    CheckpointSpec,
    RunInterrupted,
    arrays_sha1,
    load_checkpoint,
    write_checkpoint,
)
from caustica.medium import Medium
from caustica.solvers.base import (
    CWRunSpec,
    SolverDivergedError,
    SolverResult,
    interior_slices,
)
from caustica.solvers.kspace import operators as ops
from caustica.sources import CWSource, ramp_envelope

log = logging.getLogger("caustica")

#: Which numerics produced a field. Bumped whenever a change moves the
#: trajectory this engine follows, so a checkpoint from an older one is
#: refused rather than spliced and a stored result says which generation it
#: came from. /2 zeroed the Nyquist wavenumber (2026-08-24); /3 gave sources
#: per-voxel drive weights (2026-08-24).
NUMERICS_SCHEME = "cw-kspace-pstd/3"


def normalize_record_region(region: tuple[slice, ...], shape: tuple[int, ...]) -> tuple[slice, ...]:
    """Resolve record-region slices against the ACTIVE grid shape.

    The pressure array lives on the FFT-padded shape, so open-ended or
    negative slices must be resolved against the active shape FIRST —
    ``p[slice(None)]`` on a padded axis would silently include pad voxels
    (review finding, 2026-08-11).
    """
    if len(region) != len(shape):
        raise ValueError(f"record_region rank {len(region)} != grid rank {len(shape)}")
    out = []
    for sl, n_ax in zip(region, shape, strict=True):
        start, stop, step = sl.indices(n_ax)
        if step != 1:
            raise ValueError(f"record_region does not support strides, got {sl}")
        if stop <= start:
            raise ValueError(f"record_region slice {sl} is empty on an axis of {n_ax}")
        out.append(slice(start, stop))
    return tuple(out)


def cw_discretization(
    grid: Grid,
    medium: Medium,
    spec: CWRunSpec,
    f0: float,
    harmonics: tuple[int, ...] = (1,),
) -> tuple[int, float]:
    """Exact-period ``(spp, dt)`` under the CFL limit (notebook policy).

    Single source of truth shared by the engine and the planner (M8): the
    planner MUST predict the same dt the engine will use, or its step-count
    and VRAM figures drift from reality. Raises on temporal-Nyquist violation
    for the requested harmonics (h*f0 must sit strictly below spp*f0/2).
    """
    period = 1.0 / f0
    dt_cfl = spec.cfl * grid.dx / medium.c_max
    spp = max(2, int(floor(period / dt_cfl)))
    dt = period / spp
    while dt * medium.c_max / grid.dx > spec.cfl_hard_max + 1e-12 and spp < 1024:
        spp += 1
        dt = period / spp
    h_max = max(harmonics)
    if 2 * h_max >= spp:
        raise ValueError(
            f"harmonic {h_max} is at/above the temporal Nyquist for spp={spp} "
            f"(need 2*h < spp); refine dx or lower cfl to raise spp."
        )
    return spp, dt


def cw_tof_periods(
    grid: Grid,
    medium: Medium,
    source: CWSource,
    reference_point: tuple[int, ...] | None = None,
) -> int:
    """Periods of time-of-flight from the source to the farthest reference.

    With no explicit reference the wave must reach every domain corner
    (conservative default — matches the engine's settling floor).
    """
    period = 1.0 / source.f0
    src_pos = source.indices.astype(np.float64) * grid.dx
    if reference_point is not None:
        refs = np.asarray(reference_point, np.float64)[None, :] * grid.dx
    else:
        refs = np.array(list(product(*((0, n - 1) for n in grid.shape))), np.float64) * grid.dx
    dmax = max(float(np.sqrt(((src_pos - r) ** 2).sum(axis=1)).max()) for r in refs)
    return max(1, int(ceil(dmax / medium.c_min / period)))


def _guard_finite(peak: float, *, where: str, solver_name: str) -> None:
    """Stop the run the moment the field stops being a number.

    A diverged k-space run does not crash. inf/NaN propagate through every
    remaining step, the convergence test never fires (``nan < tol`` is
    False), the loop runs to its settle cap, and the run exits 0 with a
    result file full of NaN -- a wrong answer wearing the shape of an
    answer, which is the one failure mode this library must never have. The
    first GPU gate session produced exactly that (256^3 rung, 2026-08-22:
    ``peak nan``, ``exit 0``, ``result.h5`` written, and the gate suite then
    counted the run as VRAM evidence).

    The peak is already on the host at every call site, so the check costs
    nothing and fires at the FIRST bad period rather than after the run has
    burned the remaining ones.
    """
    if isfinite(peak):
        return
    raise SolverDivergedError(
        f"solver '{solver_name}' diverged during {where}: the peak pressure is "
        f"{peak}, so the field is no longer finite. Nothing downstream of this "
        f"point is a physical result. Usual causes, most common first: a time "
        f"step the scheme cannot integrate for this medium (check the CFL and "
        f"points-per-wavelength of the fastest material), a drive amplitude far "
        f"outside the range float32 can carry, a medium with a non-physical "
        f"sound speed or density, or a source sitting in or against the PML "
        f"band, where it is driven and damped at once."
    )


def run_cw_kspace_pstd(
    solver_name: str,
    grid: Grid,
    medium: Medium,
    source: CWSource,
    spec: CWRunSpec,
    backend: str = "auto",
    record_region: tuple[slice, ...] | None = None,
    reference_point: tuple[int, ...] | None = None,
    nonlinear: bool = False,
    harmonics: tuple[int, ...] = (1,),
    checkpoint: CheckpointSpec | None = None,
    progress: Callable[[dict], None] | None = None,
) -> SolverResult:
    """Execute one steady-state CW solve (see module docstring for the scheme).

    With ``checkpoint`` set, the full time-stepping state is written
    atomically every ``checkpoint.every_periods`` acoustic periods; if the
    checkpoint file already exists (and fingerprints to THIS run) the solve
    resumes from it instead of starting over, and the file is removed on
    completion. ``checkpoint.stop_when`` turns the same machinery into a
    graceful session-budget stop (:class:`~caustica.io.checkpoint.RunInterrupted`).

    ``progress`` is called with ONE dict per period boundary and once more at
    the settle -> record transition — never per step, which would force a
    device->host sync every step and destroy GPU throughput. The payload is
    the library-wide progress contract (PLAN.md section 8)::

        {period, periods_expected, step, steps_expected, peak,
         converge_delta, elapsed_s, eta_s, stage}   # stage: settle | record

    plus one key that is deliberately NOT part of the serializable contract:
    ``snapshot``, a zero-argument callable returning a 2-D (1-D in 1-D) slice
    of the live pressure field through ``reference_point``. It is lazy on
    purpose — the copy happens only if the consumer asks, so a consumer that
    renders a mid-run preview every N periods pays exactly one device->host
    copy on those periods and nothing on the others. Consumers that serialize
    the payload (``status.json``, a GUI socket) drop the key.

    Progress is independent of ``checkpoint``: an in-memory run (the facade's
    ``out=None``) has no checkpoint and still reports. A callback that raises
    warns once and the solve continues — a broken notebook widget must not
    cost hours of compute.
    """
    harmonics = tuple(dict.fromkeys(int(h) for h in harmonics))
    if not harmonics or harmonics[0] != 1 or any(h < 1 for h in harmonics):
        raise ValueError(f"harmonics must start with 1 (fundamental), got {harmonics}")
    if reference_point is not None:
        ref = tuple(reference_point)
        if len(ref) != grid.ndim or any(
            not float(r).is_integer() or not 0 <= int(r) < n
            for r, n in zip(ref, grid.shape, strict=True)
        ):
            raise ValueError(
                f"reference_point must be integer VOXEL indices inside {grid.shape}, "
                f"got {reference_point} (did you pass meters?)"
            )

    b = get_backend(backend)
    xp, fft = b.xp, b.fft
    nd = grid.ndim
    dx = grid.dx
    period = 1.0 / source.f0
    omega = 2.0 * np.pi * source.f0
    c_max = medium.c_max

    # ---- exact-period dt under the CFL limit (notebook policy) ----
    spp, dt = cw_discretization(grid, medium, spec, source.f0, harmonics)

    # ---- padded FFT domain; property maps edge-replicated ----
    active = tuple(slice(0, n) for n in grid.shape)
    padded = ops.pad_shape(grid.shape)
    dt_over_rho = xp.asarray(dt / ops.pad_volume(medium.rho, padded), dtype=xp.float32)
    rhoc2_dt = xp.asarray(
        ops.pad_volume(medium.rho * medium.c.astype(np.float64) ** 2, padded) * dt,
        dtype=xp.float32,
    )
    absorb = xp.asarray(
        np.exp(-ops.pad_volume(medium.alpha * medium.c, padded) * dt), dtype=xp.float32
    )
    beta2_dt = None
    if nonlinear and not medium.is_linear:
        beta2_dt = xp.asarray(ops.pad_volume(medium.beta, padded) * (2.0 * dt), dtype=xp.float32)

    ks = ops.k_vectors(padded, dx, xp)
    kappa = ops.kappa_sinc(ks, c_ref=c_max, dt=dt, xp=xp)
    deriv = ops.spectral_derivative_factors(ks, kappa, padded, xp)
    del ks, kappa

    pml_edge = grid.pml.edge if grid.pml is not None else 2.0
    if grid.pml_vox == 0:
        log.warning(
            "grid has no PML: the FFT domain is PERIODIC and waves wrap around, "
            "producing standing-wave interference. Attach a PMLSpec to the Grid "
            "unless periodic boundaries are intended."
        )
    sponge = ops.sponge_volume(padded, grid.pml_vox, pml_edge, xp)

    p = xp.zeros(padded, dtype=xp.float32)
    u = [xp.zeros(padded, dtype=xp.float32) for _ in range(nd)]

    src_idx = tuple(xp.asarray(source.indices[:, d]) for d in range(nd))
    src_ph = xp.asarray(source.phases, dtype=xp.float32)
    amp = float(source.amplitude)
    # Mass-source normalization (k-Wave-equivalent): a RAW additive injection
    # realizes a plane amplitude ~amp/(2*CFL_local), which couples to REMOTE
    # medium content through dt(c_max) — a far fast inclusion changed the
    # realized drive by 27% (review finding, 2026-08-11). Scaling by
    # 2*c*dt/dx at the source voxels makes the realized amplitude
    # ~= source.amplitude, grid- and medium-invariant (few-% residual).
    c_src = medium.c[tuple(source.indices[:, d] for d in range(nd))]
    # The per-voxel drive weight rides along in the same factor. It is 1.0 for
    # a uniform source, so nothing about the plane-source calibration above
    # changes; a curved source uses it to carry its own AREA instead of its
    # voxel count, which a binary mask cannot do (see caustica.geometry.offgrid).
    src_scale = xp.asarray(
        2.0 * c_src.astype(np.float64) * dt / dx * source.drive_weights.astype(np.float64),
        dtype=xp.float32,
    )

    # ---- time of flight: farthest reference the wave must reach ----
    tof_periods = cw_tof_periods(grid, medium, source, reference_point)

    # Settling bounds are derived here (not in the loop below) because the
    # progress payload has to quote them before the first period runs.
    # The convergence test must not arm before the source ramp has ended:
    # the flattening cosine-ramp tail can dip below convergence_tol while
    # the drive is still rising (review finding, 2026-08-11; the k-Wave
    # adapter's fixed schedule already guards this).
    eff_min = tof_periods + max(spec.min_settle_periods, int(ceil(source.ramp_periods)) + 1)
    eff_max = max(tof_periods + spec.max_settle_periods, eff_min)

    # ---- convergence monitoring region (interior, PML shaved) ----
    margin = grid.pml_vox + 2
    if any(n <= 2 * margin for n in grid.shape):
        margin = 0
    conv = interior_slices(grid.shape, margin)

    # ---- record region resolved EARLY: it is part of the run fingerprint ----
    rec = (
        normalize_record_region(record_region, grid.shape) if record_region is not None else active
    )

    # ---- checkpoint plumbing (M10): resume, cadence writes, graceful stop ----
    # Everything that shapes the floating-point trajectory goes into the
    # fingerprint; load_checkpoint() refuses to splice two different runs.
    fingerprint = None
    if checkpoint is not None:
        fingerprint = {
            # Bumped to /2 on 2026-08-24: zeroing the Nyquist wavenumber in the
            # collocated first derivative changed the trajectory this scheme
            # produces on every even-length axis. A checkpoint written by /1
            # holds a state the current step loop would never have reached, and
            # splicing the two would produce a field that is neither.
            "scheme": NUMERICS_SCHEME,
            "solver": solver_name,
            "backend": b.name,
            "grid_shape": list(grid.shape),
            "dx": dx,
            "pml_vox": grid.pml_vox,
            "pml_edge": pml_edge,
            "spp": spp,
            "f0": source.f0,
            "amplitude": amp,
            "ramp_periods": source.ramp_periods,
            "harmonics": list(harmonics),
            "nonlinear": beta2_dt is not None,
            "record_region": [[s.start, s.stop] for s in rec],
            "reference_point": (
                None if reference_point is None else [int(r) for r in reference_point]
            ),
            "spec": spec.model_dump(),
            "source_sha1": arrays_sha1(source.indices, source.phases, source.drive_weights),
            "medium_sha1": arrays_sha1(medium.c, medium.rho, medium.alpha, medium.beta),
        }

    n = 0
    prev_peak: float | None = None
    converged = False
    converged_period: int | None = None  # None -> defaults to eff_max below
    history: list[tuple[int, float, float]] = []
    periods_done = 0
    stage = "settle"

    if checkpoint is not None:
        state = load_checkpoint(checkpoint.path, fingerprint)
        if state is not None:
            # float32 state through uncompressed npz is bit-exact, so the
            # resumed trajectory is the SAME one, not a nearby one.
            p = xp.asarray(state["fields"]["p"])
            u = [xp.asarray(state["fields"][f"u{i}"]) for i in range(nd)]
            n = state["n"]
            periods_done = state["periods_done"]
            prev_peak = state["prev_peak"]
            converged = state["converged"]
            converged_period = state["converged_period"]
            history = state["history"]
            stage = state["stage"]

    def _write_ckpt(stage_now: str) -> None:
        write_checkpoint(
            checkpoint.path,
            stage=stage_now,
            n=n,
            periods_done=periods_done,
            prev_peak=prev_peak,
            converged=converged,
            converged_period=converged_period,
            history=history,
            fields={"p": b.to_numpy(p), **{f"u{i}": b.to_numpy(u[i]) for i in range(nd)}},
            fingerprint=fingerprint,
        )

    # ---- progress reporting (independent of checkpointing) ----
    periods_expected = eff_max
    if spec.t_end_min_us is not None:
        periods_expected = max(
            periods_expected, ceil(spec.t_end_min_us * 1e-6 / period) - spec.n_record_periods
        )
    periods_expected += spec.n_record_periods
    steps_expected = periods_expected * spp
    t_progress0 = time.monotonic()
    steps_at_start = n  # resume offset: ETA measures THIS session's cadence
    last_peak: float | None = prev_peak
    last_delta: float | None = None
    progress_warned = False

    def _snapshot() -> np.ndarray:
        """ONE device->host copy: the live field sliced through the reference.

        Lazy by contract — the consumer decides how often a mid-run preview
        is worth a copy, the engine only offers the array (the renderer lives
        outside the solver, see :mod:`caustica.progress`).
        """
        if nd == 1:
            return b.to_numpy(p[active[0]])
        if nd == 2:
            return b.to_numpy(p[active[0], active[1]])
        j = int(reference_point[1]) if reference_point is not None else grid.shape[1] // 2
        return b.to_numpy(p[active[0], j, active[2]])

    def _emit_progress(stage_now: str) -> None:
        """One payload per period boundary; a broken consumer never stops us."""
        if progress is None:
            return
        nonlocal progress_warned
        elapsed = time.monotonic() - t_progress0
        done = n - steps_at_start
        eta = None
        if done > 0 and steps_expected > n:
            eta = round((steps_expected - n) * (elapsed / done), 1)
        try:
            progress(
                {
                    "period": periods_done,
                    "periods_expected": periods_expected,
                    "step": n,
                    "steps_expected": steps_expected,
                    "peak": last_peak,
                    "converge_delta": last_delta,
                    "elapsed_s": round(elapsed, 3),
                    "eta_s": eta,
                    "stage": stage_now,
                    "snapshot": _snapshot,  # NOT serializable; see the docstring
                }
            )
        except Exception as exc:  # noqa: BLE001 - a widget must not kill a solve
            if not progress_warned:
                progress_warned = True
                warnings.warn(
                    f"progress callback raised {type(exc).__name__}: {exc}. The solve "
                    f"continues; further progress failures are silent.",
                    CausticaWarning,
                    stacklevel=2,
                )

    def _period_boundary() -> None:
        """Progress + checkpoint cadence + graceful-stop poll, once per period.

        Progress comes FIRST and unconditionally: this call site used to
        return immediately when no checkpoint was configured, which made a
        callback added here invisible to exactly the in-memory notebook run
        it exists for (trap T1 in docs/library_first_plan.md).
        """
        _emit_progress("settle")
        if checkpoint is None:
            return
        stop = checkpoint.stop_when is not None and checkpoint.stop_when()
        if stop or periods_done % checkpoint.every_periods == 0:
            _write_ckpt("settle")
        if stop:
            raise RunInterrupted(checkpoint.path, "settle", periods_done, n)

    # numpy 2 deprecates `s` without `axes` and warns that a future release
    # will read them pairwise; spelling the axes out keeps today's meaning
    # (transform every axis) the one a later numpy will also read.
    fft_axes = tuple(range(nd))

    def step(n: int) -> None:
        pk = fft.rfftn(p)
        for i in range(nd):
            grad_i = fft.irfftn(deriv[i] * pk, s=padded, axes=fft_axes)
            u[i] -= dt_over_rho * grad_i
            u[i] *= absorb
            u[i] *= sponge
        acc = None
        for i in range(nd):
            term = deriv[i] * fft.rfftn(u[i])
            acc = term if acc is None else acc + term
        divu = fft.irfftn(acc, s=padded, axes=fft_axes)
        p_local = p
        if beta2_dt is None:
            p_local -= rhoc2_dt * divu
        else:
            # Westervelt: dp = -(rho c^2 dt) div u - 2 beta dt p div u
            p_local -= (rhoc2_dt + beta2_dt * p_local) * divu
        p_local *= absorb
        p_local *= sponge
        t = n * dt
        env = ramp_envelope(t, period, source.ramp_periods)
        p_local[src_idx] += (xp.float32(amp * env) * src_scale) * xp.sin(
            xp.float32(omega * t) - src_ph
        )

    # ---- settle until the per-period peak stops moving ----
    if converged_period is None:
        converged_period = eff_max
    period_idx = periods_done  # 0 fresh; the resumed period count otherwise
    while stage == "settle" and not converged and period_idx < eff_max:
        period_idx += 1
        period_peak = xp.zeros((), dtype=xp.float32)
        for _ in range(spp):
            step(n)
            n += 1
            xp.maximum(period_peak, xp.abs(p[conv]).max(), out=period_peak)
        peak = float(period_peak)
        _guard_finite(
            peak, where=f"settling period {period_idx} (step {n})", solver_name=solver_name
        )
        periods_done = period_idx
        rel = abs(peak - prev_peak) / max(prev_peak, 1e-9) if prev_peak is not None else None
        last_peak, last_delta = peak, rel
        if period_idx >= tof_periods:
            history.append((period_idx, peak, float("nan") if rel is None else rel))
        if period_idx >= eff_min and rel is not None:
            if peak > 0.0 and rel < spec.convergence_tol:
                converged_period = period_idx
                converged = True
        if not converged:
            prev_peak = peak
        _period_boundary()

    # ---- honor an explicit t_end floor (production contract hook) ----
    if spec.t_end_min_us is not None and stage == "settle":
        need_periods = ceil(spec.t_end_min_us * 1e-6 / period) - spec.n_record_periods
        while periods_done < need_periods:
            for _ in range(spp):
                step(n)
                n += 1
            periods_done += 1
            _period_boundary()

    # ---- record window: leakage-free single-bin DFTs + time peak ----
    if stage == "settle":
        # The stage flip is reported whether or not a checkpoint exists (T1):
        # "settling done, recording now" is the one transition a watcher
        # needs, and an in-memory run has no checkpoint at all.
        stage = "record"
        _emit_progress("record")
        if checkpoint is not None:
            # Snapshot the exact pre-record state: a kill DURING the (short)
            # record window resumes by redoing the window from here, bit-exact.
            _write_ckpt("record")
            if checkpoint.stop_when is not None and checkpoint.stop_when():
                raise RunInterrupted(checkpoint.path, "record", periods_done, n)
    rec_steps = spec.n_record_periods * spp
    buffers = {h: xp.zeros(p[rec].shape, dtype=xp.complex64) for h in harmonics}
    pmax = xp.zeros(p[rec].shape, dtype=xp.float32)
    for _ in range(rec_steps):
        step(n)
        t = n * dt
        pa = p[rec]
        for h in harmonics:
            # Library-wide phasor convention: p(t) = Re{P exp(-i omega t)},
            # outgoing wave = exp(+ikx) — same as the analytic references
            # (O'Neil/Rayleigh). Accumulating exp(+i...) extracts exactly
            # that P (review finding, 2026-08-11: the old -i kernel produced
            # the CONJUGATE of the analytic convention).
            buffers[h] += pa * xp.exp(xp.complex64(+1j * h * omega * t))
        xp.maximum(pmax, xp.abs(pa), out=pmax)
        n += 1
    for h in harmonics:
        buffers[h] *= xp.complex64(2.0 / rec_steps)
    # One reduction over the recorded peak: the settle guard above cannot see
    # a divergence that starts inside the record window itself, and the record
    # window is exactly what gets written to result.h5.
    _guard_finite(float(pmax.max()), where=f"the record window (step {n})", solver_name=solver_name)

    phasors = {h: b.to_numpy(buffers[h]) for h in harmonics}
    if checkpoint is not None and not checkpoint.keep_on_success:
        # The run completed; the checkpoint would only invite a future run to
        # "resume" into an already-produced result. With keep_on_success the
        # CALLER deletes it — after the result is safely stored, so a failed
        # store stays resumable from the pre-record snapshot.
        checkpoint.path.unlink(missing_ok=True)
    return SolverResult(
        phasor=phasors[1],
        p_max=b.to_numpy(pmax),
        region=rec,
        dt=dt,
        spp=spp,
        steps_total=n,
        t_end_s=n * dt,
        tof_periods=tof_periods,
        converged_period=converged_period,
        settle_capped=not converged,
        convergence_history=history,
        phasors=phasors,
        meta={
            "solver": solver_name,
            "backend": b.name,
            "padded_shape": padded,
            "c_ref": c_max,
            "nonlinear_active": beta2_dt is not None,
            "source": source.label,
            "numerics_scheme": NUMERICS_SCHEME,
        },
    )
