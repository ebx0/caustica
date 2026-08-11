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
from itertools import product
from math import ceil, floor

import numpy as np

from hifusim.core.backend import get_backend
from hifusim.core.grid import Grid
from hifusim.medium import Medium
from hifusim.solvers.base import CWRunSpec, SolverResult, interior_slices
from hifusim.solvers.kspace import operators as ops
from hifusim.sources import CWSource, ramp_envelope

log = logging.getLogger("hifusim")


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
) -> SolverResult:
    """Execute one steady-state CW solve (see module docstring for the scheme)."""
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
    deriv = ops.spectral_derivative_factors(ks, kappa, xp)
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
    src_scale = xp.asarray(2.0 * c_src.astype(np.float64) * dt / dx, dtype=xp.float32)

    # ---- time of flight: farthest reference the wave must reach ----
    tof_periods = cw_tof_periods(grid, medium, source, reference_point)

    # ---- convergence monitoring region (interior, PML shaved) ----
    margin = grid.pml_vox + 2
    if any(n <= 2 * margin for n in grid.shape):
        margin = 0
    conv = interior_slices(grid.shape, margin)

    def step(n: int) -> None:
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
    # The convergence test must not arm before the source ramp has ended:
    # the flattening cosine-ramp tail can dip below convergence_tol while
    # the drive is still rising (review finding, 2026-08-11; the k-Wave
    # adapter's fixed schedule already guards this).
    eff_min = tof_periods + max(spec.min_settle_periods, int(ceil(source.ramp_periods)) + 1)
    eff_max = max(tof_periods + spec.max_settle_periods, eff_min)
    n = 0
    prev_peak: float | None = None
    converged = False
    converged_period = eff_max
    history: list[tuple[int, float, float]] = []
    periods_done = 0
    for period_idx in range(1, eff_max + 1):
        period_peak = xp.zeros((), dtype=xp.float32)
        for _ in range(spp):
            step(n)
            n += 1
            xp.maximum(period_peak, xp.abs(p[conv]).max(), out=period_peak)
        peak = float(period_peak)
        periods_done = period_idx
        if period_idx >= tof_periods:
            rel = (
                abs(peak - prev_peak) / max(prev_peak, 1e-9)
                if prev_peak is not None
                else float("nan")
            )
            history.append((period_idx, peak, rel))
        if period_idx >= eff_min and prev_peak is not None:
            rel = abs(peak - prev_peak) / max(prev_peak, 1e-9)
            if peak > 0.0 and rel < spec.convergence_tol:
                converged_period = period_idx
                converged = True
                break
        prev_peak = peak

    # ---- honor an explicit t_end floor (production contract hook) ----
    if spec.t_end_min_us is not None:
        need_periods = ceil(spec.t_end_min_us * 1e-6 / period) - spec.n_record_periods
        while periods_done < need_periods:
            for _ in range(spp):
                step(n)
                n += 1
            periods_done += 1

    # ---- record window: leakage-free single-bin DFTs + time peak ----
    rec = (
        normalize_record_region(record_region, grid.shape) if record_region is not None else active
    )
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

    phasors = {h: b.to_numpy(buffers[h]) for h in harmonics}
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
        },
    )
