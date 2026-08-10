"""Linear k-space PSTD solver (1/2/3-D, CW steady state, numpy or cupy).

Scheme (validated in the production notebook, nonlinear term stripped):
first-order coupled acoustic equations advanced with k-space spectral
derivatives and the kappa sinc correction (exact plane-wave propagation at
c_ref = c_max), frequency-independent exponential absorption, Gaussian
sponge PML, additive ramped CW pressure injection, exact-period dt, and a
leakage-free single-bin f0 DFT over the record window.

Per time step (dt):
    u_i <- (u_i - dt/rho * IFFT{ i k_i kappa FFT{p} }) * e^{-alpha c dt} * sponge
    p   <- (p - rho c^2 dt * IFFT{ kappa sum_i i k_i FFT{u_i} }) * e^{-alpha c dt} * sponge
    p[src] += p0 * ramp(t) * sin(omega t - phase)

ABSORPTION FIX vs the source notebook (found by the M4 unit gate): damping
only the pressure at rate a = alpha*c yields a spatial decay of alpha/2
(dispersion relation k = omega/c + i*a/(2c); equipartition halves the loss).
Damping BOTH p and u by e^{-a dt} substitutes omega -> omega + i*a exactly,
giving spatial decay alpha with unchanged phase speed. The notebook damped p
only, so its datasets realized HALF the labeled tissue absorption — recorded
as a dataset caveat, not carried into this library.

The CPU path allocates transform temporaries per step (clear > clever at
this stage); the GPU pass (M7) preallocates and fuses without changing the
math here.
"""

from __future__ import annotations

import logging
from itertools import product
from math import ceil, floor
from typing import Any

import numpy as np

from hifusim.core.backend import get_backend
from hifusim.core.grid import Grid
from hifusim.medium import Medium
from hifusim.solvers.base import (
    CWRunSpec,
    SolverBase,
    SolverCaps,
    SolverResult,
    interior_slices,
)
from hifusim.solvers.kspace import operators as ops
from hifusim.solvers.registry import register
from hifusim.sources import CWSource, ramp_envelope

log = logging.getLogger("hifusim")


@register
class LinearKSpacePSTD(SolverBase):
    """Linear full-wave k-space PSTD (registry name: ``"linear"``)."""

    name = "linear"
    caps = SolverCaps(
        ndim=frozenset({1, 2, 3}),
        nonlinear=False,
        drive=frozenset({"cw"}),
        backends=frozenset({"numpy", "cupy"}),
    )

    def run(
        self,
        grid: Grid,
        medium: Medium,
        source: CWSource,
        spec: CWRunSpec | None = None,
        backend: str = "auto",
        record_region: tuple[slice, ...] | None = None,
        reference_point: tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> SolverResult:
        if kwargs:
            raise TypeError(f"unknown run() options: {sorted(kwargs)}")
        spec = spec or CWRunSpec()
        self.validate(grid, medium, source)

        b = get_backend(backend)
        xp, fft = b.xp, b.fft
        nd = grid.ndim
        dx = grid.dx
        period = 1.0 / source.f0
        omega = 2.0 * np.pi * source.f0
        c_max, c_min = medium.c_max, medium.c_min

        # ---- exact-period dt under the CFL limit (notebook policy) ----
        dt_cfl = spec.cfl * dx / c_max
        spp = max(2, int(floor(period / dt_cfl)))
        dt = period / spp
        while dt * c_max / dx > spec.cfl_hard_max + 1e-12 and spp < 1024:
            spp += 1
            dt = period / spp

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

        ks = ops.k_vectors(padded, dx, xp)
        kappa = ops.kappa_sinc(ks, c_ref=c_max, dt=dt, xp=xp)
        deriv = ops.spectral_derivative_factors(ks, kappa, xp)
        del ks, kappa

        pml_edge = grid.pml.edge if grid.pml is not None else 2.0
        if grid.pml_vox == 0:
            log.warning(
                "grid has no PML: the FFT domain is PERIODIC and waves wrap "
                "around, producing standing-wave interference. Attach a "
                "PMLSpec to the Grid unless periodic boundaries are intended."
            )
        sponge = ops.sponge_volume(padded, grid.pml_vox, pml_edge, xp)

        p = xp.zeros(padded, dtype=xp.float32)
        u = [xp.zeros(padded, dtype=xp.float32) for _ in range(nd)]

        src_idx = tuple(xp.asarray(source.indices[:, d]) for d in range(nd))
        src_ph = xp.asarray(source.phases, dtype=xp.float32)
        amp = float(source.amplitude)

        # ---- time of flight: farthest reference the wave must reach ----
        src_pos = source.indices.astype(np.float64) * dx
        if reference_point is not None:
            refs = np.asarray(reference_point, np.float64)[None, :] * dx
        else:
            corners = np.array(list(product(*((0, n - 1) for n in grid.shape))), np.float64) * dx
        refs = refs if reference_point is not None else corners
        dmax = max(float(np.sqrt(((src_pos - r) ** 2).sum(axis=1)).max()) for r in refs)
        tof_periods = max(1, int(ceil(dmax / c_min / period)))

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
            p_local -= rhoc2_dt * divu
            p_local *= absorb
            p_local *= sponge
            t = n * dt
            env = ramp_envelope(t, period, source.ramp_periods)
            p_local[src_idx] += xp.float32(amp * env) * xp.sin(xp.float32(omega * t) - src_ph)

        # ---- settle until the per-period peak stops moving ----
        eff_min = tof_periods + spec.min_settle_periods
        eff_max = tof_periods + spec.max_settle_periods
        n = 0
        prev_peak: float | None = None
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

        # ---- record window: leakage-free single-bin DFT + time peak ----
        rec = record_region if record_region is not None else active
        for sl, n_ax in zip(rec, grid.shape, strict=True):
            if not (0 <= (sl.start or 0) and (sl.stop or n_ax) <= n_ax):
                raise ValueError(f"record_region {rec} exceeds active grid {grid.shape}")
        rec_steps = spec.n_record_periods * spp
        phasor = xp.zeros(p[rec].shape, dtype=xp.complex64)
        pmax = xp.zeros(p[rec].shape, dtype=xp.float32)
        for _ in range(rec_steps):
            step(n)
            t = n * dt
            pa = p[rec]
            phasor += pa * xp.exp(xp.complex64(-1j * omega * t))
            xp.maximum(pmax, xp.abs(pa), out=pmax)
            n += 1
        phasor *= xp.complex64(2.0 / rec_steps)

        return SolverResult(
            phasor=b.to_numpy(phasor),
            p_max=b.to_numpy(pmax),
            region=rec,
            dt=dt,
            spp=spp,
            steps_total=n,
            t_end_s=n * dt,
            tof_periods=tof_periods,
            converged_period=converged_period,
            settle_capped=converged_period >= eff_max,
            convergence_history=history,
            meta={
                "solver": self.name,
                "backend": b.name,
                "padded_shape": padded,
                "c_ref": c_max,
                "source": source.label,
            },
        )
