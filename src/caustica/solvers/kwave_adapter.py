"""k-Wave as a first-class caustica solver (registry name: ``"kwave"``).

User decision (2026-08-10): k-Wave is not only a benchmark target — it is a
selectable solver, and the validation chain centers on cross-checking our
native solvers against it (plus the analytic references). This adapter maps
caustica's Grid/Medium/CWSource onto k-wave-python's objects, drives the
precompiled kspaceFirstOrder binary (CPU/OMP by default — the GPU binary is
known-broken on Colab, see MILESTONES M12), and converts the recorded time
series back into the project's phasor contract via
:func:`caustica.spectral.single_bin_phasor`.

Unit conversions (explicit, tested):
* absorption: caustica stores Np/m (frequency-independent, v1). k-Wave wants
  dB/(MHz^y cm); we emit ``alpha_coeff = alpha * (20/ln10) / 100`` with
  ``alpha_power = 0`` (frequency-independent power law).
* nonlinearity: caustica beta = 1 + B/2A  =>  k-Wave ``BonA = 2 (beta - 1)``.

Differences from the native solvers (documented, not hidden):
* No adaptive convergence — k-Wave runs a FIXED schedule of
  ``tof + min_settle_periods`` settle periods plus the record window. Pick
  ``min_settle_periods`` generously for strongly reverberant media, and more
  so when asking for harmonics: the native engine settles until every
  requested harmonic stops moving, and a schedule fixed in advance cannot.
  Requesting 2f0 at the default settle is warned about rather than silently
  obeyed.
* k-Wave applies its own PML INSIDE the grid edge (``pml_inside=True``).
  The adapter passes ``pml_size = grid.pml_vox`` so the damped band matches
  the native sponge (falling back to k-Wave's default — 20 voxels in 2-D,
  10 in 3-D — when the grid has no PML), and REFUSES sources that sit
  inside that band: k-Wave would silently swallow them (review finding,
  2026-08-11).
* Source amplitude: k-Wave normalizes additive pressure sources internally
  (realized plane amplitude ~= the prescribed signal); the native engine
  applies the equivalent ``2 c dt / dx`` mass-source scaling (2026-08-11),
  so absolute amplitudes agree to first order across the registry.
"""

from __future__ import annotations

import warnings
from itertools import product
from math import ceil
from typing import Any

import numpy as np

from caustica.core.backend import CausticaWarning
from caustica.core.grid import Grid
from caustica.medium import Medium
from caustica.solvers.base import (
    PML_DAMPED_LIMIT,
    CWRunSpec,
    SolverBase,
    SolverCaps,
    SolverResult,
)
from caustica.solvers.registry import register
from caustica.sources import CWSource
from caustica.spectral import single_bin_phasor

_NP_TO_DB = 20.0 / np.log(10.0)  # 1 Np = 8.6859 dB


def alpha_np_m_to_kwave(alpha_np_m: np.ndarray | float) -> np.ndarray | float:
    """Np/m (frequency-independent) -> dB/(MHz^0 cm) for k-Wave."""
    return alpha_np_m * _NP_TO_DB / 100.0


def beta_to_bona(beta: np.ndarray | float) -> np.ndarray | float:
    """caustica beta (= 1 + B/2A) -> k-Wave B/A."""
    return 2.0 * (np.asarray(beta, dtype=np.float64) - 1.0)


@register
class KWaveSolver(SolverBase):
    """k-wave-python kspaceFirstOrder wrapper (CPU/OMP binary)."""

    name = "kwave"
    caps = SolverCaps(
        ndim=frozenset({2, 3}),
        nonlinear=True,
        drive=frozenset({"cw"}),
        backends=frozenset({"external"}),
    )

    def validate(self, grid: Grid, medium: Medium, source: CWSource) -> None:
        super().validate(grid, medium, source)
        if (medium.beta != 0).any() and not ((medium.beta == 0) | (medium.beta >= 1.0)).all():
            raise ValueError(
                "medium.beta must be 0 (linear) or >= 1 for the k-Wave mapping "
                "BonA = 2*(beta - 1); got values in (0, 1)."
            )

    def run(
        self,
        grid: Grid,
        medium: Medium,
        source: CWSource,
        spec: CWRunSpec | None = None,
        use_gpu_binary: bool = False,
        record_region: tuple[slice, ...] | None = None,
        reference_point: tuple[int, ...] | None = None,
        harmonics: tuple[int, ...] = (1,),
        **kwargs: Any,
    ) -> SolverResult:
        if kwargs:
            raise TypeError(f"unknown run() options: {sorted(kwargs)}")
        spec = spec or CWRunSpec()
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
        self.validate(grid, medium, source)
        try:
            import kwave
            from kwave.kgrid import kWaveGrid
            from kwave.kmedium import kWaveMedium
            from kwave.ksensor import kSensor
            from kwave.ksource import kSource
            from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
            from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
            from kwave.options.simulation_execution_options import (
                SimulationExecutionOptions,
            )
            from kwave.options.simulation_options import SimulationOptions
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "solver 'kwave' needs the optional dependency k-wave-python: "
                "pip install caustica[kwave]"
            ) from exc

        nd = grid.ndim
        dx = grid.dx
        period = 1.0 / source.f0
        c_max, c_min = medium.c_max, medium.c_min

        # ---- exact-period dt at k-Wave's customary CFL 0.3 ----
        spp = max(2, int(ceil(period / (0.3 * dx / c_max))))
        dt = period / spp
        if 2 * max(harmonics) >= spp:
            raise ValueError(
                f"harmonic {max(harmonics)} is at/above the temporal Nyquist for "
                f"spp={spp}; refine dx to raise spp."
            )

        # ---- fixed schedule: tof + min_settle + record ----
        src_pos = source.indices.astype(np.float64) * dx
        if reference_point is not None:
            refs = np.asarray(reference_point, np.float64)[None, :] * dx
        else:
            refs = np.array(list(product(*((0, n - 1) for n in grid.shape))), np.float64) * dx
        dmax = max(float(np.sqrt(((src_pos - r) ** 2).sum(axis=1)).max()) for r in refs)
        tof_periods = max(1, int(ceil(dmax / c_min / period)))
        # Fixed schedule (no adaptive convergence here): the settle window
        # must outlast the source ramp, or the record window lands on a
        # still-rising drive and the phasor is silently biased low.
        settle_periods = tof_periods + max(
            spec.min_settle_periods, int(ceil(source.ramp_periods)) + 2
        )
        # A fixed schedule cannot know when a HARMONIC has settled, and the
        # default was never chosen with one in mind. The native engine grades
        # every requested harmonic against its own amplitude and settles until
        # each stops moving; nothing here can, because the binary runs to a
        # step count decided before it starts. Measured on a focused bowl in
        # water with beta = 0, where the true 2f0 is zero: a settle short
        # enough to satisfy the peak alone left 2.1 % of the fundamental in
        # the harmonic channel, and it took roughly thirty periods past
        # time-of-flight to clear. So the default is flagged rather than
        # silently used — the number is the caller's to pick, but not by
        # accident.
        if max(harmonics) > 1 and spec.min_settle_periods <= CWRunSpec().min_settle_periods:
            warnings.warn(
                f"harmonics {harmonics} requested with min_settle_periods="
                f"{spec.min_settle_periods}: k-Wave runs a fixed schedule and cannot "
                f"detect when a harmonic has stopped moving. A settle that satisfies "
                f"the fundamental can leave a percent of it in the 2f0 channel. Raise "
                f"min_settle_periods (about 30 sufficed for a focused bowl in water) "
                f"or cross-check against a native run, which grades the harmonics.",
                CausticaWarning,
                stacklevel=2,
            )
        total_periods = settle_periods + spec.n_record_periods
        if spec.t_end_min_us is not None:
            total_periods = max(total_periods, ceil(spec.t_end_min_us * 1e-6 / period))
        nt = total_periods * spp
        rec_steps = spec.n_record_periods * spp

        # ---- k-Wave objects ----
        kgrid = kWaveGrid(np.asarray(grid.shape), np.full(nd, dx))
        kgrid.setTime(nt, dt)

        kmedium = kWaveMedium(
            sound_speed=medium.c.astype(np.float64),
            density=medium.rho.astype(np.float64),
            alpha_coeff=alpha_np_m_to_kwave(medium.alpha.astype(np.float64)),
            alpha_power=0.0,
        )
        if not medium.is_linear:
            kmedium.BonA = beta_to_bona(medium.beta)

        src_mask = np.zeros(grid.shape, dtype=bool)
        src_mask[tuple(source.indices[:, d] for d in range(nd))] = True
        # k-Wave orders mask points in Fortran (column-major) order; sort our
        # per-voxel phases to match.
        order_f = np.flatnonzero(src_mask.flatten(order="F"))
        coords_f = np.stack(np.unravel_index(order_f, grid.shape, order="F"), axis=1)
        lut = {tuple(row): i for i, row in enumerate(map(tuple, source.indices))}
        order_in_source = np.array([lut[tuple(c)] for c in map(tuple, coords_f)])
        phase_f = source.phases[order_in_source]
        # Per-voxel drive weights reorder with the phases, for the same reason:
        # k-Wave reads its source points in Fortran order and ours are stored
        # in whatever order the constructor produced.
        weight_f = source.drive_weights[order_in_source]

        t = np.arange(nt) * dt
        env = 0.5 * (1.0 - np.cos(np.pi * np.minimum(t / (source.ramp_periods * period), 1.0)))
        omega = 2.0 * np.pi * source.f0
        signals = (
            source.amplitude
            * weight_f[:, None]
            * env[None, :]
            * np.sin(omega * t[None, :] - phase_f[:, None])
        ).astype(np.float32)

        ksource = kSource()
        ksource.p_mask = src_mask
        ksource.p = signals

        # The sensor mask is the RECORD REGION, not the whole grid. k-Wave's
        # binary cannot accumulate a DFT, so it dumps every recorded step to
        # HDF5 and the adapter transforms afterwards: at the ITRUSST benchmark
        # size that is a 731 MB input file and a 1.6 GB output file per run,
        # and recording the full grid when the caller asked for a slab pays
        # all of it for nothing. Measured 2026-08-25, and the reason a dataset
        # of many runs wants a region.
        from caustica.solvers.kspace.engine import normalize_record_region  # noqa: PLC0415

        rec = (
            normalize_record_region(record_region, grid.shape)
            if record_region is not None
            else tuple(slice(0, n) for n in grid.shape)
        )
        rec_shape = tuple(sl.stop - sl.start for sl in rec)
        sensor = kSensor()
        mask = np.zeros(grid.shape, dtype=bool)
        mask[rec] = True
        sensor.mask = mask
        sensor.record = ["p"]
        sensor.record_start_index = nt - rec_steps + 1  # 1-based, inclusive

        # Match the damped band to the native sponge; k-Wave's default PML
        # (20 vox 2-D / 10 vox 3-D) applies only when the grid has none.
        pml_size = grid.pml_vox if grid.pml_vox > 0 else (20 if nd == 2 else 10)
        if grid.pml_vox == 0:
            warnings.warn(
                f"grid has no PML: k-Wave still damps its default inner band "
                f"({pml_size} voxels) — the field near the edges will differ "
                f"from the native (periodic) solvers.",
                stacklevel=2,
            )
        # Graded on the DRIVE, not on the bounding box: a band-limited source
        # has a halo whose outermost voxels carry a fraction of a percent, and
        # refusing a run because that tail touched the band would refuse every
        # off-grid bowl with a normal standoff. The threshold and the reasoning
        # are the native solvers' (caustica.solvers.base).
        idx = source.indices
        w = np.abs(source.drive_weights.astype(np.float64))
        upper = np.asarray(grid.shape) - pml_size
        outside = ~((idx >= pml_size) & (idx < upper)).all(axis=1)
        damped = float(w[outside].sum() / w.sum()) if w.sum() else 0.0
        lo, hi = idx.min(axis=0), idx.max(axis=0)
        if damped > PML_DAMPED_LIMIT:
            raise ValueError(
                f"{damped * 100:.1f}% of this source's drive (span {tuple(lo)}.."
                f"{tuple(hi)}) lies inside k-Wave's inner PML band ({pml_size} voxels "
                f"on each face of {grid.shape}, pml_inside=True) and would be silently "
                f"damped. Enlarge the grid or move the source inward."
            )
        if damped > 0.01:
            warnings.warn(
                f"{damped * 100:.1f}% of the source drive sits inside k-Wave's inner "
                f"PML band ({pml_size} voxels) and is damped as it is applied.",
                stacklevel=2,
            )
        sim_options = SimulationOptions(
            pml_inside=True, pml_size=pml_size, data_cast="single", save_to_disk=True
        )
        exec_options = SimulationExecutionOptions(
            is_gpu_simulation=use_gpu_binary, show_sim_log=False
        )
        run_fn = kspaceFirstOrder2D if nd == 2 else kspaceFirstOrder3D
        with warnings.catch_warnings():
            # k-wave-python 0.6.x: the dimension-specific entry points are the
            # documented stable API but emit a FutureWarning; the Windows
            # executor also warns that a custom binary name overrides the GPU
            # flag (we set the flag explicitly ourselves). Both are benign and
            # would otherwise flood every validation run.
            warnings.filterwarnings("ignore", category=FutureWarning, module=r"kwave.*")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"kwave.*")
            warnings.filterwarnings(
                "ignore", message=".*Custom binary name set.*", category=UserWarning
            )
            sensor_data = run_fn(
                kgrid=kgrid,
                source=ksource,
                sensor=sensor,
                medium=kmedium,
                simulation_options=sim_options,
                execution_options=exec_options,
            )

        p_rec = np.asarray(sensor_data["p"])
        n_points = int(np.prod(rec_shape))
        if rec_steps == n_points:  # square: orientation is ambiguous
            warnings.warn(
                "k-Wave sensor data is square (rec_steps == n_points); assuming "
                "time-major layout - verify against a non-square run.",
                stacklevel=2,
            )
        if p_rec.shape == (rec_steps, n_points):
            p_rec = p_rec.T
        elif p_rec.shape != (n_points, rec_steps):
            raise RuntimeError(
                f"unexpected k-Wave sensor data shape {p_rec.shape}; "
                f"expected ({n_points}, {rec_steps}) or transposed."
            )

        # ---- back to the caustica phasor contract (per requested harmonic) ----
        t0 = (sensor.record_start_index - 1) * dt
        pmax_pts = np.abs(p_rec).max(axis=1)

        # k-Wave hands back its mask points in Fortran order, so the inverse
        # map is built from the mask in that same order rather than assumed.
        rec_order = np.flatnonzero(mask.flatten(order="F"))
        rec_coords = np.unravel_index(rec_order, grid.shape, order="F")
        rec_coords = tuple(c - sl.start for c, sl in zip(rec_coords, rec, strict=True))

        phasors: dict[int, np.ndarray] = {}
        for h in harmonics:
            pts = single_bin_phasor(p_rec, dt=dt, f0=h * source.f0, t0=t0, axis=1)
            vol = np.zeros(rec_shape, dtype=np.complex64)
            vol[rec_coords] = pts.astype(np.complex64)
            phasors[h] = vol
        pmax = np.zeros(rec_shape, dtype=np.float32)
        pmax[rec_coords] = pmax_pts.astype(np.float32)

        return SolverResult(
            phasor=phasors[1],
            p_max=pmax,
            region=rec,
            dt=dt,
            spp=spp,
            steps_total=nt,
            t_end_s=nt * dt,
            tof_periods=tof_periods,
            converged_period=settle_periods,
            settle_capped=False,
            convergence_history=[],
            phasors=phasors,
            meta={
                "solver": self.name,
                "backend": "kwave-omp" if not use_gpu_binary else "kwave-cuda",
                "kwave_binary_gpu": use_gpu_binary,
                "fixed_schedule_periods": total_periods,
                "source": source.label,
                # An external engine has its own generation: ours is the
                # adapter's contract, theirs is the binary's version.
                "numerics_scheme": f"kwave/{getattr(kwave, '__version__', 'unknown')}",
            },
        )
