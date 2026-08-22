"""Acoustic sources (v1: continuous-wave pressure injection).

A :class:`CWSource` is solver-agnostic data: WHICH voxels are driven, with
WHAT phase each, at WHAT amplitude/frequency. How the drive enters the
field (additive pressure injection, ramp shape) is solver policy shared via
:func:`ramp_envelope`.

The per-voxel representation deliberately mirrors the production notebook:
transducer elements are voxelized into 1-voxel-thick curved shells and each
shell voxel carries its element's phase. The arrays module (M6) will build
CWSources from real transducer geometries; the helpers here cover the
validation geometries (plane, bowl) needed by the solver test gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caustica.analytic.geometry import spherical_cap_points
from caustica.core.grid import Grid


@dataclass(frozen=True)
class CWSource:
    """Continuous-wave multi-voxel pressure source.

    Attributes
    ----------
    indices:
        ``(n, ndim)`` integer voxel coordinates (active-grid frame).
    phases:
        ``(n,)`` per-voxel drive phase [rad].
    amplitude:
        Drive amplitude [Pa]. Solvers apply mass-source normalization
        (native: ``2 c dt / dx`` at the source voxels; k-Wave: internal),
        so the realized plane amplitude is ~= this value to the few-%
        level, independent of grid, CFL and medium (2026-08-11; before
        that the raw additive injection made it discretization-dependent).
    f0:
        Drive frequency [Hz].
    ramp_periods:
        Cosine-taper ramp length in periods (notebook default: 3).
    """

    indices: np.ndarray
    phases: np.ndarray
    amplitude: float
    f0: float
    ramp_periods: float = 3.0
    label: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        idx = np.asarray(self.indices)
        ph = np.asarray(self.phases, dtype=np.float32)
        if idx.ndim != 2:
            raise ValueError(f"indices must be (n, ndim), got shape {idx.shape}")
        if not np.issubdtype(idx.dtype, np.integer):
            raise TypeError(f"indices must be integer voxel coords, got {idx.dtype}")
        if ph.shape != (idx.shape[0],):
            raise ValueError(f"phases must be ({idx.shape[0]},), got {ph.shape}")
        if self.amplitude <= 0 or self.f0 <= 0:
            raise ValueError("amplitude and f0 must be > 0")
        if self.ramp_periods <= 0:
            raise ValueError("ramp_periods must be > 0")
        if np.unique(idx, axis=0).shape[0] != idx.shape[0]:
            # Fancy-indexed "+=" does NOT accumulate duplicates (numpy buffered
            # semantics), so duplicate voxels would silently drop drive energy.
            raise ValueError(
                "source contains duplicate voxel indices; deduplicate (and decide "
                "the phase) before constructing the CWSource."
            )
        object.__setattr__(self, "indices", np.ascontiguousarray(idx.astype(np.int64)))
        object.__setattr__(self, "phases", np.ascontiguousarray(ph))

    @property
    def n_points(self) -> int:
        return int(self.indices.shape[0])

    @property
    def ndim(self) -> int:
        return int(self.indices.shape[1])

    def check_inside(self, grid: Grid) -> None:
        """Raise when any source voxel falls outside the active grid."""
        if self.ndim != grid.ndim:
            raise ValueError(f"source is {self.ndim}-D but grid is {grid.ndim}-D")
        upper = np.asarray(grid.shape)
        if (self.indices < 0).any() or (self.indices >= upper).any():
            bad = np.where((self.indices < 0).any(1) | (self.indices >= upper).any(1))[0]
            raise ValueError(
                f"{bad.size} source voxel(s) fall outside grid {grid.shape} "
                f"(first offender: {self.indices[bad[0]].tolist()})"
            )


def ramp_envelope(t: float, period: float, ramp_periods: float) -> float:
    """Cosine taper 0 -> 1 over ``ramp_periods`` periods (notebook drive)."""
    t_ramp = ramp_periods * period
    x = min(t / t_ramp, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


def plane_cw_source(
    grid: Grid,
    f0: float,
    amplitude: float,
    axis: int = 0,
    position_vox: int | None = None,
    phase: float = 0.0,
) -> CWSource:
    """Uniform-phase plane source on one grid plane (validation workhorse).

    Default position sits just inside the PML plus a small standoff.
    """
    if not 0 <= axis < grid.ndim:
        raise ValueError(f"axis {axis} invalid for a {grid.ndim}-D grid")
    pos = position_vox if position_vox is not None else grid.pml_vox + 8
    if not 0 <= pos < grid.shape[axis]:
        raise ValueError(f"plane position {pos} outside axis of length {grid.shape[axis]}")

    other = [np.arange(n) for i, n in enumerate(grid.shape) if i != axis]
    if other:
        mesh = np.meshgrid(*other, indexing="ij")
        n_pts = mesh[0].size
        cols = iter([m.reshape(-1) for m in mesh])
    else:
        n_pts = 1
        cols = iter([])
    idx = np.empty((n_pts, grid.ndim), dtype=np.int64)
    for d in range(grid.ndim):
        idx[:, d] = pos if d == axis else next(cols)
    return CWSource(
        indices=idx,
        phases=np.full(n_pts, phase, np.float32),
        amplitude=amplitude,
        f0=f0,
        label=f"plane(axis={axis}, pos={pos})",
    )


def bowl_cw_source(
    grid: Grid,
    f0: float,
    amplitude: float,
    aperture_radius: float,
    roc: float,
    apex_vox: tuple[int, ...],
    spacing: float | None = None,
) -> CWSource:
    """Focused spherical-cap source voxelized onto a 3-D grid.

    The cap axis is +z (last grid axis); ``apex_vox`` is the voxel of the
    bowl apex, the geometric focus lands at ``apex_vox + roc/dx`` along z.
    Sub-voxel cap sampling (default dx/2) is deduplicated to one voxel set,
    uniform phase (natural geometric focusing).
    """
    if grid.ndim != 3:
        raise ValueError("bowl_cw_source requires a 3-D grid")
    if len(apex_vox) != 3:
        raise ValueError(f"apex_vox must have 3 entries, got {apex_vox}")
    ds = grid.dx / 2.0 if spacing is None else float(spacing)
    points, _normals, _areas = spherical_cap_points(aperture_radius, roc, ds)
    idx = np.round(points / grid.dx).astype(np.int64) + np.asarray(apex_vox, np.int64)
    idx = np.unique(idx, axis=0)
    src = CWSource(
        indices=idx,
        phases=np.zeros(idx.shape[0], np.float32),
        amplitude=amplitude,
        f0=f0,
        label=f"bowl(a={aperture_radius * 1e3:.1f}mm, roc={roc * 1e3:.1f}mm)",
    )
    src.check_inside(grid)
    return src
