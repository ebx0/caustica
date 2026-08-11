"""Transducer array geometry, phasing and grid voxelization.

Frame convention (shared with :mod:`hifusim.analytic`): array apex at the
origin, beam axis +z, geometric focus at ``(0, 0, focal_length)``. Element
positions lie on the spherical shell of radius ``focal_length``; normals
point at the focus.

The Archimedean-spiral builder is a verbatim port of the production
notebook's ``build_spiral_array_128`` (equal-arc-length element placement
along a spiral wound between the inner and outer aperture radii), kept
parameter-generic so small test arrays and the 128-element production
geometry come from the same code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from hifusim.analytic.rayleigh import rayleigh_pressure
from hifusim.core.grid import Grid
from hifusim.sources import CWSource


@dataclass(frozen=True)
class TransducerArray:
    """Element centers on a spherical cap + circular-piston element radius."""

    positions: np.ndarray  # (n, 3) [m], apex frame
    normals: np.ndarray  # (n, 3) unit, toward the focus
    elem_radius: float  # [m]
    focal_length: float  # [m] (radius of curvature)

    def __post_init__(self) -> None:
        pos = np.asarray(self.positions, np.float64)
        nrm = np.asarray(self.normals, np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3 or nrm.shape != pos.shape:
            raise ValueError(f"positions/normals must be matching (n, 3), got {pos.shape}")
        if self.elem_radius <= 0 or self.focal_length <= 0:
            raise ValueError("elem_radius and focal_length must be > 0")
        # Copies, not views: a frozen geometry must not shift when the caller
        # mutates the arrays it built us from (review finding, 2026-08-11).
        object.__setattr__(self, "positions", np.array(pos, dtype=np.float64, copy=True))
        object.__setattr__(self, "normals", np.array(nrm, dtype=np.float64, copy=True))

    @property
    def n_elements(self) -> int:
        return int(self.positions.shape[0])

    @property
    def focus(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.focal_length])

    # ---------- phasing ----------

    def das_phases(self, target_m: np.ndarray, f0: float, c0: float = 1500.0) -> np.ndarray:
        """Delay-and-sum focusing phases for a target point (apex frame, [m]).

        Port of the notebook's ``calc_das_phases``: phase = -k * distance,
        wrapped to [0, 2 pi) and offset so min(phase) = 0.
        """
        k0 = 2.0 * np.pi * f0 / c0
        d = np.linalg.norm(self.positions - np.asarray(target_m, np.float64), axis=1)
        phase = (-k0 * d) % (2.0 * np.pi)
        phase -= phase.min()
        return phase.astype(np.float32)

    # ---------- fast previews (no solver) ----------

    def rayleigh_preview(
        self,
        field_points: np.ndarray,
        f0: float,
        phases: np.ndarray | None = None,
        c0: float = 1500.0,
        u0: float = 1.0,
    ) -> np.ndarray:
        """Complex CW pressure at ``field_points`` from element-center pistons.

        Elements are collapsed to points carrying their full piston area —
        accurate away from the aperture (>= a few element radii), which is
        exactly the array-design use case. This is the future GUI's live
        beam preview and the KZK initial-plane projector (M9).
        """
        ph = np.zeros(self.n_elements) if phases is None else np.asarray(phases)
        v_n = u0 * np.exp(1j * ph.astype(np.float64))
        areas = np.full(self.n_elements, np.pi * self.elem_radius**2)
        k = 2.0 * np.pi * f0 / c0
        return rayleigh_pressure(self.positions, areas, v_n, field_points, k=k, c=c0)

    # ---------- grid voxelization ----------

    def voxelize(
        self,
        grid: Grid,
        apex_vox: tuple[int, int, int],
        f0: float,
        amplitude: float,
        phases: np.ndarray | None = None,
    ) -> ArraySource:
        """Project elements onto the grid as 1-voxel curved shells.

        Port of the notebook's source-placement loop: each element becomes a
        disc of in-plane radius ``elem_radius`` whose z-offset follows the
        element's normal plane; overlapping voxels keep the FIRST element's
        phase (deduplicated exactly like the notebook's ``np.unique``).
        """
        if grid.ndim != 3:
            raise ValueError("voxelize requires a 3-D grid")
        ph = np.zeros(self.n_elements, np.float32) if phases is None else phases
        ph = np.asarray(ph, np.float32)
        if ph.shape != (self.n_elements,):
            raise ValueError(f"phases must be ({self.n_elements},), got {ph.shape}")

        dx_mm = grid.dx * 1e3
        pos_mm = self.positions * 1e3
        r_vox = int(np.ceil(self.elem_radius * 1e3 / dx_mm)) + 1
        rows: list[tuple[int, int, int]] = []
        elem_ids: list[int] = []
        for i in range(self.n_elements):
            p0, nv = pos_mm[i], self.normals[i]
            ex = int(np.round(p0[0] / dx_mm)) + apex_vox[0]
            ey = int(np.round(p0[1] / dx_mm)) + apex_vox[1]
            ez = int(np.round(p0[2] / dx_mm)) + apex_vox[2]
            # Verbatim notebook port: the disc test runs in the xy-plane and
            # z is sheared onto the element plane afterwards, so a tilted
            # element's patch has area ~pi r^2 / cos(tilt) (~14% extra at the
            # production rim, ~28 deg). Kept for dataset parity; an
            # in-plane-projected test is the M12 candidate fix.
            for ox in range(-r_vox, r_vox + 1):
                for oy in range(-r_vox, r_vox + 1):
                    if (ox * dx_mm) ** 2 + (oy * dx_mm) ** 2 <= (self.elem_radius * 1e3) ** 2:
                        oz = (
                            int(np.round(-(nv[0] * ox + nv[1] * oy) / nv[2]))
                            if abs(nv[2]) > 1e-6
                            else 0
                        )
                        rows.append((ex + ox, ey + oy, ez + oz))
                        elem_ids.append(i)
        idx = np.asarray(rows, np.int64)
        elem = np.asarray(elem_ids, np.int32)
        uniq, first = np.unique(idx, axis=0, return_index=True)
        elem_of_voxel = elem[first]

        missing = sorted(set(range(self.n_elements)) - set(elem_of_voxel.tolist()))
        if missing:
            raise ValueError(
                f"{len(missing)} element(s) lost all voxels to deduplication "
                f"(first: {missing[:5]}); the grid dx is too coarse for this "
                f"element radius/pitch."
            )
        source = CWSource(
            indices=uniq,
            phases=ph[elem_of_voxel],
            amplitude=amplitude,
            f0=f0,
            label=f"array(n={self.n_elements}, r_elem={self.elem_radius * 1e3:.2f}mm)",
        )
        source.check_inside(grid)
        return ArraySource(source=source, element_of_voxel=elem_of_voxel)


@dataclass(frozen=True)
class ArraySource:
    """A voxelized array: the CWSource plus per-voxel element ownership."""

    source: CWSource
    element_of_voxel: np.ndarray  # (n_voxels,) int32

    @property
    def n_elements_represented(self) -> int:
        return int(np.unique(self.element_of_voxel).size)


def archimedean_spiral(
    n_elements: int = 128,
    d_outer: float = 0.100,
    d_inner: float = 0.044,
    roc: float = 0.100,
    active_fraction: float = 0.60,
    n_arc_samples: int = 10000,
) -> TransducerArray:
    """Spherically curved Archimedean-spiral phased array (notebook port).

    Elements are placed at equal arc-length intervals along a spiral wound
    from ``r_start`` to ``r_end`` (aperture radii inset by one element
    radius); the element radius comes from dividing the active shell area
    equally. Defaults reproduce the production 128-element geometry.
    """
    if not 0 < d_inner < d_outer <= 2 * roc:
        raise ValueError(
            f"need 0 < d_inner < d_outer <= 2*roc, got d_inner={d_inner}, "
            f"d_outer={d_outer}, roc={roc}"
        )
    if not 0 < active_fraction <= 1:
        raise ValueError(f"active_fraction must be in (0, 1], got {active_fraction}")
    r_out, r_in = d_outer / 2.0, d_inner / 2.0
    cap_h = roc - np.sqrt(roc**2 - r_out**2)
    hole_h = roc - np.sqrt(roc**2 - r_in**2)
    active_area = 2.0 * np.pi * roc * (cap_h - hole_h) * active_fraction
    elem_radius = np.sqrt((active_area / n_elements) / np.pi)
    r_start, r_end = r_in + elem_radius, r_out - elem_radius
    if r_start >= r_end:
        raise ValueError(
            f"element radius {elem_radius * 1e3:.2f} mm does not fit between the "
            f"apertures; reduce n_elements or active_fraction."
        )
    turns = np.sqrt(((r_end - r_start) * n_elements) / (np.pi * (r_end + r_start)))
    b = (r_end - r_start) / (2.0 * np.pi * turns)
    theta_hr = np.linspace(0.0, 2.0 * np.pi * turns, n_arc_samples)
    arc_len = cumulative_trapezoid(
        np.sqrt((r_start + b * theta_hr) ** 2 + b**2), theta_hr, initial=0
    )
    theta = interp1d(arc_len, theta_hr, kind="linear")(np.linspace(0.0, arc_len[-1], n_elements))
    r = r_start + b * theta
    positions = np.column_stack(
        (r * np.cos(theta), r * np.sin(theta), roc - np.sqrt(roc**2 - r**2))
    )
    normals = np.array([0.0, 0.0, roc]) - positions
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return TransducerArray(
        positions=positions, normals=normals, elem_radius=float(elem_radius), focal_length=roc
    )
