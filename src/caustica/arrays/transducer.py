"""Transducer array geometry, phasing and grid voxelization.

Frame convention (shared with :mod:`caustica.analytic`): array apex at the
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

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from caustica.analytic.rayleigh import rayleigh_pressure
from caustica.core.backend import CausticaWarning
from caustica.core.grid import Grid
from caustica.sources import CWSource

log = logging.getLogger("caustica")


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
        *,
        discretization: str = "offgrid",
        bli_tolerance: float = 0.2,
        upsampling: int = 10,
    ) -> ArraySource:
        """Project elements onto the grid.

        ``discretization`` chooses how:

        ``"offgrid"`` (default)
            Each element is a flat disc of area ``pi r^2`` sampled at its own
            position in metres and deposited through a band-limited
            interpolant, and overlapping contributions are summed as complex
            phasors rather than one of them being discarded. See
            :func:`~caustica.arrays.transducer.TransducerArray._offgrid`.

        ``"binary"``
            The pre-2026-08-24 behaviour, kept for reproducing older results:
            each element becomes a disc of in-plane radius ``elem_radius``
            whose z-offset follows the element's normal plane, and overlapping
            voxels keep the FIRST element's phase (deduplicated exactly like
            the notebook's ``np.unique``). Its element CENTRES are rounded to
            voxels, which is what costs the focus 12-18 % of its coherent sum.
        """
        if grid.ndim != 3:
            raise ValueError("voxelize requires a 3-D grid")
        if discretization not in ("offgrid", "binary"):
            raise ValueError(
                f"discretization must be 'offgrid' or 'binary', got {discretization!r}"
            )
        ph = np.zeros(self.n_elements, np.float32) if phases is None else phases
        ph = np.asarray(ph, np.float32)
        if ph.shape != (self.n_elements,):
            raise ValueError(f"phases must be ({self.n_elements},), got {ph.shape}")

        if discretization == "offgrid":
            return self._offgrid(grid, apex_vox, f0, amplitude, ph, bli_tolerance, upsampling)

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
            label=f"array(n={self.n_elements}, r_elem={self.elem_radius * 1e3:.2f}mm) binary",
        )
        source.check_inside(grid)
        return ArraySource(source=source, element_of_voxel=elem_of_voxel)

    def _offgrid(
        self,
        grid: Grid,
        apex_vox: tuple[int, int, int],
        f0: float,
        amplitude: float,
        ph: np.ndarray,
        tolerance: float,
        upsampling: int,
    ) -> ArraySource:
        """Elements at their own positions, summed as phasors.

        Three things the binary port gets wrong, and this does not.

        *Position.* ``voxelize`` rounds each element centre to a voxel, so
        every element's path to the focus is off by up to half a voxel. That
        is a phase error, it is different for every element, and it does not
        average out — it defocuses. Measured on the production 128-element
        spiral at dx = 0.5 mm and 1 MHz: 0.61 rad rms, which costs 17.6 % of
        the coherent focal sum, matching ``exp(-sigma^2/2)`` to three
        decimals. Here the quadrature points are placed in metres and the
        interpolant carries the sub-voxel offset.

        *Area.* A binary disc's voxel count is not ``pi r^2 / dx^2``: the
        notebook's in-plane test sheared onto the element plane inflates a
        tilted element by ``1/cos(tilt)`` (8 % on average at production tilts)
        while voxel quantization of a small disc takes some back. The net
        landed at +4.4 % on the production array and -2.5 % on a smaller one —
        wrong by a few percent, with a sign that depends on the grid. Each
        element now carries ``pi r^2`` exactly.

        *Overlap.* Where two elements claimed a voxel the binary path kept the
        first one's phase and dropped the other's drive (1.7 % of the pairs on
        the production array, up to 5.8 % of a single element). Contributions
        are summed here as complex phasors, which is what superposing two
        drives at one point means: ``sum_i w_i sin(wt - phi_i)`` is exactly
        ``|S| sin(wt - Phi)`` for ``S = sum_i w_i exp(-i phi_i)``.
        """
        from caustica.geometry.offgrid import band_limited_weights, disc_points  # noqa: PLC0415

        dx = grid.dx
        r_vox = self.elem_radius / dx
        if r_vox < 0.5:
            # The binary path refused this as "elements lost all voxels to
            # deduplication". There is no deduplication here — overlapping
            # elements superpose, which is what two real elements would do —
            # so the refusal is stated as what it always meant: the grid
            # cannot tell these elements apart.
            raise ValueError(
                f"element radius is {self.elem_radius * 1e3:g} mm = {r_vox:.2f} voxels at "
                f"dx={dx * 1e3:g} mm. Below half a voxel an element has no shape the grid can "
                f"carry and neighbouring elements merge into one patch. Refine dx (need "
                f"dx <= {2 * self.elem_radius * 1e3:g} mm) or use larger elements."
            )
        n_q = max(int(np.ceil(np.pi * r_vox**2 * upsampling)), 16)
        area_grid = np.pi * r_vox**2  # element area in grid squares
        apex = np.asarray(apex_vox, dtype=np.float64)

        acc: dict[tuple[int, int, int], complex] = {}
        best: dict[tuple[int, int, int], tuple[float, int]] = {}
        dropped = 0.0
        for i in range(self.n_elements):
            pts = disc_points(self.positions[i], self.normals[i], self.elem_radius, n_q)
            dep = band_limited_weights(
                grid.shape, pts / dx + apex, area_grid / n_q, tolerance=tolerance
            )
            dropped += dep.dropped
            rot = complex(np.cos(-ph[i]), np.sin(-ph[i]))
            for key, w in zip(map(tuple, dep.indices.tolist()), dep.weights, strict=True):
                acc[key] = acc.get(key, 0j) + w * rot
                mag = abs(float(w))
                if key not in best or mag > best[key][0]:
                    best[key] = (mag, i)

        if not acc:
            raise ValueError(
                f"the array deposited nothing inside grid {grid.shape}; check apex_vox"
            )
        keys = sorted(acc)
        idx = np.asarray(keys, dtype=np.int64)
        s = np.asarray([acc[k] for k in keys], dtype=np.complex128)
        total = float(np.abs(s).sum()) or 1.0
        if dropped / (self.n_elements * area_grid) > 1e-6:
            warnings.warn(
                f"{100 * dropped / (self.n_elements * area_grid):.2f}% of this array's drive "
                f"falls outside grid {grid.shape}: the band-limited elements reach a couple of "
                f"voxels beyond their discs. Enlarge the grid or move the apex inward.",
                CausticaWarning,
                stacklevel=3,
            )
        elem_of_voxel = np.asarray([best[k][1] for k in keys], dtype=np.int32)
        source = CWSource(
            indices=idx,
            # sum_i w_i sin(wt - phi_i) == |S| sin(wt - Phi), Phi = -arg(S)
            phases=np.angle(np.conj(s)).astype(np.float32),
            weights=np.abs(s).astype(np.float32),
            amplitude=amplitude,
            f0=f0,
            label=f"array(n={self.n_elements}, r_elem={self.elem_radius * 1e3:.2f}mm)",
        )
        source.check_inside(grid)
        log.debug("off-grid array: %d points, |drive| total %.1f grid squares", len(idx), total)
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
