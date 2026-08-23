"""Focal metrics — the single source of truth (M10d).

Everything here was extracted from ``apps/focus_study/analysis.py`` so the
library's ``caustica report`` command and the focus_study app compute the SAME
numbers from the same definitions (peak, -6 dB spot, gain, harmonic ratios).
focus_study now delegates to this module; changing a definition here changes
both consumers at once — that is the point.

All functions work on the solver's RECORD REGION, which may be smaller than
the grid — voxel indices are translated through ``result.region`` so the
numbers stay in the grid frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from caustica.solvers.base import SolverResult

HALF_PRESSURE = 0.5  # -6 dB in pressure

#: An ``a2`` maximum closer than this (in voxels) to the PML face is reported
#: as suspect. ``a2_peak_pa`` is a whole-interior maximum, and the harmonic
#: DFT leaves a residue against the absorber that can beat the real focal
#: second harmonic outright in a weakly nonlinear run — 6.5% of the
#: fundamental, 4 voxels from the edge, in the beta=0 sweep that found this
#: (janitor ticket 09). The value stays as measured; only the caveat is new.
A2_PML_MARGIN_WARN_VOX = 8


@dataclass(frozen=True)
class FieldFrame:
    """Where a solved field sits on the grid — the frame every metric needs.

    ``dx``/``grid_shape``/``pml_vox``/``apex_vox`` (plus the requested
    ``focus_vox``) used to travel as four or five separate keyword arguments
    through every metric, preview and figure signature; ``focus_metrics``
    alone reached eight parameters and every caller had to re-spell the same
    quartet. One frozen object instead: adding a member (an origin, say) is
    one field, not six signatures (janitor ticket 03).

    ``apex_vox`` is the voxel that mm positions are measured FROM (the
    transducer apex; the grid origin when a result file predates the stamp),
    ``focus_vox`` the REQUESTED focus — ``None`` when nothing was asked for,
    which is what makes the ``target`` metrics section optional.
    """

    dx: float
    grid_shape: tuple[int, ...]
    pml_vox: int
    apex_vox: tuple[int, int, int]
    focus_vox: tuple[int, int, int] | None = None

    @classmethod
    def from_geometry(cls, geo: Mapping[str, Any]) -> FieldFrame:
        """Build from a geometry mapping (``store._geometry``, the facade's).

        A result file's self-description and a built job's geometry carry
        exactly these keys (among others), so every reader spells the frame
        the same way instead of unpacking the mapping by hand five times.
        """
        focus = geo.get("focus_vox")
        return cls(
            dx=float(geo["dx"]),
            grid_shape=tuple(int(v) for v in geo["grid_shape"]),
            pml_vox=int(geo["pml_vox"]),
            apex_vox=tuple(int(v) for v in geo["apex_vox"]),  # type: ignore[arg-type]
            focus_vox=None if focus is None else tuple(int(v) for v in focus),  # type: ignore[arg-type]
        )


def region_origin(result: SolverResult) -> tuple[int, ...]:
    """Index of the record region's first voxel in the grid frame."""
    return tuple(int(sl.start) for sl in result.region)


def to_region(idx: tuple[int, ...], result: SolverResult) -> tuple[int, ...]:
    org = region_origin(result)
    return tuple(int(i) - o for i, o in zip(idx, org, strict=True))


def to_grid(idx: tuple[int, ...], result: SolverResult) -> tuple[int, ...]:
    org = region_origin(result)
    return tuple(int(i) + o for i, o in zip(idx, org, strict=True))


def interior_slices(
    result: SolverResult,
    frame: FieldFrame,
    *,
    extra: int = 2,
) -> tuple[slice, ...]:
    """Record-region slices with the PML (plus a margin) shaved off.

    Peak hunting must never look inside the sponge: the field there is
    damped and carries the absorber's own artefacts, so an unrestricted
    argmax can report a "harmonic maximum" sitting on the PML edge.
    """
    org = region_origin(result)
    grid_shape = frame.grid_shape
    margin = frame.pml_vox + extra
    out = []
    for d, (o, n) in enumerate(zip(org, result.amp.shape, strict=True)):
        lo = max(0, margin - o)
        hi = min(n, grid_shape[d] - margin - o)
        if hi <= lo:
            raise ValueError(
                f"axis {d}: the record region lies entirely inside the PML margin "
                f"({margin} voxels); nothing to analyze."
            )
        out.append(slice(lo, hi))
    return tuple(out)


def argmax_interior(field: np.ndarray, interior: tuple[slice, ...]) -> tuple[int, ...]:
    """argmax restricted to ``interior``, returned in the full-array frame."""
    sub = field[interior]
    loc = np.unravel_index(int(np.argmax(sub)), sub.shape)
    return tuple(int(i) + sl.start for i, sl in zip(loc, interior, strict=True))


def _frac_crossing(coord: np.ndarray, prof: np.ndarray, i_peak: int, level: float) -> float | None:
    """Coordinate where ``prof`` first drops below ``level`` right of ``i_peak``.

    Linear interpolation between the bracketing samples; ``None`` when the
    profile never drops below the level inside the recorded region (an
    honest "cannot measure" instead of a number pinned to the boundary).
    """
    for i in range(i_peak, prof.size - 1):
        if prof[i + 1] < level <= prof[i]:
            t = (prof[i] - level) / (prof[i] - prof[i + 1])
            return float(coord[i] + t * (coord[i + 1] - coord[i]))
    return None


def extent_6db(coord: np.ndarray, prof: np.ndarray, i_peak: int) -> dict:
    """-6 dB (half-pressure) extent around the peak of a 1-D profile."""
    level = HALF_PRESSURE * float(prof[i_peak])
    right = _frac_crossing(coord, prof, i_peak, level)
    left_rev = _frac_crossing(coord[::-1], prof[::-1], prof.size - 1 - i_peak, level)
    if right is None or left_rev is None:
        return {"left_mm": None, "right_mm": None, "width_mm": None, "truncated": True}
    return {
        "left_mm": round(left_rev * 1e3, 3),
        "right_mm": round(right * 1e3, 3),
        "width_mm": round((right - left_rev) * 1e3, 3),
        "truncated": False,
    }


def intensity_w_cm2(pressure_pa: float, rho: float, c: float) -> float:
    """Plane-wave CW intensity I = p^2 / (2 rho c) in W/cm^2."""
    return float(pressure_pa**2 / (2.0 * rho * c) / 1e4)


def mm_axes(result: SolverResult, frame: FieldFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Region-frame voxel coordinates in mm, measured from the apex voxel."""
    org = region_origin(result)
    shp = result.amp.shape
    dx, apex_vox = frame.dx, frame.apex_vox
    x = (np.arange(shp[0]) + org[0] - apex_vox[0]) * dx * 1e3
    y = (np.arange(shp[1]) + org[1] - apex_vox[1]) * dx * 1e3
    z = (np.arange(shp[2]) + org[2] - apex_vox[2]) * dx * 1e3
    return x, y, z


def pml_edge_distance(voxel_grid: tuple[int, ...], frame: FieldFrame) -> int:
    """Voxels between a GRID-frame voxel and the nearest PML face (min over axes).

    The sponge occupies ``[0, pml_vox)`` and ``[n - pml_vox, n)`` on each
    axis, so a voxel sitting on the first non-absorbing plane is at distance
    0 and the number grows towards the middle of the grid.
    """
    pml = frame.pml_vox
    return min(
        min(int(i) - pml, (n - pml - 1) - int(i))
        for i, n in zip(voxel_grid, frame.grid_shape, strict=True)
    )


def a2_edge_warning(distance_vox: int) -> str:
    """The text for an a2 maximum that is sitting on the absorber (ticket 09)."""
    return (
        f"the a2 peak lies {distance_vox} voxels from the PML edge "
        f"(< {A2_PML_MARGIN_WARN_VOX}) — likely a harmonic-DFT edge artifact rather "
        f"than physics (an edge residue grows LINEARLY with drive, a real second "
        f"harmonic quadratically); a2_at_fundamental_peak_pa is the trustworthy number."
    )


def focus_metrics(
    result: SolverResult,
    frame: FieldFrame,
    *,
    source_amplitude: float | None = None,
    medium: Any | None = None,
    solver: str | None = None,
) -> dict:
    """All scalar focal metrics for one run (JSON-ready).

    The optional inputs degrade honestly instead of guessing:

    * ``frame.focus_vox`` absent -> no ``target`` section (none was requested)
    * ``source_amplitude``  -> ``gain_vs_source`` is ``None``
    * ``medium`` absent     -> ``isppa_w_cm2`` is ``None`` (rho/c unknown —
      a result file does not carry the medium)
    """
    amp = result.amp
    dx, apex_vox, focus_vox = frame.dx, frame.apex_vox, frame.focus_vox
    apex_z = apex_vox[2]

    interior = interior_slices(result, frame)
    pk_r = argmax_interior(amp, interior)
    pk_g = to_grid(pk_r, result)
    p_peak = float(amp[pk_r])

    # Profiles through the realized peak (region frame, mm in the grid frame).
    z_coord = (np.arange(amp.shape[2]) + region_origin(result)[2] - apex_z) * dx
    x_coord = (np.arange(amp.shape[0]) + region_origin(result)[0] - apex_vox[0]) * dx
    y_coord = (np.arange(amp.shape[1]) + region_origin(result)[1] - apex_vox[1]) * dx
    prof_z = amp[pk_r[0], pk_r[1], :]
    prof_x = amp[:, pk_r[1], pk_r[2]]
    prof_y = amp[pk_r[0], :, pk_r[2]]

    focal_volume_mm3 = float((amp[interior] >= HALF_PRESSURE * p_peak).sum() * (dx * 1e3) ** 3)

    if medium is not None:
        rho = float(medium.rho[pk_g])
        c_loc = float(medium.c[pk_g])
        isppa = round(intensity_w_cm2(p_peak, rho, c_loc), 1)
    else:
        isppa = None

    metrics: dict = {
        "peak": {
            "p_pa": round(p_peak, 1),
            "p_mpa": round(p_peak / 1e6, 4),
            "voxel_grid": [int(i) for i in pk_g],
            "position_mm_from_apex": {
                "x": round(float(x_coord[pk_r[0]]) * 1e3, 3),
                "y": round(float(y_coord[pk_r[1]]) * 1e3, 3),
                "z": round(float(z_coord[pk_r[2]]) * 1e3, 3),
            },
            "gain_vs_source": (
                None if source_amplitude is None else round(p_peak / source_amplitude, 2)
            ),
            "isppa_w_cm2": isppa,
            "p_max_time_domain_pa": round(float(result.p_max[pk_r]), 1),
        },
    }

    if focus_vox is not None:
        tgt_r = to_region(focus_vox, result)
        in_region = all(0 <= i < n for i, n in zip(tgt_r, amp.shape, strict=True))
        p_target = float(amp[tgt_r]) if in_region else float("nan")
        metrics["target"] = {
            "voxel_grid": [int(i) for i in focus_vox],
            "p_pa": None if np.isnan(p_target) else round(p_target, 1),
            "hit_ratio": None if np.isnan(p_target) else round(p_target / p_peak, 4),
            "displacement_mm": {
                "x": round((pk_g[0] - focus_vox[0]) * dx * 1e3, 3),
                "y": round((pk_g[1] - focus_vox[1]) * dx * 1e3, 3),
                "z": round((pk_g[2] - focus_vox[2]) * dx * 1e3, 3),
            },
            "displacement_norm_mm": round(
                float(np.linalg.norm(np.array(pk_g) - np.array(focus_vox)) * dx * 1e3), 3
            ),
        }

    metrics["focal_spot"] = {
        "axial_6db": extent_6db(z_coord, prof_z, int(pk_r[2])),
        "lateral_x_6db": extent_6db(x_coord, prof_x, int(pk_r[0])),
        "lateral_y_6db": extent_6db(y_coord, prof_y, int(pk_r[1])),
        "volume_above_6db_mm3": round(focal_volume_mm3, 3),
    }
    metrics["run"] = {
        "solver": solver if solver is not None else str(result.meta.get("solver", "")),
        "dt_s": result.dt,
        "steps_per_period": result.spp,
        "steps_total": result.steps_total,
        "t_end_us": round(result.t_end_s * 1e6, 3),
        "tof_periods": result.tof_periods,
        "converged_period": result.converged_period,
        "settle_capped": bool(result.settle_capped),
        "harmonics_recorded": sorted(result.phasors),
    }

    warns: list[str] = []
    if 2 in result.phasors:
        a2 = result.harmonic_amp(2)
        a1 = result.harmonic_amp(1) if 1 in result.phasors else amp
        pk2 = argmax_interior(a2, interior)
        pk2_g = to_grid(pk2, result)
        edge_vox = pml_edge_distance(pk2_g, frame)
        metrics["harmonics"] = {
            "a2_at_fundamental_peak_pa": round(float(a2[pk_r]), 1),
            "a2_over_a1_at_peak_pct": round(100.0 * float(a2[pk_r]) / float(a1[pk_r]), 2),
            "a2_peak_pa": round(float(a2[pk2]), 1),
            "a2_peak_voxel_grid": [int(i) for i in pk2_g],
            "a2_peak_distance_to_pml_vox": edge_vox,
        }
        if edge_vox < A2_PML_MARGIN_WARN_VOX:
            warns.append(a2_edge_warning(edge_vox))

    # Always written, even when empty: a reader must be able to tell "nothing
    # to warn about" from "this file predates the channel" (D31, as plan.json
    # writes its ppw_warnings unconditionally).
    metrics["warnings"] = warns
    return metrics


def axial_profiles(result: SolverResult, frame: FieldFrame) -> dict[str, np.ndarray]:
    """Coordinate/profile arrays used by the figures (mm, Pa)."""
    amp = result.amp
    org = region_origin(result)
    dx, apex_vox, pml_vox = frame.dx, frame.apex_vox, frame.pml_vox
    pk = argmax_interior(amp, interior_slices(result, frame))
    z = (np.arange(amp.shape[2]) + org[2] - apex_vox[2]) * dx * 1e3
    x = (np.arange(amp.shape[0]) + org[0] - apex_vox[0]) * dx * 1e3
    out = {
        "z_mm": z,
        "x_mm": x,
        "axial": amp[pk[0], pk[1], :],
        "lateral": amp[:, pk[1], pk[2]],
        "peak_idx": np.array(pk),
        "pml_mm": np.array([pml_vox * dx * 1e3]),
    }
    if 2 in result.phasors:
        a2 = result.harmonic_amp(2)
        out["axial_h2"] = a2[pk[0], pk[1], :]
    return out
