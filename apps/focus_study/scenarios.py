"""Scenario builders: everything needed to run one focus study.

Each builder returns a :class:`Setup` — grid, medium, source and the
geometric bookkeeping (apex/focus voxels, aperture) the analysis and the
figures need. Scenarios only use the public caustica API.

The beam axis is ALWAYS the last grid axis (z); the transducer apex sits
just inside the PML and the geometric focus lands at ``apex + roc/dx``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from caustica import Grid, Medium, PMLSpec
from caustica.arrays import archimedean_spiral
from caustica.geometry import Ball, HalfSpace, Scene
from caustica.materials import Material, MaterialDB, breast_default
from caustica.solvers import CWRunSpec
from caustica.sources import CWSource, bowl_cw_source


@dataclass
class Knobs:
    """CLI-tunable parameters shared by every scenario."""

    dx: float = 0.4e-3
    f0: float = 1.0e6
    amplitude: float = 1.0e5  # source pressure amplitude [Pa]
    solver: str = "westervelt"
    harmonics: tuple[int, ...] = (1, 2)
    min_settle_periods: int = 8
    max_settle_periods: int = 40
    convergence_tol: float = 0.01
    pml_mm: float = 4.0


@dataclass
class Setup:
    """A fully built, ready-to-run scenario."""

    name: str
    title: str
    description: str
    grid: Grid
    medium: Medium
    source: CWSource
    spec: CWRunSpec
    knobs: Knobs
    apex_vox: tuple[int, int, int]
    focus_vox: tuple[int, int, int]  # intended (steered/geometric) focus
    roc: float
    aperture_radius: float | None  # None => not a plain bowl (no O'Neil reference)
    labels: np.ndarray | None = None  # integer label map, for the medium figure
    label_names: dict[int, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def geometry_summary(self) -> dict[str, Any]:
        g = self.grid
        return {
            "shape": list(g.shape),
            "dx_mm": round(g.dx * 1e3, 4),
            "extent_mm": [round(e * 1e3, 2) for e in g.extent],
            "n_voxels": g.n_voxels,
            "pml_vox": g.pml_vox,
            "ppw_at_f0": round(g.ppw(self.knobs.f0, self.medium.c_min), 2),
            "ppw_at_2f0": round(g.ppw(2 * self.knobs.f0, self.medium.c_min), 2),
            "roc_mm": round(self.roc * 1e3, 2),
            "aperture_radius_mm": (
                None if self.aperture_radius is None else round(self.aperture_radius * 1e3, 2)
            ),
            "f_number": (
                None
                if self.aperture_radius is None
                else round(self.roc / (2 * self.aperture_radius), 2)
            ),
            "apex_vox": list(self.apex_vox),
            "focus_vox": list(self.focus_vox),
            "source_voxels": self.source.n_points,
            "c_min_max": [self.medium.c_min, self.medium.c_max],
            "linear_medium": self.medium.is_linear,
        }


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------
def _even(n: float) -> int:
    """Round up to an even voxel count (FFT-friendlier, keeps a true center)."""
    return int(np.ceil(n / 2.0) * 2)


def _axial_grid(
    knobs: Knobs, half_width_m: float, roc: float, tail_m: float, standoff_m: float = 1.5e-3
) -> tuple[Grid, tuple[int, int, int]]:
    """Grid sized around a bowl of curvature ``roc`` + the apex voxel.

    Transverse extent covers ``half_width_m`` plus PML on both sides; the
    axis runs PML -> standoff -> apex -> focus -> ``tail_m`` -> PML.
    """
    dx, pml_m = knobs.dx, knobs.pml_mm * 1e-3
    pml_vox = int(round(pml_m / dx))
    n_xy = _even(2 * (half_width_m / dx + pml_vox + 2))
    apex_z = pml_vox + int(round(standoff_m / dx))
    n_z = _even(apex_z + (roc + tail_m) / dx + pml_vox + 2)
    grid = Grid(shape=(n_xy, n_xy, n_z), dx=dx, pml=PMLSpec(thickness=pml_m))
    return grid, (n_xy // 2, n_xy // 2, apex_z)


def _spec(knobs: Knobs) -> CWRunSpec:
    return CWRunSpec(
        min_settle_periods=knobs.min_settle_periods,
        max_settle_periods=knobs.max_settle_periods,
        convergence_tol=knobs.convergence_tol,
    )


# ---------------------------------------------------------------------------
# Scenario 1 — focused bowl in water (has an analytic reference)
# ---------------------------------------------------------------------------
def water_bowl(knobs: Knobs) -> Setup:
    """Single focused bowl radiating into homogeneous water.

    The one scenario with a closed-form reference: the axial profile is
    compared against O'Neil (1949), so it doubles as a self-check that
    the install produces physically right answers.
    """
    a, roc = 12.0e-3, 30.0e-3
    grid, apex = _axial_grid(knobs, half_width_m=a + 4e-3, roc=roc, tail_m=12e-3)
    water_nl = Material(name="water", alpha_np_m=0.5, rho=1000.0, c=1500.0, beta=3.5)
    medium = Medium.homogeneous(grid.shape, water_nl)
    src = bowl_cw_source(
        grid,
        f0=knobs.f0,
        amplitude=knobs.amplitude,
        aperture_radius=a,
        roc=roc,
        apex_vox=apex,
    )
    focus = (apex[0], apex[1], apex[2] + int(round(roc / grid.dx)))
    return Setup(
        name="water_bowl",
        title="Focused bowl in water",
        description=(
            f"Single spherical-cap transducer (a={a * 1e3:.0f} mm, ROC={roc * 1e3:.0f} mm, "
            f"F/{roc / (2 * a):.2f}) driven CW into homogeneous water "
            "(alpha=0.5 Np/m, beta=3.5). Axial profile is checked against the "
            "O'Neil closed form."
        ),
        grid=grid,
        medium=medium,
        source=src,
        spec=_spec(knobs),
        knobs=knobs,
        apex_vox=apex,
        focus_vox=focus,
        roc=roc,
        aperture_radius=a,
        notes=(
            "O'Neil assumes a lossless linear medium; the comparison is on the "
            "NORMALIZED axial shape, not absolute pressure.",
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 2 — phased array, electronically steered focus
# ---------------------------------------------------------------------------
def spiral_array(knobs: Knobs, n_elements: int = 32, steer_mm: float = 4.0) -> Setup:
    """Archimedean-spiral phased array steered off-axis by delay-and-sum.

    Exercises the arrays package: geometry -> DAS phases -> voxelized
    multi-element source. The report then measures how far the realized
    focus actually moved (steering error is a real design question).
    """
    roc, d_outer, d_inner = 35.0e-3, 40.0e-3, 16.0e-3
    array = archimedean_spiral(
        n_elements=n_elements, d_outer=d_outer, d_inner=d_inner, roc=roc, active_fraction=0.60
    )
    grid, apex = _axial_grid(knobs, half_width_m=d_outer / 2 + 4e-3, roc=roc, tail_m=12e-3)
    target_m = np.array([steer_mm * 1e-3, 0.0, roc])  # apex frame
    phases = array.das_phases(target_m, f0=knobs.f0, c0=1500.0)
    arr_src = array.voxelize(
        grid, apex_vox=apex, f0=knobs.f0, amplitude=knobs.amplitude, phases=phases
    )
    medium = Medium.homogeneous(
        grid.shape, Material(name="water", alpha_np_m=0.5, rho=1000.0, c=1500.0, beta=3.5)
    )
    focus = (
        apex[0] + int(round(target_m[0] / grid.dx)),
        apex[1] + int(round(target_m[1] / grid.dx)),
        apex[2] + int(round(target_m[2] / grid.dx)),
    )
    if array.elem_radius / grid.dx < 2.0:
        raise ValueError(
            f"dx={grid.dx * 1e3:.2f} mm resolves the {array.elem_radius * 1e3:.2f} mm element "
            f"with only {array.elem_radius / grid.dx:.1f} voxels; use a finer --dx or "
            f"fewer elements."
        )
    return Setup(
        name="spiral_array",
        title=f"{n_elements}-element spiral array, steered {steer_mm:g} mm off axis",
        description=(
            f"Archimedean-spiral array (n={n_elements}, D={d_outer * 1e3:.0f}/"
            f"{d_inner * 1e3:.0f} mm, ROC={roc * 1e3:.0f} mm, element radius "
            f"{array.elem_radius * 1e3:.2f} mm) phased by delay-and-sum onto a target "
            f"{steer_mm:g} mm off the axis, in water."
        ),
        grid=grid,
        medium=medium,
        source=arr_src.source,
        spec=_spec(knobs),
        knobs=knobs,
        apex_vox=apex,
        focus_vox=focus,
        roc=roc,
        aperture_radius=None,
        notes=(
            f"{arr_src.n_elements_represented}/{n_elements} elements survived voxelization "
            f"at dx={grid.dx * 1e3:.2f} mm.",
            "No analytic reference for a steered sparse array; the report measures the "
            "realized focus against the requested target.",
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 3 — layered tissue with a lesion target (CSG geometry)
# ---------------------------------------------------------------------------
def layered_tissue(knobs: Knobs) -> Setup:
    """Bowl focused through gel -> skin -> fat onto a muscle nodule.

    Exercises the M6b geometry system (CSG scene -> label map -> Medium
    via the breast MaterialDB) and shows what heterogeneity does to the
    focus: attenuation, speed-of-sound shift, focal displacement.
    """
    a, roc = 12.0e-3, 30.0e-3
    grid, apex = _axial_grid(knobs, half_width_m=a + 4e-3, roc=roc, tail_m=12e-3)
    dx = grid.dx
    # Scene coordinates: z measured from the apex plane, x/y from the axis.
    origin = (-apex[0] * dx, -apex[1] * dx, -apex[2] * dx)
    z_skin, z_fat = 10.0e-3, 12.0e-3
    scene = Scene(ndim=3, background=4)  # 4 = coupling gel
    scene.add(HalfSpace(point=(0.0, 0.0, z_skin), normal=(0.0, 0.0, -1.0)), label=1)  # skin
    scene.add(HalfSpace(point=(0.0, 0.0, z_fat), normal=(0.0, 0.0, -1.0)), label=2)  # fat
    scene.add(Ball(center=(0.0, 0.0, roc), radius=5.0e-3), label=3)  # muscle nodule
    db: MaterialDB = breast_default()
    labels = scene.rasterize(grid, origin=origin, supersample=3).labels
    medium = Medium.from_id_map(labels.astype(np.int64), db)
    src = bowl_cw_source(
        grid,
        f0=knobs.f0,
        amplitude=knobs.amplitude,
        aperture_radius=a,
        roc=roc,
        apex_vox=apex,
    )
    focus = (apex[0], apex[1], apex[2] + int(round(roc / dx)))
    return Setup(
        name="layered_tissue",
        title="Bowl focused through skin/fat onto a muscle nodule",
        description=(
            "CSG scene: coupling gel standoff, 2 mm skin layer at z=10 mm, fat beyond, "
            "and a 10 mm muscle nodule centred on the geometric focus. Materials from "
            "the breast phantom table (breast_default)."
        ),
        grid=grid,
        medium=medium,
        source=src,
        spec=_spec(knobs),
        knobs=knobs,
        apex_vox=apex,
        focus_vox=focus,
        roc=roc,
        aperture_radius=a,
        labels=labels,
        label_names={int(i): db[int(i)].name for i in np.unique(labels)},
        notes=(
            "Heterogeneous: the realized focus generally does NOT sit at the geometric "
            "focus (speed-of-sound refraction + attenuation weighting).",
            "v1 absorption is exponential at f0 only — harmonics are absorbed with the "
            "same alpha, so second-harmonic levels here are optimistic (M16 fixes this).",
        ),
    )


BUILDERS: dict[str, Callable[[Knobs], Setup]] = {
    "water_bowl": water_bowl,
    "spiral_array": spiral_array,
    "layered_tissue": layered_tissue,
}


def build(name: str, knobs: Knobs) -> Setup:
    try:
        return BUILDERS[name](knobs)
    except KeyError:
        raise SystemExit(
            f"unknown scenario '{name}'; available: {', '.join(sorted(BUILDERS))}"
        ) from None
