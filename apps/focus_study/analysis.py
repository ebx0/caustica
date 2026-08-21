"""Post-processing: focal metrics, profiles and the analytic cross-check.

Since M10d the metric DEFINITIONS live in :mod:`caustica.report.metrics`
(single source of truth — the runner's ``caustica report`` uses the same
functions); this module keeps only the focus_study-specific parts: the
``Setup``-flavoured wrappers and the O'Neil analytic cross-check, which
needs scenario knowledge (aperture, ROC) no result file carries.
"""

from __future__ import annotations

import numpy as np

from apps.focus_study.scenarios import Setup
from caustica.analytic import axial_pressure
from caustica.report import metrics as _metrics
from caustica.report.metrics import (
    axial_profiles,
    focus_metrics,
    region_origin,
    to_region,
)
from caustica.solvers import SolverResult

__all__ = ["interior_slices", "analyze", "profiles", "region_origin"]


def interior_slices(setup: Setup, result: SolverResult, extra: int = 2) -> tuple[slice, ...]:
    """Record-region slices with the PML (plus a margin) shaved off."""
    return _metrics.interior_slices(
        result, grid_shape=setup.grid.shape, pml_vox=setup.grid.pml_vox, extra=extra
    )


def analyze(setup: Setup, result: SolverResult) -> dict:
    """All scalar metrics for one run (JSON-ready)."""
    metrics = focus_metrics(
        result,
        dx=setup.grid.dx,
        grid_shape=setup.grid.shape,
        pml_vox=setup.grid.pml_vox,
        apex_vox=setup.apex_vox,
        focus_vox=setup.focus_vox,
        source_amplitude=setup.knobs.amplitude,
        medium=setup.medium,
        solver=setup.knobs.solver,
    )

    # --- analytic cross-check ------------------------------------------
    # Only for a plain bowl in a HOMOGENEOUS medium: O'Neil knows nothing
    # about layers, so running it against an aberrated field would dress up
    # a physical mismatch as a solver error.
    homogeneous = setup.medium.c_max == setup.medium.c_min
    if setup.aperture_radius is not None and homogeneous:
        amp = result.amp
        dx = setup.grid.dx
        z_coord = (np.arange(amp.shape[2]) + region_origin(result)[2] - setup.apex_vox[2]) * dx
        tgt_r = to_region(setup.focus_vox, result)
        on = np.abs(
            axial_pressure(
                z_coord,
                setup.aperture_radius,
                setup.roc,
                setup.knobs.f0,
                c0=1500.0,
            )
        )
        ax_on_axis = amp[tgt_r[0], tgt_r[1], :]
        valid = z_coord > 2e-3  # skip the source shell itself
        a_n = ax_on_axis[valid] / ax_on_axis[valid].max()
        o_n = on[valid] / on[valid].max()
        z_pk_sim = float(z_coord[valid][int(np.argmax(a_n))])
        z_pk_on = float(z_coord[valid][int(np.argmax(o_n))])
        # Two correlations on purpose: the full axis is dominated by the
        # near-field lobes (where a voxelized cap legitimately differs from
        # the closed form), the focal window is the number that matters.
        focal = o_n > 0.5
        metrics["vs_oneill"] = {
            "pearson_r_axial_shape": round(float(np.corrcoef(a_n, o_n)[0, 1]), 5),
            "pearson_r_focal_region": round(float(np.corrcoef(a_n[focal], o_n[focal])[0, 1]), 5),
            "focal_window_mm": [
                round(float(z_coord[valid][focal][0]) * 1e3, 2),
                round(float(z_coord[valid][focal][-1]) * 1e3, 2),
            ],
            "rel_l2_pct": round(100.0 * float(np.linalg.norm(a_n - o_n) / np.linalg.norm(o_n)), 2),
            "peak_z_sim_mm": round(z_pk_sim * 1e3, 3),
            "peak_z_oneill_mm": round(z_pk_on * 1e3, 3),
            "peak_z_diff_mm": round((z_pk_sim - z_pk_on) * 1e3, 3),
            "note": "normalized axial SHAPE comparison; O'Neil is lossless+linear",
        }

    return metrics


def profiles(setup: Setup, result: SolverResult) -> dict[str, np.ndarray]:
    """Coordinate/profile arrays used by the figures (mm, Pa)."""
    out = axial_profiles(
        result,
        dx=setup.grid.dx,
        grid_shape=setup.grid.shape,
        pml_vox=setup.grid.pml_vox,
        apex_vox=setup.apex_vox,
    )
    if setup.aperture_radius is not None and setup.medium.c_max == setup.medium.c_min:
        out["axial_oneill"] = np.abs(
            axial_pressure(
                out["z_mm"] / 1e3, setup.aperture_radius, setup.roc, setup.knobs.f0, c0=1500.0
            )
        )
        # keep the fundamental/h2 key order the figures rely on
        if "axial_h2" in out:
            out["axial_h2"] = out.pop("axial_h2")
    return out
