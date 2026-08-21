"""Metrics, preview packages and report rendering (M10d).

Single source of truth for focal metrics (:mod:`caustica.report.metrics` —
``apps/focus_study`` delegates here), the <=10 MB preview package the runner
writes next to every result (:mod:`caustica.report.preview`), and the
``caustica report`` renderer (:mod:`caustica.report.run_report`).

Import discipline (PEP 562, same rule as :mod:`caustica.io`): metric and
preview names are numpy-only and eager; anything that would pull in
matplotlib (figures, run_report) or h5py loads lazily, so the runner can
write previews on a machine with neither installed.
"""

from __future__ import annotations

from caustica.report.metrics import (
    HALF_PRESSURE,
    argmax_interior,
    axial_profiles,
    extent_6db,
    focus_metrics,
    intensity_w_cm2,
    interior_slices,
    mm_axes,
    region_origin,
    to_grid,
    to_region,
)
from caustica.report.preview import (
    DEFAULT_MAX_BYTES,
    PREVIEW_FORMAT,
    block_mean,
    load_preview,
    write_preview,
)

_LAZY = {
    "report_out_dir": "caustica.report.run_report",
    "FigureContext": "caustica.report.figures",
}

__all__ = [
    "HALF_PRESSURE",
    "argmax_interior",
    "axial_profiles",
    "extent_6db",
    "focus_metrics",
    "intensity_w_cm2",
    "interior_slices",
    "mm_axes",
    "region_origin",
    "to_grid",
    "to_region",
    "DEFAULT_MAX_BYTES",
    "PREVIEW_FORMAT",
    "block_mean",
    "load_preview",
    "write_preview",
    "report_out_dir",
    "FigureContext",
]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'caustica.report' has no attribute {name!r}")
