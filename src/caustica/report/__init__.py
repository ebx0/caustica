"""Metrics, preview packages and report rendering (M10d).

Single source of truth for focal metrics (:mod:`caustica.report.metrics` —
``apps/focus_study`` delegates here), the <=10 MB preview package the runner
writes next to every result (:mod:`caustica.report.preview`), and the
``caustica report`` renderers (:mod:`caustica.report.renderers`, with
caustica's own matplotlib implementation in :mod:`caustica.report.run_report`).

Import discipline (PEP 562, same rule as :mod:`caustica.io`): metric, preview
and renderer-registry names are numpy-only/stdlib-only and eager; anything
that would pull in matplotlib (figures, run_report) or h5py loads lazily, so
the runner can write previews on a machine with neither installed. The
renderer registry is eager on purpose — listing what can render a folder
must not import a plotting library.
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
    build_preview,
    decode_preview,
    load_preview,
    write_preview,
)
from caustica.report.renderers import DEFAULT_RENDERER, render_report, report_renderers

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
    "build_preview",
    "decode_preview",
    "load_preview",
    "write_preview",
    "DEFAULT_RENDERER",
    "render_report",
    "report_renderers",
    "report_out_dir",
    "FigureContext",
]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'caustica.report' has no attribute {name!r}")
