"""Figure set for one focus study — thin adapter over :mod:`caustica.report.figures`.

Since M10d the drawing code (and the dataviz style) lives in the library so
``caustica report`` produces the same figures; this module only translates a
scenario ``Setup`` into the library's :class:`FigureContext` and hands the
whole set off to :func:`caustica.report.figures.make_all`.
"""

from __future__ import annotations

from pathlib import Path

from apps.focus_study.analysis import field_frame
from apps.focus_study.scenarios import Setup
from caustica.report import figures as _fig
from caustica.report.figures import FigureContext
from caustica.solvers import SolverResult

__all__ = ["FigureContext", "make_all"]


def _ctx(setup: Setup) -> FigureContext:
    return FigureContext(
        frame=field_frame(setup),
        title=setup.title,
        solver=setup.knobs.solver,
        source_indices=setup.source.indices,
        labels=setup.labels,
        label_names=setup.label_names,
        sound_speed=setup.medium.c,
    )


def make_all(setup: Setup, result: SolverResult, prof: dict, outdir: Path) -> list[str]:
    return _fig.make_all(_ctx(setup), result, prof, outdir)
