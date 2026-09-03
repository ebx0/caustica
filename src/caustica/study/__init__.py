"""``caustica.study`` — one config, many runs, one stamped report.

A :class:`~caustica.study.core.Study` is the layer above
:func:`caustica.simulate`: it holds a base job, produces variants of it by
address (``"drive.amplitude_kpa"``), runs them, and writes ONE report that
puts the runs side by side. It adds no physics, no second way to build a
job and no second definition of a metric — every run goes through the
facade, and every number in the report comes from
:mod:`caustica.report.metrics`.

::

    from caustica import Study

    study = Study("p0-scan", "job.json", out="studies/p0")
    sweep = study.sweep("drive.amplitude_kpa", [50, 100, 200])
    sweep.report()          # STUDY.md + study.json + fig_sweep.png

Import discipline (same rule as :mod:`caustica.validation`): the names below
are PEP 562 lazy, so ``import caustica`` and ``import caustica.study`` cost
nothing — reaching :class:`Study` is what pulls in the runner (and h5py),
and only rendering a sweep figure pulls in matplotlib.
"""

from __future__ import annotations

__all__ = [
    "FORMAT",
    "Study",
    "StudyError",
    "StudyRun",
    "StudySweep",
    "get_by_path",
    "set_by_path",
]

_LAZY = {
    "FORMAT": "caustica.study.core",
    "Study": "caustica.study.core",
    "StudyError": "caustica.study.core",
    "StudyRun": "caustica.study.core",
    "StudySweep": "caustica.study.core",
    "get_by_path": "caustica.study.core",
    "set_by_path": "caustica.study.core",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'caustica.study' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
