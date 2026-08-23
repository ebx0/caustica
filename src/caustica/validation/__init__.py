"""Machine-checked milestone gates (M11's first piece).

A criterion in ``MILESTONES.md`` is a sentence until something measures it.
This package turns the ones that need real hardware into commands that
produce a stamped report — evidence a milestone box can be ticked against,
instead of a memory of a session that went well.

Three suites live here.

:mod:`caustica.validation.gpu_gates` closes M7's and M8's on-device criteria
in a single run — it needs a real GPU and books it for minutes::

    python -m caustica.validation gpu-gates

:mod:`caustica.validation.analytic_suite` grades the solvers against the
closed forms in :mod:`caustica.analytic` — plane wave, O'Neil bowl, linear
limit, Fubini — and runs anywhere, GPU or not, in seconds::

    python -m caustica.validation run-analytic

:mod:`caustica.validation.compare` runs ONE job on N registered solvers and
tables what they disagree about — normalized relative L2, Pearson r, focal
metrics — with a T0 sanity gate ahead of every engine and a verbatim
"environment-broken" stamp for any engine this machine cannot run::

    python -m caustica.validation compare

All three share the PASS/FAIL/SKIP algebra in
:mod:`caustica.validation._verdict`, so they cannot drift apart about what a
missing measurement means: it is a SKIP, and a SKIP is never a pass.

Nothing here needs external data: every scenario is homogeneous water with a
library-built source. Imports stay light — ``caustica.validation`` pulls in
the runner (and h5py) only when a suite actually runs, through PEP 562 lazy
attributes.
"""

from __future__ import annotations

__all__ = [
    "Check",
    "FORMAT",
    "Gate",
    "RungSpec",
    "analytic_suite",
    "build_ladder",
    "compare",
    "gpu_gates",
]

_LAZY = {
    "Check": "caustica.validation._verdict",
    "FORMAT": "caustica.validation.gpu_gates",
    "Gate": "caustica.validation._verdict",
    "RungSpec": "caustica.validation.gpu_gates",
    "analytic_suite": "caustica.validation.analytic_suite",
    "build_ladder": "caustica.validation.gpu_gates",
    "compare": "caustica.validation.compare",
    "gpu_gates": "caustica.validation.gpu_gates",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'caustica.validation' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
