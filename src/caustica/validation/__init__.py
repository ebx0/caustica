"""Machine-checked milestone gates (M11's first piece).

A criterion in ``MILESTONES.md`` is a sentence until something measures it.
This package turns the ones that need real hardware into commands that
produce a stamped report — evidence a milestone box can be ticked against,
instead of a memory of a session that went well.

The first module is :mod:`caustica.validation.gpu_gates`, which closes M7's
and M8's on-device criteria in a single run::

    python -m caustica.validation gpu-gates

Nothing here needs external data: every scenario is a homogeneous water
volume with a bowl source, sized on the spot from the device's own free VRAM.
Imports stay light — ``caustica.validation`` pulls in the runner (and h5py)
only when a suite actually runs, through PEP 562 lazy attributes.
"""

from __future__ import annotations

__all__ = [
    "Check",
    "FORMAT",
    "Gate",
    "RungSpec",
    "build_ladder",
    "gpu_gates",
]

_LAZY = {
    "Check": "caustica.validation.gpu_gates",
    "FORMAT": "caustica.validation.gpu_gates",
    "Gate": "caustica.validation.gpu_gates",
    "RungSpec": "caustica.validation.gpu_gates",
    "build_ladder": "caustica.validation.gpu_gates",
    "gpu_gates": "caustica.validation.gpu_gates",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'caustica.validation' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
