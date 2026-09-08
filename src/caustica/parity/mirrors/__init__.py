"""The mirrored k-Wave examples, one module per example.

A mirror is not a script. It is the k-Wave example's scene, its settings with
the provenance of every number, and one function per engine; the runner owns
the comparison and the metric registry owns the numbers. That split is what
keeps sixty-nine pages from becoming sixty-nine private opinions about what
agreement means.

Four mirrors live here today, from three of k-Wave's ten categories, chosen
from the eight the enumeration measured as runnable on the current tree with
no new physics. They are deliberately unlike each other: one has no acoustic
scene at all, one runs the whole pressure to dose chain, one is a driven 2-D
transducer field and one is a 3-D piston graded against the compiled k-Wave
binary. Their applicable metric subsets differ for reasons each example
states, which is the property the harness has to demonstrate before the
battery scales to the rest.

The modules are imported lazily by :func:`get`: a thermal mirror pulls in the
Pennes solver and an acoustic one pulls in the k-Wave adapter, and neither
belongs in the cost of ``import caustica.parity``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from caustica.parity.runner import ParityMirror

#: k-Wave example name -> ``module:class`` inside this package.
MIRRORS: dict[str, str] = {
    "example_diff_homogeneous_medium_diffusion": (
        "diff_homogeneous_medium_diffusion:HomogeneousDiffusionMirror"
    ),
    "example_diff_focused_ultrasound_heating": (
        "diff_focused_ultrasound_heating:FocusedUltrasoundHeatingMirror"
    ),
    "example_tvsp_transducer_field_patterns": (
        "tvsp_transducer_field_patterns:TransducerFieldPatternsMirror"
    ),
    "example_cpp_running_simulations": "cpp_running_simulations:RunningSimulationsMirror",
}


def names() -> tuple[str, ...]:
    """Every mirrored example, in the order the pages are published."""
    return tuple(MIRRORS)


def get(name: str, **kwargs: object) -> ParityMirror:
    """The mirror for one k-Wave example, by the enumeration's own name."""
    try:
        target = MIRRORS[name]
    except KeyError:
        raise KeyError(
            f"no mirror for '{name}'; this package holds {list(MIRRORS)}. The work list "
            f"is planning/reports/kwave-example-parity.json and an example on it that "
            f"has no mirror yet is a task, not a lookup failure."
        ) from None
    module_name, class_name = target.split(":")
    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, class_name)(**kwargs)  # type: ignore[no-any-return]
