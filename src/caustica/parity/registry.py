"""The thirty parity metrics, addressed by ID.

The registry is the specification made executable: M-01 to M-30 of the
comparison contract, each with its group, its "applies when" clause, its
units and two callables. A page can only name a metric that exists here, and
a metric can only be emitted through :func:`evaluate`, which is what makes
the three-state rule structural rather than a habit.

The two callables split the work the way the contract does:

``applies(case)``
    Returns ``None`` when the metric means something for this example, or the
    sentence explaining why it does not. That sentence is a statement about
    the PHYSICS and it comes from the example's declared traits, so it is
    written once, next to the scene, and printed everywhere it is needed.

``compute(case)``
    Returns the number, or raises :class:`~caustica.parity.model.NotComputed`
    with the sentence explaining what this RUN did not record. It may also
    raise :class:`~caustica.parity.model.NotApplicable` for a fact only the
    data can settle (no sidelobe in the recorded window, say).

The split is the contract's third rule, and it decides which of the two
absences a reader sees: a trait answers about the PHYSICS of the example and
produces "not applicable", while a run that simply did not record something
produces "not computed" from ``compute``. An example that never wrote a
sensor history therefore declares the time-series trait as holding and lets
the metric say what was missing, because a waveform is meaningful there and
only this run lacks it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from caustica.parity.model import (
    NOT_APPLICABLE,
    NOT_COMPUTED,
    VALUE,
    MetricOutcome,
    NotApplicable,
    NotComputed,
    ParityCase,
)

#: The metric groups of the contract, in the order pages print them. The
#: contract's prose says seven and its table lists these eight labels; the
#: table is the specification, so the registry follows it and the "numerical"
#: and "cost" rows are the ones the prose's count leaves ambiguous.
GROUPS = (
    "field agreement",
    "focus",
    "profiles",
    "spectral",
    "time domain",
    "invariants",
    "numerical",
    "cost",
)

#: How many metrics the contract defines. A registry that does not hold
#: exactly these is a registry that disagrees with the specification.
METRIC_COUNT = 30


@dataclass(frozen=True)
class ParityMetric:
    """One metric: what it is, when it applies, and how to compute it."""

    id: str
    name: str
    group: str
    applies_when: str
    units: str
    applies: Callable[[ParityCase], str | None]
    compute: Callable[[ParityCase], Any]

    def describe(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "group": self.group,
            "applies_when": self.applies_when,
            "units": self.units,
        }


_METRICS: dict[str, ParityMetric] = {}


def metric(
    metric_id: str,
    name: str,
    group: str,
    applies_when: str,
    *,
    applies: Callable[[ParityCase], str | None],
    units: str = "",
) -> Callable[[Callable[[ParityCase], Any]], Callable[[ParityCase], Any]]:
    """Register a metric function under ``metric_id``.

    The decorated function is returned unchanged, so it stays directly
    callable. Replacing what :func:`evaluate` calls is a different thing:
    the registry holds the function object, so a test swaps the whole entry
    (``dataclasses.replace(get(id), compute=...)``) rather than rebinding
    the module attribute, which :func:`evaluate` would never look at.
    """

    def wrap(fn: Callable[[ParityCase], Any]) -> Callable[[ParityCase], Any]:
        if metric_id in _METRICS:
            raise ValueError(f"parity metric '{metric_id}' is already registered")
        if group not in GROUPS:
            raise ValueError(f"metric {metric_id}: group must be one of {GROUPS}, got {group!r}")
        _METRICS[metric_id] = ParityMetric(
            id=metric_id,
            name=name,
            group=group,
            applies_when=applies_when,
            units=units,
            applies=applies,
            compute=fn,
        )
        return fn

    return wrap


def get(metric_id: str) -> ParityMetric:
    """The metric with this ID, or a lookup error naming the whole registry."""
    try:
        return _METRICS[metric_id]
    except KeyError:
        raise KeyError(
            f"no parity metric '{metric_id}'; the registry holds {ids()}. A page cannot "
            f"name a metric the contract does not define."
        ) from None


def ids() -> tuple[str, ...]:
    """Every registered metric ID, in contract order (M-01 first)."""
    return tuple(sorted(_METRICS))


def all_metrics() -> tuple[ParityMetric, ...]:
    return tuple(_METRICS[i] for i in ids())


def evaluate(metric_id: str, case: ParityCase) -> MetricOutcome:
    """One metric of one engine, in exactly one of the three states."""
    m = get(metric_id)
    reason = m.applies(case)
    if reason is not None:
        return MetricOutcome(metric_id, NOT_APPLICABLE, reason=reason)
    try:
        value = m.compute(case)
    except NotApplicable as exc:
        return MetricOutcome(metric_id, NOT_APPLICABLE, reason=str(exc))
    except NotComputed as exc:
        return MetricOutcome(metric_id, NOT_COMPUTED, reason=str(exc))
    return MetricOutcome(metric_id, VALUE, value=value)


def evaluate_all(case: ParityCase) -> dict[str, MetricOutcome]:
    """Every registered metric for one engine of one example."""
    return {i: evaluate(i, case) for i in ids()}


def registry_json() -> list[dict[str, str]]:
    """The registry as data, so a page can print what it did not use."""
    return [m.describe() for m in all_metrics()]


def check_complete() -> None:
    """Raise unless the registry holds exactly M-01 to M-30."""
    expected = tuple(f"M-{n:02d}" for n in range(1, METRIC_COUNT + 1))
    if ids() != expected:
        missing = sorted(set(expected) - set(ids()))
        extra = sorted(set(ids()) - set(expected))
        raise RuntimeError(
            f"the parity registry must hold exactly M-01 to M-{METRIC_COUNT:02d}; "
            f"missing {missing}, unexpected {extra}."
        )
