"""The data contract of a k-Wave parity comparison.

Three ideas live here and nothing else.

**A metric outcome has three states, never two.** A number, or "not
applicable" with the reason, or "not computed" with the reason. A blank, a
dash or a zero standing for any of the three is a defect, so
:class:`MetricOutcome` refuses to be built that way: a value state must carry
a finite number (or a mapping of them) and no reason, and the two absence
states must carry a reason and no value at all. Absence is therefore not
expressible as a value, which is the property the page generator rests on.

**An example declares what is true of it, once.** :class:`ParityExample`
carries the enumeration's own record plus a closed vocabulary of
:data:`TRAITS`. A metric's applicability predicate reads a trait and gets
back both the answer and the sentence to print when the answer is no, so the
reason on a page is written next to the physics, not next to the prose.

A trait is a statement about the EXAMPLE, and a trait that does not hold
makes every metric it gates "not applicable". What one particular run failed
to record is a different thing and is not a trait: the metric raises
:class:`NotComputed` and the page says "not computed". A mirror that records
no sensor history therefore declares ``time_series`` as holding, because a
waveform is meaningful for that example and only this run lacks one;
declaring the trait absent would print a statement about the physics where
the truth is a statement about the run.

**A setting is either the k-Wave example's or an inference.** k-wave.org
publishes the interesting fragment of each example, not the whole script:
the arc geometry of ``example_tvsp_transducer_field_patterns`` is on the
page, its grid size is not. :class:`Setting` makes the mirror say which is
which for every number it used, so a reader can tell a matched setting from
a reconstructed one.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: The three states of a metric outcome (comparison contract, rule 3).
VALUE = "value"
NOT_APPLICABLE = "not applicable"
NOT_COMPUTED = "not computed"
STATES = (VALUE, NOT_APPLICABLE, NOT_COMPUTED)

#: Strings a page must never print in place of a state. They are refused as
#: metric values so that "no answer" can only ever be spelled as a state.
PLACEHOLDERS = ("", "-", "--", "n/a", "N/A", "NA", "na", "none", "None", "TBD", "?")

#: What an engine is: something that integrates, or something evaluated. The
#: cost metrics read this, because a ratio of wall times against a closed
#: form would compare a solve with an evaluation.
KINDS = ("solver", "closed form")

#: Whether an engine is graded, or does the grading.
ROLES = ("native", "reference")

#: Every trait an applicability predicate may ask about, and what asking it
#: means. A spec that declares a name outside this mapping is refused, and one
#: that leaves a name out is refused too: a trait nobody declared would make
#: every metric it gates quietly applicable.
TRAITS: dict[str, str] = {
    "field": "a 2-D or 3-D field exists",
    "peak": "the graded field has a peak",
    "focused_beam": "the graded field is a focused beam",
    "source_surface": "the source has a defined surface pressure",
    "beam_axis": "a beam axis exists",
    "sidelobe": "a sidelobe is resolved",
    "nonlinear": "the example is nonlinear",
    "third_harmonic": "3f0 is resolved",
    "time_series": "a time series is recorded",
    "shock": "the waveform steepens",
    "closed_surface": "a closed surface fits in the domain",
    "integrable_loss": "the loss is zero or integrable",
    "absorbing": "the medium absorbs",
    "varies_resolution": "the example varies resolution",
}


class NotApplicable(Exception):
    """The physics of this example gives the metric no meaning."""


class NotComputed(Exception):
    """This run did not produce what the metric needs."""


def _finite(value: Any) -> bool:
    """True when ``value`` is a finite number, or a container of them."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float | np.floating | np.integer):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return bool(value) and all(_finite(v) for v in value.values())
    if isinstance(value, Sequence) and not isinstance(value, str):
        return bool(value) and all(_finite(v) for v in value)
    return False


def _jsonable(value: Any) -> Any:
    """Numpy scalars and containers into things ``json.dump`` accepts."""
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class MetricOutcome:
    """One metric of one engine: a number, or a stated absence.

    ``value`` is a finite number or a mapping/sequence of finite numbers (a
    metric may legitimately answer with a pair, as M-08 does with voxels and
    mm). ``reason`` is a sentence, and it is required exactly when the state
    is one of the two absences.
    """

    metric_id: str
    state: str
    value: Any = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ValueError(f"state must be one of {STATES}, got {self.state!r}")
        if self.state == VALUE:
            self._check_value()
        else:
            self._check_absence()

    def _check_value(self) -> None:
        if self.reason:
            raise ValueError(
                f"{self.metric_id}: a computed value carries no reason; a reason is how "
                f"an absence is stated. Got {self.reason!r}."
            )
        if self.value is None or isinstance(self.value, str):
            raise ValueError(
                f"{self.metric_id}: a metric value must be a number, not {self.value!r}. "
                f"An absence is a state, never a placeholder value."
            )
        if isinstance(self.value, Mapping | Sequence) and not self.value:
            raise ValueError(
                f"{self.metric_id}: an empty {type(self.value).__name__} is an absence "
                f"wearing the shape of a value. A metric with nothing to measure raises "
                f"NotApplicable or NotComputed with the reason."
            )
        if not _finite(self.value):
            raise ValueError(
                f"{self.metric_id}: metric values must be finite numbers, got "
                f"{self.value!r}. NaN and inf are absences and are reported as "
                f"'{NOT_COMPUTED}' with the reason."
            )

    def _check_absence(self) -> None:
        if self.value is not None:
            raise ValueError(
                f"{self.metric_id}: state {self.state!r} carries no value, got {self.value!r}."
            )
        if len(self.reason.strip()) < 10 or self.reason.strip() in PLACEHOLDERS:
            raise ValueError(
                f"{self.metric_id}: state {self.state!r} needs a reason a reader can act "
                f"on, got {self.reason!r}."
            )

    def to_json(self) -> dict[str, Any]:
        """The dict a ``metrics.json`` carries for this metric."""
        out: dict[str, Any] = {"state": self.state}
        if self.state == VALUE:
            out["value"] = _jsonable(self.value)
        else:
            out["reason"] = self.reason
        return out


@dataclass(frozen=True)
class Trait:
    """Whether a trait holds, and the sentence to print when it does not."""

    holds: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.holds and len(self.reason.strip()) < 10:
            raise ValueError(
                f"a trait that does not hold needs the reason a reader will see on the "
                f"page in place of every metric it gates; got {self.reason!r}"
            )


def holds() -> Trait:
    """A trait that holds; it gates nothing, so it carries no reason."""
    return Trait(True, "")


def absent(reason: str) -> Trait:
    """A trait that does not hold, with the sentence a page prints instead."""
    return Trait(False, reason)


@dataclass(frozen=True)
class Setting:
    """One mirrored setting and where its number came from.

    ``source`` is ``"k-Wave example page"`` when the value is quoted from
    k-wave.org, ``"inferred"`` when the published page does not carry it, and
    ``"caustica"`` when the mirror had to choose something k-Wave's script
    has no equivalent of. The last two must say what they were reconstructed
    from, because that sentence is the difference between a matched setting
    and a guess.
    """

    value: Any
    source: str
    note: str = ""

    #: The three provenances a mirrored setting can have.
    SOURCES = ("k-Wave example page", "inferred", "caustica")

    def __post_init__(self) -> None:
        if self.source not in Setting.SOURCES:
            raise ValueError(f"source must be one of {Setting.SOURCES}, got {self.source!r}")
        if self.source != "k-Wave example page" and len(self.note.strip()) < 10:
            raise ValueError(
                f"setting {self.value!r} is not quoted from the k-Wave page, so it must "
                f"say what it was reconstructed from; got note={self.note!r}"
            )

    def to_json(self) -> dict[str, Any]:
        return {"value": _jsonable(self.value), "source": self.source, "note": self.note}


@dataclass(frozen=True)
class ParityExample:
    """One k-Wave example as the enumeration recorded it, plus its traits.

    The identifying fields (``name``, ``title``, ``category``,
    ``demonstrates``, ``source_url`` and ``faithful_mirror``) are copied
    verbatim from ``kwave-example-parity.json``; that file is the work list
    and it wins against any count or wording repeated elsewhere.
    ``comparison_axis`` and ``mirror_note`` are the mirror's own restatement
    of the enumeration's verdict, written for a reader of the page rather
    than for the work list, so they say the same thing in other words.
    """

    name: str
    title: str
    category: str
    demonstrates: str
    comparison_axis: str
    source_url: str
    faithful_mirror: bool
    mirror_note: str
    traits: Mapping[str, Trait]
    graded_quantity: str
    graded_units: str

    def __post_init__(self) -> None:
        unknown = sorted(set(self.traits) - set(TRAITS))
        if unknown:
            raise ValueError(
                f"{self.name}: unknown trait(s) {unknown}; the vocabulary is {sorted(TRAITS)}."
            )
        missing = sorted(set(TRAITS) - set(self.traits))
        if missing:
            raise ValueError(
                f"{self.name}: trait(s) {missing} undeclared. Every trait is declared "
                f"explicitly, so an applicable metric is a decision and never a default."
            )

    def trait(self, name: str) -> Trait:
        return self.traits[name]

    def record_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "category": self.category,
            "demonstrates": self.demonstrates,
            "comparison_axis": self.comparison_axis,
            "source_url": self.source_url,
            "faithful_mirror": self.faithful_mirror,
            "mirror_note": self.mirror_note,
            "graded_quantity": self.graded_quantity,
            "graded_units": self.graded_units,
            "traits": {
                k: ({"holds": True} if v.holds else {"holds": False, "reason": v.reason})
                for k, v in sorted(self.traits.items())
            },
        }


@dataclass(frozen=True)
class Series:
    """One recorded time series: ``t`` [s] and the quantity, plus its label."""

    t_s: np.ndarray
    value: np.ndarray
    label: str
    units: str


@dataclass
class EngineRun:
    """What one engine produced for one example, in the form metrics read.

    An engine the example cannot express carries ``refusal`` (the capability
    matrix's own text) and nothing else; the runner emits no metrics for it,
    because a refused solver has no numbers to be honest about beyond the
    refusal itself.
    """

    engine: str
    kind: str  # see KINDS
    role: str  # see ROLES
    refusal: str | None = None
    field: np.ndarray | None = None
    dx: float = 0.0  # voxel pitch [m]; zero means this run recorded none
    beam_axis: int | None = None
    harmonics: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    series: dict[str, Series] = dataclasses.field(default_factory=dict)
    scalars: dict[str, float] = dataclasses.field(default_factory=dict)
    source_surface_pa: float | None = None
    source_power_w: float | None = None
    radiated_power_w: float | None = None
    absorbed_power_w: float | None = None
    resolution_sweep: tuple[tuple[float, float], ...] | None = None
    wall_time_s: float | None = None
    peak_device_bytes: int | None = None
    notes: tuple[str, ...] = ()
    provenance: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.engine}: kind must be one of {KINDS}, got {self.kind!r}")
        if self.role not in ROLES:
            raise ValueError(f"{self.engine}: role must be one of {ROLES}, got {self.role!r}")

    @property
    def refused(self) -> bool:
        return self.refusal is not None

    def summary_json(self) -> dict[str, Any]:
        """The engine block of ``metrics.json`` (the arrays stay out)."""
        if self.refused:
            return {"engine": self.engine, "role": self.role, "refused": self.refusal}
        return {
            "engine": self.engine,
            "role": self.role,
            "kind": self.kind,
            "field_shape": None if self.field is None else list(self.field.shape),
            "dx_m": self.dx,
            "beam_axis": self.beam_axis,
            "harmonics_recorded": sorted(self.harmonics),
            "series_recorded": sorted(self.series),
            "scalars": {k: _jsonable(v) for k, v in sorted(self.scalars.items())},
            "wall_time_s": self.wall_time_s,
            "peak_device_bytes": self.peak_device_bytes,
            "notes": list(self.notes),
            "provenance": _jsonable(self.provenance),
        }


@dataclass(frozen=True)
class ParityCase:
    """What a metric is handed: the example, one engine's run, the reference.

    The brief spells the predicate ``applies(example, result)``; this is the
    same information in one frozen object, following the pattern
    ``report.metrics.FieldFrame`` set (one object beats five parallel
    arguments repeated across sixty signatures). Adding an input later is one
    field here instead of sixty edits.
    """

    example: ParityExample
    run: EngineRun
    reference: EngineRun

    def trait(self, name: str) -> Trait:
        return self.example.trait(name)
