"""One runner for every k-Wave parity comparison.

The brief's shape, kept literally: a mirror hands the runner one example
record and one scene; the runner asks
:class:`~caustica.solvers.base.SolverCaps` which registered solvers can
express that scene, runs every one that can at the comparison CFL, runs the
reference, and writes ``metrics.json``.

Three things about it are load-bearing, and all three come from the
comparison contract rather than from convenience.

**The CFL is pinned and stated, not defaulted.** :data:`PARITY_CFL` is 0.3,
k-Wave's own ``makeTime`` default; caustica's default is 0.48. A parity page
therefore does not run either engine on a time step the other did not take.
The two engines still do not land on exactly the same one: the native
discretization takes ``floor(period / dt_cfl)`` points per period and the
k-Wave adapter takes ``ceil`` of the same quantity, so their realized CFL can
differ by one point per period. That is not left in a footnote; every engine
block carries the ``spp``, ``dt`` and realized CFL it actually ran, so a
reader can check the claim instead of believing it.

**Every registered solver appears, including the ones that cannot run.** A
solver refused by caps is recorded with the refusal the capability check
itself produced, so the table is evidence about the capability matrix rather
than a list of whatever happened to work. The consequence the brief wants is
that a page regenerated after a new solver lands gains a row with nobody
editing it.

**Nothing is emitted except through the registry.** Every number on a page
comes back from :func:`caustica.parity.registry.evaluate`, which can only
return one of the three states, so "no answer" cannot reach a page wearing
the shape of an answer.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from caustica.core.grid import Grid
from caustica.env import build_info, git_commit
from caustica.medium import Medium
from caustica.parity.model import (
    NOT_APPLICABLE,
    NOT_COMPUTED,
    VALUE,
    EngineRun,
    MetricOutcome,
    ParityCase,
    ParityExample,
    Setting,
)
from caustica.parity.registry import check_complete, evaluate_all, registry_json
from caustica.solvers.base import CWRunSpec, SolverCapabilityError
from caustica.solvers.kspace.engine import NUMERICS_SCHEME
from caustica.solvers.registry import available as available_solvers
from caustica.solvers.registry import get as get_solver
from caustica.sources import CWSource

#: The comparison CFL, pinned by the contract to k-Wave's ``makeTime``
#: default so neither engine is graded on a step the other did not take.
PARITY_CFL = 0.3

#: caustica's own default, for the sentence every page has to print.
CAUSTICA_DEFAULT_CFL = float(CWRunSpec.model_fields["cfl"].default)

#: What the runner writes, and what the page generator reads.
METRICS_FILENAME = "metrics.json"

#: Version of the ``metrics.json`` layout. A page generator pins it.
SCHEMA = "caustica-parity/1"


def parity_run_spec(**overrides: Any) -> CWRunSpec:
    """A :class:`~caustica.solvers.base.CWRunSpec` at the comparison CFL.

    Passing ``cfl`` is refused rather than obeyed: a run at another CFL is
    not a parity run, and a mirror that wants one has left the contract.
    """
    if "cfl" in overrides:
        raise ValueError(
            f"a parity run's CFL is pinned to {PARITY_CFL} by the comparison contract "
            f"(k-Wave's makeTime default); got cfl={overrides['cfl']!r}. Call the "
            f"solver directly if you want another step."
        )
    return CWRunSpec(cfl=PARITY_CFL, **overrides)


@dataclass(frozen=True)
class AcousticScene:
    """The grid, medium and source a registered solver is screened against."""

    grid: Grid
    medium: Medium
    source: CWSource
    spec: CWRunSpec
    record_region: tuple[slice, ...] | None = None

    def __post_init__(self) -> None:
        if abs(self.spec.cfl - PARITY_CFL) > 1e-12:
            raise ValueError(
                f"a parity scene runs at cfl={PARITY_CFL} (the contract's matched "
                f"setting), got {self.spec.cfl}. Build the spec with parity_run_spec()."
            )


@dataclass(frozen=True)
class ExtraEngine:
    """An engine outside the solver registry: a closed form, or a chain.

    The thermal examples need this. ``kind`` is ``"solver"`` or
    ``"closed form"``; ``refusal`` names an engine that belongs in the table
    and cannot run here, which is how k-Wave's own thermal engine appears on
    a thermal page in an environment whose k-wave-python ships no binding
    for it.
    """

    name: str
    kind: str
    role: str
    refusal: str | None = None


class ParityMirror(ABC):
    """One mirrored k-Wave example.

    A mirror owns the scene and the runs; it owns no metric and no output
    format. That split is why a metric definition can change without
    touching sixty-nine mirrors, and why a mirror cannot quietly grade
    itself on a number the registry does not define.
    """

    #: The k-Wave example name, exactly as ``kwave-example-parity.json``
    #: spells it.
    name: str = ""

    @abstractmethod
    def example(self) -> ParityExample:
        """The enumeration's record plus this example's declared traits."""

    @abstractmethod
    def settings(self) -> Mapping[str, Setting]:
        """Every mirrored setting with the provenance of its number."""

    @abstractmethod
    def reference_engine(self) -> str:
        """Which engine the others are graded against."""

    @abstractmethod
    def run_engine(self, name: str) -> EngineRun:
        """Run one engine and return what the metrics read."""

    def scene(self) -> AcousticScene | None:
        """The acoustic scene, or ``None`` for an example that has none."""
        return None

    def no_scene_reason(self) -> str:
        """Why a wave solver cannot express this example (no scene case)."""
        raise NotImplementedError(
            f"{type(self).__name__}.scene() returned None, so every registered wave "
            f"solver is refused and each refusal needs a reason a reader can act on. "
            f"Override no_scene_reason()."
        )

    def extra_engines(self) -> tuple[ExtraEngine, ...]:
        """Engines outside the solver registry (closed forms, chains)."""
        return ()

    def notes(self) -> tuple[str, ...]:
        """Sentences the page must print beside the numbers."""
        return ()


def screen_solvers(mirror: ParityMirror) -> dict[str, str | None]:
    """Every registered solver, mapped to its refusal or to ``None``.

    ``None`` means the capability check accepted this scene. The refusal is
    the capability check's own text, never a summary of it, because that text
    is the thing the project claims a user gets instead of a quietly wrong
    field.

    Only :class:`~caustica.solvers.base.SolverCapabilityError` is caught.
    Any other error from ``validate()`` is a mirror that built an
    inconsistent scene, and publishing that as a capability refusal would
    make the table say something false about the solver.
    """
    scene = mirror.scene()
    out: dict[str, str | None] = {}
    for name in sorted(available_solvers()):
        if scene is None:
            out[name] = mirror.no_scene_reason()
            continue
        try:
            get_solver(name)().validate(scene.grid, scene.medium, scene.source)
        except SolverCapabilityError as exc:
            out[name] = str(exc)
        else:
            out[name] = None
    return out


def _provenance(load_regime: str | None, mirror_backend: str | None) -> dict[str, Any]:
    info = build_info()
    return {
        "numerics_scheme": NUMERICS_SCHEME,
        "caustica_version": info.get("version"),
        # The checkout's own HEAD when there is one: a page must say which
        # library commit produced its numbers, and the build stamp goes stale
        # the moment the next commit lands.
        "caustica_commit": git_commit(),
        # The backend the mirror was built with, and None when it declares
        # none. Each engine block also carries the backend its own solver
        # reported, because that is the one the numbers came out of.
        "mirror_backend": mirror_backend,
        "load_regime": load_regime,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def _step_taken(run: EngineRun) -> dict[str, Any]:
    """The step this engine actually took, for the matched-settings rule."""
    keys = ("spp", "dt_s", "cfl_realized", "steps_total")
    return {k: run.provenance[k] for k in keys if k in run.provenance}


def _engine_block(
    run: EngineRun,
    outcomes: Mapping[str, MetricOutcome] | None,
    no_metrics_reason: str | None,
) -> dict[str, Any]:
    block = run.summary_json()
    block["step"] = _step_taken(run)
    if outcomes is None:
        if not no_metrics_reason:
            raise ValueError(f"engine '{run.engine}' carries neither metrics nor a reason")
        block["no_metrics_reason"] = no_metrics_reason
        return block
    block["metrics"] = {mid: o.to_json() for mid, o in outcomes.items()}
    block["applicable"] = [mid for mid, o in outcomes.items() if o.state == VALUE]
    block["not_applicable"] = [
        {"id": mid, "reason": o.reason} for mid, o in outcomes.items() if o.state == NOT_APPLICABLE
    ]
    block["not_computed"] = [
        {"id": mid, "reason": o.reason} for mid, o in outcomes.items() if o.state == NOT_COMPUTED
    ]
    block["state_counts"] = {
        VALUE: len(block["applicable"]),
        NOT_APPLICABLE: len(block["not_applicable"]),
        NOT_COMPUTED: len(block["not_computed"]),
    }
    return block


def _timed(mirror: ParityMirror, name: str, provenance: Mapping[str, Any]) -> EngineRun:
    """Run one engine, time it if it did not time itself, stamp provenance."""
    t0 = time.perf_counter()
    run = mirror.run_engine(name)
    elapsed = time.perf_counter() - t0
    if run.engine != name:
        raise ValueError(
            f"{type(mirror).__name__}.run_engine({name!r}) returned a run labelled "
            f"'{run.engine}'; the label is what the page prints."
        )
    if run.wall_time_s is None:
        run.wall_time_s = float(elapsed)
    merged = dict(provenance)
    merged.update(run.provenance)
    run.provenance = merged
    return run


def run_example(
    mirror: ParityMirror,
    *,
    out_dir: str | Path | None = None,
    load_regime: str | None = None,
) -> dict[str, Any]:
    """Run one mirrored example through every engine and write ``metrics.json``.

    Parameters
    ----------
    mirror:
        The mirrored example.
    out_dir:
        Where ``metrics.json`` goes. ``None`` returns the document without
        writing it, which is what a test wants.
    load_regime:
        What else was running on this machine, in the operator's own words.
        M-29 and M-30 refuse to report a timing without it, because a timing
        taken while other work shared the machine is not a measurement, so
        leaving it ``None`` is the honest state of an unattended run and not
        a missing argument.

    There is no ``backend`` argument. A mirror is built with the backend it
    runs on (``mirrors.get(name, backend="cupy")``), and the document records
    what the mirror declares plus what each solver reported, so a page cannot
    name a backend nothing ran on.
    """
    check_complete()
    example = mirror.example()
    if example.name != mirror.name:
        raise ValueError(
            f"mirror {type(mirror).__name__} is named '{mirror.name}' but its example "
            f"record says '{example.name}'; the enumeration's name is the identity."
        )

    screened = screen_solvers(mirror)
    reference_name = mirror.reference_engine()
    provenance = _provenance(load_regime, getattr(mirror, "backend", None))

    runs: dict[str, EngineRun] = {}
    for name, refusal in screened.items():
        if refusal is not None:
            runs[name] = EngineRun(
                engine=name,
                kind="solver",
                role="reference" if name == reference_name else "native",
                refusal=refusal,
            )
        else:
            runs[name] = _timed(mirror, name, provenance)
    for extra in mirror.extra_engines():
        if extra.name in runs:
            raise ValueError(
                f"extra engine '{extra.name}' collides with the registered solver of the "
                f"same name; an engine appears once in the table."
            )
        if extra.refusal is not None:
            runs[extra.name] = EngineRun(
                engine=extra.name, kind=extra.kind, role=extra.role, refusal=extra.refusal
            )
        else:
            runs[extra.name] = _timed(mirror, extra.name, provenance)

    if reference_name not in runs:
        raise ValueError(
            f"the reference engine '{reference_name}' is not among the engines this "
            f"example produced ({sorted(runs)})."
        )
    reference = runs[reference_name]
    if reference.refused:
        raise ValueError(
            f"the reference engine '{reference_name}' was refused: {reference.refusal} "
            f"Nothing on this page can be graded, so no metrics.json is written."
        )

    engines: list[dict[str, Any]] = []
    for name, run in runs.items():
        if run.refused:
            engines.append(
                _engine_block(
                    run,
                    None,
                    f"'{name}' cannot express this example, so it produced no field to "
                    f"grade. The refusal above is this row's evidence.",
                )
            )
        elif name == reference_name:
            engines.append(
                _engine_block(
                    run,
                    None,
                    f"'{name}' is the reference every other engine is graded against; "
                    f"grading it against itself would put a row of exact agreement on "
                    f"the page and call it a measurement.",
                )
            )
        else:
            outcomes = evaluate_all(ParityCase(example, run, reference))
            engines.append(_engine_block(run, outcomes, None))

    document: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": provenance,
        "contract": {
            "cfl": PARITY_CFL,
            "caustica_default_cfl": CAUSTICA_DEFAULT_CFL,
            "cfl_note": (
                f"Every run on this page is pinned to cfl = {PARITY_CFL}, which is "
                f"k-Wave's makeTime default. caustica's own default is "
                f"{CAUSTICA_DEFAULT_CFL}, so this is a comparison setting and not a "
                f"normal run. The native discretization floors the points per period "
                f"and the k-Wave adapter ceils them, so the realized step can differ by "
                f"one point per period; each engine block carries the step it took."
            ),
        },
        "example": example.record_json(),
        "graded_quantity": example.graded_quantity,
        "graded_units": example.graded_units,
        "settings": {k: v.to_json() for k, v in sorted(mirror.settings().items())},
        "reference_engine": reference_name,
        "engines": engines,
        "metric_registry": registry_json(),
        "notes": list(mirror.notes()),
    }

    if out_dir is not None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / METRICS_FILENAME).write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
    return document
