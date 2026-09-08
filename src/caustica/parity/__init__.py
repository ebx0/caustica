"""The k-Wave parity harness: one contract, one metric registry, one runner.

Decision D-030 makes the k-Wave example gallery a MEASURED comparison rather
than a picture gallery, and this package is the code every one of those
measurements goes through. Four modules carry the harness itself:

``model``
    The data contract. A metric outcome has three states (a value, "not
    applicable" with the reason, "not computed" with the reason) and the
    class refuses to be built any other way, so a blank cannot be spelled.
``registry``
    M-01 to M-30 addressed by ID, each with its group, its applicability
    clause and its two callables. :func:`~caustica.parity.registry.evaluate`
    is the only door a metric leaves by, which is what makes the three-state
    rule structural instead of a habit.
``metrics``
    The thirty metric functions. Importing this module is what fills the
    registry, so this package imports it and then checks the registry holds
    exactly the thirty the contract defines.
``runner``
    One example record in, one ``metrics.json`` out: every native solver
    :class:`~caustica.solvers.base.SolverCaps` says can express the example,
    run at the comparison CFL of 0.3, plus the reference, plus the refusal
    text for every solver that cannot express it.

Two more pieces sit beside them. :mod:`caustica.parity.scenes` holds the
scene constructors a mirrored example needs and the library has not got yet
(today, a 2-D arc source). The example mirrors live in
:mod:`caustica.parity.mirrors`, one module per k-Wave example, and are not
imported here: a mirror pulls in the thermal stack or the k-Wave adapter, and
importing the harness must stay cheap.
"""

from __future__ import annotations

from caustica.parity import metrics as _metrics  # noqa: F401  (fills the registry)
from caustica.parity.model import (
    NOT_APPLICABLE,
    NOT_COMPUTED,
    STATES,
    VALUE,
    EngineRun,
    MetricOutcome,
    NotApplicable,
    NotComputed,
    ParityCase,
    ParityExample,
    Series,
    Setting,
    Trait,
    absent,
    holds,
)
from caustica.parity.registry import (
    GROUPS,
    METRIC_COUNT,
    ParityMetric,
    all_metrics,
    check_complete,
    evaluate,
    evaluate_all,
    ids,
    registry_json,
)
from caustica.parity.runner import (
    CAUSTICA_DEFAULT_CFL,
    METRICS_FILENAME,
    PARITY_CFL,
    AcousticScene,
    ExtraEngine,
    ParityMirror,
    parity_run_spec,
    run_example,
    screen_solvers,
)

# The registry is the specification made executable, so a package that
# imported a partial one would publish pages with holes in them.
check_complete()

__all__ = [
    "CAUSTICA_DEFAULT_CFL",
    "GROUPS",
    "METRICS_FILENAME",
    "METRIC_COUNT",
    "NOT_APPLICABLE",
    "NOT_COMPUTED",
    "PARITY_CFL",
    "STATES",
    "VALUE",
    "AcousticScene",
    "EngineRun",
    "ExtraEngine",
    "MetricOutcome",
    "NotApplicable",
    "NotComputed",
    "ParityCase",
    "ParityExample",
    "ParityMetric",
    "ParityMirror",
    "Series",
    "Setting",
    "Trait",
    "absent",
    "all_metrics",
    "check_complete",
    "evaluate",
    "evaluate_all",
    "holds",
    "ids",
    "parity_run_spec",
    "registry_json",
    "run_example",
    "screen_solvers",
]
