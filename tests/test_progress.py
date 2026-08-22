"""M10j gates: the progress hook and its presentation.

The hook is the one instrumentation site the notebook, the CLI, ``status.json``
and a future GUI all read (PLAN.md section 8). These tests pin the three
properties that make it trustworthy: it fires WITHOUT a checkpoint (trap T1),
it never reaches the k-Wave adapter (trap T3), and a consumer that throws
cannot cost hours of compute.
"""

import warnings

import numpy as np
import pytest

import caustica as hs
from caustica.core.backend import CausticaWarning
from caustica.materials import water
from caustica.solvers import CWRunSpec, get
from caustica.sources import CWSource

PAYLOAD_KEYS = {
    "period",
    "periods_expected",
    "step",
    "steps_expected",
    "peak",
    "converge_delta",
    "elapsed_s",
    "eta_s",
    "stage",
}


def mini_setup(shape=(24, 24, 32)):
    grid = hs.Grid(shape=shape, dx=0.75e-3, pml=hs.PMLSpec(thickness=3e-3))
    med = hs.Medium.homogeneous(grid.shape, water())
    idx = np.array([[shape[0] // 2, shape[1] // 2, 4]], dtype=np.int32)
    src = CWSource(
        indices=idx,
        phases=np.zeros(1, np.float32),
        f0=1.0e6,
        amplitude=1.0e5,
        label="point",
    )
    return grid, med, src


def run_mini(progress=None, solver="linear", **kw):
    grid, med, src = mini_setup()
    return get(solver)().run(
        grid,
        med,
        src,
        CWRunSpec(min_settle_periods=2, max_settle_periods=5),
        backend="numpy",
        reference_point=(12, 12, 24),
        progress=progress,
        **kw,
    )


# ------------------------------------------------------------------ trap T1


def test_progress_fires_once_per_period_without_a_checkpoint():
    """T1 regression: the period boundary used to return early with no checkpoint."""
    events: list[dict] = []
    res = run_mini(progress=events.append)

    settle = [e for e in events if e["stage"] == "settle"]
    record = [e for e in events if e["stage"] == "record"]
    # Exactly one settle event per settled period, in order, no gaps, no repeats.
    assert [e["period"] for e in settle] == list(range(1, res.converged_period + 1))
    # ...and exactly one stage transition, visible as a stage change.
    assert len(record) == 1
    assert record[0]["period"] == res.converged_period
    assert [e["stage"] for e in events] == ["settle"] * len(settle) + ["record"]


def test_payload_carries_exactly_the_contract_keys():
    events: list[dict] = []
    res = run_mini(progress=events.append)
    for ev in events:
        assert PAYLOAD_KEYS <= set(ev)
        # `snapshot` is the one extra, documented as NOT serializable.
        assert set(ev) - PAYLOAD_KEYS == {"snapshot"}
        assert ev["steps_expected"] == ev["periods_expected"] * res.spp
        assert ev["elapsed_s"] >= 0.0
    first, last = events[0], events[-1]
    assert first["converge_delta"] is None  # nothing to compare against yet
    assert isinstance(last["peak"], float) and last["peak"] > 0.0
    assert last["step"] > first["step"]


def test_snapshot_is_lazy_and_slices_through_the_reference_point():
    """One device->host copy PER CALL, and only when the consumer asks."""
    events: list[dict] = []
    run_mini(progress=events.append)
    snap = events[-1]["snapshot"]()
    assert snap.shape == (24, 32)  # x-z plane, y fixed at the reference point
    assert snap.dtype == np.float32
    assert np.isfinite(snap).all()


# ------------------------------------------------------- a broken consumer


def test_a_throwing_callback_warns_once_and_the_run_completes():
    calls = {"n": 0}

    def boom(ev):
        calls["n"] += 1
        raise RuntimeError("notebook widget died")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run_mini(progress=boom)
    mine = [
        w
        for w in caught
        if issubclass(w.category, CausticaWarning) and "progress" in str(w.message)
    ]
    assert len(mine) == 1, "warn once, not once per period"
    assert calls["n"] > 1, "the run kept offering progress after the failure"
    assert res.phasor.shape == (24, 24, 32)


def test_progress_does_not_change_the_field():
    """The hook reads; it must not perturb a single float."""
    quiet = run_mini(progress=None)
    loud = run_mini(progress=lambda ev: ev["snapshot"]())
    assert np.array_equal(quiet.phasor, loud.phasor)
    assert np.array_equal(quiet.p_max, loud.p_max)
    assert quiet.converged_period == loud.converged_period
    assert quiet.steps_total == loud.steps_total


def test_both_native_solvers_accept_progress():
    """T2: the kwarg exists on linear AND westervelt, not just the engine."""
    for name in ("linear", "westervelt"):
        events: list[dict] = []
        run_mini(progress=events.append, solver=name)
        assert events, f"{name} never reported"


def test_unknown_kwarg_is_still_refused():
    with pytest.raises(TypeError, match="unknown run.. options"):
        run_mini(progresss=lambda ev: None)  # typo must not pass silently
