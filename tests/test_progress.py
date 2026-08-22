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


# ------------------------------------------------ the runner as a consumer


def test_runner_status_json_matches_the_pre_m10j_contract(tmp_path):
    """The heartbeat became a consumer; its numbers must not have moved."""
    import json

    from tests.test_runner import mini_job, opts

    from caustica.runner import EXIT_OK, run_job_file

    out = tmp_path / "out"
    assert run_job_file(mini_job(tmp_path), opts(out=out, progress="plain")) == EXIT_OK
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert set(status) == {
        "job",
        "solver",
        "backend",
        "pid",
        "ppw_warnings",
        "state",
        "periods_done",
        "steps_done",
        "steps_expected",
        "steps_worst",
        "eta_s",
        "elapsed_s",
        "written_at",
        "result",
    }
    assert status["steps_done"] == status["periods_done"] * plan["spp"]
    # The documented off-by-one survives: settle periods + the record flip.
    assert status["periods_done"] == meta["actual"]["converged_period"] + 1


def test_progress_does_not_change_the_runners_result(tmp_path):
    from tests.test_runner import mini_job, opts

    from caustica.io.store import load_result
    from caustica.runner import EXIT_OK, run_job_file

    job = mini_job(tmp_path)
    quiet, loud = tmp_path / "quiet", tmp_path / "loud"
    assert run_job_file(job, opts(out=quiet)) == EXIT_OK
    assert run_job_file(job, opts(out=loud, progress="plain")) == EXIT_OK
    a, b = load_result(quiet / "result.h5"), load_result(loud / "result.h5")
    assert np.array_equal(a.phasor, b.phasor)
    assert np.array_equal(a.p_max, b.p_max)


def test_kwave_job_with_progress_set_does_not_crash(tmp_path, monkeypatch):
    """T3: the adapter rejects unknown kwargs, so progress= must not reach it."""
    from tests.test_runner import mini_job, opts

    import caustica.solvers as solvers
    from caustica.io.store import validate_result_file
    from caustica.runner import EXIT_OK, run_job_file

    captured = {}
    orig_get = solvers.get

    class FakeExternal:
        name = "kwave"

        def run(self, grid, medium, source, spec=None, **kwargs):
            captured.update(kwargs)
            if {"backend", "checkpoint", "progress"} & set(kwargs):
                raise TypeError(f"unknown run() options: {sorted(kwargs)}")
            return orig_get("linear")().run(grid, medium, source, spec, backend="numpy", **kwargs)

    monkeypatch.setattr(solvers, "get", lambda n: FakeExternal if n == "kwave" else orig_get(n))
    out = tmp_path / "out"
    code = run_job_file(mini_job(tmp_path, solver="kwave"), opts(out=out, progress="plain"))
    assert code == EXIT_OK
    assert "progress" not in captured
    assert validate_result_file(out / "result.h5")


def test_progress_lines_go_to_stderr_never_to_stdout(tmp_path, capsys):
    """stdout stays the parseable contract (plan text, result path)."""
    from tests.test_runner import mini_job, opts

    from caustica.runner import EXIT_OK, run_job_file

    code = run_job_file(mini_job(tmp_path), opts(out=tmp_path / "a", progress="plain"))
    assert code == EXIT_OK
    cap = capsys.readouterr()
    assert "[settle" in cap.err and "preview @ period" in cap.err
    assert "[settle" not in cap.out and "preview @ period" not in cap.out
    assert "result:" in cap.out


def test_cli_shows_progress_by_default_and_no_progress_silences_it(tmp_path, capsys):
    from tests.test_runner import mini_job

    from caustica.__main__ import main

    job = mini_job(tmp_path)
    assert main(["run", str(job), "--out", str(tmp_path / "a"), "--no-measure"]) == 0
    loud = capsys.readouterr()
    # Captured output is not a tty, so the plain renderer takes over: a
    # rewriting bar in a log file is noise, not progress.
    assert "preview @ period" in loud.err and "[settle" in loud.err
    assert "preview @ period" not in loud.out

    assert (
        main(["run", str(job), "--out", str(tmp_path / "b"), "--no-measure", "--no-progress"]) == 0
    )
    quiet = capsys.readouterr()
    assert "preview @ period" not in quiet.err


# ------------------------------------------------------------ presentation


def test_resolve_refuses_an_unknown_spelling():
    from caustica.progress import resolve

    with pytest.raises(ValueError, match="not one of"):
        resolve("yes")
    assert resolve(None) is None

    def cb(ev):
        return None

    assert resolve(cb) is cb


def test_a_failing_display_does_not_mute_the_heartbeat():
    """One consumer per payload, isolated: chain() reports but keeps going."""
    from caustica.progress import chain

    seen = []

    def broken(ev):
        raise RuntimeError("no terminal")

    fan = chain(seen.append, broken)
    with pytest.raises(RuntimeError):
        fan({"period": 1})
    assert seen == [{"period": 1}], "the healthy consumer still saw the payload"


def test_a_bar_is_only_used_where_something_can_watch_it(monkeypatch):
    """A rewriting tqdm bar piped into a log is noise; plain lines are not."""
    import io

    from caustica.progress import ConsoleProgress

    monkeypatch.delitem(__import__("sys").modules, "ipykernel", raising=False)
    monkeypatch.delitem(__import__("sys").modules, "google.colab", raising=False)

    class Tty(io.StringIO):
        def isatty(self):
            return True

    assert ConsoleProgress(stream=io.StringIO())._tqdm is None  # not a tty
    # With tqdm installed a tty gets a bar; without it, plain lines either way.
    from caustica.progress import _load_tqdm

    expected = _load_tqdm()
    assert ConsoleProgress(stream=Tty())._tqdm is expected
