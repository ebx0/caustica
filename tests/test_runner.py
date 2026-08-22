"""M10c gates: the runner — plan-first, disjoint exit codes, stamp, resume.

Everything runs on numpy with a seconds-scale mini job; the CPU path and the
Colab path are the same code (`run_job_file`), only the backend differs.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from caustica.config.job import JOB_FORMAT
from caustica.io.store import load_result, validate_result_file
from caustica.runner import (
    CANCEL_FILE,
    ERROR_FILE,
    ERROR_FORMAT,
    ERROR_KEYS,
    ERROR_STAGES,
    EXIT_CONFIG,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_OOM,
    EXIT_SOLVER,
    RunnerOptions,
    run_job_file,
)


def mini_job(tmp_path: Path, name: str = "mini", **over) -> Path:
    d = {
        "format": JOB_FORMAT,
        "kind": "explicit",
        "name": name,
        "medium": {"kind": "homogeneous"},
        "grid": {"ndim": 3, "dx_mm": 0.75, "size_mm": [18, 18, 24], "pml": {"thickness_mm": 3.0}},
        "source": {
            "kind": "array",
            "array": {"kind": "bowl", "d_outer_mm": 10.0, "roc_mm": 12.0},
            "apex_mm": [9, 9, 6.0],
        },
        "drive": {"f0_mhz": 1.0, "amplitude_kpa": 100.0},
        "run": {"spec": {"min_settle_periods": 2, "max_settle_periods": 6}, "harmonics": [1]},
        "solver": "linear",
    }
    d.update(over)
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def opts(**kw) -> RunnerOptions:
    kw.setdefault("measure", False)  # skip the 20-step probe in tests
    kw.setdefault("status_interval_s", 0.0)  # write status on every period
    return RunnerOptions(**kw)


# ------------------------------------------------------------------- dry-run


def test_dry_run_writes_plan_and_nothing_else(tmp_path):
    out = tmp_path / "out"
    code = run_job_file(mini_job(tmp_path), opts(out=out, dry_run=True))
    assert code == EXIT_OK
    assert (out / "plan.json").exists() and (out / "plan.txt").exists()
    assert (out / "job.json").exists()  # the normalized copy is part of the audit
    # NOTHING was solved: no field file, no checkpoint, no status.
    assert not (out / "result.h5").exists()
    assert not (out / "checkpoint.npz").exists()
    assert not (out / "status.json").exists()
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert plan["steps_expected"] > 0 and plan["vram_gib"] >= 0.0


# ---------------------------------------------------------------- end-to-end


def test_mini_job_end_to_end_with_full_stamp(tmp_path):
    out = tmp_path / "out"
    code = run_job_file(mini_job(tmp_path), opts(out=out))
    assert code == EXIT_OK
    # Result passes the M10 contract and carries the runner stamp.
    rp = out / "result.h5"
    assert validate_result_file(rp)
    with h5py.File(rp, "r") as hf:
        assert hf.attrs["job_name"] == "mini"
        assert "git_commit" in hf.attrs and "runner" in hf.attrs
    res = load_result(rp)
    assert res.steps_total > 0 and float(np.abs(res.phasor).max()) > 0.0
    # run_meta: environment + planner-vs-actual + re-derivable geometry.
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["format"] == "caustica-run-meta/1"
    assert meta["environment"]["caustica"]
    assert meta["planner"]["steps_expected"] > 0
    assert meta["actual"]["steps_total"] == res.steps_total
    assert meta["actual"]["t_step_measured_s"] > 0
    assert "f_number" in meta["derived"]
    # status ends in 'done' and the checkpoint is gone.
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "done"
    assert not (out / "checkpoint.npz").exists()


def test_skip_guard_never_produces_twice(tmp_path):
    out = tmp_path / "out"
    job = mini_job(tmp_path)
    assert run_job_file(job, opts(out=out)) == EXIT_OK
    mtime = (out / "result.h5").stat().st_mtime_ns
    assert run_job_file(job, opts(out=out)) == EXIT_OK  # completes instantly
    assert (out / "result.h5").stat().st_mtime_ns == mtime  # untouched


# ----------------------------------------------------------------- exit codes


def test_config_error_exit_code(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text('{"format": "caustica-job/1", "kind": "explicit", "nmae": "typo"}')
    assert run_job_file(p, opts(out=tmp_path / "o1")) == EXIT_CONFIG
    assert run_job_file(tmp_path / "missing.json", opts(out=tmp_path / "o2")) == EXIT_CONFIG


def test_oom_refusal_exit_code_and_no_solve(tmp_path):
    out = tmp_path / "out"
    code = run_job_file(mini_job(tmp_path), opts(out=out, vram_limit_gib=1e-5))
    assert code == EXIT_OOM
    assert (out / "plan.json").exists()  # the plan is what refused it
    assert not (out / "result.h5").exists()  # and nothing was paid for


# -------------------------------------------------------- interrupt + resume


def test_interrupt_resume_matches_uninterrupted(tmp_path):
    job = mini_job(tmp_path)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert run_job_file(job, opts(out=out_a)) == EXIT_OK  # the baseline

    # Interrupted run: exit 5, resumable state, NO half-result.
    code = run_job_file(job, opts(out=out_b, stop_after_periods=3))
    assert code == EXIT_INTERRUPTED
    assert (out_b / "checkpoint.npz").exists()
    assert not (out_b / "result.h5").exists()
    status = json.loads((out_b / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "interrupted"
    assert status["periods_done"] == 3  # the heartbeat DID update mid-run

    # Resuming is explicit: without --resume the runner refuses.
    assert run_job_file(job, opts(out=out_b)) == EXIT_CONFIG

    # With --resume it completes, cleans up, and reproduces the baseline
    # bitwise (documented M10 band: rel < 1e-6; identical here).
    assert run_job_file(job, opts(out=out_b, resume=True)) == EXIT_OK
    assert not (out_b / "checkpoint.npz").exists()
    a, b = load_result(out_a / "result.h5"), load_result(out_b / "result.h5")
    np.testing.assert_array_equal(a.phasor, b.phasor)
    np.testing.assert_array_equal(a.p_max, b.p_max)
    assert a.steps_total == b.steps_total
    meta_b = json.loads((out_b / "run_meta.json").read_text(encoding="utf-8"))
    assert meta_b["actual"]["resumed_from_period"] == 3  # honest provenance


def test_max_hours_zero_stops_immediately_but_resumably(tmp_path):
    """--max-hours is the Colab session-budget stop; 0 fires at period 1."""
    job = mini_job(tmp_path)
    out = tmp_path / "out"
    assert run_job_file(job, opts(out=out, max_hours=0.0)) == EXIT_INTERRUPTED
    assert (out / "checkpoint.npz").exists()
    assert run_job_file(job, opts(out=out, resume=True)) == EXIT_OK
    assert validate_result_file(out / "result.h5")


# -------------------------------------------------------------- status detail


def test_status_heartbeat_fields(tmp_path):
    out = tmp_path / "out"
    run_job_file(mini_job(tmp_path), opts(out=out, stop_after_periods=2))
    s = json.loads((out / "status.json").read_text(encoding="utf-8"))
    for key in ("state", "periods_done", "steps_done", "steps_expected", "eta_s", "written_at"):
        assert key in s, f"status.json missing {key!r}"
    assert (
        s["steps_done"]
        == s["periods_done"] * json.loads((out / "plan.json").read_text(encoding="utf-8"))["spp"]
    )


# ------------------------------------------------------------------------ CLI


def test_cli_run_dry(tmp_path):
    from caustica.__main__ import main

    job = mini_job(tmp_path)
    out = tmp_path / "cli-out"
    code = main(["run", str(job), "--out", str(out), "--dry-run", "--no-measure"])
    assert code == EXIT_OK and (out / "plan.json").exists()


# ------------------------------------------- adversarial-review regressions


def test_non_native_solver_gets_no_backend_or_checkpoint_kwargs(tmp_path, monkeypatch):
    """The kwave adapter rejects unknown kwargs; the runner must not send any."""
    import caustica.solvers as solvers

    captured = {}
    orig_get = solvers.get

    class FakeExternal:
        name = "kwave"

        def run(self, grid, medium, source, spec=None, **kwargs):
            captured.update(kwargs)
            if "backend" in kwargs or "checkpoint" in kwargs:
                raise TypeError(f"unknown run() options: {sorted(kwargs)}")
            return orig_get("linear")().run(grid, medium, source, spec, backend="numpy", **kwargs)

    monkeypatch.setattr(solvers, "get", lambda n: FakeExternal if n == "kwave" else orig_get(n))
    out = tmp_path / "out"
    code = run_job_file(mini_job(tmp_path, solver="kwave"), opts(out=out))
    assert code == EXIT_OK
    assert "backend" not in captured and "checkpoint" not in captured
    assert validate_result_file(out / "result.h5")
    # Non-native: no plan, and run_meta records planner as null.
    assert not (out / "plan.json").exists()
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["planner"] is None


def test_unknown_gpu_name_is_a_config_error(tmp_path):
    code = run_job_file(mini_job(tmp_path), opts(out=tmp_path / "out", gpu="H200X"))
    assert code == EXIT_CONFIG  # classified, not a raw traceback with exit 1


def test_store_failure_keeps_checkpoint_and_resume_recovers(tmp_path, monkeypatch):
    """A Drive failure during save must not discard the finished solve."""
    import caustica.runner as runner_mod

    job = mini_job(tmp_path)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert run_job_file(job, opts(out=out_a)) == EXIT_OK  # baseline

    real_save, fail = runner_mod.save_result, {"armed": True}

    def flaky_save(*a, **kw):
        if fail["armed"]:
            fail["armed"] = False
            raise OSError("Drive FUSE mount went stale")
        return real_save(*a, **kw)

    monkeypatch.setattr(runner_mod, "save_result", flaky_save)
    code = run_job_file(job, opts(out=out_b))
    assert code == EXIT_SOLVER
    assert (out_b / "checkpoint.npz").exists()  # the solve is NOT lost
    assert not (out_b / "result.h5").exists()
    status = json.loads((out_b / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed" and "store" in status["error"]

    # Resume redoes only the record window and stores successfully.
    assert run_job_file(job, opts(out=out_b, resume=True)) == EXIT_OK
    assert not (out_b / "checkpoint.npz").exists()
    a, b = load_result(out_a / "result.h5"), load_result(out_b / "result.h5")
    np.testing.assert_array_equal(a.phasor, b.phasor)


def test_output_folder_resolves_against_the_job_file(tmp_path):
    jobdir = tmp_path / "jobs"
    jobdir.mkdir()
    job = mini_job(jobdir, output={"folder": "rel-out"})
    code = run_job_file(job, opts())  # no --out: the job's relative folder wins
    assert code == EXIT_OK
    assert (jobdir / "rel-out" / "result.h5").exists()  # next to the JOB, not the CWD


def test_preview_failure_does_not_fail_the_run(tmp_path, monkeypatch, caplog):
    """M10d contract: the preview is a bonus — its crash NEVER fails the run.

    The result must stay stored and valid, the stamp written, and the
    checkpoint cleaned up exactly as on a preview-less success.
    """
    import caustica.report.preview as preview_mod

    def boom(*a, **kw):
        raise RuntimeError("synthetic preview crash")

    monkeypatch.setattr(preview_mod, "write_preview", boom)
    out = tmp_path / "out"
    with caplog.at_level("WARNING", logger="caustica"):
        code = run_job_file(mini_job(tmp_path), opts(out=out))
    assert code == EXIT_OK
    assert validate_result_file(out / "result.h5")
    assert (out / "run_meta.json").exists()
    assert not (out / "checkpoint.npz").exists()
    assert not (out / "preview.npz").exists()
    assert any("preview package failed" in r.message for r in caplog.records)


# --------------------------------------------------------- M10l: cancel file


def test_cancel_file_stops_at_period_boundary_and_resume_is_bitwise_identical(tmp_path):
    """The GUI "Stop" button's contract: pause, do not lose.

    A ``cancel`` file in the output folder stops the solve at the NEXT period
    boundary with a checkpoint on disk and exit 5 — and the ``--resume`` that
    finishes it reproduces the uninterrupted run bit for bit, which is the
    whole reason a stop button may exist at all.
    """
    job = mini_job(tmp_path)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert run_job_file(job, opts(out=out_a)) == EXIT_OK  # uninterrupted baseline

    # The progress hook plays the part of the outside actor that presses
    # Stop, so the moment is deterministic instead of a sleep race.
    def press_stop_at_period_3(ev):
        if ev["period"] >= 3:
            (out_b / CANCEL_FILE).touch()

    code = run_job_file(job, opts(out=out_b, progress=press_stop_at_period_3))
    assert code == EXIT_INTERRUPTED
    assert (out_b / "checkpoint.npz").exists()
    assert not (out_b / "result.h5").exists()  # no half-result
    status = json.loads((out_b / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "interrupted"
    assert status["periods_done"] == 3  # it stopped at a BOUNDARY, not mid-period
    # The request is consumed: otherwise every --resume would cancel itself.
    assert not (out_b / CANCEL_FILE).exists()
    # Cancelling is not failing — no failure record is left behind.
    assert not (out_b / ERROR_FILE).exists()

    assert run_job_file(job, opts(out=out_b, resume=True)) == EXIT_OK
    assert not (out_b / "checkpoint.npz").exists()
    a, b = load_result(out_a / "result.h5"), load_result(out_b / "result.h5")
    np.testing.assert_array_equal(a.phasor, b.phasor)
    np.testing.assert_array_equal(a.p_max, b.p_max)
    assert a.steps_total == b.steps_total


def test_cancel_poll_is_one_stat_per_period_never_per_step(tmp_path, monkeypatch):
    """The cost gate: a per-step poll would put a syscall between kernels.

    Counted TWO ways, because either one alone is a hole (mutation review,
    2026-08-22). The runner asks through exactly ONE helper, so its call count
    is the poll count whatever filesystem call that helper happens to use; and
    every OTHER spelling of "does `cancel` exist?" is watched separately, so a
    poll that bypassed the helper is caught too. Instrumenting `Path.is_file`
    alone was not enough: the same regression written `os.path.exists(...)` is
    `nt._path_exists` on Windows/py3.12 — a C shortcut that never reaches
    `os.stat` — and 508 per-step polls went unseen with the whole suite green.
    Zero polls is not green either: a stop button nobody asks about is broken.
    """
    import os.path

    import caustica.runner as runner_mod

    boundaries: list[int] = []
    polls: list[str] = []  # through the runner's one poll helper
    bypass: list[str] = []  # any other filesystem question about `cancel`
    inside = False

    real_poll = runner_mod._cancel_requested

    def counting_poll(path):
        nonlocal inside
        polls.append(str(path))
        inside = True  # what the helper itself asks is not a bypass
        try:
            return real_poll(path)
        finally:
            inside = False

    def names_the_cancel_file(arg) -> bool:
        try:
            return os.path.basename(os.fspath(arg)) == CANCEL_FILE
        except TypeError:  # an open fd, not a path
            return False

    def watch(owner, attr: str) -> None:
        real = getattr(owner, attr)

        def wrapper(*a, **kw):
            if not inside and a and names_the_cancel_file(a[0]):
                bypass.append(f"{getattr(owner, '__name__', owner)}.{attr}")
            return real(*a, **kw)

        monkeypatch.setattr(owner, attr, wrapper)

    monkeypatch.setattr(runner_mod, "_cancel_requested", counting_poll)
    for attr in ("is_file", "is_dir", "exists", "stat"):
        watch(Path, attr)
    for attr in ("isfile", "isdir", "exists", "lexists"):
        watch(os.path, attr)  # the C shortcuts a pathlib-only watch misses
    for attr in ("stat", "lstat"):
        watch(os, attr)

    out = tmp_path / "out"
    code = run_job_file(
        mini_job(tmp_path), opts(out=out, progress=lambda ev: boundaries.append(ev["period"]))
    )
    assert code == EXIT_OK
    steps = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))["actual"]["steps_total"]
    spp = json.loads((out / "plan.json").read_text(encoding="utf-8"))["spp"]

    assert bypass == [], f"`{CANCEL_FILE}` is polled outside the one helper: {sorted(set(bypass))}"
    total = len(polls) + len(bypass)
    assert total > 0, "the cancel file was never polled at all"
    # One poll per period boundary (+ the one before the record window)...
    assert total <= len(boundaries) + 1
    # ...which is one poll per spp STEPS. A per-step poll would make these
    # equal; here they differ by exactly the factor the boundary buys.
    assert total < steps
    assert total * spp <= steps + spp


def test_a_stale_cancel_file_does_not_cancel_the_next_run(tmp_path):
    """A process killed between "cancel seen" and "cancel honored" must not
    poison every resume that follows."""
    out = tmp_path / "out"
    out.mkdir()
    (out / CANCEL_FILE).touch()  # leftover from a killed attempt
    assert run_job_file(mini_job(tmp_path), opts(out=out)) == EXIT_OK
    assert not (out / CANCEL_FILE).exists()


def test_a_non_native_solver_says_cancel_does_nothing(tmp_path, monkeypatch, capsys):
    """kwave takes no checkpoints, so it cannot be cancelled — say so."""
    import caustica.solvers as solvers

    orig_get = solvers.get

    class FakeExternal:
        name = "kwave"

        def run(self, grid, medium, source, spec=None, **kwargs):
            return orig_get("linear")().run(grid, medium, source, spec, backend="numpy", **kwargs)

    monkeypatch.setattr(solvers, "get", lambda n: FakeExternal if n == "kwave" else orig_get(n))
    out = tmp_path / "out"
    assert run_job_file(mini_job(tmp_path, solver="kwave"), opts(out=out)) == EXIT_OK
    assert f"a '{CANCEL_FILE}' file has no effect" in capsys.readouterr().out


# ---------------------------------------------------------- M10l: error.json


def _bad_schema(tmp_path, monkeypatch):
    p = tmp_path / "broken.json"
    p.write_text('{"format": "caustica-job/1", "kind": "explicit", "nmae": "typo"}')
    return p, opts(out=tmp_path / "out")


def _wrong_format(tmp_path, monkeypatch):
    return mini_job(tmp_path, format="caustica-job/9"), opts(out=tmp_path / "out")


def _malformed_json(tmp_path, monkeypatch):
    p = tmp_path / "notjson.json"
    p.write_text("{not json")
    return p, opts(out=tmp_path / "out")


def _unknown_backend(tmp_path, monkeypatch):
    return mini_job(tmp_path), opts(out=tmp_path / "out", backend="nope")


def _unknown_gpu(tmp_path, monkeypatch):
    return mini_job(tmp_path), opts(out=tmp_path / "out", gpu="H200X")


def _vram_refusal(tmp_path, monkeypatch):
    return mini_job(tmp_path), opts(out=tmp_path / "out", vram_limit_gib=1e-5)


def _cpu_refusal(tmp_path, monkeypatch):
    monkeypatch.setenv("CAUSTICA_CPU_LIMIT_MIN", "0")  # everything is "too slow"
    return mini_job(tmp_path), opts(out=tmp_path / "out", measure=True)


def _checkpoint_conflict(tmp_path, monkeypatch):
    job, out = mini_job(tmp_path), tmp_path / "out"
    assert run_job_file(job, opts(out=out, stop_after_periods=2)) == EXIT_INTERRUPTED
    assert (out / "checkpoint.npz").exists()
    return job, opts(out=out)  # no --resume: the conflict


def _solver_crash(tmp_path, monkeypatch):
    import caustica.solvers as solvers

    class Exploding:
        def run(self, *a, **kw):
            raise ArithmeticError("synthetic solver crash")

    monkeypatch.setattr(solvers, "get", lambda n: Exploding)
    return mini_job(tmp_path), opts(out=tmp_path / "out")


def _store_crash(tmp_path, monkeypatch):
    import caustica.runner as runner_mod

    def boom(*a, **kw):
        raise OSError("Drive FUSE mount went stale")

    monkeypatch.setattr(runner_mod, "save_result", boom)
    return mini_job(tmp_path), opts(out=tmp_path / "out")


#: (scenario, stage, exit code, error_class) — the failure classes a GUI must
#: be able to route on WITHOUT parsing stderr (M10l). Nine of them, seven
#: distinct classes; the table is asserted, not just enumerated.
ERROR_SCENARIOS = [
    (_bad_schema, "config", EXIT_CONFIG, "ValidationError"),
    (_wrong_format, "config", EXIT_CONFIG, "JobError"),
    (_malformed_json, "config", EXIT_CONFIG, "JSONDecodeError"),
    (_unknown_backend, "config", EXIT_CONFIG, "ValueError"),
    (_unknown_gpu, "plan", EXIT_CONFIG, "ValueError"),
    (_vram_refusal, "gate", EXIT_OOM, "VramRefusal"),
    (_cpu_refusal, "gate", EXIT_CONFIG, "CpuTimeRefusal"),
    (_checkpoint_conflict, "checkpoint", EXIT_CONFIG, "CheckpointConflict"),
    (_solver_crash, "solve", EXIT_SOLVER, "ArithmeticError"),
    (_store_crash, "store", EXIT_SOLVER, "OSError"),
]


@pytest.mark.parametrize(
    ("build", "stage", "code", "error_class"),
    ERROR_SCENARIOS,
    ids=[b.__name__.lstrip("_") for b, *_ in ERROR_SCENARIOS],
)
def test_every_failure_class_writes_a_conformant_error_json(
    tmp_path, monkeypatch, build, stage, code, error_class
):
    job, options = build(tmp_path, monkeypatch)
    assert run_job_file(job, options) == code  # the exit code is UNCHANGED
    payload = json.loads((Path(options.out) / ERROR_FILE).read_text(encoding="utf-8"))
    assert tuple(payload) == ERROR_KEYS  # exactly the contract's keys, in order
    assert payload["format"] == ERROR_FORMAT
    assert payload["stage"] == stage and payload["stage"] in ERROR_STAGES
    assert payload["exit_code"] == code
    assert payload["error_class"] == error_class
    assert payload["message"].strip()
    assert isinstance(payload["advice"], list)
    assert all(isinstance(a, str) and a.strip() for a in payload["advice"])


def test_the_error_table_covers_at_least_seven_distinct_classes():
    """M10l's own criterion, asserted rather than counted by hand."""
    assert len({cls for *_, cls in ERROR_SCENARIOS}) >= 7
    assert {stage for _, stage, _, _ in ERROR_SCENARIOS} == set(ERROR_STAGES)


def test_a_successful_run_writes_no_error_json_and_clears_a_stale_one(tmp_path):
    """error.json means "this folder failed" — nothing weaker."""
    job, out = mini_job(tmp_path), tmp_path / "out"
    # A real failure first, so the stale file is a real one.
    assert run_job_file(job, opts(out=out, vram_limit_gib=1e-5)) == EXIT_OOM
    assert (out / ERROR_FILE).exists()
    assert run_job_file(job, opts(out=out)) == EXIT_OK
    assert not (out / ERROR_FILE).exists()


def test_error_json_lands_even_when_the_job_never_parsed(tmp_path):
    """The GUI case: --out names the folder, so a broken job still explains
    itself in the folder the GUI is already watching."""
    p = tmp_path / "broken.json"
    p.write_text("{not json")
    out = tmp_path / "does" / "not" / "exist" / "yet"
    assert run_job_file(p, opts(out=out)) == EXIT_CONFIG
    assert json.loads((out / ERROR_FILE).read_text(encoding="utf-8"))["stage"] == "config"


def test_a_write_failure_for_error_json_changes_nothing(tmp_path, monkeypatch, caplog):
    """error.json is an ADDITION to the failure contract, never a new way to
    fail: if it cannot be written, the exit code and stderr are untouched."""
    import caustica.runner as runner_mod

    real_write = runner_mod._write_json

    def boom(path, payload):
        if Path(path).name == ERROR_FILE:
            raise OSError("read-only filesystem")
        return real_write(path, payload)

    monkeypatch.setattr(runner_mod, "_write_json", boom)
    out = tmp_path / "out"
    with caplog.at_level("WARNING", logger="caustica"):
        code = run_job_file(mini_job(tmp_path), opts(out=out, vram_limit_gib=1e-5))
    assert code == EXIT_OOM
    assert not (out / ERROR_FILE).exists()
    assert any("error.json write failed" in r.message for r in caplog.records)


def test_a_cancel_directory_cannot_livelock_the_folder(tmp_path):
    """The poll asks `is_file`, not `exists`.

    A directory named `cancel` can never be unlinked, so an `exists()` poll
    would stop every run in that folder at period 1 — forever, `--resume`
    included (review finding, 2026-08-22).
    """
    out = tmp_path / "out"
    out.mkdir()
    (out / CANCEL_FILE).mkdir()
    assert run_job_file(mini_job(tmp_path), opts(out=out)) == EXIT_OK
    assert (out / CANCEL_FILE).is_dir()  # untouched: it was never a request


# ------------------------------------------------ M10l: --dry-run is a PROBE


def test_dry_run_never_touches_the_failure_record_or_the_cancel_file(tmp_path):
    """A fit-check must not erase the diagnosis a GUI is displaying.

    `--dry-run` answers "will this fit?" — it is not an attempt on the
    folder. It must neither delete a real run's `error.json` nor write one
    of its own, and it must not eat a stop request meant for a run that is
    still going (review finding, 2026-08-22).
    """
    job, out = mini_job(tmp_path), tmp_path / "out"
    assert run_job_file(job, opts(out=out, vram_limit_gib=1e-5)) == EXIT_OOM
    real = (out / ERROR_FILE).read_text(encoding="utf-8")
    (out / CANCEL_FILE).touch()  # as if a run in another process were going

    assert run_job_file(job, opts(out=out, dry_run=True)) == EXIT_OK
    assert (out / ERROR_FILE).read_text(encoding="utf-8") == real  # untouched
    assert (out / CANCEL_FILE).exists()

    # ...and a dry run that is itself refused writes no record either: its
    # verdict is the exit code plus plan.json, which carries the advice.
    (out / ERROR_FILE).unlink()
    assert run_job_file(job, opts(out=out, dry_run=True, vram_limit_gib=1e-5)) == EXIT_OOM
    assert not (out / ERROR_FILE).exists()
    assert json.loads((out / "plan.json").read_text(encoding="utf-8"))["vram_gib"] >= 0.0


def test_dry_run_of_a_broken_job_writes_no_error_json(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not json")
    out = tmp_path / "out"
    assert run_job_file(p, opts(out=out, dry_run=True)) == EXIT_CONFIG
    assert not (out / ERROR_FILE).exists()


def test_the_skip_guard_clears_the_stale_error_but_not_a_cancel(tmp_path):
    """A complete folder did not fail — but `cancel` may belong to someone else."""
    job, out = mini_job(tmp_path), tmp_path / "out"
    assert run_job_file(job, opts(out=out)) == EXIT_OK
    (out / ERROR_FILE).write_text('{"stale": true}', encoding="utf-8")
    (out / CANCEL_FILE).touch()
    assert run_job_file(job, opts(out=out)) == EXIT_OK  # skip-guard
    assert not (out / ERROR_FILE).exists()
    assert (out / CANCEL_FILE).exists()
