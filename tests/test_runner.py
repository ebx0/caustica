"""M10c gates: the runner — plan-first, disjoint exit codes, stamp, resume.

Everything runs on numpy with a seconds-scale mini job; the CPU path and the
Colab path are the same code (`run_job_file`), only the backend differs.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import caustica.runner as runner_mod
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


# ---------------------------------------------------------- the exit-code set


def test_the_five_exit_codes_really_are_0_2_3_4_5(tmp_path):
    """The NUMBERS are the API — a queue routes on them, not on the names.

    Every other assertion in this file compares a constant against itself,
    which cannot see a renumbering: the mutation round set ``EXIT_OOM = 6``
    and nothing in this file went red (measured, 2026-08-22). So each of the
    five is pinned to its literal here, once, plus the two properties the
    contract rests on and one real run whose code is compared as a number.
    """
    assert EXIT_OK == 0
    assert EXIT_CONFIG == 2
    assert EXIT_OOM == 3
    assert EXIT_SOLVER == 4
    assert EXIT_INTERRUPTED == 5

    codes = (EXIT_OK, EXIT_CONFIG, EXIT_OOM, EXIT_SOLVER, EXIT_INTERRUPTED)
    assert len(set(codes)) == len(codes)  # disjoint: the queue's whole premise
    # 1 stays reserved for "an exception escaped the classification" — a bug
    # to report, not a verdict to retry. It is deliberately not in the set.
    assert 1 not in codes

    # ...and the numbers are what a caller actually receives, not just what
    # the module defines: a broken job returns 2, a run that cannot fit 3.
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert run_job_file(broken, opts(out=tmp_path / "o1")) == 2
    assert run_job_file(mini_job(tmp_path), opts(out=tmp_path / "o2", vram_limit_gib=1e-5)) == 3


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
    # The request is consumed, so a stopped folder does not advertise a stop
    # nobody will honor. (Not what saves the --resume — see the next test.)
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


def test_the_startup_clear_alone_would_carry_a_resume(tmp_path):
    """Which of the two `cancel` clears is load-bearing — measured, not assumed.

    The runner clears `cancel` twice: at the start of every real run, and
    again in the interrupt handler. The handler's comment used to claim IT
    was what stopped a `--resume` from cancelling itself "forever"; deleting
    that line and rerunning showed the resume completing bit-identically
    anyway (mutation review, 2026-08-22). The startup clear is the safety
    net, and this is the test that says so: the file is put BACK after the
    cancelled run, exactly as if the handler had never consumed it.

    The handler's delete stays as belt-and-braces for a different and real
    reason — a stopped folder must not advertise a stop nobody will honor —
    which `test_cancel_file_stops_at_period_boundary...` above pins.
    """
    job = mini_job(tmp_path)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert run_job_file(job, opts(out=out_a)) == EXIT_OK  # uninterrupted baseline

    def press_stop_at_period_3(ev):
        if ev["period"] >= 3:
            (out_b / CANCEL_FILE).touch()

    assert run_job_file(job, opts(out=out_b, progress=press_stop_at_period_3)) == EXIT_INTERRUPTED
    (out_b / CANCEL_FILE).touch()  # as if the handler had never run

    assert run_job_file(job, opts(out=out_b, resume=True)) == EXIT_OK
    assert not (out_b / CANCEL_FILE).exists()  # the STARTUP clear took it
    a, b = load_result(out_a / "result.h5"), load_result(out_b / "result.h5")
    np.testing.assert_array_equal(a.phasor, b.phasor)
    np.testing.assert_array_equal(a.p_max, b.p_max)


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


#: (scenario, stage, exit code, error_class, wants_advice) — the failure
#: classes a GUI must be able to route on WITHOUT parsing stderr (M10l). Ten
#: of them, seven distinct classes; the table is asserted, not just enumerated.
#: ``wants_advice`` is the measured truth per class, not an aspiration: only a
#: solver crash with no checkpoint on disk has nothing actionable to say.
ERROR_SCENARIOS = [
    (_bad_schema, "config", EXIT_CONFIG, "ValidationError", True),
    (_wrong_format, "config", EXIT_CONFIG, "JobError", True),
    (_malformed_json, "config", EXIT_CONFIG, "JSONDecodeError", True),
    (_unknown_backend, "config", EXIT_CONFIG, "ValueError", True),
    (_unknown_gpu, "plan", EXIT_CONFIG, "ValueError", True),
    (_vram_refusal, "gate", EXIT_OOM, "VramRefusal", True),
    (_cpu_refusal, "gate", EXIT_CONFIG, "CpuTimeRefusal", True),
    (_checkpoint_conflict, "checkpoint", EXIT_CONFIG, "CheckpointConflict", True),
    (_solver_crash, "solve", EXIT_SOLVER, "ArithmeticError", False),
    (_store_crash, "store", EXIT_SOLVER, "OSError", True),
]


@pytest.mark.parametrize(
    ("build", "stage", "code", "error_class", "wants_advice"),
    ERROR_SCENARIOS,
    ids=[b.__name__.lstrip("_") for b, *_ in ERROR_SCENARIOS],
)
def test_every_failure_class_writes_a_conformant_error_json(
    tmp_path, monkeypatch, build, stage, code, error_class, wants_advice
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
    if wants_advice:
        # The `all(...)` above is VACUOUSLY TRUE on an empty list, so nine of
        # these rows say what they actually promise: something to do next
        # (mutation review, 2026-08-22 — emptying every advice tuple left the
        # whole suite green).
        assert payload["advice"], f"the {stage!r} failure must tell the caller what to do"
    else:
        # The one honest exception: a solver that raised with no checkpoint on
        # disk has nothing actionable to offer, and inventing a line would be
        # worse than silence.
        assert payload["advice"] == []


@pytest.mark.parametrize("build", [_vram_refusal, _cpu_refusal], ids=["vram", "cpu"])
def test_a_refusal_prints_the_same_advice_it_writes(tmp_path, monkeypatch, capsys, build):
    """A gate that refuses without saying what to do is a dead end.

    The two pre-run gates exist to be ACTIONABLE, so both renderings are
    pinned here: the ``  -> `` lines a human reads on stderr, and the
    ``advice[]`` a GUI reads from ``error.json`` — and they must be the SAME
    list, because a copy kept only for the file is a copy that drifts.
    """
    job, options = build(tmp_path, monkeypatch)
    assert run_job_file(job, options) in (EXIT_OOM, EXIT_CONFIG)
    err = capsys.readouterr().err
    assert "REFUSED before solving" in err
    arrows = [ln.removeprefix("  -> ") for ln in err.splitlines() if ln.startswith("  -> ")]
    assert arrows, "a refusal printed no actionable line"
    written = json.loads((Path(options.out) / ERROR_FILE).read_text(encoding="utf-8"))["advice"]
    assert written == arrows  # ONE list, two renderings


def test_the_error_table_covers_at_least_seven_distinct_classes():
    """M10l's own criterion, asserted rather than counted by hand."""
    assert len({row[3] for row in ERROR_SCENARIOS}) >= 7
    assert {row[1] for row in ERROR_SCENARIOS} == set(ERROR_STAGES)


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


# ------------------------------------------------- warmup vs steady cost (A2)


def test_step_timing_takes_the_median_between_boundaries_never_the_first():
    """The interval that pays for warmup must never enter the steady rate.

    Synthetic payloads, so the arithmetic is checkable: the solve reaches its
    first boundary at t=5.0 s (a 4.5 s warmup plus 8 steps of real work) and
    then runs at a flat 0.0625 s/step. The steady answer must be the LATER
    rate, not the average that includes the first interval.
    """
    timing = runner_mod._StepTiming()
    for i, (step, elapsed) in enumerate([(8, 5.0), (16, 5.5), (24, 6.0), (32, 6.5)]):
        timing({"step": step, "elapsed_s": elapsed, "period": i + 1, "stage": "settle"})
    assert timing.steady_step_s() == pytest.approx(0.5 / 8)

    split = timing.split(elapsed_s=6.5, session_steps=32)
    assert split["t_step_steady_s"] == pytest.approx(0.0625)
    assert split["warmup_s"] == pytest.approx(4.5, abs=1e-3)
    assert split["steady_samples"] == 4


def test_step_timing_reports_nothing_it_cannot_support():
    """Fewer than two usable intervals -> None, not a number nobody can back."""
    timing = runner_mod._StepTiming()
    assert timing.steady_step_s() is None
    timing({"step": 8, "elapsed_s": 1.0})
    timing({"step": 16, "elapsed_s": 2.0})
    assert timing.steady_step_s() is None  # one interval is not a median

    # The settle->record emission repeats the same step count: a zero-step
    # interval is not a rate, and letting it through would divide by zero.
    timing({"step": 16, "elapsed_s": 2.1})
    timing({"step": 24, "elapsed_s": 3.0})
    timing({"step": 32, "elapsed_s": 4.0})
    assert timing.steady_step_s() == pytest.approx(1.0 / 8)

    split = timing.split(elapsed_s=4.0, session_steps=32)
    assert split["warmup_s"] == 0.0  # negative warmup is not a thing


def test_the_stamp_separates_warmup_from_the_steady_step_cost(tmp_path):
    """A real run's stamp carries the split, and the OLD keys are untouched.

    Fix A2: on the first Colab session ``t_step_measured_s`` read 25.9x the
    planner's per-step probe because a ~2.7 s one-time cost was averaged over
    104 steps. The bundled number stays (M8's gates and the GUI contract read
    it); what is new is the ability to see what it bundles.
    """
    out = tmp_path / "out"
    assert run_job_file(mini_job(tmp_path), opts(out=out)) == EXIT_OK
    actual = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))["actual"]

    for legacy in ("elapsed_solve_s", "elapsed_total_s", "steps_total", "t_step_measured_s"):
        assert legacy in actual, "an existing run_meta.actual key disappeared"

    assert actual["steady_samples"] >= 3
    assert actual["t_step_steady_s"] > 0.0
    assert actual["warmup_s"] >= 0.0
    # The split is exact by construction, so assert the identity rather than
    # a tolerance on a sub-second CPU micro-run: either the steady rate does
    # not explain the whole elapsed time and warmup is the remainder, or it
    # over-explains it and warmup is floored at zero.
    steady_part = actual["t_step_steady_s"] * actual["steps_total"]
    if actual["warmup_s"] > 0.0:
        assert actual["warmup_s"] + steady_part == pytest.approx(
            actual["elapsed_solve_s"], abs=0.02
        )
    else:
        assert steady_part >= actual["elapsed_solve_s"] - 0.02


def test_the_plan_reports_warmup_separately_from_the_per_step_cost(tmp_path):
    out = tmp_path / "out"
    assert run_job_file(mini_job(tmp_path), opts(out=out, dry_run=True)) == EXIT_OK
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert plan["warmup_s"] >= 0.0
    assert plan["t_expected_s"] == pytest.approx(
        plan["warmup_s"] + plan["t_step_s"] * plan["steps_expected"], rel=0.02, abs=0.05
    )
