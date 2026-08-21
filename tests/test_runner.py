"""M10c gates: the runner — plan-first, disjoint exit codes, stamp, resume.

Everything runs on numpy with a seconds-scale mini job; the CPU path and the
Colab path are the same code (`run_job_file`), only the backend differs.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from caustica.config.job import JOB_FORMAT
from caustica.io.store import load_result, validate_result_file
from caustica.runner import (
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
