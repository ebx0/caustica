"""M10i gates: the slow-CPU refusal (D5/D20) and its escape hatch.

The mini job solves in seconds, so the gate is exercised by moving the
threshold (``CAUSTICA_CPU_LIMIT_MIN``) rather than by hours-long runs; the
full-size refusal is a manual evidence run recorded in the devlog.
"""

import warnings as _warnings

from tests.test_runner import mini_job

from caustica import CausticaWarning
from caustica.runner import EXIT_CONFIG, EXIT_OK, RunnerOptions, run_job_file


def opts(**kw) -> RunnerOptions:
    kw.setdefault("measure", True)  # the gate's honest path: measured-here
    kw.setdefault("status_interval_s", 0.0)
    return RunnerOptions(**kw)


def run_recording_warnings(job, o):
    """Run and collect the RUN-level CausticaWarnings.

    The auto->numpy backend fallback is its own once-per-process warning
    (tested separately below); arming its flag first keeps these counts
    order-independent. A fresh CLI process therefore shows both: one
    fallback notice, plus exactly one gate warning per run.
    """
    from caustica.core import backend as B

    B._AUTO_FALLBACK_WARNED = True
    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        code = run_job_file(job, o)
    return code, [w for w in rec if issubclass(w.category, CausticaWarning)]


def test_cpu_gate_refuses_and_quotes_estimate_and_source(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CAUSTICA_CPU_LIMIT_MIN", "0")  # everything is "too slow"
    out = tmp_path / "out"
    code = run_job_file(mini_job(tmp_path), opts(out=out))
    err = capsys.readouterr().err
    assert code == EXIT_CONFIG  # reused code — the exit set is the queue's API
    assert "REFUSED" in err and "estimated wall time" in err
    assert "source: measured" in err  # the est.source label is quoted
    assert "--allow-slow-cpu" in err and "cupy" in err  # both escapes named
    assert not (out / "result.h5").exists()  # refused BEFORE solving


def test_cpu_gate_escape_hatch_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("CAUSTICA_CPU_LIMIT_MIN", "0")
    out = tmp_path / "out"
    code, cw = run_recording_warnings(mini_job(tmp_path), opts(out=out, allow_slow_cpu=True))
    assert code == EXIT_OK
    assert (out / "result.h5").is_file()
    accepted = [w for w in cw if "ACCEPTED via allow_slow_cpu" in str(w.message)]
    assert len(accepted) == 1  # (mini_job also warns about its 2.0 ppw — separate event)


def test_packaged_example_runs_with_exactly_one_warning(tmp_path, monkeypatch):
    """The M10i criterion verbatim: the PACKAGED example, below the
    threshold, runs with exactly ONE warning (the CPU notice — its ppw is
    3.75 so no resolution warning joins it)."""
    from caustica import examples

    monkeypatch.delenv("CAUSTICA_CPU_LIMIT_MIN", raising=False)  # default 5 min
    job = examples.copy("water_bowl_mini", tmp_path)
    code, cw = run_recording_warnings(job, opts())
    assert code == EXIT_OK
    assert len(cw) == 1  # exactly ONE warning, not a stream
    assert "numpy (CPU) backend" in str(cw[0].message)


def test_cpu_gate_uses_calibrated_cpu_entry_without_measure(tmp_path, monkeypatch, capsys):
    """--no-measure must not gate on the GPU datasheet number: the gate
    rescales through the calibrated cpu entry (absurdly slow here)."""
    from caustica.planner import calibration

    monkeypatch.setattr(
        calibration, "find_calibration_for", lambda key, path=None: {"a": 1.0, "b": 0.0}
    )
    code = run_job_file(mini_job(tmp_path), opts(out=tmp_path / "out", measure=False))
    err = capsys.readouterr().err
    assert code == EXIT_CONFIG
    assert "source: calibrated" in err


def test_cpu_gate_warns_when_it_cannot_judge(tmp_path, monkeypatch):
    """--no-measure and NO cpu calibration: no silent pass-through — a
    warning says the gate cannot judge this run."""
    from caustica.planner import calibration

    monkeypatch.setattr(calibration, "find_calibration_for", lambda key, path=None: None)
    code, cw = run_recording_warnings(mini_job(tmp_path), opts(out=tmp_path / "out", measure=False))
    assert code == EXIT_OK
    assert len([w for w in cw if "cannot judge" in str(w.message)]) == 1


def test_dry_run_previews_the_gate_without_refusing(tmp_path, monkeypatch, capsys):
    """A dry run pays nothing and planning a Colab-bound job on a CPU box is
    a legitimate flow: the gate PREVIEWS its verdict (exit stays 0, the
    dry-run contract scripts key on) instead of refusing."""
    monkeypatch.setenv("CAUSTICA_CPU_LIMIT_MIN", "0")
    code = run_job_file(mini_job(tmp_path), opts(out=tmp_path / "out", dry_run=True))
    captured = capsys.readouterr()
    assert code == EXIT_OK
    assert "would be refused" in captured.out
    assert "REFUSED before solving" not in captured.err


def test_cpu_gate_never_refuses_an_explicit_resume(tmp_path, monkeypatch):
    """A checkpointed CPU run must stay resumable even when the gate would
    refuse a fresh start — refusing strands pre-gate checkpoints and makes
    a 95%-done run unfinishable (review finding, reproduced)."""
    import json

    out = tmp_path / "out"
    job = mini_job(tmp_path)
    # phase 1: accepted slow run, interrupted at the first period boundary
    monkeypatch.setenv("CAUSTICA_CPU_LIMIT_MIN", "0")
    code, _ = run_recording_warnings(
        job, opts(out=out, allow_slow_cpu=True, stop_after_periods=1, max_hours=0.0)
    )
    assert code == 5 and (out / "checkpoint.npz").is_file()  # EXIT_INTERRUPTED
    # phase 2: the previously-valid resume command line, WITHOUT the flag
    code, cw = run_recording_warnings(job, opts(out=out, resume=True))
    assert code == EXIT_OK
    assert (out / "result.h5").is_file()
    assert any("slow-CPU gate is bypassed" in str(w.message) for w in cw)
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["actual"]["resumed_from_period"] is not None  # it truly resumed


def test_cli_wires_allow_slow_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("CAUSTICA_CPU_LIMIT_MIN", "0")
    from caustica.__main__ import main

    job = mini_job(tmp_path)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", CausticaWarning)
        assert main(["run", str(job), "--out", str(tmp_path / "o1"), "--no-measure"]) in (
            EXIT_CONFIG,
            EXIT_OK,
        )  # without the flag: refused when judgeable (calibration-dependent)
        assert (
            main(
                [
                    "run",
                    str(job),
                    "--out",
                    str(tmp_path / "o2"),
                    "--allow-slow-cpu",
                ]
            )
            == EXIT_OK
        )


# ------------------------------------------------------- env_report (M10i)


def test_env_report_returns_dict_and_keeps_the_stamp_keys():
    from caustica import env_report

    rep = env_report()
    assert isinstance(rep, dict)
    # historical stamp keys — renaming any of these breaks M8's Colab gates
    for key in ("caustica", "python", "numpy", "platform"):
        assert key in rep, f"stamp key {key!r} disappeared from env_report()"
    assert rep["resolved_backend"] in ("numpy", "cupy")
    import json

    json.dumps(rep)  # stamp must stay JSON-serializable


def test_run_meta_environment_composes_env_report(tmp_path):
    import json

    out = tmp_path / "out"
    code, _ = run_recording_warnings(mini_job(tmp_path), opts(out=out))
    assert code == EXIT_OK
    env = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))["environment"]
    for key in ("caustica", "python", "numpy", "platform"):
        assert key in env
    assert env["resolved_backend"] == "numpy"


def test_require_gpu_messages_name_the_right_fix(monkeypatch):
    from caustica import cupy_available, require_gpu

    if cupy_available():  # pragma: no cover - dev machines have no GPU
        import pytest

        pytest.skip("GPU present: refusal messages not reachable")
    # local machine: the fix is an install THE USER runs (never pip from us)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    try:
        require_gpu("test")
        raise AssertionError("should have raised")
    except RuntimeError as exc:
        assert "pip install cupy-cuda12x" in str(exc)
    # Colab: the real failure is a CPU runtime; pip cannot fix it
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-x")
    try:
        require_gpu()
        raise AssertionError("should have raised")
    except RuntimeError as exc:
        msg = str(exc)
        assert "Change runtime type" in msg
        assert "pip install cupy" not in msg


def test_auto_fallback_warns_exactly_once_per_process():
    from caustica.core import backend as B

    if B.cupy_available():  # pragma: no cover
        import pytest

        pytest.skip("GPU present: no fallback to warn about")
    old = B._AUTO_FALLBACK_WARNED
    try:
        B._AUTO_FALLBACK_WARNED = False
        with _warnings.catch_warnings(record=True) as rec:
            _warnings.simplefilter("always")
            B.get_backend("auto")
            B.get_backend("auto")
            B.get_backend("auto")
        hits = [w for w in rec if issubclass(w.category, CausticaWarning)]
        assert len(hits) == 1  # once per process, not per call
        assert "falling back to numpy" in str(hits[0].message)
    finally:
        B._AUTO_FALLBACK_WARNED = old


# -------------------------------------------------- VRAM limit uses FREE VRAM


def test_vram_refusal_uses_free_vram_and_says_so(tmp_path, monkeypatch, capsys):
    """The CUDA context eats ~1 GB before the first buffer: gating on TOTAL
    VRAM says 'fits' and dies OOM mid-run. The gate must use FREE VRAM and
    the refusal must name which limit it applied."""
    from types import SimpleNamespace

    from caustica import runner as R

    monkeypatch.setattr(R, "get_backend", lambda name=None: SimpleNamespace(name="cupy"))
    monkeypatch.setattr(
        R,
        "_gpu_environment",
        lambda backend_name: {
            "gpu_name": "FakeGPU",
            "vram_total_gib": 40.0,  # total says "fits" —
            "vram_free_gib": 0.001,  # — free says "no way"
        },
    )
    code = run_job_file(mini_job(tmp_path), opts(out=tmp_path / "out", measure=False))
    err = capsys.readouterr().err
    assert code == 3  # EXIT_OOM
    assert "free device VRAM (FakeGPU)" in err
    assert "0.00 GiB" in err  # the free number, not 40


def test_explicit_vram_limit_still_overrides(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from caustica import runner as R

    monkeypatch.setattr(R, "get_backend", lambda name=None: SimpleNamespace(name="cupy"))
    monkeypatch.setattr(
        R, "_gpu_environment", lambda backend_name: {"gpu_name": "FakeGPU", "vram_free_gib": 39.0}
    )
    code = run_job_file(
        mini_job(tmp_path), opts(out=tmp_path / "out", measure=False, vram_limit_gib=1e-4)
    )
    err = capsys.readouterr().err
    assert code == 3
    assert "requested limit (--vram-limit-gib)" in err


# ------------------------------------------- low ppw is loud in FOUR places


def _low_ppw_run(tmp_path):
    """mini_job at its default dx=0.75/f0=1 MHz is a 2.0 ppw job — the
    warning case, on purpose."""
    import json

    out = tmp_path / "out"
    code, cw = run_recording_warnings(mini_job(tmp_path), opts(out=out))
    assert code == EXIT_OK
    assert any("points per wavelength" in str(w.message) for w in cw)
    read = lambda name: json.loads((out / name).read_text(encoding="utf-8"))  # noqa: E731
    return out, read


def test_ppw_warning_in_the_plan(tmp_path):
    out, read = _low_ppw_run(tmp_path)
    assert any("points per wavelength" in w for w in read("plan.json")["ppw_warnings"])
    assert "points per wavelength" in (out / "plan.txt").read_text(encoding="utf-8")


def test_ppw_warning_in_status_json(tmp_path):
    out, read = _low_ppw_run(tmp_path)
    assert any("points per wavelength" in w for w in read("status.json")["ppw_warnings"])


def test_ppw_warning_in_run_meta(tmp_path):
    out, read = _low_ppw_run(tmp_path)
    assert any("points per wavelength" in w for w in read("run_meta.json")["ppw_warnings"])


def test_ppw_warning_at_the_head_of_the_report(tmp_path):
    import pytest

    pytest.importorskip("matplotlib")
    from caustica.report.run_report import report_out_dir

    out, _ = _low_ppw_run(tmp_path)
    report_out_dir(out)
    md = (out / "REPORT.md").read_text(encoding="utf-8")
    head = "\n".join(md.splitlines()[:10])  # the HEAD, not a footnote
    assert "WARNING" in head and "points per wavelength" in head


# ---------------------------------------- --preview-only + plan size line


def test_plan_carries_expected_result_size(tmp_path):
    import json

    out = tmp_path / "out"
    code, _ = run_recording_warnings(mini_job(tmp_path), opts(out=out, dry_run=True))
    assert code == EXIT_OK
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert plan["result_size_mb_expected"] > 0
    assert "expected result.h5 size" in (out / "plan.txt").read_text(encoding="utf-8")


def test_preview_only_skips_the_field_but_keeps_the_rest(tmp_path):
    import json

    out = tmp_path / "out"
    code, _ = run_recording_warnings(mini_job(tmp_path), opts(out=out, preview_only=True))
    assert code == EXIT_OK
    assert not (out / "result.h5").exists()  # the point of the flag
    assert (out / "preview.npz").is_file()
    assert (out / "metrics.json").is_file()
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["actual"]["preview_only"] is True
    assert not (out / "checkpoint.npz").exists()  # the run is DONE
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "done" and status["result"].endswith("preview.npz")


def test_default_output_unchanged_full_result_plus_preview(tmp_path):
    """D34: the DEFAULT does not change — full field AND preview."""
    out = tmp_path / "out"
    code, _ = run_recording_warnings(mini_job(tmp_path), opts(out=out))
    assert code == EXIT_OK
    assert (out / "result.h5").is_file() and (out / "preview.npz").is_file()


def test_vram_snapshot_is_taken_before_the_plan_probe(tmp_path, monkeypatch):
    """The measure probe fills the cupy pool with ~the run's own footprint
    and the pool keeps its blocks — free VRAM must be read BEFORE the plan
    or big jobs get falsely refused (call-order regression test)."""
    from types import SimpleNamespace

    from caustica import runner as R

    calls = []
    monkeypatch.setattr(R, "get_backend", lambda name=None: SimpleNamespace(name="cupy"))
    monkeypatch.setattr(
        R,
        "_gpu_environment",
        lambda backend_name: (
            calls.append("gpu_env") or {"gpu_name": "FakeGPU", "vram_free_gib": 0.001}
        ),
    )
    real_plan = R._plan

    def spying_plan(built, backend_name, o):
        calls.append("plan")
        return real_plan(built, backend_name, o)

    monkeypatch.setattr(R, "_plan", spying_plan)
    code = run_job_file(mini_job(tmp_path), opts(out=tmp_path / "out", measure=False))
    assert code == 3  # refused on the tiny FREE number
    assert calls.index("gpu_env") < calls.index("plan")  # snapshot BEFORE probe


def test_measure_probe_uses_the_resolved_backend(tmp_path, monkeypatch):
    """estimate(measure=True) must time the backend the run will USE — an
    'auto' probe on a GPU machine forced to numpy would feed the gate a
    cuFFT number labeled 'measured'."""
    import caustica.planner as planner_pkg
    from caustica.planner import calibration

    seen = {}
    real = calibration.measure_step_time

    def spy(shape, nonlinear=False, backend="auto", n_steps=20, warmup=3):
        seen["backend"] = backend
        return real(shape, nonlinear=nonlinear, backend=backend, n_steps=2, warmup=0)

    monkeypatch.setattr(planner_pkg, "measure_step_time", spy)
    code, _ = run_recording_warnings(mini_job(tmp_path), opts(out=tmp_path / "out", measure=True))
    assert code == EXIT_OK
    assert seen["backend"] == "numpy"  # the RESOLVED backend, not "auto"
