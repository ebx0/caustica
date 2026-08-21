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
    assert len(cw) == 1 and "ACCEPTED via allow_slow_cpu" in str(cw[0].message)


def test_cpu_run_below_threshold_emits_exactly_one_warning(tmp_path, monkeypatch):
    monkeypatch.delenv("CAUSTICA_CPU_LIMIT_MIN", raising=False)  # default 5 min
    code, cw = run_recording_warnings(mini_job(tmp_path), opts(out=tmp_path / "out"))
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
    assert len(cw) == 1 and "cannot judge" in str(cw[0].message)


def test_dry_run_is_gated_too(tmp_path, monkeypatch, capsys):
    """--dry-run answers 'would this run?' — so the gate applies, exactly
    like the VRAM refusal that precedes it."""
    monkeypatch.setenv("CAUSTICA_CPU_LIMIT_MIN", "0")
    code = run_job_file(mini_job(tmp_path), opts(out=tmp_path / "out", dry_run=True))
    assert code == EXIT_CONFIG
    assert "REFUSED" in capsys.readouterr().err


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
