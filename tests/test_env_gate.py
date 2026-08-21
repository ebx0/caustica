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
