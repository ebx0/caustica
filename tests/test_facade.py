"""M10j gates: ``caustica.simulate`` — one call, one code path.

The facade earns its place only if it describes the same world as
``caustica run``. The first test in this file is the one that says so: the
same job, once through the CLI and once in memory, must agree BIT for bit.
The rest pin the input contract, the "writes nothing" promise, and the fact
that ``out=None`` does not quietly skip the pre-run gates.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import caustica
from caustica.config.job import JOB_FORMAT, build_job, load_job, parse_job
from caustica.facade import SETUP_FORMS, SimulationError, simulate
from caustica.runner import EXIT_CONFIG, EXIT_OOM, RunnerOptions


def example_job(quantize: bool = True) -> dict:
    """The packaged zero-data example, as a dict."""
    from caustica.examples import path as example_path

    job = json.loads(example_path("water_bowl_mini").read_text(encoding="utf-8"))
    job["output"] = {"quantize": quantize}
    return job


def write_job(tmp_path: Path, job: dict, name: str = "job.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(job), encoding="utf-8")
    return p


# ------------------------------------------------------------ bit identity


def test_simulate_dict_matches_caustica_run_bit_for_bit(tmp_path):
    """The gate: one job, two entry points, identical floats.

    ``quantize=False`` only removes the STORAGE precision knob from the
    comparison — what is being compared is the solve, not the encoder.
    """
    from caustica.__main__ import main
    from caustica.io.store import load_result

    job = example_job(quantize=False)
    job_path = write_job(tmp_path, job)
    out = tmp_path / "cli"
    assert main(["run", str(job_path), "--out", str(out), "--no-measure"]) == 0
    from_cli = load_result(out / "result.h5")

    in_memory = simulate(job, out=None, progress=None).result

    assert in_memory.phasor.tobytes() == from_cli.phasor.tobytes()
    assert in_memory.p_max.tobytes() == from_cli.p_max.tobytes()
    assert in_memory.steps_total == from_cli.steps_total
    assert in_memory.converged_period == from_cli.converged_period


def test_all_four_setup_forms_produce_the_same_run(tmp_path):
    job = example_job()
    job_path = write_job(tmp_path, job)
    cfg, base_dir = load_job(job_path)
    built = build_job(cfg, base_dir=base_dir, with_medium=True)

    peaks = [
        float(np.abs(simulate(form, out=None, progress=None).result.phasor).max())
        for form in (job_path, str(job_path), job, cfg, built)
    ]
    assert len(set(peaks)) == 1, f"the five spellings disagreed: {peaks}"


def test_an_unsupported_setup_names_the_accepted_forms():
    grid = caustica.Grid(shape=(8, 8, 8), dx=1e-3)
    with pytest.raises(TypeError) as exc:
        simulate(grid)
    msg = str(exc.value)
    assert "Grid" in msg
    for form in SETUP_FORMS:
        assert form in msg
    # ...and it points at the API that DOES take built objects.
    assert "solvers.get" in msg


# --------------------------------------------------------- out=None writes 0


def test_out_none_writes_absolutely_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    res = simulate(example_job(), out=None, progress=None)
    assert set(tmp_path.rglob("*")) == before, "an in-memory run touched the disk"
    assert res.outdir is None
    # ...and it is still a complete answer.
    assert res.metrics["peak"]["p_mpa"] > 0.0
    assert res.preview()["meta"]["format"] == "caustica-preview/1"
    assert res.plan is not None and res.plan["steps_expected"] > 0


def test_out_none_still_applies_the_vram_gate(tmp_path, monkeypatch):
    """Plan-first is not an output-mode feature (the Colab refusal)."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SimulationError) as exc:
        simulate(
            example_job(),
            out=None,
            progress=None,
            options=RunnerOptions(vram_limit_gib=1e-6, measure=False),
        )
    assert exc.value.exit_code == EXIT_OOM
    assert "REFUSED before solving" in str(exc.value)
    assert not list(tmp_path.iterdir()), "a refused run wrote something"


def test_out_none_still_applies_the_cpu_time_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAUSTICA_CPU_LIMIT_MIN", "0")
    with pytest.raises(SimulationError) as exc:
        simulate(example_job(), out=None, progress=None)
    assert exc.value.exit_code == EXIT_CONFIG
    assert "allow-slow-cpu" in str(exc.value)


def test_allow_slow_cpu_is_the_documented_escape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAUSTICA_CPU_LIMIT_MIN", "0")
    res = simulate(example_job(), out=None, progress=None, allow_slow_cpu=True)
    assert res.metrics["peak"]["p_mpa"] > 0.0


# ------------------------------------------------------- out=<path> delegates


def test_out_path_produces_the_full_runner_folder(tmp_path):
    out = tmp_path / "run"
    res = simulate(write_job(tmp_path, example_job()), out=out, progress=None)
    for name in ("job.json", "plan.json", "status.json", "result.h5", "preview.npz"):
        assert (out / name).exists(), f"missing {name}"
    assert res.outdir == out
    # metrics come off the folder the runner wrote — one source, not two.
    assert res.metrics == json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert res.result.phasor.shape == res.p_max.shape
    copied = res.save(tmp_path / "copy.h5")
    assert copied.read_bytes() == (out / "result.h5").read_bytes()


def test_a_refused_delegated_run_raises_with_the_runners_exit_code(tmp_path):
    with pytest.raises(SimulationError) as exc:
        simulate(
            write_job(tmp_path, example_job()),
            out=tmp_path / "run",
            progress=None,
            options=RunnerOptions(vram_limit_gib=1e-6, measure=False),
        )
    assert exc.value.exit_code == EXIT_OOM


def test_a_job_files_relative_paths_still_resolve_against_the_job_file(tmp_path, monkeypatch):
    """T4: the half of the rule the facade can break by handing over a copy."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "elements.csv").write_text(
        "x_mm,y_mm,z_mm\n-3,0,0.3\n3,0,0.3\n0,-3,0.3\n0,3,0.3\n", encoding="utf-8"
    )
    job = example_job()
    job["source"]["array"] = {
        "kind": "elements",
        "elem_radius_mm": 1.0,
        "roc_mm": 12.0,
        "file": "elements.csv",
    }
    job_path = write_job(home, job)

    monkeypatch.chdir(tmp_path)  # a DIFFERENT cwd: only the job file can resolve it
    res = simulate(job_path, out=tmp_path / "run", progress=None)
    assert res.metrics["peak"]["p_mpa"] > 0.0

    # ...and the same job as a DICT has no file to resolve against, so it
    # looks in the CWD and fails loudly instead of silently reading
    # something else. The two forms are not interchangeable, by design.
    with pytest.raises(SimulationError) as exc:
        simulate(job, out=tmp_path / "run2", progress=None)
    assert exc.value.exit_code == EXIT_CONFIG


# ------------------------------------------------------------- job overrides


def test_overrides_are_revalidated_not_smuggled_in(tmp_path):
    job = example_job()
    with pytest.raises(ValueError, match="harmonics"):
        simulate(job, harmonics=(2, 3), out=None, progress=None)
    res = simulate(job, solver="westervelt", out=None, progress=None)
    assert res.job.solver == "westervelt"
    # The untouched job is passed through unchanged (no dump/parse round trip).
    cfg = parse_job(job)
    assert simulate(cfg, out=None, progress=None).job is cfg


def test_progress_reaches_the_facade_callback(tmp_path):
    events = []
    simulate(example_job(), out=None, progress=events.append)
    assert events and {"period", "stage", "peak"} <= set(events[0])
    assert events[-1]["stage"] == "record"


def test_the_job_format_guard_applies_to_dicts_too():
    with pytest.raises(ValueError, match=JOB_FORMAT):
        simulate({"format": "hifusim-job/1", "name": "x"}, out=None, progress=None)
