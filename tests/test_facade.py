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


def test_out_none_still_applies_the_cpu_time_gate(tmp_path, monkeypatch, no_gpu):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAUSTICA_CPU_LIMIT_MIN", "0")
    with pytest.raises(SimulationError) as exc:
        simulate(example_job(), out=None, progress=None)
    assert exc.value.exit_code == EXIT_CONFIG
    assert "allow-slow-cpu" in str(exc.value)


def test_allow_slow_cpu_is_the_documented_escape(tmp_path, monkeypatch, no_gpu):
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


# ------------------------------------------- adversarial-review regressions


def volume_job(tmp_path: Path, rel: str = "elements.csv") -> tuple[Path, dict]:
    """A job whose array table is a RELATIVE path — the T4 tripwire."""
    (tmp_path / rel).write_text(
        "x_mm,y_mm,z_mm\n-3,0,0.3\n3,0,0.3\n0,-3,0.3\n0,3,0.3\n", encoding="utf-8"
    )
    job = example_job()
    job["source"]["array"] = {
        "kind": "elements",
        "elem_radius_mm": 1.0,
        "roc_mm": 12.0,
        "file": rel,
    }
    return write_job(tmp_path, job), job


def test_a_builtjob_remembers_where_its_relative_paths_resolve(tmp_path, monkeypatch):
    """T4 through the BuiltJob form: `build_job` keeps the ORIGINAL job.

    The config it carries still holds `file: "elements.csv"`, so without a
    base_dir on the BuiltJob a re-dump resolves it against a temp directory —
    which fails loudly if nothing is there, and silently runs the WRONG table
    if something is.
    """
    home = tmp_path / "home"
    home.mkdir()
    job_path, _ = volume_job(home)
    cfg, base_dir = load_job(job_path)
    built = build_job(cfg, base_dir=base_dir, with_medium=True)
    assert built.base_dir == home

    monkeypatch.chdir(tmp_path)  # neither the job's folder nor the temp dir
    assert simulate(built, out=tmp_path / "run", progress=None).metrics["peak"]["p_mpa"] > 0.0
    # ...and a rebuild forced by an override resolves against it too.
    assert simulate(built, solver="westervelt", out=None, progress=None).exit_code == 0


def test_the_in_memory_plan_carries_ppw_warnings_like_plan_json(tmp_path):
    """D31: a consumer must tell "no warnings" from "no such field"."""
    job = example_job()
    out = tmp_path / "run"
    written = simulate(write_job(tmp_path, job), out=out, progress=None)
    on_disk = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    in_memory = simulate(job, out=None, progress=None)
    assert set(in_memory.plan) == set(on_disk)
    assert in_memory.plan["ppw_warnings"] == on_disk["ppw_warnings"]
    assert written.plan is not None and "ppw_warnings" in written.plan


def test_both_output_modes_classify_a_broken_job_the_same_way(tmp_path):
    """One failure, one exit code — whichever mode you ran it in."""
    broken = example_job()
    broken["source"]["apex_mm"] = [9.0, 9.0, 400.0]  # focus/apex outside the grid
    with pytest.raises(SimulationError) as in_mem:
        simulate(broken, out=None, progress=None)
    with pytest.raises(SimulationError) as on_disk:
        simulate(write_job(tmp_path, broken), out=tmp_path / "run", progress=None)
    assert in_mem.value.exit_code == on_disk.value.exit_code == EXIT_CONFIG
    assert in_mem.value.__cause__ is not None, "the original error is still chained"


def test_a_misspelled_backend_is_refused_before_the_medium_is_built(tmp_path, monkeypatch):
    """A typo must not cost a multi-GB medium build (runner parity)."""
    import caustica.facade as facade_mod

    built = {"n": 0}
    real = facade_mod.build_job
    monkeypatch.setattr(
        facade_mod,
        "build_job",
        lambda *a, **k: (built.__setitem__("n", built["n"] + 1), real(*a, **k))[1],
    )
    with pytest.raises(SimulationError) as exc:
        simulate(example_job(), backend="cuppy", out=None, progress=None)
    assert exc.value.exit_code == EXIT_CONFIG
    assert built["n"] == 0, "the medium was built before the backend name was checked"


def test_options_are_never_silently_dropped(tmp_path):
    """An out= asked for through options= must not run in memory instead."""
    out = tmp_path / "asked-for"
    res = simulate(example_job(), options=RunnerOptions(out=out), progress=None)
    assert res.outdir == out and (out / "result.h5").exists()


def test_options_that_need_a_folder_are_refused_not_ignored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, value in (
        ("dry_run", True),
        ("max_hours", 2.0),
        ("resume", True),
        ("preview_only", True),
        ("stop_after_periods", 3),
    ):
        with pytest.raises(ValueError, match="in memory"):
            simulate(example_job(), out=None, progress=None, options=RunnerOptions(**{name: value}))


def test_the_facade_default_is_progress_on(tmp_path, monkeypatch, capsys):
    """D21/K11: on by default here, while the library-level default is silent."""
    monkeypatch.chdir(tmp_path)
    simulate(example_job())
    assert "[settle" in capsys.readouterr().err
    # ...and an options-provided renderer is honoured rather than overwritten.
    events = []
    simulate(example_job(), options=RunnerOptions(progress=events.append))
    assert events, "options.progress was overwritten by the facade default"


def test_the_object_and_the_folder_quote_the_same_metrics(tmp_path):
    """The "one source" claim, tested where it is NOT tautological.

    `res.metrics` from a written run reads metrics.json; this compares that
    file against what the IN-MEMORY path computes for the same job.
    """
    job = example_job()
    out = tmp_path / "run"
    written = simulate(write_job(tmp_path, job), out=out, progress=None).metrics
    in_memory = simulate(job, out=None, progress=None).metrics

    def flat(d, prefix=""):
        items = {}
        for k, v in d.items():
            items.update(flat(v, f"{prefix}{k}.")) if isinstance(v, dict) else items.update(
                {f"{prefix}{k}": v}
            )
        return items

    a, b = flat(written), flat(in_memory)
    assert set(a) == set(b)
    differing = {k for k in a if a[k] != b[k]}
    # Only the timestamp may differ — and it need not, if both ran in the
    # same second.
    assert differing <= {"generated"}, f"metrics disagree on {differing - {'generated'}}"
    assert len(a) > 20, "the comparison must actually cover the metric tree"


def test_preview_is_read_back_from_the_package_a_run_wrote(tmp_path):
    """`out=<path>` must not re-derive a package that is already on disk."""
    from caustica.report.preview import load_preview

    out = tmp_path / "run"
    res = simulate(write_job(tmp_path, example_job()), out=out, progress=None)
    from_disk = load_preview(out / "preview.npz")
    via_api = res.preview()
    assert set(via_api) == set(from_disk)
    for key, arr in from_disk.items():
        if key == "meta":
            assert via_api[key] == arr
        else:
            assert np.array_equal(via_api[key], arr), key
    # Removing the file forces the in-memory rebuild, which must still agree.
    (out / "preview.npz").unlink()
    rebuilt = simulate(write_job(tmp_path, example_job(), "again.json"), out=None, progress=None)
    assert set(rebuilt.preview()) == set(from_disk)


def test_out_none_still_warns_about_low_ppw(tmp_path, monkeypatch):
    """D31: the warning is not an output-folder feature."""
    monkeypatch.chdir(tmp_path)
    # The example resolves f0 fine; its SECOND harmonic is what is coarse.
    with pytest.warns(caustica.CausticaWarning, match="low spatial resolution"):
        res = simulate(example_job(), harmonics=(1, 2), out=None, progress=None)
    assert res.warnings and "points per wavelength" in res.warnings[0]
    assert res.plan["ppw_warnings"] == list(res.warnings)
