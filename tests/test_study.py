"""``caustica.study`` — config, run(s), result, figures, report.

A Study is orchestration on top of :func:`caustica.simulate`, so the tests
that matter are the ones that would catch it becoming something else: a
second way to build a job (it must re-validate every variant through the job
model), a second definition of a metric (its numbers must be the run's own
``metrics.json``, byte for byte), or a second import cost (``import
caustica`` must stay matplotlib-free and h5py-free).

The milestone criterion itself — "a 3-point p0 sweep end to end, one combined
report" — is
:func:`test_the_three_point_p0_sweep_writes_one_combined_report`, and the
physics that makes such a sweep believable is next to it: for a linear solver
the focal peak has to track the drive exactly.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from caustica.config.job import JOB_FORMAT
from caustica.runner import EXIT_SOLVER, RunnerOptions
from caustica.study import FORMAT, Study, StudyError

MINI = {
    "format": JOB_FORMAT,
    "kind": "explicit",
    "name": "mini",
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

#: The milestone's own sweep: three drive amplitudes, one decade apart enough
#: that a proportionality check has something to fail on.
P0_KPA = (50.0, 100.0, 200.0)


def mini_job(tmp_path: Path, **over) -> Path:
    """The runner suite's mini job, on disk. Cheap enough to sweep."""
    data = {**MINI, **over}
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def study(tmp_path: Path, name: str = "s", **kw) -> Study:
    """A study over the mini job, with the planner's timing probe skipped."""
    kw.setdefault("options", RunnerOptions(measure=False, status_interval_s=0.0))
    kw.setdefault("out", tmp_path / "study")
    return Study(name, mini_job(tmp_path), **kw)


# ------------------------------------------------------------------ config


def test_a_study_needs_a_name_and_a_setup_it_can_turn_into_a_job(tmp_path):
    """Construction is where a config error belongs — not run 3 of a sweep."""
    import caustica

    with pytest.raises(StudyError, match="non-empty name"):
        Study("  ", mini_job(tmp_path))
    # The facade owns the accepted-input list; the study must not soften it.
    with pytest.raises(TypeError, match="Grid"):
        Study("s", caustica.Grid(shape=(8, 8, 8), dx=1e-3))
    # A misspelled backend costs nothing here, exactly as in the runner.
    with pytest.raises(Exception, match="torch"):
        Study("s", mini_job(tmp_path), backend="torch")


def test_an_unknown_parameter_path_names_the_fields_that_do_exist(tmp_path):
    """The job schema is far too deep to debug by guessing.

    Each of these is a different way to miss, and each has to say something
    different: a wrong leaf, a wrong root, a scalar treated as a container,
    and an index off the end of a list.
    """
    s = study(tmp_path)
    with pytest.raises(StudyError, match=r"Available: amplitude_kpa, f0_mhz, ramp_periods"):
        s.peek("drive.amplitude_kpaa")
    with pytest.raises(StudyError, match=r"not a field of <root>"):
        s.peek("drivee.x")
    with pytest.raises(StudyError, match=r"is a float, not a field container"):
        s.peek("drive.amplitude_kpa.0")
    with pytest.raises(StudyError, match=r"index 9 is out of range"):
        s.peek("source.apex_mm.9")
    with pytest.raises(StudyError, match=r"non-empty string"):
        s.peek("")
    # ...and the address that IS right reads the base job.
    assert s.peek("drive.amplitude_kpa") == 100.0
    assert s.peek("source.apex_mm.2") == 6.0


def test_an_illegal_swept_value_is_refused_before_the_first_solve(tmp_path):
    """Every variant is built and validated up front, so a typo costs 0 runs.

    A negative drive must fail with the JOB MODEL's message (the study adds
    no validation of its own), and the two legal values ahead of it must not
    have been solved: a sweep that dies half way has burned the budget it
    was given to answer a question it now cannot answer.
    """
    s = study(tmp_path)
    with pytest.raises(Exception, match="greater than 0"):
        s.sweep("drive.amplitude_kpa", [50.0, 100.0, -1.0])
    assert not (tmp_path / "study" / "runs").exists(), "a run happened before validation finished"


def test_an_override_never_touches_the_base_job(tmp_path):
    """One pristine mapping feeds every variant.

    A ``set_by_path`` that mutated in place would make run 3 inherit run 2's
    override — the classic sweep bug, and invisible in the report because
    every row would still look plausible.
    """
    s = study(tmp_path, out=None)
    s.job_for({"drive.amplitude_kpa": 999.0})
    assert s.peek("drive.amplitude_kpa") == 100.0
    assert s.base_job.drive.amplitude_kpa == 100.0
    a = s.job_for({"drive.amplitude_kpa": 10.0})
    b = s.job_for({"drive.amplitude_kpa": 20.0})
    assert (a.drive.amplitude_kpa, b.drive.amplitude_kpa) == (10.0, 20.0)


# ---------------------------------------------------------------- one run


def test_one_run_end_to_end_carries_the_stamp_and_both_halves_of_the_prediction(tmp_path):
    """The report's reason to exist: the estimate AND the outcome, together.

    Both halves filled is the assertion. A report that printed only what
    happened would be a log; one that printed only the plan would be a
    brochure.
    """
    run = study(tmp_path).run({"drive.amplitude_kpa": 200.0}, label="hot")
    assert run.ok and run.outdir is not None
    out = run.report()

    payload = json.loads((out / "study.json").read_text(encoding="utf-8"))
    assert payload["format"] == FORMAT and payload["kind"] == "run"
    # Stamp: environment + GPU slot + git, the same composition run_meta uses.
    assert payload["caustica"] and payload["git_commit"] and payload["host"]
    env = payload["environment"]
    assert env["caustica"] and env["python"] and env["resolved_backend"]

    row = payload["runs"][0]
    assert row["job_hash"] and len(row["job_hash"]) == 16
    assert row["overrides"] == {"drive.amplitude_kpa": 200.0}
    # Prediction vs reality, both sides present.
    assert row["expected"]["t_expected_s"] is not None
    assert row["expected"]["steps_expected"] > 0
    assert row["actual"]["elapsed_solve_s"] is not None
    assert row["actual"]["steps_total"] > 0
    assert row["actual"]["converged_period"] is not None
    assert row["actual"]["source"] == "run_meta.json"

    md = (out / "STUDY.md").read_text(encoding="utf-8")
    assert "caustica study — s / hot" in md
    assert row["job_hash"] in md
    assert "expected" in md and "actual" in md
    assert "peak pressure" in md  # the shared row builders, not a private copy


def test_the_report_quotes_the_runs_own_metrics_json(tmp_path):
    """One definition of "peak pressure", not two.

    ``caustica.report.metrics`` is the single source. If the study
    ever recomputed a metric, this equality is what would break — and it
    compares the whole dict, so a new key on either side is caught too.
    """
    run = study(tmp_path).run(label="only")
    on_disk = json.loads((run.outdir / "metrics.json").read_text(encoding="utf-8"))
    assert run.metrics == on_disk


def test_an_in_memory_run_says_where_its_timing_came_from(tmp_path):
    """No folder means no ``run_meta.json``, and the report must not pretend.

    The study's own wall clock stands in, and ``source`` names it — an
    in-memory number silently labelled as the runner's measurement would be
    the kind of small lie a benchmark table is built out of.
    """
    s = study(tmp_path, out=None)
    run = s.run({"drive.amplitude_kpa": 120.0}, label="mem")
    assert run.ok and run.outdir is None
    assert "wall clock" in run.actual["source"]
    assert run.actual["elapsed_solve_s"] > 0 and run.actual["steps_total"] > 0
    with pytest.raises(StudyError, match="no output folder"):
        run.report()
    out = run.report(tmp_path / "elsewhere")
    assert (out / "STUDY.md").exists() and (out / "study.json").exists()
    assert "in memory" in (out / "STUDY.md").read_text(encoding="utf-8")


# ------------------------------------------------------------------ sweep


@pytest.fixture(scope="module")
def p0_sweep(tmp_path_factory):
    """The milestone's 3-point p0 sweep, run ONCE for the tests that read it."""
    tmp_path = tmp_path_factory.mktemp("p0")
    s = study(tmp_path, name="p0-scan")
    sweep = s.sweep("drive.amplitude_kpa", P0_KPA)
    outdir = sweep.report()
    return sweep, outdir


def test_the_three_point_p0_sweep_writes_one_combined_report(p0_sweep):
    """The success criterion, end to end.

    Three runs, one report, and every swept value visible in it — the last
    part is the one worth pinning: a combined table that silently dropped a
    row would still look like a finished sweep.
    """
    sweep, outdir = p0_sweep
    assert len(sweep.runs) == 3 and sweep.ok

    md = (outdir / "STUDY.md").read_text(encoding="utf-8")
    payload = json.loads((outdir / "study.json").read_text(encoding="utf-8"))
    assert payload["format"] == FORMAT and payload["kind"] == "sweep"
    assert payload["param_path"] == "drive.amplitude_kpa"
    assert payload["values"] == list(P0_KPA)
    assert payload["n_runs"] == 3 and payload["n_ok"] == 3
    assert len(payload["runs"]) == 3

    for value in P0_KPA:
        assert str(value) in md, f"the combined table lost {value}"
    for row in payload["runs"]:
        assert row["job_hash"] in md
        assert row["metrics"]["peak"]["p_mpa"] > 0
        assert row["expected"]["steps_expected"] > 0
        assert row["actual"]["elapsed_solve_s"] is not None
    # The stamp is on the sweep, once, not smeared over the rows.
    assert payload["git_commit"] and payload["environment"]["python"]
    assert "Planner vs actual" in md and "Runs — focal metrics" in md


def test_the_swept_peak_tracks_the_drive_for_a_linear_solver(p0_sweep):
    """The physics that makes the sweep believable.

    A linear solver has no mechanism by which doubling the drive does
    anything but double the field, so the report's "departure from
    proportional" column is a self-check on the whole pipeline: source
    normalisation, the solve, and the metric that reads the peak back out.
    Loose on purpose (2%) — this is a sanity gate, not a convergence study.
    """
    sweep, outdir = p0_sweep
    peaks = sweep.peaks_mpa()
    assert all(p is not None for p in peaks)
    ratios = [p / peaks[0] for p in peaks]
    expected = [v / P0_KPA[0] for v in P0_KPA]
    for got, want in zip(ratios, expected, strict=True):
        assert abs(got / want - 1.0) < 0.02, f"peak/{want}x drive scaling drifted: {ratios}"
    # ...and the report states it, so a reader does not have to divide.
    scaling = json.loads((outdir / "study.json").read_text(encoding="utf-8"))["scaling"]
    assert len(scaling) == 3
    assert "Scaling" in (outdir / "STUDY.md").read_text(encoding="utf-8")


def test_each_value_gets_its_own_folder_and_its_own_job_hash(p0_sweep):
    """A sweep that swept nothing would still produce three tidy folders.

    Distinct hashes are the proof that three DIFFERENT jobs ran; the folders
    are what makes each one re-openable with ``caustica report``.
    """
    sweep, outdir = p0_sweep
    hashes = {r.job_hash for r in sweep.runs}
    assert len(hashes) == 3, "two runs solved the same job"
    for run in sweep.runs:
        assert run.outdir.is_dir()
        assert (run.outdir / "result.h5").exists()
        assert (run.outdir / "run_meta.json").exists()
        assert run.outdir.parent == outdir / "runs"
    # No torn writes left behind by the atomic report writer.
    assert not list(outdir.glob("*.tmp"))


def test_the_combined_figure_and_its_caption_come_from_one_place(p0_sweep):
    """The caption rule: the figure builder states the caption, once.

    ``sweep_figures`` returns ``{filename: caption}`` and the report renders
    the image list and the caption list from that same mapping, so a renamed
    figure cannot end up under someone else's caption.
    """
    _, outdir = p0_sweep
    payload = json.loads((outdir / "study.json").read_text(encoding="utf-8"))
    figs = payload["figures"]
    assert figs, "the sweep produced no figure"
    for name, caption in figs.items():
        assert (outdir / name).exists() and (outdir / name).stat().st_size > 0
        md = (outdir / "STUDY.md").read_text(encoding="utf-8")
        assert f"![{name}]({name})" in md
        assert caption in md
        assert "drive.amplitude_kpa" in caption


def test_a_sweep_report_still_writes_its_numbers_without_a_figure(tmp_path):
    """matplotlib is optional (``caustica[report]``); the numbers are not.

    ``figures=False`` is the same path a bare install takes, and it must cost
    the picture only — every table, and all of ``study.json``, still land.
    """
    s = study(tmp_path, name="nofig")
    sweep = s.sweep("drive.amplitude_kpa", [50.0, 100.0])
    outdir = sweep.report(figures=False)
    md = (outdir / "STUDY.md").read_text(encoding="utf-8")
    assert not list(outdir.glob("*.png"))
    assert "## Figures" not in md
    assert "Runs — focal metrics" in md and "Planner vs actual" in md
    payload = json.loads((outdir / "study.json").read_text(encoding="utf-8"))
    assert payload["figures"] == {}
    assert [r["metrics"]["peak"]["p_mpa"] for r in payload["runs"]]


def test_a_failed_run_is_recorded_and_the_rest_of_the_sweep_still_runs(tmp_path):
    """A four-hour sweep must not lose three good runs to the fourth's crash.

    The split the module documents: a CONFIG error raises before anything
    runs (tested above), a RUNTIME failure is recorded on its row and the
    sweep continues — carrying the runner's own exit code, so a queue
    classifies it exactly as it would a direct ``caustica run``.
    """
    s = study(tmp_path, name="mixed")
    sweep = s.sweep("solver", ["linear", "definitely-not-a-solver"])
    assert not sweep.ok and len(sweep.failures) == 1
    good, bad = sweep.runs
    assert good.ok and good.metrics["peak"]["p_mpa"] > 0
    assert not bad.ok and bad.exit_code == EXIT_SOLVER and bad.metrics is None
    with pytest.raises(StudyError, match="produced no result"):
        _ = bad.result

    outdir = sweep.report(figures=False)
    md = (outdir / "STUDY.md").read_text(encoding="utf-8")
    assert "1 / 2 runs completed" in md
    assert "FAILED" in md and "definitely-not-a-solver" in md
    payload = json.loads((outdir / "study.json").read_text(encoding="utf-8"))
    assert payload["n_ok"] == 1 and payload["runs"][1]["exit_code"] == EXIT_SOLVER


# ------------------------------------------------------------------- lazy


def test_import_caustica_does_not_import_the_study_machinery():
    """``caustica.Study`` is PEP 562 lazy, like ``caustica.simulate``.

    Reaching a Study pulls in the runner (h5py) and, at report time,
    matplotlib. Neither may be charged to a plain ``import caustica`` — the
    review found one eager report import doubling CLI startup for every
    command, ``--help`` included.
    """
    code = textwrap.dedent(
        """
        import sys
        import caustica
        assert "Study" in caustica.__all__
        for lazy in ("caustica.study", "caustica.study.core", "matplotlib", "h5py"):
            assert lazy not in sys.modules, f"{lazy} imported eagerly"
        assert caustica.Study.__name__ == "Study"      # ...and it resolves
        print("clean")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout


def test_importing_the_study_package_stays_free_of_matplotlib():
    """``import caustica.study`` must not cost a plotting library.

    Same rule as :mod:`caustica.validation`: the package door is lazy, so
    listing what a study can do never pulls in what only rendering needs.
    """
    code = textwrap.dedent(
        """
        import sys
        import caustica.study
        assert "matplotlib" not in sys.modules, "matplotlib imported by caustica.study"
        assert "caustica.study.core" not in sys.modules, "the package door is not lazy"
        assert sorted(caustica.study.__all__)[:2] == ["FORMAT", "Study"]
        caustica.study.Study                                    # now it loads
        assert "caustica.study.core" in sys.modules
        assert "matplotlib" not in sys.modules, "matplotlib came in with the core"
        print("clean")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout
