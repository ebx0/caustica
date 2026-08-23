"""Everything about the multi-engine harness except the physics it measures.

The suite has three separable risks and this file pins them separately:

* the **arithmetic** — normalized relative L2, Pearson r and the peak shift —
  driven from fabricated arrays, so an inverted comparison is caught in
  microseconds rather than after three solves;
* the **grading and the plumbing** — T0 before any comparison, a missing
  environment as a stamped SKIP and never a FAIL, a broken engine as a
  labelled FAIL, the report schema, the exit codes — driven through a fake
  registry whose "solvers" return canned fields, so the whole pipeline runs
  in milliseconds against engines that cannot possibly be right by accident;
* the **real thing**, twice at the bottom: linear vs westervelt on a linear
  medium (bit-identical by M5's guarantee, so relative L2 is exactly zero),
  and — behind the ``kwave`` marker — the live linear-vs-kwave cross-check
  the harness exists to reproduce.

Without the last two, a harness whose engines all returned the same constant
would pass every test above them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from caustica.solvers.base import (
    SolverBase,
    SolverCaps,
    SolverDivergedError,
    SolverResult,
)
from caustica.validation import compare as cmp
from caustica.validation._verdict import EXIT_ENV, EXIT_FAILED, EXIT_OK

TESTS_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------- fabricated fields


def blob(shape, *, peak_pa: float, center_frac=(0.5, 0.5, 0.55), sigma_frac=0.12) -> np.ndarray:
    """A smooth Gaussian lobe of known peak — a field-shaped array, not a field.

    Centres land on whole voxels so the maximum is exactly ``peak_pa``: the
    T0 band is a ratio against an analytic number, and a peak that quietly
    landed between voxels would make every T0 assertion approximate.
    """
    axes = []
    for n, cf in zip(shape, center_frac, strict=True):
        c = round(cf * (n - 1))
        axes.append(np.arange(n) - c)
    sig = [max(1.0, sf * n) for n, sf in zip(shape, [sigma_frac] * len(shape), strict=True)]
    grids = np.meshgrid(*[a / s for a, s in zip(axes, sig, strict=True)], indexing="ij")
    return peak_pa * np.exp(-sum(g**2 for g in grids))


def fake_solver(
    name: str,
    *,
    peak_pa: float | None = None,
    center_frac=(0.5, 0.5, 0.55),
    raises: BaseException | None = None,
    raises_after: int = 0,
    nonfinite: bool = False,
    region=None,
    caps: SolverCaps | None = None,
) -> type[SolverBase]:
    """A registry-shaped solver that returns a canned lobe instead of solving.

    ``raises_after=0`` breaks on the T0 job (the "this machine has no binary"
    shape); ``raises_after=1`` lets T0 through and breaks on the real job.
    ``validate`` is the REAL one from :class:`SolverBase`, so caps filtering
    is exercised against the genuine contract rather than a stub of it.
    """
    default_peak = cmp.t0_expected_peak_pa() if peak_pa is None else peak_pa

    class Fake(SolverBase):
        calls = 0

        def run(self, grid, medium, source, spec=None, **kwargs):
            type(self).calls += 1
            if raises is not None and type(self).calls > raises_after:
                raise raises
            amp = blob(grid.shape, peak_pa=default_peak, center_frac=center_frac)
            if nonfinite:
                amp = amp.copy()
                amp.flat[0] = np.nan
            phasor = amp.astype(np.complex64)
            rec = region or kwargs.get("record_region") or tuple(slice(0, n) for n in grid.shape)
            return SolverResult(
                phasor=phasor[rec],
                p_max=np.abs(phasor[rec]).astype(np.float32),
                region=rec,
                dt=1e-8,
                spp=10,
                steps_total=100,
                t_end_s=1e-6,
                tof_periods=1,
                converged_period=1,
                settle_capped=False,
                convergence_history=[],
                phasors={1: phasor[rec]},
                meta={"solver": name, "backend": kwargs.get("backend", "external")},
            )

    Fake.name = name
    Fake.caps = caps or SolverCaps(
        ndim=frozenset({1, 2, 3}),
        nonlinear=True,
        drive=frozenset({"cw"}),
        backends=frozenset({"numpy", "cupy"}),
    )
    Fake.__name__ = f"Fake{name.title()}"
    return Fake


def fake_harness(**solvers: type[SolverBase]) -> cmp.Harness:
    """A registry of canned engines, in the order they were passed."""
    return cmp.Harness(
        env_report=lambda: {"python": "3.12.10", "numpy": "2.2.6", "resolved_backend": "numpy"},
        backend=lambda _requested: "numpy",
        get_solver=lambda n: solvers[n],
        available=lambda: tuple(solvers),
    )


def tiny_job() -> dict:
    """The T0 geometry reused as the compared job: 40x40x48, built in ~0.1 s.

    Nothing here solves, so the job only has to be a real, buildable scene —
    and the smallest real one this module already owns is the T0 job.
    """
    return cmp.t0_job(name="fake-compare")


def run_compare(tmp_path, harness, **kwargs):
    return cmp.compare(job=tiny_job(), out=tmp_path, harness=harness, log=lambda _m: None, **kwargs)


def gate_by_id(payload: dict, gate_id: str) -> dict:
    (gate,) = [g for g in payload["gates"] if g["id"] == gate_id]
    return gate


def gate_by_id_obj(gates, gate_id):
    """The same lookup over live :class:`Gate` objects, for the pure-``evaluate`` tests."""
    (gate,) = [g for g in gates if g.id == gate_id]
    return gate


def check_by_name(payload: dict, name: str) -> dict:
    for gate in payload["gates"]:
        for check in gate["checks"]:
            if check["name"] == name:
                return check
    raise AssertionError(f"no check named {name!r} in {payload['gates']}")


# ------------------------------------------------------------- the arithmetic


def test_a_field_compared_against_itself_is_perfect_agreement():
    """The fixed point of the whole harness. If this drifts, nothing below means anything."""
    a = blob((16, 16, 20), peak_pa=1.0)
    out = cmp.compare_fields(a, a)
    assert out["rel_l2"] == 0.0
    assert out["pearson_r"] == pytest.approx(1.0, abs=1e-12)
    assert out["peak_shift_vox"] == 0.0
    assert out["peak_ratio"] == pytest.approx(1.0)
    assert out["note"] is None


def test_a_scaled_copy_agrees_because_the_comparison_is_normalized():
    """The old M12 criterion, as a test: amplitude is an engine convention.

    A field twice as loud is the SAME field for cross-engine purposes — that
    is exactly the difference between k-Wave's internal source normalization
    and the native mass-source scaling. The un-normalized peak ratio still
    reports the factor, so the convention stays visible.
    """
    a = blob((16, 16, 20), peak_pa=1.0)
    out = cmp.compare_fields(a, 2.5 * a)
    assert out["rel_l2"] == pytest.approx(0.0, abs=1e-12)
    assert out["pearson_r"] == pytest.approx(1.0, abs=1e-12)
    assert out["peak_ratio"] == pytest.approx(2.5)


def test_a_scrambled_field_fails_the_correlation_gate():
    a = blob((16, 16, 20), peak_pa=1.0)
    scrambled = np.random.default_rng(0).permutation(a.ravel()).reshape(a.shape)
    out = cmp.compare_fields(a, scrambled)
    assert out["pearson_r"] < cmp.CROSS_CORR_MIN
    assert out["rel_l2"] > 0.5
    gates = cmp.evaluate(
        {
            "ref": {
                "status": "ok",
                "t0": {"status": "ok", "finite": True, "peak_over_expected": 1.0},
            }
        },
        [{"reference": "ref", "compared": "other", "status": "ok", **out}],
        "ref",
    )
    assert gate_by_id_obj(gates, "M11.cross").verdict == "FAIL"


def test_a_shifted_peak_is_reported_in_voxels_and_fails_the_focus_gate():
    a = blob((16, 16, 20), peak_pa=1.0, center_frac=(0.5, 0.5, 0.5))
    b = blob((16, 16, 20), peak_pa=1.0, center_frac=(0.5, 0.5, 0.75))
    out = cmp.compare_fields(a, b)
    # round(0.75*19) - round(0.5*19) = 14 - 10 voxels on the last axis.
    assert out["peak_shift_vox"] == 4.0
    assert out["peak_idx_ref"] == [8, 8, 10] and out["peak_idx_cmp"] == [8, 8, 14]


@pytest.mark.parametrize(
    ("other", "fragment"),
    [
        (np.zeros((16, 16, 20)), "no positive peak"),
        (np.full((16, 16, 20), np.nan), "non-finite"),
        (np.ones((16, 16, 20)), "constant"),
        (np.zeros((4, 4, 4)), "shape mismatch"),
    ],
)
def test_an_unmeasurable_comparison_answers_none_and_says_why(other, fragment):
    """None, never zero. A null becomes a SKIP downstream, and SKIP is not a pass —
    a correlation of 0.0 against a dead engine would read as a measurement."""
    out = cmp.compare_fields(blob((16, 16, 20), peak_pa=1.0), other)
    assert out["pearson_r"] is None
    assert fragment in out["note"]


# --------------------------------------------------------------- T0 first


def test_a_dead_engine_fails_the_sanity_gate_instead_of_producing_a_correlation(tmp_path):
    """The whole point of T0: a binary that returns zeros must be LABELLED.

    Without this gate the harness would happily correlate a real field
    against a buffer of zeros, and ``r`` would come back as a number.
    """
    harness = fake_harness(good=fake_solver("good"), dead=fake_solver("dead", peak_pa=0.0))
    code, payload = run_compare(tmp_path, harness)

    assert payload["runs"]["dead"]["status"] == "sanity"
    assert check_by_name(payload, "t0.dead.peak")["verdict"] == "FAIL"
    assert gate_by_id(payload, "M11.t0")["verdict"] == "FAIL"
    # It never reached the comparison, and the pair says so rather than vanishing.
    assert payload["runs"]["dead"]["run"] is None
    (pair,) = payload["pairs"]
    assert pair["status"] == "skipped" and "T0 sanity failed" in pair["message"]
    assert check_by_name(payload, "good-vs-dead.corr")["verdict"] == "SKIP"
    assert code == EXIT_FAILED


def test_a_nan_anywhere_fails_t0_even_when_it_lands_outside_the_measured_window(tmp_path):
    """Finiteness is asked of the WHOLE recorded field, not of the interior.

    The injected NaN sits at voxel (0, 0, 0) — inside the sponge, outside the
    window the peak is measured in — so the focal peak still looks perfectly
    healthy. A NaN in the absorber still means the run is not a number, and
    an engine that produced one does not get to enter a comparison.
    """
    harness = fake_harness(good=fake_solver("good"), sick=fake_solver("sick", nonfinite=True))
    _code, payload = run_compare(tmp_path, harness)
    assert check_by_name(payload, "t0.sick.finite")["verdict"] == "FAIL"
    assert check_by_name(payload, "t0.sick.peak")["verdict"] == "PASS"
    assert payload["runs"]["sick"]["status"] == "sanity"
    assert payload["runs"]["sick"]["run"] is None


def test_an_all_nan_field_leaves_the_peak_unmeasurable_rather_than_zero(tmp_path):
    """The other half: a peak that is NaN is not a peak of zero. It is a SKIP,
    and the FAIL that labels the engine comes from the finiteness check."""
    harness = fake_harness(good=fake_solver("good"), sick=fake_solver("sick", peak_pa=float("nan")))
    _code, payload = run_compare(tmp_path, harness)
    assert check_by_name(payload, "t0.sick.finite")["verdict"] == "FAIL"
    assert check_by_name(payload, "t0.sick.peak")["verdict"] == "SKIP"
    assert payload["runs"]["sick"]["t0"]["peak_over_expected"] is None


def test_t0_runs_before_the_real_job_for_every_engine(tmp_path):
    """Ordering is the contract, not an implementation detail: an engine that
    fails T0 must cost zero seconds of the real job."""
    dead = fake_solver("dead", peak_pa=0.0)
    _code, _payload = run_compare(tmp_path, fake_harness(good=fake_solver("good"), dead=dead))
    assert dead.calls == 1, "the dead engine was asked to solve the real job anyway"


def test_a_peak_that_cannot_be_measured_is_a_t0_failure_not_a_pass(tmp_path):
    """An engine whose recorded window is swallowed by the sponge has no
    interior to grade. That is a T0 failure — it must not walk on to the real
    job carrying an unmeasured "ok"."""
    blind = fake_solver("blind", region=(slice(0, 4), slice(0, 4), slice(0, 4)))
    _code, payload = run_compare(tmp_path, fake_harness(good=fake_solver("good"), blind=blind))
    assert payload["runs"]["blind"]["status"] == "sanity"
    assert "could not be measured" in payload["runs"]["blind"]["message"]
    assert payload["runs"]["blind"]["run"] is None
    assert blind.calls == 1


def test_the_t0_band_is_wide_enough_to_be_a_garbage_detector_not_a_physics_gate():
    """A factor of five on each side. Tightening this into a physics tolerance
    would make the harness fail engines it has no standing to grade."""
    assert cmp.T0_PEAK_BAND == 5.0
    expected = cmp.t0_expected_peak_pa()
    assert expected > 0
    # Half the analytic gain still passes; a dead buffer never does.
    for factor, ok in ((1.0, True), (0.5, True), (4.9, True), (5.1, False), (0.0, False)):
        run = {"status": "ok", "finite": True, "peak_over_expected": factor}
        gates = cmp.evaluate({"s": {"status": "ok", "t0": {**run}}}, [], "s")
        assert (gate_by_id_obj(gates, "M11.t0").verdict == "PASS") is ok, factor


# ------------------------------------------------------- environment-broken


def test_a_missing_environment_is_a_stamped_skip_and_never_a_physics_failure(tmp_path):
    """The k-Wave-less machine. The row survives, carrying the error verbatim."""
    boom = RuntimeError("solver 'kwave' needs the optional dependency k-wave-python")
    harness = fake_harness(good=fake_solver("good"), kwave=fake_solver("kwave", raises=boom))
    code, payload = run_compare(tmp_path, harness)

    run = payload["runs"]["kwave"]
    assert run["status"] == "environment"
    assert str(boom) in run["message"], "the environment error was paraphrased"
    assert check_by_name(payload, "t0.kwave")["verdict"] == "SKIP"
    assert check_by_name(payload, "good-vs-kwave.corr")["verdict"] == "SKIP"
    # Nothing FAILED, so the physics is not red; nothing was compared either,
    # so the milestone cannot close: INCOMPLETE, and exit 2 not 4.
    assert [g["verdict"] for g in payload["gates"]] == ["PASS", "INCOMPLETE", "INCOMPLETE"]
    assert payload["verdict"] == "INCOMPLETE"
    assert code == EXIT_ENV
    assert any("environment" in n for n in payload["notes"])


def test_an_environment_that_breaks_after_t0_is_still_an_environment_skip(tmp_path):
    """A binary present for a 40-cube and gone for the real one is still the machine."""
    boom = FileNotFoundError("kspaceFirstOrder-OMP: no such file")
    harness = fake_harness(
        good=fake_solver("good"), kwave=fake_solver("kwave", raises=boom, raises_after=1)
    )
    code, payload = run_compare(tmp_path, harness)
    assert payload["runs"]["kwave"]["status"] == "environment"
    assert payload["runs"]["kwave"]["t0"]["status"] == "ok"
    assert str(boom) in payload["runs"]["kwave"]["message"]
    assert code == EXIT_ENV


def test_a_diverged_run_is_a_failure_and_not_an_environment_excuse(tmp_path):
    """``SolverDivergedError`` is a RuntimeError. Classifying by type alone
    would let a genuinely broken physics result hide behind exit 2."""
    assert not cmp.is_environment_error(SolverDivergedError("field went to inf"))
    assert cmp.is_environment_error(RuntimeError("binary missing"))
    assert cmp.is_environment_error(ImportError("no module named kwave"))
    assert not cmp.is_environment_error(ValueError("bad harmonics"))

    harness = fake_harness(
        good=fake_solver("good"), wild=fake_solver("wild", raises=SolverDivergedError("inf"))
    )
    code, payload = run_compare(tmp_path, harness)
    assert payload["runs"]["wild"]["status"] == "error"
    assert check_by_name(payload, "t0.wild")["verdict"] == "FAIL"
    assert code == EXIT_FAILED


def test_an_explicitly_named_solver_that_cannot_take_the_job_is_kept_and_labelled(tmp_path):
    """An explicit ``--solvers`` list is obeyed verbatim, including the entry the
    caps refuse: dropping it behind the user's back would answer a question they
    did not ask. The refusal text is the solver's own ``validate`` output, not a
    paraphrase — one capability contract, not a second opinion in the harness."""
    flat = fake_solver(
        "flat",
        caps=SolverCaps(
            ndim=frozenset({1}),
            nonlinear=True,
            drive=frozenset({"cw"}),
            backends=frozenset({"numpy"}),
        ),
    )
    code, payload = run_compare(
        tmp_path, fake_harness(good=fake_solver("good"), flat=flat), solvers=["good", "flat"]
    )
    assert payload["solvers"] == ["good", "flat"]
    assert payload["runs"]["flat"]["status"] == "unsupported"
    assert "supports [1]-D grids" in payload["runs"]["flat"]["message"]
    assert check_by_name(payload, "t0.flat")["verdict"] == "SKIP"
    assert code == EXIT_ENV  # nothing compared, nothing failed


def test_the_default_solver_list_is_the_ones_whose_caps_accept_the_job(tmp_path):
    """And the excluded ones are recorded, not silently missing from the table."""
    only_cupy = fake_solver(
        "gpuonly",
        caps=SolverCaps(
            ndim=frozenset({3}),
            nonlinear=True,
            drive=frozenset({"cw"}),
            backends=frozenset({"cupy"}),
        ),
    )
    _code, payload = run_compare(
        tmp_path, fake_harness(good=fake_solver("good"), gpuonly=only_cupy)
    )
    assert payload["solvers"] == ["good"]
    assert "cupy" in payload["excluded"]["gpuonly"]
    assert any("gpuonly" in n for n in payload["notes"])


# ------------------------------------------------------------ gate algebra


def test_a_check_that_could_not_be_measured_is_never_a_pass():
    gates = cmp.evaluate(
        {"a": {"status": "environment", "t0": {"status": "environment", "message": "no binary"}}},
        [{"reference": "a", "compared": "b", "status": "skipped", "message": "no binary"}],
        "a",
    )
    assert [g.verdict for g in gates] == ["INCOMPLETE", "INCOMPLETE", "INCOMPLETE"]
    assert all(c.verdict == "SKIP" for g in gates for c in g.checks)


def test_an_empty_comparison_cannot_pass_by_having_nothing_to_check():
    """The one-solver run. Zero pairs is not zero failures."""
    gates = cmp.evaluate(
        {"a": {"status": "ok", "t0": {"status": "ok", "finite": True, "peak_over_expected": 1.0}}},
        [],
        "a",
    )
    assert gate_by_id_obj(gates, "M11.t0").verdict == "PASS"
    assert gate_by_id_obj(gates, "M11.cross").verdict == "INCOMPLETE"


def test_the_gate_requires_a_passing_check_from_every_engine_that_could_run():
    runs = {
        "a": {"status": "ok", "t0": {"status": "ok", "finite": True, "peak_over_expected": 1.0}},
        "b": {"status": "ok", "t0": {"status": "ok", "finite": True, "peak_over_expected": 1.0}},
        "c": {"status": "environment", "t0": {"status": "environment", "message": "no binary"}},
    }
    gate = gate_by_id_obj(cmp.evaluate(runs, [], "a"), "M11.t0")
    assert gate.required == 4, "two live engines, two checks each; the broken one is not counted"
    assert gate.verdict == "PASS"


# -------------------------------------------------------------- tolerances


def test_every_gated_tolerance_names_the_test_that_established_it():
    """A harness free to choose its own limits can be tuned until it passes."""
    runs = {
        "a": {"status": "ok", "t0": {"status": "ok", "finite": True, "peak_over_expected": 1.0}},
        "b": {"status": "environment", "t0": {"status": "environment", "message": "gone"}},
    }
    pairs = [
        {"reference": "a", "compared": "b", "status": "ok", "pearson_r": 1.0, "peak_shift_vox": 0.0}
    ]
    for gate in cmp.evaluate(runs, pairs, "a"):
        for check in gate.checks:
            source = check.data.get("source", "")
            assert source.startswith("tests/"), f"{check.name} cites {source!r}"
            assert Path(source.split("::")[0]).name in {
                p.name for p in TESTS_DIR.glob("test_*.py")
            }, f"{check.name} cites a test file that does not exist: {source}"


def test_a_check_that_is_not_a_tolerance_cites_nothing_rather_than_something_irrelevant():
    """ "This engine raised" has no limit to inherit. Hanging the O'Neil gain
    citation on it would put a tolerance next to a stack trace and teach the
    reader to stop believing the column."""
    runs = {"a": {"status": "error", "t0": {"status": "error", "message": "TypeError: boom"}}}
    (check,) = cmp.evaluate(runs, [], "a")[0].checks
    assert (check.name, check.verdict) == ("t0.a", "FAIL")
    assert "source" not in check.data


def test_the_inherited_limits_are_the_ones_the_kwave_cross_check_asserts():
    """Pinned literally against the source test's text, so a quiet loosening here
    (or a tightening there) is a red test rather than a nicer-looking report."""
    assert cmp.CROSS_CORR_MIN == 0.99
    assert cmp.PEAK_SHIFT_MAX_VOX == 1.0
    text = (TESTS_DIR / "test_kwave_adapter.py").read_text(encoding="utf-8")
    assert "assert r > 0.99" in text
    assert "abs(pa[0] - pb[0]) <= 1 and abs(pa[1] - pb[1]) <= 1" in text
    # The environment/physics split is inherited from the same test's skip.
    assert "except (RuntimeError, FileNotFoundError, OSError)" in text
    assert cmp.ENV_BROKEN_SOURCE.startswith("tests/test_kwave_adapter.py")


# ------------------------------------------------------------------ report


def test_the_report_is_written_stamped_and_readable(tmp_path):
    _code, payload = run_compare(
        tmp_path, fake_harness(a=fake_solver("a"), b=fake_solver("b", center_frac=(0.5, 0.5, 0.56)))
    )
    (folder,) = tmp_path.iterdir()
    assert len(folder.name) == len("20260823-061530")

    doc = json.loads((folder / "compare.json").read_text(encoding="utf-8"))
    assert doc["format"] == cmp.FORMAT == "caustica-compare/1"
    for key in ("generated", "caustica", "git_commit", "environment", "backend", "host", "job"):
        assert doc[key], f"the report is not stamped with {key}"
    assert doc == payload
    # The job that was actually built is stored next to the verdict about it.
    assert json.loads((folder / "job_input.json").read_text(encoding="utf-8")) == doc["job"]

    md = (folder / "REPORT.md").read_text(encoding="utf-8")
    assert cmp.FORMAT in md and doc["verdict"] in md
    for gate in doc["gates"]:
        assert gate["id"] in md
        for check in gate["checks"]:
            assert check["name"] in md
    assert "focus_metrics" in md and "peak (MPa)" in md


def test_no_table_row_is_split_by_a_pipe_inside_a_cell(tmp_path):
    """Verbatim environment errors are the natural place for a pipe to arrive:
    a shell command in a "install it like this" message splits the row."""
    boom = RuntimeError("run `pip install caustica[kwave] | tee log` first")
    _code, payload = run_compare(
        tmp_path, fake_harness(a=fake_solver("a"), b=fake_solver("b", raises=boom))
    )
    md = cmp.render_markdown(payload)

    tables, current = [], []
    for line in md.splitlines():
        if line.startswith("|"):
            current.append(line)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    assert tables, "the report has no tables at all"
    for table in tables:
        width = table[0].count("|")
        for row in table:
            assert row.count("|") - row.count("\\|") == width, row
    assert "\\|" in md, "the escaping test escaped nothing"


def test_the_reproduce_line_names_the_job_that_was_actually_run():
    for source, fragment in (
        ("builtin:compare-mini", "compare --solvers"),
        ("example:water_bowl_mini", "--example water_bowl_mini"),
        ("file:/tmp/x.json", "--job /tmp/x.json"),
    ):
        line = cmp.reproduce_command(
            {"job_source": source, "solvers": ["linear"], "backend_requested": "auto"}
        )
        assert fragment in line


# --------------------------------------------------------------------- CLI


def test_the_cli_registers_the_documented_command_line():
    from caustica.validation.__main__ import build_parser

    args = build_parser().parse_args(
        [
            "compare",
            "--example",
            "water_bowl_mini",
            "--solvers",
            "linear,kwave",
            "--backend",
            "numpy",
            "--out",
            "x",
        ]
    )
    assert args.suite == "compare"
    assert (args.example, args.job, args.out, args.backend) == (
        "water_bowl_mini",
        None,
        "x",
        "numpy",
    )
    assert args.solvers == ("linear", "kwave")
    assert build_parser().parse_args(["compare"]).solvers is None
    with pytest.raises(SystemExit):  # --job and --example answer the same question
        build_parser().parse_args(["compare", "--job", "a.json", "--example", "b"])
    with pytest.raises(SystemExit):  # a backend the harness cannot resolve
        build_parser().parse_args(["compare", "--backend", "opencl"])


@pytest.mark.parametrize(
    ("second", "expected"),
    [
        (fake_solver("b", center_frac=(0.5, 0.5, 0.56)), EXIT_OK),
        (fake_solver("b", center_frac=(0.5, 0.5, 0.85)), EXIT_FAILED),
        (fake_solver("b", raises=RuntimeError("no binary here")), EXIT_ENV),
    ],
    ids=["agrees", "disagrees", "environment-broken"],
)
def test_the_cli_exit_code_is_the_whole_machine_readable_verdict(
    tmp_path, monkeypatch, second, expected
):
    from caustica.validation.__main__ import main

    harness = fake_harness(a=fake_solver("a"), b=second)
    monkeypatch.setattr(cmp, "default_harness", lambda: harness)
    monkeypatch.setattr(cmp, "mini_job", tiny_job)
    assert main(["compare", "--out", str(tmp_path)]) == expected


@pytest.mark.parametrize(
    "argv",
    [
        ["compare", "--example", "no_such_example"],
        ["compare", "--job", "no_such_file.json"],
        ["compare", "--solvers", "linear,no_such_solver"],
    ],
    ids=["unknown-example", "missing-job", "unknown-solver"],
)
def test_an_unanswerable_request_exits_two_instead_of_traceback(tmp_path, argv):
    """Exit 2 is "this machine cannot do what you asked" everywhere in this
    package; a stack trace is not a machine-readable verdict."""
    from caustica.validation.__main__ import main

    assert main([*argv, "--out", str(tmp_path)]) == EXIT_ENV


def test_the_packaged_example_is_read_in_place_and_never_written_to(tmp_path):
    """`caustica.examples` warns that running an example in place writes into
    site-packages. This suite writes only into its report folder — pinned,
    because the warning is easy to re-violate."""
    from caustica import examples

    before = sorted(p.name for p in examples.path("water_bowl_mini").parent.iterdir())
    job, base_dir, source = cmp.resolve_job(example="water_bowl_mini")
    assert source == "example:water_bowl_mini"
    assert base_dir == examples.path("water_bowl_mini").parent
    assert job["name"] == "water_bowl_mini"
    _code, _payload = run_compare(tmp_path, fake_harness(a=fake_solver("a")))
    assert sorted(p.name for p in examples.path("water_bowl_mini").parent.iterdir()) == before


def test_at_most_one_job_source_may_be_given():
    with pytest.raises(ValueError, match="at most one"):
        cmp.resolve_job({"format": "caustica-job/1"}, example="water_bowl_mini")


# ------------------------------------------------------------ end to end


@pytest.mark.slow
def test_the_real_harness_reproduces_m5_bit_identity_between_linear_and_westervelt(tmp_path):
    """The one non-k-Wave test that pays for real solves.

    Everything above runs on canned fields and would still pass if the
    engines returned constants. Here two REAL engines run the same mini job
    on a linear (beta = 0) medium, where M5 guarantees not "close" but the
    same arithmetic — so the normalized relative L2 is exactly zero, and any
    stray normalization, cropping or ordering bug in the harness would show
    up as a non-zero number rather than as a slightly different green.
    """
    code, payload = cmp.compare(
        solvers=["linear", "westervelt"], backend="numpy", out=tmp_path, log=lambda _m: None
    )
    assert payload["verdict"] == "PASS", [
        (g["id"], g["verdict"], [c["detail"] for c in g["checks"] if c["verdict"] != "PASS"])
        for g in payload["gates"]
        if g["verdict"] != "PASS"
    ]
    assert code == EXIT_OK
    assert payload["job_source"] == "builtin:compare-mini"
    assert payload["reference"] == "linear"

    (pair,) = payload["pairs"]
    assert pair["rel_l2"] == 0.0, "beta = 0 westervelt is supposed to BE the linear solver"
    assert pair["pearson_r"] == pytest.approx(1.0, abs=1e-12)
    assert pair["peak_shift_vox"] == 0.0
    assert pair["peak_ratio"] == pytest.approx(1.0)

    # Both engines really ran, and T0 really measured a focused field.
    for name in ("linear", "westervelt"):
        run = payload["runs"][name]
        assert run["status"] == "ok" and run["run"]["steps_total"] > 0
        assert 1.0 / cmp.T0_PEAK_BAND < run["t0"]["peak_over_expected"] < cmp.T0_PEAK_BAND
        assert payload["runs"][name]["metrics"]["peak"]["p_mpa"] > 0


@pytest.mark.kwave
@pytest.mark.slow
def test_the_real_harness_reproduces_the_linear_vs_kwave_cross_check(tmp_path):
    """M11's success criterion, verbatim: the existing linear-vs-kwave cross
    checks are reproduced FROM THE HARNESS at r > 0.99.

    Skips exactly the way ``tests/test_kwave_adapter.py`` skips — on a machine
    without the binary the harness stamps ``environment`` and this test reports
    that verbatim, because a measurement that could not be taken is not a pass.
    """
    pytest.importorskip("kwave", reason="k-wave-python not installed")
    code, payload = cmp.compare(
        solvers=["linear", "kwave"], backend="numpy", out=tmp_path, log=lambda _m: None
    )
    kw = payload["runs"]["kwave"]
    if kw["status"] == "environment":
        pytest.skip(f"k-Wave unavailable on this machine: {kw['message']}")

    (pair,) = payload["pairs"]
    assert pair["status"] == "ok"
    assert pair["pearson_r"] > cmp.CROSS_CORR_MIN, (
        f"linear-vs-kwave field correlation r={pair['pearson_r']:.5f} < {cmp.CROSS_CORR_MIN}"
    )
    assert pair["peak_shift_vox"] <= cmp.PEAK_SHIFT_MAX_VOX
    # The amplitude conventions differ but not wildly; the normalization is
    # what makes the shape comparison meaningful, not what hides the gap.
    assert 0.8 < pair["peak_ratio"] < 1.25
    assert kw["run"]["engine_backend"] == "kwave-omp"
    assert payload["verdict"] == "PASS"
    assert code == EXIT_OK
