"""Everything about the analytic suite except the physics it measures.

The suite's job is to grade solver output against ``caustica.analytic`` and
say so in a file somebody else can read. Those are two separable risks, and
this file pins them separately:

* the **grading** — SKIP never counts, a gate short of its required count is
  INCOMPLETE, an exception inside a scenario costs the gate and the exit code
  — is driven entirely from fabricated measurement dicts, so a wrong verdict
  is caught in milliseconds rather than after a solve;
* the **report** — its schema, its stamps, and the rule that every gated
  tolerance names the test it was inherited from.

Then exactly one test runs the real thing end to end on the quick size: a
suite whose grading is perfect and whose scenarios never actually solve would
pass every test above it.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from caustica.validation import analytic_suite as an
from caustica.validation._verdict import Check

# ------------------------------------------------------- fabricated results


def perfect_results() -> dict[str, dict]:
    """Measurements from an imaginary machine where everything agrees.

    Deliberately not "the numbers we measured": these are the numbers a
    PASSING run must look like, so a change that inverts a comparison (a
    floor read as a limit, say) shows up here as a FAIL rather than as a
    slightly different green.
    """
    return {
        "planewave": {
            "amplitude_ratio": 1.02,
            "k_measured_rad_m": 4188.9,
            "k_analytic_rad_m": 4188.79,
            "alpha_measured_np_m": 30.1,
            "alpha_target_np_m": 30.0,
            "elapsed_s": 0.1,
        },
        "oneill": {
            "axial_corr": 0.9993,
            "focus_pos_err_vox": 0.0,
            "width_solver_mm": 16.08,
            "width_oneill_mm": 15.97,
            "elapsed_s": 7.0,
        },
        "linear_limit": {
            "phasor_max_abs_diff_pa": 0.0,
            "pmax_max_abs_diff_pa": 0.0,
            "nonlinear_active": False,
            "elapsed_s": 0.03,
        },
        "fubini": {
            "stations": [
                {"index": i, "sigma": s, "measured_a2_a1": 0.1, "analytic_a2_a1": 0.1, "rel_err": e}
                for i, s, e in [
                    (150, 0.11, 0.031),
                    (250, 0.23, 0.022),
                    (350, 0.35, 0.010),
                    (450, 0.48, 0.013),
                    (540, 0.59, 0.009),
                ]
            ],
            "elapsed_s": 0.15,
        },
    }


def fake_harness(results: dict[str, dict] | None = None, backend: str = "numpy") -> an.Harness:
    """A harness whose scenarios return canned dicts instead of solving."""
    data = perfect_results() if results is None else results

    def scenario(name: str):
        def measure(_size, *, backend=backend):
            value = data[name]
            if isinstance(value, Exception):
                raise value
            return value

        return measure

    return an.Harness(
        env_report=lambda: {"python": "3.12.0", "numpy": "2.2.6", "resolved_backend": backend},
        backend=lambda: backend,
        scenarios={name: scenario(name) for name in data},
    )


def run_suite(tmp_path: Path, results: dict[str, dict] | None = None, **kwargs):
    return an.analytic_suite(
        out=tmp_path / "reports",
        size="quick",
        harness=fake_harness(results),
        log=lambda _msg: None,
        **kwargs,
    )


def gate_by_id(payload: dict, gate_id: str) -> dict:
    (gate,) = [g for g in payload["gates"] if g["id"] == gate_id]
    return gate


# ------------------------------------------------------------ SKIP is not PASS


def test_a_check_that_could_not_be_measured_is_never_a_pass():
    """The two constructors this suite added carry the same rule as the rest."""
    assert Check.at_least("corr", None, 0.99).verdict == "SKIP"
    assert Check.at_least("corr", float("nan"), 0.99).verdict == "SKIP"
    assert Check.in_range("ratio", None, 0.9, 1.12).verdict == "SKIP"
    assert Check.in_range("ratio", float("nan"), 0.9, 1.12).verdict == "SKIP"
    # ... and they still grade what they can see.
    assert Check.at_least("corr", 0.9901, 0.99).verdict == "PASS"
    assert Check.at_least("corr", 0.98, 0.99).verdict == "FAIL"
    assert Check.in_range("ratio", 1.12, 0.9, 1.12).verdict == "PASS"
    assert Check.in_range("ratio", 1.13, 0.9, 1.12).verdict == "FAIL"
    assert Check.in_range("ratio", 0.89, 0.9, 1.12).verdict == "FAIL"


def test_a_missing_scenario_leaves_its_gate_open_rather_than_passing_it():
    gates = an.evaluate({})
    assert [g.verdict for g in gates] == ["INCOMPLETE"] * 4
    assert all(c.verdict == "SKIP" for g in gates for c in g.checks)
    assert an.overall_verdict(gates) == "INCOMPLETE"


def test_a_scenario_that_raised_contributes_nothing(tmp_path):
    """An exception is recorded, not swallowed, and costs the gate."""
    results = perfect_results()
    results["oneill"] = RuntimeError("-6 dB lobe is not fully contained")
    code, payload = run_suite(tmp_path, results)

    assert code == an.EXIT_FAILED
    assert payload["verdict"] == "INCOMPLETE"
    assert gate_by_id(payload, "M4.oneill")["verdict"] == "INCOMPLETE"
    assert all(c["verdict"] == "SKIP" for c in gate_by_id(payload, "M4.oneill")["checks"])
    assert "RuntimeError" in payload["scenarios"]["oneill"]["error"]
    assert any("oneill" in note for note in payload["notes"])
    # The other three gates still closed: one bad scenario does not lose the run.
    assert [g["verdict"] for g in payload["gates"] if g["id"] != "M4.oneill"] == ["PASS"] * 3


def test_a_zero_difference_is_the_only_passing_linear_limit():
    """The one gate whose limit is exactly zero must mean exactly zero."""
    results = perfect_results()
    results["linear_limit"]["phasor_max_abs_diff_pa"] = 1e-9
    gate = [g for g in an.evaluate(results) if g.id == "M5.linear_limit"][0]
    assert gate.verdict == "FAIL"


# --------------------------------------------------------- the gate algebra


def test_a_faithful_machine_passes_every_gate_it_measured(tmp_path):
    code, payload = run_suite(tmp_path)
    assert code == an.EXIT_OK
    assert payload["verdict"] == "PASS"
    assert [g["verdict"] for g in payload["gates"]] == ["PASS"] * 4


@pytest.mark.parametrize(
    ("scenario", "key", "value", "gate_id"),
    [
        # Just outside each inherited tolerance, in the direction that matters.
        ("planewave", "amplitude_ratio", 1.13, "M4.planewave"),
        ("planewave", "amplitude_ratio", 0.89, "M4.planewave"),
        ("planewave", "alpha_measured_np_m", 30.4, "M4.planewave"),
        ("planewave", "k_measured_rad_m", 4200.0, "M4.planewave"),
        ("oneill", "axial_corr", 0.985, "M4.oneill"),
        ("oneill", "focus_pos_err_vox", 1.5, "M4.oneill"),
        ("oneill", "width_solver_mm", 17.5, "M4.oneill"),
    ],
)
def test_one_measurement_outside_its_inherited_tolerance_fails_its_gate(
    tmp_path, scenario, key, value, gate_id
):
    results = perfect_results()
    results[scenario][key] = value
    code, payload = run_suite(tmp_path, results)
    assert code == an.EXIT_FAILED
    assert gate_by_id(payload, gate_id)["verdict"] == "FAIL"
    assert payload["verdict"] == "FAIL"


def test_three_good_fubini_stations_are_not_four(tmp_path):
    """The source test asserts it exercised >= 4 sigma values; so does this."""
    results = perfect_results()
    results["fubini"]["stations"] = results["fubini"]["stations"][:3]
    code, payload = run_suite(tmp_path, results)
    gate = gate_by_id(payload, "M5.fubini")
    assert gate["n_pass"] == 3 and gate["required"] == an.FUBINI_STATIONS_REQUIRED
    assert gate["verdict"] == "INCOMPLETE"
    assert code == an.EXIT_FAILED


def test_a_fubini_scenario_with_no_pre_shock_station_says_why(tmp_path):
    results = perfect_results()
    results["fubini"]["stations"] = []
    _code, payload = run_suite(tmp_path, results)
    gate = gate_by_id(payload, "M5.fubini")
    assert gate["verdict"] == "INCOMPLETE"
    assert gate["checks"] and gate["checks"][0]["verdict"] == "SKIP"
    assert "pre-shock" in gate["checks"][0]["detail"]


# ---------------------------------------------------------------- tolerances


def test_every_gated_tolerance_names_the_test_that_established_it():
    """A suite free to choose its own limits can be tuned until it passes.

    Each check therefore carries the test file that already validated its
    number, and the report prints that column — so this is checkable, and a
    new check that invents a tolerance shows up here.
    """
    for gate in an.evaluate(perfect_results()):
        for check in gate.checks:
            source = check.data.get("source", "")
            assert source.startswith("tests/"), f"{check.name} cites {source!r}"
            assert Path(source.split("::")[0]).name in {
                p.name for p in Path(__file__).parent.glob("test_*.py")
            }, f"{check.name} cites a test file that does not exist: {source}"


def test_the_inherited_limits_are_the_ones_the_physics_tests_assert():
    """Pinned literally, so a quiet tightening here is a red test.

    These are not this suite's opinions — they are the numbers asserted in
    tests/test_linear_planewave.py, tests/test_linear_oneill_3d.py,
    tests/test_westervelt.py and tests/test_review_hardening.py.
    """
    assert an.AMPLITUDE_BAND == (0.90, 1.12)
    assert an.PHASE_SPEED_TOL_PCT == 0.1  # the test's `rel_err < 1e-3`
    assert an.ALPHA_TOL_PCT == 1.0
    assert an.ONEILL_CORR_MIN == 0.99
    assert an.ONEILL_WIDTH_TOL_PCT == 5.0
    assert an.FOCUS_POS_TOL_VOX == 1.0
    assert an.FUBINI_TOL == 0.05
    assert an.FUBINI_STATIONS_REQUIRED == 4


# -------------------------------------------------------------- the report


def test_the_report_is_written_stamped_and_readable(tmp_path):
    _code, payload = run_suite(tmp_path)
    (folder,) = (tmp_path / "reports").iterdir()
    assert folder.name.startswith("numpy-")  # <backend>-<timestamp>

    doc = json.loads((folder / "analytic.json").read_text(encoding="utf-8"))
    assert doc["format"] == an.FORMAT == "caustica-analytic/1"
    for key in ("generated", "caustica", "git_commit", "environment", "backend", "host", "size"):
        assert doc[key], f"the report is not stamped with {key}"
    assert doc["size_preset"]["bowl_shape"] == list(an.SIZES["quick"].shape)
    assert set(doc["scenarios"]) == set(perfect_results())
    assert doc == payload

    md = (folder / "REPORT.md").read_text(encoding="utf-8")
    assert an.FORMAT in md and doc["verdict"] in md
    for gate in doc["gates"]:
        assert gate["id"] in md and gate["criterion"] in md
        for check in gate["checks"]:
            assert check["source"] in md, f"{check['name']} lost its citation in the report"


def test_no_table_row_is_split_by_a_pipe_inside_a_cell():
    """`max |phasor difference|` is the natural name for a check and once
    turned its row into five columns."""
    payload = {
        "verdict": "PASS",
        "generated": "now",
        "gates": [g.as_dict() for g in an.evaluate(perfect_results())],
        "scenarios": perfect_results(),
        "environment": {},
    }
    md = an.render_markdown(payload)
    header = "| check | verdict | measured vs limit | source of the tolerance |"
    rows = md[md.index(header) :].splitlines()[2:]  # past the header separator
    body = list(itertools.takewhile(lambda r: r.startswith("|"), rows))
    assert len(body) == sum(len(g["checks"]) for g in payload["gates"])
    for row in body:
        assert row.count("|") - row.count("\\|") == 5, row
    assert any("\\|" in row for row in body), "the escaping test escaped nothing"


# ------------------------------------------------------------------- CLI


def test_the_cli_registers_the_documented_command_line():
    from caustica.validation.__main__ import build_parser

    args = build_parser().parse_args(["run-analytic", "--size", "quick", "--out", "x"])
    assert (args.suite, args.size, args.out) == ("run-analytic", "quick", "x")
    assert build_parser().parse_args(["run-analytic"]).size == "full"
    with pytest.raises(SystemExit):  # a size the suite has no preset for
        build_parser().parse_args(["run-analytic", "--size", "medium"])


def test_an_unknown_size_is_refused_by_the_suite_itself():
    with pytest.raises(ValueError, match="unknown size"):
        an.size_preset("medium")


@pytest.mark.parametrize(
    ("break_it", "expected"),
    [(False, 0), (True, 4)],
)
def test_the_cli_exits_zero_only_when_every_gate_passes(tmp_path, monkeypatch, break_it, expected):
    """What CI reads. The exit code is the whole machine-readable verdict."""
    from caustica.validation.__main__ import main

    results = perfect_results()
    if break_it:
        results["oneill"]["axial_corr"] = 0.5
    monkeypatch.setattr(an, "default_harness", lambda: fake_harness(results))
    assert main(["run-analytic", "--size", "quick", "--out", str(tmp_path)]) == expected


# ------------------------------------------------------- the size presets


@pytest.mark.parametrize("name", sorted(an.SIZES))
def test_both_bowl_presets_keep_the_source_and_its_lobe_out_of_the_sponge(name):
    """Pure arithmetic on the preset, no solve.

    A bowl whose rim lands in the absorbing band still converges — on a
    quietly wrong field — and a -6 dB lobe that runs off the measurement
    window has no width at all. Both are properties of the numbers in
    :data:`SIZES`, so they are checkable without paying for a run.
    """
    size = an.SIZES[name]
    pml_vox = int(round(size.pml_mm / size.dx_mm))
    nx, _ny, nz = size.shape

    # Lateral: the rim (aperture radius from the axis) clears the band by 2.
    assert size.apex_vox[0] + size.aperture_vox <= nx - pml_vox - 2
    assert size.apex_vox[0] - size.aperture_vox >= pml_vox
    # Axial: the apex is outside the band and the focus is well inside.
    assert size.apex_vox[2] >= pml_vox
    assert size.focus_vox[2] < nz - pml_vox - 2

    # The -6 dB depth of field is ~8 lambda F#^2 (F# = roc / 2a); both its
    # crossings must fall inside the 0.5*roc .. (grid - sponge) window the
    # width is measured in, or minus6db_width has nothing to measure.
    lam_mm = (an.C0 / an.F0) * 1e3
    f_number = size.roc_m / (2.0 * size.aperture_m)
    half_dof_mm = 4.0 * lam_mm * f_number**2
    roc_mm = size.roc_vox * size.dx_mm
    assert roc_mm - half_dof_mm > 0.5 * roc_mm, "proximal -6 dB crossing is outside the window"
    distal_mm = (nz - pml_vox - 2 - size.apex_vox[2]) * size.dx_mm
    assert roc_mm + half_dof_mm < distal_mm, "distal -6 dB crossing is inside the sponge"


def test_the_two_sizes_are_graded_against_the_same_tolerances():
    """`quick` may ask a smaller question; it may not ask an easier one.

    :func:`evaluate` takes only measurements — it never sees the size — so
    there is no place for a relaxed limit to hide. Pinned because a "quick
    mode that passes" is the obvious wrong fix for a slow suite.
    """
    import inspect

    assert "size" not in inspect.signature(an.evaluate).parameters
    assert "size" not in inspect.getsource(an.evaluate)


# ------------------------------------------------------------ end to end


@pytest.mark.slow
def test_the_quick_path_really_solves_grades_and_reports(tmp_path):
    """The one test that pays for real solves.

    Everything above it runs on fabricated numbers, which means all of it
    would still pass if the scenarios returned constants. This one drives the
    real solvers on the quick geometry (~3 s), and asserts both the verdict
    and that the numbers behind it are the physics — a suite that "passes"
    with a correlation of nan or an untouched backend is the failure mode.
    """
    code, payload = an.analytic_suite(out=tmp_path, size="quick", log=lambda _msg: None)

    assert payload["verdict"] == "PASS", [
        (g["id"], g["verdict"], [c["detail"] for c in g["checks"] if c["verdict"] != "PASS"])
        for g in payload["gates"]
        if g["verdict"] != "PASS"
    ]
    assert code == an.EXIT_OK

    scenarios = payload["scenarios"]
    assert all("error" not in s for s in scenarios.values())
    # Every scenario really ran a solver, on the backend the report claims.
    assert {s["backend"] for s in scenarios.values()} == {payload["backend"]}
    assert 0.90 < scenarios["planewave"]["amplitude_ratio"] < 1.12
    assert scenarios["oneill"]["axial_corr"] > an.ONEILL_CORR_MIN
    assert scenarios["oneill"]["steps_total"] > 0
    assert scenarios["linear_limit"]["nonlinear_active"] is False
    assert scenarios["fubini"]["nonlinear_active"] is True
    assert len(scenarios["fubini"]["stations"]) >= an.FUBINI_STATIONS_REQUIRED

    (folder,) = tmp_path.iterdir()
    assert (folder / "REPORT.md").is_file() and (folder / "analytic.json").is_file()
