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
        "absolute": {
            "drive_over_cap_area": 1.0,
            "ratio_coarse": 1.1165,
            "ratio_fine": 1.0116,
            "error_shrink_factor": 0.10,
            "elapsed_s": 9.0,
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
    assert [g.verdict for g in gates] == ["INCOMPLETE"] * 5
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
    assert [g["verdict"] for g in payload["gates"] if g["id"] != "M4.oneill"] == ["PASS"] * 4


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
    assert [g["verdict"] for g in payload["gates"]] == ["PASS"] * 5


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


# ------------------------------------------- the planner table (M11's first)
#
# The suite reports what the planner PREDICTED next to what the solve cost.
# Two separate risks again: that the table exists and reads correctly for
# every scenario (fabricated rows, below), and that its numbers come from a
# real estimate taken on the real setup before the real solve (the last two
# tests, which pay for the cheapest scenario to find out).


def planned_results() -> dict[str, dict]:
    """:func:`perfect_results` plus the planner rows a real run attaches.

    Shaped like a real numpy run's (2026-08-23): the plane wave and the
    linear limit solve twice, so their rows are sums of two legs.
    """
    results = perfect_results()
    rows = {
        "planewave": {
            "target": "cpu",
            "source": "calibrated",
            "solves": 2,
            "predicted_s": 6.01,
            "measured_s": 0.0773,
            "predicted_steps": 1640,
            "actual_steps": 1640,
            "warmup_s": 6.0,
            "vram_gib": 2.103e-05,
            "note": None,
        },
        "oneill": {
            "target": "cpu",
            "source": "calibrated",
            "solves": 1,
            "predicted_s": 5.248,
            "measured_s": 2.329,
            "predicted_steps": 152,
            "actual_steps": 152,
            "warmup_s": 3.0,
            "vram_gib": 0.02616,
            "note": None,
        },
        # Two rungs, so two solves: the absolute check measures the same bowl
        # at the suite's spacing and at half of it.
        "absolute": {
            "target": "cpu",
            "source": "calibrated",
            "solves": 2,
            "predicted_s": 11.7,
            "measured_s": 6.41,
            "predicted_steps": 456,
            "actual_steps": 456,
            "warmup_s": 6.0,
            "vram_gib": 0.2094,
            "note": None,
        },
        "linear_limit": {
            "target": "cpu",
            "source": "calibrated",
            "solves": 2,
            "predicted_s": 6.002,
            "measured_s": 0.0262,
            "predicted_steps": 768,
            "actual_steps": 768,
            "warmup_s": 6.0,
            "vram_gib": 9.374e-06,
            "note": None,
        },
        "fubini": {
            "target": "cpu",
            "source": "calibrated",
            "solves": 1,
            "predicted_s": 3.042,
            "measured_s": 0.1095,
            "predicted_steps": 2673,
            "actual_steps": 2673,
            "warmup_s": 3.0,
            "vram_gib": 6.174e-05,
            "note": None,
        },
    }
    for name, row in rows.items():
        results[name]["planner"] = row
    return results


def test_every_scenario_gets_a_planner_row_in_the_json_and_in_the_report(tmp_path):
    """One row per scenario, every field present, and the report renders it.

    A table that quietly drops the scenario it could not plan invites the
    reader to assume that one was fine.
    """
    _code, payload = run_suite(tmp_path, planned_results())
    (folder,) = (tmp_path / "reports").iterdir()
    doc = json.loads((folder / "analytic.json").read_text(encoding="utf-8"))
    block = doc["planner"]

    assert set(block["scenarios"]) == set(doc["scenarios"]) == set(perfect_results())
    assert block["target"] == "cpu" and block["source"] == "calibrated"
    for name, row in block["scenarios"].items():
        assert set(row) == set(an.PLAN_ROW_FIELDS), name
        # Every scenario here solved, so nothing in the row may be missing.
        for field in an.PLAN_ROW_FIELDS:
            if field != "note":
                assert row[field] is not None, f"{name}.{field} is null for a solved scenario"
        assert row["predicted_steps"] > 0 and row["actual_steps"] > 0
    assert doc["planner"] == payload["planner"]

    md = (folder / "REPORT.md").read_text(encoding="utf-8")
    header = (
        "| scenario | predicted t [s] | measured t [s] | deviation | predicted steps "
        "| actual steps | vram predicted [GiB] |"
    )
    assert header in md
    body = list(
        itertools.takewhile(lambda r: r.startswith("|"), md[md.index(header) :].splitlines()[2:])
    )
    assert [row.split("|")[1].strip() for row in body] == list(block["scenarios"])
    for row in body:
        assert row.count("|") - row.count("\\|") == 8, row
    # The numbers in the row are the row's, not a re-derivation.
    assert "| 152 | 152 |" in md
    assert "+125.3%" in md  # (5.248 - 2.329) / 2.329


def test_the_planner_rows_are_informational_and_no_gate_can_read_them(tmp_path):
    """A wildly wrong prediction must not cost a physics gate.

    The planner is graded on a device, by ``caustica.validation gpu-gates``
    and by the M8 ±25% criterion — not here, where the setups are hundredths
    of a second and the number is mostly a per-run constant.
    """
    results = planned_results()
    for name in results:
        results[name]["planner"]["predicted_s"] = 1.0e6  # 11 days per scenario

    code, payload = run_suite(tmp_path, results)
    assert code == an.EXIT_OK and payload["verdict"] == "PASS"
    assert [g["verdict"] for g in payload["gates"]] == ["PASS"] * 5
    assert len(payload["gates"]) == 5
    blob = json.dumps(payload["gates"])
    assert "planner" not in blob and "predicted_s" not in blob
    assert payload["planner"]["gated"] is False and payload["planner"]["informational"] is True

    (folder,) = (tmp_path / "reports").iterdir()
    md = (folder / "REPORT.md").read_text(encoding="utf-8")
    assert "Informational, not gated" in md


def test_a_scenario_the_planner_could_not_plan_still_gets_a_row_that_says_why(tmp_path):
    results = planned_results()
    results["oneill"] = RuntimeError("-6 dB lobe is not fully contained")
    del results["fubini"]["planner"]  # solved, but the estimate did not happen

    _code, payload = run_suite(tmp_path, results)
    rows = payload["planner"]["scenarios"]
    assert set(rows) == set(payload["scenarios"])

    assert rows["oneill"]["predicted_s"] is None and rows["oneill"]["actual_steps"] is None
    assert "raised before it could be planned" in rows["oneill"]["note"]
    assert "RuntimeError" in rows["oneill"]["note"]
    assert rows["fubini"]["note"] == "this scenario reported no planner estimate"
    # ... and the rows that were planned are untouched by their neighbours.
    assert rows["planewave"]["predicted_steps"] == 1640

    (folder,) = (tmp_path / "reports").iterdir()
    md = (folder / "REPORT.md").read_text(encoding="utf-8")
    assert "| oneill | -- | -- | -- | -- | -- | -- |" in md


def test_the_estimate_is_taken_on_the_real_setup_before_the_real_solve(monkeypatch):
    """The cheapest real scenario, for the one thing fabrication cannot show.

    An estimate taken AFTER the solve is a prediction written with the
    answer in hand, so the order is asserted rather than assumed; the row's
    numbers are then the real planner's on the real objects.
    """
    import caustica.solvers as solvers

    order: list[str] = []
    real_plan, real_get = an.plan_solve, solvers.get

    def spy_plan(*args, **kwargs):
        order.append("plan")
        return real_plan(*args, **kwargs)

    def spy_get(name):
        order.append("solve")
        return real_get(name)

    monkeypatch.setattr(an, "plan_solve", spy_plan)
    monkeypatch.setattr(solvers, "get", spy_get)

    data = an.measure_linear_limit(an.SIZES["quick"], backend="numpy")
    assert order == ["plan", "solve", "plan", "solve"]  # two legs, each planned first

    row = data["planner"]
    assert row["solves"] == 2 and row["target"] == "cpu"
    assert row["predicted_steps"] > 0 and row["actual_steps"] > 0
    assert row["measured_s"] > 0.0 and row["vram_gib"] > 0.0
    # A wall time is reported only when a measured model backs it: off the
    # GPU that is this machine's "cpu" calibration entry, and gpu_db.json
    # has no datasheet row for a CPU to fall back on.
    if row["source"] == "calibrated":
        assert row["predicted_s"] > 0.0 and row["warmup_s"] is not None
    else:
        assert row["source"] == "db" and row["predicted_s"] is None
        assert "no wall time" in row["note"]


def test_a_cpu_with_no_calibration_reports_no_time_instead_of_inventing_one(monkeypatch):
    """``gpu_db.json`` is a sheet of GPU datasheets; a CPU has no row in it.

    The planner's ``db`` path would happily do arithmetic over the
    placeholder throughput :func:`analytic_suite.plan_target` has to hand it,
    and the result would land in a column the reader is invited to compare
    against a stopwatch. The device-independent half of the estimate — the
    step count and the byte inventory — is real either way, so it stays.
    """
    from caustica import planner

    monkeypatch.setattr(planner, "find_calibration_for", lambda *_a, **_k: None)
    row = an.measure_linear_limit(an.SIZES["quick"], backend="numpy")["planner"]

    assert (row["source"], row["target"]) == ("db", "cpu")
    assert row["predicted_s"] is None and row["warmup_s"] is None
    assert "no wall time" in row["note"]
    assert row["predicted_steps"] > 0 and row["actual_steps"] > 0 and row["vram_gib"] > 0


def test_a_planner_that_raises_costs_its_row_and_nothing_else(monkeypatch):
    """The physics is the point; the informational table is not allowed to
    take a scenario down with it."""
    from caustica import planner

    def boom(*_args, **_kwargs):
        raise RuntimeError("gpu_db.json is unreadable")

    monkeypatch.setattr(planner, "estimate", boom)
    data = an.measure_linear_limit(an.SIZES["quick"], backend="numpy")

    assert data["phasor_max_abs_diff_pa"] == 0.0  # the scenario still measured
    assert data["planner"]["predicted_s"] is None
    assert "planner.estimate raised RuntimeError" in data["planner"]["note"]


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


# ------------------------------------- the absolute-amplitude gate (M30)


def test_the_absolute_gate_catches_a_source_that_does_not_shrink():
    """The gate exists because of a defect; it has to fail on that defect.

    A source-model error is a constant offset: refining the grid does not
    move it. That is what separates it from an honest discretization error,
    and it is what the binary bowl did — measured 1.1275 at four points per
    wavelength and 1.1932 at eight, so the error GREW. Every one of the three
    checks has to see it, or the gate is decoration.
    """
    scenarios = perfect_results()
    scenarios["absolute"] = {
        "drive_over_cap_area": 1.2146,  # the staircase factor, measured
        "ratio_coarse": 1.1275,
        "ratio_fine": 1.1932,
        "error_shrink_factor": 1.515,
        "elapsed_s": 9.0,
    }
    gate = next(g for g in an.evaluate(scenarios) if g.id == "M30.absolute")

    assert gate.verdict == "FAIL"
    assert [c.verdict for c in gate.checks] == ["FAIL", "FAIL", "FAIL"]


def test_the_absolute_gate_passes_the_shipped_source_with_room():
    """...and it must not be so tight that an honest run trips it.

    Measured on the shipped off-grid bowl at the suite's quick preset: the
    drive is the cap's area to five figures, the fine rung sits at 1.0116,
    and a 2x refinement cut the error to a tenth. The limits leave six times
    that much room on the discriminating check.
    """
    gate = next(g for g in an.evaluate(perfect_results()) if g.id == "M30.absolute")

    assert gate.verdict == "PASS"
    assert an.ABSOLUTE_SHRINK_MAX > 5 * 0.10, "no margin left on the shrink check"


@pytest.mark.parametrize(
    "field,value",
    [
        ("drive_over_cap_area", 1.02),  # 2 % of area is 20x the tolerance
        ("ratio_fine", 1.15),
        ("error_shrink_factor", 0.95),
    ],
)
def test_each_absolute_check_fails_on_its_own(field, value):
    """Three checks, three independent ways to be wrong.

    A gate whose checks all key on the same number is one check wearing
    three names — and this one is deliberately layered: the source's measure
    is exact and instant, the level is what a reader quotes, and the shrink
    is what tells the two kinds of error apart.
    """
    scenarios = perfect_results()
    scenarios["absolute"] = {**scenarios["absolute"], field: value}
    gate = next(g for g in an.evaluate(scenarios) if g.id == "M30.absolute")

    assert gate.verdict == "FAIL"
    assert sum(c.verdict == "FAIL" for c in gate.checks) == 1
