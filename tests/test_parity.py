"""The parity harness: three states, one registry, one runner.

The comparison contract's third rule is the one a test has to enforce,
because it is the one prose cannot: a metric is a value, or "not applicable"
with the reason, or "not computed" with the reason, and a blank, a dash or a
zero standing for any of the three is a defect. Everything here exists to
make that structural rather than aspirational.

Most of these tests run against a synthetic mirror: two arrays and a scene,
no solve. The two that run real engines carry the ``kwave`` marker, because
the reference on an acoustic parity page is the k-Wave binary and a machine
without it should skip rather than fail.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from collections.abc import Mapping

import numpy as np
import pytest

from caustica import Grid, Medium, PMLSpec
from caustica.materials import water
from caustica.parity import (
    NOT_APPLICABLE,
    NOT_COMPUTED,
    PARITY_CFL,
    STATES,
    VALUE,
    AcousticScene,
    EngineRun,
    ExtraEngine,
    MetricOutcome,
    ParityExample,
    ParityMirror,
    Setting,
    absent,
    check_complete,
    holds,
    ids,
    parity_run_spec,
    run_example,
    screen_solvers,
)
from caustica.parity import registry as parity_registry
from caustica.parity.mirrors import MIRRORS
from caustica.parity.mirrors import get as get_mirror
from caustica.parity.model import PLACEHOLDERS, TRAITS
from caustica.parity.scenes import arc_cw_source
from caustica.sources import CWSource

F0 = 1.0e6
DX = 0.5e-3


# --------------------------------------------------------------------------
# A synthetic mirror: the harness without a solve
# --------------------------------------------------------------------------


def _traits(**overrides: object) -> dict[str, object]:
    """Every trait declared, holding, except the ones a test switches off."""
    out = {name: holds() for name in TRAITS}
    out.update(overrides)  # type: ignore[arg-type]
    return out


def _gaussian(shape: tuple[int, int], sigma: float, scale: float = 1.0) -> np.ndarray:
    ax = [np.arange(n) - (n - 1) / 2.0 for n in shape]
    r2 = ax[0][:, None] ** 2 + ax[1][None, :] ** 2
    return scale * np.exp(-r2 / (2.0 * sigma**2))


class SyntheticMirror(ParityMirror):
    """One example, two engines, no physics: fields handed in as arrays.

    ``scene_ndim`` is ``None`` by default, so every registered solver is
    refused and exactly one engine is graded. Give it 1 or 2 to exercise the
    capability screening instead.
    """

    name = "example_synthetic"

    def __init__(
        self, *, scale: float = 1.02, scene_ndim: int | None = None, **run_fields: object
    ) -> None:
        self.scale = scale
        self.scene_ndim = scene_ndim
        self.run_fields = run_fields
        self._scene: AcousticScene | None = None

    def example(self) -> ParityExample:
        return ParityExample(
            name=self.name,
            title="A synthetic example",
            category="Test Fixtures",
            demonstrates="two Gaussian fields that differ by a stated scale factor",
            comparison_axis="the field",
            source_url="http://www.k-wave.org/documentation/",
            faithful_mirror=True,
            mirror_note="a fixture, not a page",
            graded_quantity="pressure amplitude",
            graded_units="Pa",
            traits=_traits(
                nonlinear=absent("the fixture medium is linear, so no harmonic exists."),
                third_harmonic=absent("the fixture medium is linear, so 3f0 does not exist."),
                shock=absent("the fixture waveform is a fixed Gaussian and never steepens."),
                varies_resolution=absent("the fixture runs one grid and sweeps nothing."),
            ),
        )

    def settings(self) -> Mapping[str, Setting]:
        return {
            "scale": Setting(self.scale, "caustica", "the fixture's own difference factor"),
        }

    def scene(self) -> AcousticScene | None:
        if self.scene_ndim is None:
            return None
        if self._scene is None:
            shape = (64, 64) if self.scene_ndim == 2 else (64,)
            grid = Grid(shape, DX, pml=PMLSpec(thickness=8 * DX))
            centre = tuple(n // 2 for n in shape)
            source = CWSource(
                indices=np.array([centre], dtype=np.int64),
                phases=np.zeros(1, np.float32),
                amplitude=1.0e5,
                f0=F0,
                label="one voxel",
            )
            self._scene = AcousticScene(
                grid=grid,
                medium=Medium.homogeneous(grid.shape, water()),
                source=source,
                spec=parity_run_spec(),
            )
        return self._scene

    def no_scene_reason(self) -> str:
        return (
            "this fixture has no acoustic scene: there is no source, no drive frequency "
            "and no wave for a solver to integrate."
        )

    def reference_engine(self) -> str:
        return "reference"

    def extra_engines(self) -> tuple[ExtraEngine, ...]:
        return (
            ExtraEngine("candidate", "solver", "native"),
            ExtraEngine("reference", "solver", "reference"),
        )

    def run_engine(self, name: str) -> EngineRun:
        scale = self.scale if name == "candidate" else 1.0
        return EngineRun(
            engine=name,
            kind="solver",
            role="reference" if name == "reference" else "native",
            field=_gaussian((64, 64), 8.0, scale),
            dx=DX,
            beam_axis=0,
            source_surface_pa=1.0e5,
            absorbed_power_w=1.0 * scale,
            provenance={"spp": 20, "dt_s": 5e-8, "cfl_realized": PARITY_CFL},
            **self.run_fields,  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------
# The registry is the specification, made executable
# --------------------------------------------------------------------------


def test_the_registry_holds_exactly_the_thirty_metrics_the_contract_defines():
    check_complete()
    assert ids() == tuple(f"M-{n:02d}" for n in range(1, 31))


def test_a_page_cannot_name_a_metric_the_contract_does_not_define():
    with pytest.raises(KeyError) as exc:
        parity_registry.get("M-31")
    assert "M-01" in str(exc.value), "the lookup failure must name the whole registry"


def test_every_metric_declares_a_group_the_contract_lists():
    for m in parity_registry.all_metrics():
        assert m.group in parity_registry.GROUPS
        assert m.applies_when, f"{m.id} must say when it applies"


# --------------------------------------------------------------------------
# Three states, never two: absence cannot be spelled as a value
# --------------------------------------------------------------------------


@pytest.mark.parametrize("placeholder", PLACEHOLDERS)
def test_a_placeholder_can_never_be_a_metric_value(placeholder):
    with pytest.raises(ValueError):
        MetricOutcome("M-01", VALUE, value=placeholder)


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, -math.inf, True, {}, []])
def test_a_metric_value_must_be_a_finite_number(bad):
    with pytest.raises(ValueError):
        MetricOutcome("M-01", VALUE, value=bad)


@pytest.mark.parametrize("empty", [{}, [], ()])
def test_an_empty_container_is_an_absence_and_is_refused_as_a_value(empty):
    """A mapping with no entries is 'nothing to measure' wearing a value's shape."""
    with pytest.raises(ValueError, match="absence wearing the shape of a value"):
        MetricOutcome("M-11", VALUE, value=empty)


def test_a_computed_value_carries_no_reason_and_an_absence_carries_nothing_else():
    with pytest.raises(ValueError, match="carries no reason"):
        MetricOutcome("M-01", VALUE, value=1.0, reason="because")
    with pytest.raises(ValueError, match="carries no value"):
        MetricOutcome("M-01", NOT_APPLICABLE, value=0.0, reason="a reason long enough")


@pytest.mark.parametrize("state", [NOT_APPLICABLE, NOT_COMPUTED])
@pytest.mark.parametrize("reason", ["", "-", "n/a", "  ", "too short"])
def test_an_absence_needs_a_reason_a_reader_can_act_on(state, reason):
    with pytest.raises(ValueError):
        MetricOutcome("M-01", state, reason=reason)


def test_a_trait_that_does_not_hold_must_say_why():
    with pytest.raises(ValueError, match="needs the reason"):
        absent("nope")


def test_every_trait_is_declared_explicitly_so_applicability_is_never_a_default():
    base = SyntheticMirror().example()
    partial = dict(base.traits)
    partial.pop("sidelobe")
    with pytest.raises(ValueError, match="undeclared"):
        dataclasses.replace(base, traits=partial)
    unknown = dict(base.traits)
    unknown["frequency_sweep"] = holds()
    with pytest.raises(ValueError, match="unknown trait"):
        dataclasses.replace(base, traits=unknown)


def test_a_setting_that_is_not_quoted_from_the_page_must_say_what_it_came_from():
    Setting(128, "k-Wave example page")
    with pytest.raises(ValueError, match="reconstructed"):
        Setting(128, "inferred")
    with pytest.raises(ValueError, match="source must be one of"):
        Setting(128, "guessed", "a note long enough to pass")


# --------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_document():
    return run_example(SyntheticMirror())


def test_no_metric_is_ever_emitted_as_a_blank_a_dash_or_a_zero_standing_for_absence(
    synthetic_document,
):
    """The contract's rule 3, enforced over every cell the runner emits."""
    seen = 0
    for engine in synthetic_document["engines"]:
        if "metrics" not in engine:
            assert engine["no_metrics_reason"].strip(), "a row without metrics states why"
            continue
        assert set(engine["metrics"]) == set(ids()), "a row carries every metric or none"
        for metric_id, cell in engine["metrics"].items():
            seen += 1
            assert cell["state"] in STATES, f"{metric_id}: {cell}"
            if cell["state"] == VALUE:
                assert "reason" not in cell
                assert _all_finite(cell["value"]), f"{metric_id} emitted {cell['value']!r}"
            else:
                assert "value" not in cell, f"{metric_id} carries a value beside an absence"
                reason = cell["reason"]
                assert reason.strip() not in PLACEHOLDERS
                assert len(reason.strip()) >= 10, f"{metric_id}: {reason!r}"
    assert seen == 30, "the synthetic page must have graded exactly one engine"


def _all_finite(value) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_all_finite(v) for v in value)
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def test_the_document_is_json_and_carries_the_scheme_the_cfl_and_the_load_regime(
    synthetic_document, tmp_path
):
    run_example(SyntheticMirror(), out_dir=tmp_path, load_regime="fixture, no other load")
    written = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert written["contract"]["cfl"] == PARITY_CFL
    assert written["contract"]["caustica_default_cfl"] != PARITY_CFL, (
        "the page has to be able to say the comparison CFL is not caustica's default"
    )
    assert written["provenance"]["numerics_scheme"].startswith("cw-kspace-pstd/")
    assert written["provenance"]["load_regime"] == "fixture, no other load"
    assert synthetic_document["provenance"]["load_regime"] is None
    assert {m["id"] for m in written["metric_registry"]} == set(ids())


def test_the_reference_row_carries_no_metrics_and_says_why(synthetic_document):
    ref = _engine(synthetic_document, "reference")
    assert "metrics" not in ref
    assert "grading it against itself" in ref["no_metrics_reason"]


def test_every_registered_solver_appears_with_its_own_refusal_text():
    """Contract rule 2: a solver that cannot express the example is listed."""
    doc = run_example(SyntheticMirror(scene_ndim=1))
    kwave = _engine(doc, "kwave")
    assert "supports [2, 3]-D grids, got 1-D" in kwave["refused"], (
        "the row must carry the capability check's own words, not a summary of them"
    )
    assert "metrics" in _engine(doc, "linear"), "a solver the caps accept is run, not refused"


def test_an_example_with_no_acoustic_scene_refuses_every_wave_solver_with_one_reason():
    doc = run_example(SyntheticMirror())
    for name in ("kwave", "linear", "westervelt"):
        assert "no acoustic scene" in _engine(doc, name)["refused"]
    assert "metrics" in _engine(doc, "candidate")


def test_screening_answers_none_for_a_solver_the_caps_accept():
    assert screen_solvers(SyntheticMirror(scene_ndim=2)) == {
        "kwave": None,
        "linear": None,
        "westervelt": None,
    }


def test_the_run_spec_pins_the_contract_cfl_and_refuses_another():
    assert parity_run_spec().cfl == PARITY_CFL
    with pytest.raises(ValueError, match="pinned"):
        parity_run_spec(cfl=0.48)


def test_a_scene_at_another_cfl_is_refused_before_anything_is_computed():
    from caustica.solvers import CWRunSpec

    mirror = SyntheticMirror(scene_ndim=2)
    scene = mirror.scene()
    with pytest.raises(ValueError, match="parity scene runs at cfl"):
        AcousticScene(scene.grid, scene.medium, scene.source, CWRunSpec())


def test_the_cost_metrics_stay_absent_without_a_load_regime_and_report_with_one():
    """Made structural: a timing with no load regime is not a number."""
    fields = {"wall_time_s": 2.0, "peak_device_bytes": 4096}
    without = run_example(SyntheticMirror(**fields))
    with_regime = run_example(
        SyntheticMirror(**fields), load_regime="laptop, exclusive, nothing else running"
    )
    assert _cell(without, "candidate", "M-29")["state"] == NOT_COMPUTED
    assert "load regime" in _cell(without, "candidate", "M-29")["reason"]
    assert _cell(with_regime, "candidate", "M-29") == {"state": VALUE, "value": 1.0}
    assert _cell(with_regime, "candidate", "M-30") == {"state": VALUE, "value": 1.0}


def test_mutating_a_metric_function_moves_that_number_on_the_page(monkeypatch):
    """The page is generated, not written: change the metric, the page moves."""
    before = _cell(run_example(SyntheticMirror()), "candidate", "M-07")
    assert before["state"] == VALUE
    monkeypatch.setitem(
        parity_registry._METRICS,
        "M-07",
        dataclasses.replace(parity_registry.get("M-07"), compute=lambda case: 0.5),
    )
    after = _cell(run_example(SyntheticMirror()), "candidate", "M-07")
    assert after == {"state": VALUE, "value": 0.5}
    assert after["value"] != pytest.approx(before["value"]), (
        "the mutated metric must move the number the page prints"
    )


def test_a_metric_that_stops_being_computable_becomes_not_computed_not_a_zero(monkeypatch):
    from caustica.parity.model import NotComputed

    def refuse(case):
        raise NotComputed("the fixture withheld the field this metric needs.")

    monkeypatch.setitem(
        parity_registry._METRICS,
        "M-07",
        dataclasses.replace(parity_registry.get("M-07"), compute=refuse),
    )
    cell = _cell(run_example(SyntheticMirror()), "candidate", "M-07")
    assert cell["state"] == NOT_COMPUTED
    assert "withheld" in cell["reason"]


def test_a_mirror_that_builds_an_inconsistent_scene_is_a_defect_not_a_refusal():
    """A setup bug must not be published as a statement about the capability matrix."""

    class BrokenSceneMirror(SyntheticMirror):
        def scene(self) -> AcousticScene:
            good = SyntheticMirror(scene_ndim=2).scene()
            return AcousticScene(
                grid=good.grid,
                medium=Medium.homogeneous((32, 32), water()),
                source=good.source,
                spec=parity_run_spec(),
            )

    with pytest.raises(ValueError, match="medium shape"):
        screen_solvers(BrokenSceneMirror())


def test_the_document_records_the_backend_the_mirror_declares_and_asks_for_none():
    """A page cannot name a backend nothing ran on, because nothing requests one."""
    doc = run_example(SyntheticMirror())
    assert doc["provenance"]["mirror_backend"] is None, (
        "a mirror that declares no backend must not have one stamped for it"
    )
    with pytest.raises(TypeError):
        run_example(SyntheticMirror(), backend="cupy")  # type: ignore[call-arg]


# --------------------------------------------------------------------------
# The metric functions themselves, not just the plumbing
# --------------------------------------------------------------------------


def _case_1d(**run_fields):
    """One graded pair of 1-D fields, for the metrics that need a plane."""
    from caustica.parity import ParityCase

    line = np.exp(-((np.arange(64.0) - 32.0) ** 2) / 50.0)
    common = dict(kind="solver", field=line, beam_axis=0, **run_fields)
    run = EngineRun(engine="candidate", role="native", **common)
    reference = EngineRun(engine="reference", role="reference", **common)
    return ParityCase(SyntheticMirror().example(), run, reference)


@pytest.mark.parametrize("metric_id", ["M-11", "M-14"])
def test_a_field_with_no_transverse_axis_states_it_instead_of_raising(metric_id):
    """A 1-D example is all beam axis; the lateral metrics say so."""
    outcome = parity_registry.evaluate(metric_id, _case_1d(dx=DX))
    assert outcome.state == NOT_APPLICABLE
    assert "no transverse direction" in outcome.reason


def test_a_run_without_a_voxel_pitch_reports_no_length_in_metres():
    """``dx`` defaults to zero, and a zero millimetre offset is a missing input."""
    outcome = parity_registry.evaluate("M-08", _case_1d())
    assert outcome.state == NOT_COMPUTED
    assert "no voxel pitch" in outcome.reason
    assert parity_registry.evaluate("M-08", _case_1d(dx=DX)).state == VALUE


def test_the_field_metrics_are_the_numbers_an_independent_computation_gives():
    """The metric functions, not the plumbing: every value recomputed by hand.

    The diffusion page is the cheap one (no k-Wave, no acoustic solve), and
    its two fields are available straight from the mirror, so the harness's
    M-01 to M-04 and M-07 can be checked against numpy on the same arrays.
    """
    mirror = get_mirror("example_diff_homogeneous_medium_diffusion")
    doc = run_example(mirror)
    a = np.asarray(mirror.run_engine("pennes").field, dtype=np.float64)
    b = np.asarray(mirror.run_engine("pennes-exact").field, dtype=np.float64)
    cells = _engine(doc, "pennes")["metrics"]

    peak = float(np.abs(b).max())
    expected = {
        "M-01": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
        "M-02": float(np.linalg.norm((a - b).ravel()) / np.linalg.norm(b.ravel())),
        "M-03": float(np.abs(a - b).max() / peak),
        "M-04": float(np.sqrt(np.mean((a - b) ** 2)) / peak),
        "M-07": float(a.max() / b.max()),
    }
    for metric_id, want in expected.items():
        assert cells[metric_id]["value"] == pytest.approx(want, rel=1e-12), metric_id
    assert expected["M-01"] < 1.0, "a constant 1.0 would pass a floor; this pins the number"
    assert cells["M-07"]["value"] != pytest.approx(float(b.max() / a.max()), rel=1e-9), (
        "M-07 is the run over the reference; the inverse would flip every parity claim"
    )


# --------------------------------------------------------------------------
# The mirrors
# --------------------------------------------------------------------------


def test_every_mirror_is_constructible_and_names_itself_as_the_enumeration_does():
    for name in MIRRORS:
        mirror = get_mirror(name)
        record = mirror.example()
        assert record.name == name == mirror.name
        assert record.category, f"{name} must carry its k-Wave category"
        assert record.graded_quantity and record.graded_units
        for key, setting in mirror.settings().items():
            assert isinstance(setting, Setting), f"{name}.{key}"


def test_an_example_with_no_mirror_is_a_task_not_a_lookup_failure():
    with pytest.raises(KeyError, match="work list"):
        get_mirror("example_ivp_homogeneous_medium")


def test_the_mirrors_span_three_of_k_waves_ten_categories():
    categories = {get_mirror(name).example().category for name in MIRRORS}
    assert len(categories) >= 3, categories


def test_the_thermal_diffusion_page_grades_itself_against_its_own_closed_form():
    """The whole page, end to end: it needs no k-Wave and no acoustic solve."""
    doc = run_example(get_mirror("example_diff_homogeneous_medium_diffusion"))
    assert doc["reference_engine"] == "pennes-exact"
    assert "kWaveDiffusion" in _engine(doc, "kwave-diffusion")["refused"]
    pennes = _engine(doc, "pennes")
    assert pennes["metrics"]["M-01"]["value"] > 0.999
    assert pennes["metrics"]["M-03"]["value"] < 0.01, (
        "the Pennes solve must match the closed form to well under a percent"
    )
    assert pennes["metrics"]["M-27"]["state"] == NOT_APPLICABLE
    assert pennes["metrics"]["M-21"]["state"] == NOT_COMPUTED, (
        "a waveform this run did not record is 'not computed', never 'not applicable'"
    )


#: The engine each page is graded on, and how many of the thirty metrics that
#: engine answers with a number. The counts are the acceptance criterion made
#: countable: four pages, four different subsets, none of them equal.
GRADED = {
    "example_diff_homogeneous_medium_diffusion": ("pennes", 11),
    "example_diff_focused_ultrasound_heating": ("linear", 15),
    "example_tvsp_transducer_field_patterns": ("linear", 10),
    "example_cpp_running_simulations": ("linear", 14),
}


@pytest.fixture(scope="module")
def every_page():
    """All four pages, run once: the expensive fixture the gate tests share."""
    pytest.importorskip("kwave", reason="k-wave-python not installed")
    docs = {}
    for name in MIRRORS:
        try:
            docs[name] = run_example(get_mirror(name))
        except (RuntimeError, FileNotFoundError, OSError) as exc:
            pytest.skip(f"k-Wave binary unavailable on this machine: {exc}")
    return docs


@pytest.mark.kwave
@pytest.mark.slow
@pytest.mark.parametrize("name", list(GRADED))
def test_every_page_meets_the_parity_gate_on_both_of_its_halves(every_page, name):
    """The gate this task is graded by: whole-field r > 0.99, peak within 3 %."""
    engine_name, expected_values = GRADED[name]
    engine = _engine(every_page[name], engine_name)
    r = engine["metrics"]["M-01"]["value"]
    peak = engine["metrics"]["M-07"]["value"]
    assert r > 0.99, f"{name}: whole-field r is {r}"
    assert abs(peak - 1.0) <= 0.03, f"{name}: peak ratio is {peak}"
    counts = engine["state_counts"]
    assert sum(counts.values()) == 30, f"{name}: every metric is in exactly one state"
    assert counts[VALUE] == expected_values, f"{name}: {counts}"


@pytest.mark.kwave
@pytest.mark.slow
def test_no_two_pages_apply_the_same_metric_subset_and_every_difference_is_reasoned(every_page):
    """Acceptance 1, over all four pages rather than a pair of them."""
    subsets = {
        name: frozenset(_engine(every_page[name], engine)["applicable"])
        for name, (engine, _) in GRADED.items()
    }
    assert len(set(subsets.values())) == len(subsets), (
        f"two pages apply the same metrics: {subsets}"
    )
    for left, right in itertools.combinations(GRADED, 2):
        for metric_id in subsets[left] ^ subsets[right]:
            page = left if metric_id in subsets[right] else right
            cell = _cell(every_page[page], GRADED[page][0], metric_id)
            assert cell["state"] in (NOT_APPLICABLE, NOT_COMPUTED)
            assert len(cell["reason"].strip()) >= 10, f"{page}/{metric_id} differs unreasoned"


@pytest.mark.kwave
@pytest.mark.slow
def test_an_acoustic_page_runs_every_native_solver_against_the_kwave_reference(every_page):
    doc = every_page["example_tvsp_transducer_field_patterns"]
    assert doc["reference_engine"] == "kwave"
    for name in ("linear", "westervelt"):
        engine = _engine(doc, name)
        assert engine["metrics"]["M-01"]["value"] > 0.99
        assert engine["step"]["cfl_realized"] == pytest.approx(PARITY_CFL, abs=0.02)
        assert engine["metrics"]["M-13"]["state"] == NOT_APPLICABLE, (
            "the arc's axis is diagonal, so the profile group states why it cannot report"
        )


@pytest.mark.kwave
@pytest.mark.slow
def test_the_flagship_heating_page_carries_its_dose_chain_beside_the_metrics(every_page):
    """The thermal scalars the registry has no group for still reach the page."""
    scalars = _engine(every_page["example_diff_focused_ultrasound_heating"], "linear")["scalars"]
    for key in ("peak_pressure_pa", "peak_q_w_m3", "peak_rise_k", "peak_cem43_min"):
        assert scalars[key] > 0.0, key


# --------------------------------------------------------------------------
# The 2-D arc source the mirrors need
# --------------------------------------------------------------------------


def test_the_arc_source_deposits_its_own_length_not_its_voxel_count():
    grid = Grid((128, 128), DX, pml=PMLSpec(thickness=10 * DX))
    roc, aperture = 40.0, 50.0
    source = arc_cw_source(
        grid,
        f0=F0,
        amplitude=1.0,
        apex_vox=(30.0, 64.0),
        roc_vox=roc,
        aperture_vox=aperture,
        axis=(1.0, 0.0),
    )
    expected = 2.0 * roc * math.asin(0.5 * aperture / roc)
    assert float(source.drive_weights.sum()) == pytest.approx(expected, rel=1e-3)
    assert source.discretization == "offgrid"


def test_an_arc_wider_than_its_own_diameter_is_refused():
    grid = Grid((64, 64), DX)
    with pytest.raises(ValueError, match="at most the diameter"):
        arc_cw_source(
            grid,
            f0=F0,
            amplitude=1.0,
            apex_vox=(10.0, 32.0),
            roc_vox=10.0,
            aperture_vox=40.0,
            axis=(1.0, 0.0),
        )


# --------------------------------------------------------------------------


def _engine(doc: dict, name: str) -> dict:
    for engine in doc["engines"]:
        if engine["engine"] == name:
            return engine
    raise AssertionError(f"no engine '{name}' in {[e['engine'] for e in doc['engines']]}")


def _cell(doc: dict, engine: str, metric_id: str) -> dict:
    return _engine(doc, engine)["metrics"][metric_id]
