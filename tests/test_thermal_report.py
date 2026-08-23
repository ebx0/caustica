"""M18 gate: the dose/threshold report — what it prints and what it refuses.

The report computes no physics, so every test here is about the contract:

* the medical liability note is in BOTH files, verbatim, and there is no way
  to write a report without it;
* every threshold row carries PASS or EXCEEDED, never a blank;
* the volume above a threshold is a number a reader can check by hand
  (voxel count x dx^3), and is honestly absent when the grid is not 3-D;
* the peak is read off the WHOLE history, not the final state — the same
  trap ``test_thermal.py`` pins for the dose;
* a per-tissue table appears exactly when the medium has an id map AND the
  caller named the tissues, and an unlisted tissue is NOT GRADED rather than
  graded against somebody else's limit;
* a result with no dose map is refused instead of rendered with empty rows.

The results are synthetic (built field by field) on purpose: a report test
that had to run a Pennes solve would be slow and would grade the solver a
second time. The real chain is ``tests/test_thermal_e2e.py``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from caustica.materials import Material, MaterialDB
from caustica.thermal.dose import ITRUSST_CEM43_LIMITS, ITRUSST_DELTA_T_LIMIT_C, MEDICAL_DISCLAIMER
from caustica.thermal.pennes import ARTERIAL_TEMPERATURE_C, ThermalResult
from caustica.thermal.properties import ThermalMedium
from caustica.thermal.report import (
    FORMAT,
    JSON_NAME,
    MD_NAME,
    MEDICAL_LIABILITY_NOTE,
    VERDICT_EXCEEDED,
    VERDICT_NOT_GRADED,
    VERDICT_PASS,
    itrusst_class,
    labels_from_db,
    thermal_payload,
    write_payload,
    write_thermal_report,
)

DX = 1.0e-3
SHAPE = (6, 6, 6)


def _material(name: str, perfusion: float = 0.005) -> Material:
    return Material(
        name=name,
        alpha_np_m=6.0,
        rho=1050.0,
        c=1540.0,
        beta=4.5,
        thermal_conductivity=0.5,
        specific_heat=3600.0,
        perfusion_rate=perfusion,
    )


BRAIN = _material("Brain grey matter")
BONE = _material("Cortical bone")
FAT = _material("Fat")
DB = MaterialDB(materials={1: BRAIN, 2: BONE, 3: FAT})


def _medium(shape=SHAPE, *, layered: bool = False) -> ThermalMedium:
    if not layered:
        return ThermalMedium.homogeneous(shape, BRAIN, DX)
    ids = np.ones(shape, dtype=np.int32)
    ids[..., 2:4] = 2
    ids[..., 4:] = 3
    return ThermalMedium.from_id_map(ids, DB, DX)


def _result(
    *,
    peak_c: float = 55.0,
    final_c: float = 37.2,
    peak_dose: float = 40.0,
    n_hot: int = 3,
    shape=SHAPE,
) -> ThermalResult:
    """A synthetic solve: ``n_hot`` voxels got hot and dosed, the rest did not.

    ``final`` is deliberately COLD while ``temperature_max`` is hot: that is
    the shape of every real sonication once it has cooled, and a report that
    read the endpoint would call it harmless.
    """
    t_max = np.full(shape, ARTERIAL_TEMPERATURE_C, dtype=np.float32)
    dose = np.zeros(shape, dtype=np.float32)
    flat_max, flat_dose = t_max.reshape(-1), dose.reshape(-1)
    flat_max[:n_hot] = peak_c
    flat_dose[:n_hot] = peak_dose
    return ThermalResult(
        temperature=np.full(shape, final_c, dtype=np.float32),
        temperature_max=t_max,
        dose_cem43=dose,
        samples=[],
        times=[],
        dt=0.5,
        n_steps=100,
        t_end_s=50.0,
        substeps=1,
        meta={
            "scheme": "pennes-fd-explicit/1",
            "backend": "numpy",
            "boundary": "insulated",
            "dt_stable_s": 0.6,
            "q": "HeatingSource(f0_only)",
            "perfusion_active": True,
            "arterial_temperature_c": ARTERIAL_TEMPERATURE_C,
        },
    )


# --------------------------------------------------------------------------
# The liability note
# --------------------------------------------------------------------------


def test_the_liability_note_is_in_both_files_verbatim(tmp_path):
    """The one thing this module will not let a caller leave out."""
    outdir = write_thermal_report(_result(), _medium(), tmp_path / "r")
    md = (outdir / MD_NAME).read_text(encoding="utf-8")
    payload = json.loads((outdir / JSON_NAME).read_text(encoding="utf-8"))
    assert MEDICAL_LIABILITY_NOTE in md
    assert payload["medical_liability_note"] == MEDICAL_LIABILITY_NOTE


def test_the_note_says_research_only_not_clinical_and_makes_no_medical_claim():
    """The three statements M18 requires, and one definition of the first."""
    assert MEDICAL_LIABILITY_NOTE.startswith(MEDICAL_DISCLAIMER), (
        "the research-use sentence must be caustica.thermal.dose's, not a second copy"
    )
    lowered = MEDICAL_LIABILITY_NOTE.lower()
    assert "research use only" in lowered
    assert "not a clinical decision tool" in lowered
    assert "no number in this report is a medical claim" in lowered


def test_a_payload_with_the_note_edited_out_is_refused(tmp_path):
    """Not a formatting option: removing it fails the write, loudly."""
    payload = dict(thermal_payload(_result(), _medium()))
    payload["medical_liability_note"] = "(omitted for brevity)"
    with pytest.raises(ValueError, match="not an optional field"):
        write_payload(payload, tmp_path / "r")
    assert not (tmp_path / "r" / MD_NAME).exists(), "a refused report must leave no files"


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------


def test_every_threshold_row_carries_a_verdict_and_the_itrusst_limits():
    payload = thermal_payload(_result(), _medium())
    rows = {row["name"]: row for row in payload["thresholds"]}
    assert set(rows) == {f"CEM43 ({t})" for t in ITRUSST_CEM43_LIMITS} | {
        "temperature rise (non-thermal line)"
    }
    for tissue, limit in ITRUSST_CEM43_LIMITS.items():
        assert rows[f"CEM43 ({tissue})"]["limit"] == limit
    assert rows["temperature rise (non-thermal line)"]["limit"] == ITRUSST_DELTA_T_LIMIT_C
    for row in payload["thresholds"]:
        assert row["verdict"] in (VERDICT_PASS, VERDICT_EXCEEDED), "a blank row reads as a pass"


def test_a_cold_run_passes_every_threshold_and_a_hot_one_exceeds_them():
    cold = thermal_payload(_result(peak_c=38.0, peak_dose=0.05), _medium())
    assert {row["verdict"] for row in cold["thresholds"]} == {VERDICT_PASS}
    assert cold["verdict"] == VERDICT_PASS

    hot = thermal_payload(_result(peak_c=55.0, peak_dose=40.0), _medium())
    assert {row["verdict"] for row in hot["thresholds"]} == {VERDICT_EXCEEDED}
    assert hot["verdict"] == VERDICT_EXCEEDED


def test_the_volume_above_a_threshold_is_the_voxel_count_times_the_voxel_volume():
    """A number the reader can check by hand: 3 voxels of 1 mm -> 3 mm^3."""
    payload = thermal_payload(_result(n_hot=3, peak_dose=40.0), _medium())
    brain = next(r for r in payload["thresholds"] if r["name"] == "CEM43 (brain)")
    assert brain["n_voxels"] == 3
    assert brain["volume_mm3"] == pytest.approx(3 * (DX * 1e3) ** 3)
    assert brain["fraction"] == pytest.approx(3 / np.prod(SHAPE))
    assert payload["run"]["voxel_volume_mm3"] == pytest.approx(1.0)


def test_a_dose_between_two_limits_passes_one_and_exceeds_the_other():
    """8 CEM43 is over brain (2) and under bone (16) — the table must say both."""
    payload = thermal_payload(_result(peak_dose=8.0), _medium())
    rows = {row["name"]: row["verdict"] for row in payload["thresholds"]}
    assert rows["CEM43 (brain)"] == VERDICT_EXCEEDED
    assert rows["CEM43 (bone)"] == VERDICT_PASS
    assert rows["CEM43 (skin)"] == VERDICT_PASS


def test_a_two_dimensional_grid_reports_voxels_and_no_volume(tmp_path):
    """A voxel count is only a volume in 3-D, and the report says which."""
    shape = (8, 8)
    payload = thermal_payload(_result(shape=shape), _medium(shape))
    assert payload["run"]["voxel_volume_mm3"] is None
    assert all(row["volume_mm3"] is None for row in payload["thresholds"])
    outdir = write_payload(payload, tmp_path / "r")
    assert "not 3-D: no volume" in (outdir / MD_NAME).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Temperature: the history, not the endpoint
# --------------------------------------------------------------------------


def test_the_peak_is_read_off_the_history_not_the_final_temperature():
    """The cooled-down trap: 55 C happened, and 37.2 C at the end hides it."""
    payload = thermal_payload(_result(peak_c=55.0, final_c=37.2), _medium())
    temp = payload["temperature"]
    assert temp["peak_c"] == pytest.approx(55.0)
    assert temp["peak_delta_t_c"] == pytest.approx(55.0 - ARTERIAL_TEMPERATURE_C)
    assert temp["final_peak_c"] == pytest.approx(37.2)
    assert temp["peak_is_over_history"] is True


def test_the_baseline_can_be_overridden_for_a_phantom_that_starts_cold():
    """dT is stated against a baseline; a 20 C gel is not a 37 C body."""
    payload = thermal_payload(_result(peak_c=40.0), _medium(), baseline_temperature_c=20.0)
    assert payload["temperature"]["baseline_c"] == pytest.approx(20.0)
    assert payload["temperature"]["peak_delta_t_c"] == pytest.approx(20.0)


# --------------------------------------------------------------------------
# The per-tissue table
# --------------------------------------------------------------------------


def test_the_tissue_class_is_matched_from_the_label_and_ambiguity_is_refused():
    assert itrusst_class("Cortical bone") == "bone"
    assert itrusst_class("Brain grey matter") == "brain"
    assert itrusst_class("Skin") == "skin"
    assert itrusst_class("Fat") is None
    assert itrusst_class(None) is None
    # Two classes with different limits in one name: choosing either invents
    # a limit the label does not support.
    assert itrusst_class("skin over bone") is None


def test_labels_come_from_the_material_db_so_the_names_cannot_drift():
    assert labels_from_db(DB) == {1: "Brain grey matter", 2: "Cortical bone", 3: "Fat"}
    assert labels_from_db(DB, ids=[2]) == {2: "Cortical bone"}


def test_each_tissue_is_graded_against_its_own_limit():
    """8 CEM43 in bone is fine; the same dose in brain is not."""
    med = _medium(layered=True)
    # Put the dose in the bone slab only (voxels ..., 2:4).
    t_max = np.full(SHAPE, ARTERIAL_TEMPERATURE_C, dtype=np.float32)
    dose = np.zeros(SHAPE, dtype=np.float32)
    t_max[..., 2:4] = 50.0
    dose[..., 2:4] = 8.0
    res = _result()
    res.temperature_max, res.dose_cem43 = t_max, dose

    payload = thermal_payload(res, med, tissue_labels=labels_from_db(DB))
    by_label = {t["label"]: t for t in payload["tissues"]}
    assert by_label["Cortical bone"]["itrusst_class"] == "bone"
    assert by_label["Cortical bone"]["peak_cem43"] == pytest.approx(8.0)
    assert by_label["Cortical bone"]["verdict"] == VERDICT_PASS, "8 < 16 CEM43 for bone"
    assert by_label["Brain grey matter"]["verdict"] == VERDICT_PASS, "no dose reached the brain"
    assert by_label["Fat"]["verdict"] == VERDICT_NOT_GRADED, "ITRUSST publishes no fat limit"
    assert payload["verdict_basis"].startswith("per-tissue")


def test_an_unlisted_tissue_can_be_given_a_class_explicitly():
    """The heuristic is overridable — and the override wins even when it is None."""
    med = _medium(layered=True)
    labels = labels_from_db(DB)
    payload = thermal_payload(
        _result(), med, tissue_labels=labels, tissue_classes={3: "skin", 1: None}
    )
    by_id = {t["id"]: t for t in payload["tissues"]}
    assert by_id[3]["itrusst_class"] == "skin"
    assert by_id[3]["limit_cem43"] == ITRUSST_CEM43_LIMITS["skin"]
    assert by_id[1]["itrusst_class"] is None
    assert by_id[1]["verdict"] == VERDICT_NOT_GRADED


def test_without_labels_the_report_says_so_and_falls_back_to_the_strictest_limit():
    payload = thermal_payload(_result(peak_dose=8.0), _medium(layered=True))
    assert payload["tissues"] == []
    assert any("no tissue_labels" in note for note in payload["notes"])
    assert "STRICTEST" in payload["verdict_basis"]
    # 8 CEM43 is over the brain limit, which is the strictest of the three.
    assert payload["verdict_dose"] == VERDICT_EXCEEDED


def test_a_medium_with_no_id_map_says_why_there_is_no_per_tissue_table():
    payload = thermal_payload(_result(), _medium())
    assert payload["tissues"] == []
    assert any("no id_map" in note for note in payload["notes"])


# --------------------------------------------------------------------------
# Refusals and provenance
# --------------------------------------------------------------------------


def test_a_result_that_never_accumulated_a_dose_is_refused():
    """A dose report with no dose would be a table of blanks — and blanks pass."""
    res = _result()
    res.dose_cem43 = None
    with pytest.raises(ValueError, match="dose=True"):
        thermal_payload(res, _medium())


def test_a_result_and_a_medium_of_different_shapes_are_refused():
    with pytest.raises(ValueError, match="wrong voxel"):
        thermal_payload(_result(shape=(4, 4, 4)), _medium(SHAPE))


def test_the_report_carries_the_environment_and_git_stamp(tmp_path):
    """Provenance, the same composition every other caustica report stamps."""
    outdir = write_thermal_report(_result(), _medium(), tmp_path / "r", label="stamped")
    payload = json.loads((outdir / JSON_NAME).read_text(encoding="utf-8"))
    assert payload["format"] == FORMAT == "caustica-thermal/1"
    assert payload["label"] == "stamped"
    for key in ("generated", "caustica", "git_commit", "host"):
        assert payload[key], f"missing stamp field {key}"
    env = payload["environment"]
    for key in ("python", "numpy", "platform", "resolved_backend"):
        assert key in env
    md = (outdir / MD_NAME).read_text(encoding="utf-8")
    assert str(payload["caustica"]) in md
    assert "caustica-thermal/1" in md


def test_rewriting_the_report_replaces_it_in_place(tmp_path):
    """Same folder, same two files — and no ``.tmp`` debris left behind."""
    outdir = tmp_path / "r"
    write_thermal_report(_result(peak_dose=0.01), _medium(), outdir, label="first")
    write_thermal_report(_result(peak_dose=40.0), _medium(), outdir, label="second")
    payload = json.loads((outdir / JSON_NAME).read_text(encoding="utf-8"))
    assert payload["label"] == "second"
    assert payload["verdict"] == VERDICT_EXCEEDED
    assert sorted(p.name for p in outdir.iterdir()) == sorted([JSON_NAME, MD_NAME])
