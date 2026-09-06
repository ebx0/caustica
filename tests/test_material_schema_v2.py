"""Material schema v2: the power-law absorption form, the IT'IS thermal
values, and the job section that names the exponent.

The legacy form (``alpha_np_m`` alone) is pinned by tests/test_materials.py
and must keep working unchanged; everything here is what was added next to
it.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from caustica.config.job import (
    JOB_FORMAT,
    AbsorptionConfig,
    JobError,
    build_job,
    job_schema,
    load_job,
    validate_job,
)
from caustica.materials import (
    DB_CM_TO_NP_M,
    ITIS_DB,
    TISSUE_LIBRARY,
    Material,
    MaterialDB,
    MixedAbsorptionExponentError,
    breast_default,
    breast_default_power_law,
    perfusion_ml_min_kg_to_per_s,
)

# --------------------------------------------------------------- materials

FAT_POWER_LAW = {
    "name": "power-law fat",
    "alpha0_db_cm_mhz_y": 0.5,
    "y": 1.1,
    "rho": 911.0,
    "c": 1440.0,
    "beta": 6.0,
}


def test_both_absorption_forms_round_trip_through_json():
    """Acceptance 1: a legacy material and a power-law material survive one
    JSON round trip inside the same MaterialDB, each keeping its own form."""
    db = MaterialDB(
        materials={
            0: Material(alpha_np_m=0.1, rho=1000.0, c=1500.0, beta=3.5, name="legacy gel"),
            1: Material(
                **FAT_POWER_LAW,
                thermal_conductivity=0.209,
                specific_heat=2348.333333,
                perfusion_rate=7.1e-4,
                source="test",
            ),
        }
    )
    again = MaterialDB.model_validate_json(db.model_dump_json())
    assert again == db
    assert again[0].absorption_model == "single_frequency"
    assert again[1].absorption_model == "power_law"
    assert (again[1].alpha0_db_cm_mhz_y, again[1].y) == (0.5, 1.1)
    assert again[1].thermal_conductivity == 0.209 and again[1].source == "test"


def test_a_material_declares_exactly_one_absorption_form():
    with pytest.raises(ValidationError, match="ONE form"):
        Material(alpha_np_m=1.0, alpha0_db_cm_mhz_y=0.5, y=1.1, rho=1000.0, c=1500.0, beta=0.0)
    with pytest.raises(ValidationError, match="BOTH alpha0_db_cm_mhz_y"):
        Material(alpha0_db_cm_mhz_y=0.5, rho=1000.0, c=1500.0, beta=0.0)
    with pytest.raises(ValidationError, match="BOTH alpha0_db_cm_mhz_y"):
        Material(y=1.1, rho=1000.0, c=1500.0, beta=0.0)
    with pytest.raises(ValidationError, match="must declare its absorption"):
        Material(rho=1000.0, c=1500.0, beta=0.0)


def test_power_law_alpha_and_the_baked_legacy_copy_agree():
    """alpha(f) = alpha0 f^y in dB/(cm MHz^y), converted once, at three
    frequencies; the baked copy carries the same number."""
    m = Material(**FAT_POWER_LAW)
    for f_mhz in (0.5, 1.0, 3.0):
        want = 0.5 * f_mhz**1.1 * DB_CM_TO_NP_M
        assert m.alpha_np_m_at(f_mhz * 1e6) == pytest.approx(want, rel=1e-12)
        baked = m.at_frequency(f_mhz * 1e6)
        assert baked.absorption_model == "single_frequency"
        assert baked.alpha_np_m == pytest.approx(want, rel=1e-12)
        assert (baked.rho, baked.c, baked.beta) == (m.rho, m.c, m.beta)


def test_the_legacy_form_is_frequency_independent_and_means_y_zero():
    legacy = Material(alpha_np_m=6.0, rho=932.0, c=1450.0, beta=4.5)
    assert legacy.alpha_np_m_at(2.5e6) == 6.0
    assert legacy.at_frequency(2.5e6) is legacy
    assert legacy.y_exponent == 0.0
    assert breast_default().y() == 0.0
    assert not breast_default().has_power_law


def test_material_db_y_returns_the_common_exponent_or_names_the_offenders():
    same = MaterialDB(
        materials={
            0: Material(alpha0_db_cm_mhz_y=0.0022, y=1.1, rho=1000.0, c=1500.0, beta=3.5),
            1: Material(**FAT_POWER_LAW),
        }
    )
    assert same.y() == 1.1
    mixed = MaterialDB(
        materials={
            0: Material(
                alpha0_db_cm_mhz_y=0.0022, y=2.0, rho=1000.0, c=1500.0, beta=3.5, name="Water"
            ),
            1: Material(**{**FAT_POWER_LAW, "name": "Fat"}),
        }
    )
    with pytest.raises(MixedAbsorptionExponentError) as exc:
        mixed.y()
    msg = str(exc.value)
    assert "Water" in msg and "Fat" in msg and "y = 2" in msg and "y = 1.1" in msg


def test_material_db_at_frequency_bakes_every_power_law():
    db = breast_default_power_law()
    assert db.has_power_law
    baked = db.at_frequency(1.5e6)
    assert not baked.has_power_law
    for tissue_id in db.ids:
        assert baked[tissue_id].alpha_np_m == pytest.approx(db[tissue_id].alpha_np_m_at(1.5e6))


def test_breast_default_power_law_is_the_library_and_refuses_one_exponent():
    db = breast_default_power_law()
    assert db.ids == breast_default().ids == (0, 1, 2, 3, 4)
    assert db[1].name == TISSUE_LIBRARY["skin"].name
    assert db[2].alpha0_db_cm_mhz_y == pytest.approx(0.5)  # fat midpoint of (0.40, 0.60)
    assert db[2].y == 1.1
    # water 2.0, skin and muscle 1.0, fat 1.1: the literature does not agree,
    # and the table says so instead of picking one silently.
    with pytest.raises(MixedAbsorptionExponentError):
        db.y()


def test_breast_default_legacy_numbers_are_untouched_by_the_new_form():
    """Acceptance 4, stated here too: the v1 table is bit-for-bit the same."""
    db = breast_default()
    assert [db[i].alpha_np_m for i in db.ids] == [0.1, 15.0, 6.0, 10.0, 0.1]
    assert all(db[i].absorption_model == "single_frequency" for i in db.ids)


# ---------------------------------------------------------- IT'IS thermal

#: Acceptance 3: the raw rows of the IT'IS V4.2 ASCII table
#: (Thermal_dielectric_acoustic_MR properties_database_V4.2), quoted here
#: independently of the library so a drifted library value fails.
#: (library key, database row, k [W/m/K], C [J/kg/K],
#:  heat transfer rate [mL/min/kg], database density [kg/m^3])
ITIS_V42_THERMAL = [
    ("water_37c", "Water", 0.6045, 4178.0, 0.0, 994.035466),
    ("skin", "Skin", 0.3721835, 3390.5, 106.3813131, 1109.0),
    ("fat", "Breast Fat", 0.209, 2348.333333, 47.0, 911.0),
    ("muscle", "Muscle", 0.49496875, 3421.2, 36.7382931, 1090.4),
    ("fibroglandular", "Breast Gland", 0.3345, 2960.0, 150.0, 1040.5),
    ("liver", "Liver", 0.519111111, 3540.2, 860.456666, 1078.75),
    ("brain", "Brain", 0.51325, 3630.0, 558.6063123, 1045.5),
    ("blood", "Blood", 0.516857143, 3617.0, 10000.0, 1049.75),
]


@pytest.mark.parametrize(("key", "row", "k", "cp", "htr", "rho"), ITIS_V42_THERMAL)
def test_itis_thermal_values_are_present_and_pinned(key, row, k, cp, htr, rho):
    t = TISSUE_LIBRARY[key]
    assert t.thermal_conductivity == k
    assert t.specific_heat == cp
    assert t.perfusion_rate == pytest.approx(perfusion_ml_min_kg_to_per_s(htr, rho), rel=1e-12)
    assert ITIS_DB in t.thermal_source
    assert f"row '{row}'" in t.thermal_source


def test_perfusion_conversion_is_ml_min_kg_to_per_second():
    """w_b [1/s] = rate [mL/(min kg)] * rho [kg/m^3] / (1e6 mL/m^3 * 60 s/min)."""
    assert perfusion_ml_min_kg_to_per_s(106.3813131, 1109.0) == pytest.approx(1.9663e-3, rel=1e-4)
    assert perfusion_ml_min_kg_to_per_s(860.456666, 1078.75) == pytest.approx(1.5470e-2, rel=1e-4)
    assert perfusion_ml_min_kg_to_per_s(0.0, 1000.0) == 0.0


def test_every_shipped_tissue_can_build_a_thermal_medium():
    """The point of the thermal values: no shipped tissue is missing a field."""
    from caustica.thermal.properties import ThermalMedium

    db = MaterialDB(
        materials={i: t.to_material(1e6) for i, t in enumerate(TISSUE_LIBRARY.values())}
    )
    ids = np.arange(len(db.ids), dtype=np.int64).reshape(-1, 1, 1) * np.ones(
        (1, 2, 2), dtype=np.int64
    )
    tm = ThermalMedium.from_id_map(ids, db, dx=1e-3)
    assert tm.shape == ids.shape
    assert float(tm.k.min()) > 0.0 and float(tm.specific_heat.min()) > 0.0
    assert tm.is_perfused


def test_a_tissue_material_carries_its_source_string():
    m = TISSUE_LIBRARY["liver"].to_power_law_material()
    assert ITIS_DB in m.source and "thermal:" in m.source
    assert m.thermal_conductivity == TISSUE_LIBRARY["liver"].thermal_conductivity


# ------------------------------------------------------- the job section


def _power_law_scene_job(materials: dict, absorption: dict | None = None, **over) -> dict:
    """A mini scene job whose two labels carry the given materials."""
    d = {
        "format": JOB_FORMAT,
        "kind": "explicit",
        "name": "power-law-mini",
        "medium": {
            "kind": "scene",
            "scene": {
                "ndim": 3,
                "background": 0,
                "objects": [
                    {"shape": {"kind": "ball", "center_mm": [9, 9, 14], "radius_mm": 4}, "label": 2}
                ],
            },
            "materials": materials,
        },
        "grid": {
            "ndim": 3,
            "dx_mm": 0.375,
            "size_mm": [18, 18, 24],
            "pml": {"thickness_mm": 2.25},
        },
        "source": {
            "kind": "array",
            "array": {"kind": "bowl", "d_outer_mm": 10.0, "roc_mm": 12.0},
            "apex_mm": [9, 9, 3.75],
        },
        "drive": {"f0_mhz": 1.0, "amplitude_kpa": 100.0},
        "run": {"spec": {"min_settle_periods": 3, "max_settle_periods": 8}, "harmonics": [1]},
        "solver": "linear",
    }
    if absorption is not None:
        d["absorption"] = absorption
    d.update(over)
    return d


WATER_PL = {
    "name": "Water",
    "alpha0_db_cm_mhz_y": 0.0015,
    "y": 1.1,
    "rho": 993.5,
    "c": 1522.5,
    # beta 0: these mini jobs run the linear solver, and a nonlinear medium
    # under it is a refusal of its own (tested elsewhere).
    "beta": 0.0,
}
FAT_PL = {
    "name": "Fat",
    "alpha0_db_cm_mhz_y": 0.5,
    "y": 1.1,
    "rho": 915.0,
    "c": 1442.5,
    "beta": 0.0,
}


def _write(tmp_path: Path, d: dict, name: str = "job.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_absorption_section_round_trips_and_reaches_the_schema(tmp_path):
    """Acceptance 1, job half: the section is in the generated schema and a
    job carrying it survives dump/load."""
    cfg = AbsorptionConfig(model="power_law", y=1.1)
    assert AbsorptionConfig.model_validate_json(cfg.model_dump_json()) == cfg
    assert AbsorptionConfig() == AbsorptionConfig(model="single_frequency", y=None)

    schema = job_schema()
    absorption = schema["$defs"]["AbsorptionConfig"]
    assert set(absorption["properties"]) == {"model", "y"}
    assert absorption["properties"]["model"]["enum"] == ["single_frequency", "power_law"]
    assert "absorption" in json.dumps(schema)

    # The exponent is bounded in the schema, not only in the model, so a
    # generated form or an editor refuses y = 7 before pydantic sees it.
    y_any = absorption["properties"]["y"]["anyOf"]
    number = next(b for b in y_any if b.get("type") == "number")
    assert (number["minimum"], number["maximum"]) == (0.0, 3.0)
    assert {"type": "null"} in y_any
    assert absorption["properties"]["y"]["default"] is None

    # Both absorption forms reach the material schema, and the property set is
    # pinned: dropping alpha0_db_cm_mhz_y from Material has to fail here.
    assert set(schema["$defs"]["Material"]["properties"]) == {
        "name",
        "alpha_np_m",
        "alpha0_db_cm_mhz_y",
        "y",
        "rho",
        "c",
        "beta",
        "thermal_conductivity",
        "specific_heat",
        "perfusion_rate",
        "source",
    }

    d = _power_law_scene_job({"0": WATER_PL, "2": FAT_PL}, {"model": "power_law", "y": None})
    job, base = load_job(_write(tmp_path, d))
    assert job.absorption.model == "power_law" and job.absorption.y is None
    assert load_job(_write(tmp_path, json.loads(job.model_dump_json()), "again.json"))[0] == job


def test_a_job_without_an_absorption_section_is_single_frequency(tmp_path):
    d = _power_law_scene_job(
        {
            "0": {"name": "water", "alpha_np_m": 0.0, "rho": 1000.0, "c": 1500.0, "beta": 0.0},
            "2": {"name": "fat", "alpha_np_m": 6.0, "rho": 932.0, "c": 1450.0, "beta": 0.0},
        }
    )
    job, base = load_job(_write(tmp_path, d))
    assert job.absorption.model == "single_frequency"
    built = build_job(job, base_dir=base)
    assert built.absorption_y is None
    assert built.absorption_from_power_law is False
    rep = validate_job(_write(tmp_path, d, "v.json"))
    assert rep.ok
    assert any("single frequency" in s for s in rep.summary)


def test_single_frequency_model_refuses_an_exponent():
    with pytest.raises(ValidationError, match="carries no exponent"):
        AbsorptionConfig(model="single_frequency", y=1.1)


def test_validate_refuses_mixed_y_and_names_the_offenders(tmp_path):
    """Acceptance 2: `caustica validate` refuses, naming every material."""
    water_y2 = {**WATER_PL, "y": 2.0}
    d = _power_law_scene_job({"0": water_y2, "2": FAT_PL}, {"model": "power_law", "y": None})
    path = _write(tmp_path, d, "mixed.json")
    rep = validate_job(path)
    assert not rep.ok
    blob = " ".join(rep.errors)
    assert "mixes absorption exponents" in blob
    assert "Water" in blob and "Fat" in blob
    assert "y = 2" in blob and "y = 1.1" in blob
    assert "absorption.y" in blob  # the message names the fix

    from caustica.__main__ import main

    assert main(["validate", str(path)]) == 2


def test_a_common_exponent_is_read_off_the_materials(tmp_path):
    d = _power_law_scene_job({"0": WATER_PL, "2": FAT_PL}, {"model": "power_law", "y": None})
    job, base = load_job(_write(tmp_path, d))
    built = build_job(job, base_dir=base)
    assert built.absorption_y == 1.1
    rep = validate_job(_write(tmp_path, d, "v.json"))
    assert rep.ok
    assert any("power law, y = 1.1 (read off the material table)" in s for s in rep.summary)
    assert any("absorbed at alpha(f0)" in w for w in rep.warnings)


def test_an_explicit_y_wins_over_a_mixed_table(tmp_path):
    """The whole point of the null: naming y is how a mixed table runs."""
    water_y2 = {**WATER_PL, "y": 2.0}
    d = _power_law_scene_job({"0": water_y2, "2": FAT_PL}, {"model": "power_law", "y": 1.1})
    job, base = load_job(_write(tmp_path, d))
    built = build_job(job, base_dir=base)
    assert built.absorption_y == 1.1
    rep = validate_job(_write(tmp_path, d, "v.json"))
    assert rep.ok
    assert any("power law, y = 1.1 (named by the job)" in s for s in rep.summary)


LEGACY_WATER = {"name": "water", "alpha_np_m": 0.0, "rho": 1000.0, "c": 1500.0, "beta": 0.0}
LEGACY_FAT = {"name": "fat", "alpha_np_m": 6.0, "rho": 932.0, "c": 1450.0, "beta": 0.0}


@pytest.mark.parametrize("absorption", [None, {"model": "single_frequency"}])
def test_a_power_law_table_refuses_the_single_frequency_model(tmp_path, absorption):
    """The default model takes each material's alpha_np_m as it stands, and a
    power-law table has none: baking it at f0 anyway would run the law under a
    summary line saying it does not. Review finding, 2026-09-06."""
    d = _power_law_scene_job(
        {"0": WATER_PL, "2": FAT_PL},
        absorption,
        drive={"f0_mhz": 2.0, "amplitude_kpa": 100.0},
    )
    job, base = load_job(_write(tmp_path, d))
    with pytest.raises(JobError, match="carries no alpha_np_m to take"):
        build_job(job, base_dir=base)
    rep = validate_job(_write(tmp_path, d, "v.json"))
    assert not rep.ok
    blob = " ".join(rep.errors)
    assert "absorption.model" in blob and "power_law" in blob  # the message names the fix


def test_a_legacy_table_under_the_power_law_model_says_where_alpha_came_from(tmp_path):
    """The mirror case is allowed (it is how an existing job asks for
    dispersion off its baked alpha) but must not be reported as an evaluated
    law: nothing in this alpha volume came from alpha0 f^y."""
    d = _power_law_scene_job({"0": LEGACY_WATER, "2": LEGACY_FAT}, {"model": "power_law", "y": 1.1})
    job, base = load_job(_write(tmp_path, d))
    built = build_job(job, base_dir=base)
    assert built.absorption_y == 1.1
    assert built.absorption_from_power_law is False
    assert float(built.medium.alpha.max()) == 6.0  # the legacy number, untouched
    rep = validate_job(_write(tmp_path, d, "v.json"))
    assert rep.ok
    line = next(s for s in rep.summary if s.startswith("absorption:"))
    assert "power law, y = 1.1 (named by the job)" in line
    assert "the material table is the legacy form" in line


def test_each_material_is_evaluated_at_its_own_exponent(tmp_path):
    """The convention AbsorptionConfig states, pinned away from 1 MHz where
    f^y = 1 hides it: the job's y is what the run carries, while alpha at f0
    is each material's own law. Water y = 2 next to fat y = 1.1, job y = 1.1."""
    water_y2 = {**WATER_PL, "y": 2.0}
    d = _power_law_scene_job(
        {"0": water_y2, "2": FAT_PL},
        {"model": "power_law", "y": 1.1},
        drive={"f0_mhz": 2.0, "amplitude_kpa": 100.0},
    )
    job, base = load_job(_write(tmp_path, d))
    built = build_job(job, base_dir=base)
    assert built.absorption_y == 1.1
    assert built.absorption_from_power_law is True

    at_own_y = Material(**water_y2).alpha_np_m_at(2e6)  # 0.0690776 Np/m
    at_job_y = Material(**WATER_PL).alpha_np_m_at(2e6)  # 0.0370177 Np/m
    assert at_own_y / at_job_y == pytest.approx(2.0**0.9, rel=1e-12)  # they differ by 1.87x
    assert float(built.medium.alpha.min()) == pytest.approx(at_own_y, rel=1e-6)
    assert float(built.medium.alpha.min()) != pytest.approx(at_job_y, rel=1e-3)
    # The fat row agrees with both conventions (its own y IS the job's).
    assert float(built.medium.alpha.max()) == pytest.approx(
        Material(**FAT_PL).alpha_np_m_at(2e6), rel=1e-6
    )


def test_only_the_power_law_form_needs_a_positive_frequency():
    """The legacy form ignores the frequency by definition, so it answers at
    0 Hz too; the power law refuses rather than returning 0^y."""
    assert Material(**LEGACY_FAT).alpha_np_m_at(0.0) == 6.0
    assert Material(**LEGACY_FAT).alpha_np_m_at(-1.0) == 6.0
    with pytest.raises(ValueError, match="needs a frequency > 0 Hz"):
        Material(**FAT_PL).alpha_np_m_at(0.0)


def test_medium_from_id_map_refuses_a_power_law_table():
    """The silent-NaN defect, now a refusal.

    `Medium.from_id_map` wrote `material.alpha_np_m` straight into a float32
    volume, and for a power-law material that attribute is None, which numpy
    stores as NaN. The result was a medium whose c, rho and beta were correct
    and whose absorption was NaN in every voxel, with no exception and no
    warning. It now refuses, naming the bake as the fix and listing every
    offending tissue.
    """
    from caustica.medium import Medium

    labels = np.zeros((4, 4, 4), dtype=np.int64)
    labels[..., 2:] = 2
    with pytest.raises(ValueError) as exc:
        Medium.from_id_map(labels, breast_default_power_law())
    msg = str(exc.value)
    assert "power-law absorption" in msg
    assert "at_frequency(f0_hz)" in msg
    # every tissue present is named, not just the first one to fail
    assert "id 0" in msg and "id 2" in msg


def test_medium_homogeneous_refuses_a_power_law_material():
    """The same refusal on the single-material door, which used to raise a
    bare `TypeError: unsupported operand type(s) for *: 'float' and
    'NoneType'` naming no fix."""
    from caustica.medium import Medium

    with pytest.raises(ValueError, match=r"at_frequency\(f0_hz\)"):
        Medium.homogeneous((4, 4, 4), breast_default_power_law()[2])


def test_a_scene_painted_with_a_power_law_table_refuses():
    """The route the defect was actually reachable by: a scene rasterized and
    mapped through a power-law table gave every alpha voxel NaN."""
    from caustica.core.grid import Grid
    from caustica.geometry.configs import SceneConfig

    scene = SceneConfig.model_validate(
        {
            "ndim": 3,
            "background": 0,
            "objects": [
                {
                    "shape": {"kind": "ball", "center_mm": [9, 9, 14], "radius_mm": 4},
                    "label": 2,
                }
            ],
        }
    ).build()
    grid = Grid(shape=(16, 16, 16), dx=1.5e-3)
    with pytest.raises(ValueError, match="power-law absorption"):
        scene.to_medium(grid, breast_default_power_law())


def test_a_legacy_table_still_builds_a_medium():
    """The refusal is on the power-law form only: the legacy table every
    stored job carries, and the same table baked at f0, both still build."""
    from caustica.medium import Medium

    labels = np.zeros((4, 4, 4), dtype=np.int64)
    labels[..., 2:] = 2
    med = Medium.from_id_map(labels, breast_default())
    assert not np.isnan(med.alpha).any()
    assert float(med.alpha.max()) == pytest.approx(breast_default()[2].alpha_np_m, rel=1e-6)

    baked = breast_default_power_law().at_frequency(1e6)
    med_pl = Medium.from_id_map(labels, baked)
    assert not np.isnan(med_pl.alpha).any()
    assert float(med_pl.alpha.max()) == pytest.approx(baked[2].alpha_np_m, rel=1e-6)

    hom = Medium.homogeneous((4, 4, 4), baked[2])
    assert float(hom.alpha.min()) == pytest.approx(baked[2].alpha_np_m, rel=1e-6)


def test_a_power_law_medium_builds_with_alpha_evaluated_at_f0(tmp_path):
    """The new form is not a dead letter: the built medium's alpha volume is
    alpha0 f^y at the drive frequency, per label."""
    d = _power_law_scene_job(
        {"0": WATER_PL, "2": FAT_PL},
        {"model": "power_law", "y": None},
        drive={"f0_mhz": 2.0, "amplitude_kpa": 100.0},
    )
    job, base = load_job(_write(tmp_path, d))
    built = build_job(job, base_dir=base)
    alpha = built.medium.alpha
    want_water = Material(**WATER_PL).alpha_np_m_at(2e6)
    want_fat = Material(**FAT_PL).alpha_np_m_at(2e6)
    assert float(alpha.min()) == pytest.approx(want_water, rel=1e-6)
    assert float(alpha.max()) == pytest.approx(want_fat, rel=1e-6)
    # ... and the job file still carries the law, not the baked number.
    assert job.medium.materials[2].alpha0_db_cm_mhz_y == 0.5


def test_a_homogeneous_power_law_medium_builds_too(tmp_path):
    d = _power_law_scene_job(
        {"0": WATER_PL, "2": FAT_PL},
        {"model": "power_law", "y": 1.1},
        medium={"kind": "homogeneous", "material": FAT_PL},
    )
    job, base = load_job(_write(tmp_path, d))
    built = build_job(job, base_dir=base)
    assert float(built.medium.alpha.max()) == pytest.approx(
        Material(**FAT_PL).alpha_np_m_at(1e6), rel=1e-6
    )


# ------------------------------------------------- medium_volume files


def _volume(tmp_path: Path, db: MaterialDB, name: str = "vol.npz") -> Path:
    from caustica.io.medium_volume import write_medium_volume

    labels = np.zeros((8, 8, 8), dtype=np.int32)
    labels[..., 4:] = 2
    return write_medium_volume(tmp_path / name, dx=0.5e-3, labels=labels, materials=db)


def test_a_medium_volume_file_carries_the_new_fields_and_reads_them_back(tmp_path):
    from caustica.io.medium_volume import load_medium_volume, read_material_db

    db = MaterialDB(materials={0: Material(**WATER_PL), 2: Material(**FAT_PL)})
    path = _volume(tmp_path, db)
    vol = load_medium_volume(path)
    assert vol.materials == db
    assert vol.materials.has_power_law and vol.materials.y() == 1.1
    assert read_material_db(path) == db  # the cheap door, same answer


def test_an_old_medium_volume_file_still_loads_as_legacy(tmp_path):
    from caustica.io.medium_volume import load_medium_volume

    legacy = MaterialDB(
        materials={
            0: Material(name="water", alpha_np_m=0.0, rho=1000.0, c=1500.0, beta=0.0),
            2: Material(name="fat", alpha_np_m=6.0, rho=932.0, c=1450.0, beta=4.5),
        }
    )
    vol = load_medium_volume(_volume(tmp_path, legacy, "legacy.npz"))
    assert vol.materials == legacy
    assert not vol.materials.has_power_law
    med = vol.to_medium()  # no frequency needed, and none is invented
    assert float(med.alpha.max()) == 6.0


def test_a_power_law_volume_needs_a_frequency_before_it_is_a_medium(tmp_path):
    from caustica.io.medium_volume import MediumVolumeError, load_medium_volume

    db = MaterialDB(materials={0: Material(**WATER_PL), 2: Material(**FAT_PL)})
    vol = load_medium_volume(_volume(tmp_path, db))
    with pytest.raises(MediumVolumeError, match="until a frequency is named"):
        vol.to_medium()
    med = vol.to_medium(f0_hz=1.5e6)
    assert float(med.alpha.max()) == pytest.approx(
        Material(**FAT_PL).alpha_np_m_at(1.5e6), rel=1e-6
    )


def test_a_medium_volume_job_resolves_y_from_the_file(tmp_path):
    db = MaterialDB(materials={0: Material(**WATER_PL), 2: Material(**FAT_PL)})
    path = _volume(tmp_path, db)
    d = {
        "format": JOB_FORMAT,
        "kind": "explicit",
        "name": "mv",
        "medium": {"kind": "medium_volume", "file": str(path), "pml_mm": 0.5},
        "source": {
            "kind": "array",
            "array": {"kind": "bowl", "d_outer_mm": 1.2, "roc_mm": 1.5},
            "apex_mm": [1.5, 1.25, 0.6],
        },
        "drive": {"f0_mhz": 1.0, "amplitude_kpa": 100.0},
        "absorption": {"model": "power_law", "y": None},
        "run": {"spec": {"min_settle_periods": 2, "max_settle_periods": 3}},
        "solver": "westervelt",
    }
    job, base = load_job(_write(tmp_path, d, "mv.json"))
    built = build_job(job, base_dir=base)
    assert built.absorption_y == 1.1
    assert float(built.medium.alpha.max()) == pytest.approx(
        Material(**FAT_PL).alpha_np_m_at(1e6), rel=1e-6
    )


def test_a_missing_material_table_is_a_named_refusal(tmp_path):
    """A kind that cannot show its table before the build says so, instead of
    guessing an exponent."""
    from caustica.config.job import resolve_absorption_y
    from caustica.config.kinds import MediumKindConfig

    class _Opaque(MediumKindConfig):
        kind: str = "opaque"

    with pytest.raises(JobError, match="Set absorption.y explicitly"):
        resolve_absorption_y(AbsorptionConfig(model="power_law"), _Opaque())
