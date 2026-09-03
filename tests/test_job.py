"""The caustica-job/1 contract — round-trip, scene/volume paths,
derived-geometry falsification, and validate's catches.

The stored_setup and phantom_dataset cases moved out with the
phantom package; volume-file coverage lives in tests/test_medium_volume.py.
Tests touching a local dataset npz skip when ``data/phantoms`` is empty (CI).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

import caustica.solvers as solvers
from caustica.config.job import (
    JOB_FORMAT,
    ArraySourceConfig,
    BowlArrayConfig,
    DriveConfig,
    FocusConfig,
    HomogeneousMediumConfig,
    JobConfig,
    JobError,
    MediumVolumeConfig,
    OutputConfig,
    RunConfig,
    SpiralArrayConfig,
    build_job,
    dump_job,
    load_job,
    validate_job,
)
from caustica.geometry.volumes import LabelVolume
from caustica.materials import water

REPO = Path(__file__).resolve().parents[1]
PHANTOMS = REPO / "data" / "phantoms"

needs_dataset = pytest.mark.skipif(
    not any(PHANTOMS.glob("*.npz")),
    reason="phantom dataset not built (python -m uwcem_phantoms dataset)",
)

_ADAPTER: TypeAdapter = TypeAdapter(JobConfig)

WATER = {"name": "water", "alpha_np_m": 0.0, "rho": 1000.0, "c": 1500.0, "beta": 0.0}
FAT = {"name": "fat", "alpha_np_m": 6.0, "rho": 932.0, "c": 1450.0, "beta": 4.5}


def scene_job_dict(**over) -> dict:
    """A small but complete explicit scene job (bowl in water, fat ball)."""
    d = {
        "format": JOB_FORMAT,
        "kind": "explicit",
        "name": "scene-mini",
        "medium": {
            "kind": "scene",
            "scene": {
                "ndim": 3,
                "background": 0,
                "objects": [
                    {"shape": {"kind": "ball", "center_mm": [9, 9, 14], "radius_mm": 4}, "label": 2}
                ],
            },
            "materials": {"0": WATER, "2": {**FAT, "beta": 0.0}},
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
    d.update(over)
    return d


def write_job(tmp_path: Path, d: dict, name: str = "job.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


# ---------------------------------------------------------------- round-trip


@pytest.mark.parametrize(
    "model",
    [
        DriveConfig(f0_mhz=1.2, amplitude_kpa=150.0),
        RunConfig(harmonics=(1, 2, 3), record_region_vox=((4, 40), (0, 32), (2, 60))),
        OutputConfig(folder="out/x", quantize=False),
        HomogeneousMediumConfig(material=water(c=1481.0)),
        MediumVolumeConfig(file="data/volumes/x.npz", pml_mm=4.0, linear=True),
        SpiralArrayConfig(n_elements=32, d_outer_mm=60, d_inner_mm=26.4, roc_mm=60),
        BowlArrayConfig(d_outer_mm=12, roc_mm=15),
        FocusConfig(mode="steered", target_mm=(70.0, 87.5, 60.0)),
        ArraySourceConfig(
            array=BowlArrayConfig(d_outer_mm=12, roc_mm=15), apex_mm=(9.0, 9.0, 3.75)
        ),
    ],
)
def test_every_node_round_trips_through_json(model):
    back = type(model).model_validate_json(model.model_dump_json())
    assert back == model


def test_job_union_round_trips_and_dump_load(tmp_path):
    job = _ADAPTER.validate_python(scene_job_dict())
    assert _ADAPTER.validate_json(job.model_dump_json()) == job
    p = dump_job(job, tmp_path / "j.json")
    back, base = load_job(p)
    assert back == job and base == tmp_path


def test_typo_key_is_an_error_never_a_noop(tmp_path):
    d = scene_job_dict()
    d["amplitutde_kpa"] = 5.0  # top-level typo
    with pytest.raises(ValidationError, match="amplitutde_kpa"):
        _ADAPTER.validate_python(d)
    d2 = scene_job_dict()
    d2["drive"]["ramp_period"] = 2  # nested typo (should be ramp_periods)
    with pytest.raises(ValidationError, match="ramp_period"):
        _ADAPTER.validate_python(d2)


def test_wrong_format_tag_refused(tmp_path):
    p = write_job(tmp_path, {**scene_job_dict(), "format": "caustica-job/9"})
    with pytest.raises(JobError, match="format"):
        load_job(p)


def test_scene_job_end_to_end_mini_solve(tmp_path):
    """The scene gate: SceneConfig -> Medium -> a real (tiny) CPU solve."""
    p = write_job(tmp_path, scene_job_dict())
    job, base = load_job(p)
    built = build_job(job, base_dir=base)
    assert built.medium is not None
    # The fat ball actually made it into the medium (c=1450 inside).
    assert built.medium.c.min() == pytest.approx(1450.0)
    assert built.medium.c.max() == pytest.approx(1500.0)
    res = solvers.get(built.solver)().run(
        built.grid,
        built.medium,
        built.source,
        built.spec,
        backend="numpy",
        record_region=built.record_region,
        reference_point=built.focus_vox,
        harmonics=built.harmonics,
    )
    # The focus sits where the job said, and the field is alive there.
    amp = res.amp
    assert float(amp[built.focus_vox]) > 0.5 * built.source.amplitude


def test_volume_import_job_builds_medium(tmp_path):
    labels = np.zeros((16, 16, 20), np.int32)
    labels[4:12, 4:12, 6:16] = 2  # a fat block in water, breast_default ids
    vol = LabelVolume(labels=labels, dx=0.5e-3, origin=(0.0, 0.0, 0.0))
    vol_path = tmp_path / "block.npz"
    vol.save_npz(vol_path)
    d = scene_job_dict(
        medium={
            "kind": "volume_import",
            "volume": {"format": "npz", "path": "block.npz", "position_mm": [5, 5, 8]},
            "materials": "breast_default",
        }
    )
    d["grid"] = {"ndim": 3, "dx_mm": 0.5, "size_mm": [18, 18, 24], "pml": {"thickness_mm": 2.0}}
    p = write_job(tmp_path, d)  # relative path resolves against the job file dir
    job, base = load_job(p)
    built = build_job(job, base_dir=base)
    assert built.medium is not None
    assert built.medium.c.min() == pytest.approx(1450.0)  # breast_default fat


def test_bowl_cannot_be_steered_or_phased(tmp_path):
    d = scene_job_dict()
    d["source"]["focus"] = {"mode": "steered", "target_mm": [9.0, 9.0, 18.0]}
    p = write_job(tmp_path, d)
    job, base = load_job(p)
    with pytest.raises(JobError, match="single focused element"):
        build_job(job, base_dir=base, with_medium=False)


def test_spiral_natural_vs_steered_phases(tmp_path):
    d = scene_job_dict()
    d["source"]["array"] = {
        "kind": "archimedean_spiral",
        "n_elements": 16,
        "d_outer_mm": 10.0,
        "d_inner_mm": 4.0,
        "roc_mm": 12.0,
    }
    job, base = load_job(write_job(tmp_path, d, "nat.json"))
    built = build_job(job, base_dir=base, with_medium=False)
    assert built.derived["phases"] == "zeros"
    # Natural focus means no per-element steering, which after the phasor sum
    # shows up as a REAL drive: every voxel is 0 or pi, never in between. (The
    # pi entries are the interpolant's side-lobes carrying a negative weight.)
    natural = np.abs(built.source.phases)
    assert np.all((natural < 1e-4) | (np.abs(natural - np.pi) < 1e-4))
    d["source"]["focus"] = {"mode": "steered", "target_mm": [11.0, 9.0, 15.75]}
    job, base = load_job(write_job(tmp_path, d, "steer.json"))
    built2 = build_job(job, base_dir=base, with_medium=False)
    assert built2.derived["phases"].startswith("das")
    steered = np.abs(built2.source.phases)
    assert np.mean((steered > 1e-3) & (np.abs(steered - np.pi) > 1e-3)) > 0.5
    np.testing.assert_array_equal(built2.source.indices, built.source.indices)


def test_derived_geometry_is_falsifiable(tmp_path):
    """The alpha rule generalized: recorded derived values are re-derived and checked."""
    d = scene_job_dict()
    job = _ADAPTER.validate_python(d)
    built = build_job(job, base_dir=tmp_path, with_medium=False)
    src_cfg = job.source
    src_cfg.check_derived(built.derived)  # honest record passes
    tampered = {**built.derived, "f_number": built.derived["f_number"] + 0.01}
    with pytest.raises(JobError, match="f_number"):
        src_cfg.check_derived(tampered)


# ------------------------------------------------------------------ validate


def test_validate_passes_a_good_job_and_warns_on_ppw(tmp_path):
    d = scene_job_dict()
    d["run"]["harmonics"] = [1, 2, 3]  # h3 at dx=0.375 -> under 3 ppw
    rep = validate_job(write_job(tmp_path, d))
    assert rep.ok
    assert any("harmonic 3" in w for w in rep.warnings)


def test_validate_catches_typo(tmp_path):
    d = scene_job_dict()
    d["sover"] = "linear"
    rep = validate_job(write_job(tmp_path, d))
    assert not rep.ok and any("sover" in e for e in rep.errors)


def test_validate_catches_source_buried_in_pml(tmp_path):
    d = scene_job_dict()
    d["source"]["apex_mm"] = [9.0, 9.0, 0.75]  # 2 voxels: whole bowl inside the 6-voxel band
    rep = validate_job(write_job(tmp_path, d))
    assert not rep.ok and any("PML" in e for e in rep.errors)


def test_validate_catches_unknown_solver(tmp_path):
    d = scene_job_dict(solver="kzk")  # planned but not registered yet
    rep = validate_job(write_job(tmp_path, d))
    assert not rep.ok and any("kzk" in e for e in rep.errors)


def test_validate_catches_nonlinear_medium_on_linear_solver(tmp_path):
    d = scene_job_dict()
    d["medium"]["materials"]["2"] = FAT  # beta = 4.5
    d["solver"] = "linear"
    rep = validate_job(write_job(tmp_path, d))
    assert not rep.ok and any("beta" in e for e in rep.errors)


def test_validate_fast_defers_medium_checks(tmp_path):
    d = scene_job_dict()
    d["medium"]["materials"]["2"] = FAT  # would fail caps on 'linear'...
    rep = validate_job(write_job(tmp_path, d), fast=True)
    assert rep.ok  # ...but fast mode defers that to run time
    assert any("deferred" in w for w in rep.warnings)


# ------------------------- janitor ticket 08 (2026-08-23): the beta=0 trap


def test_validate_warns_that_westervelt_on_a_beta_zero_medium_is_a_linear_solve(tmp_path):
    """The UX trap: a nonlinear solver over water() runs linear physics.

    beta=0 everywhere means the westervelt engine has no nonlinear term to
    apply, so the solve is bit-identical to `linear` (the guarantee) — a
    run labelled "westervelt" whose harmonics are numerical residue. Loud,
    but never a block: a linear reference run through the nonlinear engine is
    a legitimate thing to ask for.
    """
    d = scene_job_dict(solver="westervelt")  # every material in it has beta=0
    rep = validate_job(write_job(tmp_path, d))
    assert rep.ok  # a warning, not an error
    hits = [w for w in rep.warnings if "BIT-IDENTICAL" in w]
    assert len(hits) == 1
    assert "medium.material.beta" in hits[0] and "3.5" in hits[0]


def test_validate_stays_quiet_when_the_pairing_is_honest(tmp_path):
    """No warning for a run that never claimed nonlinearity, nor for a real one.

    The `linear` solver on a beta=0 medium is exactly what it says, and
    westervelt on beta>0 is the nonlinear run the warning asks for; a warning
    that fired for those would be noise, and noise gets filtered out.
    """
    honest_linear = scene_job_dict(solver="linear")
    rep = validate_job(write_job(tmp_path, honest_linear, "linear.json"))
    assert not any("BIT-IDENTICAL" in w for w in rep.warnings)

    nonlinear = scene_job_dict(solver="westervelt")
    nonlinear["medium"]["materials"]["2"] = FAT  # beta = 4.5
    rep = validate_job(write_job(tmp_path, nonlinear, "nonlinear.json"))
    assert rep.ok
    assert not any("BIT-IDENTICAL" in w for w in rep.warnings)


def test_cli_validate_prints_the_beta_zero_warning(tmp_path, capsys):
    """The other surface: `caustica validate` shows it, and still exits 0."""
    from caustica.__main__ import main

    path = write_job(tmp_path, scene_job_dict(solver="westervelt"), "beta0.json")
    assert main(["validate", str(path)]) == 0
    out = capsys.readouterr().out
    assert "WARNING" in out and "BIT-IDENTICAL" in out and "medium.material.beta" in out


# ----------------------------------------------------------------------- CLI


def test_cli_validate_exit_codes(tmp_path):
    from caustica.__main__ import main

    good = write_job(tmp_path, scene_job_dict(), "good.json")
    assert main(["validate", str(good)]) == 0
    bad_d = scene_job_dict()
    bad_d["sover"] = "x"
    bad = write_job(tmp_path, bad_d, "bad.json")
    assert main(["validate", str(bad)]) == 2


# ------------------------------------------- adversarial-review regressions


@needs_dataset
def test_validate_warns_on_full_grid_recording(tmp_path):
    npz = sorted(PHANTOMS.glob("*.npz"))[0]
    d = {
        "format": JOB_FORMAT,
        "kind": "explicit",
        "name": "full-rec",
        "medium": {"kind": "medium_volume", "file": str(npz)},
        "source": {
            "kind": "array",
            "array": {
                "kind": "archimedean_spiral",
                "n_elements": 64,
                "d_outer_mm": 60.0,
                "d_inner_mm": 26.4,
                "roc_mm": 60.0,
            },
            "apex_mm": [70.0, 87.5, 5.5],
        },
        "drive": {"f0_mhz": 1.0, "amplitude_kpa": 100.0},  # matches the baked f0
    }
    rep = validate_job(write_job(tmp_path, d))
    assert rep.ok
    assert any("FULL grid" in w for w in rep.warnings)
    assert any("record region" in s for s in rep.summary)
