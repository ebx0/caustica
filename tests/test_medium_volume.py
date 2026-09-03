"""The ``medium_volume`` format — read old bytes, write new.

The two bit-identity gates of the extraction (R11):

* an existing local dataset ``.npz`` loaded through ``medium_volume``
  produces a Medium BIT-identical to the pre-split ``PhantomAsset`` path
  (skipped when the local dataset / the uwcem package is absent — the same
  parity test then lives in the uwcem-phantom repository);
* ``write_medium_volume`` round-trips to a bit-identical Medium.
"""

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from caustica.io.medium_volume import (
    ACCEPTED_FORMAT_TAGS,
    MEDIUM_VOLUME_FORMAT,
    MediumVolume,
    MediumVolumeError,
    load_medium_volume,
    write_medium_volume,
)
from caustica.materials import Material, MaterialDB, water

REPO = Path(__file__).resolve().parents[1]
PHANTOMS = REPO / "data" / "phantoms"
needs_dataset = pytest.mark.skipif(
    not any(PHANTOMS.glob("*.npz")),
    reason="local aligned dataset not present (data/phantoms/*.npz)",
)


def _medium_digest(medium) -> dict[str, str]:
    """One sha256 per array — full-volume bit-identity at single-copy RAM."""
    out = {}
    for name in ("alpha", "rho", "c", "beta"):
        arr = np.ascontiguousarray(getattr(medium, name))
        out[name] = hashlib.sha256(arr.view(np.uint8)).hexdigest()
    if medium.id_map is not None:
        out["id_map"] = hashlib.sha256(
            np.ascontiguousarray(medium.id_map).view(np.uint8)
        ).hexdigest()
    return out


def _tiny_db() -> MaterialDB:
    return MaterialDB(
        materials={
            0: water(),
            2: Material(name="fatty tissue", c=1450.0, rho=950.0, alpha_np_m=4.0, beta=6.0),
        }
    )


def _tiny_labels() -> np.ndarray:
    labels = np.zeros((6, 5, 4), dtype=np.int32)
    labels[2:5, 1:4, 1:3] = 2
    return labels


# --------------------------------------------------- gate 1: read old bytes


@needs_dataset
def test_existing_dataset_file_gives_bit_identical_medium():
    """Same .npz -> medium_volume Medium == PhantomAsset Medium,
    bitwise, full volume (hashed one medium at a time)."""
    asset_mod = pytest.importorskip(
        "uwcem_phantoms.asset"
    )  # pre-split reader (skips after the split)

    npz = sorted(PHANTOMS.glob("*.npz"))[0]
    ref = asset_mod.PhantomAsset.load(npz).to_medium()
    ref_digest = _medium_digest(ref)
    del ref

    vol = load_medium_volume(npz)
    new = vol.to_medium()
    assert _medium_digest(new) == ref_digest
    # the linear switch must agree too (beta zeroed, everything else same)
    assert np.count_nonzero(vol.to_medium(linear=True).beta) == 0


@needs_dataset
def test_existing_dataset_file_grid_and_cmin():
    npz = sorted(PHANTOMS.glob("*.npz"))[0]
    vol = load_medium_volume(npz)
    assert vol.shape == (560, 700, 480)  # the aligned common grid (M6e)
    assert vol.dx == pytest.approx(0.25e-3)
    assert vol.is_continuous  # the dataset stores pval-blended properties
    assert 1400.0 < vol.c_min() < 1600.0


# ---------------------------------------------------- gate 2: write -> read


def test_roundtrip_label_mode_bit_identical(tmp_path):
    labels, db = _tiny_labels(), _tiny_db()
    p = write_medium_volume(
        tmp_path / "vol.npz", dx=0.5e-3, labels=labels, materials=db, meta={"note": "t"}
    )
    vol = load_medium_volume(p)
    assert np.array_equal(vol.labels, labels)
    assert vol.dx == 0.5e-3
    assert not vol.is_continuous
    ref = MediumVolume(labels=labels, dx=0.5e-3, materials=db).to_medium()
    assert _medium_digest(vol.to_medium()) == _medium_digest(ref)
    assert vol.meta["note"] == "t" and vol.meta["stored_properties"] is False


def test_roundtrip_continuous_mode_bit_identical(tmp_path):
    rng = np.random.default_rng(3)
    shape = (5, 6, 7)
    props = {
        "alpha": rng.uniform(0.1, 9, shape).astype(np.float32),
        "rho": rng.uniform(900, 1100, shape).astype(np.float32),
        "c": rng.uniform(1400, 1600, shape).astype(np.float32),
        "beta": rng.uniform(3, 7, shape).astype(np.float32),
    }
    p = write_medium_volume(tmp_path / "cont.npz", dx=0.3e-3, properties=props)
    vol = load_medium_volume(p)
    assert vol.is_continuous
    m = vol.to_medium()
    for name in props:
        assert np.array_equal(getattr(m, name), props[name])  # BIT-identical
    assert vol.c_min() == pytest.approx(float(props["c"].min()))


def test_writer_needs_labels_or_properties(tmp_path):
    with pytest.raises(ValueError, match="labels and/or properties"):
        write_medium_volume(tmp_path / "x.npz", dx=1e-3)


# ------------------------------------------------------------- format tags


def _retag(src: Path, dst: Path, tag: str | None) -> Path:
    """Copy an npz replacing (or dropping) its format member."""
    with np.load(src, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "format"}
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        if tag is not None:
            arrays["format"] = np.asarray(tag)
        for name, arr in arrays.items():
            with zf.open(name + ".npy", "w") as member:
                np.lib.format.write_array(member, np.asanyarray(arr), allow_pickle=False)
    return dst


def test_legacy_tags_accepted_new_tag_written(tmp_path):
    p = write_medium_volume(
        tmp_path / "v.npz", dx=1e-3, labels=_tiny_labels(), materials=_tiny_db()
    )
    with np.load(p) as data:
        assert str(data["format"]) == MEDIUM_VOLUME_FORMAT
    for legacy in sorted(ACCEPTED_FORMAT_TAGS - {MEDIUM_VOLUME_FORMAT}):
        q = _retag(p, tmp_path / f"legacy-{legacy.replace('/', '_')}.npz", legacy)
        assert np.array_equal(load_medium_volume(q).labels, _tiny_labels())
    with pytest.raises(MediumVolumeError, match="format"):
        load_medium_volume(_retag(p, tmp_path / "alien.npz", "someone-elses/9"))
    # no tag at all: accepted (pre-tag files exist in the wild)
    assert load_medium_volume(_retag(p, tmp_path / "untagged.npz", None)).dx == 1e-3


def test_not_a_medium_volume_error(tmp_path):
    p = tmp_path / "junk.npz"
    np.savez(p, foo=np.zeros(3))
    with pytest.raises(MediumVolumeError, match="missing"):
        load_medium_volume(p)


# ------------------------------------------------------------ job schema


def _mv_job(file: str, **over) -> dict:
    from caustica.config.job import JOB_FORMAT

    d = {
        "format": JOB_FORMAT,
        "kind": "explicit",
        "name": "mv",
        "medium": {"kind": "medium_volume", "file": file, "pml_mm": 0.5},
        "source": {
            "kind": "array",
            "array": {"kind": "bowl", "d_outer_mm": 1.2, "roc_mm": 1.5},
            "apex_mm": [1.5, 1.25, 0.6],
        },
        "drive": {"f0_mhz": 1.0, "amplitude_kpa": 100.0},
        "run": {"spec": {"min_settle_periods": 2, "max_settle_periods": 3}},
        "solver": "westervelt",
    }
    d.update(over)
    return d


def _write_mv_file(tmp_path: Path, shape=(30, 25, 26), dx=1e-4, meta=None) -> Path:
    labels = np.zeros(shape, dtype=np.int32)
    labels[:, :, shape[2] // 3 :] = 2  # "tissue" fills the deep two-thirds
    return write_medium_volume(
        tmp_path / "medium.npz", dx=dx, labels=labels, materials=_tiny_db(), meta=meta
    )


def test_job_grid_comes_from_the_file(tmp_path):
    from caustica.config.job import build_job, load_job

    mv = _write_mv_file(tmp_path)
    p = tmp_path / "job.json"
    p.write_text(json.dumps(_mv_job(mv.name)), encoding="utf-8")  # job-relative path
    job, base = load_job(p)
    built = build_job(job, base_dir=base)
    assert built.grid.shape == (30, 25, 26)
    assert built.grid.dx == pytest.approx(1e-4)
    assert built.medium is not None and built.medium.c.shape == (30, 25, 26)


def test_job_rejects_explicit_grid_section(tmp_path):
    from pydantic import TypeAdapter, ValidationError

    from caustica.config.job import JobConfig

    d = _mv_job("x.npz")
    d["grid"] = {"ndim": 3, "dx_mm": 0.1, "size_mm": [3, 2.5, 2]}
    with pytest.raises(ValidationError, match="fixes the grid"):
        TypeAdapter(JobConfig).validate_python(d)


def test_job_focus_in_water_refused_and_escapable(tmp_path):
    from caustica.config.job import JobError, build_job, load_job

    # Tissue only in the last two z-planes: the bowl's NATURAL focus
    # (z = 0.6 + 1.5 mm = voxel 21) lands in the label-0 water gap.
    labels = np.zeros((30, 25, 26), dtype=np.int32)
    labels[:, :, 24:] = 2
    mv = write_medium_volume(
        tmp_path / "deepwater.npz", dx=1e-4, labels=labels, materials=_tiny_db()
    )
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(_mv_job(mv.name)), encoding="utf-8")
    job, base = load_job(p)
    with pytest.raises(JobError, match="water"):
        build_job(job, base_dir=base, with_medium=False)
    # water_label: null disables the check (label 0 need not mean water)
    ok = _mv_job(mv.name)
    ok["medium"]["water_label"] = None
    p2 = tmp_path / "ok.json"
    p2.write_text(json.dumps(ok), encoding="utf-8")
    job, base = load_job(p2)
    assert build_job(job, base_dir=base, with_medium=False).grid.shape == (30, 25, 26)


def test_label_refusals_fire_before_the_medium_is_built(tmp_path, monkeypatch):
    """The speed contract: every cheap refusal runs BEFORE the GBs.

    On a full-size volume ``to_medium()`` materializes four property arrays;
    a job whose focus sits in the coupling water must be refused without
    paying for a single one of them. The ordering was verified by reading
    the code once, and nothing has held it in place since.
    """
    from caustica.config.job import JobError, build_job, load_job

    labels = np.zeros((30, 25, 26), dtype=np.int32)
    labels[:, :, 24:] = 2  # the bowl's natural focus (voxel 21) is in water
    mv = write_medium_volume(
        tmp_path / "deepwater.npz", dx=1e-4, labels=labels, materials=_tiny_db()
    )
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(_mv_job(mv.name)), encoding="utf-8")
    job, base = load_job(p)

    def never_built(*args, **kwargs):
        raise AssertionError("the medium was materialized before the label refusal")

    monkeypatch.setattr(MediumVolume, "to_medium", never_built)
    with pytest.raises(JobError, match="water"):
        build_job(job, base_dir=base)  # with_medium=True: the expensive path


def test_job_f0_guard_generalizes(tmp_path):
    from caustica.config.job import JobError, build_job, load_job

    mv = _write_mv_file(tmp_path, meta={"f0_mhz": 1.5})  # alpha baked at 1.5 MHz
    p = tmp_path / "job.json"
    p.write_text(json.dumps(_mv_job(mv.name)), encoding="utf-8")  # drives 1.0 MHz
    job, base = load_job(p)
    with pytest.raises(JobError, match="baked"):
        build_job(job, base_dir=base, with_medium=False)


def test_job_materials_override_revalidates(tmp_path):
    from caustica.config.job import build_job, load_job

    mv = _write_mv_file(tmp_path)
    d = _mv_job(mv.name)
    d["medium"]["materials"] = {"0": water().model_dump()}  # id 2 now unknown
    p = tmp_path / "job.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    job, base = load_job(p)
    with pytest.raises(Exception, match="MaterialDB"):
        build_job(job, base_dir=base, with_medium=False)


def test_job_runs_end_to_end_mini(tmp_path):
    """A medium_volume job solves on numpy in seconds (full contract)."""
    import warnings

    from caustica import CausticaWarning
    from caustica.runner import EXIT_OK, RunnerOptions, run_job_file

    mv = _write_mv_file(tmp_path)
    p = tmp_path / "job.json"
    p.write_text(json.dumps(_mv_job(mv.name)), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CausticaWarning)
        code = run_job_file(p, RunnerOptions(out=tmp_path / "out", measure=False))
    assert code == EXIT_OK
    assert (tmp_path / "out" / "result.h5").is_file()
