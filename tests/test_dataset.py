"""Tests for :mod:`uwcem_phantoms.dataset` — the standard aligned dataset.

Same two tiers as ``test_phantoms.py``: the alignment/padding/bounds logic is
proven on hand-built assets (runs everywhere), and one end-to-end dataset of
the two smallest phantoms is built at a coarse dx against the real archives
(skipped when they are not downloaded).
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from uwcem_phantoms import catalog
from uwcem_phantoms.asset import PhantomAsset, load_phantom
from uwcem_phantoms.dataset import (
    DATASET_FORMAT,
    DEPTH_LIMIT_MM,
    FRONT_GAP_MM,
    DatasetError,
    DatasetPlan,
    SurveyRow,
    _align_into_common,
    _breast_transverse_bbox,
    _check_property_bounds,
    _coupling_fill,
    _nonwater_bbox,
    build_dataset,
    dataset_filename,
    dataset_spec,
    plan_dataset,
    verify_dataset,
)
from uwcem_phantoms.heterogeneity import PROPERTY_NAMES, PropertyVolumes
from uwcem_phantoms.tissue import tissue_table

# The two smallest archives -> the cheapest real end-to-end dataset.
IDS = ("062204", "012304")


def _have_pval(ids=IDS) -> bool:
    return all(catalog.get(i).is_downloaded(with_pval=True) for i in ids)


needs_pval = pytest.mark.skipif(
    not _have_pval(),
    reason="UWCEM archives not downloaded (run: python -m uwcem_phantoms fetch --all)",
)


# --------------------------------------------------------------------------
# synthetic helpers
# --------------------------------------------------------------------------


TABLE = tissue_table("detailed", f0=1.0e6)


def synthetic_asset(
    shape=(12, 13, 15),
    block=(slice(3, 9), slice(4, 10), slice(5, 12)),
    dx=0.5e-3,
    labels: np.ndarray | None = None,
) -> PhantomAsset:
    """Water everywhere, one fatty-2 block — piecewise-mid properties."""
    if labels is None:
        labels = np.zeros(shape, dtype=np.int32)
        labels[block] = 8
    mid = TABLE.lookup("mid")
    idx = labels.astype(np.intp)
    props = PropertyVolumes(**{n: mid[n][idx] for n in PROPERTY_NAMES})
    return PhantomAsset(
        labels=labels,
        dx=dx,
        materials=TABLE.material_db(),
        properties=props,
        meta={"phantom_id": "synthetic"},
    )


def make_plan(
    common_shape, front_gap_vox, dx_mm=0.5, rows=(), depth_limit_mm=None, uncapped_nz=0
) -> DatasetPlan:
    """A plan with no depth ceiling by default; pass both depth arguments to
    make one bite (``uncapped_nz`` is what the axis would have been)."""
    return DatasetPlan(
        dx_mm=dx_mm,
        f0_mhz=1.0,
        margin_mm=5.0,
        front_gap_mm=front_gap_vox * dx_mm,
        front_gap_vox=front_gap_vox,
        common_shape=common_shape,
        rows=tuple(rows),
        depth_limit_mm=depth_limit_mm,
        uncapped_nz=uncapped_nz,
    )


# --------------------------------------------------------------------------
# alignment on synthetic assets
# --------------------------------------------------------------------------


def test_align_pads_front_and_centres_transversally():
    asset = synthetic_asset()  # tissue z starts at 5
    plan = make_plan((24, 26, 28), front_gap_vox=7)  # needs 2 voxels of front pad
    out, alignment = _align_into_common(asset, plan, TABLE)

    assert out.shape == (24, 26, 28)
    box = _nonwater_bbox(out.labels)
    assert box[2].start == 7  # front face exactly at the gap
    # breast bbox centre within half a voxel of the box centre, both axes
    bx, by = _breast_transverse_bbox(out.labels)
    for axis, sl in ((0, bx), (1, by)):
        centre = (sl.start + sl.stop - 1) / 2
        assert abs(centre - (out.shape[axis] - 1) / 2) <= 0.5
    assert alignment["front_face_vox"] == 7
    assert alignment["front_trim_vox"] == 0


def test_align_centres_the_breast_not_the_chest_slab():
    # A full-cross-section fat slab at the back (chest wall / subcutaneous
    # fat) plus a smaller protruding block. The naive non-water bbox spans
    # the whole volume; the breast bbox must ignore the slab — and stay
    # measurable AFTER padding, which is what verification relies on.
    labels = np.zeros((30, 32, 40), dtype=np.int32)
    labels[:, :, 30:] = 8  # slab: covers every transverse voxel
    labels[4:10, 6:14, 5:30] = 8  # the protruding breast
    asset = synthetic_asset(labels=labels)

    bx, by = _breast_transverse_bbox(labels)
    assert (bx.start, bx.stop) == (4, 10)
    assert (by.start, by.stop) == (6, 14)

    plan = make_plan((48, 50, 60), front_gap_vox=5)
    out, alignment = _align_into_common(asset, plan, TABLE)

    # the BREAST is centred...
    nbx, nby = _breast_transverse_bbox(out.labels)
    assert abs((nbx.start + nbx.stop - 1) / 2 - (48 - 1) / 2) <= 0.5
    assert abs((nby.start + nby.stop - 1) / 2 - (50 - 1) / 2) <= 0.5
    # ...which necessarily leaves the slab-dominated non-water bbox OFF centre
    box = _nonwater_bbox(out.labels)
    assert abs((box[0].start + box[0].stop - 1) / 2 - (48 - 1) / 2) > 2
    assert alignment["breast_center_vox"] == [23.5, 24.5]


def test_align_trims_pure_water_front():
    asset = synthetic_asset()  # tissue z starts at 5
    plan = make_plan((24, 26, 28), front_gap_vox=3)  # 2 voxels too much water
    out, alignment = _align_into_common(asset, plan, TABLE)
    assert alignment["front_trim_vox"] == 2
    assert _nonwater_bbox(out.labels)[2].start == 3


def test_align_with_tissue_already_on_the_front_face():
    # Tissue at z = 0 and front_gap 0: nothing to trim or pad on z, and a
    # stray voxel on the front face MOVES the face (the bbox sees it), so the
    # trim can never cut tissue — the in-code guard is defensive only.
    asset = synthetic_asset(block=(slice(3, 9), slice(4, 10), slice(0, 12)))
    plan = make_plan((24, 26, 28), front_gap_vox=0)
    out, alignment = _align_into_common(asset, plan, TABLE)
    assert _nonwater_bbox(out.labels)[2].start == 0
    assert alignment["front_trim_vox"] == 0


def test_align_padding_is_water_in_labels_and_properties():
    asset = synthetic_asset()
    plan = make_plan((24, 26, 28), front_gap_vox=7)
    out, _ = _align_into_common(asset, plan, TABLE)
    fill = _coupling_fill(TABLE)
    corner = (slice(0, 2), slice(0, 2), slice(0, 2))
    assert (out.labels[corner] == 0).all()
    for name in PROPERTY_NAMES:
        np.testing.assert_allclose(
            getattr(out.properties, name)[corner], np.float32(fill[name]), rtol=1e-6
        )
    # and the tissue block survived the move intact
    assert (out.labels == 8).sum() == (asset.labels == 8).sum() == 6 * 6 * 7


def test_align_raises_when_common_box_too_small():
    asset = synthetic_asset()
    plan = make_plan((10, 26, 28), front_gap_vox=7)  # x too small for a 12-wide volume
    with pytest.raises(DatasetError, match="does not fit"):
        _align_into_common(asset, plan, TABLE)


# --------------------------------------------------------------------------
# the depth ceiling: a deliberate, measured, destructive back crop
# --------------------------------------------------------------------------


def test_align_cuts_the_back_when_the_depth_limit_bites():
    # tissue spans z 5..11 in a 15-deep volume; a 10-deep box with a 5-voxel
    # front gap keeps z 0..9 -> the last two tissue planes are cut
    asset = synthetic_asset()
    plan = make_plan((24, 26, 10), front_gap_vox=5, depth_limit_mm=5.0, uncapped_nz=15)
    out, alignment = _align_into_common(asset, plan, TABLE)

    assert out.shape == (24, 26, 10)
    assert alignment["front_face_vox"] == 5  # front plane still exact
    assert alignment["back_trim_vox"] == 5
    assert alignment["truncated_tissue_vox"] == 6 * 6 * 2  # two planes of the block
    assert alignment["truncated_by_class"][8] == 6 * 6 * 2
    assert sum(alignment["truncated_by_class"][1:]) == alignment["truncated_tissue_vox"]
    assert alignment["tissue_at_back_face"] is True
    # what stayed is exactly what the count says
    assert (out.labels == 8).sum() == 6 * 6 * 7 - 6 * 6 * 2


def test_align_back_trim_of_pure_water_reports_no_loss():
    # tissue ends at z=12 of 15; cutting the three trailing water planes is free
    asset = synthetic_asset()
    plan = make_plan((24, 26, 12), front_gap_vox=5, depth_limit_mm=6.0, uncapped_nz=15)
    out, alignment = _align_into_common(asset, plan, TABLE)
    assert alignment["back_trim_vox"] == 3
    assert alignment["truncated_tissue_vox"] == 0
    assert alignment["tissue_at_back_face"] is True  # tissue now ends ON the face
    assert (out.labels == 8).sum() == 6 * 6 * 7  # nothing lost


def test_align_refuses_to_overflow_z_without_a_depth_limit():
    # same too-short box, but no ceiling was asked for: that is a survey bug,
    # and silently cropping it would be the worst possible response
    asset = synthetic_asset()
    plan = make_plan((24, 26, 10), front_gap_vox=5)
    with pytest.raises(DatasetError, match="no depth limit was set"):
        _align_into_common(asset, plan, TABLE)


def test_align_centres_on_the_kept_slab_not_the_discarded_one():
    # A volume whose protruding tissue sits at x 2..6 up front and x 14..18
    # deep. With the deep half cut off, centring must follow the SHALLOW
    # block — the verifier only ever sees what was stored.
    labels = np.zeros((24, 8, 16), dtype=np.int32)
    labels[2:6, 2:6, 4:8] = 8  # kept
    labels[14:18, 2:6, 12:16] = 8  # cut away by the ceiling
    asset = synthetic_asset(labels=labels)
    plan = make_plan((40, 12, 10), front_gap_vox=2, depth_limit_mm=5.0, uncapped_nz=16)
    out, alignment = _align_into_common(asset, plan, TABLE)

    assert alignment["truncated_tissue_vox"] == 4 * 4 * 4
    bx, _ = _breast_transverse_bbox(out.labels)
    assert abs((bx.start + bx.stop - 1) / 2 - (out.shape[0] - 1) / 2) <= 0.5


def test_prev_fft_friendly_never_exceeds_the_ceiling():
    from uwcem_phantoms.processing import next_fft_friendly, prev_fft_friendly

    for n in range(2, 500):
        m = prev_fft_friendly(n)
        assert m <= n
        assert next_fft_friendly(m) == m  # it IS transform friendly
        assert next_fft_friendly(m + 1) > n or m == n  # and it is the largest such
    assert prev_fft_friendly(400) == 400  # 100 mm at 0.25 mm, the standard depth
    assert prev_fft_friendly(401) == 400


def test_check_property_bounds_accepts_mid_and_rejects_out_of_band():
    asset = synthetic_asset()
    report = _check_property_bounds(asset, TABLE)
    assert set(report) == {"0", "8"}
    # fatty-2 c band is (1440, 1475); push one voxel above it
    asset.properties.c[5, 5, 6] = np.float32(1490.0)
    with pytest.raises(DatasetError, match="outside its band"):
        _check_property_bounds(asset, TABLE)


def test_check_property_bounds_rejects_nan():
    asset = synthetic_asset()
    asset.properties.c[5, 5, 6] = np.float32(np.nan)
    with pytest.raises(DatasetError, match="non-finite"):
        _check_property_bounds(asset, TABLE)


def test_build_dataset_refuses_when_ram_is_short(tmp_path, monkeypatch):
    from uwcem_phantoms import dataset as ds

    row = SurveyRow(
        phantom_id="062204",
        acr_class=3,
        native_cropped_shape=(1, 1, 1),
        scaled_shape=(1, 1, 1),
        half_x=1.0,
        half_y=1.0,
        back_z=1.0,
        peak_bytes=10**13,  # 10 TB: no machine passes
    )
    plan = make_plan((8, 8, 8), front_gap_vox=2, rows=(row,))
    with pytest.raises(DatasetError, match="RAM"):
        build_dataset(plan=plan, out_dir=tmp_path)
    # the rail must be a real measurement, not a constant
    monkeypatch.setattr(ds, "_available_ram_bytes", lambda: None)
    # with an unknowable platform the rail steps aside (no raise here means
    # the build would proceed; stop it before any real work by faking build)
    monkeypatch.setattr(ds, "build", lambda spec: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):
        build_dataset(plan=plan, out_dir=tmp_path)


def test_dataset_filename_tags_dx():
    assert dataset_filename("012304", 0.25) == "uwcem-012304-dx0p25mm.npz"
    assert dataset_filename("062204", 1.0) == "uwcem-062204-dx1mm.npz"


def test_dataset_spec_is_the_standard_recipe():
    spec = dataset_spec("012304")
    assert spec.simplify.tissue_model == "detailed"
    assert spec.heterogeneity.use_pval and spec.heterogeneity.noise_pct == 0
    assert spec.crop.mode == "breast"
    assert not spec.domain.fft_friendly  # the COMMON box does the final sizing
    assert spec.resolution.dx_mm == 0.25


def test_plan_dataset_rejects_unknown_ids():
    with pytest.raises(DatasetError, match="unknown phantom ids"):
        plan_dataset(["nope"])
    with pytest.raises(DatasetError, match="no phantom ids"):
        plan_dataset([])


# --------------------------------------------------------------------------
# CLI wiring (no phantom data needed)
# --------------------------------------------------------------------------


def test_dataset_cli_parses_flags():
    from uwcem_phantoms.cli import _cmd_dataset, build_parser

    args = build_parser().parse_args(["dataset", "--dry-run", "--front-gap", "7"])
    assert args.front_gap == 7.0
    assert args.dx == 0.25 and args.f0 == 1.0 and args.margin == 5.0
    assert not args.verify and not args.force
    assert args.func is _cmd_dataset

    # The CLI defaults must BE the module constants, not a copy that drifts:
    # a mismatch would make `dataset` and `build_dataset()` build different
    # geometries from the same words.
    d = build_parser().parse_args(["dataset"])
    assert d.depth == DEPTH_LIMIT_MM
    assert d.front_gap == FRONT_GAP_MM


def test_dataset_cli_routes_build_flags(monkeypatch):
    from uwcem_phantoms import cli
    from uwcem_phantoms import dataset as ds

    calls = {}

    def fake_build(
        ids=None,
        dx_mm=None,
        f0_mhz=None,
        margin_mm=None,
        front_gap_mm=None,
        depth_limit_mm=None,
        out_dir=None,
        progress=None,
        plan=None,
        force=False,
    ):
        calls.update(
            ids=ids,
            dx_mm=dx_mm,
            f0_mhz=f0_mhz,
            depth_limit_mm=depth_limit_mm,
            out_dir=out_dir,
            force=force,
        )
        return {}

    monkeypatch.setattr(ds, "build_dataset", fake_build)
    assert cli.main(["dataset", "062204", "--dx", "0.5", "--force"]) == 0
    assert calls["ids"] == ["062204"]
    assert calls["dx_mm"] == 0.5
    assert calls["depth_limit_mm"] == DEPTH_LIMIT_MM
    assert calls["force"] is True

    # --depth 0 is the escape hatch: no ceiling at all
    assert cli.main(["dataset", "062204", "--depth", "0"]) == 0
    assert calls["depth_limit_mm"] is None


def test_dataset_cli_verify_rejects_build_flags(capsys):
    from uwcem_phantoms import cli

    assert cli.main(["dataset", "062204", "--verify"]) == 2
    assert cli.main(["dataset", "--verify", "--dx", "0.5"]) == 2
    assert cli.main(["dataset", "--verify", "--depth", "80"]) == 2
    err = capsys.readouterr().err
    assert "do not apply" in err


def test_dataset_cli_verify_json_is_pure_json(monkeypatch, capsys):
    from uwcem_phantoms import cli
    from uwcem_phantoms import dataset as ds

    monkeypatch.setattr(
        ds, "verify_dataset", lambda out=None, progress=None: {"files": {"a.npz": {"ok": True}}}
    )
    assert cli.main(["dataset", "--verify", "--json"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out)["files"]["a.npz"]["ok"] is True  # parseable, nothing appended


def test_launcher_dataset_action_routes_the_depth_answer(monkeypatch):
    """The menu must reach the same ceiling the CLI does.

    A dropped keyword here would not fail loudly — it would quietly build an
    UNCAPPED dataset from the menu while the CLI capped, and the two would
    differ by 56 mm of domain.
    """
    from apps import phantom_launcher as pl
    from uwcem_phantoms import dataset as ds

    answers = iter([0.25, 70.0])  # dx, then the depth limit
    monkeypatch.setattr(pl, "ask_float", lambda *a, **k: next(answers))
    monkeypatch.setattr(pl, "ask_bool", lambda prompt, default: "verify" not in prompt and default)
    seen = {}

    def fake_plan(**kw):
        seen.update(kw)
        return make_plan((8, 8, 8), front_gap_vox=2)

    monkeypatch.setattr(ds, "plan_dataset", fake_plan)
    monkeypatch.setattr(ds, "build_dataset", lambda **kw: seen.update(built=True))
    pl.action_dataset()
    assert seen["dx_mm"] == 0.25
    assert seen["depth_limit_mm"] == 70.0
    assert seen["built"] is True


# --------------------------------------------------------------------------
# real data, coarse dx: the full pipeline end to end
# --------------------------------------------------------------------------


@needs_pval
def test_plan_covers_every_surveyed_phantom():
    plan = plan_dataset(IDS, dx_mm=1.0, depth_limit_mm=None)
    assert plan.front_gap_vox == int(round(FRONT_GAP_MM))  # dx = 1 mm here
    assert len(plan.rows) == 2
    assert plan.worst_peak_bytes > 0
    assert not plan.depth_capped and plan.clipped_ids() == []
    for row in plan.rows:
        # the common box must offer at least each phantom's own footprint
        assert plan.common_shape[0] >= row.scaled_shape[0]
        assert plan.common_shape[1] >= row.scaled_shape[1]
        assert plan.common_shape[2] >= row.back_z + plan.front_gap_vox


@needs_pval
def test_plan_depth_limit_shortens_z_and_names_the_cost():
    free = plan_dataset(IDS, dx_mm=1.0, depth_limit_mm=None)
    capped = plan_dataset(IDS, dx_mm=1.0, depth_limit_mm=60.0)

    # z lands exactly on the ceiling (60 mm is 60 voxels at this dx)
    assert capped.common_shape[2] == 60 < free.common_shape[2]
    assert capped.depth_capped and capped.uncapped_nz == free.common_shape[2]
    # x/y are surveyed on the KEPT slab, so a ceiling may resize them too —
    # what must hold is that each phantom's own footprint still fits
    for plan in (free, capped):
        for row in plan.rows:
            assert plan.common_shape[0] >= row.scaled_shape[0]
            assert plan.common_shape[1] >= row.scaled_shape[1]
    # both phantoms are deeper than 60 mm, and the plan says so BEFORE building
    clipped = dict(capped.clipped_ids())
    assert set(clipped) == set(IDS) and all(mm > 0 for mm in clipped.values())
    assert "lose tissue off the back" in capped.summary()
    # a ceiling that cannot even clear the front gap is a mistake, not a build
    with pytest.raises(DatasetError, match="front gap"):
        plan_dataset(IDS, dx_mm=1.0, depth_limit_mm=4.0)


@needs_pval
def test_build_and_verify_dataset_end_to_end(tmp_path):
    manifest = build_dataset(IDS, dx_mm=1.0, out_dir=tmp_path)

    # one common grid, both files present, manifest faithful
    assert manifest["format"] == DATASET_FORMAT
    assert len(manifest["phantoms"]) == 2
    common = tuple(manifest["common_shape"])
    on_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["common_shape"] == list(common)

    # the standard ceiling holds, and every voxel it removed is counted
    assert manifest["depth_limit_mm"] == DEPTH_LIMIT_MM
    assert common[2] * manifest["dx_mm"] <= DEPTH_LIMIT_MM
    clipped = set(manifest["truncation"]["phantoms_clipped"])
    assert clipped, "at these ids the ceiling should bite at least one phantom"
    for entry in manifest["phantoms"]:
        al = entry["alignment"]
        # the per-class breakdown must add up to the headline count, always
        assert al["truncated_tissue_vox"] == sum(al["truncated_by_class"][1:])
        was_clipped = entry["id"] in clipped
        assert (al["truncated_tissue_vox"] > 0) is was_clipped
        # tissue can only be missing off the back if it REACHED the back
        assert not was_clipped or al["tissue_at_back_face"] is True
        assert any("depth limit" in w for w in entry["warnings"]) is was_clipped

    shapes, fronts, centres = [], [], []
    for entry in manifest["phantoms"]:
        asset = load_phantom(tmp_path / entry["file"])
        assert asset.shape == common
        assert asset.properties is not None  # pval => continuous mode
        box = _nonwater_bbox(asset.labels)
        bx, by = _breast_transverse_bbox(asset.labels)
        shapes.append(asset.shape)
        fronts.append(box[2].start)
        centres.append(((bx.start + bx.stop - 1) / 2, (by.start + by.stop - 1) / 2))
        # loadable straight into a solver-ready medium
        medium = asset.to_medium()
        assert medium.c.shape == common
        # pval actually varied the tissue: fat (7-9) must not be constant
        fat = np.isin(asset.labels, (7, 8, 9))
        assert float(asset.properties.c[fat].std()) > 0.1

    assert shapes[0] == shapes[1]
    assert fronts[0] == fronts[1] == manifest["front_gap_vox"]
    target = ((common[0] - 1) / 2, (common[1] - 1) / 2)
    for cx, cy in centres:
        assert abs(cx - target[0]) <= 1.0
        assert abs(cy - target[1]) <= 1.0

    # and the independent verifier agrees
    report = verify_dataset(tmp_path)
    assert len(report["files"]) == 2
    assert all(f["ok"] for f in report["files"].values())


@needs_pval
def test_subset_rebuild_merges_and_variant_recipe_is_refused(tmp_path):
    build_dataset(IDS, dx_mm=1.0, out_dir=tmp_path)
    first = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    # rebuilding ONE id must keep the other's entry and the established grid
    build_dataset(IDS[:1], dx_mm=1.0, out_dir=tmp_path)
    merged = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert {e["id"] for e in merged["phantoms"]} == set(IDS)
    assert merged["common_shape"] == first["common_shape"]
    verify_dataset(tmp_path)  # still one coherent dataset

    # a different recipe into the same directory is refused...
    with pytest.raises(DatasetError, match="different parameters"):
        build_dataset(IDS[:1], dx_mm=1.0, f0_mhz=2.0, out_dir=tmp_path)
    # ...including a different depth ceiling, which dataset_filename does not
    # encode and which would otherwise mix two domain depths under one manifest
    with pytest.raises(DatasetError, match="depth_limit_mm"):
        build_dataset(IDS[:1], dx_mm=1.0, depth_limit_mm=80.0, out_dir=tmp_path)
    with pytest.raises(DatasetError, match="depth_limit_mm"):
        build_dataset(IDS[:1], dx_mm=1.0, depth_limit_mm=None, out_dir=tmp_path)
    # ...and the directory is untouched by the refused attempts
    verify_dataset(tmp_path)


@needs_pval
def test_verify_dataset_catches_corruption(tmp_path):
    build_dataset(IDS[:1], dx_mm=1.0, out_dir=tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = manifest["phantoms"][0]["file"]

    # 1. a wrong manifest format tag is rejected
    bad = dict(manifest)
    bad["format"] = "bogus/9"
    manifest_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(DatasetError, match="format"):
        verify_dataset(tmp_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # 2. a dataset file the manifest does not list is an error, not scenery
    orphan = tmp_path / "uwcem-999999-dx1mm.npz"
    shutil.copyfile(tmp_path / name, orphan)
    with pytest.raises(DatasetError, match="does not list"):
        verify_dataset(tmp_path)
    orphan.unlink()

    # 3. a grid deeper than the depth limit it advertises is caught
    bad = json.loads(json.dumps(manifest))
    bad["depth_limit_mm"] = (bad["common_shape"][2] - 1) * bad["dx_mm"]
    manifest_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(DatasetError, match="depth limit"):
        verify_dataset(tmp_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # 4. a phantom shifted off its front-face plane is caught
    asset = load_phantom(tmp_path / name)
    asset.labels = np.roll(asset.labels, 1, axis=2)
    asset.save(tmp_path / name)
    with pytest.raises(DatasetError, match="front face"):
        verify_dataset(tmp_path)


@needs_pval
def test_verify_catches_a_truncation_claim_the_file_contradicts(tmp_path):
    # Uncapped, so the phantom keeps its water tail and the back face is clean.
    build_dataset(IDS[:1], dx_mm=1.0, depth_limit_mm=None, out_dir=tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["phantoms"][0]["alignment"]["truncated_tissue_vox"] == 0
    verify_dataset(tmp_path)

    manifest["phantoms"][0]["alignment"]["truncated_tissue_vox"] = 1234
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetError, match="truncation record"):
        verify_dataset(tmp_path)
