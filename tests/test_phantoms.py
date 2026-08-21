"""Tests for :mod:`uwcem_phantoms`.

Two tiers, deliberately separated:

* **Synthetic** — the decoders, geometry transforms, processing operators,
  tissue table and export format are exercised on hand-built arrays. These
  run everywhere, need no download, and are where the invariants live.
* **Real data** — a handful of end-to-end checks against a downloaded UWCEM
  phantom, skipped automatically when the archive is not present so CI stays
  green without a 179 MB fetch.

The decoder tests matter most: the fast paths are byte-level tricks, so every
one of them is checked against the slow reference decoder on the same bytes.
A fast decoder that silently disagrees with the file is worse than no fast
decoder at all.
"""

from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np
import pytest
from uwcem_phantoms import catalog, heterogeneity, orientation, processing, reader, tissue
from uwcem_phantoms.asset import FORMAT_TAG, PhantomAsset, PhantomAssetError, load_phantom
from uwcem_phantoms.builder import build, dx_for_budget, plan
from uwcem_phantoms.spec import PhantomSpec

from caustica.geometry import LabelVolume, Scene
from caustica.medium import Medium

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

REFERENCE_ID = "012304"
#: The smallest archive in the catalog — used where a test copies one.
SMALL_ID = "062204"


def _have(phantom_id: str = REFERENCE_ID, with_pval: bool = False) -> bool:
    return catalog.get(phantom_id).is_downloaded(with_pval=with_pval)


needs_data = pytest.mark.skipif(
    not _have(), reason="UWCEM archive not downloaded (run: python -m uwcem_phantoms fetch --all)"
)
needs_pval = pytest.mark.skipif(not _have(with_pval=True), reason="pval archive not downloaded")


def synthetic_codes(shape=(9, 11, 13)) -> np.ndarray:
    """A miniature phantom in NATIVE (s1, s2, s3) order.

    Chest-wall muscle occupies the last two s1 slices at full area, a skin
    shell wraps a fatty core, and everything else is immersion medium — the
    same structural facts :mod:`~uwcem_phantoms.orientation` keys on.
    """
    codes = np.zeros(shape, dtype=np.int8)  # 0 = immersion
    codes[-2:, :, :] = 2  # muscle slab, full area
    codes[2:-2, 2:-2, 2:-2] = 1  # skin shell
    codes[3:-3, 3:-3, 3:-3] = 8  # fatty-2 core
    codes[4:-4, 4:-4, 4:-4] = 3  # fibroglandular-1 nucleus
    return codes


def encode_mtype(codes: np.ndarray, newline: str = "\n") -> bytes:
    """Re-encode a code volume exactly the way the repository writes it."""
    flat = codes.reshape(-1, order="F")
    body = newline.join(reader.MEDIA_TOKENS[int(c)] for c in flat)
    return body.encode("ascii") + newline.encode()


def encode_pval(values: np.ndarray, newline: str = "\n") -> bytes:
    flat = values.reshape(-1, order="F")
    return newline.join(f"{v:1.5f}" for v in flat).encode("ascii") + newline.encode()


# --------------------------------------------------------------------------
# catalog
# --------------------------------------------------------------------------


def test_catalog_covers_every_acr_class():
    assert len(catalog.CATALOG) == 9
    assert {p.acr_class for p in catalog.CATALOG.values()} == {1, 2, 3, 4}
    for info in catalog.CATALOG.values():
        assert info.n_voxels == int(np.prod(info.shape))
        assert info.url("mtype.zip").endswith(f"/{info.phantom_id}/mtype.zip")


def test_catalog_rejects_unknown_id_with_the_valid_list():
    with pytest.raises(KeyError, match="available"):
        catalog.get("nope")


def test_by_class_partitions_the_catalog():
    total = sum(len(catalog.by_class(k)) for k in (1, 2, 3, 4))
    assert total == len(catalog.CATALOG)


# --------------------------------------------------------------------------
# reader — the byte-level decoders
# --------------------------------------------------------------------------


def test_parse_breast_info_reads_every_field():
    meta = reader.parse_breast_info("breast ID=012304\ns1=215\ns2=328\ns3=212\nclassification=4\n")
    assert meta == {"id": "012304", "s1": 215, "s2": 328, "s3": 212, "classification": 4}


def test_parse_breast_info_raises_on_missing_field():
    with pytest.raises(reader.PhantomFormatError, match="missing"):
        reader.parse_breast_info("breast ID=012304\ns1=215\n")


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_fast_mtype_decoder_matches_the_slow_reference(newline):
    codes = synthetic_codes()
    raw = encode_mtype(codes, newline)
    fast = reader.decode_mtype(raw, codes.size)
    slow = reader._decode_mtype_slow(raw, codes.size)
    assert np.array_equal(fast, slow)
    assert np.array_equal(reader._as_native_volume(fast, codes.shape), codes)


def test_mtype_decoder_tolerates_a_missing_final_newline():
    codes = synthetic_codes()
    raw = encode_mtype(codes).rstrip(b"\n")
    assert np.array_equal(reader.decode_mtype(raw, codes.size), codes.reshape(-1, order="F"))


def test_mtype_decoder_refuses_an_unknown_media_number():
    codes = synthetic_codes()
    lines = encode_mtype(codes).split(b"\n")
    lines[0] = b"9.9"  # a media number Table 1 does not define
    raw = b"\n".join(lines)
    with pytest.raises(reader.PhantomFormatError, match="not in Table 1"):
        reader.decode_mtype(raw, codes.size)


def test_mtype_decoder_refuses_a_wrong_voxel_count():
    codes = synthetic_codes()
    with pytest.raises(reader.PhantomFormatError, match="lines but the grid needs"):
        reader.decode_mtype(encode_mtype(codes), codes.size + 7)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_fast_pval_decoder_matches_the_slow_reference(newline):
    rng = np.random.default_rng(0)
    values = np.round(rng.random((5, 6, 7)), 5).astype(np.float32)
    raw = encode_pval(values, newline)
    fast = reader.decode_pval(raw, values.size)
    slow = reader._decode_pval_slow(raw, values.size)
    assert np.allclose(fast, slow, atol=1e-6)
    assert np.allclose(reader._as_native_volume(fast, values.shape), values, atol=1e-6)


def test_pval_decoder_falls_back_when_the_stride_assumption_breaks():
    # Ragged widths defeat the fixed-stride path; the result must still be right.
    raw = b"0.5\n0.25\n"
    assert np.allclose(reader.decode_pval(raw, 2), [0.5, 0.25])


def test_media_number_table_is_consistent():
    assert len(reader.MEDIA_NUMBERS) == len(reader.MEDIA_TOKENS) == reader.N_CLASSES
    for number, token in zip(reader.MEDIA_NUMBERS, reader.MEDIA_TOKENS, strict=True):
        assert float(token) == pytest.approx(number)


def test_as_native_volume_is_a_free_view_of_the_fortran_layout():
    codes = synthetic_codes()
    flat = codes.reshape(-1, order="F")
    view = reader._as_native_volume(flat, codes.shape)
    assert np.array_equal(view, codes)
    assert view.base is not None  # no copy was made


# --------------------------------------------------------------------------
# orientation
# --------------------------------------------------------------------------


def test_canonical_transform_preserves_handedness():
    assert orientation.is_right_handed()


def test_canonical_round_trip_is_lossless():
    codes = synthetic_codes()
    back = orientation.from_canonical(orientation.to_canonical(codes))
    assert np.array_equal(back, codes)


def test_canonical_puts_the_chest_wall_at_high_z():
    canon = orientation.to_canonical(synthetic_codes())
    muscle_z = np.flatnonzero((canon == 2).any(axis=(0, 1)))
    assert muscle_z.min() >= canon.shape[2] - 2
    assert not (canon[:, :, 0] == 2).any()


def test_detect_propagation_axis_finds_the_slab():
    report = orientation.detect_propagation_axis(synthetic_codes(), dx=0.5e-3)
    assert report.propagation_axis == 0
    assert report.slab_area_fraction == pytest.approx(1.0)
    assert report.chest_wall_at_high_index
    assert "propagation axis = s1" in report.summary()


def test_detect_propagation_axis_raises_without_a_slab():
    codes = np.zeros((6, 6, 6), dtype=np.int8)
    codes[2, 2, 2] = 2  # a lone muscle voxel is not a chest wall
    with pytest.raises(ValueError, match="full-area muscle slab"):
        orientation.detect_propagation_axis(codes)


@needs_data
def test_every_downloaded_phantom_has_the_documented_chest_wall():
    for pid in catalog.PHANTOM_IDS:
        if not catalog.get(pid).is_downloaded():
            continue
        raw = reader.load_raw(pid)
        report = orientation.detect_propagation_axis(raw.codes, dx=raw.dx)
        assert report.propagation_axis == 0, pid
        # the manual promises a 0.5 cm muscle chest wall
        assert report.muscle_slab_mm == pytest.approx(5.0, abs=0.5), pid
        assert report.slab_area_fraction > 0.99, pid
        assert report.chest_wall_at_high_index, pid


# --------------------------------------------------------------------------
# processing
# --------------------------------------------------------------------------


def test_crop_reports_the_offset_it_introduced():
    codes = np.zeros((10, 10, 10), dtype=np.int16)
    codes[3:6, 4:8, 2:5] = 7
    res = processing.crop(codes, (7,), margin_vox=0)
    assert res.volume.shape == (3, 4, 3)
    assert res.offset == (3, 4, 2)
    assert np.array_equal(res.volume, codes[3:6, 4:8, 2:5])


def test_crop_margin_is_clamped_to_the_array():
    codes = np.zeros((6, 6, 6), dtype=np.int16)
    codes[1:5, 1:5, 1:5] = 7
    res = processing.crop(codes, (7,), margin_vox=99)
    assert res.volume.shape == codes.shape
    assert res.offset == (0, 0, 0)


def test_crop_raises_when_nothing_matches():
    with pytest.raises(ValueError, match="nothing to crop to"):
        processing.crop(np.zeros((4, 4, 4), np.int16), (5,))


def test_next_fft_friendly_only_returns_smooth_numbers():
    for n in range(1, 400):
        m = processing.next_fft_friendly(n)
        assert m >= n
        r = m
        for p in processing.FFT_PRIMES:
            while r % p == 0:
                r //= p
        assert r == 1, (n, m)
    assert processing.next_fft_friendly(251) == 252
    assert processing.next_fft_friendly(256) == 256


def test_pad_to_fft_friendly_keeps_the_content_where_it_was():
    codes = np.zeros((11, 13, 17), np.int16)
    codes[5, 6, 8] = 3
    padded, before = processing.pad_to_fft_friendly(codes, fill=0)
    assert padded[before[0] + 5, before[1] + 6, before[2] + 8] == 3
    assert padded.shape == (12, 14, 18)


def test_merge_classes_refuses_an_incomplete_table():
    codes = np.array([[[0, 3]]], dtype=np.int16)
    with pytest.raises(ValueError, match="absent from the merge table"):
        processing.merge_classes(codes, {0: 0})


def test_merge_classes_is_a_pure_relabel():
    codes = np.array([[[0, 3, 9]]], dtype=np.int16)
    out = processing.merge_classes(codes, {0: 4, 3: 2, 9: 2})
    assert np.array_equal(out, [[[4, 2, 2]]])


def test_remove_islands_dissolves_into_a_neighbour_not_the_background():
    codes = np.full((7, 7, 7), 8, dtype=np.int16)
    codes[3, 3, 3] = 3  # one stray glandular voxel inside fat
    out, removed = processing.remove_islands(codes, min_voxels=5)
    assert removed == 1
    assert out[3, 3, 3] == 8
    assert set(np.unique(out)) == {8}


def test_remove_islands_protects_listed_classes():
    codes = np.full((7, 7, 7), 8, dtype=np.int16)
    codes[3, 3, 3] = 1
    out, removed = processing.remove_islands(codes, min_voxels=5, protect=(1,))
    assert removed == 0
    assert out[3, 3, 3] == 1


def test_fill_interior_holes_only_touches_enclosed_pockets():
    codes = np.zeros((9, 9, 9), dtype=np.int16)
    codes[2:7, 2:7, 2:7] = 8
    codes[4, 4, 4] = 0  # enclosed pocket
    out = processing.fill_interior_holes(codes, background=0)
    assert out[4, 4, 4] == 8
    assert out[0, 0, 0] == 0  # the outside stays background


def test_keep_largest_component_drops_the_stray_blob():
    codes = np.zeros((12, 6, 6), dtype=np.int16)
    codes[0:6] = 8
    codes[10:12, 0:2, 0:2] = 8
    out = processing.keep_largest_component(codes, background=0)
    assert (out[0:6] == 8).all()
    assert (out[10:12] == 0).all()


def test_resample_codes_matches_label_volume_exactly():
    codes = synthetic_codes((12, 14, 16)).astype(np.int16)
    got = processing.resample_codes(codes, 0.5e-3, 0.25e-3, "smooth")
    want = LabelVolume(labels=codes.astype(np.int32), dx=0.5e-3).resample(0.25e-3, "smooth").labels
    assert np.array_equal(got, want)


def test_resample_field_preserves_a_constant():
    values = np.full((8, 8, 8), 0.375, dtype=np.float32)
    out = processing.resample_field(values, 0.5e-3, 0.3e-3)
    assert out.shape == (13, 13, 13)
    assert np.allclose(out, 0.375)


def test_interface_area_drops_when_smoothing():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 2, size=(16, 16, 16)).astype(np.int16)
    before = processing.interface_area_voxels(codes)
    after = processing.interface_area_voxels(processing.smooth_labels(codes, 1))
    assert after < before


# --------------------------------------------------------------------------
# tissue table
# --------------------------------------------------------------------------


def test_every_class_has_a_tissue_and_a_colour():
    assert len(tissue.DEFAULT_TISSUES) == reader.N_CLASSES
    assert len(tissue.DEFAULT_COLORS) == reader.N_CLASSES
    for colour in tissue.DEFAULT_COLORS:
        assert len(colour) == 7 and colour.startswith("#")


def test_attenuation_follows_the_stored_power_law():
    t = tissue.DEFAULT_TISSUES[2]  # muscle, b = 1.0
    a1 = t.alpha_np_m(1e6)
    a2 = t.alpha_np_m(2e6)
    assert a2 / a1 == pytest.approx(2.0**t.b, rel=1e-6)


def test_db_cm_and_np_m_round_trip():
    assert tissue.np_m_to_db_cm(tissue.db_cm_to_np_m(1.234)) == pytest.approx(1.234)


@pytest.mark.parametrize("model", tissue.TISSUE_MODELS)
def test_tissue_models_cover_every_class_and_serialize(model):
    table = tissue.tissue_table(model, f0=1.5e6)
    assert set(table.code_to_id) == set(range(reader.N_CLASSES))
    db = table.material_db()
    for mat_id in set(table.code_to_id.values()):
        assert mat_id in db
    restored = type(db).model_validate_json(db.model_dump_json())
    assert restored == db


def test_simple_model_matches_breast_default_id_space():
    from caustica.materials import breast_default

    table = tissue.tissue_table("simple")
    assert set(table.material_db().ids) == set(breast_default().ids)
    # id 4 is the bath and id 1 the skin, exactly as breast_default has them
    assert "coupling" in table.tissues[4].name.lower()
    assert table.tissues[1].name == "Skin"
    assert table.coupling_id == 4


def test_pval_ids_exclude_skin_muscle_and_the_bath():
    table = tissue.tissue_table("detailed")
    assert table.pval_ids == frozenset({3, 4, 5, 6, 7, 8, 9})
    assert table.coupling_id == 0


def test_pval_only_lookup_pins_non_pval_ids_to_their_midpoint():
    table = tissue.tissue_table("detailed")
    lo = table.lookup("lo", pval_only=True)
    hi = table.lookup("hi", pval_only=True)
    mid = table.lookup("mid")
    for mat_id in (0, 1, 2):  # bath, skin, muscle
        for name in ("alpha", "rho", "c", "beta"):
            assert lo[name][mat_id] == pytest.approx(mid[name][mat_id])
            assert hi[name][mat_id] == pytest.approx(mid[name][mat_id])
    assert hi["c"][9] > lo["c"][9]  # a pval class does keep its spread


@pytest.mark.parametrize("model", tissue.TISSUE_MODELS)
def test_pval_blends_within_each_media_number_not_the_merged_band(model):
    """``p`` is defined per MEDIA NUMBER, so a merge must not widen its band.

    The repository measures ``p`` against the bounding curves of one tissue
    type. Blending it across a merged id's union range (fatty-1..fatty-3
    together) stretches it ~2x and hands a pure-lipid voxel a fatty-1 sound
    speed. Regression for a review finding, 2026-08-18.
    """
    table = tissue.tissue_table(model)
    lo = table.lookup_by_code("lo")
    hi = table.lookup_by_code("hi")
    for code in tissue.PVAL_CODES:
        own = tissue.DEFAULT_TISSUES[code]
        assert lo["c"][code] == pytest.approx(own.c[0]), code
        assert hi["c"][code] == pytest.approx(own.c[1]), code
        assert lo["rho"][code] == pytest.approx(own.rho[0]), code
        assert hi["rho"][code] == pytest.approx(own.rho[1]), code
    # and the classes with no p data collapse to a single value
    for code in (0, 1, 2):
        for name in ("alpha", "rho", "c", "beta"):
            assert lo[name][code] == pytest.approx(hi[name][code]), (code, name)


def test_lookup_by_code_matches_lookup_when_nothing_is_merged():
    table = tissue.tissue_table("detailed")
    by_code = table.lookup_by_code("lo")
    by_id = table.lookup("lo", pval_only=True)
    for name in ("alpha", "rho", "c", "beta"):
        assert np.allclose(by_code[name], by_id[name][: reader.N_CLASSES])


def test_merged_models_take_the_union_of_their_members_ranges():
    grouped = tissue.tissue_table("grouped")
    fat = grouped.tissues[4]
    members = [tissue.DEFAULT_TISSUES[i] for i in (7, 8, 9)]
    assert fat.c[0] == min(t.c[0] for t in members)
    assert fat.c[1] == max(t.c[1] for t in members)


def test_tissue_table_rejects_unknown_model_and_bad_f0():
    with pytest.raises(ValueError, match="unknown tissue model"):
        tissue.tissue_table("nope")
    with pytest.raises(ValueError, match="f0 must be > 0"):
        tissue.tissue_table("detailed", f0=0.0)


def test_describe_is_json_serializable():
    rows = tissue.tissue_table("detailed").describe()
    assert json.loads(json.dumps(rows))
    assert all("color" in r and "source" in r for r in rows)


# --------------------------------------------------------------------------
# heterogeneity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("corr", [0.0, 1.5, 4.0])
def test_correlated_noise_is_zero_mean_and_unit_variance(corr):
    """Both moments matter: variance sets the noise level, and a nonzero mean
    would be a systematic bias on every tissue voxel rather than noise."""
    field = heterogeneity.correlated_noise((40, 40, 40), corr, np.random.default_rng(0))
    assert float(field.std()) == pytest.approx(1.0, abs=0.02)
    assert float(field.mean()) == pytest.approx(0.0, abs=1e-5)


def test_correlated_noise_is_smoother_when_correlated():
    rng = np.random.default_rng(2)
    white = heterogeneity.correlated_noise((32, 32, 32), 0.0, rng)
    smooth = heterogeneity.correlated_noise((32, 32, 32), 3.0, np.random.default_rng(2))
    assert np.abs(np.diff(smooth, axis=0)).mean() < np.abs(np.diff(white, axis=0)).mean()


def test_pval_interpolation_hits_both_endpoints():
    table = tissue.tissue_table("detailed")
    labels = np.full((2, 2, 2), 9, dtype=np.int32)
    lo, hi = table.lookup("lo", pval_only=True), table.lookup("hi", pval_only=True)
    for p, want in ((0.0, lo), (1.0, hi)):
        props = heterogeneity.PropertyVolumes(
            **{n: table.lookup("mid")[n][labels] for n in heterogeneity.PROPERTY_NAMES}
        )
        out = heterogeneity.apply_pval_interpolation(
            props, labels, np.full(labels.shape, p, np.float32), lo, hi
        )
        for name in heterogeneity.PROPERTY_NAMES:
            assert np.allclose(getattr(out, name), want[name][9], rtol=1e-5)


def test_scatterer_noise_leaves_masked_out_voxels_untouched():
    shape = (10, 10, 10)
    props = heterogeneity.PropertyVolumes(
        alpha=np.full(shape, 5.0, np.float32),
        rho=np.full(shape, 1000.0, np.float32),
        c=np.full(shape, 1500.0, np.float32),
        beta=np.full(shape, 4.0, np.float32),
    )
    mask = np.zeros(shape, bool)
    mask[3:7] = True
    out, meta = heterogeneity.apply_scatterer_noise(
        props, mask, 5.0, 0.0, ("rho", "c"), seed=0, coupled=True
    )
    assert meta["applied"]
    assert np.allclose(out.rho[~mask], 1000.0)
    assert np.allclose(out.c[~mask], 1500.0)
    assert out.rho[mask].std() > 0
    assert np.allclose(out.alpha, 5.0)  # not selected


def test_scatterer_noise_realizes_roughly_the_requested_percentage():
    shape = (48, 48, 48)
    props = heterogeneity.PropertyVolumes(
        alpha=np.zeros(shape, np.float32),
        rho=np.full(shape, 1000.0, np.float32),
        c=np.full(shape, 1500.0, np.float32),
        beta=np.zeros(shape, np.float32),
    )
    out, meta = heterogeneity.apply_scatterer_noise(
        props, np.ones(shape, bool), 4.0, 0.0, ("rho",), seed=1
    )
    assert meta["realized_cv"]["rho"] == pytest.approx(0.04, rel=0.15)


def test_scatterer_noise_never_produces_a_non_positive_property():
    shape = (24, 24, 24)
    props = heterogeneity.PropertyVolumes(
        alpha=np.full(shape, 1.0, np.float32),
        rho=np.full(shape, 1000.0, np.float32),
        c=np.full(shape, 1500.0, np.float32),
        beta=np.full(shape, 4.0, np.float32),
    )
    out, _ = heterogeneity.apply_scatterer_noise(
        props, np.ones(shape, bool), 50.0, 0.0, ("rho", "c", "alpha", "beta"), seed=5
    )
    for name in heterogeneity.PROPERTY_NAMES:
        assert float(getattr(out, name).min()) > 0.0


def test_scatterer_noise_rejects_an_unknown_property():
    props = heterogeneity.PropertyVolumes(
        **{n: np.ones((3, 3, 3), np.float32) for n in heterogeneity.PROPERTY_NAMES}
    )
    with pytest.raises(ValueError, match="unknown properties"):
        heterogeneity.apply_scatterer_noise(
            props, np.ones((3, 3, 3), bool), 1.0, 0.0, ("nope",), seed=0
        )


def test_coupled_noise_moves_properties_together():
    shape = (32, 32, 32)

    def make():
        return heterogeneity.PropertyVolumes(
            alpha=np.zeros(shape, np.float32),
            rho=np.full(shape, 1000.0, np.float32),
            c=np.full(shape, 1500.0, np.float32),
            beta=np.zeros(shape, np.float32),
        )

    mask = np.ones(shape, bool)
    together, _ = heterogeneity.apply_scatterer_noise(
        make(), mask, 5.0, 0.0, ("rho", "c"), seed=3, coupled=True
    )
    apart, _ = heterogeneity.apply_scatterer_noise(
        make(), mask, 5.0, 0.0, ("rho", "c"), seed=3, coupled=False
    )
    r_together = np.corrcoef(together.rho.ravel(), together.c.ravel())[0, 1]
    r_apart = np.corrcoef(apart.rho.ravel(), apart.c.ravel())[0, 1]
    assert r_together > 0.99
    assert abs(r_apart) < 0.1


# --------------------------------------------------------------------------
# spec
# --------------------------------------------------------------------------


def test_spec_round_trips_through_json():
    spec = PhantomSpec(
        phantom_id=REFERENCE_ID,
        f0_mhz=1.5,
        resolution={"dx_mm": 0.4},
        heterogeneity={"use_pval": True, "noise_pct": 2.0, "correlation_mm": 0.5},
    )
    assert PhantomSpec.model_validate_json(spec.model_dump_json()) == spec


def test_spec_rejects_unknown_keys_and_ids():
    with pytest.raises(ValueError):
        PhantomSpec(phantom_id=REFERENCE_ID, typo_key=1)
    with pytest.raises(ValueError, match="unknown phantom_id"):
        PhantomSpec(phantom_id="000000")


def test_manual_crop_requires_both_corners():
    with pytest.raises(ValueError, match="needs both start_mm and size_mm"):
        PhantomSpec(phantom_id=REFERENCE_ID, crop={"mode": "manual", "start_mm": (0, 0, 0)})


def test_default_export_name_describes_the_build():
    spec = PhantomSpec(
        phantom_id=REFERENCE_ID,
        resolution={"dx_mm": 0.3},
        heterogeneity={"use_pval": True, "noise_pct": 2.0},
    )
    name = spec.export_name()
    assert "uwcem012304" in name and "dx0.3mm" in name and "pval" in name and "noise2pct" in name


def test_heterogeneity_flag_reflects_whether_dense_volumes_are_needed():
    off = PhantomSpec(phantom_id=REFERENCE_ID, heterogeneity={"use_pval": False})
    assert not off.heterogeneity.produces_continuous_fields
    on = PhantomSpec(phantom_id=REFERENCE_ID, heterogeneity={"use_pval": False, "noise_pct": 1.0})
    assert on.heterogeneity.produces_continuous_fields


# --------------------------------------------------------------------------
# asset — the export contract
# --------------------------------------------------------------------------


def _tiny_asset(continuous: bool = False) -> PhantomAsset:
    table = tissue.tissue_table("detailed", f0=1e6)
    labels = np.zeros((6, 7, 8), dtype=np.int32)
    labels[1:5, 1:6, 1:7] = 8
    labels[2:4, 2:5, 2:6] = 1
    props = None
    if continuous:
        mid = table.lookup("mid")
        props = heterogeneity.PropertyVolumes(
            **{n: mid[n][labels] for n in heterogeneity.PROPERTY_NAMES}
        )
        props.rho[0, 0, 0] = 1234.0  # something labels alone cannot express
    return PhantomAsset(
        labels=labels,
        dx=0.5e-3,
        materials=table.material_db(),
        properties=props,
        meta={"phantom_id": "synthetic", "tissue_model": "detailed", "f0_mhz": 1.0},
    )


def test_export_round_trips(tmp_path):
    asset = _tiny_asset(continuous=True)
    path = asset.save(tmp_path / "phantom.npz")
    back = load_phantom(path)
    assert np.array_equal(back.labels, asset.labels)
    assert back.dx == asset.dx
    assert back.materials == asset.materials
    assert back.is_continuous
    assert back.properties.rho[0, 0, 0] == pytest.approx(1234.0)


def test_export_is_also_a_plain_caustica_label_volume(tmp_path):
    """The headline compatibility claim: no new code needed to import it."""
    asset = _tiny_asset()
    path = asset.save(tmp_path / "phantom.npz")
    vol = LabelVolume.load_npz(path)
    assert vol.shape == asset.shape
    assert vol.dx == asset.dx
    assert vol.origin == asset.origin
    assert np.array_equal(vol.labels, asset.labels)
    # and it drops straight into a Scene
    scene = Scene(ndim=3, background=0).add_volume(vol)
    assert scene.n_objects == 1


def test_auto_storage_skips_property_volumes_when_labels_suffice(tmp_path):
    labels_only = _tiny_asset().save(tmp_path / "a.npz")
    with np.load(labels_only) as data:
        assert "rho" not in data.files
        assert str(data["format"]) == FORMAT_TAG
    continuous = _tiny_asset(continuous=True).save(tmp_path / "b.npz")
    with np.load(continuous) as data:
        assert {"alpha", "rho", "c", "beta"} <= set(data.files)


def test_store_always_materializes_and_store_never_drops(tmp_path):
    always = _tiny_asset().save(tmp_path / "c.npz", store_properties="always")
    assert load_phantom(always).is_continuous
    never = _tiny_asset(continuous=True).save(tmp_path / "d.npz", store_properties="never")
    assert not load_phantom(never).is_continuous
    with pytest.raises(ValueError, match="auto|always|never"):
        _tiny_asset().save(tmp_path / "e.npz", store_properties="sometimes")


def test_asset_refuses_labels_without_materials():
    table = tissue.tissue_table("simple")
    labels = np.full((3, 3, 3), 7, dtype=np.int32)  # id 7 is not in the simple model
    with pytest.raises(PhantomAssetError, match="no entry in the MaterialDB"):
        PhantomAsset(labels=labels, dx=0.5e-3, materials=table.material_db())


@pytest.mark.parametrize("name", ["../escape", "a/b", "..", ".", "C:/Windows/evil", r"x\y"])
def test_export_name_cannot_escape_the_export_directory(name):
    """``name`` arrives from the GUI body and from spec.name — i.e. from outside.

    ``export_dir() / name`` is not a filename join: a relative name walks out
    and an absolute one replaces the directory entirely, after which save()
    would mkdir its way there and overwrite whatever it found.
    """
    from uwcem_phantoms.paths import export_path

    with pytest.raises(ValueError, match="plain filename"):
        export_path(name)


def test_export_name_accepts_an_ordinary_name():
    from uwcem_phantoms.paths import export_dir, export_path

    got = export_path("uwcem012304-detailed")
    assert got.name == "uwcem012304-detailed.npz"
    assert got.parent == export_dir()


def test_load_rejects_a_foreign_npz(tmp_path):
    path = tmp_path / "foreign.npz"
    np.savez(path, something=np.zeros(3))
    with pytest.raises(PhantomAssetError, match="not a caustica phantom export"):
        load_phantom(path)


def test_medium_construction_in_both_modes():
    label_mode = _tiny_asset().to_medium()
    assert isinstance(label_mode, Medium)
    assert label_mode.shape == (6, 7, 8)
    assert label_mode.id_map is not None

    cont = _tiny_asset(continuous=True).to_medium()
    assert cont.rho[0, 0, 0] == pytest.approx(1234.0)
    assert cont.id_map is not None


def test_grid_matches_the_asset():
    asset = _tiny_asset()
    grid = asset.grid()
    assert grid.shape == asset.shape
    assert grid.dx == asset.dx


def test_summary_mentions_every_populated_material():
    text = _tiny_asset().summary()
    assert "Skin" in text and "Fatty-2" in text and "Coupling" in text


# --------------------------------------------------------------------------
# builder — plan and pipeline
# --------------------------------------------------------------------------


@needs_data
@pytest.mark.parametrize(
    "simplify",
    [
        {"drop_muscle": True},
        {"drop_skin": True},
        {"keep_largest_only": True},
        {"fill_holes": True},
        {"remove_islands_vox": 50},
        {"smooth_iterations": 2},
        {"close_skin_iterations": 2},
        {"drop_skin": True, "smooth_iterations": 1},
    ],
)
def test_plan_is_an_honest_upper_bound_under_simplification(simplify):
    """With label surgery on, plan() promises >= not ==; the promise must hold.

    Dropping skin lets the outermost tissue dissolve into the bath and shrinks
    the box; smoothing and skin closing can grow it. Both directions are
    covered, and the bound is the safe one for a memory rail.
    """
    raw = reader.load_raw(REFERENCE_ID)
    spec = PhantomSpec(
        phantom_id=REFERENCE_ID,
        resolution={"dx_mm": 1.0},
        simplify=simplify,
        crop={"mode": "breast", "margin_mm": 4},
        heterogeneity={"use_pval": False},
    )
    predicted = plan(spec, raw)
    assert not predicted.exact
    assert "upper bound" in predicted.summary()
    assert predicted.n_voxels >= build(spec, raw=raw).n_voxels


@needs_data
def test_plan_predicts_the_built_shape_exactly():
    raw = reader.load_raw(REFERENCE_ID)
    for dx, standoff in ((0.5, 0.0), (0.8, 12.0), (1.2, 5.0)):
        spec = PhantomSpec(
            phantom_id=REFERENCE_ID,
            resolution={"dx_mm": dx},
            crop={"mode": "breast", "margin_mm": 4},
            domain={"standoff_mm": standoff, "fft_friendly": True},
            heterogeneity={"use_pval": False},
        )
        predicted = plan(spec, raw)
        assert predicted.exact
        assert predicted.shape == build(spec, raw=raw).shape


@needs_data
def test_dx_for_budget_respects_the_budget_and_never_refines():
    raw = reader.load_raw(REFERENCE_ID)
    spec = PhantomSpec(
        phantom_id=REFERENCE_ID, resolution={"dx_mm": 0.3}, heterogeneity={"use_pval": False}
    )
    dx = dx_for_budget(spec, 3_000_000, raw)
    assert dx >= 0.3e-3
    coarser = spec.model_copy(deep=True)
    coarser.resolution = coarser.resolution.model_copy(update={"dx_mm": dx * 1e3})
    assert plan(coarser, raw).n_voxels <= 3_000_000


@needs_data
def test_breast_crop_is_tighter_than_the_full_tissue_box():
    raw = reader.load_raw(REFERENCE_ID)

    def shape_for(mode):
        spec = PhantomSpec(
            phantom_id=REFERENCE_ID,
            crop={"mode": mode, "margin_mm": 4},
            domain={"fft_friendly": False},
            heterogeneity={"use_pval": False},
        )
        return plan(spec, raw).shape

    breast, full = shape_for("breast"), shape_for("tissue")
    assert breast[0] < full[0] and breast[1] < full[1]
    assert breast[2] == full[2]  # the beam axis keeps the whole chest wall


@needs_data
def test_max_voxels_rail_refuses_before_allocating():
    spec = PhantomSpec(
        phantom_id=REFERENCE_ID,
        resolution={"dx_mm": 0.5},
        domain={"max_voxels": 1000},
        heterogeneity={"use_pval": False},
    )
    with pytest.raises(ValueError, match="max_voxels rail"):
        build(spec)


@needs_data
def test_dropping_the_chest_wall_warns_and_removes_it():
    spec = PhantomSpec(
        phantom_id=REFERENCE_ID,
        resolution={"dx_mm": 1.0},
        simplify={"drop_muscle": True},
        heterogeneity={"use_pval": False},
    )
    asset = build(spec)
    muscle_id = asset.label_of("muscle")
    assert muscle_id is None or asset.histogram().get(muscle_id, 0) == 0
    assert any("chest wall removed" in w for w in asset.meta["warnings"])


@needs_data
def test_a_coarse_build_warns_about_the_unresolved_skin_layer():
    spec = PhantomSpec(
        phantom_id=REFERENCE_ID,
        resolution={"dx_mm": 2.0},
        heterogeneity={"use_pval": False},
    )
    asset = build(spec)
    assert any("skin" in w for w in asset.meta["warnings"])


@needs_data
def test_standoff_puts_coupling_medium_in_front_of_the_tissue():
    spec = PhantomSpec(
        phantom_id=REFERENCE_ID,
        resolution={"dx_mm": 1.0},
        domain={"standoff_mm": 20.0, "fft_friendly": False},
        heterogeneity={"use_pval": False},
    )
    asset = build(spec)
    coupling = tissue.tissue_table("detailed").coupling_id
    assert (asset.labels[:, :, :19] == coupling).all()
    assert not asset.meta["warnings"] or all("z = 0 face" not in w for w in asset.meta["warnings"])


@needs_data
def test_build_metadata_carries_the_spec_the_citation_and_the_log():
    spec = PhantomSpec(
        phantom_id=REFERENCE_ID, resolution={"dx_mm": 1.5}, heterogeneity={"use_pval": False}
    )
    meta = build(spec).meta
    assert PhantomSpec.model_validate(meta["spec"]) == spec
    assert "University of Wisconsin" in meta["citation"]
    assert meta["log"] and meta["canonical_axes"][2].startswith("propagation")
    assert meta["materials_table"]


@needs_data
def test_labels_only_build_stores_no_property_volumes():
    spec = PhantomSpec(
        phantom_id=REFERENCE_ID, resolution={"dx_mm": 1.5}, heterogeneity={"use_pval": False}
    )
    asset = build(spec)
    assert not asset.is_continuous
    assert "piecewise-constant" in " ".join(asset.meta["log"])


@needs_pval
def test_pval_build_leaves_skin_and_muscle_at_their_midpoints():
    spec = PhantomSpec(
        phantom_id=REFERENCE_ID,
        resolution={"dx_mm": 1.0},
        heterogeneity={"use_pval": True},
    )
    asset = build(spec)
    assert asset.is_continuous
    table = tissue.tissue_table("detailed", f0=spec.f0)
    mid = table.lookup("mid")
    for mat_id in (0, 1, 2):  # bath, skin, muscle — the repository gives them p = 0
        sel = asset.labels == mat_id
        if sel.any():
            assert np.allclose(asset.properties.c[sel], mid["c"][mat_id], rtol=1e-4)
    fat = asset.labels == 8
    if fat.any():
        assert asset.properties.c[fat].std() > 0  # a pval class does vary


@needs_pval
def test_noise_is_reproducible_and_seed_dependent():
    def rho_for(seed):
        spec = PhantomSpec(
            phantom_id=REFERENCE_ID,
            resolution={"dx_mm": 1.5},
            heterogeneity={
                "use_pval": False,
                "noise_pct": 3.0,
                "correlation_mm": 0.0,
                "seed": seed,
            },
        )
        return build(spec).properties.rho

    assert np.array_equal(rho_for(7), rho_for(7))
    assert not np.array_equal(rho_for(7), rho_for(8))


@needs_data
def test_end_to_end_export_imports_into_a_simulation(tmp_path):
    """The whole point of the module, checked as an outside consumer would."""
    import caustica as hs

    spec = PhantomSpec(
        phantom_id=REFERENCE_ID,
        f0_mhz=1.2,
        resolution={"dx_mm": 1.2},
        crop={"mode": "breast", "margin_mm": 4},
        domain={"standoff_mm": 6.0},
        heterogeneity={"use_pval": False},
    )
    path = build(spec).save(tmp_path / "sim.npz")

    ph = load_phantom(path)
    grid = ph.grid(pml=hs.PMLSpec(thickness=3e-3))
    medium = ph.to_medium()

    assert grid.shape == medium.shape == ph.shape
    assert grid.dx == ph.dx
    assert medium.c_min > 1000.0 and medium.c_max < 2000.0
    assert medium.alpha.min() >= 0.0
    assert not medium.is_linear
    assert medium.alpha.dtype == medium.rho.dtype == np.float32
    for vol in (medium.alpha, medium.rho, medium.c, medium.beta):
        assert vol.flags["C_CONTIGUOUS"]
        assert np.isfinite(vol).all()


def test_to_medium_linear_zeroes_beta_for_the_linear_solver():
    """A phantom medium is always nonlinear, so the linear solver rejects it."""
    asset = _tiny_asset(continuous=True)
    assert not asset.to_medium().is_linear
    assert asset.to_medium(linear=True).is_linear
    # zeroing beta must not disturb the other three volumes
    full, lin = asset.to_medium(), asset.to_medium(linear=True)
    for name in ("alpha", "rho", "c"):
        assert np.array_equal(getattr(full, name), getattr(lin, name))


@needs_data
@pytest.mark.parametrize(
    ("dx_mm", "f0_mhz", "expect"),
    [
        (1.5, 1.0, "UNUSABLE"),  # 0.9 ppw — the solve runs and produces nothing
        (1.5, 0.4, "marginal resolution"),  # 2.4 ppw
        (0.3, 1.0, None),  # ~4.8 ppw, no complaint
    ],
)
def test_build_warns_when_dx_cannot_carry_the_wave(dx_mm, f0_mhz, expect):
    spec = PhantomSpec(
        phantom_id=REFERENCE_ID,
        f0_mhz=f0_mhz,
        resolution={"dx_mm": dx_mm},
        crop={"mode": "breast", "margin_mm": 2},
        domain={"fft_friendly": False},
        heterogeneity={"use_pval": False},
        # the 0.3 mm case is only built to read its warnings, never allocated
        **({"name": "ppw-check"}),
    )
    if dx_mm < 0.5:
        # cheap path: the warning depends only on dx, f0 and the tissue table
        from uwcem_phantoms.builder import _resolution_vs_wavelength

        table = tissue.tissue_table("detailed", f0=spec.f0)
        assert _resolution_vs_wavelength(table, dx_mm * 1e-3, spec) == []
        return
    warnings = build(spec).meta["warnings"]
    hit = [w for w in warnings if expect in w]
    assert hit, warnings
    assert "points per wavelength" in hit[0]
    assert "drop dx to" in hit[0]  # the warning must say what to do


# --------------------------------------------------------------------------
# regressions from the adversarial review, 2026-08-18
# --------------------------------------------------------------------------


def test_mtype_decoder_rejects_a_token_that_only_shares_a_prefix():
    """The byte key is 3 wide, so "1.15" matches "1.1" unless length is checked."""
    raw = b"\n".join([b"1.1"] * 5 + [b"1.15"]) + b"\n"
    with pytest.raises(reader.PhantomFormatError, match="not in Table 1"):
        reader.decode_mtype(raw, 6)


@pytest.fixture
def isolated_phantom(tmp_path, monkeypatch):
    """A private data root holding one small phantom, so caching is testable."""
    src = catalog.get(SMALL_ID)
    if not src.is_downloaded():
        pytest.skip("UWCEM archive not downloaded")
    dest = tmp_path / "uwcem" / SMALL_ID
    dest.mkdir(parents=True)
    for name in ("breastInfo.txt", "mtype.zip"):
        shutil.copy(src.local(name), dest / name)
    monkeypatch.setenv("CAUSTICA_PHANTOM_DATA", str(tmp_path))
    return SMALL_ID


def test_concurrent_decodes_of_one_phantom_do_not_collide(isolated_phantom):
    """The studio fires /api/plan and /api/build at the same uncached phantom.

    A shared fixed ".part" name made the two publishes collide — PermissionError
    on Windows, a lost rename on POSIX — and the failure escaped a load that
    had already succeeded (review finding, 2026-08-18).
    """
    import concurrent.futures as cf

    from uwcem_phantoms.paths import cache_dir

    def load(_):
        return int(reader.load_raw(isolated_phantom, fetch_missing=False).codes.sum())

    with cf.ThreadPoolExecutor(4) as pool:
        sums = list(pool.map(load, range(4)))
    assert len(set(sums)) == 1, sums
    assert not list(cache_dir().glob("*.part")), "temp files left behind"


def test_a_corrupt_cache_is_re_decoded_rather_than_raised(isolated_phantom):
    """zipfile.BadZipFile is not an OSError, so a narrow except let it escape."""
    from uwcem_phantoms.paths import cache_dir

    good = int(reader.load_raw(isolated_phantom, fetch_missing=False).codes.sum())
    cached = next(cache_dir().glob("*-mtype.npz"))
    cached.write_bytes(b"PK not really a zip")
    assert int(reader.load_raw(isolated_phantom, fetch_missing=False).codes.sum()) == good


def test_cache_is_invalidated_when_the_source_archive_changes(isolated_phantom):
    """A re-download must not keep serving the array decoded from old bytes."""
    from uwcem_phantoms.paths import cache_dir

    reader.load_raw(isolated_phantom, fetch_missing=False)
    cached = next(cache_dir().glob("*-mtype.npz"))
    stamp_before = cached.stat().st_mtime_ns
    archive = catalog.get(isolated_phantom).local("mtype.zip")
    os.utime(archive, (time.time() + 10, time.time() + 10))
    reader.load_raw(isolated_phantom, fetch_missing=False)
    assert cached.stat().st_mtime_ns != stamp_before, "stale cache was served"


def test_remove_islands_reaches_a_fixed_point():
    """One pass can create a new undersized component of the donor class."""
    from scipy import ndimage

    rng = np.random.default_rng(3)
    codes = rng.integers(3, 7, size=(24, 24, 24)).astype(np.int16)
    out, removed = processing.remove_islands(codes, min_voxels=25)
    assert removed > 0
    structure = ndimage.generate_binary_structure(3, 1)
    for cls in np.unique(out):
        labeled, n = ndimage.label(out == cls, structure=structure)
        if n:
            sizes = np.bincount(labeled.reshape(-1))
            sizes[0] = 0
            assert not ((sizes > 0) & (sizes < 25)).any(), f"class {cls} still has islands"


def test_fill_from_restricts_the_donor_set():
    codes = np.zeros((1, 1, 9), dtype=np.int16)
    codes[0, 0, 3:5] = 1  # a shell with background in front and tissue behind
    codes[0, 0, 5:] = 8
    doomed = codes == 1
    got = processing.fill_from(codes, doomed, donors=(codes == 8), fallback=0)
    assert (got == 8).all()  # never the background in front


def test_pval_resample_does_not_blend_the_no_data_sentinel():
    """p = 0 in skin means 'no data'; blending it halves p in the tissue rim."""
    codes = np.full((2, 2, 24), 8, np.int16)
    codes[:, :, :6] = 1
    pval = np.full((2, 2, 24), 0.6, np.float32)
    pval[:, :, :6] = 0.0
    dx0, dx1 = 0.5e-3, 0.7e-3  # ratio 1.4 — where the unmasked blend bit
    valid = np.isin(codes, np.asarray(sorted(tissue.PVAL_CODES), dtype=codes.dtype))
    weight = processing.resample_field(valid.astype(np.float32), dx0, dx1)
    masked = processing.resample_field(pval * valid, dx0, dx1)
    np.divide(masked, weight, out=masked, where=weight > 1e-6)
    unmasked = processing.resample_field(pval, dx0, dx1)
    fat = processing.resample_codes(codes, dx0, dx1, "smooth") == 8
    assert np.allclose(masked[fat], 0.6, atol=1e-5)
    assert unmasked[fat].min() < 0.5  # the bug this guards against


def test_fft_padding_never_eats_into_the_requested_standoff():
    """Centre-anchoring z inserted FFT growth in front of the transducer."""
    from uwcem_phantoms.processing import next_fft_friendly

    codes = np.zeros((7, 7, 41), dtype=np.int16)
    codes[2:5, 2:5, 30:] = 8
    spec = PhantomSpec(
        phantom_id=REFERENCE_ID,
        domain={"standoff_mm": 10.0, "fft_friendly": True},
        heterogeneity={"use_pval": False},
    )
    from uwcem_phantoms.builder import _grow_domain

    dx = 0.5e-3
    grown, _, before = _grow_domain(codes, None, spec, dx)
    assert before[2] == int(round(10.0e-3 / dx)), "z padding must be exactly the standoff"
    assert grown.shape[2] == next_fft_friendly(codes.shape[2] + before[2])


@needs_data
def test_scatterer_noise_metadata_reports_the_noise_not_tissue_contrast():
    spec = PhantomSpec(
        phantom_id=REFERENCE_ID,
        resolution={"dx_mm": 1.5},
        heterogeneity={"use_pval": False, "noise_pct": 3.0, "correlation_mm": 0.0, "seed": 4},
    )
    realized = build(spec).meta["heterogeneity"]["noise"]["realized_cv"]
    for name, value in realized.items():
        assert value == pytest.approx(0.03, rel=0.15), (name, value)
