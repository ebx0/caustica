"""M6b gates: CSG primitives/algebra, scenes, axisymmetric, import, resample."""

import numpy as np
import pytest

from caustica import Grid, PMLSpec
from caustica.geometry import (
    Ball,
    Box,
    Cylinder,
    Ellipsoid,
    HalfSpace,
    LabelVolume,
    Scene,
    SceneConfig,
    load_labels_txt,
)


def _grid2d(n=100, dx=1e-3):
    return Grid(shape=(n, n), dx=dx)


def _points2d(n=200, span=0.1):
    x = np.linspace(0, span, n)
    gx, gy = np.meshgrid(x, x, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel()])


# ---------------- primitives ----------------


def test_primitive_membership_analytics():
    pts = np.array([[0.0, 0.0, 0.0], [0.9e-3, 0, 0], [1.1e-3, 0, 0]])
    ball = Ball((0, 0, 0), 1e-3)
    np.testing.assert_array_equal(ball.contains(pts), [True, True, False])

    box = Box((0, 0, 0), (2e-3, 4e-3, 6e-3))
    assert box.contains(np.array([[0.99e-3, 1.99e-3, 2.99e-3]]))[0]
    assert not box.contains(np.array([[1.01e-3, 0, 0]]))[0]

    cyl = Cylinder((0, 0, 0), radius=1e-3, height=4e-3, axis=2)
    assert cyl.contains(np.array([[0.5e-3, 0, 1.9e-3]]))[0]
    assert not cyl.contains(np.array([[0.5e-3, 0, 2.1e-3]]))[0]
    assert not cyl.contains(np.array([[1.1e-3, 0, 0]]))[0]

    ell = Ellipsoid((0, 0), (2e-3, 1e-3))
    assert ell.contains(np.array([[1.9e-3, 0]]))[0]
    assert not ell.contains(np.array([[0, 1.1e-3]]))[0]

    half = HalfSpace((1e-3, 0), (1, 0))
    np.testing.assert_array_equal(
        half.contains(np.array([[0.5e-3, 5], [1.5e-3, -5]])), [True, False]
    )


def test_primitive_validation():
    with pytest.raises(ValueError):
        Ball((0, 0), -1.0)
    with pytest.raises(ValueError):
        Box((0, 0), (1e-3,))
    with pytest.raises(ValueError):
        Cylinder((0, 0), 1e-3, 1e-3)  # 2 entries for a 3-D shape
    with pytest.raises(ValueError):
        HalfSpace((0, 0), (0, 0))
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        Ball((0, 0), 1e-3).contains(np.zeros((5, 3)))


def test_csg_algebra_matches_numpy_boolean():
    pts = _points2d()
    a = Ball((0.04, 0.05), 0.02)
    b = Box((0.06, 0.05), (0.03, 0.05))
    c = Ball((0.05, 0.05), 0.01)
    ma, mb, mc = a.contains(pts), b.contains(pts), c.contains(pts)
    np.testing.assert_array_equal((a | b).contains(pts), ma | mb)
    np.testing.assert_array_equal((a & b).contains(pts), ma & mb)
    np.testing.assert_array_equal((a - b).contains(pts), ma & ~mb)
    np.testing.assert_array_equal((~a).contains(pts), ~ma)
    combo = (a | b) & ~c
    np.testing.assert_array_equal(combo.contains(pts), (ma | mb) & ~mc)


def test_mixed_dimension_combination_rejected():
    with pytest.raises(ValueError, match="cannot combine"):
        _ = Ball((0, 0), 1e-3) | Ball((0, 0, 0), 1e-3)


# ---------------- transforms ----------------


def test_translate_rotate_scale():
    pts = np.array([[2e-3, 0.0]])
    ball = Ball((0.0, 0.0), 1e-3)
    assert not ball.contains(pts)[0]
    assert ball.translated((2e-3, 0)).contains(pts)[0]

    # 45-degree rotated box: corner points move in/out predictably.
    box = Box((0, 0), (2e-3, 1e-3)).rotated(np.pi / 2)
    assert box.contains(np.array([[0, 0.9e-3]]))[0]  # long side now along y
    assert not box.contains(np.array([[0.9e-3, 0]]))[0]

    big = Ball((0, 0), 1e-3).scaled(3.0)
    assert big.contains(np.array([[2.5e-3, 0]]))[0]

    # 3-D rotation about z: x-axis cylinder becomes y-axis cylinder.
    cyl = Cylinder((0, 0, 0), radius=0.5e-3, height=4e-3, axis=0)
    rot = cyl.rotated(np.pi / 2, axis=(0, 0, 1))
    assert rot.contains(np.array([[0, 1.5e-3, 0]]))[0]
    assert not rot.contains(np.array([[1.5e-3, 0, 0]]))[0]


def test_rotation_preserves_area():
    pts = _points2d(n=300)
    box = Box((0.05, 0.05), (0.03, 0.01))
    a0 = box.contains(pts).sum()
    a1 = box.rotated(0.7, center=(0.05, 0.05)).contains(pts).sum()
    assert abs(a1 - a0) / a0 < 0.02  # sampled area invariant under rotation


# ---------------- scene + rasterization ----------------


def test_scene_paint_order_and_background():
    grid = _grid2d()
    scene = Scene(ndim=2, background=0)
    scene.add(Ball((0.05, 0.05), 0.03), 1)
    scene.add(Ball((0.05, 0.05), 0.015), 2)  # later paints over
    vol = scene.rasterize(grid)
    assert vol.label_set == (0, 1, 2)
    c = vol.labels[50, 50]
    assert c == 2
    assert vol.labels[0, 0] == 0
    assert vol.labels[50, 50 + 20] == 1  # 20 mm: inside outer, outside inner


def test_sphere_volume_accuracy_and_supersampling():
    """Volume accuracy gate + supersampling convergence gate.

    Boundary over/under-inclusions cancel statistically, so VOLUME error is
    tiny even at s=1; supersampling's measurable benefit is per-voxel
    boundary classification — s=3 must sit at least as close to the
    converged (s=5) rasterization as s=1 does.
    """
    grid = Grid(shape=(48, 48, 48), dx=1e-3)
    c, r = (0.0243, 0.0241, 0.0244), 15.3e-3  # deliberately voxel-misaligned
    v_true = 4 / 3 * np.pi * r**3
    labs = {}
    for s in (1, 3, 5):
        labs[s] = Scene(ndim=3).add(Ball(c, r), 1).rasterize(grid, supersample=s).labels
        v = (labs[s] == 1).sum() * grid.dx**3
        assert abs(v - v_true) / v_true < 0.02
    d1 = int((labs[1] != labs[5]).sum())
    d3 = int((labs[3] != labs[5]).sum())
    assert d3 <= d1  # supersampling converges toward the fine rasterization
    assert d3 < 20  # and the boundary disagreement is a handful of voxels


def test_axisymmetric_scene():
    grid = _grid2d(n=80, dx=0.5e-3)
    scene = Scene(ndim=2, axisymmetric=True, background=0)
    scene.add(Ball((0.0, 0.02), 0.01), 1)  # sphere on the axis in (r, z)
    vol = scene.rasterize(grid)
    assert vol.labels[0, 40] == 1  # on-axis voxel inside
    with pytest.raises(ValueError, match="r >= 0"):
        scene.rasterize(grid, origin=(-0.01, 0.0))
    with pytest.raises(ValueError, match="2-D"):
        Scene(ndim=3, axisymmetric=True)


def test_scene_validation():
    with pytest.raises(ValueError, match="3-D shape"):
        Scene(ndim=2).add(Ball((0, 0, 0), 1e-3), 1)
    with pytest.raises(ValueError, match="supersample"):
        Scene(ndim=2).add(Ball((0, 0), 1e-3), 1).rasterize(_grid2d(), supersample=0)


# ---------------- volumes: import + resample ----------------


def test_label_volume_validation():
    with pytest.raises(TypeError):
        LabelVolume(labels=np.zeros((4, 4)), dx=1e-3)
    with pytest.raises(ValueError):
        LabelVolume(labels=np.zeros((4, 4), np.uint8), dx=-1.0)


def _txt_mapping(values):
    # An arbitrary value->label rule for the GENERIC text-import tests (the
    # production phantom's own rule moved out with its package at M10k/W0c).
    labels = np.full(values.shape, 4, dtype=np.uint8)
    labels[values == -2] = 1
    labels[values > 0] = 2
    labels[values == -4] = 3
    return labels


def test_txt_import_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    raw = rng.choice([-4.0, -2.0, 0.5, np.nan], size=(6, 5, 4))
    path = tmp_path / "phantom.txt"
    np.savetxt(path, raw.reshape(-1, order="F"))
    vol = load_labels_txt(
        path,
        shape=(6, 5, 4),
        dx=0.5e-3,
        mapping=_txt_mapping,
        order="F",
        cache=False,
    )
    expected = _txt_mapping(np.nan_to_num(raw, nan=0.0))
    np.testing.assert_array_equal(vol.labels, expected)
    assert vol.dx == 0.5e-3


def test_txt_import_cache(tmp_path):
    raw = np.full((4, 4), -2.0)
    path = tmp_path / "p.txt"
    np.savetxt(path, raw.reshape(-1, order="F"))
    v1 = load_labels_txt(path, (4, 4), 1e-3, _txt_mapping, cache=True)
    assert (tmp_path / "p.txt.labels.npz").exists()
    v2 = load_labels_txt(path, (4, 4), 1e-3, _txt_mapping, cache=True)
    np.testing.assert_array_equal(v1.labels, v2.labels)


def test_npz_roundtrip(tmp_path):
    vol = LabelVolume(labels=np.arange(24, dtype=np.uint8).reshape(4, 6) % 3, dx=0.5e-3)
    vol.save_npz(tmp_path / "v.npz")
    back = LabelVolume.load_npz(tmp_path / "v.npz")
    np.testing.assert_array_equal(back.labels, vol.labels)
    assert back.dx == vol.dx and back.origin == vol.origin


@pytest.mark.parametrize("method", ["nearest", "smooth"])
def test_resample_production_ratio(method):
    """0.5 mm -> 0.3 mm (the real dataset ratio): labels preserved, interface tight."""
    lab = np.zeros((40, 40), np.uint8)
    lab[:, 20:] = 3  # straight interface at y = 10 mm
    vol = LabelVolume(labels=lab, dx=0.5e-3)
    out = vol.resample(0.3e-3, method=method)
    assert set(out.label_set) <= {0, 3}  # never invents labels
    assert out.shape[1] == pytest.approx(40 * 0.5 / 0.3, abs=1)
    # Interface position preserved within one NEW voxel.
    edge = np.argmax(out.labels[5] == 3) * 0.3e-3
    assert abs(edge - 10e-3) <= 0.31e-3
    # Round-trip content sanity: area fraction preserved to ~1%.
    frac = (out.labels == 3).mean()
    assert abs(frac - 0.5) < 0.01


def test_resample_smooth_curved_interface_beats_nearest():
    """On a disk, smooth (one-hot vote) must not do worse than nearest."""
    n = 80
    y, x = np.mgrid[0:n, 0:n]
    disk = ((x - 40) ** 2 + (y - 40) ** 2 <= 25**2).astype(np.uint8)
    vol = LabelVolume(labels=disk, dx=1e-3)
    a_true = np.pi * 25e-3**2
    errs = {}
    for m in ("nearest", "smooth"):
        out = vol.resample(0.37e-3, method=m)  # awkward non-integer factor
        a = (out.labels == 1).sum() * 0.37e-3**2
        errs[m] = abs(a - a_true) / a_true
    assert errs["smooth"] <= errs["nearest"] + 1e-3


def test_scene_with_imported_volume():
    lab = np.full((20, 20), 7, np.uint8)
    lab[5:15, 5:15] = 2
    vol = LabelVolume(labels=lab, dx=1e-3)
    grid = _grid2d(n=60)
    scene = Scene(ndim=2, background=0)
    scene.add_volume(vol, position=(0.02, 0.02), ignore=(7,))  # 7 transparent
    out = scene.rasterize(grid)
    assert out.labels[30, 30] == 2  # inside the embedded block
    assert out.labels[22, 22] == 0  # phantom background is transparent
    assert out.labels[5, 5] == 0


# ---------------- configs ----------------


def test_scene_config_json_roundtrip_and_build_equivalence():
    cfg = SceneConfig(
        ndim=2,
        background=0,
        objects=[
            {
                "shape": {
                    "kind": "difference",
                    "base": {
                        "kind": "union",
                        "children": [
                            {"kind": "ball", "center_mm": [40, 50], "radius_mm": 20},
                            {"kind": "box", "center_mm": [60, 50], "size_mm": [30, 50]},
                        ],
                    },
                    "subtract": {"kind": "ball", "center_mm": [50, 50], "radius_mm": 10},
                },
                "label": 1,
            }
        ],
    )
    again = SceneConfig.model_validate_json(cfg.model_dump_json())
    assert again == cfg

    grid = _grid2d()
    by_config = again.build().rasterize(grid)
    manual = (
        Scene(ndim=2)
        .add(
            (Ball((0.04, 0.05), 0.02) | Box((0.06, 0.05), (0.03, 0.05))) - Ball((0.05, 0.05), 0.01),
            1,
        )
        .rasterize(grid)
    )
    np.testing.assert_array_equal(by_config.labels, manual.labels)


def test_volume_import_config_references_file(tmp_path):
    lab = np.full((8, 8), 1, np.uint8)
    LabelVolume(labels=lab, dx=0.5e-3).save_npz(tmp_path / "custom.npz")
    cfg = SceneConfig(
        ndim=2,
        imports=[
            {
                "format": "npz",
                "path": str(tmp_path / "custom.npz"),
                "resample_dx_mm": 0.3,
                "resample_method": "nearest",
            }
        ],
    )
    again = SceneConfig.model_validate_json(cfg.model_dump_json())
    assert again.imports[0].path.endswith("custom.npz")  # file REFERENCE kept
    scene = again.build()
    out = scene.rasterize(Grid(shape=(20, 20), dx=0.3e-3))
    assert 1 in out.label_set


# ---------------- integration ----------------


@pytest.mark.slow
def test_scene_to_medium_to_solver_smoke():
    import caustica.solvers as solvers
    from caustica.materials import Material, MaterialDB, water
    from caustica.solvers import CWRunSpec
    from caustica.sources import plane_cw_source

    grid = Grid(shape=(96, 96), dx=0.5e-3, pml=PMLSpec(thickness=8e-3))
    scene = Scene(ndim=2, background=0)
    scene.add(Ball((0.024, 0.024), 0.008), 1)
    db = MaterialDB(
        materials={
            0: water(),
            1: Material(name="inclusion", alpha_np_m=5.0, rho=1050.0, c=1580.0, beta=0.0),
        }
    )
    med = scene.to_medium(grid, db, supersample=3)
    assert med.c_max == pytest.approx(1580.0)
    src = plane_cw_source(grid, f0=0.5e6, amplitude=1e5, axis=0, position_vox=20)
    res = solvers.get("linear")().run(
        grid,
        med,
        src,
        CWRunSpec(min_settle_periods=6, max_settle_periods=20),
        backend="numpy",
    )
    assert np.isfinite(res.amp).all() and res.amp.max() > 0


# ---------------- review-hardening regressions (2026-08-11) ----------------


def test_resample_samples_exact_physical_positions():
    """zoom(grid_mode=False) used to stretch content ~(n_out-1)/((n_in-1)*r).

    Interface truly at 18.0 mm (input i >= 36 at dx=0.5 mm) must land at
    18.0 mm after 0.5 -> 0.3 mm resampling: first output row j with
    j*0.3 >= interface-half-cell is j=60 — NOT j=61 (the old drifted answer).
    """
    labels = np.ones((40, 4), dtype=np.int32)
    labels[36:, :] = 3
    vol = LabelVolume(labels=labels, dx=0.5e-3)
    for method in ("nearest", "smooth"):
        out = vol.resample(0.3e-3, method=method)
        assert out.shape[0] == round(40 * 0.5 / 0.3)  # extent preserved
        first3 = int(np.argmax(out.labels[:, 0] == 3))
        assert first3 == 60, f"{method}: interface at {first3 * 0.3:.1f} mm, want 18.0 mm"


def test_labelvolume_eq_and_origin_preserved():
    a = LabelVolume(labels=np.ones((4, 4), np.int32), dx=1e-3, origin=(1e-3, 2e-3))
    b = LabelVolume(labels=np.ones((4, 4), np.int32), dx=1e-3, origin=(1e-3, 2e-3))
    c = LabelVolume(labels=np.zeros((4, 4), np.int32), dx=1e-3, origin=(1e-3, 2e-3))
    assert a == b  # dataclass-default __eq__ used to raise on the ndarray
    assert a != c
    assert a.resample(0.7e-3).origin == a.origin


def test_labels_txt_cache_checks_arguments(tmp_path):
    """A cache written with different args must be re-parsed, not served."""
    path = tmp_path / "vol.txt"
    np.savetxt(path, np.arange(12, dtype=float).reshape(3, 4))

    def mapping(v):
        return (v > 5).astype(np.int32)

    v1 = load_labels_txt(path, shape=(3, 4), dx=1e-3, mapping=mapping, order="C")
    assert v1.shape == (3, 4)
    v2 = load_labels_txt(path, shape=(3, 4), dx=1e-3, mapping=mapping, order="C", transpose=(1, 0))
    assert v2.shape == (4, 3)  # stale cache used to return (3, 4)
    v3 = load_labels_txt(path, shape=(3, 4), dx=2e-3, mapping=mapping, order="C")
    assert v3.dx == 2e-3  # stale cache used to return dx=1e-3
    # unchanged args do hit the cache and agree
    v4 = load_labels_txt(path, shape=(3, 4), dx=2e-3, mapping=mapping, order="C")
    assert v4 == v3


def test_add_volume_uses_volume_origin_by_default():
    """rasterize -> add_volume round-trip must preserve placement."""
    grid = Grid(shape=(20, 20), dx=1e-3)
    org = (10e-3, 10e-3)
    scene1 = Scene(ndim=2, background=0)
    scene1.add(Ball((15e-3, 15e-3), 3e-3), 1)
    vol = scene1.rasterize(grid, origin=org)
    assert vol.labels.sum() > 0

    scene2 = Scene(ndim=2, background=0)
    scene2.add_volume(vol)  # no explicit position: volume.origin must apply
    np.testing.assert_array_equal(scene2.rasterize(grid, origin=org).labels, vol.labels)


def test_axisym_supersample_mirrors_negative_radii():
    """On-axis sub-samples at r < 0 are physically the mirror points |r|."""
    grid = Grid(shape=(4, 4), dx=3e-3)
    band = Box((1.0e-3, 6e-3), (1.0e-3, 24e-3))  # radial band r in [0.5, 1.5] mm, all z
    scene = Scene(ndim=2, axisymmetric=True, background=0)
    scene.add(band, 1)
    out = scene.rasterize(grid, supersample=3)
    # r=0 row sub-samples at |-1|,0,|1| mm -> 6 of 9 votes inside the band
    np.testing.assert_array_equal(out.labels[0, :], 1)


def test_majority_tie_prefers_later_paint():
    grid = Grid(shape=(4, 4), dx=1e-3)
    scene = Scene(ndim=2, background=0)
    scene.add(HalfSpace((2.0e-3, 0.0), (1, 0)), 2)  # x <= 2 mm, painted first
    scene.add(HalfSpace((2.0e-3, 0.0), (-1, 0)), 7)  # x >= 2 mm, painted later
    out = scene.rasterize(grid, supersample=2)  # voxel x=2 mm ties 2-2
    np.testing.assert_array_equal(out.labels[2, :], 7)


def test_chunked_rasterization_identical_across_chunk_sizes():
    grid = Grid(shape=(24, 24), dx=1e-3)
    scene = Scene(ndim=2, background=0)
    scene.add(Ball((12e-3, 12e-3), 7e-3), 1)
    scene.add(Box((5e-3, 5e-3), (6e-3, 6e-3)), 2)
    big = scene.rasterize(grid, supersample=3, chunk_voxels=10**9)
    tiny = scene.rasterize(grid, supersample=3, chunk_voxels=30)
    np.testing.assert_array_equal(big.labels, tiny.labels)


def test_halfspace_and_transform_configs_roundtrip():
    """The JSON layer must cover HalfSpace and affine transforms too."""
    cfg = SceneConfig(
        ndim=2,
        background=0,
        objects=[
            {
                "shape": {
                    "kind": "transform",
                    "child": {"kind": "ellipsoid", "center_mm": (50, 50), "semiaxes_mm": (20, 8)},
                    "scale": 1.2,
                    "angle_deg": 30.0,
                    "center_mm": (50, 50),
                    "translate_mm": (5, 0),
                },
                "label": 1,
            },
            {"shape": {"kind": "halfspace", "point_mm": (80, 0), "normal": (1, 0)}, "label": 2},
        ],
    )
    rebuilt = SceneConfig.model_validate_json(cfg.model_dump_json())
    grid = _grid2d()

    from caustica.geometry import Ellipsoid as Ell

    hand = Scene(ndim=2, background=0)
    shape = (
        Ell((50e-3, 50e-3), (20e-3, 8e-3))
        .scaled(1.2, center=(50e-3, 50e-3))
        .rotated(np.deg2rad(30.0), center=(50e-3, 50e-3))
        .translated((5e-3, 0.0))
    )
    hand.add(shape, 1)
    hand.add(HalfSpace((80e-3, 0.0), (1, 0)), 2)

    np.testing.assert_array_equal(
        rebuilt.build().rasterize(grid).labels, hand.rasterize(grid).labels
    )
