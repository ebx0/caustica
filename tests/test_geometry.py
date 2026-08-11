"""M6b gates: CSG primitives/algebra, scenes, axisymmetric, import, resample."""

import numpy as np
import pytest

from hifusim import Grid, PMLSpec
from hifusim.geometry import (
    Ball,
    Box,
    Cylinder,
    Ellipsoid,
    HalfSpace,
    LabelVolume,
    Scene,
    SceneConfig,
    breast_phantom_mapping,
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


def test_txt_import_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    raw = rng.choice([-4.0, -2.0, 0.5, np.nan], size=(6, 5, 4))
    path = tmp_path / "phantom.txt"
    np.savetxt(path, raw.reshape(-1, order="F"))
    vol = load_labels_txt(
        path,
        shape=(6, 5, 4),
        dx=0.5e-3,
        mapping=breast_phantom_mapping,
        order="F",
        cache=False,
    )
    expected = breast_phantom_mapping(np.nan_to_num(raw, nan=0.0))
    np.testing.assert_array_equal(vol.labels, expected)
    assert vol.dx == 0.5e-3


def test_txt_import_cache(tmp_path):
    raw = np.full((4, 4), -2.0)
    path = tmp_path / "p.txt"
    np.savetxt(path, raw.reshape(-1, order="F"))
    v1 = load_labels_txt(path, (4, 4), 1e-3, breast_phantom_mapping, cache=True)
    assert (tmp_path / "p.txt.labels.npz").exists()
    v2 = load_labels_txt(path, (4, 4), 1e-3, breast_phantom_mapping, cache=True)
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
    import hifusim.solvers as solvers
    from hifusim.materials import Material, MaterialDB, water
    from hifusim.solvers import CWRunSpec
    from hifusim.sources import plane_cw_source

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


@pytest.mark.slow
def test_real_mtype_phantom_if_present():
    """Load + resample the actual production phantom when it exists locally."""
    import pathlib

    from hifusim.geometry import load_breast_phantom

    path = pathlib.Path("mtype.txt")
    if not path.exists():
        pytest.skip("mtype.txt not present")
    vol = load_breast_phantom(path)  # cached as mtype.txt.labels.npz after 1st run
    assert vol.shape == (253, 355, 310)  # transposed (2,1,0) from raw
    assert set(vol.label_set) <= {1, 2, 3, 4}
    sub = LabelVolume(labels=vol.labels[100:140, 150:190, 140:180], dx=vol.dx)
    out = sub.resample(0.3e-3, method="nearest")
    assert set(out.label_set) <= set(sub.label_set)
