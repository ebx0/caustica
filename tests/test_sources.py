"""Source construction gate: validation, builders, voxelization invariants."""

import numpy as np
import pytest

from hifusim import Grid, PMLSpec
from hifusim.sources import CWSource, bowl_cw_source, plane_cw_source, ramp_envelope


def test_cwsource_validation():
    idx = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="phases"):
        CWSource(indices=idx, phases=np.zeros(3), amplitude=1.0, f0=1e6)
    with pytest.raises(TypeError, match="integer"):
        CWSource(indices=idx.astype(float), phases=np.zeros(2), amplitude=1.0, f0=1e6)
    with pytest.raises(ValueError, match="duplicate"):
        CWSource(indices=np.array([[1, 2], [1, 2]]), phases=np.zeros(2), amplitude=1.0, f0=1e6)
    with pytest.raises(ValueError):
        CWSource(indices=idx, phases=np.zeros(2), amplitude=-1.0, f0=1e6)


def test_check_inside():
    g = Grid(shape=(16, 16), dx=1e-3)
    src = CWSource(indices=np.array([[0, 0], [15, 15]]), phases=np.zeros(2), amplitude=1.0, f0=1e6)
    src.check_inside(g)  # boundary voxels are inside
    bad = CWSource(indices=np.array([[16, 0]]), phases=np.zeros(1), amplitude=1.0, f0=1e6)
    with pytest.raises(ValueError, match="outside grid"):
        bad.check_inside(g)
    with pytest.raises(ValueError, match="2-D"):
        src.check_inside(Grid(shape=(8, 8, 8), dx=1e-3))


def test_ramp_envelope_shape():
    period = 1e-6
    assert ramp_envelope(0.0, period, 3.0) == 0.0
    assert ramp_envelope(1.5 * period, period, 3.0) == pytest.approx(0.5)
    assert ramp_envelope(3.0 * period, period, 3.0) == pytest.approx(1.0)
    assert ramp_envelope(10.0 * period, period, 3.0) == 1.0  # clamped


def test_plane_source_2d_covers_full_plane():
    g = Grid(shape=(32, 24), dx=0.5e-3, pml=PMLSpec(thickness=4e-3))  # pml 8 vox
    src = plane_cw_source(g, f0=1e6, amplitude=1e5, axis=0)
    assert src.n_points == 24
    assert np.all(src.indices[:, 0] == g.pml_vox + 8)
    np.testing.assert_array_equal(np.sort(src.indices[:, 1]), np.arange(24))


def test_plane_source_1d_is_single_voxel():
    g = Grid(shape=(64,), dx=0.5e-3)
    src = plane_cw_source(g, f0=1e6, amplitude=1e5, position_vox=20)
    assert src.n_points == 1
    assert src.indices.tolist() == [[20]]


def test_bowl_source_voxelization_invariants():
    g = Grid(shape=(64, 64, 80), dx=0.5e-3)
    a, roc = 6e-3, 15e-3  # 12 and 30 voxels
    apex = (32, 32, 10)
    src = bowl_cw_source(g, f0=1e6, amplitude=1e5, aperture_radius=a, roc=roc, apex_vox=apex)
    assert src.n_points > 100
    # Unique voxels only (CWSource enforces), all inside the grid.
    src.check_inside(g)
    # Depth span ~ bowl depth h; transverse extent within the aperture.
    h_vox = (roc - np.sqrt(roc**2 - a**2)) / g.dx
    assert src.indices[:, 2].min() == apex[2]
    assert src.indices[:, 2].max() <= apex[2] + int(np.ceil(h_vox)) + 1
    r_trans = np.hypot(src.indices[:, 0] - 32, src.indices[:, 1] - 32) * g.dx
    assert r_trans.max() <= a + g.dx


def test_bowl_source_requires_3d():
    with pytest.raises(ValueError, match="3-D"):
        bowl_cw_source(
            Grid(shape=(32, 32), dx=1e-3),
            f0=1e6,
            amplitude=1.0,
            aperture_radius=5e-3,
            roc=10e-3,
            apex_vox=(16, 16),
        )
