"""Source construction gate: validation, builders, voxelization invariants."""

import numpy as np
import pytest

from caustica import Grid, PMLSpec
from caustica.sources import CWSource, bowl_cw_source, plane_cw_source, ramp_envelope


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
    """The binary shell: one voxel thick, apex on its voxel, inside the aperture."""
    g = Grid(shape=(64, 64, 80), dx=0.5e-3)
    a, roc = 6e-3, 15e-3  # 12 and 30 voxels
    apex = (32, 32, 10)
    src = bowl_cw_source(
        g,
        f0=1e6,
        amplitude=1e5,
        aperture_radius=a,
        roc=roc,
        apex_vox=apex,
        discretization="binary",
    )
    assert src.n_points > 100
    assert src.weights is None  # a binary shell drives every voxel alike
    # Unique voxels only (CWSource enforces), all inside the grid.
    src.check_inside(g)
    # Depth span ~ bowl depth h; transverse extent within the aperture.
    h_vox = (roc - np.sqrt(roc**2 - a**2)) / g.dx
    assert src.indices[:, 2].min() == apex[2]
    assert src.indices[:, 2].max() <= apex[2] + int(np.ceil(h_vox)) + 1
    r_trans = np.hypot(src.indices[:, 0] - 32, src.indices[:, 1] - 32) * g.dx
    assert r_trans.max() <= a + g.dx


def test_the_offgrid_bowl_carries_the_caps_own_area():
    """The default discretization, and the property that makes it worth having.

    A binary shell's strength is its voxel count, which for a curved surface
    is 13-25 % more than its area and does not converge as dx shrinks. The
    off-grid source carries the closed-form area instead, so the sum of its
    weights IS that area in grid squares — the quantity O'Neil integrates
    over. Everything downstream follows from that one identity.
    """
    g = Grid(shape=(64, 64, 80), dx=0.5e-3)
    a, roc = 6e-3, 15e-3
    apex = (32, 32, 10)
    src = bowl_cw_source(g, f0=1e6, amplitude=1e5, aperture_radius=a, roc=roc, apex_vox=apex)

    area = 2.0 * np.pi * roc**2 * (1.0 - np.sqrt(1.0 - (a / roc) ** 2))
    assert src.weights is not None
    assert float(src.drive_weights.sum()) == pytest.approx(area / g.dx**2, rel=1e-3)
    src.check_inside(g)
    # It is a halo, not a shell: several times the points, and signed.
    shell = bowl_cw_source(
        g,
        f0=1e6,
        amplitude=1e5,
        aperture_radius=a,
        roc=roc,
        apex_vox=apex,
        discretization="binary",
    )
    assert src.n_points > 3 * shell.n_points
    assert src.drive_weights.min() < 0.0  # the interpolant's side-lobes
    # ...and it still sits where the bowl was ordered: the drive's centre of
    # mass is on the axis, at the depth the cap's own centroid has.
    w = src.drive_weights.astype(np.float64)
    com = (src.indices.astype(np.float64) * w[:, None]).sum(axis=0) / w.sum()
    assert com[0] == pytest.approx(apex[0], abs=0.05)
    assert com[1] == pytest.approx(apex[1], abs=0.05)
    sag_vox = (roc - np.sqrt(roc**2 - a**2)) / g.dx
    assert apex[2] < com[2] < apex[2] + sag_vox


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
