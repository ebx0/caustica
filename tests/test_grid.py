"""M1 gate: Grid geometry and k-space definitions."""

import numpy as np
import pytest

from caustica import Grid, PMLSpec


def test_k_axis_matches_fftfreq_analytics():
    g = Grid(shape=(32, 48, 64), dx=0.5e-3)
    for ax in range(3):
        expected = 2.0 * np.pi * np.fft.fftfreq(g.shape[ax], d=g.dx)
        np.testing.assert_allclose(g.k_axis(ax), expected, rtol=1e-12, atol=0.0)
    expected_r = 2.0 * np.pi * np.fft.rfftfreq(64, d=g.dx)
    np.testing.assert_allclose(g.k_axis_r(2), expected_r, rtol=1e-12, atol=0.0)


def test_k_vectors_broadcast_to_rfft_layout():
    g = Grid(shape=(8, 12, 16), dx=1e-3)
    ks = g.k_vectors(rfft_last=True)
    assert ks[0].shape == (8, 1, 1)
    assert ks[1].shape == (1, 12, 1)
    assert ks[2].shape == (1, 1, 16 // 2 + 1)
    k2 = g.k_squared(rfft_last=True)
    assert k2.shape == (8, 12, 9)
    assert k2.min() == 0.0
    # |k|^2 at the corner equals sum of per-axis Nyquist-bin squares.
    manual = g.k_axis(0)[4] ** 2 + g.k_axis(1)[6] ** 2 + g.k_axis_r(2)[8] ** 2
    np.testing.assert_allclose(k2[4, 6, 8], manual, rtol=1e-12)


def test_ppw_reproduces_dataset_design_numbers():
    # Notebook design rule: dx=0.30 mm, f0=1.1 MHz, c_min=1450 -> 4.39 ppw,
    # and 2.20 ppw at the second harmonic.
    g = Grid(shape=(16, 16, 16), dx=0.30e-3)
    assert g.ppw(1.1e6, 1450.0) == pytest.approx(4.394, abs=1e-3)
    assert g.ppw(2.2e6, 1450.0) == pytest.approx(2.197, abs=1e-3)


def test_geometry_helpers():
    g = Grid(shape=(10, 10), dx=1e-3, pml=PMLSpec(thickness=5e-3))
    assert g.ndim == 2
    assert g.extent == (0.01, 0.01)
    assert g.n_voxels == 100
    assert g.pml_vox == 5
    assert g.k_max == pytest.approx(np.pi / 1e-3)
    x = g.axis_coords(0, centered=True)
    assert x[10 // 2] == 0.0
    x0 = g.axis_coords(0, centered=False)
    assert x0[0] == 0.0 and x0[-1] == pytest.approx(9e-3)
    assert g.voxels_for(3.2e-3) == 3


def test_grid_validation():
    with pytest.raises(ValueError, match="1..3 dimensions"):
        Grid(shape=(4, 4, 4, 4), dx=1e-3)
    with pytest.raises(ValueError, match=">= 4 voxels"):
        Grid(shape=(3, 8), dx=1e-3)
    with pytest.raises(ValueError, match="dx must be"):
        Grid(shape=(8, 8), dx=-1.0)
    with pytest.raises(ValueError):
        Grid(shape=(8, 8), dx=1e-3).ppw(0.0, 1500.0)
