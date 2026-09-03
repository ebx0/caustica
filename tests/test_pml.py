"""M1 gate: sponge PML spec + profile (port-parity with the notebook)."""

import numpy as np
import pytest

from caustica.core.pml import PMLSpec, sponge_profile_1d


def test_spec_thickness_vox_rounding():
    # Notebook value: 5.0 mm at dx=0.30 mm -> 17 voxels.
    assert PMLSpec(thickness=5.0e-3).thickness_vox(0.30e-3) == 17
    assert PMLSpec(thickness=0.0).thickness_vox(1e-3) == 0


def test_spec_validation():
    with pytest.raises(ValueError):
        PMLSpec(thickness=-1e-3)
    with pytest.raises(ValueError):
        PMLSpec(thickness=1e-3, edge=0.0)
    with pytest.raises(ValueError):
        PMLSpec(thickness=1e-3).thickness_vox(0.0)


def test_profile_interior_is_one_edges_damp():
    n, w, edge = 64, 10, 2.0
    s = sponge_profile_1d(n, w, edge)
    assert s.dtype == np.float32
    np.testing.assert_array_equal(s[w:-w], 1.0)
    # Outermost cell damps hardest: exp(-edge).
    assert s[0] == pytest.approx(np.exp(-edge), rel=1e-6)
    assert s[-1] == pytest.approx(np.exp(-edge), rel=1e-6)
    # Monotone recovery toward the interior, symmetric ends.
    assert np.all(np.diff(s[:w]) > 0)
    np.testing.assert_allclose(s[:w], s[-w:][::-1], rtol=0, atol=0)
    assert np.all(s > 0) and np.all(s <= 1.0)


def test_profile_zero_width_is_all_ones():
    np.testing.assert_array_equal(sponge_profile_1d(32, 0), 1.0)


def test_profile_validation():
    with pytest.raises(ValueError):
        sponge_profile_1d(0, 0)
    with pytest.raises(ValueError):
        sponge_profile_1d(16, -1)
    with pytest.raises(ValueError, match="does not fit"):
        sponge_profile_1d(16, 9)


def test_missing_pml_warns_about_periodic_wraparound(caplog):
    """The periodic-boundary footgun (found 2026-08-10) must stay loud."""
    import numpy as np

    from caustica import Grid, Medium
    from caustica.materials import water
    from caustica.solvers import CWRunSpec, get
    from caustica.sources import CWSource

    grid = Grid(shape=(48,), dx=0.5e-3, pml=None)
    medium = Medium.homogeneous(grid.shape, water())
    src = CWSource(
        indices=np.array([[8]], dtype=np.int32),
        phases=np.zeros(1, dtype=np.float32),
        amplitude=1e4,
        f0=1e6,
    )
    with caplog.at_level("WARNING", logger="caustica"):
        get("linear")().run(
            grid, medium, src, CWRunSpec(min_settle_periods=1, max_settle_periods=2)
        )
    assert any("PERIODIC" in r.message for r in caplog.records)
