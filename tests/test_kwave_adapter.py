"""M4b gates: unit conversions (always run) + live k-Wave cross-check (auto-skip).

The live test is the FIRST k-Wave cross-validation of the native solver:
same grid, same medium, same voxel source -> normalized focal patterns must
agree. It is skipped, with a visible reason, when k-wave-python or its
binary is unavailable — the suite stays green without it (M4b criterion).
"""

import numpy as np
import pytest

import hifusim.solvers as solvers
from hifusim import Grid, Medium, PMLSpec
from hifusim.materials import water
from hifusim.solvers import CWRunSpec
from hifusim.solvers.kwave_adapter import alpha_np_m_to_kwave, beta_to_bona
from hifusim.sources import CWSource

F0 = 0.5e6
C0 = 1500.0
DX = 0.5e-3  # 6 ppw at 500 kHz


def test_alpha_unit_conversion():
    # 1 Np = 20/ln(10) dB; per meter -> per cm.
    assert alpha_np_m_to_kwave(1.0) == pytest.approx(0.0868589, rel=1e-5)
    assert alpha_np_m_to_kwave(0.0) == 0.0
    np.testing.assert_allclose(
        alpha_np_m_to_kwave(np.array([10.0, 100.0])), [0.868589, 8.68589], rtol=1e-5
    )


def test_beta_to_bona_mapping():
    # hifusim beta = 1 + B/2A  =>  B/A = 2 (beta - 1); water beta=3.5 -> B/A=5.
    assert beta_to_bona(3.5) == pytest.approx(5.0)
    assert beta_to_bona(1.0) == pytest.approx(0.0)


def _disc_source(center, radius_vox, ndim=2):
    offs = np.arange(-radius_vox, radius_vox + 1)
    mesh = np.meshgrid(*([offs] * ndim), indexing="ij")
    mask = sum(m**2 for m in mesh) <= radius_vox**2
    pts = np.stack([m[mask] for m in mesh], axis=1) + np.asarray(center)
    return CWSource(
        indices=pts.astype(np.int64),
        phases=np.zeros(pts.shape[0], np.float32),
        amplitude=1.0e5,
        f0=F0,
        label="disc",
    )


@pytest.mark.kwave
@pytest.mark.slow
def test_kwave_vs_linear_2d_water():
    pytest.importorskip("kwave", reason="k-wave-python not installed")
    # 20-voxel sponge, mirroring k-Wave's default 20-voxel inner PML.
    grid = Grid(shape=(96, 96), dx=DX, pml=PMLSpec(thickness=10e-3))
    medium = Medium.homogeneous(grid.shape, water(c=C0))
    source = _disc_source(center=(30, 48), radius_vox=3)
    spec = CWRunSpec(min_settle_periods=10, n_record_periods=2)

    res_lin = solvers.get("linear")().run(grid, medium, source, spec, backend="numpy")
    try:
        res_kw = solvers.get("kwave")().run(grid, medium, source, spec)
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        pytest.skip(f"k-Wave binary unavailable on this machine: {exc}")

    # Compare normalized |phasor| away from PMLs (k-Wave PML ~20 vox inside).
    inner = (slice(26, 70), slice(26, 70))
    a = res_lin.amp[inner]
    b = res_kw.amp[inner]
    a = a / a.max()
    b = b / b.max()

    # Peak location must agree (also catches Fortran-order mapping bugs).
    pa = np.unravel_index(np.argmax(a), a.shape)
    pb = np.unravel_index(np.argmax(b), b.shape)
    assert abs(pa[0] - pb[0]) <= 1 and abs(pa[1] - pb[1]) <= 1

    r = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    assert r > 0.99, f"linear-vs-kwave field correlation r={r:.5f} < 0.99"

    assert res_kw.meta["backend"] == "kwave-omp"
