"""M5 gates: Westervelt nonlinearity vs Fubini + linear-limit equivalence.

Resolution note (calibrated empirically, see devlog 2026-08-10 session 5):
harmonic-cascade accuracy needs headroom above f0 — at 8 ppw the 3rd
harmonic sits at 2.67 ppw and aliases, inflating A2/A1 by ~10%. At 16 ppw
(2f0 at 8 ppw, 3f0 at 5.3 ppw) A2/A1 lands within 1-3% of Fubini across
sigma 0.06-0.61. The production dx rule (2f0 above Nyquist) bounds p_max
capture, not harmonic-ratio physics — this gate pins the physics.
"""

import numpy as np
import pytest

import caustica.solvers as solvers
from caustica import Grid, Medium, PMLSpec
from caustica.analytic import fubini_harmonic, shock_distance
from caustica.materials import water
from caustica.solvers import CWRunSpec
from caustica.sources import plane_cw_source

F0, C0, BETA = 1.0e6, 1500.0, 3.5


def test_beta_zero_is_bitwise_identical_to_linear():
    # Same engine, same code path when the medium is linear: not just close —
    # identical.
    dx = C0 / (F0 * 4.0)
    grid = Grid(shape=(128,), dx=dx, pml=PMLSpec(thickness=16 * dx))
    med = Medium.homogeneous(grid.shape, water(c=C0))  # beta = 0
    src = plane_cw_source(grid, f0=F0, amplitude=1e5, position_vox=24)
    spec = CWRunSpec(min_settle_periods=20, max_settle_periods=40)
    r_lin = solvers.get("linear")().run(grid, med, src, spec, backend="numpy")
    r_wes = solvers.get("westervelt")().run(grid, med, src, spec, backend="numpy")
    np.testing.assert_array_equal(r_lin.phasor, r_wes.phasor)
    np.testing.assert_array_equal(r_lin.p_max, r_wes.p_max)
    assert r_wes.meta["nonlinear_active"] is False


@pytest.fixture(scope="module")
def fubini_run():
    ppw = 16.0
    dx = C0 / (F0 * ppw)
    grid = Grid(shape=(600,), dx=dx, pml=PMLSpec(thickness=40 * dx))
    med = Medium.homogeneous(grid.shape, water(c=C0, beta=BETA))
    src = plane_cw_source(grid, f0=F0, amplitude=2.0e6, position_vox=60)
    spec = CWRunSpec(min_settle_periods=45, max_settle_periods=100, convergence_tol=0.003)
    res = solvers.get("westervelt")().run(
        grid, med, src, spec, backend="numpy", harmonics=(1, 2, 3)
    )
    return res, dx, 60


def _sigma_axis(res, dx, src_vox):
    a1 = res.harmonic_amp(1)
    i0 = 90  # near-source reference; sigma there is ~0.11 -> B1 correction ~0.2%
    sig0_guess = ((i0 - src_vox) * dx) / shock_distance(a1[i0], F0, C0, beta=BETA)
    p0_eff = a1[i0] / fubini_harmonic(1, sig0_guess)
    x_sh = shock_distance(p0_eff, F0, C0, beta=BETA)
    return a1, x_sh


@pytest.mark.slow
def test_fubini_second_harmonic_within_5_percent(fubini_run):
    res, dx, src_vox = fubini_run
    a1, x_sh = _sigma_axis(res, dx, src_vox)
    a2 = res.harmonic_amp(2)
    assert res.meta["nonlinear_active"] is True
    checked = 0
    for iv in (150, 250, 350, 450, 540):
        sigma = ((iv - src_vox) * dx) / x_sh
        if not 0.05 <= sigma <= 0.95:
            continue
        measured = a2[iv] / a1[iv]
        analytic = fubini_harmonic(2, sigma) / fubini_harmonic(1, sigma)
        rel = abs(measured - analytic) / analytic
        assert rel < 0.05, f"A2/A1 at sigma={sigma:.3f}: {rel * 100:.2f}% >= 5%"
        checked += 1
    assert checked >= 4  # the gate must actually exercise a sigma range


@pytest.mark.slow
def test_fubini_third_harmonic_reasonable(fubini_run):
    # Secondary (informative) gate: A3 is tiny and resolution-hungrier; 10%.
    res, dx, src_vox = fubini_run
    a1, x_sh = _sigma_axis(res, dx, src_vox)
    a3 = res.harmonic_amp(3)
    for iv in (350, 450, 540):
        sigma = ((iv - src_vox) * dx) / x_sh
        measured = a3[iv] / a1[iv]
        analytic = fubini_harmonic(3, sigma) / fubini_harmonic(1, sigma)
        rel = abs(measured - analytic) / analytic
        assert rel < 0.10, f"A3/A1 at sigma={sigma:.3f}: {rel * 100:.1f}% >= 10%"


@pytest.mark.slow
def test_amp_pmax_ceiling_invariant_nonlinear(fubini_run):
    # amp(f0) can never exceed the discrete-sampled time peak by more than
    # the cos(pi/spp) sampling ceiling — anywhere in the >10% band.
    res, _dx, _src = fubini_run
    band = res.p_max > 0.10 * res.p_max.max()
    ratio = res.amp[band] / res.p_max[band]
    ceiling = 1.0 / np.cos(np.pi / res.spp)
    assert ratio.max() <= ceiling * (1 + 1e-3)
    # Mild-nonlinearity sanity: near the source (sigma~0.1) the f0 fraction
    # of the crest is still >= 0.85 (notebook's production band).
    assert 0.85 <= res.amp[90] / res.p_max[90] <= 1.0 + 1e-3


def test_harmonics_api_contract():
    dx = C0 / (F0 * 4.0)
    grid = Grid(shape=(128,), dx=dx, pml=PMLSpec(thickness=16 * dx))
    med = Medium.homogeneous(grid.shape, water(c=C0))
    src = plane_cw_source(grid, f0=F0, amplitude=1e5, position_vox=24)
    spec = CWRunSpec(min_settle_periods=10, max_settle_periods=30)
    res = solvers.get("westervelt")().run(grid, med, src, spec, backend="numpy", harmonics=(1, 2))
    assert set(res.phasors) == {1, 2}
    np.testing.assert_array_equal(res.phasors[1], res.phasor)
    with pytest.raises(KeyError, match="harmonic 3"):
        res.harmonic_amp(3)
    with pytest.raises(ValueError, match="fundamental"):
        solvers.get("westervelt")().run(grid, med, src, spec, backend="numpy", harmonics=(2,))
