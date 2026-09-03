"""M5 gates: Westervelt nonlinearity vs Fubini + linear-limit equivalence.

Resolution note (calibrated empirically, 2026-08-10):
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
from caustica.sources import bowl_cw_source, plane_cw_source

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


# --------------------------------------------------------------------------
# the same nonlinear term, in the geometry it is actually used in
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_second_harmonic_grows_as_the_square_of_the_first_in_a_focused_beam():
    """Fubini is exact in one dimension; a focused beam has no such series.

    What survives the geometry is the SCALING. While 2f0 is a small
    perturbation fed by f0, doubling the drive quadruples it, whatever the
    beam is doing in between -- so a log-log slope of 2 is a prediction with
    no free parameters, available in exactly the geometry the library is
    used in and the one Fubini cannot follow.

    It fails loudly for the errors worth catching: a Westervelt term carrying
    the wrong power of pressure gives a slope of 1 or 3, and a term scaled by
    the wrong constant shows up here as soon as it stops being a pure
    multiplier. The 8 % band is set by the coarse grid this runs on -- at
    6 points per wavelength the harmonic sits at 3, which is enough to grade
    a slope and not enough to grade an amplitude.
    """
    ppw, aperture, roc = 6.0, 5.0e-3, 10.0e-3
    dx = C0 / (F0 * ppw)
    pml_mm, margin = 3.0e-3, 4
    pml_vox = int(round(pml_mm / dx))
    n_xy = 2 * (int(np.ceil(aperture / dx)) + pml_vox + margin) + 1
    apex_z = pml_vox + margin
    n_z = apex_z + int(round(1.5 * roc / dx)) + pml_vox + margin
    grid = Grid(shape=(n_xy, n_xy, n_z), dx=dx, pml=PMLSpec(thickness=pml_mm))
    med = Medium.homogeneous(grid.shape, water(c=C0, beta=BETA))
    apex = (n_xy // 2, n_xy // 2, apex_z)
    focus = (apex[0], apex[1], apex_z + int(round(roc / dx)))
    spec = CWRunSpec(min_settle_periods=10, max_settle_periods=40, n_record_periods=2)

    p1, p2 = [], []
    for drive in (5.0e4, 1.0e5, 2.0e5):
        src = bowl_cw_source(grid, F0, drive, aperture, roc, apex)
        res = solvers.get("westervelt")().run(
            grid, med, src, spec, reference_point=focus, harmonics=(1, 2)
        )
        # The on-axis maximum, not the geometric focus: nonlinearity moves
        # the peak, and a fixed sample would read the shift as a slope.
        lobe = slice(apex_z + int(round(0.4 * roc / dx)), n_z - pml_vox - 2)
        p1.append(float(res.harmonic_amp(1)[apex[0], apex[1], lobe].max()))
        p2.append(float(res.harmonic_amp(2)[apex[0], apex[1], lobe].max()))

    slope = float(np.polyfit(np.log(p1), np.log(p2), 1)[0])
    assert slope == pytest.approx(2.0, abs=0.08), f"quasi-linear slope {slope:.3f}, expected 2"
    assert p2[-1] / p1[-1] > 5e-3, "no harmonic was generated; the slope fit means nothing"


def test_settling_waits_for_the_harmonic_and_not_only_for_the_peak():
    """A linear medium has no second harmonic. A short settle reports one anyway.

    The settle test grades the peak of the total field, which the fundamental
    dominates. A residual transient worth one part in a hundred of that peak
    is worth tens of parts in a hundred of a second harmonic that is itself a
    few percent of the peak -- so a run could pass the peak test with a 2f0
    channel made mostly of transient, and nothing in the suite would say so.

    What sent this looking was a focused bowl at 1 MHz and 12 points per
    wavelength, in water with beta = 0 where the true answer is exactly zero:
    the peak test alone stopped at period 27 and the 2f0 channel read 2.1 %
    of the fundamental. Grading the harmonic against its own amplitude took
    the same run to period 35 and 0.003 %.

    That geometry is far too big for a test, so the bar here comes from this
    one. The peak test alone stops this run at period 16 and leaves 3.9e-4;
    asking for the harmonic carries it to 49 and 8.2e-6. The 1e-4 bar sits
    between them with a factor of four above and twelve below, so it fails if
    the harmonic criterion is removed and does not fail on drift.

    Two claims, because either alone is weak. The physical one: what comes
    back from a linear medium is not a harmonic. The structural one: asking
    for a harmonic can only lengthen the settle, never shorten it -- which is
    what makes the first claim hold for a reason rather than by luck.
    """
    dx = C0 / (F0 * 8.0)
    grid = Grid(shape=(72, 72), dx=dx, pml=PMLSpec(thickness=8 * dx))
    med = Medium.homogeneous(grid.shape, water(c=C0))  # beta = 0: no 2f0 exists
    src = plane_cw_source(grid, f0=F0, amplitude=2e5, axis=0, position_vox=12)
    spec = CWRunSpec(min_settle_periods=1, max_settle_periods=120, convergence_tol=0.01)

    alone = solvers.get("linear")().run(grid, med, src, spec, backend="numpy", harmonics=(1,))
    with_h2 = solvers.get("linear")().run(grid, med, src, spec, backend="numpy", harmonics=(1, 2))

    assert with_h2.converged_period > alone.converged_period
    assert not with_h2.settle_capped, "the settle hit its cap; the criterion never fired"
    leak = float(np.abs(with_h2.phasors[2]).max() / np.abs(with_h2.phasors[1]).max())
    assert leak < 1e-4, f"a linear medium returned 2f0 at {leak:.3e} of the fundamental"
