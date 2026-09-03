"""M4 physics gates on 1-D plane waves: dispersion, absorption, PML, stability.

All runs use the numpy backend on small 1-D grids — seconds, not minutes —
and each asserts one isolated physical property.
"""

import numpy as np
import pytest

import caustica.solvers as solvers
from caustica import Grid, Medium, PMLSpec
from caustica.materials import water
from caustica.solvers import CWRunSpec
from caustica.sources import plane_cw_source

F0 = 1.0e6
C0 = 1500.0
PPW = 4.0
DX = C0 / (F0 * PPW)  # 0.375 mm -> lambda = 4 voxels
K0 = 2.0 * np.pi * F0 / C0


def run_1d(
    alpha=0.0,
    n=256,
    pml_vox=24,
    src_vox=40,
    min_settle=40,
    max_settle=90,
    tol=0.005,
    record=2,
):
    grid = Grid(shape=(n,), dx=DX, pml=PMLSpec(thickness=pml_vox * DX))
    medium = Medium.homogeneous(grid.shape, water(alpha_np_m=alpha, c=C0))
    source = plane_cw_source(grid, f0=F0, amplitude=1.0e5, position_vox=src_vox)
    spec = CWRunSpec(
        min_settle_periods=min_settle,
        max_settle_periods=max_settle,
        convergence_tol=tol,
        n_record_periods=record,
    )
    solver = solvers.get("linear")()
    return solver.run(grid, medium, source, spec, backend="numpy")


@pytest.fixture(scope="module")
def lossless():
    return run_1d(alpha=0.0)


SPAN = slice(80, 200)  # measurement window: 30 wavelengths, PML/source excluded


def test_exact_period_dt_policy(lossless):
    # ppw/cfl = 4/0.48 -> floor 8 steps per period, effective CFL 0.5.
    assert lossless.spp == 8
    assert lossless.dt == pytest.approx(1.0 / F0 / 8, rel=1e-12)
    assert lossless.dt * C0 / DX == pytest.approx(0.5, rel=1e-9)


def test_plane_wave_dispersion_below_0p1_percent(lossless):
    # Library convention (2026-08-11): p(t)=Re{P e^{-iwt}}, so right of the
    # source the outgoing +x wave is P(x) ~ exp(+i k x).
    ph = lossless.phasor[SPAN]
    dphi = np.angle(ph[1:] * np.conj(ph[:-1]))
    k_num = dphi.mean() / DX
    rel_err = abs(k_num - K0) / K0
    assert rel_err < 1e-3, f"phase-speed error {rel_err * 100:.4f}% >= 0.1%"


def test_phasor_consistent_with_time_peak(lossless):
    # amp <= p_max / cos(pi/spp) (discrete-sampling ceiling, notebook contract)
    amp = lossless.amp[SPAN]
    pmax = lossless.p_max[SPAN]
    ratio = amp / pmax
    ceiling = 1.0 / np.cos(np.pi / lossless.spp)
    assert ratio.max() <= ceiling * (1 + 1e-3)
    assert ratio.min() > 0.90


def test_absorption_matches_configured_alpha_within_1_percent():
    alpha = 30.0  # Np/m -> e^{-1.35} over the 45 mm window
    res = run_1d(alpha=alpha, pml_vox=32, n=288, min_settle=45, max_settle=95)
    x = np.arange(*SPAN.indices(288)) * DX
    lnamp = np.log(res.amp[SPAN])
    slope = np.polyfit(x, lnamp, 1)[0]
    rel_err = abs(-slope - alpha) / alpha
    assert rel_err < 0.01, f"measured alpha off by {rel_err * 100:.2f}%"


def test_pml_reflection_below_3_percent(lossless):
    # A reflection R makes a standing ripple: (max-min)/(max+min) ~ |R|.
    amp = lossless.amp[SPAN]
    ripple = (amp.max() - amp.min()) / (amp.max() + amp.min())
    assert ripple < 0.03, f"PML ripple {ripple * 100:.2f}% >= 3%"


@pytest.mark.slow
def test_long_run_stability_no_energy_growth():
    # 200 settle periods with convergence disabled: peak must not drift.
    res = run_1d(alpha=0.0, min_settle=200, max_settle=200, tol=0.0)
    hist = np.array([peak for (_p, peak, _r) in res.convergence_history])
    periods = np.array([p for (p, _peak, _r) in res.convergence_history])
    late = hist[periods >= periods.max() - 100]
    drift = (late.max() - late.min()) / late.mean()
    assert res.settle_capped
    assert drift < 0.01, f"late-time peak drift {drift * 100:.2f}% >= 1%"


def test_convergence_metadata_is_honest(lossless):
    assert lossless.tof_periods >= 50  # 80.6 mm to the far corner at 1500 m/s
    assert lossless.converged_period >= lossless.tof_periods
    assert not lossless.settle_capped
    assert lossless.steps_total == pytest.approx(
        (lossless.converged_period + 2) * lossless.spp, abs=0
    )
    assert lossless.meta["backend"] == "numpy"
