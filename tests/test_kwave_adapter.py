"""Unit conversions (always run) + live k-Wave cross-check (auto-skip).

The live test is the FIRST k-Wave cross-validation of the native solver:
same grid, same medium, same voxel source -> normalized focal patterns must
agree. It is skipped, with a visible reason, when k-wave-python or its
binary is unavailable — the suite stays green without it.
"""

import contextlib
import warnings

import numpy as np
import pytest

import caustica.solvers as solvers
from caustica import Grid, Medium, PMLSpec
from caustica.core.backend import CausticaWarning
from caustica.materials import water
from caustica.solvers import CWRunSpec
from caustica.solvers.kwave_adapter import alpha_np_m_to_kwave, beta_to_bona
from caustica.sources import CWSource, bowl_cw_source

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
    # caustica beta = 1 + B/2A  =>  B/A = 2 (beta - 1); water beta=3.5 -> B/A=5.
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


@pytest.mark.kwave
def test_the_sensor_records_the_region_and_not_the_whole_grid():
    """k-Wave dumps every recorded step to disk; the region decides how much.

    The binary cannot accumulate a DFT, so a run writes ``n_sensor_points x
    record_steps`` floats to HDF5 and the adapter transforms afterwards. At
    the ITRUSST benchmark size that was a 731 MB input file and a 1.6 GB
    output file per run (measured 2026-08-25), and recording the whole grid
    when the caller asked for a slab paid all of it for nothing. The field
    over the region has to come back unchanged either way.
    """
    grid = Grid(shape=(40, 40, 56), dx=0.5e-3, pml=PMLSpec(thickness=3e-3))
    medium = Medium.homogeneous(grid.shape, water())
    src = bowl_cw_source(
        grid, f0=1e6, amplitude=1e5, aperture_radius=4e-3, roc=10e-3, apex_vox=(20, 20, 8)
    )
    spec = CWRunSpec(min_settle_periods=2, max_settle_periods=6, n_record_periods=2)
    region = (slice(10, 30), slice(10, 30), slice(20, 44))

    whole = solvers.get("kwave")().run(grid, medium, src, spec)
    part = solvers.get("kwave")().run(grid, medium, src, spec, record_region=region)

    assert whole.phasor.shape == grid.shape
    assert part.phasor.shape == tuple(sl.stop - sl.start for sl in region)
    np.testing.assert_allclose(
        np.abs(part.phasor), np.abs(whole.phasor[region]), rtol=1e-5, atol=1.0
    )


class _Captured(Exception):
    """Raised once the medium has been built, to stop before the solver runs."""


@pytest.mark.kwave
def test_the_adapter_asks_kwave_for_the_absorption_law_the_engine_implements():
    """``alpha_power = 0`` is a decision, and an expensive one to reverse quietly.

    This library absorbs at a rate that does not depend on frequency, so the
    adapter tells k-Wave to do the same. That makes the cross-check a
    comparison of two implementations of one model rather than of two models,
    which is the only thing a cross-check can honestly be.

    It also means neither code is right about tissue, which absorbs roughly
    as ``f^1.1``: both under-absorb 2f0 together, by a factor near ``2^1.1``,
    and no amount of agreement between them will say so. Raising the exponent
    HERE alone would not fix that -- it would only hide it, by making the two
    codes disagree at the harmonic for a reason that looks like a numerics
    error. `scripts/dev_nonlinear.py` N5 measures the gap; closing it means
    changing the engine, and this test is what makes that a deliberate act.
    """
    km = pytest.importorskip("kwave.kmedium")
    seen: dict = {}
    original = km.kWaveMedium

    class Capture(original):
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)
            raise _Captured

    dx = C0 / (F0 * 6.0)
    grid = Grid(shape=(48, 48), dx=dx, pml=PMLSpec(thickness=6 * dx))
    med = Medium.homogeneous(grid.shape, water(c=C0, alpha_np_m=5.0, beta=3.5))
    src = _disc_source((24, 8), 3, ndim=2)

    km.kWaveMedium = Capture
    try:
        with pytest.raises(_Captured):
            solvers.get("kwave")().run(grid, med, src, CWRunSpec(min_settle_periods=2))
    finally:
        km.kWaveMedium = original

    assert seen["alpha_power"] == 0.0
    # The coefficient is then a plain dB/cm at every frequency, which is what
    # alpha_power = 0 makes it mean.
    np.testing.assert_allclose(seen["alpha_coeff"], alpha_np_m_to_kwave(5.0), rtol=1e-9)
    np.testing.assert_allclose(seen["sound_speed"], C0)


def test_asking_kwave_for_a_harmonic_at_the_default_settle_is_flagged():
    """A fixed schedule cannot know when a harmonic has stopped moving.

    The native engine grades every requested harmonic against its own
    amplitude and settles until each one stops changing. k-Wave's binary runs
    to a step count decided before it starts, so the caller has to pick that
    count -- and the default was chosen for a fundamental. Measured on a
    focused bowl in water with beta = 0, where the true 2f0 is zero: a settle
    short enough to satisfy the peak alone left 2.1 % of the fundamental in
    the harmonic channel.

    The warning fires before any binary runs, so this needs neither k-Wave nor
    a GPU -- which is the point of raising it where the schedule is computed.
    """
    grid = Grid(shape=(48, 48), dx=DX, pml=PMLSpec(thickness=6 * DX))
    med = Medium.homogeneous(grid.shape, water(c=C0, beta=3.5))
    src = _disc_source((24, 8), 3, ndim=2)

    with pytest.warns(CausticaWarning, match="fixed schedule"):
        with contextlib.suppress(Exception):
            solvers.get("kwave")().run(grid, med, src, CWRunSpec(), harmonics=(1, 2))

    # Ask for the fundamental alone, or settle deliberately, and it stays quiet.
    for spec, harmonics in ((CWRunSpec(), (1,)), (CWRunSpec(min_settle_periods=30), (1, 2))):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with contextlib.suppress(Exception):
                solvers.get("kwave")().run(grid, med, src, spec, harmonics=harmonics)
        assert not [w for w in caught if "fixed schedule" in str(w.message)]
