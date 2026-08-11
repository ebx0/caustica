"""Gates added from the 2026-08-11 adversarial review round.

Each test pins a specific defect class the review showed the previous suite
would have missed: padded-array record regions, harmonic aliasing, unit
mistakes in reference_point, phase-map collisions, untested k-Wave phase
reordering and medium mapping.
"""

import numpy as np
import pytest

import hifusim.solvers as solvers
from hifusim import Grid, Medium, PMLSpec
from hifusim.arrays import archimedean_spiral, build_phase_maps
from hifusim.materials import Material, MaterialDB, water
from hifusim.solvers import CWRunSpec
from hifusim.solvers.kspace.engine import normalize_record_region
from hifusim.sources import CWSource, plane_cw_source

C0 = 1500.0


def test_record_region_resolves_against_active_not_padded_shape():
    # 97 pads to a 5-smooth 100: slice(None) must yield 97 voxels, not 100.
    dx = C0 / (1e6 * 4.0)
    grid = Grid(shape=(97,), dx=dx, pml=PMLSpec(thickness=12 * dx))
    med = Medium.homogeneous(grid.shape, water(c=C0))
    src = plane_cw_source(grid, f0=1e6, amplitude=1e5, position_vox=20)
    spec = CWRunSpec(min_settle_periods=4, max_settle_periods=10)
    res = solvers.get("linear")().run(
        grid, med, src, spec, backend="numpy", record_region=(slice(None),)
    )
    assert res.phasor.shape == (97,)
    assert res.region == (slice(0, 97),)


def test_normalize_record_region_contract():
    assert normalize_record_region((slice(None),), (50,)) == (slice(0, 50),)
    assert normalize_record_region((slice(-10, None),), (50,)) == (slice(40, 50),)
    with pytest.raises(ValueError, match="strides"):
        normalize_record_region((slice(0, 50, 2),), (50,))
    with pytest.raises(ValueError, match="empty"):
        normalize_record_region((slice(30, 10),), (50,))
    with pytest.raises(ValueError, match="rank"):
        normalize_record_region((slice(None),), (50, 50))


def test_harmonics_above_temporal_nyquist_rejected():
    # 4 ppw -> spp=8; harmonic 4 sits AT Nyquist and must be refused.
    dx = C0 / (1e6 * 4.0)
    grid = Grid(shape=(128,), dx=dx, pml=PMLSpec(thickness=16 * dx))
    med = Medium.homogeneous(grid.shape, water(c=C0))
    src = plane_cw_source(grid, f0=1e6, amplitude=1e5, position_vox=24)
    with pytest.raises(ValueError, match="Nyquist"):
        solvers.get("linear")().run(grid, med, src, CWRunSpec(), backend="numpy", harmonics=(1, 4))


def test_reference_point_in_meters_rejected():
    dx = C0 / (1e6 * 4.0)
    grid = Grid(shape=(128,), dx=dx, pml=PMLSpec(thickness=16 * dx))
    med = Medium.homogeneous(grid.shape, water(c=C0))
    src = plane_cw_source(grid, f0=1e6, amplitude=1e5, position_vox=24)
    with pytest.raises(ValueError, match="meters"):
        solvers.get("linear")().run(
            grid, med, src, CWRunSpec(), backend="numpy", reference_point=(0.015,)
        )


def test_nonlinear_term_smoke_fast():
    # Fast (non-slow) coverage of the Westervelt branch: 2f0 energy appears.
    dx = C0 / (1e6 * 8.0)
    grid = Grid(shape=(240,), dx=dx, pml=PMLSpec(thickness=24 * dx))
    med = Medium.homogeneous(grid.shape, water(c=C0, beta=3.5))
    src = plane_cw_source(grid, f0=1e6, amplitude=2e6, position_vox=32)
    spec = CWRunSpec(min_settle_periods=10, max_settle_periods=40)
    res = solvers.get("westervelt")().run(grid, med, src, spec, backend="numpy", harmonics=(1, 2))
    assert res.meta["nonlinear_active"] is True
    mid = slice(80, 160)
    ratio = res.harmonic_amp(2)[mid].max() / res.harmonic_amp(1)[mid].max()
    assert 0.005 < ratio < 0.5  # nonzero, sane 2nd-harmonic content


def test_build_phase_maps_refuses_collisions():
    arr = archimedean_spiral()
    with pytest.raises(ValueError, match="collisions"):
        build_phase_maps(np.zeros(arr.n_elements), arr.positions, 16, 0.100)


def test_transducer_array_copies_inputs():
    arr0 = archimedean_spiral(n_elements=16, d_outer=0.03, d_inner=0.01, roc=0.03)
    pos = arr0.positions.copy()
    from hifusim.arrays.transducer import TransducerArray

    arr = TransducerArray(
        positions=pos,
        normals=arr0.normals.copy(),
        elem_radius=arr0.elem_radius,
        focal_length=arr0.focal_length,
    )
    pos[:, 2] += 1.0  # caller mutates its buffer...
    assert arr.positions[:, 2].max() < 0.1  # ...frozen geometry unaffected


def _phased_disc(center, radius_vox, f0, k0):
    """Disc whose phases advance linearly along x -> steered wavefront."""
    o = np.arange(-radius_vox, radius_vox + 1)
    mx, my = np.meshgrid(o, o, indexing="ij")
    m = mx**2 + my**2 <= radius_vox**2
    pts = np.stack([mx[m], my[m]], 1) + np.asarray(center)
    phases = (0.9 * k0 * 0.5e-3 * pts[:, 0].astype(np.float64)) % (2 * np.pi)
    return CWSource(
        indices=pts.astype(np.int64),
        phases=phases.astype(np.float32),
        amplitude=1e5,
        f0=f0,
        label="phased-disc",
    )


@pytest.mark.kwave
@pytest.mark.slow
def test_kwave_phase_reordering_with_nonuniform_phases():
    """Nonzero per-voxel phases through the adapter's Fortran-order LUT.

    Any permutation bug in the mask reordering scrambles the phases, the
    steered beam collapses, and the cross-correlation with the native
    solver drops far below the gate.
    """
    pytest.importorskip("kwave", reason="k-wave-python not installed")
    f0 = 0.5e6
    k0 = 2 * np.pi * f0 / C0
    grid = Grid(shape=(96, 96), dx=0.5e-3, pml=PMLSpec(thickness=10e-3))
    med = Medium.homogeneous(grid.shape, water(c=C0))
    src = _phased_disc((30, 48), 4, f0, k0)
    spec = CWRunSpec(min_settle_periods=10)
    r_lin = solvers.get("linear")().run(grid, med, src, spec, backend="numpy")
    try:
        r_kw = solvers.get("kwave")().run(grid, med, src, spec)
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        pytest.skip(f"k-Wave binary unavailable: {exc}")
    inner = (slice(26, 70), slice(26, 70))
    a = r_lin.amp[inner] / r_lin.amp[inner].max()
    b = r_kw.amp[inner] / r_kw.amp[inner].max()
    r = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    rel_l2 = np.linalg.norm(a - b) / np.linalg.norm(b)
    assert r > 0.99, f"phased-source correlation r={r:.5f}"
    assert rel_l2 < 0.05, f"phased-source relL2={rel_l2 * 100:.1f}%"


@pytest.mark.kwave
@pytest.mark.slow
def test_kwave_heterogeneous_absorbing_medium():
    """Heterogeneous rho/c + nonzero alpha through the adapter's unit maps."""
    pytest.importorskip("kwave", reason="k-wave-python not installed")
    f0 = 0.5e6
    grid = Grid(shape=(96, 96), dx=0.5e-3, pml=PMLSpec(thickness=10e-3))
    id_map = np.zeros(grid.shape, dtype=np.uint8)
    id_map[:, 48:] = 1
    db = MaterialDB(
        materials={
            0: water(c=C0),
            1: Material(name="fatlike", alpha_np_m=8.0, rho=950.0, c=1440.0, beta=0.0),
        }
    )
    med = Medium.from_id_map(id_map, db)
    o = np.arange(-3, 4)
    mx, my = np.meshgrid(o, o, indexing="ij")
    msk = mx**2 + my**2 <= 9
    pts = np.stack([mx[msk], my[msk]], 1) + np.array([30, 30])
    src = CWSource(
        indices=pts.astype(np.int64),
        phases=np.zeros(msk.sum(), np.float32),
        amplitude=1e5,
        f0=f0,
    )
    spec = CWRunSpec(min_settle_periods=14)
    r_lin = solvers.get("linear")().run(grid, med, src, spec, backend="numpy")
    try:
        r_kw = solvers.get("kwave")().run(grid, med, src, spec)
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        pytest.skip(f"k-Wave binary unavailable: {exc}")
    inner = (slice(26, 70), slice(26, 70))
    a = r_lin.amp[inner] / r_lin.amp[inner].max()
    b = r_kw.amp[inner] / r_kw.amp[inner].max()
    r = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    rel_l2 = np.linalg.norm(a - b) / np.linalg.norm(b)
    assert r > 0.99, f"hetero correlation r={r:.5f}"
    assert rel_l2 < 0.05, f"hetero relL2={rel_l2 * 100:.1f}%"


# ------------- 2026-08-11 physics-review regressions (round 2) -------------


def test_mass_source_normalization_and_phase_convention():
    """Three contracts in one 2-D water run + one perturbed rerun:

    1. realized plane amplitude ~= source.amplitude (mass-source scaling);
    2. the amplitude is INVARIANT to remote medium content that only
       changes dt through c_max (the old raw injection shifted ~20% here);
    3. phasor convention p(t)=Re{P e^{-iwt}}: the outgoing wave's unwrapped
       phase must INCREASE along propagation at slope +k.
    """
    shape = (96, 64)
    dx = 0.25e-3
    f0 = 1e6
    amp = 1e5
    grid = Grid(shape=shape, dx=dx, pml=PMLSpec(thickness=2e-3))  # 8 voxels
    src = plane_cw_source(grid, f0=f0, amplitude=amp, position_vox=14)
    spec = CWRunSpec(min_settle_periods=6, n_record_periods=2)
    win = (slice(30, 70), slice(20, 44))

    med1 = Medium.homogeneous(shape, water(c=C0))
    res1 = solvers.get("linear")().run(grid, med1, src, spec, record_region=win)
    plateau1 = float(np.median(np.abs(res1.phasor)))
    assert 0.90 * amp < plateau1 < 1.12 * amp, f"realized {plateau1 / amp:.3f}x amp"

    # fast block deep inside the FAR sponge: physically invisible to the
    # window, but raises c_max -> smaller dt -> different spp
    c2 = med1.c.copy()
    c2[-4:, :] = 1800.0
    med2 = Medium(alpha=med1.alpha, rho=med1.rho, c=c2, beta=med1.beta)
    res2 = solvers.get("linear")().run(grid, med2, src, spec, record_region=win)
    assert res2.spp != res1.spp  # the coupling channel really did change
    plateau2 = float(np.median(np.abs(res2.phasor)))
    rel = abs(plateau2 - plateau1) / plateau1
    assert rel < 0.02, f"remote medium changed the drive by {rel * 100:.1f}%"

    line = res1.phasor[:, res1.phasor.shape[1] // 2]
    phase = np.unwrap(np.angle(line))
    x_m = np.arange(phase.size) * dx
    slope = float(np.polyfit(x_m, phase, 1)[0])
    k = 2.0 * np.pi * f0 / C0
    assert slope == pytest.approx(+k, rel=0.02), f"phase slope {slope:.0f} vs +k={k:.0f}"


def test_settle_capped_reports_convergence_honestly():
    shape = (32, 32)
    grid = Grid(shape=shape, dx=1e-3, pml=PMLSpec(thickness=4e-3))
    med = Medium.homogeneous(shape, water(c=C0))
    src = plane_cw_source(grid, f0=0.5e6, amplitude=1e5)

    ok = solvers.get("linear")().run(
        grid, med, src, CWRunSpec(min_settle_periods=1, max_settle_periods=40, n_record_periods=1)
    )
    assert not ok.settle_capped
    assert ok.converged_period < ok.tof_periods + 40

    capped = solvers.get("linear")().run(
        grid,
        med,
        src,
        CWRunSpec(
            min_settle_periods=1,
            max_settle_periods=5,
            convergence_tol=0.0,  # unattainable: rel change is never < 0
            n_record_periods=1,
        ),
    )
    assert capped.settle_capped


@pytest.mark.kwave
def test_kwave_rejects_source_inside_its_pml_band():
    pytest.importorskip("kwave")
    grid = Grid(shape=(64, 64), dx=1e-3, pml=PMLSpec(thickness=5e-3))  # 5 voxels
    med = Medium.homogeneous((64, 64), water(c=C0))
    src = plane_cw_source(grid, f0=0.5e6, amplitude=1e5, position_vox=3)  # inside the band
    with pytest.raises(ValueError, match="PML band"):
        solvers.get("kwave")().run(grid, med, src, CWRunSpec())
