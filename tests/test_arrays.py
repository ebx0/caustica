"""M6 gates: spiral geometry regression, DAS focusing, voxelization, phase maps."""

import numpy as np
import pytest

import caustica.solvers as solvers
from caustica import Grid, Medium, PMLSpec
from caustica.arrays import archimedean_spiral, build_phase_maps, select_phase_map_size
from caustica.materials import water
from caustica.solvers import CWRunSpec

F0, C0 = 1.0e6, 1500.0


@pytest.fixture(scope="module")
def production_array():
    """The 128-element production geometry (notebook defaults)."""
    return archimedean_spiral()


@pytest.fixture(scope="module")
def small_array():
    """A compact spiral that fits cheap 3-D test grids."""
    return archimedean_spiral(
        n_elements=32, d_outer=0.030, d_inner=0.010, roc=0.030, active_fraction=0.60
    )


def test_production_geometry_regression(production_array):
    arr = production_array
    assert arr.n_elements == 128
    # Element radius from the equal-area rule (hand-derived from the
    # notebook's formulas): ~3.20 mm for the production parameters.
    assert arr.elem_radius == pytest.approx(3.205e-3, rel=1e-2)
    # All elements between the inset apertures, on the spherical shell.
    r = np.hypot(arr.positions[:, 0], arr.positions[:, 1])
    # Spiral must respect the one-element-radius inset from both apertures.
    assert r.min() >= 0.022 + arr.elem_radius - 1e-9
    assert r.max() <= 0.050 - arr.elem_radius + 1e-9
    shell = np.linalg.norm(arr.positions - arr.focus, axis=1)
    np.testing.assert_allclose(shell, 0.100, rtol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(arr.normals, axis=1), 1.0, rtol=1e-9)


def test_spiral_validation():
    with pytest.raises(ValueError, match="d_inner"):
        archimedean_spiral(d_inner=0.2, d_outer=0.1)
    with pytest.raises(ValueError, match="active_fraction"):
        archimedean_spiral(active_fraction=0.0)


def test_das_phases_steer_by_the_commanded_displacement(small_array):
    """DAS steering gate, bias-cancelled.

    A finite-aperture focus peaks PROXIMAL of the phase target (classic
    focal shift), so "peak == target" is the wrong gate. The right one:
    the peak DISPLACEMENT between uniform phasing and DAS phasing must
    equal the commanded displacement within lambda/2 — the systematic
    focal-shift bias cancels in the difference.
    """
    arr = small_array
    lam = C0 / F0
    target = np.array([3e-3, -2e-3, arr.focal_length - 4e-3])
    commanded = target - arr.focus
    phases = arr.das_phases(target, f0=F0, c0=C0)
    assert phases.min() == 0.0 and phases.max() < 2 * np.pi

    span = np.arange(-8e-3, 8e-3 + 1e-9, lam / 6)

    def peak_of(center, ph):
        gx, gy, gz = np.meshgrid(
            center[0] + span, center[1] + span, center[2] + span, indexing="ij"
        )
        pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        p = np.abs(arr.rayleigh_preview(pts, f0=F0, phases=ph, c0=C0))
        return pts[np.argmax(p)]

    peak_uniform = peak_of(arr.focus, None)
    peak_das = peak_of(target, phases)
    realized = peak_das - peak_uniform
    lat_err = np.linalg.norm((realized - commanded)[:2])
    ax_err = abs((realized - commanded)[2])
    # Lateral steering is sub-wavelength; axial position of a CW focus is
    # diffraction-dominated (the -6 dB focal length here is ~7 lambda), so
    # one wavelength is the honest axial gate.
    assert lat_err <= lam / 2, f"lateral steering error {lat_err * 1e3:.2f} mm"
    assert ax_err <= lam, f"axial steering error {ax_err * 1e3:.2f} mm"
    # And the field at the target must actually have moved there:
    at_target = abs(arr.rayleigh_preview(target[None, :], f0=F0, phases=phases, c0=C0)[0])
    at_target_uniform = abs(arr.rayleigh_preview(target[None, :], f0=F0, c0=C0)[0])
    assert at_target > 3.0 * at_target_uniform


def test_voxelization_invariants(small_array):
    grid = Grid(shape=(96, 96, 96), dx=0.5e-3, pml=PMLSpec(thickness=5e-3))
    asrc = small_array.voxelize(grid, apex_vox=(48, 48, 12), f0=F0, amplitude=1e5)
    assert asrc.n_elements_represented == small_array.n_elements  # 32/32 present
    assert asrc.source.n_points == asrc.element_of_voxel.size
    # Curved shell: z-span roughly the cap depth, radial extent inside aperture.
    idx = asrc.source.indices
    r_mm = np.hypot(idx[:, 0] - 48, idx[:, 1] - 48) * 0.5
    assert r_mm.max() <= (0.030 / 2) * 1e3 + 1.0
    # Curved shell: z must actually follow the bowl depth (cap height),
    # not sit in a flat plane (review finding: z geometry was unasserted).
    h_vox = (0.030 - np.sqrt(0.030**2 - 0.015**2)) / 0.5e-3
    z_span = idx[:, 2].max() - idx[:, 2].min()
    assert 0.5 * h_vox <= z_span <= h_vox + 2
    asrc.source.check_inside(grid)


def test_voxelization_phase_passthrough(small_array):
    grid = Grid(shape=(96, 96, 96), dx=0.5e-3, pml=PMLSpec(thickness=5e-3))
    phases = np.linspace(0, 2 * np.pi, small_array.n_elements, endpoint=False).astype(np.float32)
    asrc = small_array.voxelize(grid, (48, 48, 12), f0=F0, amplitude=1e5, phases=phases)
    # Every voxel carries exactly its owning element's phase.
    np.testing.assert_array_equal(asrc.source.phases, phases[asrc.element_of_voxel])


def test_phase_map_size_selection_production(production_array):
    """Regression: 32x32 FAILS placement (95 offset offenders) -> fallback 64.

    This pins the notebook's actual runtime behavior: the production spiral
    does not meet the 0.25-pixel centering tolerance at 32x32, so the
    dataset's phase maps are 64x64 (the selector's fallback).
    """
    size, stats = select_phase_map_size(production_array.positions, d_outer=0.100)
    assert size == 64
    assert stats["n_collisions"] == 0
    assert stats["n_offset_offenders"] == 0
    from caustica.arrays.phasemaps import project_elements

    xi, yi, offsets_mm, _pitch = project_elements(production_array.positions, 32, 0.100)
    tol_mm = 0.25 * (0.100 / 31) * 1e3
    assert int((offsets_mm.max(axis=1) > tol_mm).sum()) == 95  # why 32 was rejected


def test_phase_maps_roundtrip(production_array):
    rng = np.random.default_rng(7)
    phases = rng.uniform(0, 2 * np.pi, production_array.n_elements).astype(np.float32)
    maps, phase_map = build_phase_maps(phases, production_array.positions, 32, 0.100)
    assert maps.shape == (2, 32, 32) and phase_map.shape == (32, 32)
    from caustica.arrays.phasemaps import project_elements

    xi, yi, _, _ = project_elements(production_array.positions, 32, 0.100)
    np.testing.assert_allclose(maps[0][xi, yi], np.sin(phases), atol=1e-6)
    np.testing.assert_allclose(maps[1][xi, yi], np.cos(phases), atol=1e-6)
    # sin^2 + cos^2 == 1 exactly at element pixels, 0 far outside the aperture.
    assert maps[0][0, 0] == 0.0 and maps[1][0, 0] == 0.0


@pytest.mark.slow
def test_spiral_solver_integration_focuses_at_geometric_focus(small_array):
    """End-to-end: voxelized spiral + linear solver -> focus at apex + roc."""
    grid = Grid(shape=(96, 96, 96), dx=0.5e-3, pml=PMLSpec(thickness=5e-3))
    medium = Medium.homogeneous(grid.shape, water(c=C0))
    apex = (48, 48, 12)
    focus_vox = (48, 48, 12 + 60)  # roc = 30 mm = 60 voxels
    asrc = small_array.voxelize(grid, apex, f0=F0, amplitude=1e5)
    spec = CWRunSpec(min_settle_periods=8, max_settle_periods=30)
    res = solvers.get("linear")().run(
        grid, medium, asrc.source, spec, backend="numpy", reference_point=focus_vox
    )
    peak = np.unravel_index(np.argmax(res.amp), res.amp.shape)
    assert abs(peak[0] - focus_vox[0]) <= 1
    assert abs(peak[1] - focus_vox[1]) <= 1
    # Axial gate: between the O'Neil-predicted peak (full-bowl proximal
    # bound; the spiral's annular aperture pulls slightly distal of it)
    # and one voxel past the geometric focus.
    from caustica.analytic import axial_pressure

    z_fine = np.linspace(0.010, 0.045, 2001)
    z_oneill = z_fine[np.argmax(np.abs(axial_pressure(z_fine, 0.015, 0.030, F0, C0)))]
    z_lo = apex[2] + z_oneill / 0.5e-3 - 1.0
    assert z_lo <= peak[2] <= focus_vox[2] + 1, (
        f"axial peak {peak[2]} outside [O'Neil {z_lo:.1f}, geo+1 {focus_vox[2] + 1}]"
    )
