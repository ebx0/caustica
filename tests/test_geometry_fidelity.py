"""What was ordered, against what was built.

Every geometry this library makes is a request turned into voxels or points
by code that has to round, and the interesting question is never "did it
return something" but "how far is what it returned from what was asked for".
So each test here names the closed-form answer, measures the constructed one,
and pins the gap.

The bounds are measurements, not aspirations: they come from
``scripts/dev_geometry.py``, which reports the same quantities in full and
writes them to ``benchmarks/reports/geometry/``. Tightening one of them is a
claim that needs a measurement behind it, same as loosening one.
"""

from __future__ import annotations

import numpy as np
import pytest

from caustica.analytic.geometry import spherical_cap_points
from caustica.arrays import archimedean_spiral
from caustica.core.grid import Grid
from caustica.geometry import Ball, Box, Cylinder, Ellipsoid, Scene

MM = 1e-3


def bowl_voxels(dx: float, aperture: float, roc: float, spacing: float | None = None):
    """The library's own bowl digitization, in voxels about the apex."""
    ds = dx / 2.0 if spacing is None else spacing
    points, _n, _a = spherical_cap_points(aperture, roc, ds)
    return np.unique(np.round(points / dx).astype(np.int64), axis=0)


def rasterize(shape, grid: Grid, origin) -> np.ndarray:
    return np.asarray(Scene(3).add(shape, 1).rasterize(grid, origin=origin).labels) == 1


# ---------------------------------------------------------------- the bowl


@pytest.mark.parametrize("dx_mm", [0.5, 0.25, 0.1, 0.05])
def test_the_bowl_shell_sits_on_the_sphere_that_was_ordered(dx_mm):
    """The number that matters is the distance to the KNOWN surface.

    Not the radius a curve fit infers from the voxels — a cap this shallow
    does not determine its own curvature from a voxel shell, and an
    algebraic sphere fit reads 19 % short at dx = 0.5 mm while every one of
    those voxels is within three quarters of a voxel of the sphere it was
    asked for. Measured, 2026-08-24: rms 0.335-0.338 voxels and max 0.72 at
    every dx from 0.5 mm to 0.05 mm, which is a digitization error and
    nothing else.
    """
    dx, aperture, roc = dx_mm * MM, 5.0 * MM, 12.0 * MM
    p = bowl_voxels(dx, aperture, roc).astype(float) * dx
    err = np.linalg.norm(p - np.array([0.0, 0.0, roc]), axis=1) - roc

    assert np.sqrt(np.mean(err**2)) / dx < 0.40, "the shell has drifted off the ordered sphere"
    assert np.abs(err).max() / dx < 0.80, "a voxel sits more than a voxel off the sphere"


@pytest.mark.parametrize("dx_mm", [0.5, 0.25, 0.1])
def test_the_apex_is_a_flat_disc_of_the_size_the_curvature_implies(dx_mm):
    """The apex plane is not one voxel, and the width of it is predictable.

    A cap sags by r^2/2R, so every point out to r = sqrt(R dx) is within half
    a voxel of the apex plane and rounds into it. At R = 12 mm and
    dx = 0.25 mm that is 1.73 mm, seven voxels across — which is the shape a
    digitized shallow bowl has, not a defect. Worth pinning, because a
    reader who expects the apex to be a single voxel will misread every
    near-field plot the library draws.
    """
    dx, aperture, roc = dx_mm * MM, 5.0 * MM, 12.0 * MM
    idx = bowl_voxels(dx, aperture, roc)
    assert idx[:, 2].min() == 0, "the apex must land on the voxel it was indexed from"

    disc = np.linalg.norm(idx[idx[:, 2] == 0][:, :2], axis=1).max() * dx
    assert disc == pytest.approx(np.sqrt(roc * dx), abs=1.5 * dx)


def test_the_bowls_aperture_does_not_exceed_the_one_requested_by_more_than_a_voxel():
    dx, aperture, roc = 0.25 * MM, 5.0 * MM, 12.0 * MM
    p = bowl_voxels(dx, aperture, roc).astype(float) * dx
    rim = float(np.linalg.norm(p[:, :2], axis=1).max())
    assert aperture - dx <= rim <= aperture + dx


def test_the_shipped_cap_sampling_leaves_holes_in_the_shell():
    """A characterization test: this is a KNOWN defect, pinned deliberately.

    ``bowl_cw_source`` samples the cap at dx/2 and rounds to voxels. Sampling
    the same cap sixteen times finer reaches voxels the shipped spacing never
    does — measured 2026-08-24 as 10.4 % to 12.1 % of the shell across dx from
    0.5 mm to 0.05 mm, roughly independent of dx because it is a property of
    the sampling ratio and not of the grid. Those voxels are undriven, so the
    bowl radiates from a porous shell.

    The bound below is the current behaviour, not the desired one. It exists
    so that closing the holes has to be a deliberate act with a measurement
    attached — refining the default sampling also raises the voxel count
    above the cap's own area, which makes the drive normalization worse
    before it makes it better (see the staircase test below).
    """
    dx, aperture, roc = 0.25 * MM, 5.0 * MM, 12.0 * MM
    shipped = {tuple(v) for v in bowl_voxels(dx, aperture, roc)}
    fine = {tuple(v) for v in bowl_voxels(dx, aperture, roc, spacing=dx / 16.0)}

    missing = len(fine - shipped) / len(fine)
    assert 0.08 < missing < 0.15, f"the hole fraction moved to {missing:.3f}; re-measure and say so"
    assert not shipped - fine, "the shipped sampling reached a voxel the dense one did not"


@pytest.mark.parametrize("denom,low,high", [(2, 1.10, 1.25), (16, 1.28, 1.40)])
def test_a_digitized_cap_carries_more_voxels_than_its_area(denom, low, high):
    """The staircase factor, characterized: it is why a bowl over-drives.

    A flat source has exactly one voxel per dx^2 of aperture. A tilted
    surface crosses more, and the engine drives every source voxel with the
    same normalized amplitude — so a bowl radiates in proportion to its voxel
    count, not to its area. Measured 2026-08-24 on an f/1.2 cap: 1.18 voxels
    per dx^2 at the shipped dx/2 sampling and 1.33 at dx/16, and the focal
    pressure sits 1.15x and 1.25x O'Neil's closed form respectively — the
    excess tracks the ratio, and it does NOT fall as dx shrinks, because it
    is a property of digitizing a tilted surface rather than a discretization
    error.

    Pinned rather than fixed: correcting it rescales every focused-bowl
    pressure this library has ever produced.
    """
    dx, aperture, roc = 0.25 * MM, 5.0 * MM, 12.0 * MM
    n = len(bowl_voxels(dx, aperture, roc, spacing=dx / denom))
    cap_area = 2.0 * np.pi * roc**2 * (1.0 - np.sqrt(1.0 - (aperture / roc) ** 2))

    assert low < n * dx**2 / cap_area < high


# ------------------------------------------------------------ the cap cloud


@pytest.mark.parametrize("aperture_mm,roc_mm", [(5.0, 12.0), (32.0, 64.0), (1.0, 50.0)])
def test_the_cap_point_cloud_carries_the_exact_cap_area(aperture_mm, roc_mm):
    """Every analytic reference in this library is weighted by these areas.

    A cap whose sample areas do not sum to 2 pi R^2 (1 - cos theta_max) is a
    cap radiating the wrong power, and the "analytic" curve the solver is
    graded against would be wrong by the same factor.
    """
    a, f = aperture_mm * MM, roc_mm * MM
    exact = 2.0 * np.pi * f**2 * (1.0 - np.sqrt(1.0 - (a / f) ** 2))
    pts, nrm, areas = spherical_cap_points(a, f, f / 200.0)

    assert areas.sum() == pytest.approx(exact, rel=1e-12)
    assert np.ptp(areas) == 0.0, "equal-area sampling is what the Rayleigh sum assumes"
    assert np.abs(np.linalg.norm(pts - np.array([0.0, 0.0, f]), axis=1) - f).max() < 1e-15
    assert np.abs(np.linalg.norm(nrm, axis=1) - 1.0).max() < 1e-12
    toward = np.array([0.0, 0.0, f]) - pts
    toward /= np.linalg.norm(toward, axis=1)[:, None]
    assert np.abs((nrm * toward).sum(1) - 1.0).max() < 1e-12
    assert np.linalg.norm(pts[:, :2], axis=1).max() <= a


# --------------------------------------------------------------- primitives


@pytest.mark.parametrize(
    "name,shape,exact",
    [
        ("ball", Ball((0.0, 0.0, 0.0), 4.0 * MM), 4.0 / 3.0 * np.pi * (4.0 * MM) ** 3),
        (
            "ellipsoid",
            Ellipsoid((0.0, 0.0, 0.0), (4.0 * MM, 2.4 * MM, 5.6 * MM)),
            4.0 / 3.0 * np.pi * (4.0 * MM) * (2.4 * MM) * (5.6 * MM),
        ),
        (
            "cylinder",
            Cylinder((0.0, 0.0, 0.0), 4.0 * MM, 6.0 * MM, axis=2),
            np.pi * (4.0 * MM) ** 2 * 6.0 * MM,
        ),
    ],
)
def test_a_curved_primitive_rasterizes_to_its_closed_form_volume(name, shape, exact):
    """Within the price of binary voxels, which is what the bound is.

    A voxel is in or out by its centre, so the error is a boundary term.
    Measured 2026-08-24 at dx = 0.25 mm: 0.03 % to 0.96 % depending on how
    the grid happens to fall against the surface, with no misclassified voxel
    anywhere against a 9x9x9 occupancy truth — the residual is the volume a
    binary approximation simply cannot represent, not a rasterizer error.
    """
    dx, half = 0.25 * MM, 6.0 * MM
    n = int(round(2 * half / dx))
    grid = Grid(shape=(n, n, n), dx=dx)
    origin = tuple(-half + dx / 2 for _ in range(3))

    volume = float(rasterize(shape, grid, origin).sum()) * dx**3
    assert volume == pytest.approx(exact, rel=0.02)


def test_a_box_rasterizes_exactly():
    """No curvature, no boundary term: a flat-faced primitive has no excuse."""
    dx, half = 0.25 * MM, 6.0 * MM
    n = int(round(2 * half / dx))
    grid = Grid(shape=(n, n, n), dx=dx)
    origin = tuple(-half + dx / 2 for _ in range(3))
    size = (4.0 * MM, 6.0 * MM, 2.0 * MM)

    volume = float(rasterize(Box((0.0, 0.0, 0.0), size), grid, origin).sum()) * dx**3
    assert volume == pytest.approx(size[0] * size[1] * size[2], rel=1e-12)


# ------------------------------------------------------------------ algebra


def test_the_boolean_algebra_agrees_with_the_rasterizer_voxel_for_voxel():
    """Set operations get no discretization excuse.

    On a fixed grid, the rasterization of ``A | B`` has to be exactly the
    union of the two rasterizations — every voxel, no tolerance. Anything
    less would mean the algebra and the rasterizer disagree about what a
    shape is, and every scene built from more than one primitive inherits it.
    """
    dx, half, n = 0.25 * MM, 6.0 * MM, 48
    grid = Grid(shape=(n, n, n), dx=dx)
    origin = tuple(-half + dx / 2 for _ in range(3))
    a = Ball((-1.0 * MM, 0.0, 0.0), 3.0 * MM)
    b = Box((1.0 * MM, 0.0, 0.0), (4.0 * MM, 4.0 * MM, 4.0 * MM))
    ma, mb = rasterize(a, grid, origin), rasterize(b, grid, origin)

    assert np.array_equal(rasterize(a | b, grid, origin), ma | mb)
    assert np.array_equal(rasterize(a & b, grid, origin), ma & mb)
    assert np.array_equal(rasterize(a - b, grid, origin), ma & ~mb)
    assert np.array_equal(rasterize(~a, grid, origin), ~ma)
    assert np.array_equal(rasterize(~(a | b), grid, origin), rasterize(~a & ~b, grid, origin))
    assert int((ma | mb).sum()) == int(ma.sum()) + int(mb.sum()) - int((ma & mb).sum())


def test_a_rigid_motion_keeps_the_volume_and_lands_on_the_target():
    """Volume is the invariant of a rigid motion; the centroid is the aim."""
    dx, half, n = 0.2 * MM, 6.0 * MM, 60
    grid = Grid(shape=(n, n, n), dx=dx)
    origin = tuple(-half + dx / 2 for _ in range(3))
    coords = [np.arange(n) * dx + origin[d] for d in range(3)]

    def measure(shape):
        m = rasterize(shape, grid, origin)
        pts = np.argwhere(m)
        world = np.column_stack([coords[d][pts[:, d]] for d in range(3)])
        return int(m.sum()), world.mean(0)

    ball = Ball((0.0, 0.0, 0.0), 3.0 * MM)
    target = np.array([1.4 * MM, -0.8 * MM, 2.0 * MM])
    n0, c0 = measure(ball)
    n_t, c_t = measure(ball.translated(tuple(target)))
    n_r, _ = measure(ball.rotated(0.7, axis=(1.0, 1.0, 0.0)))

    assert n_t == pytest.approx(n0, rel=0.01)
    assert n_r == pytest.approx(n0, rel=0.01)
    assert np.abs(c_t - (c0 + target)).max() < 0.5 * dx


def test_a_quarter_turn_swaps_an_ellipsoids_semiaxes():
    """The sharpest statement of "the rotation is the one that was asked for"."""
    dx, half, n = 0.2 * MM, 6.0 * MM, 60
    grid = Grid(shape=(n, n, n), dx=dx)
    origin = tuple(-half + dx / 2 for _ in range(3))

    def extent(shape):
        pts = np.argwhere(rasterize(shape, grid, origin)).astype(float) * dx
        return pts.max(0) - pts.min(0)

    ell = Ellipsoid((0.0, 0.0, 0.0), (4.0 * MM, 1.5 * MM, 2.5 * MM))
    before = extent(ell)
    after = extent(ell.rotated(np.pi / 2, axis=(0.0, 0.0, 1.0)))

    assert after[0] == pytest.approx(before[1], abs=dx)
    assert after[1] == pytest.approx(before[0], abs=dx)
    assert after[2] == pytest.approx(before[2], abs=dx)


# -------------------------------------------------------------- the array


@pytest.mark.parametrize(
    "n_el,d_out_mm,d_in_mm,roc_mm", [(128, 100.0, 44.0, 100.0), (64, 40.0, 12.0, 50.0)]
)
def test_the_spiral_array_honours_every_term_of_its_datasheet(n_el, d_out_mm, d_in_mm, roc_mm):
    """The one geometry a user orders in the vocabulary of a real transducer.

    Element count, shell radius, outer diameter and central hole are all
    constraints, and each is checked as one. ``active_fraction`` is the only
    term that is a target rather than a bound (it lands at 0.63-0.65 against
    a 0.60 request), so it is left to the report rather than asserted here.
    """
    arr = archimedean_spiral(
        n_elements=n_el,
        d_outer=d_out_mm * MM,
        d_inner=d_in_mm * MM,
        roc=roc_mm * MM,
        active_fraction=0.6,
    )
    pos = np.asarray(arr.positions)
    center = np.array([0.0, 0.0, roc_mm * MM])
    transverse = np.linalg.norm(pos[:, :2], axis=1)

    assert len(pos) == n_el
    assert arr.focal_length == pytest.approx(roc_mm * MM, rel=1e-12)
    assert np.abs(np.linalg.norm(pos - center, axis=1) - roc_mm * MM).max() < 1e-12
    assert (transverse + arr.elem_radius).max() <= d_out_mm * MM / 2 + 1e-9
    assert (transverse - arr.elem_radius).min() >= d_in_mm * MM / 2 - 1e-9
    toward = center - pos
    toward /= np.linalg.norm(toward, axis=1)[:, None]
    assert np.abs((np.asarray(arr.normals) * toward).sum(1) - 1.0).max() < 1e-12


# ---------------------------------------------------- geometry inside a job


@pytest.mark.parametrize(
    "dx_mm,d_outer_mm,roc_mm,apex_mm", [(0.5, 12.0, 10.0, 6.0), (0.1, 5.0, 6.0, 2.0)]
)
def test_the_bowl_a_job_orders_is_the_bowl_the_builder_makes(dx_mm, d_outer_mm, roc_mm, apex_mm):
    """The whole path, in the job file's own units.

    A job names millimetres; the builder converts to voxels, and a rounding
    convention lives at each step. The focus the builder REPORTS is what the
    runner plans around and what the report prints, so it has to be the one
    the job file implies, not merely close to it.
    """
    from caustica.config.job import build_job, parse_job

    size = [max(4.0 * roc_mm, 2.5 * d_outer_mm)] * 2 + [roc_mm + apex_mm + 6.0]
    job = parse_job(
        {
            "format": "caustica-job/1",
            "kind": "explicit",
            "name": "geom",
            "medium": {"kind": "homogeneous"},
            "grid": {
                "ndim": 3,
                "dx_mm": dx_mm,
                "size_mm": size,
                "pml": {"thickness_mm": 1.5},
            },
            "source": {
                "kind": "array",
                "array": {"kind": "bowl", "d_outer_mm": d_outer_mm, "roc_mm": roc_mm},
                "apex_mm": [size[0] / 2, size[1] / 2, apex_mm],
            },
            "drive": {"f0_mhz": 1.0, "amplitude_kpa": 100.0},
            "run": {"harmonics": [1]},
            "solver": "linear",
        },
        "geom",
    )
    built = build_job(job, base_dir=None, with_medium=False)
    dx = built.grid.dx
    idx = np.asarray(built.source.indices)
    apex_expected = np.round(np.array([size[0] / 2, size[1] / 2, apex_mm]) * MM / dx)
    focus_expected = apex_expected + np.array([0.0, 0.0, round(roc_mm * MM / dx)])

    assert list(built.focus_vox) == [int(v) for v in focus_expected]
    assert idx[:, 2].min() == int(apex_expected[2])
    err = np.linalg.norm(idx * dx - focus_expected * dx, axis=1) - roc_mm * MM
    assert np.abs(err).max() / dx < 0.80


def test_no_shipped_job_puts_a_source_voxel_in_the_pml():
    """The mistake that cost a night, made a gate.

    A bowl whose rim lands inside the absorbing layer is not a bowl; it is a
    bowl with its edge dissolved, producing a field that is plausible, stable
    and answering a question nobody asked. On 2026-08-24 that was reported as
    a 26 % sensitivity to PML thickness before anyone looked at the geometry.
    The control below is a job deliberately built that way, so the check is
    known to be able to fail.
    """
    from caustica import examples
    from caustica.config.job import build_job, load_job, parse_job

    def source_voxels_in_the_pml(built) -> int:
        idx = np.asarray(built.source.indices)
        shape = np.asarray(built.grid.shape)
        pml = int(built.grid.pml_vox)
        return int(((idx < pml) | (idx >= shape - pml)).any(1).sum())

    for name in examples.available():
        job, base = load_job(examples.path(name))
        built = build_job(job, base_dir=base, with_medium=False)
        assert source_voxels_in_the_pml(built) == 0, f"{name} drives voxels inside the sponge"

    control = parse_job(
        {
            "format": "caustica-job/1",
            "kind": "explicit",
            "name": "pml-overlap-control",
            "medium": {"kind": "homogeneous"},
            "grid": {
                "ndim": 3,
                "dx_mm": 0.5,
                "size_mm": [19.0, 19.0, 24.0],
                "pml": {"thickness_mm": 3.0},
            },
            "source": {
                "kind": "array",
                "array": {"kind": "bowl", "d_outer_mm": 16.0, "roc_mm": 12.0},
                "apex_mm": [9.5, 9.5, 3.0],
            },
            "drive": {"f0_mhz": 1.0, "amplitude_kpa": 100.0},
            "run": {"harmonics": [1]},
            "solver": "linear",
        },
        "control",
    )
    built = build_job(control, base_dir=None, with_medium=False)
    assert source_voxels_in_the_pml(built) > 0, "the control no longer fails; this check is blind"
