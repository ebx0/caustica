"""The v2 transducer model: four element shapes, their quadrature and their deposit.

The gates here are the ones that make an element model worth having, and they
are graded against references that were not derived from the model:

* the closed-form area of every shape against a tensor Gauss-Legendre integral
  over the same surface, written out here from the shape's own parametrization;
* the deposited grid weights against that closed-form area (V-17), at three
  orientations per shape, because the whole point of an off-grid source is that
  a tilted surface still carries its own area;
* the Rayleigh integral the model builds over its quadrature against the same
  Gauss-Legendre integral, so the reference a solve is graded against is itself
  graded;
* and, at the end and marked ``slow``, the k-space solve of a steered
  sixteen-element array of each shape against that Rayleigh integral (V-11).
"""

import pathlib
import warnings

import numpy as np
import pytest

import caustica.solvers as solvers
from caustica import CausticaWarning, Grid, Medium, PMLSpec
from caustica.analytic.rayleigh import rayleigh_pressure
from caustica.materials import water
from caustica.solvers import CWRunSpec
from caustica.sources import bowl_cw_source, disc_cw_source
from caustica.transducers import (
    Element,
    Frame,
    Transducer,
    TransducerMeta,
    archimedean_spiral,
    deposit_cw,
    deposit_transient,
    disc,
    element_deposit,
    element_points,
    rect,
    ring_sector,
    spherical_segment,
    tangent_frame,
)

F0, C0, RHO0 = 0.5e6, 1500.0, 1000.0
LAM = C0 / F0
DX = LAM / 8.0  # 8 points per wavelength, the rung V-11 is stated at
AMP = 1.0e5

#: A normal that is not an axis, so nothing below can pass by symmetry.
TILTED = np.array([0.3, -0.4, 0.866])
CENTER = np.array([1.0e-3, -2.0e-3, 0.5e-3])


def _rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def four_shapes(center=CENTER, normal=TILTED):
    """One element of each shape, all at the same place and tilt."""
    return {
        "disc": disc(center, normal, 3.0e-3),
        "rect": rect(center, normal, 4.0e-3, 2.5e-3, rotation=0.7),
        "ring_sector": ring_sector(center, normal, 1.2e-3, 4.0e-3, 0.4, 0.4 + 2.1),
        "spherical_segment": spherical_segment(
            center, normal, 20.0e-3, 5.0e-3, r_in=1.8e-3, phi0=-0.3, phi1=-0.3 + 2.6
        ),
    }


# --------------------------------------------------------- independent reference


def gauss_surface(el: Element, n: int = 160):
    """``(points, areas)``: a tensor Gauss-Legendre integral over the element.

    Written from the shape's own parametrization and its Jacobian rather than
    from anything in :mod:`caustica.transducers`, so it can disagree with the
    model. It is the reference for both the area and the Rayleigh integral
    below; a shared bug would have to live in the tangent frame alone, which
    the frame test pins separately.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    a, wa = 0.5 * (x + 1.0), 0.5 * w
    u, v, nv = tangent_frame(el.normal)
    grid_a, grid_b = np.meshgrid(a, a, indexing="ij")
    weight = np.outer(wa, wa)

    if el.shape == "disc":
        r = el.size[0] * grid_a
        phi = 2.0 * np.pi * grid_b
        pts = el.center_m + (r * np.cos(phi))[..., None] * u + (r * np.sin(phi))[..., None] * v
        jac = r * el.size[0] * 2.0 * np.pi
    elif el.shape == "rect":
        width, height, rot = el.size
        cx, cy = (grid_a - 0.5) * width, (grid_b - 0.5) * height
        cr, sr = np.cos(rot), np.sin(rot)
        pts = el.center_m + (cx * cr - cy * sr)[..., None] * u + (cx * sr + cy * cr)[..., None] * v
        jac = np.full_like(grid_a, width * height)
    elif el.shape == "ring_sector":
        r_in, r_out, phi0, phi1 = el.size
        r = r_in + (r_out - r_in) * grid_a
        phi = phi0 + (phi1 - phi0) * grid_b
        pts = el.center_m + (r * np.cos(phi))[..., None] * u + (r * np.sin(phi))[..., None] * v
        jac = r * (r_out - r_in) * (phi1 - phi0)
    else:
        r_in, r_out, phi0, phi1 = el.size
        roc = float(el.curvature)
        t_in, t_out = np.arcsin(r_in / roc), np.arcsin(r_out / roc)
        theta = t_in + (t_out - t_in) * grid_a
        phi = phi0 + (phi1 - phi0) * grid_b
        lateral = roc * np.sin(theta)
        pts = (
            el.center_m
            + (lateral * np.cos(phi))[..., None] * u
            + (lateral * np.sin(phi))[..., None] * v
            + (roc * (1.0 - np.cos(theta)))[..., None] * nv
        )
        jac = roc**2 * np.sin(theta) * (t_out - t_in) * (phi1 - phi0)
    return pts.reshape(-1, 3), (jac * weight).ravel()


# ------------------------------------------------------------ frame and geometry


def test_the_tangent_frame_is_right_handed_and_gives_back_the_world_axes():
    """Everything in-plane (a rectangle's rotation, a sector's phi) is measured
    in this frame, so a wrong handedness would mirror every sector silently."""
    u, v, n = tangent_frame([0.0, 0.0, 3.0])
    np.testing.assert_allclose(u, [1.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-15)
    for normal in (TILTED, [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.9, 0.1, -0.4]):
        u, v, n = tangent_frame(normal)
        basis = np.stack([u, v, n])
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(basis) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError, match="non-zero"):
        tangent_frame([0.0, 0.0, 0.0])


@pytest.mark.parametrize("name", ["disc", "rect", "ring_sector", "spherical_segment"])
def test_every_shapes_closed_form_area_matches_an_independent_integral(name):
    """The area is the number the whole source model rests on (V-17).

    A closed form nobody checks is a place for a factor of two to live, so it
    is graded against a Gauss-Legendre integral of the same surface written
    from its parametrization.
    """
    el = four_shapes()[name]
    _pts, areas = gauss_surface(el)
    assert float(areas.sum()) == pytest.approx(el.area, rel=1e-12)


@pytest.mark.parametrize("name", ["disc", "rect", "ring_sector", "spherical_segment"])
def test_quadrature_points_sit_on_the_surface_and_carry_equal_area(name):
    """Equal-area by construction is what lets a deposit divide by the count."""
    el = four_shapes()[name]
    pts = element_points(el, 40000)
    u, v, n = el.frame
    local = pts - el.center_m
    a, b, c = local @ u, local @ v, local @ n
    if el.shape == "spherical_segment":
        roc = float(el.curvature)
        centre = el.center_m + roc * n
        np.testing.assert_allclose(np.linalg.norm(pts - centre, axis=1), roc, rtol=1e-12)
        r_in, r_out, phi0, phi1 = el.size
        radial = np.hypot(a, b)
        assert radial.min() >= r_in - 1e-9 and radial.max() <= r_out + 1e-9
    else:
        np.testing.assert_allclose(c, 0.0, atol=1e-15)  # flat: exactly in its own plane
        if el.shape == "disc":
            assert np.hypot(a, b).max() <= el.size[0] + 1e-12
        elif el.shape == "rect":
            width, height, rot = el.size
            cr, sr = np.cos(-rot), np.sin(-rot)
            assert np.abs(a * cr - b * sr).max() <= width / 2 + 1e-12
            assert np.abs(a * sr + b * cr).max() <= height / 2 + 1e-12
        else:
            r_in, r_out, _p0, _p1 = el.size
            radial = np.hypot(a, b)
            assert radial.min() >= r_in - 1e-9 and radial.max() <= r_out + 1e-9

    # Equal area: the empirical centroid of the point set is the area centroid,
    # which it would not be if the lattice were denser anywhere. The tolerance
    # is the sampling error of a finite point set, a part in a few thousand of
    # the element at forty thousand points.
    np.testing.assert_allclose(pts.mean(axis=0), el.centroid, atol=1e-3 * el.outer_radius)


def test_an_element_refuses_a_size_tuple_that_is_not_its_own():
    with pytest.raises(ValueError, match="unknown element shape"):
        Element(shape="ellipse", center_m=CENTER, normal=TILTED, size=(1e-3,))
    with pytest.raises(ValueError, match=r"takes 1 size value"):
        Element(shape="disc", center_m=CENTER, normal=TILTED, size=(1e-3, 2e-3))
    with pytest.raises(ValueError, match="radius must be > 0"):
        disc(CENTER, TILTED, -1e-3)
    with pytest.raises(ValueError, match="r_out > r_in"):
        ring_sector(CENTER, TILTED, 3e-3, 1e-3)
    with pytest.raises(ValueError, match="phi1 - phi0"):
        ring_sector(CENTER, TILTED, 1e-3, 3e-3, 0.5, 0.5)
    with pytest.raises(ValueError, match="needs curvature"):
        Element(
            shape="spherical_segment",
            center_m=CENTER,
            normal=TILTED,
            size=(0.0, 3e-3, 0.0, 6.0),
        )
    with pytest.raises(ValueError, match="a 'disc' element is flat"):
        Element(shape="disc", center_m=CENTER, normal=TILTED, size=(1e-3,), curvature=10e-3)
    with pytest.raises(ValueError, match="passes the curvature centre"):
        spherical_segment(CENTER, TILTED, 5e-3, 6e-3)


# ------------------------------------------------------------------- the deposit

ORIENTATIONS = {
    "on axis": np.array([0.0, 0.0, 1.0]),
    "30 degrees": _rotation([0.0, 1.0, 0.0], np.deg2rad(30.0)) @ np.array([0.0, 0.0, 1.0]),
    "55 degrees off diagonal": (
        _rotation([1.0, 1.0, 0.0], np.deg2rad(55.0)) @ np.array([0.0, 0.0, 1.0])
    ),
}


@pytest.mark.parametrize("name", ["disc", "rect", "ring_sector", "spherical_segment"])
@pytest.mark.parametrize("orientation", list(ORIENTATIONS))
def test_a_deposited_element_carries_its_own_area_at_any_orientation(name, orientation):
    """V-17 for the element model: grid weights sum to the physical area.

    This is the property a binary voxel mask cannot have, and it is what makes
    an absolute amplitude mean anything. Measured worst case over the twelve
    cases: 1.2e-16 relative.

    Read that number for what it is. With ``normalize=True``, which is what
    ``element_deposit`` uses, each point's truncated kernel is divided by its
    own sum, so a point deposits exactly the weight it was handed and the total
    is an identity once nothing falls off the grid: 1.2e-16 is float64
    round-off in the plumbing, not a fidelity measurement. What it does gate is
    the two halves the identity does not cover, that the closed-form area is
    right and that nothing is dropped. Deposit the same quadrature with
    ``normalize=False`` and the same total is 2.9e-2 off at tolerance 0.2, all
    four shapes, which is the truncation scale error normalization exists to
    remove.

    Where the element actually lies is gated by
    ``test_the_deposited_weights_lie_where_the_element_is`` below, and the
    closed-form area itself by
    ``test_every_shapes_closed_form_area_matches_an_independent_integral``.
    """
    el = four_shapes(center=np.zeros(3), normal=ORIENTATIONS[orientation])[name]
    grid = Grid(shape=(120, 120, 120), dx=0.25e-3)
    dep = element_deposit(el, grid, (60, 60, 60), upsampling=10)
    assert dep.dropped == 0.0, "the element hung off the grid; the gate below would be vacuous"
    assert float(dep.weights.sum()) * grid.dx**2 == pytest.approx(el.area, rel=1e-3)


@pytest.mark.parametrize("name", ["disc", "rect", "ring_sector", "spherical_segment"])
def test_the_deposited_weights_lie_where_the_element_is(name):
    """The area gate cannot see WHERE the surface is; this can.

    A normalized deposit carries the requested total whatever point set it was
    handed, so a quadrature that sampled a disc at half its true radius would
    still pass V-17 above. The first two moments of the deposited weights are
    compared here with the same moments of the independent Gauss-Legendre
    surface: the weighted centroid to within 0.05 voxels (measured 0.001 to
    0.005) and the radius of gyration to 1 % (measured 0.12 % to 0.41 %, the
    residual being the band-limited halo, which spreads weight outward and can
    only widen the figure).
    """
    el = four_shapes(center=np.zeros(3))[name]
    grid = Grid(shape=(160, 160, 160), dx=0.25e-3)
    origin = np.array([80, 80, 80], float)
    dep = element_deposit(el, grid, (80, 80, 80), upsampling=10)
    assert dep.dropped == 0.0

    pts, areas = gauss_surface(el)
    c_ref = (pts * areas[:, None]).sum(0) / areas.sum()
    rg_ref = np.sqrt((((pts - c_ref) ** 2).sum(1) * areas).sum() / areas.sum())

    pos = (dep.indices.astype(np.float64) - origin) * grid.dx
    w = dep.weights
    c = (pos * w[:, None]).sum(0) / w.sum()
    rg = np.sqrt((((pos - c) ** 2).sum(1) * w).sum() / w.sum())

    assert np.linalg.norm(c - c_ref) < 0.05 * grid.dx, f"centroid off by {c - c_ref}"
    assert rg / rg_ref == pytest.approx(1.0, rel=0.01), f"radius of gyration {rg / rg_ref:.5f}"


def test_an_element_thinner_than_a_voxel_is_refused_with_the_dx_that_would_work():
    grid = Grid(shape=(64, 64, 64), dx=0.5e-3)
    thin = rect(np.zeros(3), [0.0, 0.0, 1.0], 4e-3, 0.2e-3)
    with pytest.raises(ValueError, match="Refine dx"):
        element_deposit(thin, grid, (32, 32, 32))


def test_two_elements_over_one_voxel_superpose_as_phasors():
    """Two coincident half-strength elements are one full-strength element.

    The v1 voxelizer kept the first element's phase and dropped the other's
    drive. Here the coherent sum over the grid has to reproduce the analytic
    phasor sum of the element drives, which is what superposition means.
    """
    grid = Grid(shape=(96, 96, 96), dx=DX)
    left = disc((-1e-3, 0.0, 0.0), [0.0, 0.0, 1.0], 3e-3, phase=0.0)
    right = disc((1e-3, 0.0, 0.0), [0.0, 0.0, 1.0], 3e-3, phase=1.1)
    td = Transducer((left, right))
    src = td.deposit_cw(grid, (48, 48, 20), f0=F0, amplitude=AMP).source
    got = np.sum(src.drive_weights.astype(np.float64) * np.exp(-1j * src.phases.astype(np.float64)))
    area_grid = np.pi * (3e-3 / grid.dx) ** 2
    want = area_grid * (1.0 + np.exp(-1j * 1.1))
    assert got == pytest.approx(want, rel=2e-3)


def test_a_continuous_wave_folds_a_delay_into_the_phase_and_a_transient_does_not():
    """A delay and a phase are the same thing on the carrier and only there.

    ``deposit_cw`` may therefore fold ``omega * delay`` into the voxel phase;
    ``deposit_transient`` may not, and keeps one row per element and voxel with
    the delay attached.
    """
    grid = Grid(shape=(96, 96, 96), dx=DX)
    delay = 0.37 / F0
    by_delay = Transducer((disc(np.zeros(3), [0.0, 0.0, 1.0], 3e-3, delay_s=delay),))
    by_phase = Transducer(
        (disc(np.zeros(3), [0.0, 0.0, 1.0], 3e-3, phase=2.0 * np.pi * F0 * delay),)
    )
    a = by_delay.deposit_cw(grid, (48, 48, 20), f0=F0, amplitude=AMP).source
    b = by_phase.deposit_cw(grid, (48, 48, 20), f0=F0, amplitude=AMP).source
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_allclose(a.phases, b.phases, atol=1e-6)

    tr = deposit_transient(by_delay, grid, (48, 48, 20))
    assert tr.n_elements == 1
    np.testing.assert_allclose(tr.delays_s, delay)
    np.testing.assert_allclose(tr.phases, 0.0)
    # The transient rows collapsed on the carrier are the CW deposit again.
    collapsed: dict[tuple[int, ...], complex] = {}
    for idx, w, d, p in zip(
        map(tuple, tr.indices.tolist()), tr.weights, tr.delays_s, tr.phases, strict=True
    ):
        collapsed[idx] = collapsed.get(idx, 0j) + w * np.exp(-1j * (p + 2 * np.pi * F0 * d))
    cw = {
        tuple(i): w * np.exp(-1j * ph)
        for i, w, ph in zip(
            a.indices.tolist(),
            a.drive_weights.astype(np.float64),
            a.phases.astype(np.float64),
            strict=True,
        )
    }
    assert set(collapsed) == set(cw)
    diff = max(abs(collapsed[k] - cw[k]) for k in cw)
    assert diff < 1e-5 * max(abs(v) for v in cw.values())


def test_a_transient_deposit_keeps_two_delays_at_one_voxel_apart():
    """The reason the transient path exists at all."""
    grid = Grid(shape=(96, 96, 96), dx=DX)
    td = Transducer(
        (
            disc((-0.5e-3, 0.0, 0.0), [0.0, 0.0, 1.0], 3e-3, delay_s=0.0),
            disc((0.5e-3, 0.0, 0.0), [0.0, 0.0, 1.0], 3e-3, delay_s=1e-6),
        )
    )
    tr = deposit_transient(td, grid, (48, 48, 20))
    shared = set(map(tuple, tr.indices[tr.element == 0].tolist())) & set(
        map(tuple, tr.indices[tr.element == 1].tolist())
    )
    assert shared, "the two discs overlap; a test that assumed they did not proves nothing"
    assert sorted(set(tr.delays_s.tolist())) == [0.0, 1e-6]


def test_apodizing_one_element_to_zero_removes_exactly_its_own_area():
    grid = Grid(shape=(96, 96, 96), dx=DX)
    els = tuple(disc((x, 0.0, 0.0), [0.0, 0.0, 1.0], 2e-3) for x in (-5e-3, 0.0, 5e-3, 10e-3))
    td = Transducer(els)
    full = td.deposit_cw(grid, (40, 48, 20), f0=F0, amplitude=AMP).source
    off = (
        td.with_drive(amplitudes=[1.0, 1.0, 0.0, 1.0])
        .deposit_cw(grid, (40, 48, 20), f0=F0, amplitude=AMP)
        .source
    )

    def coherent(src):
        return np.sum(
            src.drive_weights.astype(np.float64) * np.exp(-1j * src.phases.astype(np.float64))
        )

    area_grid = np.pi * (2e-3 / grid.dx) ** 2
    assert abs(coherent(full)) - abs(coherent(off)) == pytest.approx(area_grid, rel=2e-3)


def test_a_silenced_element_leaves_no_voxels_behind():
    """Apodizing to zero removes the element from the source, not only its area.

    The coherent-sum gate above cannot see a voxel carrying weight exactly 0,
    so it passed while a silenced element still owned 550 of 2499 voxels and
    ``n_elements_represented`` still counted it. A job report prints both of
    those numbers as a description of what radiates, so they have to describe
    the driven elements and nothing else: a three-disc transducer with the
    middle one switched off must deposit exactly what the two-disc transducer
    deposits, voxel for voxel and phasor for phasor.
    """
    grid = Grid(shape=(96, 96, 96), dx=DX)
    els = tuple(disc((x, 0.0, 0.0), [0.0, 0.0, 1.0], 2e-3) for x in (-5e-3, 0.0, 5e-3))
    off = (
        Transducer(els)
        .with_drive(amplitudes=[1.0, 0.0, 1.0])
        .deposit_cw(grid, (40, 48, 20), f0=F0, amplitude=AMP)
    )
    two = Transducer((els[0], els[2])).deposit_cw(grid, (40, 48, 20), f0=F0, amplitude=AMP)
    full = Transducer(els).deposit_cw(grid, (40, 48, 20), f0=F0, amplitude=AMP)

    assert off.source.n_points < full.source.n_points
    assert off.n_elements_represented == 2
    assert not np.any(off.source.drive_weights == 0.0)
    np.testing.assert_array_equal(off.source.indices, two.source.indices)

    def drive(src):
        return src.drive_weights.astype(np.float64) * np.exp(-1j * src.phases.astype(np.float64))

    np.testing.assert_allclose(drive(off.source), drive(two.source), rtol=0, atol=0)

    tr = (
        Transducer(els).with_drive(amplitudes=[1.0, 0.0, 1.0]).deposit_transient(grid, (40, 48, 20))
    )
    assert set(np.unique(tr.element).tolist()) == {0, 2}
    assert tr.n_elements == 3


def test_a_transducer_with_every_element_silenced_is_refused():
    """All zeros is a drive mistake, and the message says which one."""
    grid = Grid(shape=(96, 96, 96), dx=DX)
    els = tuple(disc((x, 0.0, 0.0), [0.0, 0.0, 1.0], 2e-3) for x in (-5e-3, 5e-3))
    td = Transducer(els).with_drive(amplitudes=[0.0, 0.0])
    with pytest.raises(ValueError, match="apodized to zero"):
        td.deposit_cw(grid, (40, 48, 20), f0=F0, amplitude=AMP)
    with pytest.raises(ValueError, match="apodized to zero"):
        td.deposit_transient(grid, (40, 48, 20))


def test_a_silenced_element_is_still_graded_against_the_voxel_size():
    """Switching an element off must not hide a geometry the grid cannot carry."""
    grid = Grid(shape=(96, 96, 96), dx=DX)
    els = (
        disc((-5e-3, 0.0, 0.0), [0.0, 0.0, 1.0], 2e-3),
        disc((5e-3, 0.0, 0.0), [0.0, 0.0, 1.0], 0.1 * DX),
    )
    td = Transducer(els).with_drive(amplitudes=[1.0, 0.0])
    with pytest.raises(ValueError, match="narrowest"):
        td.deposit_cw(grid, (40, 48, 20), f0=F0, amplitude=AMP)


def test_the_drop_warning_names_the_callers_own_line():
    """A warning about the caller's grid has to point at the caller's line.

    ``deposit_cw`` is reached three ways, so the fixed ``stacklevel`` this
    warning used to carry attributed two of them to library code: a user who
    put a transducer half outside the domain was shown a line of
    ``transducers/model.py``. The stack level is now walked out to the first
    frame outside the package, so all three routes name this file.
    """
    grid = Grid(shape=(64, 64, 64), dx=0.5e-3)
    td = Transducer((disc((0.0, 0.0, 0.0), [0.0, 0.0, 1.0], 6e-3),))
    arr = archimedean_spiral(n_elements=16, d_outer=0.030, d_inner=0.010, roc=0.030)
    here = str(pathlib.Path(__file__).resolve())

    for call in (
        lambda: td.deposit_cw(grid, (3, 32, 32), f0=F0, amplitude=AMP),
        lambda: deposit_cw(td, grid, (3, 32, 32), F0, AMP),
        lambda: arr.voxelize(grid, (1, 32, 32), f0=F0, amplitude=AMP),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        assert caught, "the transducer hangs off the grid and nothing was said"
        assert str(pathlib.Path(caught[0].filename).resolve()) == here, caught[0].filename


# -------------------------------------------------------------- the transducer


def test_a_transducer_reports_the_geometry_a_report_has_to_print():
    els = tuple(
        spherical_segment(
            np.zeros(3),
            [0.0, 0.0, 1.0],
            30e-3,
            15e-3,
            r_in=10e-3,
            phi0=j * np.pi / 2,
            phi1=(j + 1) * np.pi / 2,
        )
        for j in range(4)
    )
    td = Transducer(els, meta=TransducerMeta(name="quad", source="test", nominal=True))
    assert td.n_elements == 4
    assert td.area == pytest.approx(sum(e.area for e in els))
    assert td.aperture == pytest.approx(30e-3, rel=1e-12)
    np.testing.assert_allclose(td.focus, [0.0, 0.0, 30e-3], atol=1e-12)
    assert td.label().startswith("quad: ")
    table = td.element_table()
    assert table["center_m"].shape == (4, 3)
    np.testing.assert_allclose(table["curvature_m"], 30e-3)
    assert table["shape"] == ["spherical_segment"] * 4


def test_a_flat_probe_has_no_geometric_focus_and_says_so():
    """A least-squares intersection of parallel axes is a huge number, not a
    focus, so the model answers None instead of inventing one."""
    els = tuple(
        rect((x, 0.0, 0.0), [0.0, 0.0, 1.0], 0.4e-3, 5e-3) for x in np.linspace(-8e-3, 8e-3, 32)
    )
    assert Transducer(els).focus is None


def test_a_placed_transducer_moves_rigidly_and_deposits_the_same_area():
    els = four_shapes(center=np.zeros(3), normal=[0.0, 0.0, 1.0])
    td = Transducer(tuple(els.values()))
    frame = Frame(origin_m=[2e-3, -1e-3, 3e-3], rotation=_rotation([0.0, 1.0, 0.0], 0.4))
    placed = td.placed(frame)
    world = placed.world_elements()
    np.testing.assert_allclose(
        world[0].center_m, frame.rotation @ td.elements[0].center_m + frame.origin_m, atol=1e-15
    )
    assert sum(e.area for e in world) == pytest.approx(td.area, rel=1e-12)

    # Both placements deposit the same total surface. Compared as the COHERENT
    # sum, because these four elements sit on top of each other and the stored
    # weights are magnitudes: the interpolant's negative side-lobes come back
    # as a positive weight at phase pi, so summing magnitudes counts them twice.
    grid = Grid(shape=(160, 160, 160), dx=0.25e-3)

    def coherent_area(source):
        drive = source.drive_weights.astype(np.float64) * np.exp(
            -1j * source.phases.astype(np.float64)
        )
        return float(abs(drive.sum())) * grid.dx**2

    here = td.deposit_cw(grid, (80, 80, 60), f0=F0, amplitude=AMP).source
    there = placed.deposit_cw(grid, (80, 80, 60), f0=F0, amplitude=AMP).source
    assert coherent_area(here) == pytest.approx(td.area, rel=1e-3)
    assert coherent_area(there) == pytest.approx(td.area, rel=1e-3)


def test_a_frame_refuses_a_mirror_and_a_non_rotation():
    with pytest.raises(ValueError, match="orthonormal"):
        Frame(rotation=np.diag([1.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="right-handed"):
        Frame(rotation=np.diag([1.0, 1.0, -1.0]))


def test_with_drive_checks_its_lengths():
    td = Transducer((disc(np.zeros(3), [0.0, 0.0, 1.0], 1e-3),))
    with pytest.raises(ValueError, match=r"amplitudes must be \(1,\)"):
        td.with_drive(amplitudes=[1.0, 1.0])


# ------------------------------------------------------------ the Rayleigh side


@pytest.mark.parametrize("name", ["disc", "rect", "ring_sector", "spherical_segment"])
def test_the_rayleigh_reference_converges_to_an_independent_surface_integral(name):
    """The reference is graded before anything is graded against it.

    ``Transducer.rayleigh_field`` integrates over the model's own equal-area
    quadrature; here the same integral is taken over a 160x160 Gauss-Legendre
    lattice built from the shape's parametrization. Measured relative
    difference at four field points, over quadrature densities of 12, 48 and
    192 points per wavelength:

        disc               1.59e-2  1.62e-3  3.99e-5
        rect               2.75e-2  3.45e-3  1.28e-4
        ring_sector        1.69e-2  1.95e-3  5.38e-5
        spherical_segment  1.87e-2  1.38e-3  1.17e-4
    """
    el = four_shapes()[name]
    field = np.array(
        [[0.0, 0.0, 20e-3], [4e-3, 3e-3, 25e-3], [-6e-3, 2e-3, 12e-3], [1e-3, -8e-3, 40e-3]]
    )
    pts, areas = gauss_surface(el)
    want = rayleigh_pressure(pts, areas, 1.0, field, k=2 * np.pi * F0 / C0, rho=RHO0, c=C0)
    td = Transducer((el,))
    errors = [
        float(
            np.abs(
                td.rayleigh_field(field, F0, c0=C0, rho=RHO0, u0=1.0, points_per_wavelength=ppw)
                - want
            ).max()
            / np.abs(want).max()
        )
        for ppw in (12.0, 48.0, 192.0)
    ]
    assert errors[0] < 5e-2
    assert errors[2] < 5e-4
    assert errors[2] < errors[0], "the quadrature did not converge to the reference"


def _sidelobe_db(profile):
    """Level of the highest secondary maximum, in dB below the peak."""
    profile = np.asarray(profile, np.float64)
    i_peak = int(np.argmax(profile))
    best = 0.0
    for step in (+1, -1):
        i = i_peak
        while 0 < i < profile.size - 1 and profile[i + step] < profile[i]:
            i += step
        j = i
        while 0 < j < profile.size - 1 and profile[j + step] > profile[j]:
            j += step
        best = max(best, float(profile[j]))
    return 20.0 * np.log10(best / profile[i_peak])


def test_a_hann_apodized_linear_probe_drops_its_sidelobe_by_the_analytic_amount():
    """Apodization, graded against the textbook aperture-window numbers.

    A uniformly driven continuous line aperture has its first sidelobe at
    -13.26 dB and a Hann-weighted one at -31.47 dB, so the reduction to expect
    is 18.21 dB. Measured here on 32 rectangular elements at half-wavelength
    pitch, focused at f/2.5: -13.13 dB uniform, -30.52 dB Hann, a reduction of
    17.38 dB, 0.83 dB short of the continuous-aperture figure. The shortfall is
    the window's own sampling and not the model: the point-source array factor
    of the same 32 positions and weights gives 17.18 dB, which the model
    tracks to 0.21 dB.

    Those three fields are all surface integrals, so the last block repeats the
    measurement on the GRID DEPOSIT: the voxels are propagated as point sources
    carrying their own stored weight and phase. That is the half of the
    criterion an integral over the element surfaces cannot state, and it reads
    -13.151 dB and -30.548 dB, a reduction of 17.397 dB.
    """
    f0 = 1.5e6
    lam = C0 / f0
    n, pitch, kerf, elevation = 32, lam / 2, 0.05e-3, 5e-3
    focus = np.array([0.0, 0.0, 40e-3])
    x_e = (np.arange(n) - (n - 1) / 2.0) * pitch

    def probe(amplitudes):
        els = tuple(
            rect((xi, 0.0, 0.0), [0.0, 0.0, 1.0], pitch - kerf, elevation, amplitude=float(a))
            for xi, a in zip(x_e, amplitudes, strict=True)
        )
        td = Transducer(els)
        return td.with_drive(phases=td.das_phases(focus, f0, C0))

    x = np.linspace(-15e-3, 15e-3, 3001)
    field = np.column_stack([x, np.zeros_like(x), np.full_like(x, focus[2])])
    uniform = np.ones(n)
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * (np.arange(n) + 0.5) / n))

    levels = {}
    for label, amps in (("uniform", uniform), ("hann", hann)):
        pressure = np.abs(
            probe(amps).rayleigh_field(
                field, f0, c0=C0, rho=RHO0, u0=1.0, points_per_wavelength=12.0
            )
        )
        levels[label] = _sidelobe_db(pressure)

    assert levels["uniform"] == pytest.approx(-13.26, abs=0.5)
    reduction = levels["uniform"] - levels["hann"]
    assert reduction == pytest.approx(-13.26 - (-31.47), abs=1.0)

    # ...and against the array factor of the same 32 point drives, which is the
    # sharper statement because it carries the window's sampling too.
    k = 2 * np.pi / lam

    def array_factor(amps):
        d = np.sqrt((x[:, None] - x_e[None, :]) ** 2 + focus[2] ** 2)
        d0 = np.sqrt(x_e**2 + focus[2] ** 2)
        return np.abs((amps * np.exp(1j * k * (d - d0[None, :]))).sum(axis=1))

    analytic = _sidelobe_db(array_factor(uniform)) - _sidelobe_db(array_factor(hann))
    assert reduction == pytest.approx(analytic, abs=0.5)

    # ...and once more from the GRID DEPOSIT rather than the surface integral,
    # which is the half of the criterion the two fields above cannot reach: the
    # deposited voxels are propagated as point sources of their own stored
    # drive (weight and phase), so this fails if apodization does not survive
    # voxelization. Measured -13.151 dB uniform and -30.548 dB Hann, a
    # reduction of 17.397 dB, 0.015 dB from the surface integral's 17.382.
    dx = lam / 6
    grid = Grid(shape=(144, 64, 16), dx=dx)
    origin = np.array([72, 32, 4])
    deposited = {}
    for label, amps in (("uniform", uniform), ("hann", hann)):
        src = probe(amps).deposit_cw(grid, tuple(origin), f0=f0, amplitude=1.0).source
        pos = (src.indices.astype(np.float64) - origin) * dx
        # rayleigh_pressure propagates with exp(+i k R), so the deposit's
        # stored phasor is conjugated here, as Transducer.rayleigh_field does.
        drive = src.drive_weights.astype(np.float64) * np.exp(1j * src.phases.astype(np.float64))
        pressure = np.empty(len(field), np.complex128)
        for lo in range(0, len(field), 128):  # chunked: 17643 voxels x 1501 points
            block = field[lo : lo + 128]
            distance = np.linalg.norm(block[:, None, :] - pos[None, :, :], axis=2)
            pressure[lo : lo + 128] = ((drive / distance) * np.exp(1j * k * distance)).sum(axis=1)
        deposited[label] = _sidelobe_db(np.abs(pressure))
    assert deposited["uniform"] == pytest.approx(levels["uniform"], abs=0.1)
    assert deposited["uniform"] - deposited["hann"] == pytest.approx(reduction, abs=0.2)


def test_the_package_docstring_example_runs_and_stays_inside_its_grid():
    """The four lines in ``caustica.transducers.__doc__`` are executable.

    A docstring is API documentation, so it is graded like the rest: the
    example is parsed out of the module and run, and it has to deposit
    without the halo hanging off the domain (which would raise here, because
    the drop warning is promoted to an error).
    """
    import textwrap
    import warnings

    import caustica.transducers as pkg

    lines = pkg.__doc__.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("import numpy"))
    stop = next(i for i, ln in enumerate(lines[start:], start) if ln.strip().startswith("src ="))
    body = textwrap.dedent("".join(lines[start : stop + 1]))
    assert "deposit_cw" in body

    scope: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", CausticaWarning)
        exec(compile(body, "<transducers docstring>", "exec"), scope)  # noqa: S102
    src = scope["src"]
    assert src.n_points > 0
    assert scope["probe"].n_elements == 64


# ------------------------------------------------------- the v1 path underneath


def test_the_flat_piston_source_and_a_one_element_transducer_are_the_same_source():
    """``disc_cw_source`` predates the element model; the model reproduces it.

    Reproduced as the signed complex drive, because the two store it
    differently: the piston keeps the interpolant's negative side-lobes as
    negative weights at phase zero, while the transducer deposit stores a
    magnitude and an angle and therefore books those side-lobes as positive
    weights at phase pi. The physics is ``w * exp(-i phi)`` either way, and
    that agrees to 8.6e-9 relative.
    """
    grid = Grid(shape=(100, 88, 128), dx=DX, pml=PMLSpec(thickness=4.5e-3))
    radius, origin = 9e-3, (44, 40, 16)
    piston = disc_cw_source(grid, f0=F0, amplitude=AMP, radius=radius, center_vox=origin)
    modelled = (
        Transducer((disc(np.zeros(3), [0.0, 0.0, 1.0], radius),))
        .deposit_cw(grid, origin, f0=F0, amplitude=AMP)
        .source
    )
    np.testing.assert_array_equal(piston.indices, modelled.indices)

    def drive(src):
        return src.drive_weights.astype(np.float64) * np.exp(-1j * src.phases.astype(np.float64))

    a, b = drive(piston), drive(modelled)
    assert float(np.abs(a - b).max() / np.abs(a).max()) < 1e-6


def test_the_focused_bowl_source_is_one_spherical_segment_element():
    """``bowl_cw_source`` predates the element model too, and is a special case.

    A whole bowl is ``spherical_segment(apex, +z, roc, aperture)``: one
    element, a full turn, no inner radius. The two reach the same voxels
    through different quadratures (the cap sampler against the element
    lattice), so this is not an identity of construction but a measurement.
    Measured on an f/1.25 cap at dx = 0.5 mm: the same 3877 voxels, the
    largest per-voxel drive difference 2.0e-8 of the peak voxel, and coherent
    totals of 472.095768 grid squares apiece, agreeing to 1.6e-15.
    """
    g = Grid(shape=(64, 64, 80), dx=0.5e-3)
    aperture, roc, apex = 6e-3, 15e-3, (32, 32, 10)
    old = bowl_cw_source(g, f0=F0, amplitude=AMP, aperture_radius=aperture, roc=roc, apex_vox=apex)
    modelled = (
        Transducer((spherical_segment(np.zeros(3), [0.0, 0.0, 1.0], roc, aperture),))
        .deposit_cw(g, apex, f0=F0, amplitude=AMP)
        .source
    )
    np.testing.assert_array_equal(old.indices, modelled.indices)

    def drive(src):
        return src.drive_weights.astype(np.float64) * np.exp(-1j * src.phases.astype(np.float64))

    a, b = drive(old), drive(modelled)
    assert float(np.abs(a - b).max() / np.abs(a).max()) < 1e-6
    assert abs(a.sum()) == pytest.approx(abs(b.sum()), rel=1e-9)
    cap_area = 2.0 * np.pi * roc**2 * (1.0 - np.sqrt(1.0 - (aperture / roc) ** 2))
    assert abs(b.sum()) * g.dx**2 == pytest.approx(cap_area, rel=1e-3)


def test_the_spiral_deposit_reproduces_the_v1_off_grid_numbers():
    """The v1 array voxelizer's own numbers, pinned before it was deleted.

    ``TransducerArray.voxelize`` now runs through the element model. The
    constants below were measured from the pre-v2 ``_offgrid`` implementation
    at ``ddd07b0`` on this 32-element spiral, and a full per-voxel comparison
    against that implementation gave a relative difference of exactly 0 with
    uniform phases and 1.8e-7 with delay-and-sum phases. That residual is the
    v1 path's float32 phase cast, which v2 does in float64 (float32 epsilon is
    1.2e-7): it is invisible in the totals, which agree to 1.2e-9, and shows up
    only in the de-rotated total below, which multiplies each voxel by the
    exact float64 phase and therefore isolates the cast at 3.1e-6.
    """
    arr = archimedean_spiral(n_elements=32, d_outer=0.030, d_inner=0.010, roc=0.030)
    grid = Grid(shape=(96, 96, 96), dx=0.5e-3, pml=PMLSpec(thickness=5e-3))
    apex = (48, 48, 12)
    phases = arr.das_phases(np.array([3e-3, -2e-3, 0.026]), f0=1.0e6, c0=C0)

    for label, ph, n_points, coherent, magnitude, derotated in (
        ("uniform", None, 13721, 1628.437326, 2625.599313, 1628.437326),
        ("das", phases, 13721, 50.751237, 2539.089661, 1542.094854),
    ):
        asrc = arr.voxelize(grid, apex, f0=1.0e6, amplitude=AMP, phases=ph)
        src = asrc.source
        w = src.drive_weights.astype(np.float64)
        d = w * np.exp(-1j * src.phases.astype(np.float64))
        own = (np.zeros(arr.n_elements) if ph is None else np.asarray(ph, np.float64))[
            asrc.element_of_voxel
        ]
        assert src.n_points == n_points, label
        assert float(np.abs(d.sum())) == pytest.approx(coherent, rel=1e-6), label
        assert float(w.sum()) == pytest.approx(magnitude, rel=1e-6), label
        # De-rotating each voxel by its owner's drive phase turns the phased
        # case back into a non-cancelling total, so the pin above is not the
        # small difference of two large numbers.
        assert float(abs(np.sum(d * np.exp(1j * own)))) == pytest.approx(derotated, rel=1e-5), label


# ------------------------------------------------------------------ V-11, slow

ROC, APER = 30e-3, 15e-3
TARGET = np.array([5e-3, 0.0, ROC])
V11_SHAPE = (136, 120, 136)
V11_APEX = (64, 60, 16)


def _equal_area_radii(n, r_max, roc=None):
    """Radii cutting a disc (or a spherical cap) into ``n`` equal-area rings."""
    if roc is None:
        return r_max * np.sqrt(np.arange(n + 1) / n)
    depth = (1.0 - np.sqrt(1.0 - (r_max / roc) ** 2)) * np.arange(n + 1) / n
    return roc * np.sqrt(np.clip(1.0 - (1.0 - depth) ** 2, 0.0, None))


def _shell_seats():
    """Sixteen seats on the focusing shell: two rings of eight, staggered."""
    for ring, r_ring in enumerate((APER * 0.45, APER * 0.82)):
        z = ROC - np.sqrt(ROC**2 - r_ring**2)
        for j in range(8):
            angle = 2.0 * np.pi * j / 8 + (np.pi / 8 if ring else 0.0)
            seat = np.array([r_ring * np.cos(angle), r_ring * np.sin(angle), z])
            yield seat, np.array([0.0, 0.0, ROC]) - seat


def sixteen_elements(name):
    """A sixteen-element array of one shape, all with the same f/1 aperture."""
    axis = np.array([0.0, 0.0, 1.0])
    phi = np.linspace(0.0, 2.0 * np.pi, 9)
    if name == "disc":
        return tuple(disc(p, n, APER * 0.17) for p, n in _shell_seats())
    if name == "rect":
        return tuple(rect(p, n, APER * 0.27, APER * 0.27) for p, n in _shell_seats())
    if name == "ring_sector":
        r = _equal_area_radii(2, APER)
        return tuple(
            ring_sector(np.zeros(3), axis, r[i], r[i + 1], phi[j], phi[j + 1])
            for i in range(2)
            for j in range(8)
        )
    r = _equal_area_radii(2, APER, ROC)
    return tuple(
        spherical_segment(np.zeros(3), axis, ROC, r[i + 1], r_in=r[i], phi0=phi[j], phi1=phi[j + 1])
        for i in range(2)
        for j in range(8)
    )


#: V-11's peak-ratio gate at 8 points per wavelength, the same number for every
#: shape. It is NOT widened for the shape that misses it: the flat annular array
#: reads 1.0253 and its case below is a strict xfail, so the suite records the
#: gate as failed and fails loudly the day the engine is fixed.
V11_RATIO_TOL = 0.02

#: Why the flat annular array misses it, measured on the same grid, rung and
#: comparison: the library's own ``disc_cw_source`` (a 15 mm piston with no
#: element model anywhere in it) reads 1.0340 at 8 points per wavelength and
#: 1.0173 at 16, while a one-element curved bowl of the same aperture reads
#: 1.0136 and 1.0014. The annular array sits below the flat control at both
#: rungs and above the curved one, so the excess belongs to the absolute
#: amplitude of a flat source lying in one grid plane and not to the element
#: model. Re-evaluating the same solved fields with the Rayleigh quadrature at
#: 8, 16, 32 and 64 points per wavelength moves the reference by at most 4e-4,
#: so it is not reference quadrature error either.
V11_RING_SECTOR_XFAIL = (
    "V-11 peak ratio 1.0253 against 1.00 +/- 0.02 for a flat in-plane array. "
    "The engine's absolute amplitude for a flat source, not the element model: "
    "disc_cw_source reads 1.0340 on the same grid and rung. Delete this marker "
    "when that is fixed."
)


@pytest.mark.slow
@pytest.mark.parametrize(
    "name",
    [
        "disc",
        "rect",
        pytest.param(
            "ring_sector",
            marks=pytest.mark.xfail(strict=True, reason=V11_RING_SECTOR_XFAIL),
        ),
        "spherical_segment",
    ],
)
def test_a_steered_sixteen_element_array_radiates_what_the_rayleigh_integral_says(name):
    """V-11, one array per element shape, at the rung the gate is stated at.

    Sixteen elements of one shape on (or under) an f/1 aperture, 30 mm radius
    of curvature, delay-and-sum phased to (5, 0, 30) mm, solved with the linear
    k-space solver at 8 points per wavelength and compared with the Rayleigh
    integral over the SAME element surfaces. Measured 2026-09-06 on both the
    numpy and the cupy backend, which agree to four decimals:

        shape              ratio   lateral   axial   at ref   SLL solve  SLL ref
        disc              1.0100   0.000mm  0.375mm  0.9962    -13.371   -13.347
        rect              1.0118   0.000mm  0.375mm  0.9928    -13.489   -13.461
        ring_sector       1.0253   0.000mm  0.375mm  0.9875    -11.643   -11.762
        spherical_segment 1.0144   0.000mm  0.375mm  0.9954    -11.615   -11.581

    and, at 16 points per wavelength, ratios of 0.9998, 0.9993, 1.0160 and
    1.0049: the excess is the engine's own absolute-amplitude error at a coarse
    grid (``validation.analytic_suite.measure_absolute_amplitude`` measures it
    as ~3.7 ppw^-2.5 for a bowl), not the source model. The ``ring_sector`` row
    fails the peak-ratio gate and its case is a strict xfail; the other three
    gates run for it too, which is why the ratio is asserted last.

    The ratios carry a systematic of about 2e-3 from settle detection: dropping
    ``reference_point`` from the run moves ring_sector at 16 points per
    wavelength from 1.0160 to 1.0184 and the flat control from 1.0177 to
    1.0191. That is inside V-11's declared +/- 0.5 % repeatability, but it is
    why the fourth decimal here is not a promise.

    The axial offset is one voxel for every shape at both rungs and does not
    shrink in millimetres, because at this f-number the axial lobe is flat: the
    Rayleigh field varies by 0.1 % over +/- 1.5 mm about its own maximum. So
    the axial gate here is that the solver's field at the reference's peak
    voxel is within 2 % of the solver's own peak, and the lateral gate is
    V-11's dx/2.
    """
    td = Transducer(sixteen_elements(name))
    td = td.with_drive(phases=td.das_phases(TARGET, F0, C0))
    grid = Grid(shape=V11_SHAPE, dx=DX, pml=PMLSpec(thickness=4.5e-3))
    medium = Medium.homogeneous(grid.shape, water(c=C0))
    source = td.deposit_cw(grid, V11_APEX, f0=F0, amplitude=AMP).source
    spec = CWRunSpec(min_settle_periods=8, max_settle_periods=30, n_record_periods=2)
    reference_point = (
        V11_APEX[0] + int(round(TARGET[0] / DX)),
        V11_APEX[1],
        V11_APEX[2] + int(round(ROC / DX)),
    )
    result = solvers.get("linear")().run(
        grid, medium, source, spec, backend="auto", reference_point=reference_point
    )
    amp = np.asarray(result.amp, np.float64)

    pad = grid.pml_vox + 2
    window = (
        slice(pad, V11_SHAPE[0] - pad),
        slice(pad, V11_SHAPE[1] - pad),
        slice(V11_APEX[2] + int(round(0.5 * ROC / DX)), V11_SHAPE[2] - pad),
    )
    inner = np.unravel_index(int(np.argmax(amp[window])), amp[window].shape)
    peak = tuple(int(w.start + i) for w, i in zip(window, inner, strict=True))

    half = 10
    axes = [np.arange(p - half, p + half + 1) for p in peak]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([(m.ravel() - o) * DX for m, o in zip(mesh, V11_APEX, strict=True)])
    reference = np.abs(
        td.rayleigh_field(
            points, F0, c0=C0, rho=RHO0, u0=AMP / (RHO0 * C0), points_per_wavelength=8.0
        )
    ).reshape(mesh[0].shape)
    i_ref = np.unravel_index(int(np.argmax(reference)), reference.shape)
    assert all(0 < i < n - 1 for i, n in zip(i_ref, reference.shape, strict=True)), (
        "the reference peaked on the edge of the comparison box; widen it"
    )
    ref_peak = tuple(int(a[i]) for a, i in zip(axes, i_ref, strict=True))

    offset = np.array(peak) - np.array(ref_peak)
    lateral = float(np.hypot(offset[0], offset[1]) * DX)
    assert lateral <= DX / 2, f"lateral focus offset {lateral * 1e3:.3f} mm"
    assert float(amp[ref_peak] / amp[peak]) > 0.98, "the two peaks are not the same lobe"

    line = np.arange(pad, V11_SHAPE[0] - pad)
    cut = np.column_stack(
        [
            (line - V11_APEX[0]) * DX,
            np.full(line.size, (ref_peak[1] - V11_APEX[1]) * DX),
            np.full(line.size, (ref_peak[2] - V11_APEX[2]) * DX),
        ]
    )
    ref_line = np.abs(
        td.rayleigh_field(cut, F0, c0=C0, rho=RHO0, u0=AMP / (RHO0 * C0), points_per_wavelength=8.0)
    )
    solved_line = amp[line, ref_peak[1], ref_peak[2]]
    delta = abs(_sidelobe_db(solved_line) - _sidelobe_db(ref_line))
    assert delta < 0.5, f"first sidelobe differs by {delta:.3f} dB"

    # Last, so that the position and sidelobe halves of V-11 are exercised for
    # every shape, including the one whose peak ratio is a known failure.
    ratio = float(amp[peak] / reference[i_ref])
    assert ratio == pytest.approx(1.0, abs=V11_RATIO_TOL), f"peak ratio {ratio:.4f}"
