"""M4 headline gate: 3-D focused bowl in water vs O'Neil / Rayleigh.

Pressure-injected PSTD sources have no closed-form absolute amplitude, so
the comparison is on NORMALIZED field shapes (the notebook's validated
methodology): axial profile correlation, focal position, and -6 dB widths.
"""

import numpy as np
import pytest

import caustica.solvers as solvers
from caustica import Grid, Medium, PMLSpec
from caustica.analytic import axial_pressure, rayleigh_pressure, spherical_cap_points
from caustica.materials import water
from caustica.solvers import CWRunSpec
from caustica.sources import bowl_cw_source

F0 = 1.0e6
C0 = 1500.0
DX = 0.375e-3  # 4 ppw
A = 7.5e-3  # aperture radius (20 vox)
ROC = 18e-3  # focal length (48 vox)
APEX = (40, 40, 16)
SHAPE = (80, 80, 120)  # z long enough that the -6 dB lobe clears the PML
FOCUS_VOX = (40, 40, 16 + 48)


def minus6db_width(coord, prof):
    """Width of the -6 dB (half-amplitude) main lobe around the peak."""
    prof = np.asarray(prof, dtype=np.float64)
    i_pk = int(np.argmax(prof))
    half = prof[i_pk] / 2.0
    lo = i_pk
    while lo > 0 and prof[lo] > half:
        lo -= 1
    hi = i_pk
    while hi < prof.size - 1 and prof[hi] > half:
        hi += 1
    if lo == 0 or hi == prof.size - 1:
        raise AssertionError("-6 dB lobe not fully contained in the profile")

    def cross(i0, i1):
        f = (half - prof[i0]) / (prof[i1] - prof[i0])
        return coord[i0] + f * (coord[i1] - coord[i0])

    return cross(hi - 1, hi) - cross(lo + 1, lo)


@pytest.fixture(scope="module")
def solved():
    grid = Grid(shape=SHAPE, dx=DX, pml=PMLSpec(thickness=5e-3))  # 13 vox
    medium = Medium.homogeneous(grid.shape, water(c=C0))
    source = bowl_cw_source(grid, f0=F0, amplitude=1.0e5, aperture_radius=A, roc=ROC, apex_vox=APEX)
    spec = CWRunSpec(min_settle_periods=8, max_settle_periods=30, n_record_periods=2)
    res = solvers.get("linear")().run(
        grid, medium, source, spec, backend="numpy", reference_point=FOCUS_VOX
    )
    return res


@pytest.mark.slow
def test_focus_lands_within_one_voxel(solved):
    amp = solved.amp
    peak = np.unravel_index(np.argmax(amp), amp.shape)
    # O'Neil's true maximum sits slightly proximal of the geometric focus.
    z_phys = (np.arange(SHAPE[2]) - APEX[2]) * DX
    p_oneill = np.abs(axial_pressure(z_phys[24:100], A, ROC, F0, C0))
    z_true = z_phys[24:100][np.argmax(p_oneill)]
    z_true_vox = APEX[2] + z_true / DX
    assert abs(peak[0] - FOCUS_VOX[0]) <= 1
    assert abs(peak[1] - FOCUS_VOX[1]) <= 1
    assert abs(peak[2] - z_true_vox) <= 1.0


@pytest.mark.slow
def test_axial_profile_matches_oneill(solved):
    amp = solved.amp
    axial = amp[APEX[0], APEX[1], :]
    z_phys = (np.arange(SHAPE[2]) - APEX[2]) * DX
    # Never measure inside the sponge: stay 2 voxels clear of the PML start.
    z_pml_limit = (SHAPE[2] - 13 - 2 - APEX[2]) * DX
    sel = (z_phys > 0.5 * ROC) & (z_phys < min(1.5 * ROC, z_pml_limit))
    p_oneill = np.abs(axial_pressure(z_phys[sel], A, ROC, F0, C0))

    a = axial[sel] / axial[sel].max()
    b = p_oneill / p_oneill.max()
    r = np.corrcoef(a, b)[0, 1]
    assert r > 0.99, f"axial correlation r={r:.5f} < 0.99"

    # Width measurement needs the full distal -6 dB crossing: use the widest
    # window that stays clear of the sponge.
    sel_w = (z_phys > 0.5 * ROC) & (z_phys < z_pml_limit)
    aw = axial[sel_w] / axial[sel_w].max()
    bw = np.abs(axial_pressure(z_phys[sel_w], A, ROC, F0, C0))
    bw = bw / bw.max()
    w_solver = minus6db_width(z_phys[sel_w], aw)
    w_oneill = minus6db_width(z_phys[sel_w], bw)
    rel = abs(w_solver - w_oneill) / w_oneill
    assert rel < 0.05, f"-6dB axial width off by {rel * 100:.1f}%"


@pytest.mark.slow
def test_lateral_profile_matches_rayleigh(solved):
    amp = solved.amp
    z_pk = int(np.argmax(amp[APEX[0], APEX[1], :]))
    lateral = amp[:, APEX[1], z_pk]
    assert abs(int(np.argmax(lateral)) - APEX[0]) <= 1

    x_phys = (np.arange(SHAPE[0]) - APEX[0]) * DX
    z_phys = (z_pk - APEX[2]) * DX
    points, _n, areas = spherical_cap_points(A, ROC, spacing=C0 / F0 / 5.0)
    targets = np.column_stack([x_phys, np.zeros_like(x_phys), np.full_like(x_phys, z_phys)])
    p_ray = np.abs(rayleigh_pressure(points, areas, 1.0, targets, k=2 * np.pi * F0 / C0, c=C0))

    sel = np.abs(x_phys) < 6e-3  # main lobe + first sidelobes
    a = lateral[sel] / lateral[sel].max()
    b = p_ray[sel] / p_ray[sel].max()
    w_solver = minus6db_width(x_phys[sel], a)
    w_ray = minus6db_width(x_phys[sel], b)
    rel = abs(w_solver - w_ray) / w_ray
    assert rel < 0.05, f"-6dB lateral width off by {rel * 100:.1f}%"

    r = np.corrcoef(a, b)[0, 1]
    assert r > 0.98, f"lateral correlation r={r:.5f}"


@pytest.mark.slow
def test_run_metadata(solved):
    assert solved.meta["padded_shape"] == SHAPE  # already 2/3/5-smooth
    # roc 18 mm at 1500 m/s and 1 MHz -> 12 periods; voxel rounding of the
    # cap may push the farthest source voxel slightly beyond roc -> 13.
    assert solved.tof_periods in (12, 13)
    assert not solved.settle_capped
