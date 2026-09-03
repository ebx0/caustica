"""M3 gate: O'Neil closed form vs numerical Rayleigh integral (cross-check).

The two implementations are independent (closed-form algebra vs point-cloud
quadrature), so agreement validates BOTH: geometry conventions, the
prefactor, the focal-gain limit and the equal-area cap sampling. Criteria:
  * focal-region correlation r > 0.999 (normalized |p|),
  * peak position difference < the axial sampling step,
  * focal amplitude relative difference < 2%.
"""

import numpy as np
import pytest

from caustica.analytic import (
    axial_pressure,
    focal_gain,
    rayleigh_pressure,
    spherical_cap_points,
)

# Shared test bowl: modest f-number, exactly representable everywhere.
A = 15e-3  # aperture radius [m]
ROC = 60e-3  # radius of curvature / focal length [m]
F0 = 1.0e6  # [Hz]
C0 = 1500.0  # [m/s]
RHO0 = 1000.0  # [kg/m^3]
U0 = 0.10  # [m/s]
K = 2.0 * np.pi * F0 / C0
LAMBDA = C0 / F0


@pytest.fixture(scope="module")
def cap():
    return spherical_cap_points(A, ROC, spacing=LAMBDA / 5.0)


def test_cap_sampling_invariants(cap):
    points, normals, areas = cap
    h = ROC - np.sqrt(ROC**2 - A**2)
    cap_area = 2.0 * np.pi * ROC * h
    assert areas.sum() == pytest.approx(cap_area, rel=1e-12)
    # Every point lies on the sphere centered at the focus.
    r = np.linalg.norm(points - np.array([0.0, 0.0, ROC]), axis=1)
    np.testing.assert_allclose(r, ROC, rtol=1e-12)
    # Normals are unit and point at the focus.
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=1e-12)
    # z spans [0, h]; radial extent stays inside the aperture.
    assert points[:, 2].min() >= 0.0 and points[:, 2].max() <= h + 1e-12
    assert np.hypot(points[:, 0], points[:, 1]).max() <= A + 1e-9


def test_focal_gain_matches_kh():
    h = ROC - np.sqrt(ROC**2 - A**2)
    assert focal_gain(A, ROC, F0, C0) == pytest.approx(K * h, rel=1e-12)
    # And the closed form reproduces it at the focus.
    p_focus = axial_pressure(ROC, A, ROC, F0, C0, RHO0, U0)
    assert abs(p_focus) == pytest.approx(RHO0 * C0 * U0 * K * h, rel=1e-12)


def test_oneill_vs_rayleigh_axial_profile(cap):
    points, _normals, areas = cap
    dz = 0.2e-3
    z = np.arange(20e-3, 100e-3 + dz / 2, dz)
    targets = np.zeros((z.size, 3))
    targets[:, 2] = z

    p_oneill = np.abs(axial_pressure(z, A, ROC, F0, C0, RHO0, U0))
    p_rayleigh = np.abs(rayleigh_pressure(points, areas, U0, targets, k=K, rho=RHO0, c=C0))

    # (1) Normalized-shape correlation over the focal region [0.7F, 1.3F].
    sel = (z > 0.7 * ROC) & (z < 1.3 * ROC)
    a = p_oneill[sel] / p_oneill[sel].max()
    b = p_rayleigh[sel] / p_rayleigh[sel].max()
    r = np.corrcoef(a, b)[0, 1]
    assert r > 0.999, f"focal-region correlation too low: r={r:.6f}"

    # (2) Peak position agreement within one axial step.
    z_peak_oneill = z[np.argmax(p_oneill)]
    z_peak_rayleigh = z[np.argmax(p_rayleigh)]
    assert abs(z_peak_oneill - z_peak_rayleigh) <= dz

    # (3) Focal amplitude within 2%.
    i_focus = np.argmin(np.abs(z - ROC))
    rel = abs(p_rayleigh[i_focus] - p_oneill[i_focus]) / p_oneill[i_focus]
    assert rel < 0.02, f"focal amplitude off by {rel * 100:.2f}%"


def test_rayleigh_input_validation(cap):
    points, _normals, areas = cap
    with pytest.raises(ValueError, match="field_points"):
        rayleigh_pressure(points, areas, U0, np.zeros((3, 2)), k=K)
    with pytest.raises(ValueError, match="src_areas"):
        rayleigh_pressure(points, areas[:-1], U0, np.zeros((3, 3)), k=K)
    with pytest.raises(ValueError, match="positive real part"):
        rayleigh_pressure(points, areas, U0, np.zeros((3, 3)), k=0.0)
    # A complex k carries absorption; a NEGATIVE imaginary part would be an
    # amplifying medium, which is not a thing this integral will pretend to.
    with pytest.raises(ValueError, match="amplifying"):
        rayleigh_pressure(points, areas, U0, np.zeros((3, 3)), k=K - 5j)
    with pytest.raises(ValueError, match="coincides"):
        rayleigh_pressure(points, areas, U0, points[:1], k=K)


def test_a_complex_wavenumber_attenuates_the_rayleigh_field(cap):
    """Absorption enters as +i*alpha and decays every source-to-field path.

    Not as ``exp(-alpha z)`` applied afterwards, and the difference has no
    fixed sign: each path carries its own length, and for a CONCAVE source
    those can be shorter than the axial distance. Measured on this f/1 cap
    at 1 dB/cm: 4.6 % below the naive decay at 30 mm, 1.5 % above it at
    90 mm, crossing near the focus. A flat piston sits on the other side of
    it throughout, since its paths are never shorter than z.
    """
    points, _normals, areas = cap
    z = np.array([0.03, 0.06, 0.09])
    field = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    alpha = 11.5129  # 1 dB/cm at 500 kHz, the ITRUSST BM2 value

    lossless = np.abs(rayleigh_pressure(points, areas, U0, field, k=K))
    lossy = np.abs(rayleigh_pressure(points, areas, U0, field, k=K + 1j * alpha))

    assert np.all(lossy < lossless), "a lossy medium cannot raise the field"
    ratio = lossy / lossless
    naive = np.exp(-alpha * z)
    assert np.all(np.abs(ratio / naive - 1.0) < 0.10), "not the absorption asked for"
    assert ratio[0] / naive[0] < 1.0 < ratio[-1] / naive[-1], "path lengths cross z"


def test_oneill_input_validation():
    with pytest.raises(ValueError):
        axial_pressure(0.05, aperture_radius=0.07, roc=0.06, f0=F0)
    with pytest.raises(ValueError):
        focal_gain(0.0, 0.06, F0)
    with pytest.raises(ValueError):
        spherical_cap_points(0.07, 0.06, 1e-3)
