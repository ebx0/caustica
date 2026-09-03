"""Source-surface discretizations for analytic integrals.

Currently: uniform-area point sampling of a spherical cap (focused bowl).
Used by the Rayleigh integral (validation + beam preview) and later by the
KZK initial-plane projection.
"""

from __future__ import annotations

import numpy as np

_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def spherical_cap_points(
    aperture_radius: float,
    roc: float,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a concave spherical cap with ~equal-area points (Fibonacci lattice).

    Geometry convention (shared with :mod:`caustica.analytic.oneill`): the bowl
    apex sits at the origin, the beam axis is +z, the geometric focus is at
    ``(0, 0, roc)``. The cap spans polar angles ``theta in [0, theta_max]``
    with ``sin(theta_max) = aperture_radius / roc``.

    Parameters
    ----------
    aperture_radius:
        Outer aperture radius ``a`` [m]; must satisfy ``a < roc``.
    roc:
        Radius of curvature (focal length) [m].
    spacing:
        Target inter-point spacing [m]; use <= lambda/4 for Rayleigh sums.
        Point count is ``ceil(cap_area / spacing**2)``.

    Returns
    -------
    points, normals, areas:
        ``(N, 3)`` positions [m], ``(N, 3)`` unit normals (toward the focus),
        ``(N,)`` per-point areas [m^2] summing to the exact cap area.
    """
    a, f = float(aperture_radius), float(roc)
    if not 0 < a < f:
        raise ValueError(f"need 0 < aperture_radius < roc, got a={a}, roc={f}")
    if spacing <= 0:
        raise ValueError(f"spacing must be > 0, got {spacing}")

    cos_tmax = np.sqrt(1.0 - (a / f) ** 2)
    cap_area = 2.0 * np.pi * f**2 * (1.0 - cos_tmax)
    n = int(np.ceil(cap_area / spacing**2))
    n = max(n, 16)

    j = np.arange(n, dtype=np.float64)
    # cos(theta) uniform in [cos_tmax, 1] -> equal solid angle == equal area.
    cos_t = 1.0 - (1.0 - cos_tmax) * (j + 0.5) / n
    sin_t = np.sqrt(np.clip(1.0 - cos_t**2, 0.0, None))
    phi = j * _GOLDEN_ANGLE

    points = np.column_stack((f * sin_t * np.cos(phi), f * sin_t * np.sin(phi), f * (1.0 - cos_t)))
    center = np.array([0.0, 0.0, f])
    normals = (center - points) / f
    areas = np.full(n, cap_area / n)
    return points, normals, areas
