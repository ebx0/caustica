"""O'Neil (1949) on-axis solution for a focused spherical-cap radiator.

Closed-form linear lossless axial pressure of a concave bowl with uniform
normal velocity ``u0`` — the exact on-axis evaluation of the Rayleigh
integral over the cap. This is validation gate #1 for every full-wave
solver in this library, and the sanity anchor for the
Rayleigh point-cloud discretization (the two must agree to <2%).

Geometry (shared with :mod:`caustica.analytic.geometry`): bowl apex at z=0,
beam axis +z, geometric focus at ``z = roc``. ``h = roc - sqrt(roc^2 - a^2)``
is the bowl depth; the rim sits in the plane ``z = h`` at radius ``a``.

Formula (``e^{-i omega t}`` convention)::

    p(z) = rho c u0 / (1 - z/roc) * (exp(i k z) - exp(i k d_rim(z)))
    d_rim(z) = sqrt(a^2 + (z - h)^2)

with the removable singularity at the focus handled analytically::

    p(roc) = -i k h rho c u0 exp(i k roc)      (=> focal gain |p|/(rho c u0) = k h)
"""

from __future__ import annotations

import numpy as np


def _check_bowl(aperture_radius: float, roc: float) -> None:
    if not 0 < aperture_radius < roc:
        raise ValueError(f"need 0 < aperture_radius < roc, got a={aperture_radius}, roc={roc}")


def axial_pressure(
    z: np.ndarray | float,
    aperture_radius: float,
    roc: float,
    f0: float,
    c0: float = 1500.0,
    rho0: float = 1000.0,
    u0: float = 1.0,
    focus_tol: float = 1e-9,
) -> np.ndarray:
    """Complex on-axis pressure ``p(z)`` [Pa] for axial positions ``z`` [m].

    ``z`` is measured from the bowl apex along the beam axis. Points inside
    the source region (z < 0) are allowed mathematically but physically
    meaningless; the caller decides the range.
    """
    _check_bowl(aperture_radius, roc)
    if f0 <= 0 or c0 <= 0 or rho0 <= 0:
        raise ValueError("f0, c0, rho0 must be > 0")

    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    k = 2.0 * np.pi * f0 / c0
    a, f = float(aperture_radius), float(roc)
    h = f - np.sqrt(f**2 - a**2)
    d_rim = np.sqrt(a**2 + (z_arr - h) ** 2)

    denom = 1.0 - z_arr / f
    near_focus = np.abs(denom) < focus_tol
    # Avoid 0/0 warnings: patch the denominator, overwrite those points below.
    denom_safe = np.where(near_focus, 1.0, denom)

    p = rho0 * c0 * u0 / denom_safe * (np.exp(1j * k * z_arr) - np.exp(1j * k * d_rim))
    p_focus = -1j * k * h * rho0 * c0 * u0 * np.exp(1j * k * f)
    p = np.where(near_focus, p_focus, p)
    return p if np.ndim(z) else p[0]


def focal_gain(aperture_radius: float, roc: float, f0: float, c0: float = 1500.0) -> float:
    """Linear focal pressure gain ``|p(focus)| / (rho c u0) = k h`` [-]."""
    _check_bowl(aperture_radius, roc)
    if f0 <= 0 or c0 <= 0:
        raise ValueError("f0 and c0 must be > 0")
    k = 2.0 * np.pi * f0 / c0
    h = roc - np.sqrt(roc**2 - aperture_radius**2)
    return float(k * h)
