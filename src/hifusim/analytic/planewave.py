"""Plane-wave references: exponential absorption and Fubini nonlinear growth.

These two are the cheapest, sharpest solver probes we have:

* The absorption law validates the dissipative part of a solver in
  isolation (M4 gate: measured alpha within 1% of configured alpha).
* The Fubini solution gives the exact pre-shock harmonic amplitudes of an
  initially sinusoidal plane wave in a lossless fluid — the reference for
  the Westervelt nonlinear term (M5 gate) and for KZK's Burgers step (M9).
"""

from __future__ import annotations

import numpy as np
from scipy.special import jv


def attenuate(p0: float, alpha_np_m: float, x: np.ndarray | float) -> np.ndarray | float:
    """Plane-wave amplitude after distance ``x`` [m]: ``p0 * exp(-alpha x)``."""
    if alpha_np_m < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha_np_m}")
    return p0 * np.exp(-alpha_np_m * np.asarray(x, dtype=np.float64))


def shock_distance(
    p0: float, f0: float, c0: float = 1500.0, rho0: float = 1000.0, beta: float = 3.5
) -> float:
    """Shock formation distance ``x_sh = rho0 c0^3 / (beta omega p0)`` [m].

    Equivalent to ``1 / (beta k M)`` with acoustic Mach number
    ``M = p0 / (rho0 c0^2)``.
    """
    if min(p0, f0, c0, rho0, beta) <= 0:
        raise ValueError("p0, f0, c0, rho0, beta must all be > 0")
    omega = 2.0 * np.pi * f0
    return rho0 * c0**3 / (beta * omega * p0)


def fubini_harmonic(n: int, sigma: np.ndarray | float) -> np.ndarray | float:
    """Fubini harmonic amplitude ``B_n(sigma) = 2 J_n(n sigma) / (n sigma)``.

    ``sigma = x / x_sh`` is the dimensionless propagation distance; the
    series is exact for ``0 <= sigma <= 1`` (up to shock formation).
    ``B_n`` is the amplitude of the n-th harmonic relative to the SOURCE
    amplitude ``p0``; small-sigma limits: ``B1 -> 1``, ``B2 -> sigma/2``.
    ``sigma = 0`` is handled by the analytic limit (B1=1, Bn>1=0).
    """
    if n < 1:
        raise ValueError(f"harmonic index must be >= 1, got {n}")
    s = np.asarray(sigma, dtype=np.float64)
    if np.any(s < 0) or np.any(s > 1):
        raise ValueError("Fubini solution is valid for 0 <= sigma <= 1 (pre-shock)")
    with np.errstate(invalid="ignore", divide="ignore"):
        b = 2.0 * jv(n, n * s) / (n * s)
    limit = 1.0 if n == 1 else 0.0
    b = np.where(s == 0, limit, b)
    return b if np.ndim(sigma) else float(b)
