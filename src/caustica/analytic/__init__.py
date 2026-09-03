"""Analytic acoustic references.

These are the library's ground truth: solvers are validated against them,
and they double as fast preview engines
(Rayleigh) for array design. CPU/numpy only by design — they must run
anywhere, instantly, with zero GPU dependencies.
"""

from caustica.analytic.geometry import spherical_cap_points
from caustica.analytic.oneill import axial_pressure, focal_gain
from caustica.analytic.planewave import attenuate, fubini_harmonic, shock_distance
from caustica.analytic.rayleigh import rayleigh_pressure

__all__ = [
    "attenuate",
    "axial_pressure",
    "focal_gain",
    "fubini_harmonic",
    "rayleigh_pressure",
    "shock_distance",
    "spherical_cap_points",
]
