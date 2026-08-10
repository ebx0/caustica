"""Analytic acoustic references.

These are the library's ground truth: solvers are validated against them
(milestone gates in MILESTONES.md), and they double as fast preview engines
(Rayleigh) for array design. CPU/numpy only by design — they must run
anywhere, instantly, with zero GPU dependencies.
"""

from hifusim.analytic.geometry import spherical_cap_points
from hifusim.analytic.oneill import axial_pressure, focal_gain
from hifusim.analytic.planewave import attenuate, fubini_harmonic, shock_distance
from hifusim.analytic.rayleigh import rayleigh_pressure

__all__ = [
    "attenuate",
    "axial_pressure",
    "focal_gain",
    "fubini_harmonic",
    "rayleigh_pressure",
    "shock_distance",
    "spherical_cap_points",
]
