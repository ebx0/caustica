"""Absorbing boundary: Gaussian sponge PML specification and profile.

The production notebook (hifu_pred_dx300_t128 v12) established this exact
sponge as numerically robust for the k-space PSTD scheme: a multiplicative
1-D Gaussian ramp applied to pressure and velocity each step, with profile

    s(r) = exp(-edge * ((width - r) / width)^2),   r = 0 .. width-1

so the OUTERMOST cell damps hardest (s = exp(-edge)) and the profile blends
smoothly to 1.0 at the inner edge. It is applied separably per axis.

This module owns only the *specification* (physical thickness, edge factor)
and the profile construction; solvers decide where/when to apply it.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

import numpy as np


@dataclass(frozen=True)
class PMLSpec:
    """Sponge-PML specification in physical units.

    Parameters
    ----------
    thickness:
        Sponge thickness in meters (converted to voxels against a grid's dx).
        The notebook default was 5.0 mm at dx=0.30 mm -> 17 voxels.
    edge:
        Gaussian edge factor; larger = stronger damping at the outer cell.
    """

    thickness: float
    edge: float = 2.0

    def __post_init__(self) -> None:
        if self.thickness < 0:
            raise ValueError(f"PML thickness must be >= 0, got {self.thickness}")
        if self.edge <= 0:
            raise ValueError(f"PML edge factor must be > 0, got {self.edge}")

    def thickness_vox(self, dx: float) -> int:
        """Sponge thickness in whole voxels for a grid spacing ``dx`` [m]."""
        if dx <= 0:
            raise ValueError(f"dx must be > 0, got {dx}")
        return int(round(self.thickness / dx))


def sponge_profile_1d(n: int, width: int, edge: float = 2.0, xp: ModuleType = np) -> np.ndarray:
    """1-D multiplicative sponge profile of length ``n`` (float32).

    Ones in the interior; symmetric Gaussian ramps of ``width`` cells at both
    ends. ``width=0`` returns all ones. Direct port of the validated notebook
    implementation (``_make_sponge_1d``), generalized over the array module
    so the same function builds numpy or cupy profiles.
    """
    if n <= 0:
        raise ValueError(f"profile length must be > 0, got {n}")
    if width < 0:
        raise ValueError(f"sponge width must be >= 0, got {width}")
    if 2 * width > n:
        raise ValueError(f"sponge width {width} does not fit twice into length {n}")
    s = xp.ones(n, dtype=xp.float32)
    if width > 0:
        r = xp.arange(width, dtype=xp.float32)
        ramp = xp.exp(-edge * ((width - r) / width) ** 2)
        s[:width] = ramp
        s[-width:] = ramp[::-1]
    return s
