"""Exact plane-wave solutions for layered fluid media.

The heterogeneous counterpart to :mod:`~caustica.analytic.oneill` and
:mod:`~caustica.analytic.rayleigh`. Those give a solver something exact to be
graded against in a *uniform* medium; nothing here did the same once the
medium had structure in it, which for a library aimed at breast phantoms is
the wrong gap to have.

A stratified fluid has a closed form. Across one layer of thickness ``d``,
wavenumber ``k`` and impedance ``Z``, pressure and normal velocity transform
by the transfer matrix

    M = [[cos(kd),        i Z sin(kd)],
         [i sin(kd) / Z,  cos(kd)   ]]

and a stack is the product of its layers' matrices. Everything else — the
reflection and transmission coefficients, the half-wave transparency of a
slab, the quarter-wave matching layer — falls out of that one object, so
there is a single thing to get right rather than a family of special cases.

Absorption enters as a complex wavenumber ``k + i*alpha``, matching the
library's convention ``p(t) = Re{P e^{-i omega t}}`` with an outgoing wave
``e^{+ikx}``: the product ``e^{i(k + i alpha)x}`` then decays as
``e^{-alpha x}``, and ``alpha`` is the SPATIAL decay of pressure in Np/m,
the same quantity :class:`~caustica.materials.Material` stores.

Normal incidence only. Oblique incidence needs the same matrices with
``k_z = k cos(theta)`` and ``Z / cos(theta)``, which is a small extension and
not one anything in this library needs yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Layer:
    """One stratum: how thick, and what it is made of."""

    thickness: float  # [m]
    c: float  # [m/s]
    rho: float  # [kg/m^3]
    alpha_np_m: float = 0.0  # spatial pressure decay [Np/m]

    def __post_init__(self) -> None:
        if self.thickness < 0:
            raise ValueError(f"thickness must be >= 0, got {self.thickness}")
        if self.c <= 0 or self.rho <= 0:
            raise ValueError("c and rho must be > 0")
        if self.alpha_np_m < 0:
            raise ValueError("alpha must be >= 0; a negative one amplifies")

    @property
    def impedance(self) -> float:
        return self.rho * self.c

    def wavenumber(self, f0: float) -> complex:
        """``omega/c - i alpha``, and the sign is set by the matrix below.

        The transfer matrix expresses the FRONT face in terms of the BACK
        one, so the transmission coefficient comes out as ``exp(-i k d)``
        rather than ``exp(+i k d)`` — the opposite orientation from the
        Rayleigh integral, where the kernel carries ``exp(+i k r)`` and
        absorption enters as ``+i alpha``. Getting it backwards here makes an
        absorbing layer AMPLIFY: a matched absorber returns ``|T| = e^{+alpha d}``
        instead of ``e^{-alpha d}``, which is what
        ``test_a_matched_absorber_only_attenuates`` exists to catch. It was
        backwards for the first hour this module existed, and a 1-D solver
        run through three tissue layers is what caught it.
        """
        return complex(2.0 * np.pi * f0 / self.c, -self.alpha_np_m)

    def matrix(self, f0: float) -> np.ndarray:
        kd = self.wavenumber(f0) * self.thickness
        z = self.impedance
        return np.array(
            [
                [np.cos(kd), 1j * z * np.sin(kd)],
                [1j * np.sin(kd) / z, np.cos(kd)],
            ],
            dtype=np.complex128,
        )


def interface_coefficients(z1: float, z2: float) -> tuple[float, float]:
    """Pressure reflection and transmission at one interface, normal incidence.

    ``R = (Z2 - Z1) / (Z2 + Z1)`` and ``T = 1 + R``. The second is worth
    writing that way rather than as ``2 Z2 / (Z1 + Z2)``: it says the
    transmitted pressure is the incident plus the reflected, which is the
    continuity condition the whole thing rests on, and makes ``T > 1`` at a
    hard interface obviously right rather than obviously suspicious. Power is
    still conserved — the transmitted particle velocity falls by more than
    the pressure rises.
    """
    if z1 <= 0 or z2 <= 0:
        raise ValueError(f"impedances must be > 0, got {z1}, {z2}")
    r = (z2 - z1) / (z2 + z1)
    return r, 1.0 + r


def stack_matrix(layers: Sequence[Layer], f0: float) -> np.ndarray:
    """Product of the layers' transfer matrices, first layer first."""
    m = np.eye(2, dtype=np.complex128)
    for layer in layers:
        m = m @ layer.matrix(f0)
    return m


def stack_coefficients(
    layers: Sequence[Layer], z_in: float, z_out: float, f0: float
) -> tuple[complex, complex]:
    """Complex pressure reflection and transmission through a stack.

    ``z_in`` and ``z_out`` are the semi-infinite media on either side. With no
    layers this reduces to :func:`interface_coefficients`, which is the first
    thing worth checking about it.
    """
    if z_in <= 0 or z_out <= 0:
        raise ValueError(f"impedances must be > 0, got {z_in}, {z_out}")
    m = stack_matrix(layers, f0)
    # Input impedance seen at the front face, with the substrate closing the
    # stack: v = p / z_out there.
    num = m[0, 0] * z_out + m[0, 1]
    den = m[1, 0] * z_out + m[1, 1]
    z_input = num / den
    r = (z_input - z_in) / (z_input + z_in)
    # p_front = p_inc (1 + r), and p_front = p_out * (M11 + M12 / z_out)
    t = (1.0 + r) / (m[0, 0] + m[0, 1] / z_out)
    return complex(r), complex(t)


def half_wave_thickness(c: float, f0: float, n: int = 1) -> float:
    """Thickness at which a slab is transparent: ``n`` half-wavelengths.

    At ``kd = n pi`` the layer's matrix is ``+-I``, so the stack behaves as
    though it were not there whatever its impedance — the sharpest prediction
    this module makes, and the one a solver with a phase-accumulation error
    cannot reproduce.
    """
    return n * 0.5 * c / f0


def quarter_wave_impedance(z1: float, z2: float) -> float:
    """Impedance of the layer that makes ``z1`` and ``z2`` reflectionless.

    ``sqrt(z1 z2)`` at a quarter-wave thickness. The complement of the
    half-wave case: there the layer disappears, here it cancels an interface
    that would otherwise reflect.
    """
    return float(np.sqrt(z1 * z2))
