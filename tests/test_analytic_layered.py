"""The exact plane-wave solution for a stratified fluid.

This is a reference other things are graded against, so it is graded against
the cases where the answer is known by inspection rather than by the formula
being tested: a stack of nothing is one interface, a half-wave layer is not
there, a quarter-wave layer of the geometric-mean impedance cancels an
interface, and an absorber only ever absorbs.

The last of those is here because it caught a real sign error on the day the
module was written. The transfer matrix expresses the front face in terms of
the back one, so its transmission carries ``exp(-i k d)`` — the opposite
orientation from the Rayleigh integral — and a wavenumber written
``omega/c + i alpha`` for both makes an absorbing layer AMPLIFY.
"""

from __future__ import annotations

import numpy as np
import pytest

from caustica.analytic.layered import (
    Layer,
    half_wave_thickness,
    interface_coefficients,
    quarter_wave_impedance,
    stack_coefficients,
    stack_matrix,
)

F0 = 1.0e6
WATER = (1500.0, 1000.0)
Z_WATER = WATER[0] * WATER[1]
MUSCLE = (1580.0, 1050.0)
Z_MUSCLE = MUSCLE[0] * MUSCLE[1]
FAT = (1450.0, 932.0)


def test_one_interface_conserves_power():
    """``T = 1 + R`` is continuity; power still balances behind it.

    The transmitted PRESSURE exceeds the incident one at a hard interface,
    which is correct and looks wrong: the particle velocity falls by more
    than the pressure rises, and ``|R|^2 + (z1/z2)|T|^2 = 1`` says so.
    """
    for z2 in (Z_MUSCLE, 3.0e6, 0.5e6):
        r, t = interface_coefficients(Z_WATER, z2)
        assert t == pytest.approx(1.0 + r)
        assert r**2 + (Z_WATER / z2) * t**2 == pytest.approx(1.0)


def test_a_stack_of_nothing_is_one_interface():
    r, t = stack_coefficients([], Z_WATER, Z_MUSCLE, F0)
    r0, t0 = interface_coefficients(Z_WATER, Z_MUSCLE)

    assert r.real == pytest.approx(r0) and r.imag == pytest.approx(0.0)
    assert t.real == pytest.approx(t0) and t.imag == pytest.approx(0.0)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_a_half_wave_layer_is_not_there(n):
    """At ``kd = n pi`` the layer's matrix is +-I whatever its impedance.

    The sharpest prediction the module makes, and the one a phase-accumulation
    error cannot survive: the layer's own wavelength decides the thickness, so
    getting it right means the wavenumber inside the layer is right.
    """
    c, rho = FAT
    d = half_wave_thickness(c, F0, n)
    r, t = stack_coefficients([Layer(d, c, rho)], Z_WATER, Z_MUSCLE, F0)
    r0, _ = interface_coefficients(Z_WATER, Z_MUSCLE)

    assert abs(r) == pytest.approx(abs(r0), rel=1e-9)
    # The matrix itself is +-I. Its off-diagonals carry impedance, so they are
    # scaled by Z before being called zero — comparing |M| to the identity
    # elementwise would be comparing a pressure to a velocity.
    m = stack_matrix([Layer(d, c, rho)], F0)
    z = rho * c
    assert abs(abs(m[0, 0]) - 1.0) < 1e-9 and abs(abs(m[1, 1]) - 1.0) < 1e-9
    assert abs(m[0, 1]) / z < 1e-12
    assert abs(m[1, 0]) * z < 1e-12


def test_a_quarter_wave_layer_of_the_mean_impedance_cancels_an_interface():
    z_layer = quarter_wave_impedance(Z_WATER, Z_MUSCLE)
    c = 1500.0
    r, _t = stack_coefficients([Layer(0.25 * c / F0, c, z_layer / c)], Z_WATER, Z_MUSCLE, F0)

    assert abs(r) < 1e-12
    assert z_layer == pytest.approx(np.sqrt(Z_WATER * Z_MUSCLE))


@pytest.mark.parametrize("alpha", [5.0, 15.0, 30.0])
@pytest.mark.parametrize("d_mm", [2.0, 10.0])
def test_a_matched_absorber_only_attenuates(alpha, d_mm):
    """An impedance-matched absorber reflects nothing and passes ``exp(-alpha d)``.

    Written the other way round, this is the test that fails when the
    wavenumber's imaginary part has the wrong sign: the layer amplifies by
    the same factor it should attenuate by. That is exactly what the module
    did before a three-layer tissue stack in the 1-D solver disagreed with it
    by a factor that turned out to be ``exp(2 alpha L)``.
    """
    c, rho = WATER
    layer = Layer(d_mm * 1e-3, c, rho, alpha)
    r, t = stack_coefficients([layer], Z_WATER, Z_WATER, F0)

    assert abs(r) < 1e-12, "a matched layer cannot reflect"
    assert abs(t) == pytest.approx(np.exp(-alpha * d_mm * 1e-3), rel=1e-9)
    assert abs(t) < 1.0, "an absorber that amplifies has its sign backwards"


def test_absorption_and_impedance_compose():
    """Two effects in one layer, and neither swallows the other."""
    c, rho = FAT
    d = 8.0e-3
    lossless = abs(stack_coefficients([Layer(d, c, rho)], Z_WATER, Z_WATER, F0)[1])
    lossy = abs(stack_coefficients([Layer(d, c, rho, 12.0)], Z_WATER, Z_WATER, F0)[1])

    assert lossy < lossless
    # The loss is the single pass through the layer's own thickness, to within
    # what its internal multiples reshuffle. Those are worth a fraction of a
    # percent here and their SIGN is not fixed — absorption weakens every
    # multiple, but whether a weaker multiple helps or hurts is set by ``kd``.
    # So the claim is a magnitude, not a direction: asserting the direction is
    # how this test read before the numbers were looked at.
    single_pass = np.exp(-12.0 * d)
    assert lossy / lossless == pytest.approx(single_pass, rel=0.02)
    assert lossy / lossless < 1.0


def test_a_stack_is_the_product_of_its_layers():
    """Order matters, and the matrices multiply in beam order."""
    a = Layer(3e-3, *FAT)
    b = Layer(2e-3, *MUSCLE)

    np.testing.assert_allclose(stack_matrix([a, b], F0), a.matrix(F0) @ b.matrix(F0), rtol=1e-12)
    forward = stack_coefficients([a, b], Z_WATER, Z_WATER, F0)
    reversed_ = stack_coefficients([b, a], Z_WATER, Z_WATER, F0)
    # Transmission through a reciprocal stack is order-independent; reflection
    # is not, because it is read from the side the wave comes in on.
    assert abs(forward[1]) == pytest.approx(abs(reversed_[1]), rel=1e-12)


def test_the_inputs_that_cannot_mean_anything_are_refused():
    with pytest.raises(ValueError, match="thickness"):
        Layer(-1e-3, *WATER)
    with pytest.raises(ValueError, match="c and rho"):
        Layer(1e-3, 0.0, 1000.0)
    with pytest.raises(ValueError, match="amplifies"):
        Layer(1e-3, *WATER, alpha_np_m=-1.0)
    with pytest.raises(ValueError, match="impedances"):
        interface_coefficients(0.0, Z_WATER)
    with pytest.raises(ValueError, match="impedances"):
        stack_coefficients([], Z_WATER, -1.0, F0)
