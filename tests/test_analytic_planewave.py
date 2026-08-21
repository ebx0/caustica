"""M3 gate: plane-wave references — attenuation exact, Fubini limits."""

import numpy as np
import pytest

from caustica.analytic import attenuate, fubini_harmonic, shock_distance


def test_attenuation_is_exact_exponential():
    x = np.linspace(0.0, 0.1, 11)
    p = attenuate(2.0e6, alpha_np_m=30.0, x=x)
    np.testing.assert_allclose(p, 2.0e6 * np.exp(-30.0 * x), rtol=1e-15)
    with pytest.raises(ValueError):
        attenuate(1.0, alpha_np_m=-1.0, x=0.1)


def test_shock_distance_formula():
    # Hand-computed: rho c^3 / (beta omega p0)
    p0, f0, c0, rho0, beta = 1.0e6, 1.1e6, 1500.0, 1000.0, 3.5
    expected = rho0 * c0**3 / (beta * 2 * np.pi * f0 * p0)
    assert shock_distance(p0, f0, c0, rho0, beta) == pytest.approx(expected, rel=1e-12)
    assert shock_distance(p0, f0) == pytest.approx(
        1000.0 * 1500.0**3 / (3.5 * 2 * np.pi * f0 * p0), rel=1e-12
    )


def test_fubini_small_sigma_limits():
    # B1 -> 1 and B2 -> sigma/2 as sigma -> 0.
    assert fubini_harmonic(1, 1e-4) == pytest.approx(1.0, abs=1e-6)
    s = 0.01
    assert fubini_harmonic(2, s) == pytest.approx(s / 2.0, rel=1e-2)
    assert fubini_harmonic(1, 0.0) == 1.0
    assert fubini_harmonic(3, 0.0) == 0.0


def test_fubini_fundamental_monotone_decreasing_up_to_shock():
    s = np.linspace(1e-3, 1.0, 200)
    b1 = fubini_harmonic(1, s)
    assert np.all(np.diff(b1) < 0)
    assert b1[-1] == pytest.approx(2.0 * 0.44005, rel=1e-3)  # 2*J1(1)


def test_fubini_energy_never_exceeds_source():
    # Sum of harmonic energies stays <= source energy pre-shock.
    for s in (0.2, 0.5, 0.9, 1.0):
        total = sum(fubini_harmonic(n, s) ** 2 for n in range(1, 21))
        assert total <= 1.0 + 1e-9


def test_fubini_domain_checks():
    with pytest.raises(ValueError):
        fubini_harmonic(0, 0.5)
    with pytest.raises(ValueError):
        fubini_harmonic(1, 1.5)
