"""Phasor-contract gate: single-bin DFT recovers CW amplitude and phase."""

import numpy as np
import pytest

from hifusim.spectral import single_bin_phasor

F0 = 1.1e6
OMEGA = 2.0 * np.pi * F0


def _series(amp, phi0, spp, n_periods, t0=0.0):
    dt = 1.0 / F0 / spp
    t = t0 + dt * np.arange(spp * n_periods)
    return amp * np.sin(OMEGA * t + phi0), dt


def test_amplitude_and_phase_recovered_exactly():
    # p = A sin(wt + phi0) = Re{ A e^{i(phi0 - pi/2)} e^{iwt} }
    amp, phi0 = 3.7e6, 0.9
    x, dt = _series(amp, phi0, spp=10, n_periods=2)
    ph = single_bin_phasor(x, dt, F0)
    assert abs(ph) == pytest.approx(amp, rel=1e-9)
    expected_angle = phi0 - np.pi / 2.0
    diff = np.angle(np.exp(1j * (np.angle(ph) - expected_angle)))
    assert abs(diff) < 1e-9


def test_absolute_time_origin_is_honored():
    amp, phi0 = 1.0, 0.3
    t0 = 17.25 / F0  # non-integer period offset
    x, dt = _series(amp, phi0, spp=8, n_periods=3, t0=t0)
    ph = single_bin_phasor(x, dt, F0, t0=t0)
    diff = np.angle(np.exp(1j * (np.angle(ph) - (phi0 - np.pi / 2.0))))
    assert abs(ph) == pytest.approx(amp, rel=1e-9)
    assert abs(diff) < 1e-9


def test_harmonics_are_rejected_by_the_bin():
    # Pure 2f0 content must integrate to ~zero at the f0 bin.
    x, dt = _series(1.0, 0.0, spp=12, n_periods=2)
    x2 = np.sin(2 * OMEGA * (dt * np.arange(x.size)))
    ph = single_bin_phasor(x2, dt, F0)
    assert abs(ph) < 1e-9


def test_multidim_axis_handling():
    amp = np.array([[1.0], [2.0]])
    x, dt = _series(1.0, 0.0, spp=10, n_periods=1)
    stack = amp * x[None, :]
    ph = single_bin_phasor(stack, dt, F0, axis=1)
    np.testing.assert_allclose(np.abs(ph), [1.0, 2.0], rtol=1e-9)


def test_non_integer_window_refused():
    x, dt = _series(1.0, 0.0, spp=10, n_periods=2)
    with pytest.raises(ValueError, match="integer"):
        single_bin_phasor(x[:-3], dt, F0)
