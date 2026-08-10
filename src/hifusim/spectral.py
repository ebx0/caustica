"""Spectral utilities shared by solvers, sensors and adapters.

The single-bin DFT here IS the project's phasor contract: solvers accumulate
it streaming during the record window, external adapters (k-Wave) apply it
to recorded time series, and tests validate both against it. Keeping one
implementation prevents convention drift (sign, normalization, time origin).
"""

from __future__ import annotations

import numpy as np


def single_bin_phasor(
    series: np.ndarray,
    dt: float,
    f0: float,
    t0: float = 0.0,
    axis: int = -1,
) -> np.ndarray:
    """Complex amplitude of the ``f0`` component of ``series``.

    Convention: ``p(t) ~ Re{ P * exp(+i omega t) }`` — matching the solver's
    streaming accumulation ``sum p * exp(-i omega t) * 2 / N``. The window
    must span an INTEGER number of periods for a leakage-free result
    (exact-period dt policy); this is asserted to 1e-6 period tolerance.

    Parameters
    ----------
    series: real time series, shape (..., n_t) along ``axis``.
    dt: sample spacing [s].
    f0: extraction frequency [Hz].
    t0: absolute time of the first sample (phase reference) [s].
    """
    x = np.asarray(series)
    n = x.shape[axis]
    if n < 2:
        raise ValueError(f"need at least 2 samples, got {n}")
    if dt <= 0 or f0 <= 0:
        raise ValueError("dt and f0 must be > 0")
    n_periods = n * dt * f0
    if abs(n_periods - round(n_periods)) > 1e-6 * max(1.0, n_periods):
        raise ValueError(
            f"window covers {n_periods:.6f} periods — not an integer; "
            f"single-bin DFT would leak. Use an exact-period record window."
        )
    t = t0 + dt * np.arange(n)
    kernel = np.exp(-1j * 2.0 * np.pi * f0 * t)
    shape = [1] * x.ndim
    shape[axis] = n
    return (x * kernel.reshape(shape)).sum(axis=axis) * (2.0 / n)
