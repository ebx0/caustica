"""k-space PSTD building blocks, backend-generic (numpy or cupy).

Everything here is a pure function of (shape, dx, dt, medium constants) and
an array module ``xp`` — no solver state. The linear and Westervelt solvers
compose these; the GPU backend reuses them unchanged.

Ported, with the numerics intact, from the production notebook:
* 2/3/5-smooth FFT sizes (fast paths in cuFFT AND pocketfft),
* edge-replicated padding of the property maps into the FFT region,
* the kappa sinc dispersion correction referenced to c_max,
* separable Gaussian sponge combined into one multiplicative volume.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

from caustica.core.pml import sponge_profile_1d


def optimal_fft_size(n: int) -> int:
    """Smallest size >= n whose prime factors are all in {2, 3, 5}."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def pad_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """FFT-friendly padded shape for an active-grid shape."""
    return tuple(optimal_fft_size(n) for n in shape)


def pad_volume(vol: np.ndarray, padded: tuple[int, ...]) -> np.ndarray:
    """Edge-replicate ``vol`` up to ``padded`` (appends at the high end only).

    High-end-only padding keeps active-grid voxel indices valid in the padded
    frame — sources and sensors need no coordinate translation.
    """
    if len(padded) != vol.ndim:
        raise ValueError(f"padded rank {len(padded)} != volume rank {vol.ndim}")
    pad = []
    for have, want in zip(vol.shape, padded, strict=True):
        if want < have:
            raise ValueError(f"padded shape {padded} smaller than volume {vol.shape}")
        pad.append((0, want - have))
    if all(p == (0, 0) for p in pad):
        return vol
    return np.pad(vol, pad, mode="edge")


def k_vectors(shape: tuple[int, ...], dx: float, xp: ModuleType) -> list:
    """Broadcast-ready angular k-vectors (float32) for an rfft-last layout."""
    ks = []
    nd = len(shape)
    for ax, n in enumerate(shape):
        if ax == nd - 1:
            k = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx)
        else:
            k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
        arr = xp.asarray(k, dtype=xp.float32)
        sh = [1] * nd
        sh[ax] = arr.size
        ks.append(arr.reshape(sh))
    return ks


def kappa_sinc(ks: list, c_ref: float, dt: float, xp: ModuleType):
    """k-space dispersion correction ``sinc(c_ref |k| dt / 2)`` (float32).

    With this factor the discrete scheme propagates plane waves EXACTLY at
    speed ``c_ref`` for any stable dt (Tabei/Mast/Waag); heterogeneous media
    see a small O((c - c_ref)/c_ref) residual — the standard k-space trade.
    ``xp.sinc`` is the normalized sinc, hence the ``/ (2 pi)``.
    """
    k2 = ks[0] ** 2
    for k in ks[1:]:
        k2 = k2 + k**2
    return xp.sinc(c_ref * xp.sqrt(k2) * (dt / (2.0 * np.pi))).astype(xp.float32)


def spectral_derivative_factors(ks: list, kappa, shape: tuple[int, ...], xp: ModuleType) -> list:
    """Per-axis complex factors ``i k_ax kappa`` (complex64) for grad/div.

    The Nyquist wavenumber is DROPPED on every even-length axis. That is one
    fix with two faces, and both of them matter here:

    * Numerically, the Nyquist mode ``cos(pi x / dx)`` has a derivative that
      samples to all zeros on this grid, so a collocated spectral first
      derivative cannot represent it and the textbook remedy is to zero that
      wavenumber. (k-Wave never meets this because it is staggered: the
      half-sample shift ``exp(i k dx/2)`` rotates ``i k_Nyq`` onto the real
      axis. This engine is collocated, so it must zero explicitly.)
    * Structurally, ``i k`` is purely imaginary and odd in the index, while
      the bins that pair with THEMSELVES under ``j -> -j`` -- index ``n/2`` of
      an even axis -- have to stay conjugate-symmetric or ``irfftn`` has no
      real inverse to return. Leaving them in handed the transform an input
      violating that symmetry by ~150% of its own magnitude.

    Which axis you are on decides what the violation COSTS, and the two axis
    classes cost different things (measured 2026-08-24, see
    ``benchmarks/reports/campaign/2026-08-24-night``):

    * On the LAST axis, ``irfftn`` ends in a complex-to-real transform, and a
      C2R is free to resolve a non-Hermitian input however it likes. pocketfft
      discards the offending part, so the CPU answer was unaffected; cuFFT does
      not, so at 256^3 the two parted company at step 2 (1.1e-2 on
      ``irfftn(deriv2*pk)``) and by period 2 the GPU run was at NaN while the
      identical CPU run sat at 45 kPa -- the divergence that outlived three GPU
      sessions (2026-08-23, dev_validate U1b).
    * On the TRANSVERSE axes, ``irfftn`` is a plain complex-to-complex inverse
      and nothing discards anything: the illegal content passed straight into
      the real output on EVERY backend. That is where the CPU field actually
      moved -- 2.7% of peak at 64^3, and 1.4% on the library's own O'Neil bowl,
      concentrated on the voxelized source shell. A grid whose transverse axes
      are odd is bit-identical before and after; one whose beam axis is odd is
      not.

    So this is not a GPU-only repair. It is a correctness repair whose most
    visible symptom happened to be a GPU divergence.

    ``kappa`` keeps the true Nyquist ``|k|``: the dispersion correction is an
    even function of a magnitude, not an odd derivative, and zeroing it there
    would change the physics rather than repair a contract.
    """
    out = []
    for ax, k in enumerate(ks):
        if shape[ax] % 2 == 0:
            k = k.copy()
            idx = [0] * k.ndim
            idx[ax] = shape[ax] // 2  # rfft or fft layout: the Nyquist bin either way
            k[tuple(idx)] = 0.0
        out.append((1j * k * kappa).astype(xp.complex64))
    return out


def sponge_volume(shape: tuple[int, ...], width: int, edge: float, xp: ModuleType):
    """Full multiplicative sponge volume (float32) as a separable product.

    One dense volume trades memory for step-loop simplicity; the GPU path
    may switch back to separable per-axis multiplies later if VRAM demands.
    """
    out = xp.ones(shape, dtype=xp.float32)
    for ax, n in enumerate(shape):
        prof = sponge_profile_1d(n, width, edge, xp=xp)
        sh = [1] * len(shape)
        sh[ax] = n
        out = out * prof.reshape(sh)
    return out
