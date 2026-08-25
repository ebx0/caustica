"""Rayleigh–Sommerfeld integral for CW pressure from a discretized source.

Physics
-------
For a source surface vibrating with normal velocity ``v_n`` at angular
frequency ``omega`` in a medium ``(rho, c)``, the time-harmonic pressure
(``e^{-i omega t}`` convention) at a field point ``x`` is

    p(x) = -i * rho * c * k / (2 pi) * sum_j v_j * dS_j * exp(i k R_j) / R_j

with ``R_j = |x - x_j|``. This is the free-field reference the solvers are
validated against, and the "instant beam preview" engine for array design:
it is exact for a rigid-baffled planar source and the standard reference for
gently curved HIFU bowls (O'Neil's solution is its on-axis closed form).

Implementation notes
--------------------
* Fully vectorized; field points are processed in chunks so the (n_src x
  n_field) distance matrix never exceeds ``max_pair`` elements (~256 MB of
  float64 at the default). Complexity is O(n_src * n_field) — fine for
  10^4 x 10^4; use coarser sampling for quick previews.
* Element phasing = complex ``v_n`` (amplitude * exp(i phase) per point).
"""

from __future__ import annotations

import numpy as np


def rayleigh_pressure(
    src_points: np.ndarray,
    src_areas: np.ndarray,
    v_n: np.ndarray | complex,
    field_points: np.ndarray,
    k: complex,
    rho: float = 1000.0,
    c: float = 1500.0,
    max_pair: int = 2**25,
) -> np.ndarray:
    """Complex CW pressure at ``field_points`` [Pa].

    Parameters
    ----------
    src_points: ``(n_src, 3)`` source point positions [m].
    src_areas: ``(n_src,)`` per-point areas [m^2].
    v_n: scalar or ``(n_src,)`` complex normal velocity [m/s].
    field_points: ``(n_field, 3)`` evaluation positions [m].
    k: angular wavenumber [rad/m]. Real for a lossless medium. A COMPLEX
        ``omega/c + 1j*alpha`` carries absorption: ``exp(1j k r)`` then
        contributes the ``exp(-alpha r)`` decay along every source-to-field
        path, which is the exact lossy Rayleigh integral rather than a
        decay applied to a lossless answer afterwards. ``alpha`` is in Np/m
        and must be >= 0 — a negative one would be an amplifying medium.
    rho, c: medium density and sound speed.
    max_pair: chunking threshold on n_src * chunk_size.
    """
    src = np.asarray(src_points, dtype=np.float64)
    fld = np.atleast_2d(np.asarray(field_points, dtype=np.float64))
    areas = np.asarray(src_areas, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src_points must be (n, 3), got {src.shape}")
    if fld.shape[1] != 3:
        raise ValueError(f"field_points must be (m, 3), got {fld.shape}")
    if areas.shape != (src.shape[0],):
        raise ValueError(f"src_areas must be ({src.shape[0]},), got {areas.shape}")
    k = complex(k)
    if k.real <= 0:
        raise ValueError(f"k must have a positive real part, got {k}")
    if k.imag < 0:
        raise ValueError(
            f"k has a negative imaginary part ({k.imag:g}), which is an amplifying "
            f"medium; absorption enters as +1j*alpha with alpha >= 0"
        )

    v = np.broadcast_to(np.asarray(v_n, dtype=np.complex128), (src.shape[0],))
    # Per-source complex strength folded once (v_j * dS_j).
    strength = v * areas

    n_src = src.shape[0]
    chunk = max(1, int(max_pair // max(n_src, 1)))
    out = np.empty(fld.shape[0], dtype=np.complex128)
    prefactor = -1j * rho * c * k / (2.0 * np.pi)

    for start in range(0, fld.shape[0], chunk):
        stop = min(start + chunk, fld.shape[0])
        diff = fld[start:stop, None, :] - src[None, :, :]
        r = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        if np.any(r < 1e-12):
            raise ValueError(
                "a field point coincides with a source point (R -> 0); "
                "evaluate off the source surface."
            )
        out[start:stop] = (np.exp(1j * k * r) / r) @ strength

    return prefactor * out
