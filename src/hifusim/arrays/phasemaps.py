"""Phase-map (sin/cos image) representation of array element phases.

This is the AI-input encoding of the dataset pipeline: element phases are
splatted onto an S x S pixel grid at the elements' projected aperture
positions, as separate sin/cos channels (continuous across the 2 pi wrap),
masked to the circular aperture. Verbatim port of the notebook's
``_project_elements`` / ``select_phase_map_size`` / ``build_phase_maps``
with logging swapped for returned diagnostics.
"""

from __future__ import annotations

import numpy as np


def apply_circular_mask(image: np.ndarray) -> np.ndarray:
    size = image.shape[0]
    y, x = np.ogrid[-size / 2 : size / 2, -size / 2 : size / 2]
    out = image.copy()
    out[~((x * x + y * y) <= (size / 2) ** 2)] = 0.0
    return out


def project_elements(
    positions: np.ndarray, size: int, d_outer: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Map element xy positions to pixel indices on an S x S aperture image."""
    radius = d_outer / 2.0
    pitch = (2.0 * radius) / (size - 1)
    xf = (positions[:, 0] + radius) / pitch
    yf = (positions[:, 1] + radius) / pitch
    xi = np.clip(np.round(xf), 0, size - 1).astype(int)
    yi = np.clip(np.round(yf), 0, size - 1).astype(int)
    offsets_mm = np.column_stack((np.abs(xf - xi), np.abs(yf - yi))) * pitch * 1000.0
    return xi, yi, offsets_mm, pitch * 1000.0


def select_phase_map_size(
    positions: np.ndarray,
    d_outer: float,
    default_size: int = 32,
    fallback_size: int = 64,
    center_offset_frac: float = 0.25,
) -> tuple[int, dict]:
    """Smallest map size that places every element on its own pixel, accurately.

    Tries ``default_size`` then ``fallback_size``; a size is accepted when no
    two elements collide on a pixel AND every element center lands within
    ``center_offset_frac`` of the default-size pixel pitch. Raises when even
    the fallback fails (geometry too dense for this representation).
    """
    radius = d_outer / 2.0
    tol_mm = center_offset_frac * ((2.0 * radius) / (default_size - 1)) * 1e3

    last: dict = {}
    for size in (default_size, fallback_size):
        xi, yi, offsets_mm, pitch_mm = project_elements(positions, size, d_outer)
        lin = xi * size + yi
        uniq, counts = np.unique(lin, return_counts=True)
        collided = np.where(np.isin(lin, uniq[counts > 1]))[0]
        offenders = np.where(offsets_mm.max(axis=1) > tol_mm)[0]
        last = dict(
            size=size,
            pitch_mm=pitch_mm,
            max_offset_mm=float(offsets_mm.max()),
            mean_offset_mm=float(offsets_mm.mean()),
            n_collisions=int(collided.size),
            n_offset_offenders=int(offenders.size),
        )
        if collided.size == 0 and offenders.size == 0:
            return size, last
    raise ValueError(
        f"phase-map placement failed even at {fallback_size}x{fallback_size}: "
        f"{last['n_collisions']} collisions, {last['n_offset_offenders']} offset "
        f"offenders (tol {tol_mm:.2f} mm)."
    )


def build_phase_maps(
    phases: np.ndarray, positions: np.ndarray, size: int, d_outer: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sin_cos_maps (2, S, S), phase_map (S, S))`` float32 images."""
    xi, yi, _, _ = project_elements(positions, size, d_outer)
    sin_map = np.zeros((size, size), np.float32)
    cos_map = np.zeros((size, size), np.float32)
    phase_map = np.zeros((size, size), np.float32)
    sin_map[xi, yi] = np.sin(phases).astype(np.float32)
    cos_map[xi, yi] = np.cos(phases).astype(np.float32)
    phase_map[xi, yi] = np.asarray(phases, np.float32)
    return (
        np.stack([apply_circular_mask(sin_map), apply_circular_mask(cos_map)], axis=0),
        apply_circular_mask(phase_map),
    )
