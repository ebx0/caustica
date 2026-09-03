"""Dynamic float16 quantization with a measured-error contract.

Port of the notebook's validated ``_try_float16``: the array is divided by
its own peak before the cast, so the full float16 mantissa is spent on the
range that actually exists (no fixed-exponent underflow). The round-trip
error is MEASURED, not assumed; if it exceeds ``max_norm_err * |peak|`` the
field is stored as float32 instead — correctness beats the 2x saving every
time. The measured error travels with the dataset so a reader never has to
guess what it lost.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

#: Project-wide default contract: max round-trip error, as a fraction of the
#: field's own peak (0.1 %). A release gate.
DEFAULT_MAX_NORM_ERR = 1e-3


class Quantized(NamedTuple):
    """One array ready for storage, plus everything needed to reload it."""

    stored: np.ndarray  # float16 (scaled) or float32 (verbatim)
    scale: float  # multiply after the float32 cast to reconstruct
    dtype_name: str  # "float16" | "float32"
    norm_err: float  # measured max |round-trip - original| / |peak|

    def restore(self) -> np.ndarray:
        """Reconstruct the float32 field (the documented reload recipe)."""
        return restore(self.stored, self.scale)


def restore(stored: np.ndarray, scale: float) -> np.ndarray:
    """The one reload recipe: ``stored.astype(float32) * float32(scale)``.

    Every reader (HDF5 store, preview package) goes through this so the
    reconstruction can never drift between formats.
    """
    return np.asarray(stored).astype(np.float32) * np.float32(scale)


def try_float16(arr: np.ndarray, max_norm_err: float = DEFAULT_MAX_NORM_ERR) -> Quantized:
    """Quantize to float16 with a dynamic scale, but only if it is safe.

    Falls back to float32 (scale 1.0, err 0.0) when the measured round-trip
    error exceeds the contract. An all-zero field is trivially exact.
    """
    arr = np.asarray(arr, dtype=np.float32)
    peak = float(np.abs(arr).max()) if arr.size else 0.0
    if peak <= 0.0:
        return Quantized(arr.astype(np.float16), 1.0, "float16", 0.0)
    scaled = (arr / peak).astype(np.float16)
    err = float(np.abs(scaled.astype(np.float32) * peak - arr).max() / peak)
    if err <= max_norm_err:
        return Quantized(scaled, peak, "float16", err)
    return Quantized(arr, 1.0, "float32", 0.0)
