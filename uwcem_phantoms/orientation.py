"""Anatomical axes: from the repository's index order to a HIFU frame.

The UWCEM files index voxels as ``ARRAY(s1, s2, s3)``. The manual never
states which anatomical direction each axis runs, so this module DERIVES it
from the data and pins the result with an assertion any caller can re-run.

What the data says (verified on every phantom by
:func:`detect_propagation_axis`):

* **s1 is the nipple -> chest-wall axis.** The 0.5 cm muscle chest wall the
  manual promises appears as a full-area slab occupying the LAST ten s1
  slices (10 x 0.5 mm = 5 mm) and nowhere else; along s2/s3 muscle is smeared
  over every index at a few percent area. Index ``s1 = 0`` is 100% immersion
  medium — free space in front of the breast.
* s2 and s3 are the two transverse axes (the coronal plane, patient prone).
  Their individual anatomical labels (superior-inferior vs left-right) are
  not recoverable from the files and do not affect an axisymmetric-ish HIFU
  setup, so they are reported as ``transverse-1`` / ``transverse-2``.

Canonical simulation frame produced by :func:`to_canonical`:

    ``(x, y, z)`` with **z = beam/propagation axis**, ``z = 0`` anterior (in
    the immersion medium, where the transducer sits) and ``z = z_max`` at the
    chest wall — the direction a breast-HIFU beam actually travels.

The transform is ``flip(transpose(2, 1, 0), axis=0)``. Both halves matter:

* ``transpose(2, 1, 0)`` moves the propagation axis to the end;
* the flip restores HANDEDNESS. A single axis swap is an ODD permutation
  (determinant -1), i.e. a mirror image — a left breast silently becomes a
  right breast. One extra flip brings the determinant back to +1. This is
  the same transform :func:`caustica.geometry.load_breast_phantom` already
  applies, so phantoms exported here land in the frame the rest of the
  library already uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Axis roles of the canonical frame, in order.
CANONICAL_AXES = ("transverse-2", "transverse-1", "propagation (anterior -> chest wall)")

#: Axis permutation applied to the native ``(s1, s2, s3)`` volume.
CANONICAL_TRANSPOSE = (2, 1, 0)

#: Axes flipped AFTER the transpose (handedness repair).
CANONICAL_FLIP = (0,)


@dataclass(frozen=True)
class AxisReport:
    """Evidence for which native axis carries the chest wall."""

    propagation_axis: int
    muscle_slab: tuple[int, int]
    muscle_slab_mm: float
    slab_area_fraction: float
    chest_wall_at_high_index: bool

    def summary(self) -> str:
        lo, hi = self.muscle_slab
        side = "high" if self.chest_wall_at_high_index else "low"
        return (
            f"propagation axis = s{self.propagation_axis + 1}; muscle slab at "
            f"indices [{lo}, {hi}] ({self.muscle_slab_mm:.1f} mm thick, "
            f"{self.slab_area_fraction:.0%} of the slice), chest wall at the "
            f"{side}-index end"
        )


def detect_propagation_axis(codes: np.ndarray, muscle_code: int = 2, dx: float = 0.5e-3):
    """Find the chest-wall axis of a NATIVE ``(s1, s2, s3)`` code volume.

    The chest wall is the only structure in the phantom that is a full-area
    slab, so the axis along which muscle covers ~100% of a slice — and only
    a contiguous run of slices — is the propagation axis. Returns an
    :class:`AxisReport`; raises when no axis looks like a slab (a phantom
    without a chest wall, or a wrong ``muscle_code``).
    """
    if codes.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got {codes.ndim}-D")
    best: AxisReport | None = None
    for axis in range(3):
        others = tuple(i for i in range(3) if i != axis)
        area = codes.shape[others[0]] * codes.shape[others[1]]
        per_slice = (codes == muscle_code).sum(axis=others)
        if per_slice.max() == 0:
            continue
        frac = float(per_slice.max()) / area
        present = np.flatnonzero(per_slice > 0)
        lo, hi = int(present[0]), int(present[-1])
        report = AxisReport(
            propagation_axis=axis,
            muscle_slab=(lo, hi),
            muscle_slab_mm=(hi - lo + 1) * dx * 1e3,
            slab_area_fraction=frac,
            chest_wall_at_high_index=lo > codes.shape[axis] - 1 - hi,
        )
        if best is None or report.slab_area_fraction > best.slab_area_fraction:
            best = report
    if best is None or best.slab_area_fraction < 0.5:
        raise ValueError(
            "no axis carries a full-area muscle slab; this does not look like a "
            "UWCEM phantom (or muscle_code is wrong)"
        )
    return best


def to_canonical(volume: np.ndarray, copy: bool = True) -> np.ndarray:
    """Native ``(s1, s2, s3)`` -> canonical ``(x, y, z)``, z = propagation.

    ``copy=True`` returns a C-contiguous array — do this once, at the end of
    decoding, because every downstream operation (resampling, morphology,
    slicing for the viewer) is far faster on contiguous memory.
    """
    out = np.flip(volume.transpose(*CANONICAL_TRANSPOSE), axis=CANONICAL_FLIP[0])
    return np.ascontiguousarray(out) if copy else out


def from_canonical(volume: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_canonical` (for writing data back in file order)."""
    return np.flip(volume, axis=CANONICAL_FLIP[0]).transpose(*CANONICAL_TRANSPOSE)


def is_right_handed() -> bool:
    """The canonical transform preserves chirality (permutation x flip = +1)."""
    perm = np.zeros((3, 3))
    for i, j in enumerate(CANONICAL_TRANSPOSE):
        perm[i, j] = 1.0
    det = np.linalg.det(perm) * (-1.0) ** len(CANONICAL_FLIP)
    return bool(det > 0)
