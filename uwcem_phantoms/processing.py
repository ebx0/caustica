"""Label-domain surgery: crop, simplify, clean, resample, pad.

Everything here operates on an integer CLASS-CODE volume and is deliberately
free of physics — it decides WHERE tissue is, never what tissue sounds like.
That split is what lets the same operations serve both the detailed 10-class
tissue model and a merged 3-class one.

Two rules the whole module obeys:

1. **Physical position is tracked, never assumed.** Every operation that
   changes the array's extent returns the voxel offset it introduced, so the
   caller can keep the phantom's origin honest. A crop that forgets to move
   the origin puts the focus in the wrong place.
2. **No operation may invent a class.** Cleaning replaces voxels with a class
   that already borders them; it never averages two tissues into a third.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from caustica.geometry.volumes import LabelVolume, resample_scalar

#: FFT-friendly sizes are products of these primes. The k-space PSTD solver
#: FFTs every axis every step, and a size with a large prime factor (e.g.
#: 251, a real phantom dimension) makes pocketfft fall back to Bluestein —
#: several times slower than the neighbouring 252 = 2^2 * 3^2 * 7.
FFT_PRIMES = (2, 3, 5, 7)


# ------------------------------------------------------------------ cropping


@dataclass(frozen=True)
class CropResult:
    """A cropped volume plus the voxel offset of its new first voxel."""

    volume: np.ndarray
    offset: tuple[int, int, int]
    slices: tuple[slice, slice, slice]


def bounding_box(codes: np.ndarray, keep: tuple[int, ...]) -> tuple[slice, ...]:
    """Tight slices around every voxel whose class is in ``keep``."""
    mask = np.isin(codes, np.asarray(keep, dtype=codes.dtype))
    if not mask.any():
        raise ValueError(f"no voxel carries any of the classes {keep}; nothing to crop to")
    out = []
    for axis in range(codes.ndim):
        others = tuple(i for i in range(codes.ndim) if i != axis)
        present = np.flatnonzero(mask.any(axis=others))
        out.append(slice(int(present[0]), int(present[-1]) + 1))
    return tuple(out)


def crop(
    codes: np.ndarray,
    keep: tuple[int, ...],
    margin_vox: tuple[int, int, int] | int = 0,
) -> CropResult:
    """Crop to the bounding box of ``keep``, grown by ``margin_vox``.

    The margin is clamped to the array — asking for 20 mm of standoff on a
    phantom that only has 5 mm of immersion medium gives you the 5 mm that
    exists, not an index error. Use :func:`pad` to ADD medium.
    """
    box = bounding_box(codes, keep)
    margins = (margin_vox,) * codes.ndim if isinstance(margin_vox, int) else tuple(margin_vox)
    if len(margins) != codes.ndim:
        raise ValueError(f"margin_vox needs {codes.ndim} entries, got {margin_vox}")
    grown = []
    for axis, (sl, m) in enumerate(zip(box, margins, strict=True)):
        lo = max(0, sl.start - int(m))
        hi = min(codes.shape[axis], sl.stop + int(m))
        grown.append(slice(lo, hi))
    sl3 = tuple(grown)
    return CropResult(
        volume=np.ascontiguousarray(codes[sl3]),
        offset=tuple(s.start for s in sl3),  # type: ignore[arg-type]
        slices=sl3,  # type: ignore[arg-type]
    )


def pad(
    codes: np.ndarray,
    before: tuple[int, int, int],
    after: tuple[int, int, int],
    fill: int,
) -> np.ndarray:
    """Pad with a constant class (typically the immersion/coupling medium)."""
    widths = tuple((int(b), int(a)) for b, a in zip(before, after, strict=True))
    if any(b < 0 or a < 0 for b, a in widths):
        raise ValueError(f"pad widths must be >= 0, got before={before} after={after}")
    return np.pad(codes, widths, mode="constant", constant_values=fill)


def next_fft_friendly(n: int, primes: tuple[int, ...] = FFT_PRIMES) -> int:
    """Smallest ``m >= n`` whose prime factors all lie in ``primes``."""
    if n <= 1:
        return 1
    m = n
    while True:
        r = m
        for p in primes:
            while r % p == 0:
                r //= p
        if r == 1:
            return m
        m += 1


def prev_fft_friendly(n: int, primes: tuple[int, ...] = FFT_PRIMES) -> int:
    """Largest ``m <= n`` whose prime factors all lie in ``primes``.

    The mirror of :func:`next_fft_friendly`, for the one case where growing is
    not allowed: a hard size CEILING (the dataset's depth limit). Rounding up
    there would exceed a limit the caller asked for in millimetres, so we round
    down to the nearest transform-friendly size instead.
    """
    if n <= 1:
        return 1
    m = n
    while m > 1:
        r = m
        for p in primes:
            while r % p == 0:
                r //= p
        if r == 1:
            return m
        m -= 1
    return 1


def pad_to_fft_friendly(
    codes: np.ndarray,
    fill: int,
    primes: tuple[int, ...] = FFT_PRIMES,
    anchor: str = "center",
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Grow each axis to the next FFT-friendly size; returns (volume, before).

    ``anchor="center"`` splits the growth evenly (keeps the breast centred in
    the transverse plane); ``anchor="start"`` appends everything at the high
    index, which is what you want when the low-index end is the transducer
    face and must stay at a known depth.
    """
    if anchor not in ("center", "start"):
        raise ValueError(f"anchor must be 'center' or 'start', got {anchor!r}")
    before, after = [], []
    for n in codes.shape:
        extra = next_fft_friendly(n, primes) - n
        b = extra // 2 if anchor == "center" else 0
        before.append(b)
        after.append(extra - b)
    padded = pad(codes, tuple(before), tuple(after), fill)  # type: ignore[arg-type]
    return padded, tuple(before)  # type: ignore[return-value]


# --------------------------------------------------------------- simplifying


def merge_classes(codes: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Relabel via a complete ``{old: new}`` table (vectorized lookup).

    Every class present must appear in ``mapping``; a missing key raises
    rather than passing the old code through, because a silently unmapped
    tissue becomes an unknown material id at ``Medium.from_id_map`` time —
    far from here, with no clue why.
    """
    present = np.unique(codes)
    missing = [int(v) for v in present if int(v) not in mapping]
    if missing:
        raise ValueError(f"classes {missing} are present but absent from the merge table")
    lut_size = int(max(int(k) for k in mapping) + 1)
    lut = np.zeros(lut_size, dtype=np.int16)
    for old, new in mapping.items():
        lut[int(old)] = int(new)
    return lut[codes.astype(np.intp)]


def remove_islands(
    codes: np.ndarray,
    min_voxels: int,
    protect: tuple[int, ...] = (),
    connectivity: int = 1,
    max_passes: int = 8,
) -> tuple[np.ndarray, int]:
    """Dissolve connected components smaller than ``min_voxels``.

    A removed island takes the class of its nearest surviving NEIGHBOUR, so a
    lone glandular voxel inside fat becomes fat rather than becoming
    background. Classes listed in ``protect`` are never dissolved (skin is one
    voxel thick in places and would evaporate). Returns ``(volume, n_removed)``.

    Iterated to a fixed point, because one pass does not finish the job:
    dissolving an island of A into B can create a NEW sub-threshold component
    of B, and the class loop is over a snapshot taken before any writes. A
    single pass on phantom 012304 at ``min_voxels=30`` reported 63,502 voxels
    dissolved while leaving 7,669 voxels — 12% of the claim — still sitting in
    undersized components (review finding, 2026-08-18). Each pass now reads
    the CURRENT volume, and the count returned is what was actually changed.
    """
    from scipy import ndimage

    if min_voxels <= 1:
        return codes, 0
    structure = ndimage.generate_binary_structure(codes.ndim, connectivity)
    out = codes
    removed = 0
    for _ in range(max_passes):
        doomed = np.zeros(out.shape, dtype=bool)
        for cls in np.unique(out):
            if int(cls) in protect:
                continue
            labeled, n = ndimage.label(out == cls, structure=structure)
            if n == 0:
                continue
            sizes = np.bincount(labeled.reshape(-1))
            small = np.flatnonzero(sizes < min_voxels)
            small = small[small != 0]
            if small.size:
                doomed |= np.isin(labeled, small)
        if not doomed.any():
            break
        # Everything undersized goes at once, and the donors are the voxels
        # that SURVIVE this pass — so no island is ever replaced by another
        # island that is itself about to be dissolved.
        out = out.copy()
        out[doomed] = _neighbour_vote(out, doomed, exclude=-1)
        removed += int(doomed.sum())
    return out, removed


def _neighbour_vote(codes: np.ndarray, doomed: np.ndarray, exclude: int) -> np.ndarray:
    """Class of each doomed voxel taken from its 6-neighbourhood majority."""
    from scipy import ndimage

    # Nearest non-doomed voxel: exact, cheap, and never invents a class.
    # (A true majority vote needs one dilation per class; the distance
    # transform gives the same answer for the thin islands this targets.)
    keep = ~doomed
    if not keep.any():
        return np.full(int(doomed.sum()), exclude, dtype=codes.dtype)
    _, indices = ndimage.distance_transform_edt(doomed, return_indices=True)
    return codes[tuple(idx[doomed] for idx in indices)]


def fill_from(
    codes: np.ndarray, doomed: np.ndarray, donors: np.ndarray, fallback: int
) -> np.ndarray:
    """Class of the nearest DONOR voxel, for each doomed voxel.

    ``_neighbour_vote`` takes whatever is nearest; this restricts the answer
    to a chosen set, which is what "replace this shell with the tissue behind
    it" actually requires — the nearest voxel to the outer face of a skin
    shell is the water in front of it.
    """
    from scipy import ndimage

    if not doomed.any():
        return np.empty(0, dtype=codes.dtype)
    if not donors.any():
        return np.full(int(doomed.sum()), fallback, dtype=codes.dtype)
    _, indices = ndimage.distance_transform_edt(~donors, return_indices=True)
    return codes[tuple(idx[doomed] for idx in indices)]


def fill_interior_holes(codes: np.ndarray, background: int) -> np.ndarray:
    """Turn background pockets fully enclosed by tissue into their surround.

    MRI segmentation leaves air-like specks inside the breast; at HIFU
    amplitudes a stray "water" voxel inside fat is a spurious scatterer.
    """
    from scipy import ndimage

    bg = codes == background
    filled = ndimage.binary_fill_holes(~bg)
    interior = filled & bg
    if not interior.any():
        return codes
    out = codes.copy()
    out[interior] = _neighbour_vote(codes, interior, exclude=background)
    return out


def close_class(codes: np.ndarray, cls: int, iterations: int = 1) -> np.ndarray:
    """Morphologically close one class (seals 1-voxel gaps in, e.g., skin)."""
    from scipy import ndimage

    if iterations < 1:
        return codes
    mask = codes == cls
    closed = ndimage.binary_closing(mask, iterations=iterations)
    gained = closed & ~mask
    if not gained.any():
        return codes
    out = codes.copy()
    out[gained] = cls
    return out


def smooth_labels(codes: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Majority-filter the label field (removes staircase speckle).

    Implemented as a per-class 3x3x3 box count + argmax, which is exactly a
    majority vote and cannot produce a class that was not already in the
    neighbourhood.
    """
    from scipy import ndimage

    out = codes
    for _ in range(max(0, iterations)):
        classes = np.unique(out)
        best = None
        winner = None
        for cls in classes:
            score = ndimage.uniform_filter((out == cls).astype(np.float32), size=3, mode="nearest")
            if best is None:
                best = score
                winner = np.full(out.shape, cls, dtype=out.dtype)
            else:
                take = score > best
                winner[take] = cls  # type: ignore[index]
                np.maximum(best, score, out=best)
        out = winner  # type: ignore[assignment]
    return out


def keep_largest_component(codes: np.ndarray, background: int) -> np.ndarray:
    """Drop every disconnected blob of tissue except the biggest one."""
    from scipy import ndimage

    mask = codes != background
    structure = ndimage.generate_binary_structure(codes.ndim, 1)
    labeled, n = ndimage.label(mask, structure=structure)
    if n <= 1:
        return codes
    sizes = np.bincount(labeled.reshape(-1))
    sizes[0] = 0
    biggest = int(np.argmax(sizes))
    out = codes.copy()
    out[mask & (labeled != biggest)] = background
    return out


# --------------------------------------------------------------- resampling


def resample_codes(
    codes: np.ndarray,
    dx_old: float,
    dx_new: float,
    method: str = "smooth",
) -> np.ndarray:
    """Resample a class-code volume, delegating to :class:`LabelVolume`.

    Reuse is the point: ``LabelVolume.resample`` carries the exact-position
    sampling rule (and its regression history) that keeps interfaces from
    drifting, and the rest of the library resamples imported volumes through
    it. A second implementation here would be a second set of bugs.
    """
    vol = LabelVolume(labels=codes.astype(np.int32), dx=dx_old)
    return np.ascontiguousarray(vol.resample(dx_new, method=method).labels)


def resample_field(values: np.ndarray, dx_old: float, dx_new: float) -> np.ndarray:
    """Resample a continuous field with the label-matched interpolator."""
    return np.ascontiguousarray(resample_scalar(values, dx_old, dx_new))


# ------------------------------------------------------------------ metrics


def class_histogram(codes: np.ndarray, n_classes: int, chunk: int = 1 << 22) -> np.ndarray:
    """Voxel count per class code, length ``n_classes``.

    Counted in chunks: ``np.bincount`` needs an ``intp`` array, so the obvious
    one-liner makes an int64 copy of the WHOLE volume — 174 MB of scratch to
    produce a ten-element histogram on a 22 Mvox build, and it showed up as
    the peak of every 3-D texture request (review finding, 2026-08-18).
    """
    flat = codes.reshape(-1)
    out = np.zeros(n_classes, dtype=np.int64)
    for start in range(0, flat.size, chunk):
        part = flat[start : start + chunk].astype(np.intp, copy=False)
        out += np.bincount(part, minlength=n_classes)[:n_classes]
    return out


def interface_area_voxels(codes: np.ndarray) -> int:
    """Count of face-adjacent voxel pairs with different classes.

    A blunt but useful staircase metric: simplification and smoothing should
    lower it, and a resample that shatters interfaces raises it sharply.
    """
    total = 0
    for axis in range(codes.ndim):
        a = np.take(codes, np.arange(codes.shape[axis] - 1), axis=axis)
        b = np.take(codes, np.arange(1, codes.shape[axis]), axis=axis)
        total += int((a != b).sum())
    return total
