"""Labeled voxel volumes: import, storage, and dx-resampling.

A :class:`LabelVolume` is a heterogeneous multi-class voxel map (integer
tissue labels) with a physical spacing and origin — the bridge between
imported phantom files (mtype-style text, npz) and the Scene/Medium chain.
Resampling is its own concern here, deliberately separate from geometry
construction: the user picks dx at simulation time, so every volume must
resample well, with a selectable method:

* ``"nearest"`` — label-safe nearest neighbor (fast; blocky interfaces).
* ``"smooth"``  — one-hot linear interpolation + argmax: area-weighted
  class vote, smoother interfaces at non-integer factors; never invents
  labels that were not present.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RESAMPLE_METHODS = ("nearest", "smooth")


def _axis_samples(n_in: int, n_out: int, ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact resample positions ``j * ratio`` along one axis.

    Returns clamped lower/upper neighbor indices and the linear weight.
    Positions are ABSOLUTE fractional input indices — never the
    endpoint-aligned rescale that ``scipy.ndimage.zoom(grid_mode=False)``
    applies, which stretches content by ``(n_out-1)/((n_in-1)*ratio)``
    (review finding, 2026-08-11: a 0.5->0.3 mm phantom resample drifted a
    full voxel at 18 mm depth).
    """
    pos = np.arange(n_out, dtype=np.float64) * ratio
    i0 = np.clip(np.floor(pos).astype(np.intp), 0, n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    w = np.clip(pos - i0, 0.0, 1.0).astype(np.float32)
    return i0, i1, w


def _resample_linear(vol: np.ndarray, out_shape: tuple[int, ...], ratio: float) -> np.ndarray:
    """Separable linear interpolation of ``vol`` at exact positions."""
    out = vol
    for ax, m in enumerate(out_shape):
        i0, i1, w = _axis_samples(vol.shape[ax], m, ratio)
        a0 = np.take(out, i0, axis=ax)
        a1 = np.take(out, i1, axis=ax)
        sh = [1] * out.ndim
        sh[ax] = m
        w = w.reshape(sh)
        out = a0 * (1.0 - w) + a1 * w
    return out


def resample_scalar(values: np.ndarray, dx_old: float, dx_new: float) -> np.ndarray:
    """Linearly resample a CONTINUOUS volume the same way labels are resampled.

    Shares :func:`_axis_samples` with :meth:`LabelVolume.resample`, so a
    property volume (rho, c, ...) and its label map stay voxel-aligned. Using
    a different interpolator for the two would drift interfaces apart — the
    exact failure the exact-position sampling above exists to prevent.
    """
    if dx_old <= 0 or dx_new <= 0:
        raise ValueError(f"spacings must be > 0, got {dx_old} -> {dx_new}")
    ratio = dx_new / dx_old
    if abs(ratio - 1.0) < 1e-12:
        return values
    out_shape = tuple(max(1, int(round(n * dx_old / dx_new))) for n in values.shape)
    return _resample_linear(values.astype(np.float32, copy=False), out_shape, ratio)


@dataclass(frozen=True, eq=False)
class LabelVolume:
    """Integer label volume with isotropic spacing [m] and physical origin."""

    labels: np.ndarray
    dx: float
    origin: tuple[float, ...] = field(default=None)  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        # dataclass-default __eq__ would compare the ndarray field and raise
        # "truth value of an array is ambiguous" (review finding, 2026-08-11)
        if not isinstance(other, LabelVolume):
            return NotImplemented
        return (
            self.dx == other.dx
            and self.origin == other.origin
            and np.array_equal(self.labels, other.labels)
        )

    def __post_init__(self) -> None:
        lab = np.asarray(self.labels)
        if not np.issubdtype(lab.dtype, np.integer):
            raise TypeError(f"labels must be integer-typed, got {lab.dtype}")
        if lab.ndim not in (2, 3):
            raise ValueError(f"labels must be 2-D or 3-D, got {lab.ndim}-D")
        if self.dx <= 0:
            raise ValueError(f"dx must be > 0, got {self.dx}")
        origin = self.origin if self.origin is not None else (0.0,) * lab.ndim
        if len(origin) != lab.ndim:
            raise ValueError(f"origin rank {len(origin)} != volume rank {lab.ndim}")
        object.__setattr__(self, "labels", np.ascontiguousarray(lab))
        object.__setattr__(self, "origin", tuple(float(o) for o in origin))

    @property
    def ndim(self) -> int:
        return self.labels.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.labels.shape

    @property
    def extent(self) -> tuple[float, ...]:
        return tuple(n * self.dx for n in self.shape)

    @property
    def label_set(self) -> tuple[int, ...]:
        return tuple(int(v) for v in np.unique(self.labels))

    def resample(self, dx_new: float, method: str = "nearest") -> LabelVolume:
        """Return the volume resampled to spacing ``dx_new`` [m].

        Output voxel ``j`` samples the input at the EXACT physical position
        ``origin + j*dx_new`` (origin = first voxel center, preserved), so
        interfaces stay where the source data put them. Output shape is
        ``round(n * dx / dx_new)`` per axis (same physical extent).
        """
        if dx_new <= 0:
            raise ValueError(f"dx_new must be > 0, got {dx_new}")
        if method not in RESAMPLE_METHODS:
            raise ValueError(f"unknown method {method!r}; pick from {RESAMPLE_METHODS}")
        ratio = dx_new / self.dx
        if abs(ratio - 1.0) < 1e-12:
            return self
        out_shape = tuple(max(1, int(round(n * self.dx / dx_new))) for n in self.shape)
        if method == "nearest":
            idx = [
                np.clip(np.round(np.arange(m) * ratio).astype(np.intp), 0, n - 1)
                for m, n in zip(out_shape, self.shape, strict=True)
            ]
            out = self.labels[np.ix_(*idx)]
        else:  # smooth: one-hot linear + argmax (never invents labels)
            best_score = None
            out = None
            for lab in self.label_set:
                score = _resample_linear((self.labels == lab).astype(np.float32), out_shape, ratio)
                if out is None:
                    out = np.full(score.shape, lab, dtype=self.labels.dtype)
                    best_score = score
                else:
                    take = score > best_score
                    out[take] = lab
                    np.maximum(best_score, score, out=best_score)
        return LabelVolume(labels=out.astype(self.labels.dtype), dx=dx_new, origin=self.origin)

    # ---- IO ----

    def save_npz(self, path: str | Path) -> None:
        np.savez_compressed(path, labels=self.labels, dx=self.dx, origin=np.asarray(self.origin))

    @staticmethod
    def load_npz(path: str | Path) -> LabelVolume:
        with np.load(path) as data:
            return LabelVolume(
                labels=data["labels"],
                dx=float(data["dx"]),
                origin=tuple(data["origin"].tolist()),
            )


def load_labels_txt(
    path: str | Path,
    shape: tuple[int, ...],
    dx: float,
    mapping: Callable[[np.ndarray], np.ndarray],
    order: str = "F",
    transpose: tuple[int, ...] | None = None,
    flip_axes: tuple[int, ...] = (),
    nan_value: float = 0.0,
    cache: bool = True,
) -> LabelVolume:
    """Import an mtype-style whitespace text volume as labels.

    The raw file holds one float per voxel (any layout); ``mapping`` turns
    the float volume into integer labels — the ONLY part that knows what
    the numbers mean. Because parsing 100+ MB of text is slow, the result
    is cached as ``<path>.labels.npz`` next to the source (``cache=True``)
    and reloaded from there when the cache is newer than the source AND was
    built with the same arguments (shape/dx/order/transpose/flip/nan/
    mapping) — an argument fingerprint is stored in the npz so a cache
    written by different arguments is re-parsed, not silently served
    (review finding, 2026-08-11).
    """
    path = Path(path)
    cache_path = path.with_suffix(path.suffix + ".labels.npz")
    fingerprint = repr(
        (
            tuple(int(n) for n in shape),
            float(dx),
            str(order),
            tuple(transpose) if transpose is not None else None,
            tuple(flip_axes),
            float(nan_value),
            getattr(mapping, "__qualname__", type(mapping).__name__),
        )
    )
    if cache and cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        with np.load(cache_path) as data:
            if "fingerprint" in data.files and str(data["fingerprint"]) == fingerprint:
                return LabelVolume(
                    labels=data["labels"],
                    dx=float(data["dx"]),
                    origin=tuple(data["origin"].tolist()),
                )
        # stale-by-arguments: fall through and re-parse

    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if values.size != int(np.prod(shape)):
        raise ValueError(
            f"file holds {values.size} values but shape {shape} needs {int(np.prod(shape))}"
        )
    values = np.nan_to_num(values, nan=nan_value)
    vol = values.reshape(shape, order=order)
    labels = np.asarray(mapping(vol))
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("mapping must return an integer label array")
    if labels.shape != vol.shape:
        raise ValueError(f"mapping changed the shape: {vol.shape} -> {labels.shape}")
    if transpose is not None:
        labels = np.transpose(labels, transpose)
    for ax in flip_axes:
        labels = np.flip(labels, axis=ax)
    out = LabelVolume(labels=np.ascontiguousarray(labels), dx=dx)
    if cache:
        np.savez_compressed(
            cache_path,
            labels=out.labels,
            dx=out.dx,
            origin=np.asarray(out.origin),
            fingerprint=np.asarray(fingerprint),
        )
    return out


def breast_phantom_mapping(values: np.ndarray) -> np.ndarray:
    """The production breast phantom's value->tissue rule (notebook port).

    -2 -> 1 (skin), >0 -> 2 (fat/glandular), -4 -> 3 (muscle),
    everything else -> 4 (coupling gel / background).
    """
    labels = np.full(values.shape, 4, dtype=np.uint8)
    labels[values == -2] = 1
    labels[values > 0] = 2
    labels[values == -4] = 3
    return labels


def load_breast_phantom(path: str | Path, cache: bool = True) -> LabelVolume:
    """Load the production mtype phantom (0.5 mm, 310x355x253, Fortran order)."""
    return load_labels_txt(
        path,
        shape=(310, 355, 253),
        dx=0.5e-3,
        mapping=breast_phantom_mapping,
        order="F",
        transpose=(2, 1, 0),
        flip_axes=(0,),
        cache=cache,
    )
