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
from scipy.ndimage import zoom

RESAMPLE_METHODS = ("nearest", "smooth")


@dataclass(frozen=True)
class LabelVolume:
    """Integer label volume with isotropic spacing [m] and physical origin."""

    labels: np.ndarray
    dx: float
    origin: tuple[float, ...] = field(default=None)  # type: ignore[assignment]

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
        """Return the volume resampled to spacing ``dx_new`` [m]."""
        if dx_new <= 0:
            raise ValueError(f"dx_new must be > 0, got {dx_new}")
        if method not in RESAMPLE_METHODS:
            raise ValueError(f"unknown method {method!r}; pick from {RESAMPLE_METHODS}")
        factor = self.dx / dx_new
        if abs(factor - 1.0) < 1e-12:
            return self
        if method == "nearest":
            out = zoom(self.labels, zoom=factor, order=0, mode="nearest")
        else:  # smooth: one-hot linear + argmax (never invents labels)
            present = self.label_set
            best_score = None
            out = None
            for lab in present:
                score = zoom(
                    (self.labels == lab).astype(np.float32),
                    zoom=factor,
                    order=1,
                    mode="nearest",
                )
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
    and reloaded from there when the cache is newer than the source.
    """
    path = Path(path)
    cache_path = path.with_suffix(path.suffix + ".labels.npz")
    if cache and cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        return LabelVolume.load_npz(cache_path)

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
        out.save_npz(cache_path)
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
