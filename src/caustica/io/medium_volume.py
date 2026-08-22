"""``medium_volume`` — the ONE door for volume media (M10k/W0a, D16/D28).

A single ``.npz`` that a simulation eats directly, owned by caustica. Every
phantom source (UWCEM or anyone else's) enters the library through this
format; nothing source-specific lives here.

Layout (a single ``.npz``)::

    labels        int32  (nx, ny, nz)   material ids
    dx            float64 scalar        isotropic spacing [m]
    origin        float64 (3,)          physical position of voxel (0,0,0) [m]
    materials     str    JSON           a caustica MaterialDB, verbatim
    meta          str    JSON           provenance, statistics, citation, ...
    format        str                   "caustica-medium-volume/1"
    alpha,rho,c,beta  float32 (nx,ny,nz)  OPTIONAL dense property volumes

Two honest modes, recorded in the file itself:

* **label mode** — ``labels`` + ``materials`` reconstruct the medium exactly
  (piecewise-constant tissue);
* **continuous mode** — the four dense volumes are stored because they carry
  information the labels cannot (measured per-voxel heterogeneity, noise);
  ``labels`` stays as the id map for tissue-wise analysis.

The grid belongs to the file: shape and dx are read, never chosen, so a job
cannot silently run a resampled ghost of the data (the ``medium_volume`` job
kind rejects an explicit ``grid`` section).

Compatibility: this is the ``uwcem_phantoms`` export layout promoted into the
library — the reader accepts the pre-split tags (``caustica-phantom/1`` and
the pre-rename ``hifusim-phantom/1``) so the existing multi-GB local datasets
load unchanged, byte for byte, with no rebuild (T7). The writer is public
(D28): ``write_medium_volume(...)`` is the single source of the format, and
the ``uwcem-phantom`` repository calls it instead of carrying its own.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from caustica.core.grid import Grid
from caustica.core.pml import PMLSpec
from caustica.geometry.volumes import LabelVolume
from caustica.materials import MaterialDB, water
from caustica.medium import Medium

MEDIUM_VOLUME_FORMAT = "caustica-medium-volume/1"
#: Tags written by the pre-split ``uwcem_phantoms`` exporter (and its
#: pre-rename spelling). Readers accept them so the existing local datasets
#: — hours of CPU work — never need a rebuild just to restamp a string.
LEGACY_FORMAT_TAGS = frozenset({"caustica-phantom/1", "hifusim-phantom/1"})
#: What the reader accepts — retire aliases by shrinking THIS set only.
ACCEPTED_FORMAT_TAGS = frozenset({MEDIUM_VOLUME_FORMAT}) | LEGACY_FORMAT_TAGS

#: The four dense volumes a :class:`~caustica.medium.Medium` is built from,
#: in the order they are stored.
PROPERTY_NAMES = ("alpha", "rho", "c", "beta")


class MediumVolumeError(RuntimeError):
    """A medium-volume file is missing something it must have."""


def _label_histogram(labels: np.ndarray, n: int) -> np.ndarray:
    """Counts per id in ``[0, n)``, chunked — no full-volume sort/copy."""
    counts = np.zeros(n, dtype=np.int64)
    flat = labels.ravel()
    step = 16_000_000
    for start in range(0, flat.size, step):
        counts += np.bincount(flat[start : start + step], minlength=n)
    return counts


@dataclass
class MediumVolume:
    """A simulation-ready volume medium: labels + materials (+ dense props)."""

    labels: np.ndarray
    dx: float
    materials: MaterialDB
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    properties: dict[str, np.ndarray] | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError(f"labels must be integer-typed, got {labels.dtype}")
        if labels.ndim != 3:
            raise ValueError(f"medium volume labels must be 3-D, got {labels.ndim}-D")
        if self.dx <= 0:
            raise ValueError(f"dx must be > 0, got {self.dx}")
        self.labels = np.ascontiguousarray(labels, dtype=np.int32)
        self.origin = tuple(float(o) for o in self.origin)  # type: ignore[assignment]
        if len(self.origin) != 3:
            raise ValueError(f"origin needs 3 entries, got {self.origin}")
        if self.labels.size:
            n = max(int(i) for i in self.materials.ids) + 1
            lo_id, hi_id = int(self.labels.min()), int(self.labels.max())
            if lo_id < 0 or hi_id >= n:
                raise MediumVolumeError(
                    f"labels span ids [{lo_id}, {hi_id}] with no entry in the MaterialDB "
                    f"(known: {list(self.materials.ids)})"
                )
            present = np.flatnonzero(_label_histogram(self.labels, n))
            missing = [int(i) for i in present if int(i) not in self.materials]
            if missing:
                raise MediumVolumeError(
                    f"labels use ids {missing} with no entry in the MaterialDB "
                    f"(known: {list(self.materials.ids)})"
                )
        if self.properties is not None:
            if set(self.properties) != set(PROPERTY_NAMES):
                raise ValueError(
                    f"properties must carry exactly {PROPERTY_NAMES}, got {sorted(self.properties)}"
                )
            self.properties = {
                n: np.ascontiguousarray(self.properties[n], dtype=np.float32)
                for n in PROPERTY_NAMES
            }
            shapes = {n: v.shape for n, v in self.properties.items()}
            if set(shapes.values()) != {self.labels.shape}:
                raise ValueError(f"property volumes {shapes} != labels {self.labels.shape}")

    # ---------------------------------------------------------- geometry

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.labels.shape  # type: ignore[return-value]

    @property
    def is_continuous(self) -> bool:
        """True when per-voxel properties are stored (not derivable from labels)."""
        return self.properties is not None

    def grid(self, pml: PMLSpec | None = None) -> Grid:
        """A :class:`~caustica.core.grid.Grid` matching this volume exactly."""
        return Grid(shape=self.shape, dx=self.dx, pml=pml)

    def label_volume(self) -> LabelVolume:
        return LabelVolume(labels=self.labels, dx=self.dx, origin=self.origin)

    def c_min(self) -> float:
        if self.properties is not None:
            return float(self.properties["c"].min())
        return min(m.c for m in self.materials.materials.values())

    # ------------------------------------------------------------- medium

    def to_medium(self, linear: bool = False) -> Medium:
        """Build the :class:`~caustica.medium.Medium` the solvers consume.

        Continuous mode passes the stored volumes straight through (keeping
        the labels as ``id_map``); label mode goes through
        :meth:`Medium.from_id_map`, which validates every id against the
        MaterialDB. ``linear=True`` zeroes ``beta`` (a cheap linear pass
        before committing to Westervelt — hand-zeroing volumes at call sites
        is how someone eventually zeroes the wrong one).
        """
        if self.properties is None:
            medium = Medium.from_id_map(self.labels.astype(np.int64), self.materials)
        else:
            p = self.properties
            medium = Medium(
                alpha=p["alpha"], rho=p["rho"], c=p["c"], beta=p["beta"], id_map=self.labels
            )
        if linear:
            medium.beta = np.zeros_like(medium.beta)
        return medium


# ------------------------------------------------------------------------ IO


def write_medium_volume(
    path: str | Path,
    *,
    dx: float,
    labels: np.ndarray | None = None,
    materials: MaterialDB | None = None,
    properties: dict[str, np.ndarray] | None = None,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    meta: dict | None = None,
    compresslevel: int = 1,
) -> Path:
    """Write a ``medium_volume`` file from plain numpy arrays (D28).

    Give ``labels`` + ``materials`` (label mode), ``properties`` (a dict with
    ``alpha``/``rho``/``c``/``beta`` float32 volumes, continuous mode), or
    both. With properties only, ``labels`` defaults to all-zeros and
    ``materials`` to a single water entry — bookkeeping the format requires;
    the medium is built from the dense volumes regardless.

    ``compresslevel`` is the zip deflate level: numpy's ``savez_compressed``
    pins 6, which measured ~3 MB/s on pval-like float32 versus ~20 MB/s at
    level 1 for the same output size. The file stays a plain ``.npz``.
    """
    if labels is None and properties is None:
        raise ValueError("write_medium_volume needs labels and/or properties")
    if properties is not None:
        if set(properties) != set(PROPERTY_NAMES):
            raise ValueError(
                f"properties must carry exactly {PROPERTY_NAMES}, got {sorted(properties)}"
            )
        shape = next(iter(properties.values())).shape
        if labels is None:
            labels = np.zeros(shape, dtype=np.int32)
    if materials is None:
        materials = MaterialDB(materials={0: water()})

    # Validation (id coverage, shapes, dtypes) lives in ONE place:
    vol = MediumVolume(
        labels=labels,
        dx=float(dx),
        materials=materials,
        origin=origin,
        properties=properties,
        meta=dict(meta or {}),
    )

    file_meta = dict(vol.meta)
    file_meta["stored_properties"] = vol.properties is not None
    file_meta["shape"] = list(vol.shape)
    file_meta["dx_mm"] = vol.dx * 1e3

    arrays: dict[str, np.ndarray] = {
        "labels": vol.labels,
        "dx": np.asarray(vol.dx, dtype=np.float64),
        "origin": np.asarray(vol.origin, dtype=np.float64),
        "materials": np.asarray(vol.materials.model_dump_json()),
        "meta": np.asarray(json.dumps(file_meta, sort_keys=True, default=str)),
        "format": np.asarray(MEDIUM_VOLUME_FORMAT),
    }
    if vol.properties is not None:
        arrays.update(vol.properties)

    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.part")
    # Hand-rolled npz (zip of .npy members, identical layout to
    # np.savez_compressed) so `compresslevel` is honoured — numpy exposes no
    # way to set it. Streamed member-by-member; the temp name + replace keeps
    # the write atomic.
    with open(tmp, "wb") as fh:
        with zipfile.ZipFile(
            fh, "w", zipfile.ZIP_DEFLATED, allowZip64=True, compresslevel=compresslevel
        ) as zf:
            for name, arr in arrays.items():
                with zf.open(name + ".npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, np.asanyarray(arr), allow_pickle=False)
    tmp.replace(path)
    return path


def load_medium_volume(path: str | Path) -> MediumVolume:
    """Read a ``medium_volume`` file (current or legacy-tagged)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        for required in ("labels", "dx", "materials"):
            if required not in keys:
                raise MediumVolumeError(
                    f"{path} is not a caustica medium volume (missing {required!r}); "
                    f"it holds {sorted(keys)}"
                )
        tag = str(data["format"]) if "format" in keys else "unknown"
        if tag not in ACCEPTED_FORMAT_TAGS and tag != "unknown":
            raise MediumVolumeError(
                f"{path} has format {tag!r}, expected one of {sorted(ACCEPTED_FORMAT_TAGS)}"
            )
        materials = MaterialDB.model_validate_json(str(data["materials"]))
        meta = json.loads(str(data["meta"])) if "meta" in keys else {}
        origin = tuple(data["origin"].tolist()) if "origin" in keys else (0.0, 0.0, 0.0)
        props = None
        if all(n in keys for n in PROPERTY_NAMES):
            props = {n: data[n] for n in PROPERTY_NAMES}
        return MediumVolume(
            labels=data["labels"],
            dx=float(data["dx"]),
            materials=materials,
            origin=origin,  # type: ignore[arg-type]
            properties=props,
            meta=meta,
        )
