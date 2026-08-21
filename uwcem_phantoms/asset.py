"""The export: one ``.npz`` a simulation can eat.

This file is the whole point of the submodule, so its format is designed
around one requirement — **it must import into the main simulation without
new code** — and one nicety: it should also carry everything a human or a GUI
needs to understand what they are looking at.

Layout (a single ``.npz``)::

    labels        int32  (nx, ny, nz)   material ids
    dx            float64 scalar        isotropic spacing [m]
    origin        float64 (3,)          physical position of voxel (0,0,0) [m]
    materials     str    JSON           a caustica MaterialDB, verbatim
    meta          str    JSON           spec, provenance, statistics, citation
    format        str                   "caustica-phantom/1"
    alpha,rho,c,beta  float32 (nx,ny,nz)  OPTIONAL dense property volumes

The first three keys are exactly the ones
:meth:`caustica.geometry.volumes.LabelVolume.load_npz` reads, and numpy's npz
reader ignores keys it was not asked for. So the same file is simultaneously:

* ``LabelVolume.load_npz(path)`` — works today, no new code;
* ``PhantomAsset.load(path).to_medium()`` — richer: brings the MaterialDB and
  any continuous per-voxel properties along.

Why continuous volumes are optional: with heterogeneity off, the medium is
exactly reconstructible from ``labels`` + ``materials``, and storing four
float32 volumes would inflate a 2 MB file to 500 MB for nothing. With pval or
noise on, the per-voxel values are NOT derivable from the labels, so they are
stored — that is the only honest choice, and the metadata records which mode
the file is in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from caustica.core.grid import Grid
from caustica.core.pml import PMLSpec
from caustica.geometry.volumes import LabelVolume
from caustica.materials import MaterialDB
from caustica.medium import Medium
from uwcem_phantoms.heterogeneity import PROPERTY_NAMES, PropertyVolumes
from uwcem_phantoms.paths import ensure_dir, export_path

FORMAT_TAG = "caustica-phantom/1"
#: Tags written before the library was renamed (2026-08-21, pre-public).
#: Readers accept them so the existing 5.66 GB aligned dataset — ~1 h of
#: CPU work — does not have to be rebuilt just to restamp a string.
LEGACY_FORMAT_TAGS = frozenset({"hifusim-phantom/1"})
#: What a reader accepts — retire the aliases by shrinking THIS set only.
ACCEPTED_FORMAT_TAGS = frozenset({FORMAT_TAG}) | LEGACY_FORMAT_TAGS


class PhantomAssetError(RuntimeError):
    """An export file is missing something it must have."""


@dataclass
class PhantomAsset:
    """A simulation-ready phantom: labels + materials (+ dense properties)."""

    labels: np.ndarray
    dx: float
    materials: MaterialDB
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    properties: PropertyVolumes | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError(f"labels must be integer-typed, got {labels.dtype}")
        if labels.ndim != 3:
            raise ValueError(f"phantom labels must be 3-D, got {labels.ndim}-D")
        if self.dx <= 0:
            raise ValueError(f"dx must be > 0, got {self.dx}")
        self.labels = np.ascontiguousarray(labels, dtype=np.int32)
        self.origin = tuple(float(o) for o in self.origin)  # type: ignore[assignment]
        if len(self.origin) != 3:
            raise ValueError(f"origin needs 3 entries, got {self.origin}")
        # Presence via range check + chunked histogram, not np.unique: the
        # full sort copies and sorts the whole volume (945 MB and 5.7 s at the
        # dataset's 236 Mvox common grid) to answer "which of ~10 ids appear",
        # and this constructor runs on every build, align and load (review
        # finding, 2026-08-19).
        if self.labels.size:
            from uwcem_phantoms.processing import class_histogram

            n = max(int(i) for i in self.materials.ids) + 1
            lo_id, hi_id = int(self.labels.min()), int(self.labels.max())
            if lo_id < 0 or hi_id >= n:
                raise PhantomAssetError(
                    f"labels span ids [{lo_id}, {hi_id}] with no entry in the MaterialDB "
                    f"(known: {list(self.materials.ids)})"
                )
            present = np.flatnonzero(class_histogram(self.labels, n))
            missing = [int(i) for i in present if int(i) not in self.materials]
            if missing:
                raise PhantomAssetError(
                    f"labels use ids {missing} with no entry in the MaterialDB "
                    f"(known: {list(self.materials.ids)})"
                )
        if self.properties is not None and self.properties.shape != self.labels.shape:
            raise ValueError(
                f"property volumes {self.properties.shape} != labels {self.labels.shape}"
            )

    # ---------------------------------------------------------- geometry

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.labels.shape  # type: ignore[return-value]

    @property
    def n_voxels(self) -> int:
        return int(self.labels.size)

    @property
    def extent(self) -> tuple[float, float, float]:
        return tuple(n * self.dx for n in self.shape)  # type: ignore[return-value]

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        return tuple(round(e * 1e3, 2) for e in self.extent)  # type: ignore[return-value]

    @property
    def is_continuous(self) -> bool:
        """True when per-voxel properties are stored (pval and/or noise)."""
        return self.properties is not None

    # ------------------------------------------------ caustica conversions

    def grid(self, pml: PMLSpec | None = None) -> Grid:
        """A :class:`~caustica.core.grid.Grid` matching this phantom exactly."""
        return Grid(shape=self.shape, dx=self.dx, pml=pml)

    def label_volume(self) -> LabelVolume:
        """The label map as a :class:`~caustica.geometry.volumes.LabelVolume`."""
        return LabelVolume(labels=self.labels, dx=self.dx, origin=self.origin)

    def to_medium(self, linear: bool = False) -> Medium:
        """Build the :class:`~caustica.medium.Medium` the solvers consume.

        Continuous mode passes the stored volumes straight through (keeping
        ``id_map`` so tissue-wise analysis still works); label mode goes
        through :meth:`Medium.from_id_map`, which validates every id against
        the MaterialDB rather than guessing.

        ``linear=True`` zeroes ``beta``. Real tissue has ``beta`` around 4-6,
        so a phantom medium is ALWAYS nonlinear and the ``linear`` solver
        rejects it outright (``SolverCapabilityError``). That refusal is
        correct — but a cheap linear pass before committing to Westervelt is
        a normal way to work, and hand-zeroing four volumes at every call
        site is how someone eventually zeroes the wrong one.
        """
        if self.properties is None:
            medium = Medium.from_id_map(self.labels.astype(np.int64), self.materials)
        else:
            p = self.properties
            medium = Medium(alpha=p.alpha, rho=p.rho, c=p.c, beta=p.beta, id_map=self.labels)
        if linear:
            medium.beta = np.zeros_like(medium.beta)
        return medium

    def label_of(self, name: str) -> int | None:
        """Material id whose name contains ``name`` (case-insensitive)."""
        for i, m in self.materials.materials.items():
            if name.lower() in m.name.lower():
                return int(i)
        return None

    # ------------------------------------------------------------- stats

    def histogram(self) -> dict[int, int]:
        from uwcem_phantoms.processing import class_histogram

        counts = class_histogram(self.labels, max(self.materials.ids) + 1)
        return {int(i): int(counts[i]) for i in self.materials.ids if i < counts.size}

    def summary(self) -> str:
        mode = "continuous properties" if self.is_continuous else "labels + MaterialDB"
        lines = [
            f"phantom  : {self.meta.get('phantom_id', '?')} "
            f"(ACR class {self.meta.get('acr_class', '?')})",
            f"grid     : {self.shape[0]}x{self.shape[1]}x{self.shape[2]} voxels "
            f"@ {self.dx * 1e3:g} mm  =  "
            f"{self.extent_mm[0]}x{self.extent_mm[1]}x{self.extent_mm[2]} mm",
            f"voxels   : {self.n_voxels:,}",
            f"mode     : {mode}",
            f"materials: {len(self.materials.materials)} "
            f"({self.meta.get('tissue_model', '?')} model, "
            f"f0 = {self.meta.get('f0_mhz', '?')} MHz)",
        ]
        total = self.n_voxels
        for i, n in sorted(self.histogram().items()):
            if n == 0:
                continue
            m = self.materials[i]
            lines.append(
                f"  id {i:<2} {m.name:<38} {n:>12,}  {100 * n / total:5.2f}%  "
                f"c={m.c:6.1f}  rho={m.rho:6.1f}  a={m.alpha_np_m:6.2f} Np/m  b={m.beta:.2f}"
            )
        if self.is_continuous:
            st = self.properties.stats()  # type: ignore[union-attr]
            lines.append("  continuous ranges:")
            for name in PROPERTY_NAMES:
                s = st[name]
                lines.append(
                    f"    {name:<6} [{s['min']:.4g}, {s['max']:.4g}]  "
                    f"mean {s['mean']:.4g}  std {s['std']:.4g}"
                )
        return "\n".join(lines)

    # ---------------------------------------------------------------- IO

    def save(
        self,
        path: str | Path | None = None,
        store_properties: str = "auto",
        compresslevel: int = 1,
    ) -> Path:
        """Write the single importable file; returns the path actually used.

        ``store_properties``: ``"auto"`` writes the dense volumes only when
        they carry information the labels cannot ("continuous" mode),
        ``"always"`` forces them, ``"never"`` drops them (the medium then
        falls back to piecewise-constant on load — a real change, so the
        metadata records it).

        ``compresslevel``: zlib level for the zip members. numpy's
        ``savez_compressed`` pins level 6, which measured ~3 MB/s on pval-like
        float32 versus ~20 MB/s at level 1 for the SAME output size — on a
        4.7 GB dataset export that difference is minutes per file (review
        finding, 2026-08-19). The file stays a plain ``.npz``: ``np.load``,
        ``LabelVolume.load_npz`` and :meth:`load` all read it unchanged.
        """
        if store_properties not in ("auto", "always", "never"):
            raise ValueError(
                f"store_properties must be auto|always|never, got {store_properties!r}"
            )
        if path is None:
            name = self.meta.get("name") or "phantom"
            path = export_path(name)
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        ensure_dir(path.parent)

        props = self.properties
        if store_properties == "never":
            props = None
        elif store_properties == "always" and props is None:
            props = self._materialize_properties()

        meta = dict(self.meta)
        meta["stored_properties"] = props is not None
        meta["shape"] = list(self.shape)
        meta["dx_mm"] = self.dx * 1e3
        meta["extent_mm"] = list(self.extent_mm)
        meta["histogram"] = {str(k): v for k, v in self.histogram().items()}

        arrays: dict[str, np.ndarray] = {
            "labels": self.labels,
            "dx": np.asarray(self.dx, dtype=np.float64),
            "origin": np.asarray(self.origin, dtype=np.float64),
            "materials": np.asarray(self.materials.model_dump_json()),
            "meta": np.asarray(json.dumps(meta, sort_keys=True, default=str)),
            "format": np.asarray(FORMAT_TAG),
        }
        if props is not None:
            arrays.update(props.asdict())

        tmp = path.with_suffix(".npz.part")
        # Hand-rolled npz (zip of .npy members, identical layout to
        # np.savez_compressed) so `compresslevel` is honoured — numpy exposes
        # no way to set it. Streamed member-by-member; nothing is buffered
        # beyond zlib's window. File OBJECT + explicit temp name keeps the
        # write atomic (np.savez_compressed would also append ".npz" to a
        # bare temp name and strand it).
        import zipfile

        with open(tmp, "wb") as fh:
            with zipfile.ZipFile(
                fh, "w", zipfile.ZIP_DEFLATED, allowZip64=True, compresslevel=compresslevel
            ) as zf:
                for name, arr in arrays.items():
                    with zf.open(name + ".npy", "w", force_zip64=True) as member:
                        np.lib.format.write_array(member, np.asanyarray(arr), allow_pickle=False)
        tmp.replace(path)
        return path

    def _materialize_properties(self) -> PropertyVolumes:
        """Expand labels + MaterialDB into dense volumes (for store=always)."""
        med = Medium.from_id_map(self.labels.astype(np.int64), self.materials)
        return PropertyVolumes(alpha=med.alpha, rho=med.rho, c=med.c, beta=med.beta)

    @staticmethod
    def load(path: str | Path) -> PhantomAsset:
        """Read a file written by :meth:`save`."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            for required in ("labels", "dx", "materials"):
                if required not in keys:
                    raise PhantomAssetError(
                        f"{path} is not a caustica phantom export (missing {required!r}); "
                        f"it holds {sorted(keys)}"
                    )
            tag = str(data["format"]) if "format" in keys else "unknown"
            if tag not in ACCEPTED_FORMAT_TAGS and tag != "unknown":
                raise PhantomAssetError(f"{path} has format {tag!r}, expected {FORMAT_TAG!r}")
            materials = MaterialDB.model_validate_json(str(data["materials"]))
            meta = json.loads(str(data["meta"])) if "meta" in keys else {}
            origin = tuple(data["origin"].tolist()) if "origin" in keys else (0.0, 0.0, 0.0)
            props = None
            if all(n in keys for n in PROPERTY_NAMES):
                props = PropertyVolumes(**{n: data[n] for n in PROPERTY_NAMES})
            return PhantomAsset(
                labels=data["labels"],
                dx=float(data["dx"]),
                materials=materials,
                origin=origin,  # type: ignore[arg-type]
                properties=props,
                meta=meta,
            )


def load_phantom(path: str | Path) -> PhantomAsset:
    """Read a phantom export (module-level convenience for simulation code)."""
    return PhantomAsset.load(path)
