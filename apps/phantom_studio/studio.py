"""Studio state: builds, caches, and the arrays the browser draws.

Kept deliberately free of HTTP so it can be exercised from a REPL or a test
without a socket. :mod:`apps.phantom_studio.server` is a thin translation
layer on top of this.

Two caches earn their keep:

* decoded phantoms (``RawPhantom``) — decoding costs seconds, and every
  parameter change in the UI would otherwise pay it again;
* finished builds, keyed by the spec's own JSON — dragging a slider back and
  forth then costs nothing.

Both are small and bounded; the build cache is an LRU of three because a
single 0.3 mm build can hold a gigabyte.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from uwcem_phantoms import catalog
from uwcem_phantoms.asset import PhantomAsset
from uwcem_phantoms.builder import build, dx_for_budget, plan
from uwcem_phantoms.heterogeneity import PROPERTY_NAMES
from uwcem_phantoms.paths import export_dir, export_path
from uwcem_phantoms.processing import class_histogram
from uwcem_phantoms.reader import RawPhantom, load_raw
from uwcem_phantoms.spec import PhantomSpec
from uwcem_phantoms.tissue import tissue_table

#: Scalar fields the viewer can display, beyond the label map.
SCALAR_FIELDS = (*PROPERTY_NAMES, "impedance")
FIELDS = ("labels", *SCALAR_FIELDS)

MAX_BUILDS = 3
#: Decoded phantoms held in memory (see Studio.raw).
MAX_RAW = 2


@dataclass
class BuildRecord:
    """One finished build plus everything the viewer needs about it."""

    key: str
    asset: PhantomAsset
    spec: PhantomSpec
    requested_dx_mm: float
    preview: bool
    seconds: float
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    def field_plane(self, name: str, axis: int, index: int) -> np.ndarray:
        """One plane of the named field, WITHOUT materializing the volume.

        A piecewise-constant build stores only labels, so the obvious
        ``lut[labels]`` builds a fresh full-size float32 array and throws all
        but one plane away: 199 MB and 131 ms to return a 132 kB slice on a
        0.4 mm build. Taking the label plane first and expanding THAT is the
        same answer for 1/300th of the work (review finding, 2026-08-18).
        """
        if name == "labels":
            return np.take(self.asset.labels, index, axis=axis)
        if name not in SCALAR_FIELDS:
            raise KeyError(f"unknown field {name!r}; pick from {FIELDS}")
        props = self.asset.properties
        if props is not None:
            if name == "impedance":
                return np.take(props.rho, index, axis=axis) * np.take(props.c, index, axis=axis)
            return np.take(getattr(props, name), index, axis=axis)
        return self._lut(name)[np.take(self.asset.labels, index, axis=axis)]

    def field_volume(self, name: str) -> np.ndarray:
        """The named field as a 3-D array (materialized on demand)."""
        if name == "labels":
            return self.asset.labels
        if name not in SCALAR_FIELDS:
            raise KeyError(f"unknown field {name!r}; pick from {FIELDS}")
        props = self.asset.properties
        if props is not None:
            if name == "impedance":
                return props.impedance()
            return getattr(props, name)
        # Piecewise-constant build: expand through a lookup instead of
        # storing four volumes we can regenerate exactly.
        lut = self._lut(name)
        return lut[self.asset.labels]

    def _lut(self, name: str) -> np.ndarray:
        db = self.asset.materials
        size = max(db.ids) + 1
        out = np.zeros(size, dtype=np.float32)
        for i in db.ids:
            m = db[i]
            out[i] = {
                "alpha": m.alpha_np_m,
                "rho": m.rho,
                "c": m.c,
                "beta": m.beta,
                "impedance": m.rho * m.c,
            }[name]
        return out

    def bath_level(self, name: str) -> float | None:
        """The coupling medium's value on this field's 0..255 display scale."""
        if name == "labels":
            return None
        bath = next(
            (
                i
                for i in self.asset.materials.ids
                if "coupling" in self.asset.materials[i].name.lower()
            ),
            None,
        )
        if bath is None:
            return None
        value = self._lut(name)[bath]
        vmin, vmax = self.range_of(name)
        span = vmax - vmin
        return float(np.clip((value - vmin) * 255.0 / span, 0, 255)) if span > 0 else 0.0

    def range_of(self, name: str) -> tuple[float, float]:
        """Global min/max of a field, cached.

        Global, not per-slice: a colour scale that rescales while the user
        drags the slice slider makes tissues appear to change properties.
        """
        if name not in self.ranges:
            props = self.asset.properties
            if props is None:
                # Piecewise-constant: the extremes are in the ~10-entry lookup,
                # over the ids that actually occur. Expanding five full volumes
                # just to take a min and a max cost 1.17 s and 395 MB per build.
                lut = self._lut(name)
                present = np.unique(self.asset.labels)
                vals = lut[present]
                self.ranges[name] = (float(vals.min()), float(vals.max()))
            else:
                v = self.field_volume(name)
                self.ranges[name] = (float(v.min()), float(v.max()))
        return self.ranges[name]

    def describe(self) -> dict:
        a = self.asset
        meta = a.meta
        total = a.n_voxels
        hist = a.histogram()
        materials = []
        for row in meta.get("materials_table", []):
            n = hist.get(row["id"], 0)
            materials.append({**row, "voxels": n, "fraction": n / total if total else 0.0})
        return {
            "key": self.key,
            "shape": list(a.shape),
            "dx_mm": a.dx * 1e3,
            "requested_dx_mm": self.requested_dx_mm,
            "preview": self.preview,
            "extent_mm": list(a.extent_mm),
            "n_voxels": total,
            "continuous": a.is_continuous,
            "phantom_id": meta.get("phantom_id"),
            "acr_class": meta.get("acr_class"),
            "tissue_model": meta.get("tissue_model"),
            "f0_mhz": meta.get("f0_mhz"),
            "materials": materials,
            "warnings": meta.get("warnings", []),
            "log": meta.get("log", []),
            "heterogeneity": meta.get("heterogeneity", {}),
            "name": meta.get("name"),
            "seconds": round(self.seconds, 2),
            "ranges": {f: list(self.range_of(f)) for f in FIELDS if f != "labels"},
            "summary": a.summary(),
        }


class Studio:
    """Everything the GUI needs, with the caches that make it feel instant."""

    def __init__(self, preview_mvox: float = 10.0):
        self.preview_mvox = preview_mvox
        # One decoded phantom is up to 1.05 GB with pval; keeping every id the
        # user clicked through would pin ~9 GB. Two is enough to make
        # "compare these two phantoms" fast without hoarding.
        self._raw: OrderedDict[str, RawPhantom] = OrderedDict()
        self._builds: OrderedDict[str, BuildRecord] = OrderedDict()
        # Two locks, different lifetimes. The server is threaded, so a slice
        # request can land mid-build: holding ONE lock for the whole build
        # would freeze the viewer for the duration, and holding none would let
        # a reader walk the OrderedDict while build() calls move_to_end or
        # popitem on it.
        self._build_lock = threading.Lock()  # serializes builds (long)
        self._cache_lock = threading.Lock()  # guards the dict itself (short)
        self.progress: tuple[str, float] = ("idle", 0.0)

    # ------------------------------------------------------------- catalog

    def state(self) -> dict:
        return {
            "catalog": catalog.status(),
            "acr_classes": catalog.ACR_CLASSES,
            "fields": list(FIELDS),
            "defaults": PhantomSpec(phantom_id=catalog.PHANTOM_IDS[-1]).model_dump(mode="json"),
            "schema": PhantomSpec.model_json_schema(),
            "citation": catalog.CITATION,
            "preview_mvox": self.preview_mvox,
            "export_dir": str(export_dir()),
            "exports": self.exports(),
        }

    def exports(self) -> list[dict]:
        d = export_dir()
        if not d.exists():
            return []
        return sorted(
            (
                {"name": p.name, "path": str(p), "mb": round(p.stat().st_size / 1e6, 2)}
                for p in d.glob("*.npz")
            ),
            key=lambda r: r["name"],
        )

    def fetch(self, phantom_id: str, with_pval: bool = True) -> dict:
        catalog.fetch(phantom_id, with_pval=with_pval)
        return {"ok": True, "catalog": catalog.status()}

    def raw(self, phantom_id: str, with_pval: bool) -> RawPhantom:
        with self._cache_lock:
            cached = self._raw.get(phantom_id)
            if cached is not None:
                self._raw.move_to_end(phantom_id)
        if cached is None or (with_pval and not cached.has_pval):
            cached = load_raw(phantom_id, with_pval=with_pval)
            with self._cache_lock:
                self._raw[phantom_id] = cached
                self._raw.move_to_end(phantom_id)
                while len(self._raw) > MAX_RAW:
                    self._raw.popitem(last=False)
        return cached

    def tissues(self, model: str, f0_mhz: float) -> list[dict]:
        return tissue_table(model, f0=f0_mhz * 1e6).describe()

    # --------------------------------------------------------------- build

    def plan(self, spec: PhantomSpec) -> dict:
        raw = self.raw(spec.phantom_id, with_pval=False)
        p = plan(spec, raw)
        return {
            "shape": list(p.shape),
            "dx_mm": p.dx * 1e3,
            "n_voxels": p.n_voxels,
            "extent_mm": list(p.extent_mm),
            "peak_gb": round(p.peak_bytes / 1e9, 3),
            "exact": p.exact,
            "summary": p.summary(),
        }

    def build(self, spec: PhantomSpec, preview: bool) -> BuildRecord:
        """Build (or return a cached build) for ``spec``.

        In preview mode the resolution is coarsened until the domain fits the
        preview budget. The record keeps BOTH the effective and the requested
        dx so the UI can say "showing 0.7 mm, export will be 0.3 mm" instead
        of quietly lying about what was computed.
        """
        requested_dx_mm = spec.resolution.dx_mm or (catalog.NATIVE_DX * 1e3)
        effective = spec
        with self._build_lock:
            raw_light = self.raw(spec.phantom_id, with_pval=False)
            if preview:
                dx = dx_for_budget(spec, int(self.preview_mvox * 1e6), raw_light)
                if dx * 1e3 > requested_dx_mm * 1.001:
                    effective = spec.model_copy(deep=True)
                    effective.resolution = effective.resolution.model_copy(
                        update={"dx_mm": round(dx * 1e3, 4)}
                    )
            key = f"{effective.model_dump_json()}|preview={preview}"
            with self._cache_lock:
                hit = self._builds.get(key)
                if hit is not None:
                    self._builds.move_to_end(key)
            if hit is not None:
                return hit

            raw = self.raw(spec.phantom_id, with_pval=effective.heterogeneity.use_pval)

            def on_progress(msg: str, frac: float) -> None:
                self.progress = (msg, frac)

            import time

            t0 = time.perf_counter()
            asset = build(effective, raw=raw, progress=on_progress)
            record = BuildRecord(
                key=key,
                asset=asset,
                spec=effective,
                requested_dx_mm=requested_dx_mm,
                preview=preview and effective is not spec,
                seconds=time.perf_counter() - t0,
            )
            with self._cache_lock:
                self._builds[key] = record
                while len(self._builds) > MAX_BUILDS:
                    self._builds.popitem(last=False)
            self.progress = ("idle", 1.0)
            return record

    def get(self, key: str) -> BuildRecord:
        with self._cache_lock:
            record = self._builds.get(key)
        if record is None:
            raise KeyError("that build has expired from the cache; rebuild it")
        return record

    def export(self, key: str, name: str | None, store_properties: str) -> dict:
        record = self.get(key)
        asset = record.asset
        if record.preview:
            raise ValueError(
                "this is a PREVIEW build at a coarsened resolution; press 'Build full' "
                "before exporting so the file matches what you asked for"
            )
        if name:
            asset.meta["name"] = name
        path = asset.save(
            None if not name else export_path(name), store_properties=store_properties
        )
        return {
            "path": str(path),
            "mb": round(path.stat().st_size / 1e6, 2),
            "exports": self.exports(),
        }

    # --------------------------------------------------------------- views

    def slice_bytes(self, key: str, axis: int, index: int, fieldname: str):
        """One slice as uint8 plus the scaling needed to read it back.

        The browser gets bytes and a (vmin, vmax); it does the colour mapping.
        That keeps the server out of the palette business and makes changing
        colour maps instant — no round trip.
        """
        record = self.get(key)
        if not 0 <= axis < 3:
            raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
        n_axis = record.asset.shape[axis]
        index = int(np.clip(index, 0, n_axis - 1))
        plane = record.field_plane(fieldname, axis, index)
        if fieldname == "labels":
            data = np.ascontiguousarray(plane.astype(np.uint8))
            vmin, vmax = 0.0, 255.0
        else:
            vmin, vmax = record.range_of(fieldname)
            span = vmax - vmin
            scaled = (plane - vmin) * (255.0 / span) if span > 0 else np.zeros_like(plane)
            data = np.ascontiguousarray(np.clip(scaled, 0, 255).astype(np.uint8))
        return data, {
            "width": data.shape[1],
            "height": data.shape[0],
            "vmin": vmin,
            "vmax": vmax,
            "index": index,
            "n": n_axis,
        }

    def volume_bytes(self, key: str, max_dim: int, fieldname: str):
        """A downsampled uint8 volume for the WebGL ray caster."""
        record = self.get(key)
        vol = record.field_volume(fieldname)
        if fieldname == "labels":
            priority = _rarity_priority(vol, int(vol.max()) + 1)
            small = downsample_priority(vol, max_dim, priority)
            data = np.ascontiguousarray(small.astype(np.uint8))
            vmin, vmax = 0.0, 255.0
        else:
            small = downsample_mean(vol, max_dim)
            vmin, vmax = record.range_of(fieldname)
            span = vmax - vmin
            scaled = (small - vmin) * (255.0 / span) if span > 0 else np.zeros_like(small)
            data = np.ascontiguousarray(np.clip(scaled, 0, 255).astype(np.uint8))
        # The block reduction drops the remainder voxels, so the texture
        # covers a slightly SMALLER box than the build. Report the extent of
        # what the texture actually holds, or the viewer stretches it and the
        # anatomy drifts against the slice planes.
        block = [_block(n, max_dim) for n in vol.shape]
        covered = [d * b for d, b in zip(data.shape, block, strict=True)]
        return data, {
            "shape": list(data.shape),
            "full_shape": list(vol.shape),
            "covered_shape": covered,
            "vmin": vmin,
            "vmax": vmax,
            "dx_mm": record.asset.dx * 1e3,
            # Physical size of the FULL volume: only the axes above the budget
            # get block-reduced, so the viewer cannot recover the box from the
            # texture dimensions alone without squashing the anatomy.
            "extent_mm": [n * record.asset.dx * 1e3 for n in covered],
            # Where the coupling bath sits on this field's 0..255 scale. The
            # 3-D view needs it: on a scalar field the bath is 60% of the
            # voxels and its value is not extreme (water's sound speed sits
            # mid-range), so without knowing it the ray caster renders a
            # featureless block of water with the breast buried inside.
            "bath_u8": record.bath_level(fieldname),
        }


# ------------------------------------------------------------- downsampling


def _block(n: int, max_dim: int) -> int:
    return max(1, int(np.ceil(n / max_dim)))


def _rarity_priority(labels: np.ndarray, n_ids: int) -> np.ndarray:
    """Priority for block voting: rarer labels win.

    Without this, striding or majority voting erases exactly the structures
    that matter visually — the ~1.5 mm skin shell and thin glandular strands
    lose every vote to the fat and water around them.
    """
    counts = class_histogram(labels, n_ids).astype(np.float64)
    counts[counts == 0] = np.inf
    return (1.0 / counts).astype(np.float32)


#: Output rows per chunk when block-reducing. Keeps the scratch bounded:
#: scoring a whole 22 Mvox label volume at once needs an int64 index copy plus
#: a float32 score array — ~350 MB to produce a 1.8 MB texture.
CHUNK_ROWS = 8


def downsample_priority(labels: np.ndarray, max_dim: int, priority: np.ndarray) -> np.ndarray:
    """Block-reduce a label volume, keeping the highest-priority label.

    Chunked along the first axis: the intermediate score array is the
    expensive part, and it only ever has to exist for the rows being reduced.
    """
    b = tuple(_block(n, max_dim) for n in labels.shape)
    if b == (1, 1, 1):
        return labels
    out_shape = tuple(n // f for n, f in zip(labels.shape, b, strict=True))
    out = np.empty(out_shape, dtype=labels.dtype)
    for r0 in range(0, out_shape[0], CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, out_shape[0])
        slab = labels[
            r0 * b[0] : r1 * b[0],
            : out_shape[1] * b[1],
            : out_shape[2] * b[2],
        ]
        blocks = (
            slab.reshape(r1 - r0, b[0], out_shape[1], b[1], out_shape[2], b[2])
            .transpose(0, 2, 4, 1, 3, 5)
            .reshape(r1 - r0, out_shape[1], out_shape[2], -1)
        )
        pick = np.argmax(priority[blocks.astype(np.intp)], axis=-1)
        out[r0:r1] = np.take_along_axis(blocks, pick[..., None], axis=-1)[..., 0]
    return out


def downsample_mean(values: np.ndarray, max_dim: int) -> np.ndarray:
    """Block-average a continuous volume (anti-aliased, unlike striding)."""
    b = tuple(_block(n, max_dim) for n in values.shape)
    if b == (1, 1, 1):
        return values
    out_shape = tuple(n // f for n, f in zip(values.shape, b, strict=True))
    out = np.empty(out_shape, dtype=np.float32)
    for r0 in range(0, out_shape[0], CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, out_shape[0])
        slab = values[
            r0 * b[0] : r1 * b[0],
            : out_shape[1] * b[1],
            : out_shape[2] * b[2],
        ]
        out[r0:r1] = slab.reshape(r1 - r0, b[0], out_shape[1], b[1], out_shape[2], b[2]).mean(
            axis=(1, 3, 5)
        )
    return out
