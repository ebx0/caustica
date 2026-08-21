"""One grid for nine breasts: the standard aligned dataset caustica consumes.

:mod:`uwcem_phantoms.builder` builds ONE phantom well; a comparative study
needs all nine on the SAME grid, so that a transducer position, a focus, a
slicing script written for one phantom is valid for every other. This module
produces that dataset:

* every export shares one shape and one ``dx`` (default 0.25 mm);
* the skin front face sits at the same z-plane in every file (``front_gap_mm``
  of coupling water in front), so one transducer standoff fits all;
* the propagation axis stops at ``depth_limit_mm`` (default 100 mm). Behind
  the chest wall every phantom is pure coupling water — 2.75 mm of it in the
  deepest, 51.75 mm in the shallowest — and a HIFU study never focuses there,
  so the natural union box spends most of its depth on water and on the
  full-width chest-wall slab. Capping the axis is a DESTRUCTIVE crop by
  design: it drops that dead space, and in the deeper phantoms it also cuts
  real tissue off the back. What it cuts is counted per phantom, per class,
  and recorded in the manifest — never silently (see
  :func:`_align_into_common`);
* the PROTRUDING BREAST is centred transversally: near the chest wall the
  subcutaneous-fat slab spans the whole transverse extent, so the naive
  "centre the tissue bounding box" centres the crop window rather than the
  breast (review finding, 2026-08-19) — alignment therefore centres the
  bounding box of the protruding slices only (see
  :func:`_breast_transverse_bbox`);
* pval heterogeneity is ON and the property volumes are stored, so the file
  is directly loadable: ``load_phantom(path).to_medium()``.

Two-phase build, deliberately:

1. **Survey** (seconds) — measure every phantom's post-crop box at the native
   0.5 mm from the cached ``mtype`` volumes, scale by ``0.5 / dx``, and take
   the union. The common box is decided BEFORE any expensive build, so it
   cannot depend on build order; the survey also carries the builder's own
   fitted peak-RAM model, and :func:`build_dataset` refuses up front when the
   worst build would not fit in the RAM that is actually free.
2. **Build** (minutes per phantom) — run the normal pipeline per phantom
   (breast crop, smooth resample, pval), then pad into the common box with
   coupling medium. The survey adds :data:`SAFETY_VOX` per side because the
   smooth resample can move a tissue boundary by a voxel; the build MEASURES
   the real bounding box and raises rather than clipping if the survey was
   ever wrong.

Padding is physics, not bookkeeping: labels pad with the coupling-medium id
and each property volume pads with that material's own values — the same
numbers the in-volume water voxels carry — so the padded region is
indistinguishable from water the crop kept.

Outputs land in ``data/phantoms/`` at the repo root (see
:func:`uwcem_phantoms.paths.dataset_dir`) plus a ``manifest.json`` recording
the common grid, the per-phantom alignment actually achieved, and the
citation. Rebuilding a SUBSET merges into the existing manifest and reuses
its established grid; a rebuild with different parameters is refused rather
than silently mixing incompatible files. ``verify_dataset`` re-checks all of
it from disk — including files the manifest does NOT list, which are exactly
the ones a stale rebuild leaves behind.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from uwcem_phantoms import processing
from uwcem_phantoms.asset import PhantomAsset
from uwcem_phantoms.builder import _crop, _project_shape, build
from uwcem_phantoms.builder import plan as builder_plan
from uwcem_phantoms.catalog import CITATION, NATIVE_DX, PHANTOM_IDS
from uwcem_phantoms.heterogeneity import PROPERTY_NAMES
from uwcem_phantoms.orientation import to_canonical
from uwcem_phantoms.paths import dataset_dir, ensure_dir
from uwcem_phantoms.reader import N_CLASSES, load_raw
from uwcem_phantoms.spec import PhantomSpec
from uwcem_phantoms.tissue import TissueTable, tissue_table

#: Bumped to /2 when the depth limit landed: a /1 file has an uncapped z axis
#: and a manifest with no truncation record, so the two must not mix in one
#: directory even though the filenames collide.
DATASET_FORMAT = "caustica-phantom-dataset/2"
#: Pre-rename tag (2026-08-21): the existing dataset on disk carries it, both
#: in manifest.json and inside every npz's meta — accepted on read so the
#: 5.66 GB build survives the rename. New builds write DATASET_FORMAT.
LEGACY_DATASET_FORMATS = frozenset({"hifusim-phantom-dataset/2"})
#: What a reader accepts — retire the aliases by shrinking THIS set only.
ACCEPTED_DATASET_FORMATS = frozenset({DATASET_FORMAT}) | LEGACY_DATASET_FORMATS

#: The standard recipe. One place, so the CLI, the launcher and the tests all
#: mean the same dataset when they say "the dataset".
DATASET_DX_MM = 0.25
DATASET_F0_MHZ = 1.0
MARGIN_MM = 5.0
TISSUE_MODEL = "detailed"

#: Coupling water in front of the skin [mm]. This is the TRANSDUCER BUDGET, not
#: cosmetic padding: the PML sponge sits INSIDE the grid, and a focused bowl
#: needs room for its own shell in front of the apex. At 0.25 mm a typical PML
#: is 20 voxels (5 mm) and the production 128-element spiral is 11.6 mm deep
#: apex-to-rim, so 20 mm leaves ~3 mm of clearance for it and fits any bowl up
#: to about F/1 at ROC 60. A 5 mm gap — the value this dataset shipped with
#: first — leaves exactly zero usable water once a PML is attached.
FRONT_GAP_MM = 20.0

#: Hard ceiling on the propagation axis [mm], ``None`` to let the union box
#: decide. The uncapped union is 156.25 mm deep, and the back half of it is
#: water plus the flat chest-wall slab; 120 mm pays for the 20 mm transducer
#: budget above and still leaves 100 mm of tissue. This is a CROP, not padding:
#: phantoms deeper than the limit lose tissue at the back, which
#: :func:`_align_into_common` measures and reports rather than discarding
#: quietly.
DEPTH_LIMIT_MM: float | None = 120.0

#: A z-slice whose non-water voxel count exceeds this fraction of the
#: volume-wide transverse non-water bbox area is treated as chest wall when
#: measuring the breast centre. Mirrors ``CropSpec.breast_coverage``.
BREAST_COVERAGE = 0.9

#: Voxels added per side beyond the survey's union box. The survey scales
#: native-resolution measurements by ``0.5 / dx``; the smooth resample can
#: shift a tissue boundary by about one target voxel relative to that
#: arithmetic, and non-integer ratios round once more. Three voxels covers
#: both with room to spare; the build re-measures and raises if not.
SAFETY_VOX = 3

#: Absolute slack for the property-bounds check [SI units of each property].
#: float32 storage plus the ``lo + (hi - lo) * p`` blend round at ~1e-4 on a
#: sound speed of 1500; 1e-3 is comfortably above that and far below any
#: physical contrast.
BOUNDS_ATOL = 1e-3

#: How far [voxels] a file's measured breast centre may sit from the box
#: centre before verification fails. Half a voxel is the build's own rounding;
#: the other half absorbs the resample's boundary jitter between builds.
CENTER_TOL_VOX = 1.0

ProgressFn = Callable[[str], None]


class DatasetError(RuntimeError):
    """The dataset build or verification found something it must not ignore."""


def dataset_spec(
    phantom_id: str,
    dx_mm: float = DATASET_DX_MM,
    f0_mhz: float = DATASET_F0_MHZ,
    margin_mm: float = MARGIN_MM,
) -> PhantomSpec:
    """The per-phantom build recipe the dataset uses.

    ``fft_friendly`` is OFF here on purpose: the COMMON box does the final
    sizing (and is made FFT-friendly there), so per-phantom padding would
    only wedge asymmetric water between the breast and the alignment math.
    """
    return PhantomSpec(
        phantom_id=phantom_id,
        f0_mhz=f0_mhz,
        simplify={"tissue_model": TISSUE_MODEL},
        resolution={"dx_mm": dx_mm, "method": "smooth"},
        crop={"mode": "breast", "margin_mm": margin_mm},
        domain={"fft_friendly": False},
        heterogeneity={"use_pval": True, "noise_pct": 0.0},
        name=f"uwcem-{phantom_id}-dataset",
    )


def dataset_filename(phantom_id: str, dx_mm: float = DATASET_DX_MM) -> str:
    """``uwcem-<id>-dx<tag>mm.npz``. Only dx is encoded; other recipe
    parameters (f0, margins) are guarded by the manifest instead — a build
    into a directory whose manifest disagrees on ANY of them is refused, so
    two parameter variants can never silently share a directory."""
    tag = f"{dx_mm:g}".replace(".", "p")
    return f"uwcem-{phantom_id}-dx{tag}mm.npz"


# --------------------------------------------------------------- geometry


def _mask_bbox(mask: np.ndarray) -> list[slice]:
    """Tight bounding slices of a boolean mask (raises when empty)."""
    out = []
    for axis in range(mask.ndim):
        others = tuple(i for i in range(mask.ndim) if i != axis)
        present = np.flatnonzero(mask.any(axis=others))
        if present.size == 0:
            raise DatasetError("volume holds no tissue at all")
        out.append(slice(int(present[0]), int(present[-1]) + 1))
    return out


def _nonwater_bbox(volume: np.ndarray) -> tuple[slice, slice, slice]:
    """Bounding box of everything that is not the immersion medium.

    ``volume != 0`` instead of ``np.isin`` over the nine tissue classes:
    identical result (0 IS the immersion class code / coupling id, asserted
    where it matters), and measured 0.2 s vs 11.7 s at the 236 Mvox
    common-grid size (review finding, 2026-08-19).
    """
    return tuple(_mask_bbox(volume != 0))  # type: ignore[return-value]


def _breast_transverse_bbox(volume: np.ndarray) -> tuple[slice, slice]:
    """x/y bounding box of the PROTRUDING breast, chest-wall slab excluded.

    The subcutaneous-fat slab near the chest wall spans every transverse
    slice of the crop window, so the plain non-water bbox in x/y is always
    the window itself and "centre the tissue bbox" would centre the window,
    not the breast (review finding, 2026-08-19). This mirrors ``_crop``'s
    'breast' rule in a form that is STABLE UNDER WATER PADDING: a slice is
    chest wall when its non-water count exceeds :data:`BREAST_COVERAGE` of
    the volume-wide transverse non-water bbox area. Padding with water
    changes neither the counts nor that bbox, so the build (pre-padding) and
    the verifier (post-padding) select the same slices — which is what lets
    verification re-measure the quantity the build actually aligned.

    Falls back to the full non-water bbox when nothing protrudes.
    """
    mask = volume != 0
    box = _mask_bbox(mask)
    area = (box[0].stop - box[0].start) * (box[1].stop - box[1].start)
    counts = mask.sum(axis=(0, 1))
    protruding = (counts > 0) & (counts <= BREAST_COVERAGE * area)
    if protruding.any():
        sub = _mask_bbox(mask[:, :, protruding])
        return sub[0], sub[1]
    return box[0], box[1]


def _bbox_center(sl: slice) -> float:
    return (sl.start + sl.stop - 1) / 2


# ------------------------------------------------------------------- survey


@dataclass(frozen=True)
class SurveyRow:
    """One phantom's space requirement, measured at 0.5 mm, scaled to ``dx``."""

    phantom_id: str
    acr_class: int
    native_cropped_shape: tuple[int, int, int]
    scaled_shape: tuple[int, int, int]
    #: Voxels needed from the transverse BREAST centre to the farthest volume
    #: edge (i.e. the half-width the common box must offer), per axis.
    half_x: float
    half_y: float
    #: Voxels from the tissue front face to the back of the built volume.
    back_z: float
    #: The builder's fitted peak-RAM estimate for this build [bytes].
    peak_bytes: int


@dataclass(frozen=True)
class DatasetPlan:
    """The common grid every phantom will be padded into."""

    dx_mm: float
    f0_mhz: float
    margin_mm: float
    front_gap_mm: float
    front_gap_vox: int
    common_shape: tuple[int, int, int]
    rows: tuple[SurveyRow, ...]
    #: The requested depth ceiling [mm], or None when the union box decided.
    depth_limit_mm: float | None = None
    #: What the depth axis WOULD have been without the ceiling — so the plan
    #: can say how much is being cut before anything is built.
    uncapped_nz: int = 0

    @property
    def dx(self) -> float:
        return self.dx_mm * 1e-3

    @property
    def depth_capped(self) -> bool:
        """True when the ceiling actually shortened the axis."""
        return bool(self.uncapped_nz and self.common_shape[2] < self.uncapped_nz)

    @property
    def n_voxels(self) -> int:
        return self.common_shape[0] * self.common_shape[1] * self.common_shape[2]

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        return tuple(round(n * self.dx_mm, 2) for n in self.common_shape)  # type: ignore[return-value]

    @property
    def uncompressed_bytes_per_file(self) -> int:
        """labels int32 + four float32 property volumes, before compression."""
        return (4 + 4 * 4) * self.n_voxels

    @property
    def worst_peak_bytes(self) -> int:
        """The largest per-phantom build peak (builder's fitted model)."""
        return max((r.peak_bytes for r in self.rows), default=0)

    def clipped_ids(self) -> list[tuple[str, float]]:
        """``(id, mm cut off the back)`` for every phantom the ceiling truncates.

        Surveyed, so the caller can see the cost BEFORE spending an hour on
        the build. The depths come from the same native-resolution
        measurement the common box does, so they are accurate to about a
        voxel, not exact; the build re-measures and records the real counts.
        """
        if not self.depth_capped:
            return []
        out = []
        for r in self.rows:
            over = self.front_gap_vox + r.back_z - self.common_shape[2]
            if over > 0:
                out.append((r.phantom_id, round(over * self.dx_mm, 2)))
        return out

    def summary(self) -> str:
        sh = self.common_shape
        text = (
            f"common grid {sh[0]}x{sh[1]}x{sh[2]} @ {self.dx_mm:g} mm "
            f"= {self.extent_mm[0]:g}x{self.extent_mm[1]:g}x{self.extent_mm[2]:g} mm, "
            f"{self.n_voxels / 1e6:.1f} Mvox, "
            f"~{self.uncompressed_bytes_per_file / 1e9:.2f} GB/file uncompressed, "
            f"worst build peak ~{self.worst_peak_bytes / 1e9:.1f} GB RAM "
            f"({len(self.rows)} phantoms)"
        )
        if self.depth_capped:
            clipped = self.clipped_ids()
            text += (
                f"\ndepth limited to {self.depth_limit_mm:g} mm "
                f"({self.common_shape[2]} vox, uncapped would be {self.uncapped_nz} "
                f"= {self.uncapped_nz * self.dx_mm:g} mm)"
            )
            if clipped:
                worst = max(mm for _, mm in clipped)
                text += (
                    f"; {len(clipped)}/{len(self.rows)} phantom(s) lose tissue off the "
                    f"back, up to ~{worst:g} mm: "
                    + ", ".join(f"{pid} ~{mm:g} mm" for pid, mm in clipped)
                )
        return text


def plan_dataset(
    ids: Iterable[str] | None = None,
    dx_mm: float = DATASET_DX_MM,
    f0_mhz: float = DATASET_F0_MHZ,
    margin_mm: float = MARGIN_MM,
    front_gap_mm: float = FRONT_GAP_MM,
    depth_limit_mm: float | None = DEPTH_LIMIT_MM,
) -> DatasetPlan:
    """Measure every phantom and decide the common box. Cheap (cached mtype).

    The scaling below mirrors the builder's own arithmetic on purpose:
    ``scaled_shape`` comes from the same :func:`_project_shape` the build
    uses, so the volume sizes are EXACT; only the position of the tissue
    inside them is an estimate (voxel-centre mapping of the native bounding
    box), which is what :data:`SAFETY_VOX` is for.

    ``depth_limit_mm`` caps the propagation axis. The cap is a ceiling, so it
    rounds DOWN to a transform-friendly size (``prev_fft_friendly``) — rounding
    up the way the other two axes do would hand back a domain deeper than the
    millimetres the caller asked for.
    """
    ids = list(ids) if ids is not None else list(PHANTOM_IDS)
    if not ids:
        raise DatasetError("no phantom ids given")
    unknown = [i for i in ids if i not in PHANTOM_IDS]
    if unknown:
        raise DatasetError(f"unknown phantom ids {unknown}; pick from {list(PHANTOM_IDS)}")

    dx = dx_mm * 1e-3
    ratio = NATIVE_DX / dx
    front_gap_vox = int(round(front_gap_mm * 1e-3 / dx))
    cap_vox = None
    if depth_limit_mm is not None:
        cap_vox = processing.prev_fft_friendly(int(round(depth_limit_mm * 1e-3 / dx)))
        if cap_vox <= front_gap_vox:
            raise DatasetError(
                f"depth limit {depth_limit_mm:g} mm is "
                f"{int(round(depth_limit_mm * 1e-3 / dx))} voxel(s) at dx={dx_mm:g} mm, "
                f"which does not even clear the {front_gap_mm:g} mm front gap "
                f"({front_gap_vox} voxels)"
            )

    rows: list[SurveyRow] = []
    for pid in ids:
        spec = dataset_spec(pid, dx_mm=dx_mm, f0_mhz=f0_mhz, margin_mm=margin_mm)
        raw = load_raw(pid, with_pval=False)
        codes = to_canonical(raw.codes, copy=False)
        cropped, _, _ = _crop(codes, None, spec)
        scaled = _project_shape(cropped.shape, dx, spec)

        z_box = _nonwater_bbox(cropped)[2]
        # The transverse centre must be measured on the slab that will SURVIVE
        # the depth ceiling: the breast is widest near its base, so including
        # slices the crop is about to discard would size and centre the common
        # box around tissue no file ends up holding.
        keep_native = cropped.shape[2]
        if cap_vox is not None:
            keep_native = min(
                keep_native, z_box.start + int(np.ceil((cap_vox - front_gap_vox) / ratio))
            )
        bx, by = _breast_transverse_bbox(cropped[:, :, :keep_native])
        # Voxel-centre coordinates in the TARGET grid: native centre c maps to
        # (c + 0.5) * ratio - 0.5; a native front FACE at index z0 maps to
        # z0 * ratio.
        cx = (_bbox_center(bx) + 0.5) * ratio - 0.5
        cy = (_bbox_center(by) + 0.5) * ratio - 0.5
        z0 = z_box.start * ratio
        rows.append(
            SurveyRow(
                phantom_id=pid,
                acr_class=raw.acr_class,
                native_cropped_shape=tuple(cropped.shape),  # type: ignore[arg-type]
                scaled_shape=scaled,
                half_x=max(cx + 0.5, scaled[0] - cx - 0.5),
                half_y=max(cy + 0.5, scaled[1] - cy - 0.5),
                back_z=scaled[2] - z0,
                peak_bytes=builder_plan(spec, raw).peak_bytes,
            )
        )

    nx = int(np.ceil(2 * max(r.half_x for r in rows))) + 2 * SAFETY_VOX
    ny = int(np.ceil(2 * max(r.half_y for r in rows))) + 2 * SAFETY_VOX
    nz = front_gap_vox + int(np.ceil(max(r.back_z for r in rows))) + SAFETY_VOX
    cx, cy, cz = (processing.next_fft_friendly(n) for n in (nx, ny, nz))
    uncapped_nz = cz
    if cap_vox is not None:
        cz = min(cz, cap_vox)
    return DatasetPlan(
        dx_mm=dx_mm,
        f0_mhz=f0_mhz,
        margin_mm=margin_mm,
        front_gap_mm=front_gap_mm,
        front_gap_vox=front_gap_vox,
        common_shape=(cx, cy, cz),
        rows=tuple(rows),
        depth_limit_mm=depth_limit_mm,
        uncapped_nz=uncapped_nz,
    )


# ------------------------------------------------------------------ aligning


def _coupling_fill(table: TissueTable) -> dict[str, float]:
    """The property values of the coupling medium — what padding must carry."""
    mid = table.lookup("mid")
    cid = table.coupling_id
    return {name: float(mid[name][cid]) for name in PROPERTY_NAMES}


def _align_into_common(
    asset: PhantomAsset, plan: DatasetPlan, table: TissueTable
) -> tuple[PhantomAsset, dict]:
    """Pad ``asset`` into the common box, trimming both z faces as needed.

    z front: the first non-water voxel lands exactly at ``front_gap_vox``. The
    crop margin already put ~that much water in front, so this trims or pads a
    voxel or two of pure water; trimming anything that is NOT water raises.
    z back: whatever no longer fits under the plan's depth ceiling is cut. That
    cut is allowed to take tissue — that is what a depth limit MEANS — but only
    when the plan actually set one; without a ceiling an overflowing z is the
    survey having under-measured, and raises like x and y do. Either way the
    removed voxels are counted per class and returned, so the manifest can
    record exactly what left the domain.
    x/y: the PROTRUDING BREAST's bounding-box centre lands on the box centre,
    to the nearest voxel (see :func:`_breast_transverse_bbox` for why the
    plain tissue bbox would centre the wrong thing).
    """
    labels = asset.labels
    props = asset.properties
    if props is None:
        raise DatasetError("dataset builds are continuous (pval); properties are missing")
    cid = table.coupling_id
    if cid != 0:
        # Padding fills labels with `cid`, and the != 0 bboxes above assume it.
        raise DatasetError(f"expected coupling id 0 in the {TISSUE_MODEL!r} model, got {cid}")

    nx, ny, nz = labels.shape
    cxc, cyc, czc = plan.common_shape
    z0 = _nonwater_bbox(labels)[2].start

    # --- z: put the front face at front_gap_vox ---------------------------
    trim = max(0, z0 - plan.front_gap_vox)
    if trim:
        front = labels[:, :, :trim]
        if (front != cid).any():
            raise DatasetError(
                f"{asset.meta.get('phantom_id')}: front-face alignment would trim "
                f"{int((front != cid).sum())} non-water voxels; refusing"
            )
    z_before = max(0, plan.front_gap_vox - z0)
    nz_eff = nz - trim
    z_after = czc - nz_eff - z_before

    # --- z: cut whatever overflows the depth ceiling off the back ---------
    back_trim = max(0, -z_after)
    z_after = max(0, z_after)
    if back_trim:
        if not plan.depth_capped:
            raise DatasetError(
                f"{asset.meta.get('phantom_id')}: overflows the common box on z by "
                f"{back_trim} voxel(s) and no depth limit was set; the survey "
                f"under-measured — raise SAFETY_VOX and rebuild"
            )
        if back_trim >= nz_eff:
            raise DatasetError(
                f"{asset.meta.get('phantom_id')}: a depth limit of "
                f"{plan.depth_limit_mm:g} mm would remove the whole volume"
            )
        nz_eff -= back_trim
    cut_by_class = np.bincount(
        labels[:, :, nz - back_trim :].reshape(-1).astype(np.intp, copy=False),
        minlength=N_CLASSES,
    ).tolist()
    cut_tissue = int(sum(cut_by_class[1:]))
    keep = slice(trim, nz - back_trim)

    # --- x/y: centre the protruding breast --------------------------------
    # measured on the KEPT slab, not the built volume: a depth limit can cut
    # away the widest protruding slices, and aligning on slices that will not
    # be stored would leave the file off-centre by the verifier's own measure.
    bx, by = _breast_transverse_bbox(labels[:, :, keep])
    cx, cy = _bbox_center(bx), _bbox_center(by)
    x_before = int(round((cxc - 1) / 2 - cx))
    y_before = int(round((cyc - 1) / 2 - cy))
    x_after = cxc - nx - x_before
    y_after = cyc - ny - y_before

    for name, before, after in (
        ("x", x_before, x_after),
        ("y", y_before, y_after),
        ("z", z_before, z_after),
    ):
        if before < 0 or after < 0:
            raise DatasetError(
                f"{asset.meta.get('phantom_id')}: does not fit the common box on {name} "
                f"(pad before={before}, after={after}); the survey under-measured — "
                f"raise SAFETY_VOX and rebuild"
            )

    sl = (
        slice(x_before, x_before + nx),
        slice(y_before, y_before + ny),
        slice(z_before, z_before + nz_eff),
    )
    new_labels = np.full(plan.common_shape, cid, dtype=np.int32)
    new_labels[sl] = labels[:, :, keep]

    fill = _coupling_fill(table)
    # One property at a time: the padded copy replaces the source before the
    # next allocation, so peak scratch is ONE common-shape volume, not four.
    for name in PROPERTY_NAMES:
        src = getattr(props, name)
        out = np.full(plan.common_shape, np.float32(fill[name]), dtype=np.float32)
        out[sl] = src[:, :, keep]
        setattr(props, name, out)
        del src

    new_box = _nonwater_bbox(new_labels)
    nbx, nby = _breast_transverse_bbox(new_labels)
    alignment = {
        "front_gap_vox": plan.front_gap_vox,
        "front_face_vox": int(new_box[2].start),
        "tissue_bbox_vox": [[int(s.start), int(s.stop)] for s in new_box],
        "breast_bbox_vox": [[int(nbx.start), int(nbx.stop)], [int(nby.start), int(nby.stop)]],
        "breast_center_vox": [_bbox_center(nbx), _bbox_center(nby)],
        "box_center_vox": [(cxc - 1) / 2, (cyc - 1) / 2],
        "pad_before_vox": [x_before, y_before, z_before],
        "front_trim_vox": trim,
        "back_trim_vox": back_trim,
        "back_trim_mm": round(back_trim * plan.dx_mm, 4),
        "truncated_tissue_vox": cut_tissue,
        "truncated_by_class": cut_by_class,
        # tissue reaching the last plane means the domain now ENDS inside the
        # body: the back boundary (and any PML on it) sits in tissue, not water
        "tissue_at_back_face": bool(int(new_box[2].stop) == czc),
    }

    meta = dict(asset.meta)
    meta["dataset"] = {
        "format": DATASET_FORMAT,
        "dx_mm": plan.dx_mm,
        "f0_mhz": plan.f0_mhz,
        "margin_mm": plan.margin_mm,
        "front_gap_mm": plan.front_gap_mm,
        "depth_limit_mm": plan.depth_limit_mm,
        "common_shape": list(plan.common_shape),
        "alignment": alignment,
    }
    if cut_tissue:
        warnings = list(meta.get("warnings", []))
        warnings.append(
            f"depth limit {plan.depth_limit_mm:g} mm cut {back_trim} voxel(s) "
            f"({back_trim * plan.dx_mm:g} mm) off the back, removing "
            f"{cut_tissue:,} tissue voxels; the domain ends inside the body"
        )
        meta["warnings"] = warnings
    out_asset = PhantomAsset(
        labels=new_labels,
        dx=asset.dx,
        materials=asset.materials,
        origin=(0.0, 0.0, 0.0),
        properties=props,
        meta=meta,
    )
    return out_asset, alignment


# ------------------------------------------------------------------ checking


def _check_property_bounds(asset: PhantomAsset, table: TissueTable) -> dict[str, dict]:
    """Every voxel's properties must lie inside its own class's band.

    The pval blend is ``lo + (hi - lo) * p`` with ``p`` in [0, 1], per media
    number; classes without pval collapse to their midpoint. So per class
    code, min/max across the volume must sit inside ``[lo, hi]`` (a point
    interval for skin, muscle and the bath). Non-finite values fail
    explicitly — every band comparison is False for NaN, so without the
    isfinite gate a NaN volume would verify clean (review finding,
    2026-08-19). This is the "pvals are computed properly" check, run on
    every build and every verify.

    One chunked pass per property: per-class count/sum/sumsq via
    ``np.bincount`` and min/max via ``ufunc.at`` on ten-element accumulators.
    The per-class boolean masks this replaces cost 72 s and ~0.5 GB scratch
    per phantom at the 236 Mvox common-grid size; this form measures ~8 s
    with one chunk of scratch (review finding, 2026-08-19).
    """
    labels = asset.labels
    props = asset.properties
    if props is None:
        raise DatasetError("no property volumes to check")
    pid = asset.meta.get("phantom_id")
    lo = table.lookup_by_code("lo")
    hi = table.lookup_by_code("hi")

    flat = labels.reshape(-1)
    if flat.size == 0:
        raise DatasetError("empty label volume")
    lo_id, hi_id = int(flat.min()), int(flat.max())
    if lo_id < 0 or hi_id >= N_CLASSES:
        raise DatasetError(f"{pid}: labels span [{lo_id}, {hi_id}] outside 0..{N_CLASSES - 1}")

    counts = np.zeros(N_CLASSES, dtype=np.int64)
    sums = {n: np.zeros(N_CLASSES, dtype=np.float64) for n in PROPERTY_NAMES}
    sumsq = {n: np.zeros(N_CLASSES, dtype=np.float64) for n in PROPERTY_NAMES}
    mins = {n: np.full(N_CLASSES, np.inf, dtype=np.float32) for n in PROPERTY_NAMES}
    maxs = {n: np.full(N_CLASSES, -np.inf, dtype=np.float32) for n in PROPERTY_NAMES}
    flats = {n: getattr(props, n).reshape(-1) for n in PROPERTY_NAMES}

    chunk = 1 << 24  # 16 Mvox: big enough to amortize, small enough to fit
    for start in range(0, flat.size, chunk):
        li = flat[start : start + chunk].astype(np.intp, copy=False)
        counts += np.bincount(li, minlength=N_CLASSES)
        for name in PROPERTY_NAMES:
            v = flats[name][start : start + chunk]
            if not np.isfinite(v).all():
                raise DatasetError(f"{pid}: non-finite {name} values in the volume")
            sums[name] += np.bincount(li, weights=v, minlength=N_CLASSES)
            v64 = v.astype(np.float64)
            sumsq[name] += np.bincount(li, weights=v64 * v64, minlength=N_CLASSES)
            np.minimum.at(mins[name], li, v)
            np.maximum.at(maxs[name], li, v)

    report: dict[str, dict] = {}
    for code in np.flatnonzero(counts):
        code = int(code)
        cnt = int(counts[code])
        row: dict = {}
        for name in PROPERTY_NAMES:
            vmin, vmax = float(mins[name][code]), float(maxs[name][code])
            lo_c, hi_c = float(lo[name][code]), float(hi[name][code])
            if vmin < lo_c - BOUNDS_ATOL or vmax > hi_c + BOUNDS_ATOL:
                raise DatasetError(
                    f"{pid}: class {code} has {name} in "
                    f"[{vmin:.6g}, {vmax:.6g}] outside its band [{lo_c:.6g}, {hi_c:.6g}]"
                )
            mean = sums[name][code] / cnt
            var = max(0.0, sumsq[name][code] / cnt - mean * mean)
            row[name] = {"min": vmin, "max": vmax, "std": float(np.sqrt(var))}
        report[str(code)] = row
    return report


# ------------------------------------------------------------------ building


def _available_ram_bytes() -> int | None:
    """Physical RAM currently free, or None when the platform cannot say."""
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32)] + [
                (n, ctypes.c_uint64)
                for n in (
                    "ullTotalPhys",
                    "ullAvailPhys",
                    "ullTotalPageFile",
                    "ullAvailPageFile",
                    "ullTotalVirtual",
                    "ullAvailVirtual",
                    "ullAvailExtendedVirtual",
                )
            ]

        stat = MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return int(stat.ullAvailPhys)
    except (AttributeError, OSError):
        pass
    try:
        import os

        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None


#: Manifest keys that define the recipe; a directory whose manifest disagrees
#: on any of them holds a DIFFERENT dataset and is refused, not merged.
#: ``depth_limit_mm`` is nullable, so it compares by identity-or-value rather
#: than numerically — a missing key (a /1 manifest) must read as "different",
#: not as "no limit requested".
_RECIPE_KEYS = (
    "dx_mm",
    "f0_mhz",
    "tissue_model",
    "margin_mm",
    "front_gap_mm",
    "depth_limit_mm",
)


def _reconcile_with_existing(
    plan: DatasetPlan, out: Path, force: bool
) -> tuple[DatasetPlan, dict | None]:
    """Merge policy for a directory that already holds a dataset.

    Same recipe -> adopt the ESTABLISHED grid (so a one-phantom rebuild lands
    on the grid the other eight already use) and keep their manifest entries.
    Different recipe -> refuse unless ``force``, because dataset_filename
    encodes only dx and a silent overwrite would leave, say, a 3 MHz alpha
    volume under a 1 MHz manifest (review finding, 2026-08-19).
    """
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return plan, None
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise DatasetError(
            f"{manifest_path} exists but is unreadable ({exc}); delete it or use a fresh --out"
        ) from exc

    wanted = {
        "format": DATASET_FORMAT,
        "dx_mm": plan.dx_mm,
        "f0_mhz": plan.f0_mhz,
        "tissue_model": TISSUE_MODEL,
        "margin_mm": plan.margin_mm,
        "front_gap_mm": plan.front_gap_mm,
        "depth_limit_mm": plan.depth_limit_mm,
    }
    diffs = []
    for k in ("format", *_RECIPE_KEYS):
        have = existing.get(k, "<missing>")
        want = wanted[k]
        if want is None or isinstance(want, str):
            if have != want:
                diffs.append(k)
        # `have` may be missing or non-numeric; NaN tricks (nan > x is False)
        # must not let a missing key count as a match.
        elif not isinstance(have, (int, float)) or abs(float(have) - want) > 1e-9:
            diffs.append(k)
    if diffs:
        if force:
            return plan, None  # deliberate overwrite: start the directory over
        raise DatasetError(
            f"{out} already holds a dataset built with different parameters "
            f"({', '.join(f'{k}={existing.get(k)!r} vs {wanted[k]!r}' for k in diffs)}); "
            f"use a fresh --out for a variant, delete the directory, or force=True/--force "
            f"to overwrite"
        )

    established = tuple(existing["common_shape"])
    if any(e < c for e, c in zip(established, plan.common_shape, strict=True)):
        raise DatasetError(
            f"the established grid {established} cannot hold this build "
            f"(needs {plan.common_shape}); rebuild every id into a fresh --out"
        )
    return replace(plan, common_shape=established), existing  # type: ignore[arg-type]


def build_dataset(
    ids: Iterable[str] | None = None,
    dx_mm: float = DATASET_DX_MM,
    f0_mhz: float = DATASET_F0_MHZ,
    margin_mm: float = MARGIN_MM,
    front_gap_mm: float = FRONT_GAP_MM,
    depth_limit_mm: float | None = DEPTH_LIMIT_MM,
    out_dir: str | Path | None = None,
    progress: ProgressFn | None = None,
    plan: DatasetPlan | None = None,
    force: bool = False,
) -> dict:
    """Build the aligned dataset; returns (and writes) the manifest dict.

    Phantoms build one at a time — a single 0.25 mm build peaks at 6-15 GB
    of scratch by the builder's fitted model, and nine in flight would be
    nine times that for zero wall clock (the pipeline is memory-bound, not
    CPU-bound). The worst peak is checked against the RAM actually free
    BEFORE any build starts; ``force=True`` skips that rail (and lets a
    different-recipe directory be overwritten, see
    :func:`_reconcile_with_existing`).
    """
    say = progress or (lambda _msg: None)
    out = Path(out_dir) if out_dir is not None else dataset_dir()
    ensure_dir(out)

    if plan is None:
        say("surveying phantoms for the common box...")
        plan = plan_dataset(
            ids,
            dx_mm=dx_mm,
            f0_mhz=f0_mhz,
            margin_mm=margin_mm,
            front_gap_mm=front_gap_mm,
            depth_limit_mm=depth_limit_mm,
        )
    plan, existing = _reconcile_with_existing(plan, out, force)
    say(plan.summary())
    if existing is not None and len(plan.rows) < len(existing.get("phantoms", [])):
        say(f"subset rebuild: adopting the established grid {tuple(existing['common_shape'])}")

    avail = _available_ram_bytes()
    if avail is not None and plan.worst_peak_bytes > avail and not force:
        raise DatasetError(
            f"the largest build peaks at ~{plan.worst_peak_bytes / 1e9:.1f} GB but only "
            f"~{avail / 1e9:.1f} GB RAM is free; close other work, build a subset, or pass "
            f"force=True / --force to try anyway"
        )

    table = tissue_table(TISSUE_MODEL, f0=plan.f0_mhz * 1e6)
    entries: list[dict] = []
    for i, row in enumerate(plan.rows, 1):
        pid = row.phantom_id
        t0 = time.perf_counter()
        say(f"[{i}/{len(plan.rows)}] {pid}: building @ {plan.dx_mm:g} mm ...")
        spec = dataset_spec(pid, dx_mm=plan.dx_mm, f0_mhz=plan.f0_mhz, margin_mm=plan.margin_mm)
        asset = build(spec)
        say(f"[{i}/{len(plan.rows)}] {pid}: built {asset.shape}, aligning into {plan.common_shape}")
        asset, alignment = _align_into_common(asset, plan, table)
        bounds = _check_property_bounds(asset, table)

        path = out / dataset_filename(pid, plan.dx_mm)
        say(f"[{i}/{len(plan.rows)}] {pid}: writing {path.name} ...")
        asset.save(path)
        seconds = time.perf_counter() - t0
        entries.append(
            {
                "id": pid,
                "acr_class": row.acr_class,
                "file": path.name,
                "bytes": path.stat().st_size,
                "built_shape": list(row.scaled_shape),
                "alignment": alignment,
                "warnings": asset.meta.get("warnings", []),
                "property_stats_by_class": bounds,
                "build_seconds": round(seconds, 1),
            }
        )
        say(
            f"[{i}/{len(plan.rows)}] {pid}: done in {seconds:.0f} s "
            f"({path.stat().st_size / 1e6:.0f} MB)"
        )
        if alignment["truncated_tissue_vox"]:
            say(
                f"[{i}/{len(plan.rows)}] {pid}: WARNING depth limit removed "
                f"{alignment['back_trim_mm']:g} mm off the back "
                f"({alignment['truncated_tissue_vox']:,} tissue voxels)"
            )
        del asset

    # Merge, never clobber: a subset rebuild keeps the other phantoms' entries
    # (their files are untouched on disk and still on the same grid).
    by_id = {e["id"]: e for e in (existing.get("phantoms", []) if existing else [])}
    for e in entries:
        by_id[e["id"]] = e
    ordered = [by_id[pid] for pid in PHANTOM_IDS if pid in by_id]
    ordered += [e for i, e in by_id.items() if i not in PHANTOM_IDS]

    manifest = {
        "format": DATASET_FORMAT,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dx_mm": plan.dx_mm,
        "f0_mhz": plan.f0_mhz,
        "tissue_model": TISSUE_MODEL,
        "margin_mm": plan.margin_mm,
        "front_gap_mm": plan.front_gap_mm,
        "front_gap_vox": plan.front_gap_vox,
        "depth_limit_mm": plan.depth_limit_mm,
        "common_shape": list(plan.common_shape),
        "extent_mm": list(plan.extent_mm),
        "alignment": {
            "z": "first non-water voxel at index front_gap_vox on every phantom",
            "xy": "protruding-breast bbox centre at the box centre (nearest voxel)",
            "depth": (
                f"z axis capped at {plan.depth_limit_mm:g} mm; anything deeper is cut "
                f"off the back and counted in each phantom's truncated_* fields"
                if plan.depth_limit_mm is not None
                else "uncapped: the union of every phantom's depth"
            ),
        },
        "truncation": {
            "phantoms_clipped": sorted(
                e["id"] for e in ordered if e["alignment"].get("truncated_tissue_vox")
            ),
            "worst_back_trim_mm": max(
                (e["alignment"].get("back_trim_mm", 0.0) for e in ordered), default=0.0
            ),
        },
        "use_pval": True,
        "noise_pct": 0.0,
        "citation": CITATION,
        "phantoms": ordered,
    }
    manifest_path = out / "manifest.json"
    tmp = manifest_path.with_suffix(".json.part")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)
    say(f"manifest written: {manifest_path}")
    clipped = manifest["truncation"]["phantoms_clipped"]
    if clipped:
        say(
            f"depth limit {plan.depth_limit_mm:g} mm clipped {len(clipped)}/{len(ordered)} "
            f"phantom(s) at the back: {', '.join(clipped)} "
            f"(worst {manifest['truncation']['worst_back_trim_mm']:g} mm)"
        )
    return manifest


# ----------------------------------------------------------------- verifying


def verify_dataset(out_dir: str | Path | None = None, progress: ProgressFn | None = None) -> dict:
    """Re-check an existing dataset from disk; raises :class:`DatasetError`.

    Checks: the manifest's and every file's dataset format tag, the common
    shape and dx, the stored property volumes, finiteness, the front-face
    plane, the breast centring (within :data:`CENTER_TOL_VOX` of the box
    centre), the water padding's property values, the per-class property
    bounds (the pval contract), the depth ceiling and its truncation record —
    and that the directory holds NO dataset files the manifest does not list,
    which is exactly what a stale or interrupted rebuild leaves behind.
    Returns a report dict.

    The truncation record cannot be re-derived from the file alone (the tissue
    that left is gone), but it IS falsifiable in one direction: a phantom the
    manifest says lost tissue must have tissue on its last z-plane, because the
    body is contiguous along z. A manifest claiming a clip on a file whose back
    face is clean water is therefore caught here.
    """
    say = progress or (lambda _msg: None)
    out = Path(out_dir) if out_dir is not None else dataset_dir()
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        raise DatasetError(f"no manifest at {manifest_path}; build the dataset first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fmt = manifest.get("format")
    if fmt not in ACCEPTED_DATASET_FORMATS:
        raise DatasetError(f"{manifest_path}: format {fmt!r} != {DATASET_FORMAT!r}")

    listed = {e["file"] for e in manifest["phantoms"]}
    orphans = sorted(p.name for p in out.glob("uwcem-*.npz") if p.name not in listed)
    if orphans:
        raise DatasetError(
            f"{out} holds dataset files the manifest does not list: {orphans}; "
            f"they are from an older or interrupted build — delete them or rebuild"
        )

    common = tuple(manifest["common_shape"])
    dx_mm = float(manifest["dx_mm"])
    front_gap_vox = int(manifest["front_gap_vox"])
    depth_limit_mm = manifest.get("depth_limit_mm")
    if depth_limit_mm is not None:
        # The stored grid must actually honour the limit it advertises.
        if common[2] * dx_mm > depth_limit_mm + 1e-9:
            raise DatasetError(
                f"{manifest_path}: depth limit {depth_limit_mm:g} mm but the grid is "
                f"{common[2]} x {dx_mm:g} mm = {common[2] * dx_mm:g} mm deep"
            )
    table = tissue_table(manifest["tissue_model"], f0=float(manifest["f0_mhz"]) * 1e6)
    fill = _coupling_fill(table)

    report: dict = {
        "files": {},
        "common_shape": list(common),
        "dx_mm": dx_mm,
        "depth_limit_mm": depth_limit_mm,
    }
    for entry in manifest["phantoms"]:
        path = out / entry["file"]
        say(f"verifying {path.name} ...")
        if not path.exists():
            raise DatasetError(f"{path} is listed in the manifest but missing on disk")
        asset = PhantomAsset.load(path)

        dfmt = asset.meta.get("dataset", {}).get("format")
        if dfmt not in ACCEPTED_DATASET_FORMATS:
            raise DatasetError(f"{path.name}: dataset format {dfmt!r} != {DATASET_FORMAT!r}")
        if asset.shape != common:
            raise DatasetError(f"{path.name}: shape {asset.shape} != common {common}")
        if abs(asset.dx * 1e3 - dx_mm) > 1e-9:
            raise DatasetError(f"{path.name}: dx {asset.dx * 1e3} mm != {dx_mm} mm")
        if asset.properties is None:
            raise DatasetError(f"{path.name}: property volumes are missing")

        z_box = _nonwater_bbox(asset.labels)[2]
        z0 = z_box.start
        if z0 != front_gap_vox:
            raise DatasetError(
                f"{path.name}: tissue front face at z={z0}, expected {front_gap_vox}"
            )
        at_back = int(z_box.stop) == common[2]
        clipped = int(entry.get("alignment", {}).get("truncated_tissue_vox", 0))
        if clipped and not at_back:
            raise DatasetError(
                f"{path.name}: the manifest records {clipped:,} truncated tissue voxels, "
                f"but tissue stops at z={z_box.stop} of {common[2]} — the file does not "
                f"match its own truncation record"
            )
        bx, by = _breast_transverse_bbox(asset.labels)
        centre = (_bbox_center(bx), _bbox_center(by))
        for axis in (0, 1):
            target = (common[axis] - 1) / 2
            if abs(centre[axis] - target) > CENTER_TOL_VOX:
                raise DatasetError(
                    f"{path.name}: breast centre {centre[axis]} on axis {axis} is "
                    f"{abs(centre[axis] - target):.2f} voxels off the box centre {target}"
                )

        # The padded water must carry the coupling medium's own values — a
        # zero-padded property volume would be a vacuum, and the solver would
        # happily refract through it.
        corner = (slice(0, 2), slice(0, 2), slice(0, 2))
        if (asset.labels[corner] != table.coupling_id).any():
            raise DatasetError(f"{path.name}: the domain corner is not coupling medium")
        for name in PROPERTY_NAMES:
            v = float(getattr(asset.properties, name)[corner].mean())
            if not np.isfinite(v) or abs(v - fill[name]) > BOUNDS_ATOL:
                raise DatasetError(
                    f"{path.name}: padded water has {name}={v:.6g}, expected {fill[name]:.6g}"
                )

        bounds = _check_property_bounds(asset, table)

        # pval must have actually varied something: a within-class std of 0 on
        # every pval class means the blend silently fell back to midpoints.
        pval_codes = [str(c) for c in sorted(table.pval_ids)]
        if not any(bounds[c]["c"]["std"] > 0 for c in pval_codes if c in bounds):
            raise DatasetError(f"{path.name}: no within-class variation — pval was not applied")

        report["files"][path.name] = {
            "id": entry["id"],
            "shape": list(asset.shape),
            "front_face_vox": int(z0),
            "back_face_vox": int(z_box.stop),
            "tissue_at_back_face": at_back,
            "truncated_tissue_vox": clipped,
            "breast_center_vox": list(centre),
            "property_stats_by_class": bounds,
            "ok": True,
        }
        del asset
    say(f"{len(report['files'])} file(s) verified against {manifest_path.name}")
    return report
