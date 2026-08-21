"""The pipeline: :class:`PhantomSpec` in, :class:`PhantomAsset` out.

One function, one fixed order, every step logged. The order is argued for in
:mod:`uwcem_phantoms.spec`; the logging exists because a phantom build
quietly changes physics — a crop that clipped the chest wall or a resample
that halved the skin thickness produces a medium that still *runs*, and still
gives wrong answers. Every stage therefore appends a line to
``asset.meta["log"]`` saying what it did and what it cost, and the skin/tissue
sanity checks below turn the most damaging of those into explicit warnings
that travel inside the export file.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from uwcem_phantoms import processing
from uwcem_phantoms.asset import PhantomAsset
from uwcem_phantoms.catalog import CITATION, NATIVE_DX
from uwcem_phantoms.heterogeneity import (
    PROPERTY_NAMES,
    PropertyVolumes,
    apply_pval_interpolation,
    apply_scatterer_noise,
)
from uwcem_phantoms.orientation import CANONICAL_AXES, to_canonical
from uwcem_phantoms.reader import N_CLASSES, RawPhantom, load_raw
from uwcem_phantoms.spec import PhantomSpec
from uwcem_phantoms.tissue import (
    PVAL_CODES,
    AcousticTissue,
    TissueTable,
    tissue_table,
)

#: Class code of the immersion medium in the raw data (media number -1).
IMMERSION_CODE = 0
SKIN_CODE = 1
MUSCLE_CODE = 2
#: Every code that is breast tissue proper (glandular .. fatty).
TISSUE_CODES = tuple(range(3, N_CLASSES))

ProgressFn = Callable[[str, float], None]


class BuildWarning(str):
    """A note that the build changed the physics in a way worth reading."""


@dataclass(frozen=True)
class BuildPlan:
    """What a spec would cost, computed without doing the expensive parts.

    The crop box is measured for real (a couple of reductions on the native
    volume); resampling, property expansion and noise — the parts that
    dominate both time and memory — are only ARITHMETIC here. That is enough
    for an interactive UI to pick a resolution it can actually afford, and
    for a script to refuse a build before it allocates 4 GB.

    ``exact`` says which promise this plan is making. With no label
    simplification requested it is an EQUALITY: ``plan(spec).shape ==
    build(spec).shape``. With simplification on it is an UPPER BOUND —
    measured, not hand-waved: dropping skin lets the outermost tissue
    dissolve into the bath and shrink the box (90 -> 84 voxels on phantom
    012304), while skin closing and majority smoothing can grow it by at most
    one voxel per pass, which is added here. An upper bound is the safe
    direction for a memory rail; an equality claim that is quietly false is
    not.
    """

    cropped_shape: tuple[int, int, int]
    dx: float
    shape: tuple[int, int, int]
    n_voxels: int
    label_bytes: int
    property_bytes: int
    exact: bool = True
    resamples: bool = False
    source_voxels: int = 0
    decodes_pval: bool = False

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        return tuple(round(n * self.dx * 1e3, 2) for n in self.shape)  # type: ignore[return-value]

    @property
    def peak_bytes(self) -> int:
        """Peak host memory of the whole build, not just of what it returns.

        Two stages compete for the peak and the estimate takes the larger:

        * **resample** — ``LabelVolume.resample("smooth")`` holds a float32
          one-hot volume per class, its separable-interpolation temporaries at
          the intermediate shapes, and ``out``/``best_score`` at the output
          shape. It dominates whenever ``dx`` differs from the native 0.5 mm,
          which is the normal case.
        * **everything else** — labels, the int16 code volume, the ``intp``
          index copy the lookups make, and (continuous mode) four property
          volumes plus the pval blend's scratch.

        A third stage joins them the first time a phantom's ``pval`` is
        decoded: that cost scales with the SOURCE phantom, not the output
        grid, so at a coarse ``dx`` it is the peak all by itself (a 0.7 mm
        build measured 733 MB against a 363 MB grid-based estimate). It is
        counted whenever the spec asks for pval — the safe direction, since a
        second build reusing the same ``RawPhantom`` does not pay it again.

        The per-voxel coefficients are FITTED to measured ``tracemalloc``
        peaks, not guessed: an earlier analytic-looking model that ignored the
        resample under-reported by 4.4x at dx = 0.35 mm (267 MB planned,
        1180 MB actual), and this number is what the GUI shows a user sizing
        a machine (review finding, 2026-08-18). The 1.25 factor keeps it on
        the safe side of every case measured.
        """
        n = self.n_voxels
        src = self.cropped_shape[0] * self.cropped_shape[1] * self.cropped_shape[2]
        # labels(4) + codes int16(2) + one intp index copy(8)
        per_voxel = 14
        if self.property_bytes:
            # 4 property volumes(16) + the pval blend's lo/hi/temporary
            # scratch(24) + the float64 draw inside correlated_noise before
            # its float32 cast(8). Measured, not derived: 36 was still 10%
            # short of the real continuous peak.
            per_voxel += 48
        stage_other = per_voxel * n
        stage_resample = 4 * (4 * src + 8 * n) if self.resamples else 0
        # 26 B per NATIVE voxel: the zip's decompressed bytes, the decoder's
        # accumulator and live digit temporary, the retained float32 result
        # and its reorientation copy. Measured across two phantoms.
        stage_decode = 26 * self.source_voxels if self.decodes_pval else 0
        return int(1.25 * max(stage_other, stage_resample, stage_decode))

    def summary(self) -> str:
        bound = "" if self.exact else " (upper bound: simplification may shrink it)"
        return (
            f"{self.shape[0]}x{self.shape[1]}x{self.shape[2]} @ {self.dx * 1e3:g} mm "
            f"= {self.n_voxels / 1e6:.1f} Mvox, ~{self.peak_bytes / 1e9:.2f} GB peak{bound}"
        )


def _project_shape(
    cropped_shape: tuple[int, ...], dx: float, spec: PhantomSpec
) -> tuple[int, int, int]:
    """Final grid shape a cropped volume will reach, by arithmetic only.

    Shared by :func:`plan` and :func:`build` on purpose: the plan's promise is
    that it predicts the build EXACTLY, and two copies of this arithmetic
    would drift the first time a domain option was added to one of them.
    """
    scaled = tuple(max(1, int(round(n * NATIVE_DX / dx))) for n in cropped_shape)
    d = spec.domain
    lat = int(round(d.lateral_margin_mm * 1e-3 / dx))
    grown = [
        scaled[0] + 2 * lat,
        scaled[1] + 2 * lat,
        scaled[2] + int(round(d.standoff_mm * 1e-3 / dx)) + int(round(d.backing_mm * 1e-3 / dx)),
    ]
    if d.fft_friendly:
        grown = [processing.next_fft_friendly(n) for n in grown]
    return (grown[0], grown[1], grown[2])


def plan(spec: PhantomSpec, raw: RawPhantom | None = None) -> BuildPlan:
    """Estimate the built size of ``spec`` without resampling anything.

    Deliberate approximation: label simplification is SKIPPED, because it
    changes the crop box by at most a voxel or two while costing as much as
    the build itself. Everything else — crop mode, target dx, standoff,
    lateral margin, FFT padding — is computed exactly.
    """
    if raw is None:
        raw = load_raw(spec.phantom_id, with_pval=False)
    codes = to_canonical(raw.codes, copy=False)
    cropped, _, _ = _crop(codes, None, spec)
    dx = spec.resolution.dx() or NATIVE_DX

    simp = spec.simplify
    # Morphology that can GROW the label field grows the box with it; every
    # other simplification only removes tissue, so the unsimplified box
    # already bounds it.
    grow = 2 * (simp.close_skin_iterations + simp.smooth_iterations)
    cropped_shape = tuple(
        min(n + grow, limit) for n, limit in zip(cropped.shape, codes.shape, strict=True)
    )
    shape = _project_shape(cropped_shape, dx, spec)
    n = shape[0] * shape[1] * shape[2]
    props = 4 * 4 * n if spec.heterogeneity.produces_continuous_fields else 0
    return BuildPlan(
        cropped_shape=cropped_shape,  # type: ignore[arg-type]
        dx=dx,
        shape=shape,
        n_voxels=n,
        label_bytes=4 * n,
        property_bytes=props,
        exact=not simp.touches_labels,
        resamples=abs(dx - NATIVE_DX) > 1e-12,
        source_voxels=int(raw.n_voxels),
        decodes_pval=spec.heterogeneity.use_pval,
    )


def dx_for_budget(
    spec: PhantomSpec,
    max_voxels: int,
    raw: RawPhantom | None = None,
    step: float = 1.15,
    max_iterations: int = 40,
) -> float:
    """Smallest dx >= the spec's own that keeps the build under ``max_voxels``.

    Used by the GUI's preview mode: coarsen until it fits, then say so out
    loud rather than silently rendering something the user did not ask for.
    """
    if raw is None:
        raw = load_raw(spec.phantom_id, with_pval=False)
    trial = spec.model_copy(deep=True)
    dx_mm = (spec.resolution.dx() or NATIVE_DX) * 1e3
    for _ in range(max_iterations):
        trial.resolution = trial.resolution.model_copy(update={"dx_mm": dx_mm})
        if plan(trial, raw).n_voxels <= max_voxels:
            return dx_mm * 1e-3
        dx_mm *= step
    return dx_mm * 1e-3


def build(
    spec: PhantomSpec,
    raw: RawPhantom | None = None,
    overrides: dict[int, AcousticTissue] | None = None,
    progress: ProgressFn | None = None,
) -> PhantomAsset:
    """Execute ``spec`` and return the simulation-ready asset.

    ``raw`` lets a caller (the GUI, a sweep) reuse an already-decoded phantom
    across many builds instead of paying the decode every time. When it is
    given, its id must match the spec's.
    """
    t_start = time.perf_counter()
    log: list[str] = []
    warnings: list[str] = []

    def step(msg: str, frac: float) -> None:
        log.append(msg)
        if progress is not None:
            progress(msg, frac)

    need_pval = spec.heterogeneity.use_pval
    if raw is None:
        raw = load_raw(spec.phantom_id, with_pval=need_pval)
    elif raw.phantom_id != spec.phantom_id:
        raise ValueError(f"raw phantom is {raw.phantom_id!r}, spec asks for {spec.phantom_id!r}")
    elif need_pval and not raw.has_pval:
        raw = load_raw(spec.phantom_id, with_pval=True)
    step(f"loaded {raw.phantom_id}: {raw.shape} @ 0.5 mm (ACR class {raw.acr_class})", 0.05)

    # ---- 1. orientation ---------------------------------------------------
    codes = to_canonical(raw.codes).astype(np.int16)
    pval = to_canonical(raw.pval) if (need_pval and raw.pval is not None) else None
    step(f"reoriented to canonical axes {CANONICAL_AXES} -> {codes.shape}", 0.12)

    # ---- 2. label simplification (at native dx) ---------------------------
    codes, simp_log, simp_warn = _simplify(codes, spec)
    log.extend(simp_log)
    warnings.extend(simp_warn)
    if progress is not None:
        progress("simplified", 0.3)

    # ---- 3. crop ----------------------------------------------------------
    codes, pval, offset_vox = _crop(codes, pval, spec)
    step(f"cropped to {codes.shape} (offset {offset_vox} voxels at 0.5 mm)", 0.4)

    # ---- 4. resample ------------------------------------------------------
    dx = spec.resolution.dx() or NATIVE_DX
    # The rail is checked HERE, on the projected shape, not after the fact:
    # resampling is the biggest allocation in the pipeline, so a guard that
    # fires downstream of it has already spent the memory it exists to save.
    projected = _project_shape(codes.shape, dx, spec)
    n_projected = projected[0] * projected[1] * projected[2]
    if spec.domain.max_voxels is not None and n_projected > spec.domain.max_voxels:
        raise ValueError(
            f"this spec builds a {projected[0]}x{projected[1]}x{projected[2]} grid "
            f"({n_projected:,} voxels), above the max_voxels rail of "
            f"{spec.domain.max_voxels:,}. Increase dx_mm, crop tighter, or raise the rail."
        )
    if abs(dx - NATIVE_DX) > 1e-12:
        before = codes.shape
        if pval is not None:
            # MASKED resample. p = 0 in skin/muscle/bath means "no data", not
            # "lowest value", so blending it with real p drags the tissue
            # touching those interfaces toward its class minimum — a
            # one-voxel rim around the entire breast surface, exactly where
            # the skin/fat reflection lives, and only at non-integer dx
            # ratios so it was intermittent (review finding, 2026-08-18).
            # Resample p*valid and valid separately, then divide.
            valid = np.isin(codes, np.asarray(sorted(PVAL_CODES), dtype=codes.dtype))
            weight = processing.resample_field(valid.astype(np.float32), NATIVE_DX, dx)
            pval = processing.resample_field(pval * valid, NATIVE_DX, dx)
            np.divide(pval, weight, out=pval, where=weight > 1e-6)
            pval[weight <= 1e-6] = 0.0
            np.clip(pval, 0.0, 1.0, out=pval)
            del valid, weight
        codes = processing.resample_codes(codes, NATIVE_DX, dx, spec.resolution.method)
        step(
            f"resampled {before} @ 0.5 mm -> {codes.shape} @ {dx * 1e3:g} mm "
            f"({spec.resolution.method})",
            0.55,
        )
        warnings.extend(_resolution_warnings(dx, spec))
    else:
        step("kept native 0.5 mm spacing", 0.55)

    # ---- 5. domain growth -------------------------------------------------
    codes, pval, pad_before = _grow_domain(codes, pval, spec, dx)
    step(f"domain {codes.shape} @ {dx * 1e3:g} mm ({codes.size:,} voxels)", 0.65)

    # ---- 6. labels -> material ids ---------------------------------------
    table = tissue_table(spec.simplify.tissue_model, f0=spec.f0, overrides=overrides)
    labels = processing.merge_classes(codes, table.code_to_id).astype(np.int32)
    step(
        f"mapped {len(set(table.code_to_id.values()))} material ids "
        f"({spec.simplify.tissue_model} model)",
        0.72,
    )

    # ---- 7. properties + heterogeneity ------------------------------------
    props, het_meta, het_log = _properties(labels, codes, pval, table, spec, dx)
    log.extend(het_log)
    if progress is not None:
        progress("properties built", 0.9)

    warnings.extend(_geometry_warnings(codes, dx, spec))
    warnings.extend(_resolution_vs_wavelength(table, dx, spec))

    meta = {
        "format_note": "caustica phantom export; see uwcem_phantoms.asset for the key contract",
        "phantom_id": raw.phantom_id,
        "acr_class": raw.acr_class,
        "name": spec.export_name(),
        "note": spec.note,
        "tissue_model": spec.simplify.tissue_model,
        "f0_mhz": spec.f0_mhz,
        "dx_mm": dx * 1e3,
        "native_dx_mm": NATIVE_DX * 1e3,
        "canonical_axes": list(CANONICAL_AXES),
        "source_offset_vox": list(offset_vox),
        "pad_before_vox": list(pad_before),
        "spec": spec.model_dump(mode="json"),
        "materials_table": table.describe(),
        "heterogeneity": het_meta,
        "log": log,
        "warnings": warnings,
        "citation": CITATION,
        "build_seconds": round(time.perf_counter() - t_start, 3),
    }

    asset = PhantomAsset(
        labels=labels,
        dx=dx,
        materials=table.material_db(),
        origin=(0.0, 0.0, 0.0),
        properties=props,
        meta=meta,
    )
    if progress is not None:
        progress("done", 1.0)
    return asset


# --------------------------------------------------------------------- steps


def _simplify(codes: np.ndarray, spec: PhantomSpec):
    """Label-domain cleanup at the native spacing."""
    s = spec.simplify
    log: list[str] = []
    warn: list[str] = []
    if not s.touches_labels:
        return codes, log, warn

    if s.drop_muscle:
        n = int((codes == MUSCLE_CODE).sum())
        codes = codes.copy()
        codes[codes == MUSCLE_CODE] = IMMERSION_CODE
        log.append(f"dropped the chest wall ({n:,} muscle voxels -> coupling medium)")
        warn.append(
            "chest wall removed: reflections from the muscle interface and the "
            "rib-cage standoff are no longer modelled"
        )
    if s.drop_skin:
        n = int((codes == SKIN_CODE).sum())
        doomed = codes == SKIN_CODE
        if doomed.any():
            # Fill from the nearest TISSUE, not the nearest non-skin voxel.
            # Skin is a ~3-voxel shell with bath on the outside, so a plain
            # nearest-neighbour fill sent 57.5% of it to the coupling medium
            # and pulled the whole breast surface back ~1.2 mm — moving the
            # water/tissue interface, which is exactly where the beam enters
            # (review finding, 2026-08-18). Filling with tissue keeps the
            # breast the size it was and simply removes the skin layer, which
            # is what "remove skin" is asked to mean.
            codes = codes.copy()
            donors = np.isin(codes, np.asarray((MUSCLE_CODE, *TISSUE_CODES), dtype=codes.dtype))
            codes[doomed] = processing.fill_from(codes, doomed, donors, fallback=IMMERSION_CODE)
        log.append(f"dropped skin ({n:,} voxels -> the tissue behind it)")
        warn.append(
            "skin removed: the strongest impedance step in the path is gone, so "
            "transmitted pressure will be optimistic"
        )
    if s.keep_largest_only:
        before = int((codes != IMMERSION_CODE).sum())
        codes = processing.keep_largest_component(codes, IMMERSION_CODE)
        after = int((codes != IMMERSION_CODE).sum())
        log.append(f"kept the largest tissue component ({before - after:,} stray voxels removed)")
    if s.fill_holes:
        before = int((codes == IMMERSION_CODE).sum())
        codes = processing.fill_interior_holes(codes, IMMERSION_CODE)
        after = int((codes == IMMERSION_CODE).sum())
        log.append(f"filled {before - after:,} enclosed coupling-medium pockets")
    if s.remove_islands_vox > 1:
        codes, removed = processing.remove_islands(
            codes, s.remove_islands_vox, protect=(SKIN_CODE, IMMERSION_CODE)
        )
        log.append(f"dissolved islands < {s.remove_islands_vox} voxels ({removed:,} voxels)")
    if s.close_skin_iterations:
        before = int((codes == SKIN_CODE).sum())
        codes = processing.close_class(codes, SKIN_CODE, s.close_skin_iterations)
        log.append(
            f"closed the skin layer x{s.close_skin_iterations} "
            f"(+{int((codes == SKIN_CODE).sum()) - before:,} voxels)"
        )
    if s.smooth_iterations:
        before = processing.interface_area_voxels(codes)
        codes = processing.smooth_labels(codes, s.smooth_iterations)
        after = processing.interface_area_voxels(codes)
        log.append(
            f"majority-smoothed x{s.smooth_iterations} (interface faces {before:,} -> {after:,})"
        )
    return np.ascontiguousarray(codes), log, warn


def _bbox_slices(mask: np.ndarray, margin: int, limits: tuple[int, ...]) -> list[slice]:
    """Bounding-box slices of a boolean mask, grown by ``margin`` and clamped.

    ``limits`` is the shape of the volume the slices will index, which is not
    always ``mask.shape`` — the breast-crop measures its transverse box on a
    z-subset of the mask, and clamping to that subset's depth would be wrong.
    """
    out = []
    for axis in range(mask.ndim):
        others = tuple(i for i in range(mask.ndim) if i != axis)
        present = np.flatnonzero(mask.any(axis=others))
        lo = max(0, int(present[0]) - margin)
        hi = min(limits[axis], int(present[-1]) + 1 + margin)
        out.append(slice(lo, hi))
    return out


def _crop(codes: np.ndarray, pval: np.ndarray | None, spec: PhantomSpec):
    """Reduce to the region of interest; keeps ``pval`` in lockstep."""
    c = spec.crop
    if c.mode == "none":
        return codes, pval, (0, 0, 0)
    if c.mode in ("tissue", "breast"):
        # The chest wall is a full-area slab, and the subcutaneous fat layer
        # behind the breast covers the whole transverse slice too, so a plain
        # bounding box over "all tissue" cannot crop transversally at all
        # (measured: 212x328 in, 212x328 out). 'breast' measures the
        # TRANSVERSE box only on the protruding slices — those whose tissue
        # coverage stays below `breast_coverage` — while the propagation
        # extent still spans everything.
        margin = int(round(c.margin_mm * 1e-3 / NATIVE_DX))
        soft = np.asarray((SKIN_CODE, *TISSUE_CODES), dtype=codes.dtype)
        mask = np.isin(codes, soft)
        if not mask.any():
            raise ValueError("the phantom holds no skin or breast tissue to crop to")
        sl = list(_bbox_slices(mask, margin, codes.shape))
        if c.mode == "breast":
            protruding = mask.mean(axis=(0, 1)) <= c.breast_coverage
            if protruding.any() and mask[:, :, protruding].any():
                trans = _bbox_slices(mask[:, :, protruding], margin, codes.shape)
                sl[0], sl[1] = trans[0], trans[1]
            # else: every slice looks like chest wall (a manual pre-crop, an
            # odd phantom) — keep the full box rather than returning nothing.
        # Always extend the propagation axis over whatever chest wall exists:
        # the muscle reflection is part of the physics, not scenery.
        muscle = codes == MUSCLE_CODE
        if muscle.any():
            z = np.flatnonzero(muscle.any(axis=(0, 1)))
            sl[2] = slice(min(sl[2].start, int(z[0])), max(sl[2].stop, int(z[-1]) + 1))
        sl = tuple(sl)
    else:  # manual
        sl = tuple(
            slice(
                max(0, int(round(st * 1e-3 / NATIVE_DX))),
                min(n, int(round((st + sz) * 1e-3 / NATIVE_DX))),
            )
            for st, sz, n in zip(c.start_mm, c.size_mm, codes.shape, strict=True)  # type: ignore[arg-type]
        )
        if any(s.stop - s.start < 1 for s in sl):
            raise ValueError(f"manual crop {c.start_mm} + {c.size_mm} mm selects an empty region")
    out = np.ascontiguousarray(codes[sl])
    out_pval = np.ascontiguousarray(pval[sl]) if pval is not None else None
    return out, out_pval, tuple(s.start for s in sl)


def _grow_domain(codes: np.ndarray, pval: np.ndarray | None, spec: PhantomSpec, dx: float):
    """Add standoff/backing/lateral coupling medium, then FFT-friendly padding.

    ``pval`` is padded with 0, which is exactly what the repository uses for
    non-breast voxels — so added coupling medium interpolates to its own LOW
    bound rather than picking up a neighbour's texture.
    """
    d = spec.domain
    lat = int(round(d.lateral_margin_mm * 1e-3 / dx))
    before = [lat, lat, int(round(d.standoff_mm * 1e-3 / dx))]
    after = [lat, lat, int(round(d.backing_mm * 1e-3 / dx))]
    if d.fft_friendly:
        # Settle the PROPAGATION axis here, entirely at the back, before the
        # centre-anchored pass below ever sees it. Centring z would split the
        # FFT growth and insert half of it in FRONT of the transducer face,
        # inflating the user's standoff by an amount that depends on dx: at
        # dx = 0.55 mm a requested 20 mm standoff silently became 23.65 mm
        # (~2.6 wavelengths at 1 MHz), and sweeping dx moved it between 0 and
        # 3.85 mm with no pattern (review finding, 2026-08-18). Padding behind
        # the chest wall costs the same voxels and changes nothing anyone
        # looks at.
        nz = codes.shape[2] + before[2] + after[2]
        after[2] += processing.next_fft_friendly(nz) - nz
    if any(before) or any(after):
        codes = processing.pad(codes, tuple(before), tuple(after), IMMERSION_CODE)  # type: ignore[arg-type]
        if pval is not None:
            pval = np.pad(
                pval,
                tuple((b, a) for b, a in zip(before, after, strict=True)),
                mode="constant",
                constant_values=0.0,
            )
    if d.fft_friendly:
        # z is already smooth (handled above), so this pass only grows x and
        # y — where centring is what you want, to keep the breast in the
        # middle of the transverse plane.
        grown, extra_before = processing.pad_to_fft_friendly(codes, IMMERSION_CODE, anchor="center")
        if grown.shape != codes.shape:
            widths = tuple(
                (b, g - n - b)
                for n, g, b in zip(codes.shape, grown.shape, extra_before, strict=True)
            )
            if pval is not None:
                pval = np.pad(pval, widths, mode="constant", constant_values=0.0)
            codes = grown
            before = [b + e for b, e in zip(before, extra_before, strict=True)]
    return np.ascontiguousarray(codes), pval, tuple(before)


def _properties(
    labels: np.ndarray,
    codes: np.ndarray,
    pval: np.ndarray | None,
    table: TissueTable,
    spec: PhantomSpec,
    dx: float,
):
    """Dense property volumes, or ``None`` when labels alone suffice."""
    h = spec.heterogeneity
    log: list[str] = []
    if not h.produces_continuous_fields:
        log.append("piecewise-constant medium (labels + MaterialDB); no dense volumes stored")
        return None, {"mode": "labels"}, log

    idx = labels.astype(np.intp)
    mid = table.lookup("mid")
    props = PropertyVolumes(**{n: mid[n][idx] for n in PROPERTY_NAMES})
    meta: dict = {"mode": "continuous", "use_pval": bool(h.use_pval and pval is not None)}

    if h.use_pval and pval is not None:
        props = apply_pval_interpolation(
            props,
            codes,
            pval,
            table.lookup_by_code("lo"),
            table.lookup_by_code("hi"),
        )
        interpolated = np.isin(codes, np.asarray(sorted(PVAL_CODES), dtype=codes.dtype))
        log.append(
            f"pval interpolation applied to {100 * float(interpolated.mean()):.1f}% of voxels, "
            f"within each MEDIA NUMBER's own property band (codes {sorted(PVAL_CODES)}); skin, "
            f"muscle and the coupling medium stay at their class midpoints because the "
            f"repository ships them p = 0 as 'no data'"
        )
        meta["pval_codes"] = sorted(PVAL_CODES)
        meta["pval_ids"] = sorted(table.pval_ids)
    elif h.use_pval:
        log.append("pval requested but unavailable; properties stay at class midpoints")
        meta["pval_missing"] = True

    if h.noise_pct > 0:
        # Ask the table which id is the bath — it is 0 in the detailed and
        # grouped models but 4 in 'simple' (which mirrors breast_default's
        # id space), and hardcoding 0 would have put scatterers in the water
        # and left the tissue smooth.
        coupling = table.coupling_id
        mask = labels != coupling if h.tissue_only else np.ones(labels.shape, dtype=bool)
        corr_vox = h.correlation_mm * 1e-3 / dx
        props, noise_meta = apply_scatterer_noise(
            props,
            mask,
            amplitude_pct=h.noise_pct,
            correlation_vox=corr_vox,
            properties=tuple(h.properties),
            seed=h.seed,
            coupled=h.coupled,
        )
        meta["noise"] = noise_meta
        log.append(
            f"scatterer noise {h.noise_pct:g}% on {list(h.properties)}, "
            f"correlation {h.correlation_mm:g} mm = {corr_vox:.2f} vox, seed {h.seed}"
        )
        if 0 < corr_vox < 0.5:
            log.append(
                "note: the requested correlation length is below one voxel, so the "
                "noise is effectively white at this dx"
            )
    return props, meta, log


# ------------------------------------------------------------------ warnings


def _resolution_warnings(dx: float, spec: PhantomSpec) -> list[str]:
    out = []
    if dx > NATIVE_DX:
        factor = dx / NATIVE_DX
        if factor >= 3.0:
            out.append(
                f"dx = {dx * 1e3:g} mm is {factor:.1f}x the 0.5 mm source data: the ~1.5 mm skin "
                f"layer is thinner than one voxel and will be partly or wholly lost"
            )
        elif factor > 1.5:
            out.append(
                f"dx = {dx * 1e3:g} mm under-resolves the ~1.5 mm skin layer "
                f"({1.5e-3 / dx:.1f} voxels across it)"
            )
    return out


def _resolution_vs_wavelength(table: TissueTable, dx: float, spec: PhantomSpec) -> list[str]:
    """Warn when ``dx`` cannot carry a wave at ``f0`` through the slowest tissue.

    This is the failure that does not look like a failure: the build succeeds,
    the medium is valid, the solver runs to completion — and returns a field
    with no focus in it, because at 1.3 points per wavelength the scheme
    cannot propagate. Caught the first time this module's own end-to-end check
    was run at dx = 1.2 mm and 1 MHz (peak pressure landed ON the transducer).

    The k-space PSTD scheme here is usable down to ~2 ppw; the library's own
    design rule of thumb is ~4.4 ppw at f0. Both thresholds are reported
    against the SLOWEST material present, since that is where the wavelength
    is shortest.
    """
    c_min = min(AcousticTissue._pick(t.c, "lo") for t in table.tissues.values())
    ppw = c_min / (spec.f0 * dx)
    if ppw >= 4.0:
        return []
    advice = (
        f"drop dx to {c_min / (spec.f0 * 4.0) * 1e3:.2f} mm, or build for "
        f"f0 <= {c_min / (4.0 * dx) / 1e6:.2f} MHz"
    )
    if ppw < 2.0:
        return [
            f"UNUSABLE at f0 = {spec.f0_mhz:g} MHz: dx = {dx * 1e3:g} mm gives only "
            f"{ppw:.1f} points per wavelength in the slowest tissue ({c_min:.0f} m/s). "
            f"Below ~2 ppw the k-space PSTD scheme cannot propagate the wave — the solve "
            f"will run and produce a field with no focus in it. {advice}."
        ]
    return [
        f"marginal resolution at f0 = {spec.f0_mhz:g} MHz: {ppw:.1f} points per wavelength "
        f"in the slowest tissue ({c_min:.0f} m/s); the library's design rule is ~4.4. "
        f"Harmonics are worse by their order. {advice}."
    ]


def _geometry_warnings(codes: np.ndarray, dx: float, spec: PhantomSpec) -> list[str]:
    out = []
    if not spec.simplify.drop_skin:
        skin_vox = int((codes == SKIN_CODE).sum())
        if skin_vox == 0:
            out.append("no skin voxels survived the build")
    if not spec.simplify.drop_muscle:
        muscle = codes == MUSCLE_CODE
        if muscle.any():
            z = np.flatnonzero(muscle.any(axis=(0, 1)))
            thickness_mm = (int(z[-1]) - int(z[0]) + 1) * dx * 1e3
            if thickness_mm < 3.0:
                out.append(
                    f"the chest wall is only {thickness_mm:.1f} mm thick after the build "
                    f"(source: 5 mm) — the crop or resample clipped it"
                )
        else:
            out.append("no chest-wall voxels survived the build")
    tissue = np.isin(codes, np.asarray(TISSUE_CODES, dtype=codes.dtype))
    if tissue.any():
        z = np.flatnonzero(tissue.any(axis=(0, 1)))
        if int(z[0]) == 0:
            out.append(
                "breast tissue touches the z = 0 face: there is no coupling medium in "
                "front of it, so a transducer has nowhere to sit (raise domain.standoff_mm)"
            )
    return out
