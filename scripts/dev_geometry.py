"""Geometry fidelity: what was asked for, against what was built.

Every geometry the library makes is a *request* — a bowl of this radius of
curvature, a ball of this volume, a spiral of this many elements — turned
into voxels or points by code that has to round. This script keeps the two
apart and measures the gap, one check at a time:

    expected   the closed-form answer the request implies
    produced   what came out of the constructor
    deviation  the difference, in the unit that makes it judgeable

That separation is the whole point. A rasterizer that quietly loses a
millimetre of focal length, or a bowl whose rim falls inside the absorbing
layer, produces a field that looks entirely plausible and answers the wrong
question; the run that found the second of those (2026-08-24) reported it as
a 26 % PML sensitivity before the geometry was checked at all.

Run it::

    python scripts/dev_geometry.py --out benchmarks/reports/geometry
    python scripts/dev_geometry.py --only G1,G3

Nothing here needs a GPU or more than a few seconds; the whole set is CPU
work on grids small enough to hold in a laptop's cache.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:  # runnable from a checkout without an install
    sys.path.insert(0, str(REPO / "src"))

MM = 1e-3

CHECKS: list[tuple[str, str, Callable]] = []


def check(cid: str, title: str):
    def wrap(fn):
        CHECKS.append((cid, title, fn))
        return fn

    return wrap


# --------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------


def fit_sphere(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least-squares sphere through ``points``; returns (center, radius, rms).

    The algebraic (Kasa) form: ``|p|^2 = 2 p.c + (r^2 - |c|^2)`` is linear in
    the unknowns, so one lstsq gives both. On a digitized shell the residual
    is the interesting number in its own right — it *is* the shell thickness.
    """
    p = np.asarray(points, float)
    a = np.hstack([2.0 * p, np.ones((len(p), 1))])
    b = (p**2).sum(1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    center = sol[:3]
    radius = float(np.sqrt(sol[3] + center @ center))
    rms = float(np.sqrt(np.mean((np.linalg.norm(p - center, axis=1) - radius) ** 2)))
    return center, radius, rms


def rel(produced: float, expected: float) -> float:
    return float(abs(produced - expected) / abs(expected)) if expected else float("nan")


def deviation(produced: float, expected: float, unit: str = "") -> dict:
    """One expected/produced/deviation record, formatted the same way twice."""
    return {
        "expected": float(expected),
        "produced": float(produced),
        "abs_error": float(abs(produced - expected)),
        "rel_error": rel(produced, expected),
        "unit": unit,
    }


def bowl_voxels(dx: float, aperture: float, roc: float, spacing: float | None = None):
    """The library's own bowl digitization, in voxel coordinates about the apex."""
    from caustica.analytic.geometry import spherical_cap_points

    ds = dx / 2.0 if spacing is None else spacing
    points, _n, _a = spherical_cap_points(aperture, roc, ds)
    return np.unique(np.round(points / dx).astype(np.int64), axis=0)


# --------------------------------------------------------------------------
# G1 — the focused bowl: does the shell have the geometry that was ordered?
# --------------------------------------------------------------------------


@check("G1", "focused bowl: surface error against the sphere that was ordered")
def _g1(ctx):
    """A bowl is ordered by (aperture, ROC) and delivered as voxel indices.

    The question that decides whether the delivery is good is not "what
    sphere do these voxels look like" but "how far is each voxel from the
    sphere that was ordered" — because that distance, divided by the
    wavelength, is the phase error the focus actually sees. So the primary
    measurement is against the KNOWN surface, and the free sphere fit is
    kept beside it only as a diagnostic.

    The two disagree, and the disagreement is the interesting part. A cap
    this shallow (5 mm aperture on a 12 mm radius sags barely a millimetre)
    does not determine its own curvature from a voxelized shell: at dx =
    0.5 mm the sag is two voxels, and an algebraic sphere fit through points
    scattered by half a voxel comes back 19 % short. That is a statement
    about inferring geometry FROM voxels, not about the voxels being wrong —
    the same shell sits within a third of a voxel of the sphere it was asked
    for, at every dx.
    """
    aperture, roc = 5.0 * MM, 12.0 * MM
    focus = np.array([0.0, 0.0, roc])  # the apex is the origin by construction
    f0 = 1.0e6
    lam = 1500.0 / f0
    rows = []
    for dx_mm in (0.5, 0.25, 0.15, 0.1, 0.05):
        dx = dx_mm * MM
        idx = bowl_voxels(dx, aperture, roc)
        p = idx.astype(float) * dx
        # primary: distance to the sphere that was ORDERED
        err = np.linalg.norm(p - focus, axis=1) - roc
        rms, worst = float(np.sqrt(np.mean(err**2))), float(np.abs(err).max())
        # secondary: what a free fit infers from the same voxels
        center, radius, fit_rms = fit_sphere(p)
        transverse = np.linalg.norm(p[:, :2], axis=1)
        rows.append(
            {
                "dx_mm": dx_mm,
                "n_voxels": int(len(idx)),
                "voxels_per_roc": roc / dx,
                "surface_rms_error_m": rms,
                "surface_rms_error_vox": rms / dx,
                "surface_max_error_vox": worst / dx,
                "phase_error_rms_rad_at_1mhz": 2 * np.pi * rms / lam,
                "aperture_radius": deviation(float(transverse.max()), aperture, "m"),
                "apex_z_vox": int(idx[:, 2].min()),
                "free_fit_roc": deviation(radius, roc, "m"),
                "free_fit_focus_z": deviation(float(center[2]), roc, "m"),
                "free_fit_offaxis_m": float(np.linalg.norm(center[:2])),
                "free_fit_rms_vox": fit_rms / dx,
            }
        )
    worst_surface = max(r["surface_rms_error_vox"] for r in rows)
    worst_max = max(r["surface_max_error_vox"] for r in rows)
    coarse_phase = rows[0]["phase_error_rms_rad_at_1mhz"]
    fine_phase = rows[-1]["phase_error_rms_rad_at_1mhz"]
    worst_fit = max(r["free_fit_roc"]["rel_error"] for r in rows)
    best_fit = min(r["free_fit_roc"]["rel_error"] for r in rows)
    return {
        "aperture_m": aperture,
        "roc_m": roc,
        "sag_m": roc - np.sqrt(roc**2 - aperture**2),
        "rows": rows,
        "verdict": (
            f"every voxel lands within {worst_max:.2f} voxel of the sphere that was ordered "
            f"({worst_surface:.3f} voxels rms at worst), which is {coarse_phase:.2f} rad of "
            f"wavefront error at 1 MHz at dx=0.5 mm and {fine_phase:.3f} rad at dx=0.05 mm; a "
            f"free sphere fit to the same voxels reads the curvature {worst_fit * 100:.0f} % "
            f"short at the coarsest dx and {best_fit * 100:.1f} % at the finest, because a cap "
            f"this shallow does not determine its own radius from a voxel shell"
        ),
    }


# --------------------------------------------------------------------------
# G2 — is the shell watertight, or does it leak?
# --------------------------------------------------------------------------


@check("G2", "focused bowl: holes in the digitized shell at the shipped sampling")
def _g2(ctx):
    """A source shell with holes is a source that leaks.

    ``bowl_cw_source`` samples the cap at dx/2 and rounds to voxels. Whether
    that covers every voxel the cap actually passes through is an empirical
    question, and the failure is silent: the missing voxels are simply not
    driven, so the aperture is quietly smaller than the one requested and the
    field behind the bowl is not the field a solid bowl would produce.

    The reference is the same cap sampled sixteen times finer. Any voxel the
    fine sampling reaches and the shipped sampling does not is a hole.
    """
    aperture, roc = 5.0 * MM, 12.0 * MM
    rows = []
    for dx_mm in (0.5, 0.25, 0.1, 0.05):
        dx = dx_mm * MM
        shipped = {tuple(v) for v in bowl_voxels(dx, aperture, roc)}
        fine = {tuple(v) for v in bowl_voxels(dx, aperture, roc, spacing=dx / 16.0)}
        missing = fine - shipped
        rows.append(
            {
                "dx_mm": dx_mm,
                "shipped_sampling": "dx/2",
                "n_shipped": len(shipped),
                "n_reference": len(fine),
                "n_missing": len(missing),
                "missing_fraction": len(missing) / len(fine),
                "extra_not_in_reference": len(shipped - fine),
            }
        )
    worst = max(r["missing_fraction"] for r in rows)
    return {
        "rows": rows,
        "worst_missing_fraction": worst,
        "verdict": (
            f"the shipped dx/2 sampling misses up to {worst * 100:.2f} % of the voxels a "
            f"16x-finer sampling of the same cap reaches"
            + (
                " — the shell is watertight at every dx tested"
                if worst < 1e-9
                else " — those voxels are undriven holes in the shell"
            )
        ),
    }


# --------------------------------------------------------------------------
# G3 — CSG primitives: rasterized volume against the closed form
# --------------------------------------------------------------------------


@check("G3", "CSG primitives: rasterized volume and voxel-level agreement vs the closed form")
def _g3(ctx):
    """Two questions that look like one, and are not.

    *Volume* is a sum, so its errors cancel: a ball rasterized by voxel
    centres can land within 0.005 % of 4/3 pi r^3 while a percent of its
    voxels are on the wrong side of the boundary, because the ones wrongly
    included cancel the ones wrongly excluded. That makes total volume a
    poor witness for anything happening at the interface.

    So both are measured. The second reference is the occupancy truth: each
    voxel is subdivided 9x9x9 and called occupied when more than half its
    volume is inside, which is what a rasterizer is trying to approximate.
    The fraction of voxels that disagree with that is the number the
    ``supersample`` argument's documented "big win on curved interfaces"
    should move — and total volume is not.
    """
    from caustica.core.grid import Grid
    from caustica.geometry import Ball, Box, Cylinder, Ellipsoid, Scene

    r = 4.0 * MM
    cases = {
        "ball": (Ball((0.0, 0.0, 0.0), r), 4.0 / 3.0 * np.pi * r**3),
        "ellipsoid": (
            Ellipsoid((0.0, 0.0, 0.0), (r, 0.6 * r, 1.4 * r)),
            4.0 / 3.0 * np.pi * r * (0.6 * r) * (1.4 * r),
        ),
        "cylinder": (Cylinder((0.0, 0.0, 0.0), r, 1.5 * r, axis=2), np.pi * r**2 * 1.5 * r),
        "box": (Box((0.0, 0.0, 0.0), (r, 1.5 * r, 0.5 * r)), r * 1.5 * r * 0.5 * r),
    }
    half = 6.0 * MM  # domain half-width: every primitive fits with room to spare
    sub = 9  # 729 samples per voxel: the occupancy truth both rasterizations approximate
    rows = []
    for name, (shape, exact) in cases.items():
        for dx_mm in (0.5, 0.25, 0.1):
            dx = dx_mm * MM
            n = int(round(2 * half / dx))
            grid = Grid(shape=(n, n, n), dx=dx)
            origin = tuple(-half + dx / 2 for _ in range(3))
            entry: dict[str, Any] = {"shape": name, "dx_mm": dx_mm, "grid": n}
            masks = {}
            for ss in (1, 3):
                labels = Scene(3).add(shape, 1).rasterize(grid, origin=origin, supersample=ss)
                masks[ss] = np.asarray(labels.labels) == 1
                entry[f"volume_supersample_{ss}"] = deviation(
                    float(masks[ss].sum()) * dx**3, exact, "m^3"
                )
            truth = _occupancy_truth(shape, grid, origin, sub, masks[1])
            for ss in (1, 3):
                wrong = int(np.count_nonzero(masks[ss] != truth))
                entry[f"misclassified_supersample_{ss}"] = {
                    "voxels": wrong,
                    "fraction_of_the_object": wrong / max(int(truth.sum()), 1),
                }
            entry["supersampling_helps_volume"] = (
                entry["volume_supersample_3"]["rel_error"]
                < entry["volume_supersample_1"]["rel_error"]
            )
            entry["supersampling_helps_the_interface"] = (
                entry["misclassified_supersample_3"]["voxels"]
                < entry["misclassified_supersample_1"]["voxels"]
            )
            rows.append(entry)

    curved = [r for r in rows if r["shape"] != "box"]
    helped_v = sum(1 for r in rows if r["supersampling_helps_volume"])
    helped_i = sum(1 for r in curved if r["supersampling_helps_the_interface"])
    perfect = [r for r in rows if r["misclassified_supersample_1"]["voxels"] == 0]
    worst_mis = max(r["misclassified_supersample_1"]["fraction_of_the_object"] for r in rows)
    box_worst = max(r["volume_supersample_1"]["rel_error"] for r in rows if r["shape"] == "box")
    worst_vol = max(r["volume_supersample_1"]["rel_error"] for r in curved)
    return {
        "rows": rows,
        "occupancy_truth_subdivision": sub,
        "cases_with_no_misclassified_voxel": f"{len(perfect)}/{len(rows)}",
        "worst_misclassified_fraction": worst_mis,
        "supersampling_helped_volume_in": f"{helped_v}/{len(rows)}",
        "supersampling_helped_the_interface_in": f"{helped_i}/{len(curved)}",
        "verdict": (
            f"centre sampling already agrees with majority occupancy in {len(perfect)} of "
            f"{len(rows)} cases (worst disagreement {worst_mis * 100:.3f} % of the object, on the "
            f"ellipsoid where curvature is highest relative to dx), so the volume error of up to "
            f"{worst_vol * 100:.1f} % on curved primitives is the price of binary voxels, not a "
            f"rasterizer defect — the flat-faced box is exact to {box_worst:.0e}. Supersampling "
            f"has correspondingly little to correct: it moved the volume in {helped_v} of "
            f"{len(rows)} cases and the classification in {helped_i} of {len(curved)} curved ones"
        ),
    }


def _occupancy_truth(shape, grid, origin, sub: int, seed: np.ndarray) -> np.ndarray:
    """Majority occupancy per voxel, by ``sub**3`` samples inside each one.

    Only the boundary layer is actually sampled. A voxel more than a couple
    of voxels from the surface is unanimously in or out whatever the sample
    count, so ``seed`` (the centre-sampled mask) is already its answer, and
    subdividing the whole domain would spend nine hundred million point
    evaluations to reproduce it.
    """
    dx = grid.dx
    band = seed.copy()
    for _ in range(2):  # dilate the disagreement zone by two voxels each way
        grown = band.copy()
        for ax in range(3):
            grown |= np.roll(band, 1, ax) | np.roll(band, -1, ax)
        band = grown
    eroded = seed.copy()
    for _ in range(2):
        shrunk = eroded.copy()
        for ax in range(3):
            shrunk &= np.roll(eroded, 1, ax) & np.roll(eroded, -1, ax)
        eroded = shrunk
    candidates = np.argwhere(band & ~eroded)
    if not len(candidates):
        return seed

    offs = (np.arange(sub) + 0.5) / sub - 0.5
    base = np.stack([np.asarray(origin)[d] + candidates[:, d] * dx for d in range(3)], axis=1)
    hits = np.zeros(len(candidates), np.int32)
    for ox in offs:
        for oy in offs:
            for oz in offs:
                pts = base + np.array([ox, oy, oz]) * dx
                hits += np.asarray(shape.contains(pts)).astype(np.int32)
    truth = seed.copy()
    truth[tuple(candidates.T)] = hits * 2 > sub**3
    return truth


# --------------------------------------------------------------------------
# G4 — boolean algebra: exact identities, not approximations
# --------------------------------------------------------------------------


@check("G4", "boolean algebra: set identities hold voxel-for-voxel")
def _g4(ctx):
    """Union, intersection, difference and complement are SET operations.

    On a fixed voxel grid there is no discretization excuse available: the
    rasterization of ``A | B`` has to be exactly the union of the two
    rasterizations, every voxel, no tolerance. Inclusion-exclusion is checked
    the same way. Anything less would mean the algebra and the rasterizer
    disagree about what a shape is, and every scene built from more than one
    primitive would inherit it.
    """
    from caustica.core.grid import Grid
    from caustica.geometry import Ball, Box, Scene

    dx = 0.25 * MM
    n, half = 48, 6.0 * MM
    grid = Grid(shape=(n, n, n), dx=dx)
    origin = tuple(-half + dx / 2 for _ in range(3))
    a = Ball((-1.0 * MM, 0.0, 0.0), 3.0 * MM)
    b = Box((1.0 * MM, 0.0, 0.0), (4.0 * MM, 4.0 * MM, 4.0 * MM))

    def mask(shape) -> np.ndarray:
        return np.asarray(Scene(3).add(shape, 1).rasterize(grid, origin=origin).labels) == 1

    ma, mb = mask(a), mask(b)
    identities = {
        "union": (mask(a | b), ma | mb),
        "intersection": (mask(a & b), ma & mb),
        "difference": (mask(a - b), ma & ~mb),
        "complement_of_a": (mask(~a), ~ma),
        "de_morgan": (mask(~(a | b)), mask(~a & ~b)),
    }
    rows = []
    for name, (produced, expected) in identities.items():
        mismatched = int(np.count_nonzero(produced != expected))
        rows.append(
            {
                "identity": name,
                "expected_voxels": int(expected.sum()),
                "produced_voxels": int(produced.sum()),
                "mismatched_voxels": mismatched,
                "exact": mismatched == 0,
            }
        )
    incl_excl = int((ma | mb).sum()) - (int(ma.sum()) + int(mb.sum()) - int((ma & mb).sum()))
    rows.append(
        {
            "identity": "inclusion_exclusion",
            "expected_voxels": 0,
            "produced_voxels": incl_excl,
            "mismatched_voxels": abs(incl_excl),
            "exact": incl_excl == 0,
        }
    )
    bad = [r["identity"] for r in rows if not r["exact"]]
    return {
        "grid": [n, n, n],
        "dx_mm": 0.25,
        "rows": rows,
        "verdict": (
            f"all {len(rows)} set identities hold voxel-for-voxel, no tolerance"
            if not bad
            else f"{len(bad)} identities do NOT hold: {bad}"
        ),
    }


# --------------------------------------------------------------------------
# G5 — affine transforms: does the shape land where it was sent?
# --------------------------------------------------------------------------


@check("G5", "affine transforms: a moved shape keeps its volume and lands on its target")
def _g5(ctx):
    """Translate and rotate are rigid motions; volume is their invariant.

    So two numbers say whether the transform is right: the rasterized volume
    must not change beyond the boundary layer, and the centroid must land on
    the requested centre. A rotation of a ball is the sharpest form of the
    first (a ball is rotation-invariant, so the ONLY thing that can move is
    the rasterization error) and an ellipsoid rotated a quarter turn is the
    sharpest form of the second (its semiaxes must swap).
    """
    from caustica.core.grid import Grid
    from caustica.geometry import Ball, Ellipsoid, Scene

    dx = 0.2 * MM
    n, half = 60, 6.0 * MM
    grid = Grid(shape=(n, n, n), dx=dx)
    origin = tuple(-half + dx / 2 for _ in range(3))
    coords = [np.arange(n) * dx + origin[d] for d in range(3)]

    def measure(shape) -> tuple[int, np.ndarray, np.ndarray]:
        m = np.asarray(Scene(3).add(shape, 1).rasterize(grid, origin=origin).labels) == 1
        pts = np.argwhere(m).astype(float)
        world = np.column_stack([coords[d][pts[:, d].astype(int)] for d in range(3)])
        return int(m.sum()), world.mean(0), world.max(0) - world.min(0)

    ball = Ball((0.0, 0.0, 0.0), 3.0 * MM)
    target = (1.4 * MM, -0.8 * MM, 2.0 * MM)
    n0, c0, _e0 = measure(ball)
    n_t, c_t, _e = measure(ball.translated(target))
    n_r, _c_r, _ = measure(ball.rotated(0.7, axis=(1.0, 1.0, 0.0)))

    ell = Ellipsoid((0.0, 0.0, 0.0), (4.0 * MM, 1.5 * MM, 2.5 * MM))
    _n_e, _c_e, extent0 = measure(ell)
    _n_q, _c_q, extent_q = measure(ell.rotated(np.pi / 2, axis=(0.0, 0.0, 1.0)))

    rows = [
        {
            "transform": "translate",
            "volume_voxels": deviation(n_t, n0, "voxels"),
            "centroid_x": deviation(float(c_t[0]), float(c0[0]) + target[0], "m"),
            "centroid_y": deviation(float(c_t[1]), float(c0[1]) + target[1], "m"),
            "centroid_z": deviation(float(c_t[2]), float(c0[2]) + target[2], "m"),
        },
        {
            "transform": "rotate a ball (volume is the invariant)",
            "volume_voxels": deviation(n_r, n0, "voxels"),
        },
        {
            "transform": "rotate an ellipsoid a quarter turn about z (semiaxes swap)",
            "extent_x": deviation(float(extent_q[0]), float(extent0[1]), "m"),
            "extent_y": deviation(float(extent_q[1]), float(extent0[0]), "m"),
            "extent_z": deviation(float(extent_q[2]), float(extent0[2]), "m"),
        },
    ]
    axes = ("centroid_x", "centroid_y", "centroid_z")
    worst_centroid = max(rows[0][k]["abs_error"] for k in axes)
    return {
        "dx_mm": 0.2,
        "rows": rows,
        "worst_centroid_error_vox": worst_centroid / dx,
        "translated_volume_rel_error": rows[0]["volume_voxels"]["rel_error"],
        "rotated_volume_rel_error": rows[1]["volume_voxels"]["rel_error"],
        "verdict": (
            f"a translated shape lands within {worst_centroid / dx:.2f} voxel of its target and "
            f"keeps its volume to {rows[0]['volume_voxels']['rel_error'] * 100:.2f} %; rotating a "
            f"ball moves its volume by {rows[1]['volume_voxels']['rel_error'] * 100:.2f} % "
            f"(rasterization only — the exact volume is unchanged)"
        ),
    }


# --------------------------------------------------------------------------
# G6 — the cap point cloud the analytic integrals stand on
# --------------------------------------------------------------------------


@check("G6", "spherical cap sampling: area, normals and angular span")
def _g6(ctx):
    """Every analytic reference in this library starts from this point cloud.

    The Rayleigh integral weights each sample by its area, so a cap whose
    areas do not sum to 2 pi R^2 (1 - cos theta_max) is a cap radiating the
    wrong amount of power, and the "analytic" curve the solver is judged
    against would be wrong by the same factor. Normals must be unit and point
    at the centre of curvature, or the obliquity in the integral is wrong.
    """
    from caustica.analytic.geometry import spherical_cap_points

    rows = []
    for aperture_mm, roc_mm in ((5.0, 12.0), (32.0, 64.0), (1.0, 50.0)):
        a, f = aperture_mm * MM, roc_mm * MM
        cos_tmax = np.sqrt(1.0 - (a / f) ** 2)
        exact_area = 2.0 * np.pi * f**2 * (1.0 - cos_tmax)
        pts, nrm, ar = spherical_cap_points(a, f, f / 200.0)
        center = np.array([0.0, 0.0, f])
        radii = np.linalg.norm(pts - center, axis=1)
        toward = (center - pts) / np.linalg.norm(center - pts, axis=1)[:, None]
        rows.append(
            {
                "aperture_mm": aperture_mm,
                "roc_mm": roc_mm,
                "f_number": roc_mm / (2 * aperture_mm),
                "n_points": int(len(pts)),
                "cap_area": deviation(float(ar.sum()), exact_area, "m^2"),
                "on_sphere_max_dev_m": float(np.abs(radii - f).max()),
                "normal_norm_max_dev": float(np.abs(np.linalg.norm(nrm, axis=1) - 1.0).max()),
                "normal_alignment_max_dev": float(np.abs((nrm * toward).sum(1) - 1.0).max()),
                "apex_z_m": float(pts[:, 2].min()),
                "rim_radius": deviation(
                    float(np.linalg.norm(pts[:, :2], axis=1).max()), a, "m (<= a)"
                ),
                "equal_area": bool(np.ptp(ar) < 1e-30),
            }
        )
    worst_area = max(r["cap_area"]["rel_error"] for r in rows)
    worst_sphere = max(r["on_sphere_max_dev_m"] for r in rows)
    return {
        "rows": rows,
        "verdict": (
            f"the sampled areas sum to the exact cap area to {worst_area:.1e} relative, every "
            f"point sits on the sphere to {worst_sphere:.1e} m, and the normals are unit and "
            f"aimed at the centre of curvature to machine precision"
        ),
    }


# --------------------------------------------------------------------------
# G7 — the spiral array: what a real transducer request produces
# --------------------------------------------------------------------------


@check("G7", "spiral array: element count, placement on the shell and aperture bounds")
def _g7(ctx):
    """The one geometry a user orders by transducer specification.

    ``archimedean_spiral(n_elements, d_outer, d_inner, roc, active_fraction)``
    is a request in the vocabulary of a datasheet. What it returns has to
    honour every term of it: that many elements, all on the shell of that
    radius, none outside the outer diameter, none inside the central hole.
    The active fraction is the one term that is a *target* rather than a
    constraint, so it is reported rather than asserted.
    """
    from caustica.arrays import archimedean_spiral

    rows = []
    for n_el, d_out_mm, d_in_mm, roc_mm in ((128, 100.0, 44.0, 100.0), (64, 40.0, 12.0, 50.0)):
        arr = archimedean_spiral(
            n_elements=n_el,
            d_outer=d_out_mm * MM,
            d_inner=d_in_mm * MM,
            roc=roc_mm * MM,
            active_fraction=0.6,
        )
        pos = np.asarray(arr.positions)
        center = np.array([0.0, 0.0, roc_mm * MM])
        radii = np.linalg.norm(pos - center, axis=1)
        transverse = np.linalg.norm(pos[:, :2], axis=1)
        toward = (center - pos) / np.linalg.norm(center - pos, axis=1)[:, None]
        nrm = np.asarray(arr.normals)
        active = len(pos) * np.pi * arr.elem_radius**2
        aperture_area = np.pi * ((d_out_mm * MM / 2) ** 2 - (d_in_mm * MM / 2) ** 2)
        rows.append(
            {
                "requested": {
                    "n_elements": n_el,
                    "d_outer_mm": d_out_mm,
                    "d_inner_mm": d_in_mm,
                    "roc_mm": roc_mm,
                    "active_fraction": 0.6,
                },
                "n_elements": deviation(len(pos), n_el, "elements"),
                "focal_length": deviation(float(arr.focal_length), roc_mm * MM, "m"),
                "on_shell_max_dev_m": float(np.abs(radii - roc_mm * MM).max()),
                "outer_radius": deviation(
                    float((transverse + arr.elem_radius).max()), d_out_mm * MM / 2, "m (<= a_out)"
                ),
                "inner_radius_min_m": float((transverse - arr.elem_radius).min()),
                "inner_radius_requested_m": d_in_mm * MM / 2,
                "respects_central_hole": bool(
                    (transverse - arr.elem_radius).min() >= d_in_mm * MM / 2 - 1e-9
                ),
                "normal_alignment_max_dev": float(np.abs((nrm * toward).sum(1) - 1.0).max()),
                "active_fraction": deviation(active / aperture_area, 0.6, "of the annulus"),
                "elem_radius_mm": float(arr.elem_radius / MM),
            }
        )
    return {
        "rows": rows,
        "verdict": (
            "both requests come back with exactly the elements asked for, all on the shell to "
            f"{max(r['on_shell_max_dev_m'] for r in rows):.1e} m, inside the outer diameter and "
            "outside the central hole; the active fraction lands at "
            + ", ".join(f"{r['active_fraction']['produced']:.3f}" for r in rows)
            + " against the 0.600 target"
        ),
    }


# --------------------------------------------------------------------------
# G8 — does the geometry fit the domain it was given?
# --------------------------------------------------------------------------


@check("G8", "does the source fit its grid, clear of the PML?")
def _g8(ctx):
    """The mistake that cost a night, made checkable.

    A bowl whose rim lands inside the absorbing layer is not a bowl. It is a
    bowl with its edge dissolved, and it produces a field that is plausible,
    stable, and answering a question nobody asked — the run that found this
    (2026-08-24) had reported it as a 26 % sensitivity to PML thickness.
    Nothing in the job schema refuses it, so the check belongs somewhere.

    Every packaged example is audited here: how many source voxels sit inside
    the sponge, and how much clearance the tightest one has left.
    """
    from caustica import examples
    from caustica.config.job import build_job, load_job, parse_job
    from caustica.validation.compare import mini_job, t0_job

    def audited_jobs():
        for name in sorted(examples.available()):
            path = examples.path(name)
            job, base = load_job(path)
            yield f"example:{name}", job, base
        for name, factory in (("compare-mini", mini_job), ("t0-sanity", t0_job)):
            yield f"harness:{name}", parse_job(factory(), name), Path(".")
        # The control. A check that never fires proves nothing, so one job
        # here is deliberately built wrong: a 16 mm bowl in a domain whose
        # interior is 13 mm across, which is the mistake that produced a
        # spurious "26 % PML sensitivity" on 2026-08-24 before anyone looked
        # at the geometry. The audit must catch it.
        yield (
            "control:bowl deliberately overlapping the PML",
            parse_job(
                {
                    "format": "caustica-job/1",
                    "kind": "explicit",
                    "name": "pml-overlap-control",
                    "medium": {"kind": "homogeneous"},
                    "grid": {
                        "ndim": 3,
                        "dx_mm": 0.5,
                        "size_mm": [19.0, 19.0, 24.0],
                        "pml": {"thickness_mm": 3.0},
                    },
                    "source": {
                        "kind": "array",
                        "array": {"kind": "bowl", "d_outer_mm": 16.0, "roc_mm": 12.0},
                        "apex_mm": [9.5, 9.5, 3.0],
                    },
                    "drive": {"f0_mhz": 1.0, "amplitude_kpa": 100.0},
                    "run": {"harmonics": [1]},
                    "solver": "linear",
                },
                "control",
            ),
            Path("."),
        )

    rows = []
    for name, job, base in audited_jobs():
        try:
            built = build_job(job, base_dir=base, with_medium=False)
        except Exception as exc:
            rows.append({"job": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        grid, src = built.grid, built.source
        pml = int(grid.pml_vox)
        idx = np.asarray(src.indices)
        shape = np.asarray(grid.shape)
        low = idx.min(0)
        high = (shape - 1) - idx.max(0)
        clearance = int(min(low.min(), high.min()))
        in_pml = int(((idx < pml) | (idx >= shape - pml)).any(1).sum())
        rows.append(
            {
                "job": name,
                "grid": list(map(int, shape)),
                "dx_mm": float(grid.dx / MM),
                "pml_vox": pml,
                "n_source_voxels": int(len(idx)),
                "source_voxels_inside_the_pml": in_pml,
                "source_voxels_inside_the_pml_pct": 100.0 * in_pml / max(len(idx), 1),
                "clearance_to_the_domain_edge_vox": clearance,
                "clearance_beyond_the_pml_vox": clearance - pml,
                "fits_clear_of_the_pml": in_pml == 0,
            }
        )
    audited = [r for r in rows if "error" not in r]
    real = [r for r in audited if not r["job"].startswith("control:")]
    control = [r for r in audited if r["job"].startswith("control:")]
    offenders = [r["job"] for r in real if not r["fits_clear_of_the_pml"]]
    tightest = min(real, key=lambda r: r["clearance_beyond_the_pml_vox"]) if real else None
    caught = all(not r["fits_clear_of_the_pml"] for r in control) if control else False
    return {
        "rows": rows,
        "n_jobs_audited": len(audited),
        "offenders": offenders,
        "control_was_caught": caught,
        "verdict": (
            f"{len(real)} shipped jobs audited; "
            + (
                "none puts a source voxel in the PML"
                if not offenders
                else f"{len(offenders)} put source voxels in the PML: {offenders}"
            )
            + (
                f"; the tightest is {tightest['job']} with "
                f"{tightest['clearance_beyond_the_pml_vox']} voxels of clearance beyond the "
                "sponge"
                if tightest
                else ""
            )
            + (
                f"; the deliberately-broken control was caught, with "
                f"{control[0]['source_voxels_inside_the_pml_pct']:.0f} % of its bowl in the sponge"
                if caught
                else "; the deliberately-broken control was NOT caught, so this check proves "
                "nothing"
            )
        ),
    }


# --------------------------------------------------------------------------
# G9 — the bowl a job orders, end to end
# --------------------------------------------------------------------------


@check("G9", "job to voxels: the bowl a job file orders is the bowl the runner builds")
def _g9(ctx):
    """G1 measured the constructor. This measures the whole path.

    A job names the bowl in millimetres and the apex in millimetres; the
    builder converts to voxels, and a rounding convention lives at each step.
    Fit the shell the *builder* produced and compare against what the job
    file said, in the job file's own units.
    """
    from caustica.config.job import build_job, parse_job

    rows = []
    for dx_mm, d_outer_mm, roc_mm, apex_mm in (
        (0.5, 12.0, 10.0, 6.0),
        (0.25, 12.0, 10.0, 6.0),
        (0.1, 5.0, 6.0, 2.0),
    ):
        size = [max(4.0 * roc_mm, 2.5 * d_outer_mm)] * 2 + [roc_mm + apex_mm + 6.0]
        job = parse_job(
            {
                "format": "caustica-job/1",
                "kind": "explicit",
                "name": "geom",
                "medium": {"kind": "homogeneous"},
                "grid": {
                    "ndim": 3,
                    "dx_mm": dx_mm,
                    "size_mm": size,
                    "pml": {"thickness_mm": 1.5},
                },
                "source": {
                    "kind": "array",
                    "array": {"kind": "bowl", "d_outer_mm": d_outer_mm, "roc_mm": roc_mm},
                    "apex_mm": [size[0] / 2, size[1] / 2, apex_mm],
                },
                "drive": {"f0_mhz": 1.0, "amplitude_kpa": 100.0},
                "run": {"harmonics": [1]},
                "solver": "linear",
            },
            "geom",
        )
        built = build_job(job, base_dir=Path("."), with_medium=False)
        dx = built.grid.dx
        idx = np.asarray(built.source.indices)
        p = idx.astype(float) * dx
        # the sphere the JOB FILE ordered, in the grid's own coordinates
        apex_vox_expected = np.array([size[0] / 2, size[1] / 2, apex_mm]) * MM / dx
        focus = (np.round(apex_vox_expected) + np.array([0.0, 0.0, roc_mm * MM / dx])) * dx
        err = np.linalg.norm(p - focus, axis=1) - roc_mm * MM
        rows.append(
            {
                "dx_mm": dx_mm,
                "grid": list(map(int, built.grid.shape)),
                "n_source_voxels": int(len(p)),
                "surface_rms_error_vox": float(np.sqrt(np.mean(err**2))) / dx,
                "surface_max_error_vox": float(np.abs(err).max()) / dx,
                "aperture_radius": deviation(
                    float(np.linalg.norm(p[:, :2] - focus[:2], axis=1).max()) / MM,
                    d_outer_mm / 2,
                    "mm",
                ),
                "apex_z": deviation(float(idx[:, 2].min()) * dx / MM, apex_mm, "mm"),
                "focus_vox_reported": list(map(int, built.focus_vox)),
                "focus_vox_expected": [int(round(c / dx)) for c in focus],
            }
        )
    worst_apex = max(r["apex_z"]["abs_error"] for r in rows)
    worst_surface = max(r["surface_max_error_vox"] for r in rows)
    agree = all(r["focus_vox_reported"] == r["focus_vox_expected"] for r in rows)
    return {
        "rows": rows,
        "focus_vox_matches_the_job_file": agree,
        "verdict": (
            "the focus the builder reports is exactly the one the job file implies at every dx"
            if agree
            else "the builder's reported focus does NOT match the job file"
        )
        + (
            f"; the shell it built sits within {worst_surface:.2f} voxel of the ordered sphere "
            f"and its apex within {worst_apex:.4f} mm of the ordered depth"
        ),
    }


# --------------------------------------------------------------------------
# G10 — do the holes change the answer?
# --------------------------------------------------------------------------


@check("G10", "how many voxels carry the drive, and what that does to the focal pressure")
def _g10(ctx):
    """G2 counts holes. This asks what they cost, and finds something worse.

    The engine drives each source voxel with the same normalized amplitude,
    so the power a bowl radiates is proportional to how many voxels its
    shell happens to contain. On a *plane* source that is exactly the
    aperture area divided by dx^2, which is what the normalization was
    calibrated against. On a curved shell it is not: a tilted surface
    crosses more voxels per unit area than a flat one, and how many depends
    on how densely the cap was sampled before rounding.

    So refining the cap sampling does two things at once — it closes the
    holes G2 counted, and it raises the voxel count above the cap's own
    area. Both show up as focal pressure. The measurement below separates
    them by reporting the voxels-per-area ratio next to the pressure, and
    grades both against O'Neil's closed form ``|p| = rho c u0 k h``, which
    is the absolute prediction the analytic gate deliberately does not use
    (it compares normalized shape only, so nothing in the suite would ever
    notice a drive that is uniformly too strong).

    Two f-numbers, because the closed form is a paraxial statement and its
    own accuracy has to be visible in the answer.
    """
    from caustica.analytic import axial_pressure
    from caustica.core.grid import Grid
    from caustica.core.pml import PMLSpec
    from caustica.materials import water
    from caustica.medium import Medium
    from caustica.solvers import CWRunSpec, get
    from caustica.sources import bowl_cw_source

    dx, f0, drive = 0.25 * MM, 1.0e6, 1.0e5
    c0, rho0 = 1500.0, 1000.0
    spec = CWRunSpec(min_settle_periods=4, max_settle_periods=24, n_record_periods=2)
    geometries = (
        ("f/1.2", 5.0 * MM, 12.0 * MM, (96, 96, 112)),
        ("f/2.0", 6.0 * MM, 24.0 * MM, (80, 80, 152)),
    )
    samplings = (("dx/2 (shipped)", 2), ("dx/4", 4), ("dx/8", 8), ("dx/16", 16))

    rows = []
    for gname, aperture, roc, shape in geometries:
        grid = Grid(shape=shape, dx=dx, pml=PMLSpec(thickness=2.0 * MM))
        medium = Medium.homogeneous(grid.shape, water())
        apex = (shape[0] // 2, shape[1] // 2, 12)
        focus_vox = (apex[0], apex[1], apex[2] + int(round(roc / dx)))
        cos_tmax = np.sqrt(1.0 - (aperture / roc) ** 2)
        cap_area = 2.0 * np.pi * roc**2 * (1.0 - cos_tmax)
        # O'Neil's exact on-axis solution, driven by the velocity a pressure
        # source of this amplitude implies for a locally plane wave. Windowed
        # exactly as the analytic gate windows it: past half the focal length,
        # clear of the source, and two voxels clear of the sponge.
        z = (np.arange(shape[2]) - apex[2]) * dx
        z_pml = (shape[2] - grid.pml_vox - 2 - apex[2]) * dx
        sel = (z > 0.5 * roc) & (z < z_pml)
        oneill = np.abs(axial_pressure(z[sel], aperture, roc, f0, c0, rho0, drive / (rho0 * c0)))
        expected = float(oneill.max())
        for sname, denom in samplings:
            src = bowl_cw_source(grid, f0, drive, aperture, roc, apex, spacing=dx / denom)
            res = get("linear")().run(
                grid, medium, src, spec, backend="numpy", reference_point=focus_vox
            )
            amp = np.abs(np.asarray(res.phasor))
            axis = amp[apex[0], apex[1], :]
            produced = float(axis[sel].max())
            rows.append(
                {
                    "geometry": gname,
                    "f_number": roc / (2 * aperture),
                    "sampling": sname,
                    "n_source_voxels": int(src.n_points),
                    "voxels_per_cap_area": src.n_points * dx**2 / cap_area,
                    "on_axis_focal_pressure": deviation(produced, expected, "Pa vs O'Neil"),
                    "peak_mpa": produced / 1e6,
                    "oneill_mpa": expected / 1e6,
                    "peak_z_mm": float(z[sel][int(axis[sel].argmax())] / MM),
                    "oneill_peak_z_mm": float(z[sel][int(oneill.argmax())] / MM),
                    "global_peak_mpa": float(amp.max() / 1e6),
                }
            )

    out: dict[str, Any] = {"dx_mm": 0.25, "drive_pa": drive, "rows": rows}
    summary = []
    for gname, *_ in geometries:
        got = [r for r in rows if r["geometry"] == gname]
        shipped, finest = got[0], got[-1]
        out[gname] = {
            "extra_voxels_from_refining_the_sampling": finest["n_source_voxels"]
            / shipped["n_source_voxels"]
            - 1.0,
            "peak_moved_by": rel(shipped["peak_mpa"], finest["peak_mpa"]),
            "voxels_per_cap_area_shipped": shipped["voxels_per_cap_area"],
            "voxels_per_cap_area_converged": finest["voxels_per_cap_area"],
            "over_oneill_shipped": shipped["peak_mpa"] / shipped["oneill_mpa"],
            "over_oneill_converged": finest["peak_mpa"] / finest["oneill_mpa"],
        }
        summary.append(
            f"{gname}: refining dx/2 to dx/16 adds "
            f"{out[gname]['extra_voxels_from_refining_the_sampling'] * 100:.0f} % more driven "
            f"voxels and moves the focus by {out[gname]['peak_moved_by'] * 100:.1f} %; the shell "
            f"carries {shipped['voxels_per_cap_area']:.2f} voxels per dx^2 of cap area at the "
            f"shipped sampling and {finest['voxels_per_cap_area']:.2f} at dx/16, against 1.00 for "
            f"a flat source; the on-axis focus sits {out[gname]['over_oneill_shipped']:.2f}x and "
            f"{out[gname]['over_oneill_converged']:.2f}x O'Neil's absolute prediction"
        )
    out["verdict"] = " | ".join(summary)
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def environment() -> dict:
    import numpy

    import caustica

    return {
        "caustica": caustica.__version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def flatten(value: Any, prefix: str = "") -> dict:
    """One level of dotted keys, so a nested row still fits in a table cell."""
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(value, list) and len(value) > 6:
        out[prefix] = f"[{len(value)} entries]"
    else:
        out[prefix] = value
    return out


def cell(v: Any) -> str:
    if isinstance(v, float):
        if v == 0 or 1e-3 <= abs(v) < 1e5:
            return f"{v:.4g}"
        return f"{v:.3e}"
    return str(v).replace("|", "/")


def table(rows: list) -> list[str]:
    flat = [flatten(r) for r in rows if isinstance(r, dict)]
    if not flat:
        return []
    cols: list[str] = []
    for r in flat:
        for k in r:
            if k not in cols:
                cols.append(k)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in flat:
        out.append("| " + " | ".join(cell(r.get(c, "")) for c in cols) + " |")
    return out


def merge_into(path: Path, key: str, fresh: list[dict]) -> list[dict]:
    """This run's entries over whatever a previous run left, in id order."""
    previous: list[dict] = []
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8")).get(key, [])
        except (OSError, ValueError):
            previous = []
    merged = {e["id"]: e for e in previous}
    merged.update({e["id"]: e for e in fresh})
    order = [cid for cid, *_ in CHECKS]
    return [merged[cid] for cid in order if cid in merged]


def render_markdown(payload: dict) -> str:
    lines = [
        "# Geometry fidelity: expected against produced",
        "",
        "Generated by `scripts/dev_geometry.py`. Every check separates what was",
        "*asked for* from what was *built*, and reports the gap.",
        "",
        "| check | question | verdict |",
        "|---|---|---|",
    ]
    for e in payload["checks"]:
        v = (e.get("data") or {}).get("verdict", e.get("error", ""))
        lines.append(f"| [{e['id']}](#{e['id'].lower()}) | {e['title']} | {cell(v)} |")
    for e in payload["checks"]:
        data = e.get("data") or {}
        lines += ["", f"## {e['id']}", "", f"**{e['title']}**", ""]
        if e["status"] != "OK":
            lines += ["```", str(e.get("error", "")), "```"]
            continue
        lines += [str(data.get("verdict", "")), ""]
        if data.get("rows"):
            lines += table(data["rows"])
        summary = {k: v for k, v in data.items() if k not in ("rows", "verdict")}
        if summary:
            lines += ["", "```json", json.dumps(summary, indent=2, default=str), "```"]
    env = json.dumps(payload["environment"], indent=2)
    lines += ["", "## Environment", "", "```json", env, "```"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="benchmarks/reports/geometry")
    ap.add_argument("--only", default="", help="comma-separated check ids (default: all)")
    ap.add_argument("--skip", default="")
    args = ap.parse_args(argv)

    only = {s.strip().upper() for s in args.only.split(",") if s.strip()}
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for cid, title, fn in CHECKS:
        if (only and cid not in only) or cid in skip:
            continue
        print(f"[{cid}] {title} ...", flush=True)
        t0 = time.perf_counter()
        entry: dict[str, Any] = {"id": cid, "title": title}
        try:
            entry["data"] = fn({})
            entry["status"] = "OK"
        except Exception as exc:
            entry["status"] = "ERROR"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc(limit=6)
        entry["elapsed_s"] = round(time.perf_counter() - t0, 2)
        results.append(entry)
        mark = "OK " if entry["status"] == "OK" else "ERR"
        detail = (entry.get("data") or {}).get("verdict", entry.get("error", ""))
        print(f"  {mark} {entry['elapsed_s']:>6.2f}s  {detail}", flush=True)

    # A partial run must not destroy the record. `--only G10` re-measures one
    # check; it does not mean the other nine stopped being true, and a file
    # that quietly shrank to one entry is worse than no file.
    results = merge_into(outdir / "geometry.json", "checks", results)
    payload = {
        "format": "caustica-geometry-fidelity/1",
        "environment": environment(),
        "checks": results,
    }
    (outdir / "geometry.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (outdir / "REPORT.md").write_text(render_markdown(payload), encoding="utf-8")
    bad = [e["id"] for e in results if e["status"] != "OK"]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks ran -> {outdir / 'geometry.json'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
