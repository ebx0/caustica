#!/usr/bin/env python3
"""Measure what the project has been ASSUMING — a one-shot validation campaign.

``dev_validate.py`` probes open questions and ``master_test.py`` gates the
milestones. This script does the third job: it takes claims the library rests on
and puts a number against each. Most of them concern the change of 2026-08-24,
which altered the spectral derivative operator itself and therefore every number
the library produces:

    the collocated first derivative used to carry a live Nyquist wavenumber on
    every even-length axis, which made ``deriv * rfftn(p)`` an illegal input to
    a real-to-complex inverse transform

Zeroing that bin is defensible on paper. What was never measured is what it does
to the answers: whether accuracy against the analytic references moved, by how
much the field changes, whether odd grids are untouched as predicted, which grid
sizes cuFFT was mishandling before, and — the only genuinely independent check
available — whether the change moved this library toward or away from k-Wave, a
staggered-grid implementation that never had the defect because its half-sample
shift rotates the Nyquist factor onto the real axis.

Every experiment runs the CURRENT operator and, where the comparison is the
point, the PRE-FIX operator restored by :func:`legacy_operator`, so both are
measured under identical conditions in one process.

Usage::

    python scripts/dev_campaign.py --list
    python scripts/dev_campaign.py --experiments E1,E2
    python scripts/dev_campaign.py --all --out benchmarks/reports/campaign

Results stream into ``campaign.json`` as each experiment finishes, so a run that
is interrupted still leaves everything it had measured.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "caustica-dev-campaign/1"
C0, RHO0, ALPHA0 = 1500.0, 1000.0, 0.025  # water: m/s, kg/m^3, Np/m
BETA_WATER = 3.5

EXPERIMENTS: list[tuple[str, str, bool, Callable]] = []


def experiment(eid: str, title: str, *, gpu: bool = False):
    def deco(fn):
        EXPERIMENTS.append((eid, title, gpu, fn))
        return fn

    return deco


# --------------------------------------------------------------------------
# the pre-fix operator, restored
# --------------------------------------------------------------------------


@contextlib.contextmanager
def legacy_operator():
    """Put the pre-2026-08-24 derivative factors back for the duration.

    The old code multiplied by ``i k kappa`` for every bin, Nyquist included.
    Restoring it here — rather than checking out the old commit — is what lets a
    single process measure both operators against the same reference, on the
    same machine, in the same float32 arithmetic.
    """
    from caustica.solvers.kspace import operators as ops

    original = ops.spectral_derivative_factors

    def pre_fix(ks, kappa, shape, xp):
        return [(1j * k * kappa).astype(xp.complex64) for k in ks]

    ops.spectral_derivative_factors = pre_fix
    try:
        yield
    finally:
        ops.spectral_derivative_factors = original


# --------------------------------------------------------------------------
# scenario helpers
# --------------------------------------------------------------------------


def water_medium(shape, *, beta: float = 0.0, alpha: float = ALPHA0):
    from caustica import Medium

    ones = np.ones(shape, np.float32)
    return Medium(alpha=ones * alpha, rho=ones * RHO0, c=ones * C0, beta=ones * float(beta))


def layered_medium(shape, *, beta: float = 0.0):
    """Water, then a fat-like slab, then a muscle-like half space."""
    from caustica import Medium

    c = np.full(shape, C0, np.float32)
    rho = np.full(shape, RHO0, np.float32)
    alpha = np.full(shape, ALPHA0, np.float32)
    z = shape[-1]
    lo, hi = int(0.40 * z), int(0.58 * z)
    c[..., lo:hi], rho[..., lo:hi], alpha[..., lo:hi] = 1450.0, 950.0, 6.0
    c[..., hi:], rho[..., hi:], alpha[..., hi:] = 1580.0, 1050.0, 9.0
    return Medium(alpha=alpha, rho=rho, c=c, beta=np.full(shape, float(beta), np.float32))


def make_grid(shape, dx, pml_vox=10):
    from caustica import Grid, PMLSpec

    return Grid(shape, dx, pml=PMLSpec(thickness=pml_vox * dx))


def run_spec(**kw):
    from caustica.solvers import CWRunSpec

    base = dict(min_settle_periods=4, max_settle_periods=14, n_record_periods=2)
    base.update(kw)
    return CWRunSpec(**base)


def solver(name: str):
    """``caustica.solvers.get`` hands back the CLASS; the runs want an instance."""
    from caustica.solvers import get

    obj = get(name)
    return obj() if isinstance(obj, type) else obj


def smooth_size(n: int) -> int:
    from caustica.solvers.kspace import operators as ops

    return ops.optimal_fft_size(int(n))


def grid_for_bowl(aperture, roc, dx, *, pml_vox=10, margin_vox=6, z_reach=1.7):
    """A grid the bowl actually FITS in, sized from the transducer outwards.

    Getting this wrong is quiet and expensive: a rim that lands inside the
    absorbing band is driven and damped at once, and the run then measures the
    sponge instead of the physics. An early pass of this campaign lost two
    experiments that way — a 16 mm bowl inside a 13 mm interior, reported as a
    26% "PML sensitivity" that was really a clipped source.
    """
    transverse = smooth_size(int(np.ceil(2 * aperture / dx)) + 2 * (pml_vox + margin_vox))
    axial = smooth_size(int(np.ceil(z_reach * roc / dx)) + 2 * (pml_vox + margin_vox))
    return (transverse, transverse, axial)


def bowl_scene(
    shape,
    dx,
    *,
    f0=1e6,
    amp=1.0e5,
    aperture=0.006,
    roc=0.012,
    medium=None,
    pml_vox=10,
    margin_vox=3,
):
    """A focused bowl on the beam axis, plus the focus voxel to settle against.

    The aperture is shrunk when it would not fit the grid it was handed, because
    several experiments vary the SHAPE and every rung of those must stay
    physically valid; what was actually used comes back in the meta dict.

    The reference point matters more than it looks. Without one the engine takes
    time of flight to the farthest CORNER of the domain, which at 256^3 and
    0.3 mm is 89 acoustic periods before settling can even begin — minutes per
    run, for a wave that reached the focus in eleven. Every scenario here is
    about what happens at or before the focus, so that is what they settle to.
    """
    from caustica.sources import bowl_cw_source

    g = make_grid(shape, dx, pml_vox)
    apex_z = pml_vox + 2
    room = (min(shape[0], shape[1]) - 2 * (pml_vox + margin_vox)) * dx / 2.0
    used_aperture = max(3 * dx, min(aperture, room))
    used_roc = max(roc, 1.15 * used_aperture)
    src = bowl_cw_source(
        g,
        f0=f0,
        amplitude=amp,
        aperture_radius=used_aperture,
        roc=used_roc,
        apex_vox=(shape[0] // 2, shape[1] // 2, apex_z),
    )
    med = water_medium(shape) if medium is None else medium
    focus_z = min(shape[-1] - 1 - pml_vox, apex_z + int(round(used_roc / dx)))
    meta = {
        "aperture_mm": used_aperture * 1e3,
        "roc_mm": used_roc * 1e3,
        "clipped": bool(used_aperture < aperture - 1e-12),
        "f_number": used_roc / (2 * used_aperture),
        "n_source_voxels": int(len(src.indices)),
    }
    return g, med, src, apex_z, (shape[0] // 2, shape[1] // 2, focus_z), meta


def axial_profile(phasor, shape, apex_z, dx):
    """|P| along the beam axis, with distance measured from the bowl apex."""
    prof = np.abs(np.asarray(phasor)[shape[0] // 2, shape[1] // 2, :]).astype(np.float64)
    return (np.arange(shape[2]) - apex_z) * dx, prof


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den else float("nan")


def width_6db(z, prof):
    """Axial extent where |P| stays above half the peak, around the peak."""
    i = int(np.argmax(prof))
    half = prof[i] / 2.0
    lo, hi = i, i
    while lo > 0 and prof[lo] >= half:
        lo -= 1
    while hi < len(prof) - 1 and prof[hi] >= half:
        hi += 1
    return float(z[hi] - z[lo])


def field_diff(a, b):
    """(max relative deviation, relative L2) between two fields."""
    a, b = np.asarray(a), np.asarray(b)
    scale = float(np.abs(a).max()) or 1.0
    return (
        float(np.abs(a - b).max()) / scale,
        float(np.linalg.norm(a - b) / (np.linalg.norm(a) or 1.0)),
    )


def oneill_axial(z, aperture, roc, f0):
    from caustica.analytic import axial_pressure

    return np.abs(
        axial_pressure(z, aperture_radius=aperture, roc=roc, f0=f0, c0=C0, rho0=RHO0, u0=1.0)
    )


def profile_metrics(field, shape, apex_z, dx, aperture, roc, f0):
    z, prof = axial_profile(field, shape, apex_z, dx)
    keep = (z > 0.35 * roc) & (z < 1.7 * roc)
    ana = oneill_axial(z[keep], aperture, roc, f0)
    p = prof[keep]
    r = pearson(p / p.max(), ana / ana.max())
    return {
        "correlation_r": r,
        "one_minus_r": 1.0 - r,
        "peak_z_mm": float(z[keep][int(np.argmax(p))] * 1e3),
        "peak_z_error_mm": float((z[keep][int(np.argmax(p))] - z[keep][int(np.argmax(ana))]) * 1e3),
        "width_6db_mm": width_6db(z[keep], p) * 1e3,
        "width_6db_error_pct": 100.0
        * (width_6db(z[keep], p) - width_6db(z[keep], ana))
        / width_6db(z[keep], ana),
        "peak_pa": float(p.max()),
    }


def gpu_available() -> bool:
    try:
        import cupy

        cupy.cuda.runtime.getDeviceProperties(0)
        return True
    except Exception:
        return False


def sh(args, *, cwd=None, timeout=1800):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        p = subprocess.run(
            args,
            cwd=None if cwd is None else str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout} s"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# --------------------------------------------------------------------------
# E1 — the accuracy question the fix has to answer
# --------------------------------------------------------------------------


@experiment("E1", "focused bowl vs O'Neil: fixed operator against the pre-fix one")
def _e1(ctx):
    f0, aperture, roc, dx = 1.0e6, 0.006, 0.012, 1.5e-4
    shape = grid_for_bowl(aperture, roc, dx)
    out: dict[str, Any] = {
        "scenario": {
            "shape": list(shape),
            "dx_mm": dx * 1e3,
            "f0_MHz": f0 / 1e6,
            "ppw": C0 / f0 / dx,
            "f_number": roc / (2 * aperture),
        }
    }
    fields, apex_z = {}, None
    for label, cm in (("fixed", contextlib.nullcontext()), ("legacy", legacy_operator())):
        g, med, src, apex_z, ref, meta = bowl_scene(shape, dx, f0=f0, aperture=aperture, roc=roc)
        out["scenario"]["source"] = meta
        with cm:
            res = solver("linear").run(
                g, med, src, run_spec(), backend="numpy", reference_point=ref
            )
        fields[label] = np.asarray(res.phasor)

    for label in ("fixed", "legacy"):
        out[label] = profile_metrics(fields[label], shape, apex_z, dx, aperture, roc, f0)
    lin, l2 = field_diff(fields["fixed"], fields["legacy"])
    out["fixed_vs_legacy"] = {"max_rel": lin, "rel_l2": l2}
    closer = out["fixed"]["one_minus_r"] <= out["legacy"]["one_minus_r"]
    out["fixed_is_closer_to_oneill"] = bool(closer)
    out["verdict"] = (
        f"correlation with O'Neil: fixed {out['fixed']['correlation_r']:.6f}, "
        f"legacy {out['legacy']['correlation_r']:.6f}; -6 dB width error "
        f"{out['fixed']['width_6db_error_pct']:+.2f}% vs "
        f"{out['legacy']['width_6db_error_pct']:+.2f}%; the fields differ by "
        f"{lin * 100:.3f}%; fixed is {'closer' if closer else 'FURTHER'}"
    )
    return out


# --------------------------------------------------------------------------
# E2 — how far does the change reach, shape by shape?
# --------------------------------------------------------------------------


@experiment("E2", "blast radius: how much the field moves, over a shape matrix")
def _e2(ctx):
    """Odd PADDED axes have no Nyquist bin, so they must be bit-identical.

    The padding is the point. The engine rounds every axis up to a 2/3/5-smooth
    size, and almost every such size is even — 81 and 135 are rare exceptions —
    so a grid the user thinks of as odd is usually solved on an even one.
    """
    from caustica.solvers.kspace import operators as ops

    cases = [
        ("cubic-64", (64, 64, 64)),
        ("cubic-96", (96, 96, 96)),
        ("cubic-81 (3^4, stays odd)", (81, 81, 81)),
        ("cubic-135 (odd smooth)", (135, 135, 135)),
        ("cubic-63 -> pads to 64", (63, 63, 63)),
        ("beam-axis-odd", (64, 64, 81)),
        ("beam-axis-even", (81, 81, 64)),
        ("padded-100", (100, 100, 100)),
        ("anisotropic", (64, 80, 96)),
    ]
    dx = 2.0e-4
    rows = []
    for name, shape in cases:
        padded = ops.pad_shape(shape)
        g, med, src, _apex, ref, meta = bowl_scene(shape, dx, aperture=0.006, roc=0.012)
        spec = run_spec(min_settle_periods=3, max_settle_periods=8, n_record_periods=1)
        kw = dict(backend="numpy", reference_point=ref)
        a = np.asarray(solver("linear").run(g, med, src, spec, **kw).phasor)
        with legacy_operator():
            b = np.asarray(solver("linear").run(g, med, src, spec, **kw).phasor)
        lin, l2 = field_diff(a, b)
        parity = ["even" if n % 2 == 0 else "odd" for n in padded]
        rows.append(
            {
                "case": name,
                "shape": list(shape),
                "padded": list(padded),
                "padded_parity": parity,
                "even_axes": sum(1 for p in parity if p == "even"),
                "aperture_mm": meta["aperture_mm"],
                "clipped": meta["clipped"],
                "max_rel": lin,
                "rel_l2": l2,
                "identical": lin == 0.0,
                "peak_kpa": float(np.abs(a).max() / 1e3),
            }
        )
    all_odd = [r for r in rows if r["even_axes"] == 0]
    any_even = [r for r in rows if r["even_axes"] > 0]
    worst = max(any_even, key=lambda r: r["max_rel"])
    return {
        "rows": rows,
        "all_odd_padded_are_identical": all(r["identical"] for r in all_odd),
        "n_all_odd_cases": len(all_odd),
        "even_cases_that_did_not_move": [r["case"] for r in any_even if r["identical"]],
        "worst": {"case": worst["case"], "max_rel": worst["max_rel"]},
        "verdict": (
            f"{len(all_odd)} all-odd-padded shapes bit-identical: "
            f"{all(r['identical'] for r in all_odd)}; every shape with an even padded axis "
            f"moved, worst {worst['max_rel'] * 100:.2f}% ({worst['case']})"
        ),
    }


# --------------------------------------------------------------------------
# E3 — plane wave in 1-D
# --------------------------------------------------------------------------


@experiment("E3", "plane wave in 1-D: phase speed and absorption, fixed vs pre-fix")
def _e3(ctx):
    """Dispersion and absorption where nothing else can contaminate them.

    A 1-D grid has no diffraction to confuse a phase fit, and its single axis IS
    the real-to-complex axis, so the removed bin sits in the most direct
    position to matter. Both parities run; an odd grid has no Nyquist bin.
    """
    from caustica.sources import plane_cw_source

    dx, f0, alpha = 2.0e-4, 1.0e6, 10.0
    k_true = 2 * np.pi * f0 / C0
    rows, gaps = [], {}
    for name, n in (("even-512", 512), ("odd-405", 405)):
        shape = (n,)
        g = make_grid(shape, dx, pml_vox=16)
        med = water_medium(shape, alpha=alpha)
        src = plane_cw_source(g, f0=f0, amplitude=1.0e5, axis=0)
        fields = {}
        for label, cm in (("fixed", contextlib.nullcontext()), ("legacy", legacy_operator())):
            with cm:
                res = solver("linear").run(
                    g,
                    med,
                    src,
                    run_spec(min_settle_periods=6, max_settle_periods=30),
                    backend="numpy",
                    reference_point=(n - 20,),
                )
            ph = np.asarray(res.phasor).ravel()
            fields[label] = ph
            lo, hi = int(0.30 * n), int(0.80 * n)
            z = np.arange(lo, hi) * dx
            amp = np.abs(ph[lo:hi])
            slope = np.polyfit(z, np.log(amp), 1)[0]
            k_fit = float(abs(np.polyfit(z, np.unwrap(np.angle(ph[lo:hi])), 1)[0]))
            rows.append(
                {
                    "grid": name,
                    "padded": smooth_size(n),
                    "operator": label,
                    "alpha_measured_np_m": float(-slope),
                    "alpha_error_pct": 100.0 * (-slope - alpha) / alpha,
                    "k_measured_rad_m": k_fit,
                    "k_error_pct": 100.0 * (k_fit - k_true) / k_true,
                    "c_implied_m_s": float(2 * np.pi * f0 / k_fit) if k_fit else None,
                }
            )
        gaps[name] = field_diff(fields["fixed"], fields["legacy"])

    def worst(op, key):
        return max(abs(r[key]) for r in rows if r["operator"] == op)

    return {
        "rows": rows,
        "fixed_vs_legacy": {k: {"max_rel": v[0], "rel_l2": v[1]} for k, v in gaps.items()},
        "k_expected_rad_m": k_true,
        "alpha_configured_np_m": alpha,
        "odd_grid_untouched": gaps["odd-405"][0] == 0.0,
        "verdict": (
            f"phase speed error: fixed {worst('fixed', 'k_error_pct'):.4f}%, "
            f"legacy {worst('legacy', 'k_error_pct'):.4f}%; absorption: fixed "
            f"{worst('fixed', 'alpha_error_pct'):.3f}%, legacy "
            f"{worst('legacy', 'alpha_error_pct'):.3f}%"
        ),
    }


# --------------------------------------------------------------------------
# E4 — the independent referee: k-Wave
# --------------------------------------------------------------------------


@experiment("E4", "k-Wave cross-check: does the fix move us toward an independent code?")
def _e4(ctx):
    """k-Wave is staggered, so its Nyquist factor is real and it never had this
    defect. That makes it the one referee available that can say whether zeroing
    the bin moved this library toward or away from an implementation that got
    the boundary case right by construction."""
    from caustica.solvers import available

    if "kwave" not in available():
        return {"skipped": "the kwave solver is not registered"}
    try:
        import kwave  # noqa: F401
    except Exception as exc:
        return {"skipped": f"k-wave-python not importable: {type(exc).__name__}: {exc}"}

    f0, aperture, roc, dx = 1.0e6, 0.006, 0.012, 2.0e-4
    shape = grid_for_bowl(aperture, roc, dx)
    rows = []
    for name, med_kind in (("water", "water"), ("layered", "layered")):
        med = layered_medium(shape) if med_kind == "layered" else water_medium(shape)
        g, _m, src, apex_z, ref, meta = bowl_scene(
            shape, dx, f0=f0, aperture=aperture, roc=roc, medium=med
        )
        spec = run_spec()
        entry: dict[str, Any] = {"scenario": name, "shape": list(shape), "source": meta}
        try:
            # The external engine drives itself and refuses a `backend` option.
            gold = np.asarray(solver("kwave").run(g, med, src, spec, reference_point=ref).phasor)
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
            rows.append(entry)
            continue
        entry["kwave_peak_kpa"] = float(np.abs(gold).max() / 1e3)
        for label, cm in (("fixed", contextlib.nullcontext()), ("legacy", legacy_operator())):
            with cm:
                mine = np.asarray(
                    solver("linear")
                    .run(g, med, src, spec, backend="numpy", reference_point=ref)
                    .phasor
                )
            a, b = np.abs(mine), np.abs(gold)
            _, l2 = field_diff(b, a)
            zc, pc = axial_profile(mine, shape, apex_z, dx)
            _, pr = axial_profile(gold, shape, apex_z, dx)
            keep = (zc > 0.35 * roc) & (zc < 1.7 * roc)
            entry[label] = {
                "rel_l2_vs_kwave": l2,
                "pearson_r": pearson(a.ravel(), b.ravel()),
                "peak_ratio": float(a.max() / (b.max() or 1.0)),
                "axial_r": pearson(pc[keep], pr[keep]),
                "peak_z_diff_mm": float(
                    (zc[keep][int(np.argmax(pc[keep]))] - zc[keep][int(np.argmax(pr[keep]))]) * 1e3
                ),
            }
        if "fixed" in entry and "legacy" in entry:
            better = entry["fixed"]["rel_l2_vs_kwave"] < entry["legacy"]["rel_l2_vs_kwave"]
            entry["fix_moved_us"] = "closer to k-Wave" if better else "further from k-Wave"
            entry["l2_improvement_pct"] = 100.0 * (
                1.0
                - entry["fixed"]["rel_l2_vs_kwave"] / (entry["legacy"]["rel_l2_vs_kwave"] or 1.0)
            )
        rows.append(entry)
    moved = [r.get("fix_moved_us") for r in rows if r.get("fix_moved_us")]
    return {
        "rows": rows,
        "all_closer": bool(moved) and all(v == "closer to k-Wave" for v in moved),
        "verdict": "; ".join(
            f"{r['scenario']}: {r.get('fix_moved_us', r.get('error', '?'))}"
            + (
                f" ({r['l2_improvement_pct']:+.1f}% on rel L2)"
                if r.get("l2_improvement_pct") is not None
                else ""
            )
            for r in rows
        ),
    }


# --------------------------------------------------------------------------
# E5 — which sizes was cuFFT actually mishandling?
# --------------------------------------------------------------------------


@experiment("E5", "cuFFT failure map: which grid sizes the pre-fix operator broke", gpu=True)
def _e5(ctx):
    """Only 256^3 was ever caught. Which sizes were actually broken?

    The question needs no CPU reference, and that is what makes it affordable:
    the pre-fix operator's failure mode ON THE GPU *is* divergence, so running
    each size on cupy with the old factors and asking whether the field stays
    finite answers it directly. A numpy leg at every size would have cost hours
    -- a single 288^3 step costs seconds on this laptop's CPU and milliseconds
    on its GPU -- and would only have re-derived what the NaN already says.
    Two small sizes keep a CPU anchor so the comparison is not GPU-only.
    """
    from caustica.solvers.base import SolverDivergedError

    sizes = [96, 128, 144, 160, 162, 180, 192, 200, 216, 240, 243, 250, 256, 270, 288, 320]
    anchors = {96, 128}
    dx = 3.0e-4
    rows = []
    for n in sizes:
        shape = (n, n, n)
        if n**3 * 4 * 14 > 5.6 * 2**30:
            rows.append({"n": n, "skipped": "would not fit this card's VRAM"})
            continue
        entry: dict[str, Any] = {
            "n": n,
            "factors": factorize(n),
            "power_of_two": (n & (n - 1)) == 0,
        }
        backends = ("cupy", "numpy") if n in anchors else ("cupy",)
        for label, cm in (("legacy", legacy_operator()), ("fixed", contextlib.nullcontext())):
            g, _m, src, apex_z, _ref, _meta = bowl_scene(shape, dx, aperture=0.008, roc=0.016)
            med = water_medium(shape, beta=BETA_WATER)
            spec = run_spec(min_settle_periods=1, max_settle_periods=1, n_record_periods=1)
            near = (n // 2, n // 2, min(n - 1, apex_z + 2))
            side: dict[str, Any] = {}
            for backend in backends:
                try:
                    with cm:
                        res = solver("westervelt").run(
                            g, med, src, spec, backend=backend, reference_point=near
                        )
                    side[backend] = float(np.abs(np.asarray(res.phasor)).max())
                except SolverDivergedError:
                    side[backend] = float("nan")
                except Exception as exc:
                    side[backend] = None
                    side[backend + "_error"] = type(exc).__name__
            cu = side.get("cupy")
            np_ = side.get("numpy")
            entry[label] = {
                "cupy_peak_pa": cu,
                "numpy_peak_pa": np_,
                "diverged": bool(cu is not None and not np.isfinite(cu)),
                "backend_gap": (
                    abs(cu - np_) / np_
                    if (cu and np_ and np.isfinite(cu) and np.isfinite(np_))
                    else None
                ),
            }
        rows.append(entry)
    ran = [r for r in rows if "legacy" in r]
    broke = [r["n"] for r in ran if r["legacy"]["diverged"]]
    still = [r["n"] for r in ran if r["fixed"]["diverged"]]
    gaps = [
        (r["n"], r["legacy"]["backend_gap"])
        for r in ran
        if r["legacy"].get("backend_gap") is not None
    ]
    return {
        "rows": rows,
        "sizes_tested": [r["n"] for r in ran],
        "sizes_diverging_before_the_fix": broke,
        "sizes_diverging_after_the_fix": still,
        "cpu_anchor_backend_gaps_before": gaps,
        "verdict": (
            f"with the pre-fix operator {len(broke)} of {len(ran)} sizes diverged on this "
            f"GPU ({broke}); with the fix in place {len(still)} do ({still})"
        ),
    }


def factorize(n: int) -> str:
    out, m = [], n
    for p in (2, 3, 5, 7, 11, 13):
        while m % p == 0:
            out.append(p)
            m //= p
    if m > 1:
        out.append(m)
    return "*".join(str(x) for x in out)


# --------------------------------------------------------------------------
# E6 — the parity matrix nobody ran
# --------------------------------------------------------------------------


@experiment("E6", "numpy/cupy parity across rank, solver, shape, drive and medium", gpu=True)
def _e6(ctx):
    from caustica.sources import bowl_cw_source, plane_cw_source

    rows = []

    def compare(name, run, extra=None):
        try:
            a, b = run("numpy"), run("cupy")
        except Exception as exc:
            rows.append({"case": name, "error": f"{type(exc).__name__}: {exc}"[:200]})
            return
        lin, l2 = field_diff(a, b)
        rows.append(
            {
                "case": name,
                "max_rel": lin,
                "rel_l2": l2,
                "peak_pa": float(np.abs(a).max()),
                "pass": bool(l2 < 1e-4),
                **(extra or {}),
            }
        )

    for shape in [(256,), (96, 128), (64, 64, 80)]:
        g = make_grid(shape, 3.0e-4, pml_vox=10)
        med = water_medium(shape)
        src = plane_cw_source(g, f0=1e6, amplitude=1e5, axis=len(shape) - 1)
        spec = run_spec(min_settle_periods=2, max_settle_periods=8, n_record_periods=1)
        ref = tuple(n - 12 if i == len(shape) - 1 else n // 2 for i, n in enumerate(shape))
        compare(
            f"rank-{len(shape)}D",
            lambda bk, g=g, med=med, src=src, spec=spec, ref=ref: np.asarray(
                solver("linear").run(g, med, src, spec, backend=bk, reference_point=ref).phasor
            ),
        )

    dx = 2.5e-4
    shape = grid_for_bowl(0.006, 0.012, dx)
    g, _m, src0, apex_z, ref3, _meta = bowl_scene(shape, dx, aperture=0.006, roc=0.012)
    spec = run_spec(min_settle_periods=3, max_settle_periods=10, n_record_periods=2)
    for amp_kpa in (1.0, 200.0, 1500.0):
        src = bowl_cw_source(
            g,
            f0=1e6,
            amplitude=amp_kpa * 1e3,
            aperture_radius=0.006,
            roc=0.012,
            apex_vox=(shape[0] // 2, shape[1] // 2, apex_z),
        )
        med = water_medium(shape, beta=BETA_WATER)
        for h in (1, 2, 3):
            compare(
                f"westervelt-{amp_kpa:g}kPa-h{h}",
                lambda bk, med=med, src=src, h=h: np.asarray(
                    solver("westervelt")
                    .run(g, med, src, spec, backend=bk, harmonics=(1, 2, 3), reference_point=ref3)
                    .phasors[h]
                ),
                {"drive_kpa": amp_kpa, "harmonic": h},
            )

    med_l = layered_medium(shape, beta=BETA_WATER)
    compare(
        "layered-westervelt",
        lambda bk, med=med_l: np.asarray(
            solver("westervelt").run(g, med, src0, spec, backend=bk, reference_point=ref3).phasor
        ),
    )

    for name, sh_ in (("odd-padded-81", (81, 81, 81)), ("mixed-64-81", (64, 81, 96))):
        gg, mm, ss, _a, rr, _mt = bowl_scene(sh_, 2.5e-4, aperture=0.005, roc=0.010)
        sp = run_spec(min_settle_periods=2, max_settle_periods=8, n_record_periods=1)
        compare(
            name,
            lambda bk, gg=gg, mm=mm, ss=ss, sp=sp, rr=rr: np.asarray(
                solver("linear").run(gg, mm, ss, sp, backend=bk, reference_point=rr).phasor
            ),
        )

    bad = [r["case"] for r in rows if r.get("pass") is False or "error" in r]
    good = [r for r in rows if "rel_l2" in r]
    worst = max(good, key=lambda r: r["rel_l2"], default=None)
    return {
        "rows": rows,
        "failing": bad,
        "verdict": (
            f"{len(good) - len(bad)}/{len(rows)} combinations agree to rel L2 < 1e-4"
            + (f"; worst {worst['case']} at {worst['rel_l2']:.2e}" if worst else "")
        ),
    }


# --------------------------------------------------------------------------
# E7 — the time hypothesis, measured directly
# --------------------------------------------------------------------------


@experiment("E7", "warmup: a fresh process pays what a warm one never sees", gpu=True)
def _e7(ctx):
    """The standing explanation for time is that calibration probes run inside
    a process that has ALREADY paid the CUDA context, module-load and plan costs,
    so the warmup they measure is not the warmup a real run pays."""
    rows = []
    for shape in [(128, 128, 128), (192, 192, 192), (256, 256, 256)]:
        warm = warmup_here(shape)
        fresh = warmup_fresh_process(shape)
        if warm is None or fresh is None:
            rows.append({"shape": list(shape), "error": "probe failed"})
            continue
        rows.append(
            {
                "shape": list(shape),
                "p_elems": int(np.prod(shape)),
                "warm_process_s": round(warm, 3),
                "fresh_process_s": round(fresh, 3),
                "ratio": round(fresh / warm, 2) if warm > 1e-6 else None,
                "missing_s": round(fresh - warm, 3),
            }
        )
    ok = [r for r in rows if r.get("ratio")]
    return {
        "rows": rows,
        "supports_warm_process_hypothesis": bool(ok) and all(r["ratio"] > 1.5 for r in ok),
        "verdict": (
            f"a fresh process pays {min(r['ratio'] for r in ok):.1f}-"
            f"{max(r['ratio'] for r in ok):.1f}x the warmup a warm one measures "
            f"({min(r['missing_s'] for r in ok):.1f}-{max(r['missing_s'] for r in ok):.1f} s "
            f"the calibration never sees)"
            if ok
            else "no shape produced both numbers"
        ),
    }


def warmup_here(shape):
    try:
        import cupy

        a = cupy.zeros(shape, cupy.float32)
        axes = tuple(range(len(shape)))
        cupy.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        cupy.fft.irfftn(cupy.fft.rfftn(a), s=shape, axes=axes)
        cupy.cuda.runtime.deviceSynchronize()
        cold = time.perf_counter() - t0
        t1 = time.perf_counter()
        for _ in range(3):
            cupy.fft.irfftn(cupy.fft.rfftn(a), s=shape, axes=axes)
        cupy.cuda.runtime.deviceSynchronize()
        steady = (time.perf_counter() - t1) / 3.0
        del a
        cupy.get_default_memory_pool().free_all_blocks()
        return max(0.0, cold - steady)
    except Exception:
        return None


_FRESH = """
import time, sys
t_start = time.perf_counter()
import cupy
shape = tuple(int(x) for x in sys.argv[1].split(","))
axes = tuple(range(len(shape)))
a = cupy.zeros(shape, cupy.float32)
cupy.cuda.runtime.deviceSynchronize()
t0 = time.perf_counter()
cupy.fft.irfftn(cupy.fft.rfftn(a), s=shape, axes=axes)
cupy.cuda.runtime.deviceSynchronize()
cold = time.perf_counter() - t0
t1 = time.perf_counter()
for _ in range(3):
    cupy.fft.irfftn(cupy.fft.rfftn(a), s=shape, axes=axes)
cupy.cuda.runtime.deviceSynchronize()
steady = (time.perf_counter() - t1) / 3.0
print(max(0.0, cold - steady) + (t0 - t_start))
"""


def warmup_fresh_process(shape):
    """The probe prints one number; cupy prints warnings around it, so scan."""
    rc, out = sh([sys.executable, "-c", _FRESH, ",".join(str(n) for n in shape)], timeout=900)
    for line in reversed(out.strip().splitlines()):
        try:
            return float(line.strip())
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------
# E8 — planner VRAM against the pool, on a card it has never seen
# --------------------------------------------------------------------------


@experiment("E8", "planner VRAM inventory against the measured pool peak", gpu=True)
def _e8(ctx):
    import cupy

    from caustica.planner import estimate, spec_for_device

    # This card is in no datasheet, so plan against what it actually reports.
    name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    live = spec_for_device(name)
    if live is None:
        return {"skipped": f"no GPUSpec could be built for {name!r}"}
    rows = []
    for shape in [(96, 96, 96), (128, 128, 128), (160, 160, 160), (192, 192, 192)]:
        g, med, src, _apex, ref, _meta = bowl_scene(
            shape, 3.0e-4, aperture=0.008, roc=0.016, medium=water_medium(shape, beta=BETA_WATER)
        )
        est = estimate(
            g, med, src, run_spec(), solver="westervelt", gpu=live, reference_point=ref
        )
        pool = cupy.get_default_memory_pool()
        pool.free_all_blocks()
        try:
            solver("westervelt").run(
                g,
                med,
                src,
                run_spec(min_settle_periods=1, max_settle_periods=2, n_record_periods=1),
                backend="cupy",
                reference_point=ref,
            )
        except Exception as exc:
            rows.append({"shape": list(shape), "error": type(exc).__name__})
            continue
        peak = pool.total_bytes()
        rows.append(
            {
                "shape": list(shape),
                "planned_gib": est.vram_bytes / 2**30,
                "measured_pool_gib": peak / 2**30,
                "dev_pct": 100.0 * (est.vram_bytes - peak) / peak,
                "plan_source": est.source,
            }
        )
        pool.free_all_blocks()
    ok = [r for r in rows if "dev_pct" in r]
    return {
        "rows": rows,
        "verdict": (
            f"worst VRAM deviation {max(abs(r['dev_pct']) for r in ok):+.1f}% over "
            f"{len(ok)} shapes on an unknown card"
            if ok
            else "no shape produced a measurement"
        ),
    }


# --------------------------------------------------------------------------
# E9 — reproduce the hosted-runtime test failure
# --------------------------------------------------------------------------


@experiment("E9", "clean environment: the wheel plus the checkout's tests")
def _e9(ctx):
    """On Colab the suite failed with three collection errors while the same
    suite is green in an editable install. A wheel install with only the
    checkout's tests is exactly that situation, so build one and read them."""
    repo = Path(__file__).resolve().parent.parent
    work = Path(ctx["work"]) / "cleanenv"
    work.mkdir(parents=True, exist_ok=True)
    wheels = work / "wheels"
    rc, out = sh(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(wheels)],
        cwd=repo,
        timeout=1800,
    )
    built = sorted(wheels.glob("caustica-*.whl"))
    if rc != 0 or not built:
        return {"error": f"wheel build failed (exit {rc})", "tail": out[-800:]}
    venv = work / "venv"
    sh([sys.executable, "-m", "venv", str(venv)], timeout=900)
    py = venv / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
    rc, out = sh([str(py), "-m", "pip", "install", "-q", str(built[0]), "pytest"], timeout=1800)
    if rc != 0:
        return {"error": f"install failed (exit {rc})", "tail": out[-800:]}
    rc, out = sh([str(py), "-m", "pytest", "-q", "-p", "no:warnings", "--co"], cwd=repo)
    errors = [ln.strip() for ln in out.splitlines() if ln.startswith("ERROR ")]
    causes = sorted(
        {
            ln.strip()
            for ln in out.splitlines()
            if "ModuleNotFoundError" in ln or "ImportError" in ln
        }
    )
    rc_run, out_run = sh(
        [str(py), "-m", "pytest", "-q", "-p", "no:warnings"], cwd=repo, timeout=5400
    )
    summary = next(
        (
            ln.strip()
            for ln in reversed(out_run.splitlines())
            if "passed" in ln or "error" in ln or "failed" in ln
        ),
        f"exit {rc_run}",
    )
    return {
        "wheel": built[0].name,
        "collect_exit": rc,
        "collection_errors": errors,
        "causes": causes[:8],
        "run_exit": rc_run,
        "summary": summary,
        "verdict": (
            f"{len(errors)} collection error(s) with a wheel install: "
            + ("; ".join(causes[:3]) if causes else summary)
        ),
    }


# --------------------------------------------------------------------------
# E10 — the knobs the library calls safe defaults
# --------------------------------------------------------------------------


@experiment("E10", "sensitivity of the answer to CFL, PML thickness and resolution")
def _e10(ctx):
    """A default is only safe if the answer stops moving when you change it.

    The PML arm holds the interior domain fixed and grows the grid around it, so
    a thicker absorber never eats the transducer — the mistake that made an
    earlier pass report a 26% "PML sensitivity" that was a clipped source.
    """
    f0, aperture, roc = 1.0e6, 0.006, 0.012
    rows = []

    def one(tag, *, dx, pml_vox, cfl):
        shape = grid_for_bowl(aperture, roc, dx, pml_vox=pml_vox)
        g, med, src, apex_z, ref, meta = bowl_scene(
            shape, dx, f0=f0, aperture=aperture, roc=roc, pml_vox=pml_vox
        )
        res = solver("linear").run(
            g,
            med,
            src,
            run_spec(cfl=cfl, cfl_hard_max=max(cfl, 0.5)),
            backend="numpy",
            reference_point=ref,
        )
        rows.append(
            {
                "knob": tag,
                "shape": list(shape),
                "dx_mm": dx * 1e3,
                "ppw": C0 / f0 / dx,
                "pml_vox": pml_vox,
                "cfl": cfl,
                "clipped": meta["clipped"],
                **profile_metrics(res.phasor, shape, apex_z, dx, aperture, roc, f0),
            }
        )

    base = dict(dx=1.5e-4, pml_vox=10, cfl=0.48)
    one("baseline", **base)
    for cfl in (0.15, 0.30):
        one(f"cfl={cfl}", **{**base, "cfl": cfl})
    for pml in (6, 16, 24):
        one(f"pml={pml}", **{**base, "pml_vox": pml})
    for dx, tag in ((2.5e-4, "ppw=6"), (1.0e-4, "ppw=15")):
        one(tag, **{**base, "dx": dx})

    ref0 = rows[0]
    for r in rows:
        r["peak_dev_pct"] = 100.0 * (r["peak_pa"] - ref0["peak_pa"]) / ref0["peak_pa"]
        r["focus_shift_mm"] = r["peak_z_mm"] - ref0["peak_z_mm"]

    def spread(prefix):
        sel = [r for r in rows if r["knob"].startswith(prefix)]
        return max(abs(r["peak_dev_pct"]) for r in sel) if sel else 0.0

    return {
        "rows": rows,
        "any_clipped": [r["knob"] for r in rows if r["clipped"]],
        "pml_spread_pct": spread("pml"),
        "cfl_spread_pct": spread("cfl"),
        "ppw_spread_pct": spread("ppw"),
        "verdict": (
            f"focal peak moves at most {spread('pml'):.1f}% with PML thickness, "
            f"{spread('cfl'):.1f}% with CFL and {spread('ppw'):.1f}% with resolution "
            f"(ppw 6 to 15); correlation with O'Neil stays above "
            f"{min(r['correlation_r'] for r in rows):.4f}"
        ),
    }


# --------------------------------------------------------------------------
# E11 — does the disagreement vanish as the grid refines?
# --------------------------------------------------------------------------


@experiment("E11", "grid refinement: the two operators against O'Neil as dx shrinks")
def _e11(ctx):
    """Two discretizations of one continuum operator must converge together.

    If the gap between the fixed and pre-fix fields shrinks with dx, the change
    is a discretization artefact rather than a change of physics — and whichever
    approaches the analytic solution faster is the better discretization.
    """
    f0, aperture, roc = 1.0e6, 0.006, 0.012
    rows = []
    for dx in (3.0e-4, 2.0e-4, 1.5e-4):
        shape = grid_for_bowl(aperture, roc, dx)
        fields, apex_z = {}, None
        for label, cm in (("fixed", contextlib.nullcontext()), ("legacy", legacy_operator())):
            g, med, src, apex_z, ref, _meta = bowl_scene(
                shape, dx, f0=f0, aperture=aperture, roc=roc
            )
            with cm:
                res = solver("linear").run(
                    g, med, src, run_spec(), backend="numpy", reference_point=ref
                )
            fields[label] = np.asarray(res.phasor)
        row: dict[str, Any] = {"shape": list(shape), "dx_mm": dx * 1e3, "ppw": C0 / f0 / dx}
        for label in ("fixed", "legacy"):
            row[label] = profile_metrics(fields[label], shape, apex_z, dx, aperture, roc, f0)
        lin, l2 = field_diff(fields["fixed"], fields["legacy"])
        row["fixed_vs_legacy"] = {"max_rel": lin, "rel_l2": l2}
        row["fixed_is_closer"] = bool(row["fixed"]["one_minus_r"] <= row["legacy"]["one_minus_r"])
        rows.append(row)

    gaps = [r["fixed_vs_legacy"]["rel_l2"] for r in rows]
    wins = sum(1 for r in rows if r["fixed_is_closer"])
    return {
        "rows": rows,
        "gap_by_ppw": {f"{r['ppw']:.0f}": r["fixed_vs_legacy"]["rel_l2"] for r in rows},
        "gap_shrinks_with_refinement": bool(gaps[-1] < gaps[0]),
        "fixed_closer_at": f"{wins}/{len(rows)}",
        "verdict": (
            f"the gap falls {gaps[0] * 100:.2f}% -> {gaps[-1] * 100:.2f}% as ppw goes "
            f"{rows[0]['ppw']:.0f} -> {rows[-1]['ppw']:.0f}; the fixed operator is closer "
            f"to O'Neil in {wins}/{len(rows)} resolutions"
        ),
    }


# --------------------------------------------------------------------------
# E12 — where the fix should bite hardest: the harmonics
# --------------------------------------------------------------------------


@experiment("E12", "harmonics: the higher the harmonic, the more the Nyquist bin mattered")
def _e12(ctx):
    """A 3f0 field sits three times closer to the grid Nyquist than the drive.

    If the change really is about the top of the spectrum, its size has to grow
    with harmonic order. If it does not, the explanation is wrong.
    """
    dx = 2.0e-4
    shape = grid_for_bowl(0.006, 0.012, dx)
    g, _m, src, _apex, ref, meta = bowl_scene(
        shape, dx, f0=1.0e6, amp=8.0e5, aperture=0.006, roc=0.012
    )
    med = water_medium(shape, beta=BETA_WATER)
    spec = run_spec()
    out = {}
    for label, cm in (("fixed", contextlib.nullcontext()), ("legacy", legacy_operator())):
        with cm:
            res = solver("westervelt").run(
                g, med, src, spec, backend="numpy", harmonics=(1, 2, 3), reference_point=ref
            )
        out[label] = {h: np.asarray(res.phasors[h]) for h in (1, 2, 3)}
    rows = []
    for h in (1, 2, 3):
        lin, l2 = field_diff(out["fixed"][h], out["legacy"][h])
        pf = float(np.abs(out["fixed"][h]).max())
        pl = float(np.abs(out["legacy"][h]).max())
        rows.append(
            {
                "harmonic": h,
                "ppw_at_this_harmonic": C0 / (h * 1.0e6) / dx,
                "peak_kpa_fixed": pf / 1e3,
                "peak_kpa_legacy": pl / 1e3,
                "peak_shift_pct": 100.0 * (pf - pl) / pl,
                "max_rel": lin,
                "rel_l2": l2,
            }
        )
    grows = all(a["rel_l2"] <= b["rel_l2"] for a, b in zip(rows, rows[1:], strict=False))
    return {
        "rows": rows,
        "scenario": {"shape": list(shape), "dx_mm": dx * 1e3, "drive_kpa": 800.0, "source": meta},
        "grows_with_harmonic_order": grows,
        "verdict": (
            (
                "the change grows with harmonic order as predicted: "
                if grows
                else "the change does NOT grow monotonically with harmonic order: "
            )
            + " -> ".join(f"h{r['harmonic']} {r['rel_l2'] * 100:.2f}%" for r in rows)
        ),
    }


# --------------------------------------------------------------------------
# E13 — the GPU paths nobody compared: resume and thermal
# --------------------------------------------------------------------------


@experiment("E13", "checkpoint resume and the thermal solver, on the GPU", gpu=True)
def _e13(ctx):
    """Two claims that were only ever exercised on the CPU.

    "A resumed run is bit-identical" and "the thermal solver never touches numpy
    for state maths" are both backend claims, and both were measured on one
    backend. float32 reductions are not associative, so neither is free.
    """
    import numpy as np

    from caustica.io import CheckpointSpec, RunInterrupted

    out = {}

    # ---- resume on cupy ----
    dx = 3.0e-4
    shape = grid_for_bowl(0.005, 0.010, dx)
    g, med, src, _apex, ref, _meta = bowl_scene(shape, dx, aperture=0.005, roc=0.010)
    spec = run_spec(min_settle_periods=3, max_settle_periods=3, n_record_periods=1)
    ck = Path(ctx["work"]) / "gpu_resume.npz"
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.unlink(missing_ok=True)
    seen = {"n": 0}

    def stop_when():
        seen["n"] += 1
        return seen["n"] == 2

    try:
        solver("linear").run(
            g,
            med,
            src,
            spec,
            backend="cupy",
            reference_point=ref,
            checkpoint=CheckpointSpec(path=ck, every_periods=1, stop_when=stop_when),
        )
        interrupted = False
    except RunInterrupted:
        interrupted = True
    if ck.exists():
        resumed = np.asarray(
            solver("linear")
            .run(
                g,
                med,
                src,
                spec,
                backend="cupy",
                reference_point=ref,
                checkpoint=CheckpointSpec(path=ck, every_periods=1),
            )
            .phasor
        )
        straight = np.asarray(
            solver("linear").run(g, med, src, spec, backend="cupy", reference_point=ref).phasor
        )
        lin, l2 = field_diff(straight, resumed)
        out["resume_on_cupy"] = {
            "interrupted": interrupted,
            "max_rel": lin,
            "rel_l2": l2,
            "bit_exact": lin == 0.0,
        }
    else:
        out["resume_on_cupy"] = {"skipped": "the stop hook never fired"}

    # ---- Pennes on both backends ----
    from caustica import PennesSolver, ThermalMedium

    n, tdx = 64, 1e-3
    ones = np.ones((n, n, n), np.float32)
    tmed = ThermalMedium(
        k=ones * 0.5,
        rho=ones * 1050.0,
        specific_heat=ones * 3600.0,
        perfusion=ones * 0.5,
        dx=tdx,
    )
    q = np.zeros((n, n, n), np.float32)
    q[n // 2 - 2 : n // 2 + 2, n // 2 - 2 : n // 2 + 2, n // 2 - 2 : n // 2 + 2] = 2.0e7
    t0 = np.full((n, n, n), 37.0, np.float32)
    res = {}
    for bk in ("numpy", "cupy"):
        r = PennesSolver(backend=bk).solve(t0, q, tmed, dt=0.05, n_steps=40, dose=True)
        res[bk] = (
            np.asarray(r.temperature, dtype=np.float64),
            np.asarray(r.dose_cem43, dtype=np.float64) if r.dose_cem43 is not None else None,
        )
    lin_t, l2_t = field_diff(res["numpy"][0] - 37.0, res["cupy"][0] - 37.0)
    entry = {
        "temperature_max_rel": lin_t,
        "temperature_rel_l2": l2_t,
        "peak_c_numpy": float(res["numpy"][0].max()),
        "peak_c_cupy": float(res["cupy"][0].max()),
        "bit_exact": lin_t == 0.0,
    }
    if res["numpy"][1] is not None and res["cupy"][1] is not None:
        lin_d, _ = field_diff(res["numpy"][1], res["cupy"][1])
        entry["cem43_max_rel"] = lin_d
        entry["cem43_peak_min"] = float(res["numpy"][1].max())
    out["thermal_backends"] = entry

    r = out["resume_on_cupy"]
    if "bit_exact" in r:
        state = "bit-exact" if r["bit_exact"] else "NOT bit-exact"
        parts = [f"resume on cupy: max rel {r['max_rel']:.2e} ({state})"]
    else:
        parts = ["resume not exercised"]
    parts.append(f"thermal numpy vs cupy: max rel {entry['temperature_max_rel']:.2e} on the rise")
    out["verdict"] = "; ".join(parts)
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def environment() -> dict:
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("caustica", "numpy", "scipy", "h5py"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import cupy

        info["cupy"] = cupy.__version__
        info["gpu_name"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        free, total = cupy.cuda.runtime.memGetInfo()
        info["vram_free_gib"] = round(free / 2**30, 2)
        info["vram_total_gib"] = round(total / 2**30, 2)
    except Exception as exc:
        info["cupy"] = None
        info["gpu_note"] = f"{type(exc).__name__}: {exc}"[:160]
    rc, out = sh(["git", "rev-parse", "HEAD"], timeout=60)
    info["git_commit"] = out.strip() if rc == 0 else None
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dev_campaign.py")
    ap.add_argument("--experiments", default=None, help="comma-separated ids")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default="benchmarks/reports/campaign")
    args = ap.parse_args(argv)

    if args.list:
        for eid, title, gpu, _ in EXPERIMENTS:
            print(f"  {eid:<4} {'GPU' if gpu else '   '}  {title}")
        return 0
    want = {s.strip() for s in args.experiments.split(",")} if args.experiments else None
    if want is None and not args.all:
        ap.error("pass --experiments, --all or --list")

    out = Path(args.out).resolve() / time.strftime("%Y%m%d-%H%M%S")
    (out / "work").mkdir(parents=True, exist_ok=True)
    env = environment()
    have_gpu = gpu_available()

    print("=" * 78)
    print("caustica dev campaign")
    print("=" * 78)
    print(f"  caustica {env.get('caustica')} @ {str(env.get('git_commit'))[:12]}")
    print(f"  GPU: {env.get('gpu_name') or 'none'} ({env.get('vram_total_gib')} GiB)")
    print(f"  out: {out}\n", flush=True)

    report: dict[str, Any] = {
        "format": FORMAT,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": env,
        "experiments": [],
    }
    for eid, title, needs_gpu, fn in EXPERIMENTS:
        if want is not None and eid not in want:
            continue
        if needs_gpu and not have_gpu:
            print(f"[{eid}] SKIP (no GPU) -- {title}", flush=True)
            report["experiments"].append(
                {"id": eid, "title": title, "status": "SKIP", "data": {"skipped": "no usable GPU"}}
            )
            continue
        print(f"[{eid}] {title} ...", flush=True)
        t0 = time.monotonic()
        try:
            data = fn({"work": out / "work", "out": out})
            status = "SKIP" if isinstance(data, dict) and data.get("skipped") else "OK"
        except Exception:
            data = {"traceback": traceback.format_exc()[-2500:]}
            status = "ERROR"
        dt = time.monotonic() - t0
        report["experiments"].append(
            {"id": eid, "title": title, "status": status, "elapsed_s": round(dt, 1), "data": data}
        )
        verdict = (data or {}).get("verdict") or (data or {}).get("skipped") or status
        print(f"      -> {status} in {dt:.1f} s: {verdict}\n", flush=True)
        (out / "campaign.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )

    print("=" * 78)
    for e in report["experiments"]:
        print(f"  {e['id']:<4} {e['status']:<6} {e['title']}")
    print(f"\n  report: {out / 'campaign.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
