"""Heterogeneous media: several different ways of asking whether it is right.

Everything this library has been graded against so far lives in a uniform
medium — O'Neil, Rayleigh, Fubini, the ITRUSST water benchmarks. The dataset
it is meant to produce does not: breast phantoms are layers and inclusions
with real impedance contrasts (fat 1.35, skin 1.77, muscle 1.66 MRayl against
water's 1.50). Nothing standing in the suite covers that, and the one
heterogeneous cross-check on record is a figure from 2026-08-10, which
predates both of this week's numerics generations.

One comparison would be a coincidence, so this asks in several different
ways, of different kinds:

    H1  one interface, against the closed-form reflection coefficient
    H2  a slab, swept through its half-wave resonance, against the exact
        transfer matrix -- the discriminating one, since it tests phase
        accumulation and internal multiples rather than an amplitude
    H3  a skin/fat/muscle stack with real absorption, same reference
    H4  a focused bowl in a layered medium, against k-Wave, on ABSOLUTE
        amplitude rather than the normalized correlation the shipped
        cross-check grades
    H5  does the heterogeneous answer converge as dx shrinks

H1 to H3 run in 1-D, which is not a limitation but the point: the transfer
matrix describes exactly that geometry, so the comparison is about layered
physics and nothing else. Every heterogeneous run is divided by a homogeneous
run of the same geometry, which cancels the source calibration and leaves
only what the layers did.

Run it::

    python scripts/dev_hetero.py --out benchmarks/reports/hetero
    python scripts/dev_hetero.py --only H2
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
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

MM = 1e-3
F0 = 1.0e6
C_W, RHO_W = 1500.0, 1000.0
Z_W = C_W * RHO_W
DRIVE = 1.0e5

CHECKS: list[tuple[str, str, Callable]] = []


def check(cid: str, title: str):
    def wrap(fn):
        CHECKS.append((cid, title, fn))
        return fn

    return wrap


# --------------------------------------------------------------------------
# a 1-D stratified run, and the same run with the layers taken out
# --------------------------------------------------------------------------


#: Sponge thickness, and the water run-up before the stack, in mm. The run-up
#: has to hold a full standing-wave period clear of both the source and the
#: first interface, or the extremes the reflection is read from are not both
#: inside the window.
PML_MM = 8.0
PRE_MM = 9.0
POST_MM = 9.0

#: The background, and the default substrate.
WATER = (C_W, RHO_W, 0.0)


def run_1d(layers, dx: float, substrate=WATER, *, backend="numpy"):
    """Drive a plane wave through a stack into a half-space.

    ``layers`` is a list of ``(thickness_m, c, rho, alpha)``; ``substrate`` is
    the ``(c, rho, alpha)`` filling everything past them, sponge included.

    That last part is the whole trick: terminating the substrate INSIDE the
    PML rather than in another slab of water is what makes it a half-space.
    The first version of this ran a 40 mm layer with water behind it and
    called one — its far face sent a second reflection back into the region
    the coefficient was being read from, and the answer came out 91 % wrong
    in a way that looked exactly like a solver defect.

    An empty ``layers`` with a water substrate is the homogeneous reference.
    """
    from caustica.core.grid import Grid
    from caustica.core.pml import PMLSpec
    from caustica.medium import Medium
    from caustica.solvers import CWRunSpec, get
    from caustica.sources import plane_cw_source

    pml_vox = int(round(PML_MM * MM / dx))
    stack_mm = sum(t for t, *_ in layers) / MM
    n = int(round((PML_MM + PRE_MM + stack_mm + POST_MM + PML_MM) * MM / dx))
    grid = Grid(shape=(n,), dx=dx, pml=PMLSpec(thickness=PML_MM * MM))

    c = np.full(n, C_W, np.float32)
    rho = np.full(n, RHO_W, np.float32)
    alpha = np.zeros(n, np.float32)
    start = pml_vox + int(round(PRE_MM * MM / dx))
    at = start
    for thickness, c_l, rho_l, a_l in layers:
        stop = at + int(round(thickness / dx))
        c[at:stop], rho[at:stop], alpha[at:stop] = c_l, rho_l, a_l
        at = stop
    c_s, rho_s, a_s = substrate
    c[at:], rho[at:], alpha[at:] = c_s, rho_s, a_s
    medium = Medium(alpha=alpha, rho=rho, c=c, beta=np.zeros(n, np.float32))

    src = plane_cw_source(grid, f0=F0, amplitude=DRIVE, axis=0, position_vox=pml_vox + 4)
    spec = CWRunSpec(min_settle_periods=24, max_settle_periods=200, n_record_periods=2)
    res = get("linear")().run(grid, medium, src, spec, backend=backend)
    return np.arange(n) * dx, np.asarray(res.phasor, dtype=np.complex128), (start, at)


def measured_coefficients(layers, dx: float, substrate=WATER, backend="numpy"):
    """|R| from the standing wave before the stack, |T| from the wave past it.

    ``|T|`` is read against a homogeneous run of the same geometry, so the
    source's own calibration divides out and only the stack is left. ``|R|``
    comes from the standing-wave ratio, which needs no reference at all.
    """
    x, het, (start, stop) = run_1d(layers, dx, substrate, backend=backend)
    # The reference keeps the layers' THICKNESSES and replaces their contents
    # with water, so both runs get the same grid and sample the same points.
    # Dropping the layers instead shortens the domain, and the transmitted
    # window then indexes off the end of the shorter one.
    water_layers = [(t, C_W, RHO_W, 0.0) for t, *_ in layers]
    _x, hom, _ = run_1d(water_layers, dx, WATER, backend=backend)
    lam = C_W / F0
    src_end = int(round((PML_MM + 2.0) * MM / dx))
    inc = slice(src_end, start - int(round(0.15 * lam / dx)))
    a_het = np.abs(het[inc])
    swr = float(a_het.max() / a_het.min())
    r_meas = (swr - 1.0) / (swr + 1.0)
    # Transmitted: inside the substrate, clear of the last interface and of
    # the sponge, sampled at the same points in both runs. An absorbing
    # substrate decays across the window, so the decay is divided out rather
    # than left to bias the mean.
    post = slice(
        stop + int(round(0.3 * substrate[0] / F0 / dx)),
        len(x) - int(round((PML_MM + 1.0) * MM / dx)),
    )
    if post.stop - post.start < 4:
        raise ValueError(f"transmitted window holds {post.stop - post.start} points; widen POST_MM")
    decay = np.exp(-substrate[2] * (x[post] - x[stop]))
    t_meas = float(np.mean(np.abs(het[post]) / decay) / np.abs(hom[post]).mean())
    return r_meas, t_meas, dict(swr=swr, n=len(x), inc_pts=int(inc.stop - inc.start))


def exact(layers, z_out=Z_W):
    from caustica.analytic.layered import Layer, stack_coefficients

    ls = [Layer(t, c, rho, a) for t, c, rho, a in layers]
    return stack_coefficients(ls, Z_W, z_out, F0)


def rel(a, b):
    return float(abs(a - b) / abs(b)) if b else float("nan")


def pair(measured: float, exact_value: float, extra: str = "rel_error") -> dict:
    """One measured/exact/error record, formatted the same way everywhere."""
    e = abs(exact_value)
    return {
        "exact": e,
        "measured": measured,
        extra: rel(measured, e) if extra == "rel_error" else abs(measured - e),
    }


# --------------------------------------------------------------------------
# H1 — one interface
# --------------------------------------------------------------------------


@check("H1", "one interface: reflection and transmission against the closed form")
def _h1(ctx):
    """The most basic heterogeneous fact there is, and it has an exact answer.

    ``R = (Z2 - Z1) / (Z2 + Z1)``. A solver that gets an impedance step wrong
    gets everything downstream of it wrong, and in a breast phantom there is
    an impedance step every few millimetres.
    """
    from caustica.analytic.layered import interface_coefficients

    dx = C_W / F0 / ctx["ppw"]
    rows = []
    for name, c2, rho2 in (
        ("fat", 1450.0, 932.0),
        ("skin", 1600.0, 1109.0),
        ("muscle", 1580.0, 1050.0),
        ("hard (2x Z)", 1500.0, 2000.0),
    ):
        z2 = c2 * rho2
        r_ex, t_ex = interface_coefficients(Z_W, z2)
        # No layers at all: medium 2 is the SUBSTRATE, so the interface under
        # test is the only one in the domain and its far side is the sponge.
        r_me, t_me, extra = measured_coefficients(
            [], dx, substrate=(c2, rho2, 0.0), backend=ctx["backend"]
        )
        rows.append(
            {
                "second_medium": name,
                "z2_mrayl": z2 / 1e6,
                "reflection": pair(r_me, r_ex),
                "transmission": pair(t_me, t_ex),
                "standing_wave_ratio": extra["swr"],
            }
        )
    worst_r = max(r["reflection"]["rel_error"] for r in rows)
    worst_t = max(r["transmission"]["rel_error"] for r in rows)
    return {
        "points_per_wavelength": ctx["ppw"],
        "rows": rows,
        "verdict": (
            f"across four impedance steps from {rows[0]['z2_mrayl']:.2f} to "
            f"{rows[-1]['z2_mrayl']:.2f} MRayl the reflection coefficient is within "
            f"{worst_r * 100:.2f} % of the closed form and the transmission within "
            f"{worst_t * 100:.2f} %"
        ),
    }


# --------------------------------------------------------------------------
# H2 — a slab, swept through resonance
# --------------------------------------------------------------------------


@check("H2", "slab thickness sweep: the half-wave resonance, against the transfer matrix")
def _h2(ctx):
    """The discriminating one.

    A slab's transmission is not a number, it is a curve: transparent at
    every half-wavelength of its own, most reflective at the quarter. Getting
    that curve right means the solver accumulates phase correctly inside a
    medium whose wavelength is not the background's, and sums the internal
    multiples correctly. An amplitude-only check cannot see either.
    """
    dx = C_W / F0 / ctx["ppw"]
    c2, rho2 = 1450.0, 932.0  # fat
    lam2 = c2 / F0
    rows = []
    for frac in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0):
        d = frac * lam2
        layers = [(d, c2, rho2, 0.0)]
        r_ex, t_ex = exact(layers)
        r_me, t_me, _ = measured_coefficients(layers, dx, backend=ctx["backend"])
        rows.append(
            {
                "thickness_over_lambda": frac,
                "thickness_mm": d / MM,
                "reflection": pair(r_me, r_ex),
                "transmission": pair(t_me, t_ex, "abs_error"),
            }
        )
    half = next(r for r in rows if r["thickness_over_lambda"] == 0.5)
    quarter = next(r for r in rows if r["thickness_over_lambda"] == 0.25)
    worst_t = max(r["transmission"]["abs_error"] for r in rows)
    return {
        "points_per_wavelength": ctx["ppw"],
        "layer": "fat in water",
        "rows": rows,
        "verdict": (
            f"the transmission curve is followed to {worst_t:.4f} in amplitude across the "
            f"sweep; at the half-wave thickness the slab is transparent (measured "
            f"|T| {half['transmission']['measured']:.4f}, exact "
            f"{half['transmission']['exact']:.4f}) and at the quarter it reflects most "
            f"(|R| {quarter['reflection']['measured']:.4f} vs "
            f"{quarter['reflection']['exact']:.4f})"
        ),
    }


# --------------------------------------------------------------------------
# H3 — a breast-like stack, with real absorption
# --------------------------------------------------------------------------


@check("H3", "skin/fat/muscle stack with absorption, against the transfer matrix")
def _h3(ctx):
    """The phantom's own materials, in the order a beam meets them."""
    from caustica.materials import breast_default

    db = breast_default().materials
    skin, fat, muscle = db[1], db[2], db[3]
    dx = C_W / F0 / ctx["ppw"]
    stacks = {
        "skin 2 mm": [(2.0 * MM, skin.c, skin.rho, skin.alpha_np_m)],
        "skin 2 + fat 10": [
            (2.0 * MM, skin.c, skin.rho, skin.alpha_np_m),
            (10.0 * MM, fat.c, fat.rho, fat.alpha_np_m),
        ],
        "skin 2 + fat 10 + muscle 5": [
            (2.0 * MM, skin.c, skin.rho, skin.alpha_np_m),
            (10.0 * MM, fat.c, fat.rho, fat.alpha_np_m),
            (5.0 * MM, muscle.c, muscle.rho, muscle.alpha_np_m),
        ],
    }
    rows = []
    for name, layers in stacks.items():
        r_ex, t_ex = exact(layers)
        r_me, t_me, _ = measured_coefficients(layers, dx, backend=ctx["backend"])
        rows.append(
            {
                "stack": name,
                "n_layers": len(layers),
                "reflection": pair(r_me, r_ex),
                "transmission": pair(t_me, t_ex),
            }
        )
    worst_t = max(r["transmission"]["rel_error"] for r in rows)
    worst_r = max(r["reflection"]["rel_error"] for r in rows)
    return {
        "points_per_wavelength": ctx["ppw"],
        "rows": rows,
        "verdict": (
            f"through one, two and three absorbing layers the transmitted amplitude is "
            f"within {worst_t * 100:.2f} % of the transfer matrix and the reflection "
            f"within {worst_r * 100:.2f} %"
        ),
    }


# --------------------------------------------------------------------------
# H4 — a focused bowl through tissue, against an independent code
# --------------------------------------------------------------------------


@check("H4", "focused bowl through skin and fat, against k-Wave on absolute amplitude")
def _h4(ctx):
    """The one comparison with no closed form, and the one closest to the job.

    H1 to H3 are exact but one-dimensional: a plane wave meeting flat layers.
    A real phantom refracts and aberrates a converging beam, and there is no
    analytic answer for that — only another implementation. k-Wave is the one
    available, and it shares nothing with this engine but the source voxels.

    Graded on ABSOLUTE amplitude. The shipped cross-check (`cross`) grades
    normalized correlation, which is how a 13-18 % amplitude error survived
    months of green gates; repeating that mistake in the heterogeneous case
    would be careless.
    """
    from caustica.core.grid import Grid
    from caustica.core.pml import PMLSpec
    from caustica.materials import breast_default, water
    from caustica.medium import Medium
    from caustica.solvers import CWRunSpec, get
    from caustica.sources import bowl_cw_source

    db = breast_default().materials
    skin, fat = db[1], db[2]
    dx = C_W / F0 / ctx["h4_ppw"]
    aperture, roc = 10.0 * MM, 25.0 * MM
    pml_mm = 5.0
    pml_vox = int(round(pml_mm * MM / dx))
    margin = 4
    n_xy = 2 * (int(np.ceil(aperture / dx)) + pml_vox + margin) + 1
    apex_z = pml_vox + margin
    n_z = apex_z + int(round(1.5 * roc / dx)) + pml_vox + margin
    shape = (n_xy, n_xy, n_z)
    grid = Grid(shape=shape, dx=dx, pml=PMLSpec(thickness=pml_mm * MM))
    apex = (n_xy // 2, n_xy // 2, apex_z)

    # Water standoff, then skin, then fat all the way through the focus: the
    # order a beam meets a breast.
    c = np.full(shape, C_W, np.float32)
    rho = np.full(shape, RHO_W, np.float32)
    alpha = np.zeros(shape, np.float32)
    z_skin = apex_z + int(round(6.0 * MM / dx))
    z_fat = z_skin + int(round(2.0 * MM / dx))
    for lo, hi, mat in ((z_skin, z_fat, skin), (z_fat, n_z, fat)):
        c[:, :, lo:hi], rho[:, :, lo:hi] = mat.c, mat.rho
        alpha[:, :, lo:hi] = mat.alpha_np_m
    layered = Medium(alpha=alpha, rho=rho, c=c, beta=np.zeros(shape, np.float32))
    uniform = Medium.homogeneous(shape, water(c=C_W, rho=RHO_W))

    src = bowl_cw_source(grid, F0, DRIVE, aperture, roc, apex)
    spec = CWRunSpec(min_settle_periods=10, max_settle_periods=60, n_record_periods=2)
    focus = (apex[0], apex[1], apex[2] + int(round(roc / dx)))
    z = (np.arange(n_z) - apex_z) * dx
    sel = (z > 0.3 * roc) & (z < (n_z - pml_vox - 2 - apex_z) * dx)

    rows = []
    fields = {}
    for medium_name, medium in (("water", uniform), ("skin + fat", layered)):
        for solver in ("linear", "kwave"):
            kw = {"use_gpu_binary": True} if solver == "kwave" else {"backend": ctx["backend"]}
            t0 = time.perf_counter()
            try:
                res = get(solver)().run(grid, medium, src, spec, reference_point=focus, **kw)
            except Exception as exc:
                rows.append({"medium": medium_name, "solver": solver, "error": f"{exc}"[:120]})
                continue
            amp = np.abs(np.asarray(res.phasor)).astype(np.float64)
            axis = amp[apex[0], apex[1], :][sel]
            fields[(medium_name, solver)] = axis
            rows.append(
                {
                    "medium": medium_name,
                    "solver": solver,
                    "peak_mpa": float(axis.max() / 1e6),
                    "peak_z_mm": float(z[sel][int(axis.argmax())] / MM),
                    "elapsed_s": round(time.perf_counter() - t0, 1),
                }
            )

    agree = {}
    for medium_name in ("water", "skin + fat"):
        a = fields.get((medium_name, "linear"))
        b = fields.get((medium_name, "kwave"))
        if a is None or b is None:
            continue
        agree[medium_name] = {
            "peak_rel_difference": float(abs(a.max() - b.max()) / b.max()),
            "profile_rms_rel": float(np.sqrt(np.mean((a - b) ** 2)) / b.max()),
            "profile_correlation": float(np.corrcoef(a / a.max(), b / b.max())[0, 1]),
        }
    w = agree.get("water", {}).get("peak_rel_difference")
    t = agree.get("skin + fat", {}).get("peak_rel_difference")
    return {
        "points_per_wavelength": ctx["h4_ppw"],
        "grid": list(map(int, shape)),
        "megavoxels": grid.n_voxels / 1e6,
        "layers": "6 mm water, 2 mm skin, then fat through the focus",
        "rows": rows,
        "agreement": agree,
        "verdict": (
            "no pair completed"
            if w is None or t is None
            else (
                f"on absolute focal pressure the two codes differ by {w * 100:.2f} % in water "
                f"and {t * 100:.2f} % through skin and fat, so the layers cost "
                f"{abs(t - w) * 100:.2f} points of agreement; axial profiles correlate "
                f"{agree['skin + fat']['profile_correlation']:.5f} in tissue"
            )
        ),
    }


# --------------------------------------------------------------------------
# H5 — does the heterogeneous answer converge?
# --------------------------------------------------------------------------


@check("H5", "does the layered answer converge as dx shrinks?")
def _h5(ctx):
    """An interface is a step, and a spectral method meets a step badly.

    Whatever the error is at one spacing, the question that separates a
    discretization error from a modelling error is whether refining removes
    it — the same question the absolute-amplitude gate asks about a source.
    """
    c2, rho2 = 1450.0, 932.0
    layers = [(0.25 * c2 / F0, c2, rho2, 0.0)]
    r_ex, t_ex = exact(layers)
    rows = []
    for ppw in ctx["ladder"]:
        dx = C_W / F0 / ppw
        r_me, t_me, _ = measured_coefficients(layers, dx, backend=ctx["backend"])
        rows.append(
            {
                "points_per_wavelength": ppw,
                "dx_mm": dx / MM,
                "reflection_rel_error": rel(r_me, abs(r_ex)),
                "transmission_rel_error": rel(t_me, abs(t_ex)),
            }
        )
    first, last = rows[0], rows[-1]
    shrink = (
        last["transmission_rel_error"] / first["transmission_rel_error"]
        if first["transmission_rel_error"]
        else 0.0
    )
    return {
        "layer": "quarter-wave fat slab in water",
        "rows": rows,
        "transmission_error_shrink": shrink,
        "verdict": (
            f"from {first['points_per_wavelength']} to {last['points_per_wavelength']} points "
            f"per wavelength the transmission error goes "
            f"{first['transmission_rel_error'] * 100:.3f} % -> "
            f"{last['transmission_rel_error'] * 100:.3f} % "
            f"(x{shrink:.3f}) and the reflection error "
            f"{first['reflection_rel_error'] * 100:.3f} % -> "
            f"{last['reflection_rel_error'] * 100:.3f} %"
        ),
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def environment() -> dict:
    import numpy

    import caustica
    from caustica.core.backend import cupy_available

    env = {
        "caustica": caustica.__version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if cupy_available():
        import cupy

        env["gpu"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    return env


def cell(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}" if v == 0 or 1e-3 <= abs(v) < 1e5 else f"{v:.3e}"
    return str(v).replace("|", "/")


def flatten(value: Any, prefix: str = "") -> dict:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = value
    return out


def render_markdown(payload: dict) -> str:
    lines = [
        "# Heterogeneous media: several different ways",
        "",
        "Generated by `scripts/dev_hetero.py`. H1-H3 and H5 are graded against the",
        "exact transfer-matrix solution for a stratified fluid",
        "(`caustica.analytic.layered`); H4 against k-Wave.",
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
        rows = [flatten(r) for r in data.get("rows", [])]
        if rows:
            cols: list[str] = []
            for r in rows:
                for k in r:
                    if k not in cols:
                        cols.append(k)
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("|" + "---|" * len(cols))
            for r in rows:
                lines.append("| " + " | ".join(cell(r.get(c, "")) for c in cols) + " |")
    env = json.dumps(payload["environment"], indent=2)
    lines += ["", "## Environment", "", "```json", env, "```"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="benchmarks/reports/hetero")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--ppw", type=float, default=32.0, help="points per wavelength for H1-H3")
    ap.add_argument("--h4-ppw", type=float, default=6.0, help="points per wavelength for H4")
    ap.add_argument("--ladder", default="8,12,16,24,32,48", help="ppw ladder for H5")
    args = ap.parse_args(argv)

    from caustica.core.backend import cupy_available

    ctx = {
        "backend": "cupy" if cupy_available() else "numpy",
        "ppw": args.ppw,
        "h4_ppw": args.h4_ppw,
        "ladder": [float(s) for s in args.ladder.split(",") if s.strip()],
    }
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
            entry["data"] = fn(ctx)
            entry["status"] = "OK"
        except Exception as exc:
            entry["status"] = "ERROR"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc(limit=8)
        entry["elapsed_s"] = round(time.perf_counter() - t0, 2)
        results.append(entry)
        mark = "OK " if entry["status"] == "OK" else "ERR"
        detail = (entry.get("data") or {}).get("verdict", entry.get("error", ""))
        print(f"  {mark} {entry['elapsed_s']:>7.2f}s  {detail}", flush=True)

    path = outdir / "hetero.json"
    previous = []
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8")).get("checks", [])
        except (OSError, ValueError):
            previous = []
    merged = {e["id"]: e for e in previous}
    merged.update({e["id"]: e for e in results})
    order = [cid for cid, *_ in CHECKS]
    results = [merged[c] for c in order if c in merged]

    payload = {
        "format": "caustica-hetero/1",
        "environment": environment(),
        "checks": results,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (outdir / "REPORT.md").write_text(render_markdown(payload), encoding="utf-8")
    bad = [e["id"] for e in results if e["status"] != "OK"]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks ran -> {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
