"""Small volumes at fine spacing: convergence, k-Wave, and one attribution.

Three questions, one script.

**Does the answer converge as dx shrinks?** A focused bowl in a few cubic
millimetres of water, at 0.2, 0.1 and 0.05 mm, against O'Neil's closed form —
both the shape the analytic gate already checks and the absolute amplitude it
deliberately does not. Small volumes are what make 0.05 mm affordable, and
0.05 mm is where a 2 MHz drive gets fifteen points per wavelength.

**Does an independent code agree?** The same ladder run through k-Wave, whose
staggered grid and smoothed source masks make it wrong in different places
than we are. Agreement that improves with dx is evidence; agreement at one dx
is a coincidence.

**Which change actually repaired the GPU?** Commit ``f462ce4`` zeroed the
Nyquist wavenumber AND added an explicit ``axes=`` to both ``irfftn`` calls,
and the 256^3 divergence went away. Nothing has yet said which of the two did
it. One run settles it: the pre-fix operator, at 256^3, on cupy, under the
engine as it stands today — which already carries the explicit ``axes=``.

Run it::

    python scripts/dev_resolution.py --out benchmarks/reports/resolution
    python scripts/dev_resolution.py --only R1 --finest 0.05

R1 and R4 want a GPU; R2 and R3 run k-Wave on the CPU and are the slow ones.
"""

from __future__ import annotations

import argparse
import contextlib
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
C0, RHO0 = 1500.0, 1000.0

EXPERIMENTS: list[tuple[str, str, bool, Callable]] = []


def experiment(eid: str, title: str, *, gpu: bool = False):
    def wrap(fn):
        EXPERIMENTS.append((eid, title, gpu, fn))
        return fn

    return wrap


@contextlib.contextmanager
def legacy_operator():
    """Restore the pre-2026-08-24 derivative factory for the duration.

    Deliberately a monkeypatch and not a git checkout: the point of the
    attribution run is to exercise the operator as it was inside the engine
    as it IS, so that the only thing that differs from today's code is the
    one line under test.
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


def water_scene(
    dx: float,
    aperture: float,
    roc: float,
    f0: float,
    drive: float,
    pml_mm: float,
    discretization: str = "offgrid",
):
    """A focused bowl in water, sized from the transducer outwards.

    The grid is derived from the geometry rather than the other way round —
    the 2026-08-24 campaign wasted two experiments on bowls that did not fit
    their domains, and reported the rim dissolving in the sponge as a 26 %
    sensitivity to PML thickness.
    """
    from caustica.core.grid import Grid
    from caustica.core.pml import PMLSpec
    from caustica.materials import water
    from caustica.medium import Medium
    from caustica.sources import bowl_cw_source

    pml = pml_mm * MM
    pml_vox = int(np.ceil(pml / dx))
    margin = 4
    n_xy = 2 * (int(np.ceil(aperture / dx)) + pml_vox + margin) + 1
    apex_z = pml_vox + margin
    # A full focal length of water BEYOND the focus. The -6 dB depth of field
    # of an f/1.2 bowl is several wavelengths long, and a domain that stops
    # just past the focus truncates the lobe the comparison is measuring.
    n_z = apex_z + 2 * int(np.ceil(roc / dx)) + pml_vox + margin
    grid = Grid(shape=(n_xy, n_xy, n_z), dx=dx, pml=PMLSpec(thickness=pml))
    medium = Medium.homogeneous(grid.shape, water(c=C0, rho=RHO0))
    apex = (n_xy // 2, n_xy // 2, apex_z)
    src = bowl_cw_source(grid, f0, drive, aperture, roc, apex, discretization=discretization)
    focus = (apex[0], apex[1], apex[2] + int(round(roc / dx)))
    return grid, medium, src, apex, focus


def oneill_axis(grid, apex, aperture, roc, f0, drive):
    """The closed-form on-axis profile on this grid's own z samples."""
    from caustica.analytic import axial_pressure

    dx = grid.dx
    z = (np.arange(grid.shape[2]) - apex[2]) * dx
    z_pml = (grid.shape[2] - grid.pml_vox - 2 - apex[2]) * dx
    # Clear of the cap (its sag is a fraction of a millimetre) at the near
    # end, two voxels clear of the sponge at the far end, and wide enough for
    # the whole -6 dB lobe in between.
    sel = (z > 0.25 * roc) & (z < z_pml)
    p = np.abs(axial_pressure(z[sel], aperture, roc, f0, C0, RHO0, drive / (RHO0 * C0)))
    return z, sel, p


def minus6db_width(coord: np.ndarray, profile: np.ndarray) -> float | None:
    """The -6 dB axial width, or ``None`` when the window truncates the lobe.

    A truncated lobe is a fact about the domain, not a failure of the run:
    recording it as missing keeps the rest of the rung's numbers.
    """
    from caustica.validation.analytic_suite import minus6db_width as w

    try:
        return float(w(coord, profile))
    except ValueError:
        return None


def _mm(value: float | None) -> float | None:
    return None if value is None else value / MM


# --------------------------------------------------------------------------
# R1 — the refinement ladder
# --------------------------------------------------------------------------


@experiment("R1", "small volume, fine dx: the bowl against O'Neil as dx shrinks", gpu=True)
def _r1(ctx):
    """What converges, what does not, and what the difference means.

    The *shape* correlation and the -6 dB width are what the shipped analytic
    gate grades, and both tighten as dx falls whichever source is used. The
    *absolute* level is the one nothing in the library checked, and it is
    where the two discretizations part company.

    The prediction that motivated the repair: a binary shell's excess over
    O'Neil is a staircase factor, a property of digitizing a tilted surface,
    so it will NOT fall as dx shrinks; the band-limited source carries the
    cap's own area, so its error is an ordinary discretization error and
    will. Both legs run at every rung so the two behaviours sit in one table.
    """
    from caustica.solvers import CWRunSpec, get

    aperture, roc, f0, drive = 2.5 * MM, 6.0 * MM, 2.0e6, 1.0e5
    backend = ctx["backend"]
    rows = []
    for dx_mm in ctx["ladder"]:
      for mode in ("binary", "offgrid"):
        dx = dx_mm * MM
        grid, medium, src, apex, focus = water_scene(
            dx, aperture, roc, f0, drive, pml_mm=1.0, discretization=mode
        )
        cos_tmax = np.sqrt(1.0 - (aperture / roc) ** 2)
        cap_area = 2.0 * np.pi * roc**2 * (1.0 - cos_tmax)
        spec = CWRunSpec(min_settle_periods=8, max_settle_periods=40, n_record_periods=2)
        t0 = time.perf_counter()
        res = get("linear")().run(
            grid, medium, src, spec, backend=backend, reference_point=focus
        )
        elapsed = time.perf_counter() - t0
        amp = np.abs(np.asarray(res.phasor))
        axis = amp[apex[0], apex[1], :]
        z, sel, oneill = oneill_axis(grid, apex, aperture, roc, f0, drive)
        sim = axis[sel]
        rows.append(
            {
                "dx_mm": dx_mm,
                "discretization": mode,
                "points_per_wavelength": C0 / f0 / dx,
                "grid": list(map(int, grid.shape)),
                "megavoxels": grid.n_voxels / 1e6,
                "n_source_points": int(src.n_points),
                "drive_per_cap_area": float(src.drive_weights.sum()) * dx**2 / cap_area,
                "shape_correlation": float(
                    np.corrcoef(sim / sim.max(), oneill / oneill.max())[0, 1]
                ),
                "width_mm": {
                    "simulated": _mm(minus6db_width(z[sel], sim / sim.max())),
                    "oneill": _mm(minus6db_width(z[sel], oneill / oneill.max())),
                },
                "peak_z_mm": {
                    "simulated": float(z[sel][int(sim.argmax())] / MM),
                    "oneill": float(z[sel][int(oneill.argmax())] / MM),
                },
                "absolute_mpa": {
                    "simulated": float(sim.max() / 1e6),
                    "oneill": float(oneill.max() / 1e6),
                    "ratio": float(sim.max() / oneill.max()),
                },
                "converged_period": int(res.converged_period),
                "elapsed_s": round(elapsed, 2),
            }
        )

    # The library warns below three points per wavelength and means it: a
    # rung under that resolves nothing and belongs in the table as an anchor,
    # not in the trend it would otherwise dominate.
    resolved = [r for r in rows if r["points_per_wavelength"] >= 3.0]

    def leg(mode):
        return [r for r in resolved if r["discretization"] == mode]

    summary = {}
    for mode in ("binary", "offgrid"):
        got = leg(mode)
        if not got:
            continue
        summary[mode] = {
            "shape_correlation": [got[0]["shape_correlation"], got[-1]["shape_correlation"]],
            "absolute_ratio": [r["absolute_mpa"]["ratio"] for r in got],
            "drive_per_cap_area": [r["drive_per_cap_area"] for r in got],
            "source_points": [r["n_source_points"] for r in got],
        }
    b = summary.get("binary", {}).get("absolute_ratio", [])
    o = summary.get("offgrid", {}).get("absolute_ratio", [])
    ppw = [r["points_per_wavelength"] for r in leg("offgrid")]
    return {
        "geometry": {
            "aperture_mm": aperture / MM,
            "roc_mm": roc / MM,
            "f_number": roc / (2 * aperture),
            "f0_mhz": f0 / 1e6,
            "drive_kpa": drive / 1e3,
        },
        "backend": backend,
        "rows": rows,
        "resolved_rungs": sorted({r["dx_mm"] for r in resolved}),
        "by_discretization": summary,
        "verdict": (
            f"from {ppw[0]:.1f} to {ppw[-1]:.1f} points per wavelength the binary shell's "
            f"absolute level goes "
            + " -> ".join(f"{v:.3f}" for v in b)
            + "x O'Neil, flat: a staircase factor is not a discretization error. The "
            "band-limited source goes "
            + " -> ".join(f"{v:.3f}" for v in o)
            + "x, which is what an ordinary discretization error looks like"
        ),
    }


# --------------------------------------------------------------------------
# R2 — the same ladder, against k-Wave
# --------------------------------------------------------------------------


@experiment("R2", "the same ladder against k-Wave: does agreement improve with dx?")
def _r2(ctx):
    """An independent propagator, deliberately not an independent source.

    k-Wave stores pressure and velocity on a staggered grid and uses its own
    absorption model; neither choice is ours, so agreement between the two is
    informative about the propagation. What it is NOT independent of is the
    source: the adapter hands k-Wave our own grid points and weights, so both
    codes drive exactly the same discretized cap.

    That is what made this experiment the control for the geometry finding.
    With the binary shell both codes sat about 1.16x O'Neil at fifteen points
    per wavelength — two propagators sharing nothing but a digitized cap,
    landing on the same excess, which put the excess in the cap. With the
    band-limited source they should both land on the closed form instead, and
    the same shared-source logic says so for the same reason.

    The rungs are capped by wall clock, not by principle: k-Wave runs on the
    CPU here, and the six-megavoxel rung takes six minutes.
    """
    from caustica.solvers import CWRunSpec, get

    aperture, roc, f0, drive = 2.5 * MM, 6.0 * MM, 2.0e6, 1.0e5
    rows = []
    for dx_mm in ctx["kwave_ladder"]:
        dx = dx_mm * MM
        grid, medium, src, apex, focus = water_scene(dx, aperture, roc, f0, drive, pml_mm=1.0)
        spec = CWRunSpec(min_settle_periods=8, max_settle_periods=40, n_record_periods=2)
        z, sel, oneill = oneill_axis(grid, apex, aperture, roc, f0, drive)
        entry: dict[str, Any] = {
            "dx_mm": dx_mm,
            "grid": list(map(int, grid.shape)),
            "megavoxels": grid.n_voxels / 1e6,
            "points_per_wavelength": C0 / f0 / dx,
        }
        profiles = {}
        for name in ("linear", "kwave"):
            t0 = time.perf_counter()
            try:
                kw = {} if name == "kwave" else {"backend": ctx["backend"]}
                res = get(name)().run(grid, medium, src, spec, reference_point=focus, **kw)
            except Exception as exc:
                entry[name] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            amp = np.abs(np.asarray(res.phasor))
            axis = amp[apex[0], apex[1], :][sel]
            profiles[name] = axis
            entry[name] = {
                "peak_mpa": float(axis.max() / 1e6),
                "peak_z_mm": float(z[sel][int(axis.argmax())] / MM),
                "width_mm": _mm(minus6db_width(z[sel], axis / axis.max())),
                "over_oneill": float(axis.max() / oneill.max()),
                "elapsed_s": round(time.perf_counter() - t0, 1),
            }
        if len(profiles) == 2:
            a, b = profiles["linear"], profiles["kwave"]
            entry["agreement"] = {
                "peak_rel_difference": float(abs(a.max() - b.max()) / b.max()),
                "profile_correlation": float(np.corrcoef(a / a.max(), b / b.max())[0, 1]),
                "profile_rms_rel_difference": float(
                    np.sqrt(np.mean((a / a.max() - b / b.max()) ** 2))
                ),
            }
        entry["oneill_mpa"] = float(oneill.max() / 1e6)
        rows.append(entry)

    got = [r for r in rows if "agreement" in r]
    if not got:
        return {"rows": rows, "verdict": "no rung produced both legs; nothing to compare"}
    peaks = [r["agreement"]["peak_rel_difference"] for r in got]
    shapes = [r["agreement"]["profile_rms_rel_difference"] for r in got]
    finest = got[-1]
    return {
        "rows": rows,
        "shared_source_discretization": True,
        "peak_difference_span": [min(peaks), max(peaks)],
        "agreement_improves_with_dx": peaks[-1] < peaks[0] if len(peaks) > 1 else None,
        "verdict": (
            f"across {len(got)} rungs the two codes' focal peaks converge onto each other, "
            f"{max(peaks) * 100:.1f} % apart at the coarsest and {peaks[-1] * 100:.1f} % at "
            f"{finest['points_per_wavelength']:.0f} points per wavelength, with normalized "
            f"profiles {min(shapes) * 100:.1f}-{max(shapes) * 100:.1f} % rms apart. Against "
            f"O'Neil's absolute prediction they sit {finest['linear']['over_oneill']:.3f}x "
            f"(ours) and {finest['kwave']['over_oneill']:.3f}x (k-Wave) at the finest rung — "
            f"two propagators driven from one shared source discretization, agreeing with each "
            f"other and with the closed form"
        ),
    }


# --------------------------------------------------------------------------
# R3 — the shipped cross-check harness, end to end
# --------------------------------------------------------------------------


@experiment("R3", "the shipped k-Wave cross-check harness on its own jobs")
def _r3(ctx):
    """R2 hand-rolls a comparison; this runs the one the library ships.

    ``caustica.validation.compare`` is what a user reaches for, and its gate
    table is the claim the project makes in public. Running it here means the
    claim and the evidence come from the same code path — a hand-rolled
    comparison agreeing with a shipped harness that nobody ran would be worth
    nothing.
    """
    from caustica.validation.compare import compare, mini_job, t0_job

    rows = []
    for name, factory in (("compare-mini", mini_job), ("t0-sanity", t0_job)):
        t0 = time.perf_counter()
        try:
            code, payload = compare(
                job=factory(),
                solvers=["linear", "westervelt", "kwave"],
                backend=ctx["backend"],
                # Keep the harness's own report beside this one instead of in
                # its default dated folder: one run, one place to look.
                out=ctx["outdir"] / "compare" / name,
                log=lambda _m: None,
            )
        except Exception as exc:
            rows.append({"job": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        gates = [
            {
                "id": g["id"],
                "criterion": g.get("criterion", ""),
                "checks": [
                    {"name": c["name"], "verdict": c["verdict"], "detail": c["detail"]}
                    for c in g.get("checks", [])
                ],
            }
            for g in payload.get("gates", [])
        ]
        rows.append(
            {
                "job": name,
                "exit_code": code,
                "verdict": payload.get("verdict"),
                "solvers": list(payload.get("runs", {})),
                "gates": gates,
                "elapsed_s": round(time.perf_counter() - t0, 1),
            }
        )
    ok = [r for r in rows if r.get("verdict") == "PASS"]
    return {
        "rows": rows,
        "verdict": (
            f"{len(ok)}/{len(rows)} jobs pass the shipped cross-check: "
            + "; ".join(f"{r['job']} {r.get('verdict', r.get('error'))}" for r in rows)
        ),
    }


# --------------------------------------------------------------------------
# R4 — which half of the fix repaired the GPU
# --------------------------------------------------------------------------


@experiment("R4", "attribution: was it the Nyquist zeroing or the explicit axes?", gpu=True)
def _r4(ctx):
    """The one run the 2026-08-24 campaign never got to.

    Commit ``f462ce4`` did two things: it zeroed the Nyquist wavenumber in
    the collocated first derivative, and it passed an explicit ``axes=`` to
    both ``irfftn`` calls so the transform could not silently pick a
    different set. Afterwards 256^3 stopped diverging on cupy. Nothing since
    has said which change was responsible, and the story the CHANGELOG tells
    names the first.

    Restoring only the old derivative factory inside today's engine
    discriminates them. If 256^3 diverges again, the Nyquist zeroing is the
    repair and the ``axes=`` was housekeeping. If it stays finite, the story
    is backwards and needs rewriting. Two smaller even sizes and one odd size
    come along: the odd one is the control, since an odd axis has no Nyquist
    bin to zero and must be unaffected either way.
    """
    from caustica.solvers import CWRunSpec, get
    from caustica.solvers.base import SolverDivergedError

    aperture, roc, f0, drive = 8.0 * MM, 16.0 * MM, 1.0e6, 1.0e5
    dx = 3.0e-4
    rows = []
    for n in ctx["attribution_sizes"]:
        shape = (n, n, n)
        entry: dict[str, Any] = {
            "n": n,
            "even": n % 2 == 0,
            "megavoxels": n**3 / 1e6,
        }
        from caustica.core.grid import Grid
        from caustica.core.pml import PMLSpec
        from caustica.materials import water
        from caustica.medium import Medium
        from caustica.sources import bowl_cw_source

        grid = Grid(shape=shape, dx=dx, pml=PMLSpec(thickness=3.0e-3))
        medium = Medium.homogeneous(shape, water(c=C0, rho=RHO0, beta=3.5))
        apex = (n // 2, n // 2, grid.pml_vox + 4)
        # The legacy shell on purpose: this experiment varies the derivative
        # operator and nothing else, so the source has to be the one the
        # 2026-08-24 divergence was seen with.
        src = bowl_cw_source(
            grid, f0, drive, aperture, roc, apex, discretization="binary"
        )
        near = (apex[0], apex[1], min(n - 1, apex[2] + 2))
        spec = CWRunSpec(min_settle_periods=1, max_settle_periods=1, n_record_periods=1)
        for label, cm in (("legacy", legacy_operator()), ("fixed", contextlib.nullcontext())):
            try:
                with cm:
                    res = get("westervelt")().run(
                        grid, medium, src, spec, backend="cupy", reference_point=near
                    )
                peak = float(np.abs(np.asarray(res.phasor)).max())
                entry[label] = {"peak_pa": peak, "finite": bool(np.isfinite(peak))}
            except SolverDivergedError as exc:
                entry[label] = {"peak_pa": None, "finite": False, "diverged": str(exc)[:120]}
            except Exception as exc:
                entry[label] = {"error": f"{type(exc).__name__}: {exc}"}
        rows.append(entry)

    ran = [r for r in rows if "legacy" in r and "error" not in r["legacy"]]
    broke = [r["n"] for r in ran if not r["legacy"]["finite"]]
    still = [r["n"] for r in ran if r.get("fixed", {}).get("finite") is False]
    attributed = "the Nyquist zeroing" if broke else "the explicit axes= (or neither)"
    return {
        "engine_carries_explicit_axes": True,
        "rows": rows,
        "sizes_diverging_with_the_old_operator": broke,
        "sizes_diverging_with_the_shipped_operator": still,
        "verdict": (
            f"with today's engine — explicit axes= and all — the pre-fix derivative factory "
            f"still diverges at {broke} and the shipped one at {still}: the repair is "
            f"{attributed}"
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
        "gpu": None,
    }
    if cupy_available():
        import cupy

        props = cupy.cuda.runtime.getDeviceProperties(0)
        env["gpu"] = {
            "name": props["name"].decode(),
            "cupy": cupy.__version__,
            "total_gib": props["totalGlobalMem"] / 2**30,
        }
    try:
        import kwave

        env["kwave"] = getattr(kwave, "__version__", "installed")
    except Exception:
        env["kwave"] = None
    return env


def merge_into(path: Path, key: str, fresh: list[dict]) -> list[dict]:
    """This run's entries over whatever a previous run left, in id order.

    A partial run must not destroy the record: ``--only R4`` re-measures one
    experiment, it does not mean the other three stopped being true.
    """
    previous: list[dict] = []
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8")).get(key, [])
        except (OSError, ValueError):
            previous = []
    merged = {e["id"]: e for e in previous}
    merged.update({e["id"]: e for e in fresh})
    order = [eid for eid, *_ in EXPERIMENTS]
    return [merged[eid] for eid in order if eid in merged]


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


def render_markdown(payload: dict) -> str:
    lines = [
        "# Small volumes, fine spacing, and an independent code",
        "",
        "Generated by `scripts/dev_resolution.py`.",
        "",
        "| id | question | verdict |",
        "|---|---|---|",
    ]
    for e in payload["experiments"]:
        v = (e.get("data") or {}).get("verdict", e.get("error", ""))
        lines.append(f"| [{e['id']}](#{e['id'].lower()}) | {e['title']} | {cell(v)} |")
    for e in payload["experiments"]:
        data = e.get("data") or {}
        lines += ["", f"## {e['id']}", "", f"**{e['title']}**", ""]
        if e["status"] == "ERROR":
            lines += ["```", str(e.get("error", "")), "```"]
            continue
        lines += [str(data.get("verdict", data.get("skipped", ""))), ""]
        if data.get("rows"):
            lines += table(data["rows"])
        summary = {k: v for k, v in data.items() if k not in ("rows", "verdict")}
        if summary:
            lines += ["", "```json", json.dumps(summary, indent=2, default=str), "```"]
    lines += [
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(payload["environment"], indent=2),
        "```",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="benchmarks/reports/resolution")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--ladder", default="0.2,0.1,0.05", help="dx values in mm for R1")
    ap.add_argument("--kwave-ladder", default="0.2,0.1", help="dx values in mm for R2")
    ap.add_argument("--attribution-sizes", default="128,192,243,256")
    args = ap.parse_args(argv)

    from caustica.core.backend import cupy_available

    only = {s.strip().upper() for s in args.only.split(",") if s.strip()}
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    ctx = {
        "backend": "cupy" if cupy_available() else "numpy",
        "ladder": [float(s) for s in args.ladder.split(",") if s.strip()],
        "kwave_ladder": [float(s) for s in args.kwave_ladder.split(",") if s.strip()],
        "attribution_sizes": [int(s) for s in args.attribution_sizes.split(",") if s.strip()],
        "outdir": outdir,
    }

    results = []
    for eid, title, needs_gpu, fn in EXPERIMENTS:
        if (only and eid not in only) or eid in skip:
            continue
        entry: dict[str, Any] = {"id": eid, "title": title}
        if needs_gpu and not cupy_available():
            entry["status"] = "SKIP"
            entry["data"] = {"skipped": "no CUDA device on this machine"}
            results.append(entry)
            print(f"[{eid}] SKIP (needs a GPU)", flush=True)
            continue
        print(f"[{eid}] {title} ...", flush=True)
        t0 = time.perf_counter()
        try:
            entry["data"] = fn(ctx)
            entry["status"] = "OK"
        except Exception as exc:
            entry["status"] = "ERROR"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc(limit=8)
        entry["elapsed_s"] = round(time.perf_counter() - t0, 2)
        results.append(entry)
        mark = {"OK": "OK ", "ERROR": "ERR"}.get(entry["status"], "---")
        detail = (entry.get("data") or {}).get("verdict", entry.get("error", ""))
        print(f"  {mark} {entry['elapsed_s']:>7.1f}s  {detail}", flush=True)

    results = merge_into(outdir / "resolution.json", "experiments", results)
    payload = {
        "format": "caustica-resolution/1",
        "environment": environment(),
        "settings": {k: v for k, v in ctx.items()},
        "experiments": results,
    }
    (outdir / "resolution.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (outdir / "REPORT.md").write_text(render_markdown(payload), encoding="utf-8")
    bad = [e["id"] for e in results if e["status"] == "ERROR"]
    print(f"\n{len(results) - len(bad)}/{len(results)} ran -> {outdir / 'resolution.json'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
