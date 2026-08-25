"""ITRUSST PH1 water benchmarks: the first two, run as gates.

The International Transcranial Ultrasonic Stimulation Safety and Standards
consortium defined nine benchmarks of increasing geometric complexity and ran
eleven modelling codes through them (Aubry et al., *Benchmark problems for
transcranial ultrasound simulation: intercomparison of compressional wave
models*, J. Acoust. Soc. Am. 152(2), 1003, 2022; arXiv:2202.04552). The
milestone ladder puts all nine at the v0.1 gate, at the end.

The first two are in water, and that is what makes them worth pulling
forward: water has an exact reference. The Rayleigh surface integral over the
transducer's own geometry IS the answer for a baffled source in a homogeneous
medium, with or without absorption, so these two can be graded today without
waiting for a skull. The intercomparison's own reference for the water cases
was FOCUS, which is a Rayleigh-integral code — the same idea.

They are also the standing form of the lesson from 2026-08-24: two independent
source-model errors of 13-18 % of ABSOLUTE amplitude lived in this library for
months because every gate compared normalized shape. These benchmarks compare
a field in pascals.

**Definitions**, verbatim from the paper where numbers are quoted:

* water: 1500 m/s, 1000 kg/m^3.
* BM1 lossless; BM2 "uniform artificial absorption of 1 dB/cm at 500 kHz".
  The paper accepts either a frequency-independent absorption or a power law
  in frequency squared; this library's absorption is frequency-independent,
  the first of those, and at a single drive frequency the two coincide.
* drive 500 kHz, surface velocity 0.04 m/s, i.e. 60 kPa through the water
  impedance — and 1000 * 1500 * 0.04 = 60000 exactly, so the paper's two
  statements of the drive are one statement.
* SC1: focused bowl, "64 mm radius of curvature and a 64 mm aperture
  diameter" (f-number 1.0).
* SC2: plane piston, "diameter of 20 mm".
* comparison domain 120 mm axial by 70 mm lateral at 0.5 mm, i.e. 241 x 141
  points, with "the center of the source (rear of the bowl or center of the
  piston)" on the first axial plane, laterally centred. Benchmarks 1 to 6 are
  axisymmetric, so the paper compares on that 2-D plane; the runs themselves
  are 3-D, and so are these.
* the compared quantity is the complex pressure at 500 kHz over that domain,
  "from the transducer exit plane onward". That last clause is load-bearing:
  the Rayleigh integral is not valid ON the source surface, and the simulated
  field there is the injected drive rather than a radiated one. Measured on
  the piston: including the source plane puts L-infinity at 32 %, all of it
  from a single row of voxels on the disc face; starting one voxel past it
  gives 3.8 %, and that number does not move again for the rest of the
  domain. Both are reported below, so the choice is visible rather than
  quietly favourable.

**What agreement means here.** The paper reports L-infinity against FOCUS: for
the bowl, "seven models have L-infinity values of less than 1%" and "all
values less than 10%"; for the piston, "four models have L-infinity values of
less than 1%" with a maximum of 15%. Those are the numbers a new model reads
itself against — not a tolerance invented here.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from caustica.validation._verdict import EXIT_FAILED, EXIT_OK, Check, Gate, overall_verdict

_PAPER = (
    "Aubry et al., J. Acoust. Soc. Am. 152(2), 1003 (2022); arXiv:2202.04552 — "
    "reported L-infinity spread across eleven models against FOCUS"
)

#: Water, both benchmarks.
WATER_C = 1500.0  # [m/s]
WATER_RHO = 1000.0  # [kg/m^3]

F0 = 500.0e3  # [Hz]
U0 = 0.04  # surface normal velocity [m/s]
DRIVE_PA = WATER_RHO * WATER_C * U0  # 60 kPa, which the paper also states

#: 1 dB/cm at 500 kHz, as Np/m: 1 dB = ln(10)/20 Np, and per-cm to per-m is x100.
BM2_ALPHA_NP_M = 100.0 * np.log(10.0) / 20.0

#: The comparison domain, in the paper's own units.
DOMAIN_AXIAL_MM = 120.0
DOMAIN_LATERAL_MM = 70.0
DX_MM = 0.5

#: L-infinity spreads the eleven models showed against FOCUS, as reported.
REPORTED_SPREAD = {
    "SC1": {"best_seven_under_pct": 1.0, "all_under_pct": 10.0},
    "SC2": {"best_four_under_pct": 1.0, "all_under_pct": 15.0},
}


@dataclass(frozen=True)
class Benchmark:
    """One PH1 water benchmark."""

    id: str
    alpha_np_m: float
    note: str


@dataclass(frozen=True)
class SourceCondition:
    """One of the two transducers the benchmarks are driven with."""

    id: str
    kind: str  # "bowl" | "piston"
    roc_mm: float | None
    diameter_mm: float


BM1 = Benchmark("BM1", 0.0, "lossless water")
BM2 = Benchmark("BM2", BM2_ALPHA_NP_M, "water with 1 dB/cm at 500 kHz")
SC1 = SourceCondition("SC1", "bowl", 64.0, 64.0)
SC2 = SourceCondition("SC2", "piston", None, 20.0)

BENCHMARKS = {b.id: b for b in (BM1, BM2)}
SOURCES = {s.id: s for s in (SC1, SC2)}


def _geometry(dx: float, pml_mm: float, margin: int):
    """Grid, source voxel and comparison window for the paper's domain.

    The paper's domain is where the field is COMPARED; the sponge and the
    band-limited source's halo have to live outside it, or the comparison
    would be reading damped voxels.
    """
    from caustica.core.grid import Grid  # noqa: PLC0415
    from caustica.core.pml import PMLSpec  # noqa: PLC0415

    n_ax = int(round(DOMAIN_AXIAL_MM * 1e-3 / dx)) + 1
    n_lat = int(round(DOMAIN_LATERAL_MM * 1e-3 / dx)) + 1
    pad = int(np.ceil(pml_mm * 1e-3 / dx)) + margin
    shape = (n_lat + 2 * pad, n_lat + 2 * pad, n_ax + 2 * pad)
    grid = Grid(shape=shape, dx=dx, pml=PMLSpec(thickness=pml_mm * 1e-3))
    apex = (shape[0] // 2, shape[1] // 2, pad)
    window = (slice(pad, pad + n_lat), slice(pad, pad + n_ax))
    return grid, apex, window, (n_lat, n_ax)


def _source(sc: SourceCondition, grid, apex, discretization: str):
    from caustica.sources import bowl_cw_source, disc_cw_source  # noqa: PLC0415

    if sc.kind == "bowl":
        return bowl_cw_source(
            grid,
            f0=F0,
            amplitude=DRIVE_PA,
            aperture_radius=sc.diameter_mm / 2 * 1e-3,
            roc=sc.roc_mm * 1e-3,
            apex_vox=apex,
            discretization=discretization,
        )
    return disc_cw_source(
        grid,
        f0=F0,
        amplitude=DRIVE_PA,
        radius=sc.diameter_mm / 2 * 1e-3,
        center_vox=apex,
        discretization=discretization,
    )


def reference_field(sc: SourceCondition, bm: Benchmark, field_points: np.ndarray) -> np.ndarray:
    """The Rayleigh integral over the transducer's continuous surface.

    Exact for a baffled source in a homogeneous medium. Absorption enters as
    the imaginary part of the wavenumber, so the decay is integrated along
    every source-to-field path rather than applied to a lossless answer
    afterwards — which for a focused field are not the same thing.
    """
    from caustica.analytic import rayleigh_pressure  # noqa: PLC0415
    from caustica.analytic.geometry import spherical_cap_points  # noqa: PLC0415
    from caustica.geometry.offgrid import disc_points  # noqa: PLC0415

    lam_8 = WATER_C / F0 / 8.0  # the SUM's convergence, not the grid's
    if sc.kind == "bowl":
        a, roc = sc.diameter_mm / 2 * 1e-3, sc.roc_mm * 1e-3
        pts, _n, areas = spherical_cap_points(a, roc, lam_8)
    else:
        r = sc.diameter_mm / 2 * 1e-3
        n = max(int(np.ceil(np.pi * r**2 / lam_8**2)), 256)
        pts = disc_points(np.zeros(3), np.array([0.0, 0.0, 1.0]), r, n)
        areas = np.full(len(pts), np.pi * r**2 / n)
    k = 2.0 * np.pi * F0 / WATER_C + 1j * bm.alpha_np_m
    return rayleigh_pressure(pts, areas, U0, field_points, k, WATER_RHO, WATER_C)


def measure(
    benchmark: str = "BM1",
    source_condition: str = "SC1",
    *,
    solver: str = "linear",
    backend: str = "auto",
    dx_mm: float = DX_MM,
    pml_mm: float = 8.0,
    margin: int = 4,
    discretization: str = "offgrid",
    use_gpu_binary: bool = False,
) -> dict[str, Any]:
    """Run one benchmark and grade it against the Rayleigh integral.

    Returns metrics, not the plane: a gate reads numbers, and a 241 x 141
    array inside a JSON report helps nobody.
    """
    from caustica.materials import water  # noqa: PLC0415
    from caustica.medium import Medium  # noqa: PLC0415
    from caustica.solvers import CWRunSpec, get  # noqa: PLC0415

    bm, sc = BENCHMARKS[benchmark], SOURCES[source_condition]
    dx = dx_mm * 1e-3
    grid, apex, window, (n_lat, n_ax) = _geometry(dx, pml_mm, margin)
    medium = Medium.homogeneous(
        grid.shape, water(c=WATER_C, rho=WATER_RHO, alpha_np_m=bm.alpha_np_m)
    )
    src = _source(sc, grid, apex, discretization)

    # NO reference_point. The settle schedule is driven by how far the wave
    # must travel, and here that is the FAR END of a 120 mm comparison domain,
    # not the focus at 64 mm. Pointing it at the focus stops the run 32
    # periods in, and the field past ~95 mm is then simply one the wavefront
    # has not reached yet: measured 0.3 kPa where the Rayleigh integral says
    # 104 kPa, which looks exactly like a solver error and is not one.
    # Passing None makes the engine wait for every domain corner.
    spec = CWRunSpec(min_settle_periods=8, max_settle_periods=60, n_record_periods=2)
    t0 = time.perf_counter()
    kw: dict[str, Any] = (
        {"use_gpu_binary": use_gpu_binary} if solver == "kwave" else {"backend": backend}
    )
    res = get(solver)().run(grid, medium, src, spec, **kw)
    elapsed = time.perf_counter() - t0

    lat_sl, ax_sl = window
    amp = np.abs(np.asarray(res.phasor, dtype=np.complex128))
    plane = amp[lat_sl, apex[1], ax_sl]  # (lateral, axial)

    lat = (np.arange(n_lat) - (apex[0] - lat_sl.start)) * dx
    ax = np.arange(n_ax) * dx
    ll, aa = np.meshgrid(lat, ax, indexing="ij")
    field_points = np.column_stack([ll.ravel(), np.zeros(ll.size), aa.ravel()])
    ref = np.abs(reference_field(sc, bm, field_points)).reshape(plane.shape)

    # "From the transducer exit plane onward": past the rim for a bowl, past
    # the disc for a piston. See the module docstring for what including the
    # source plane does to the number and why it is an artefact of comparing
    # a radiated field against a surface integral on the surface itself.
    exit_m = (
        sc.roc_mm * 1e-3 - np.sqrt((sc.roc_mm * 1e-3) ** 2 - (sc.diameter_mm / 2 * 1e-3) ** 2)
        if sc.kind == "bowl"
        else 0.0
    )
    graded = ax > exit_m + 0.5 * dx

    scale = float(ref.max())
    ip, ja = np.unravel_index(int(plane.argmax()), plane.shape)
    ir, jr = np.unravel_index(int(ref.argmax()), ref.shape)
    return {
        "benchmark": bm.id,
        "source_condition": sc.id,
        "solver": solver,
        "backend": res.meta.get("backend"),
        "discretization": discretization,
        "note": bm.note,
        "grid": list(map(int, grid.shape)),
        "dx_mm": dx_mm,
        "points_per_wavelength": WATER_C / F0 / dx,
        "source_points": int(src.n_points),
        "comparison_plane": [int(n_lat), int(n_ax)],
        "comparison_start_mm": float(exit_m * 1e3),
        "l_inf_pct": float(100.0 * np.abs(plane[:, graded] - ref[:, graded]).max() / scale),
        "l2_pct": float(100.0 * np.sqrt(np.mean((plane[:, graded] - ref[:, graded]) ** 2)) / scale),
        "l_inf_pct_including_the_source_plane": float(100.0 * np.abs(plane - ref).max() / scale),
        "peak_pa": {"simulated": float(plane.max()), "reference": scale},
        "peak_ratio": float(plane.max() / scale),
        "peak_axial_mm": {"simulated": float(ax[ja] * 1e3), "reference": float(ax[jr] * 1e3)},
        "peak_lateral_mm": {"simulated": float(lat[ip] * 1e3), "reference": float(lat[ir] * 1e3)},
        "reported_spread_pct": REPORTED_SPREAD[sc.id],
        "steps_total": int(res.steps_total),
        "elapsed_s": round(elapsed, 2),
    }


# ------------------------------------------------------------------ gates


def evaluate(cases: dict[str, dict]) -> list[Gate]:
    """Turn the four measurements into gate verdicts. Pure — no solving.

    Graded against the intercomparison's own reported spread, not against a
    tolerance invented here: for the bowl "all values less than 10%", for the
    piston a maximum of 15%. Landing inside those is the claim "this model
    belongs in that table"; landing under 1% would be the claim "with the
    best of them", which is a different and stronger thing to say.
    """
    gates = []
    for sc_id in ("SC1", "SC2"):
        limit = REPORTED_SPREAD[sc_id]["all_under_pct"]
        checks = []
        for bm_id in ("BM1", "BM2"):
            key = f"{bm_id}-{sc_id}"
            data = cases.get(key) or {}
            checks.append(
                Check.at_most(
                    f"{key}: L-infinity vs the Rayleigh integral",
                    None if data.get("error") else data.get("l_inf_pct"),
                    limit,
                    " %",
                ).citing(_PAPER)
            )
            checks.append(
                Check.at_most(
                    f"{key}: peak position error",
                    None
                    if data.get("error")
                    else abs(
                        data.get("peak_axial_mm", {}).get("simulated", float("nan"))
                        - data.get("peak_axial_mm", {}).get("reference", float("nan"))
                    ),
                    2 * DX_MM,
                    " mm",
                ).citing(_PAPER)
            )
        gates.append(
            Gate(
                id=f"M21.PH1-{sc_id}",
                criterion=(
                    f"ITRUSST PH1 benchmarks 1 and 2 with source condition {sc_id}: the "
                    f"field over the paper's comparison domain agrees with the Rayleigh "
                    f"integral to within the spread the intercomparison itself reported "
                    f"({limit:g} % L-infinity across all eleven models), and the peak "
                    f"lands within two voxels"
                ),
                required=4,
                checks=checks,
            )
        )
    return gates


def run(
    *,
    out: str | Path | None = None,
    solver: str = "linear",
    backend: str = "auto",
    dx_mm: float = DX_MM,
    use_gpu_binary: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[int, dict]:
    """Run BM1 and BM2 against both source conditions and write a report."""
    import platform  # noqa: PLC0415
    from datetime import datetime, timezone  # noqa: PLC0415

    from caustica.env import env_report, git_commit  # noqa: PLC0415

    started = time.perf_counter()
    env = env_report()
    log("--- environment ----------------------------------------------")
    for key, value in env.items():
        log(f"  {key}: {value}")

    stamp = datetime.now(timezone.utc)
    root = Path(out) if out is not None else Path("benchmarks/reports/itrusst")
    tag = (env.get("resolved_backend") or platform.node() or "run").lower()
    outdir = root / f"{tag}-{stamp.strftime('%Y%m%d-%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"\nsolver: {solver}   dx: {dx_mm} mm   report folder: {outdir}")

    cases: dict[str, dict] = {}
    for bm_id in ("BM1", "BM2"):
        for sc_id in ("SC1", "SC2"):
            key = f"{bm_id}-{sc_id}"
            log(f"\n=== {key} ({BENCHMARKS[bm_id].note}, {SOURCES[sc_id].kind}) ===")
            try:
                cases[key] = measure(bm_id, sc_id, solver=solver, backend=backend, dx_mm=dx_mm)
            except Exception as exc:  # noqa: BLE001 - one case must not lose the rest
                message = f"{type(exc).__name__}: {exc}"
                log(f"  raised {message}")
                cases[key] = {"error": message}
                continue
            d = cases[key]
            log(f"  L-infinity: {d['l_inf_pct']:.2f} %   L2: {d['l2_pct']:.2f} %")
            log(
                f"  peak: {d['peak_pa']['simulated'] / 1e3:.1f} kPa vs "
                f"{d['peak_pa']['reference'] / 1e3:.1f} kPa (x{d['peak_ratio']:.4f}) at "
                f"{d['peak_axial_mm']['simulated']:.1f} mm vs "
                f"{d['peak_axial_mm']['reference']:.1f} mm"
            )
            log(f"  {d['steps_total']} steps in {d['elapsed_s']} s on {d['backend']}")

    gates = evaluate(cases)
    payload = {
        "format": "caustica-itrusst/1",
        "generated": stamp.isoformat(timespec="seconds"),
        "caustica": env.get("caustica"),
        "git_commit": git_commit(),
        "host": platform.node(),
        "solver": solver,
        "dx_mm": dx_mm,
        "reference": (
            "Rayleigh surface integral over the transducer's own geometry, which is "
            "exact for a baffled source in a homogeneous medium; the intercomparison "
            "used FOCUS, a Rayleigh-integral code, for the same reason"
        ),
        "paper": _PAPER,
        "cases": cases,
        "gates": [g.as_dict() for g in gates],
        "verdict": overall_verdict(gates),
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    (outdir / "itrusst.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (outdir / "REPORT.md").write_text(render_markdown(payload), encoding="utf-8")

    log("\n--- verdict --------------------------------------------------")
    for g in gates:
        log(f"  {g.verdict:<11} {g.id}: {g.criterion}")
    log(f"\noverall: {payload['verdict']}  ({payload['elapsed_s']} s)")
    log(f"report: {outdir / 'REPORT.md'}")
    return (EXIT_OK if payload["verdict"] == "PASS" else EXIT_FAILED), payload


def render_markdown(payload: dict) -> str:
    lines = [
        "# ITRUSST PH1 water benchmarks",
        "",
        f"Reference: {payload['reference']}.",
        "",
        f"Paper: {payload['paper']}",
        "",
        "| case | medium | source | L-inf % | L2 % | peak kPa (sim/ref) | ratio | peak z mm |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, d in payload["cases"].items():
        if d.get("error"):
            lines.append(f"| {key} | | | | | {d['error']} | | |")
            continue
        lines.append(
            f"| {key} | {d['note']} | {SOURCES[d['source_condition']].kind} | "
            f"{d['l_inf_pct']:.2f} | {d['l2_pct']:.2f} | "
            f"{d['peak_pa']['simulated'] / 1e3:.1f} / {d['peak_pa']['reference'] / 1e3:.1f} | "
            f"{d['peak_ratio']:.4f} | "
            f"{d['peak_axial_mm']['simulated']:.1f} / {d['peak_axial_mm']['reference']:.1f} |"
        )
    lines += ["", "## Gates", ""]
    for g in payload["gates"]:
        lines.append(f"**{g['verdict']}  {g['id']}** — {g['criterion']}")
        lines.append("")
        for c in g["checks"]:
            lines.append(f"- `{c['verdict']}` {c['name']}: {c['detail']}")
        lines.append("")
    env = {k: payload[k] for k in ("caustica", "git_commit", "host", "solver", "dx_mm")}
    lines += ["## Run", "", "```json", json.dumps(env, indent=2, default=str), "```"]
    return "\n".join(lines) + "\n"
