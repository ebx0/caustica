"""Command line: build a scenario, plan it, run it, write the result folder.

    python -m apps.focus_study --list
    python -m apps.focus_study water_bowl --dry-run
    python -m apps.focus_study water_bowl --out out/water

Design rule: NOTHING expensive happens before the planner has spoken. The
plan is printed (and saved) first, so a run that would take an hour is
visible before it starts, not after.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import caustica as hs
import caustica.solvers as solvers
from apps.focus_study import __version__, analysis, figures, report, scenarios
from apps.focus_study.scenarios import Knobs, Setup
from caustica import planner

DEFAULT_OUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"


def _parse_harmonics(text: str) -> tuple[int, ...]:
    try:
        vals = tuple(sorted({int(t) for t in text.split(",") if t.strip()}))
    except ValueError:
        raise SystemExit(f"--harmonics must be comma-separated integers, got '{text}'") from None
    if not vals or vals[0] < 1:
        raise SystemExit(f"--harmonics must be positive integers, got '{text}'")
    return vals


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m apps.focus_study",
        description="Characterize a HIFU focus with caustica (CPU / numpy backend).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("scenario", nargs="?", default="water_bowl", help="scenario name")
    p.add_argument("--list", action="store_true", help="list scenarios and exit")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output folder (default: apps/outputs/<scenario>-<timestamp>)",
    )
    p.add_argument("--dx", type=float, default=0.4, help="grid spacing [mm]")
    p.add_argument("--f0", type=float, default=1.0, help="drive frequency [MHz]")
    p.add_argument("--amplitude", type=float, default=0.1, help="source pressure amplitude [MPa]")
    p.add_argument("--solver", choices=("linear", "westervelt"), default="westervelt")
    p.add_argument("--harmonics", default="1,2", help="harmonics to record, comma separated")
    p.add_argument("--pml-mm", type=float, default=4.0, help="PML thickness [mm]")
    p.add_argument(
        "--min-settle", type=int, default=8, help="minimum settling periods after time-of-flight"
    )
    p.add_argument("--max-settle", type=int, default=40, help="settling cap [periods]")
    p.add_argument("--tol", type=float, default=0.01, help="steady-state convergence tolerance")
    p.add_argument("--gpu", default="A100", help="GPU to also report a datasheet estimate for")
    p.add_argument("--no-measure", action="store_true", help="skip the ~20-step local timing probe")
    p.add_argument("--dry-run", action="store_true", help="plan only: no solve, no figures")
    p.add_argument("--n-elements", type=int, default=32, help="spiral_array: element count")
    p.add_argument(
        "--steer-mm", type=float, default=4.0, help="spiral_array: off-axis steering [mm]"
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"focus_study {__version__} / caustica {hs.__version__}",
    )
    return p


def make_setup(args: argparse.Namespace) -> Setup:
    knobs = Knobs(
        dx=args.dx * 1e-3,
        f0=args.f0 * 1e6,
        amplitude=args.amplitude * 1e6,
        solver=args.solver,
        harmonics=_parse_harmonics(args.harmonics),
        min_settle_periods=args.min_settle,
        max_settle_periods=args.max_settle,
        convergence_tol=args.tol,
        pml_mm=args.pml_mm,
    )
    if args.scenario == "spiral_array":
        return scenarios.spiral_array(knobs, n_elements=args.n_elements, steer_mm=args.steer_mm)
    return scenarios.build(args.scenario, knobs)


def resolution_warnings(setup: Setup) -> list[str]:
    """Spatial resolution checks the user should see BEFORE paying for a run.

    The k-space PSTD scheme is accurate down to ~3 points per wavelength;
    every recorded harmonic needs that at ITS OWN frequency, which is what
    silently degrades multi-harmonic runs on a grid sized for f0.
    """
    out = []
    grid, c_min = setup.grid, setup.medium.c_min
    for n in setup.knobs.harmonics:
        ppw = grid.ppw(n * setup.knobs.f0, c_min)
        if ppw < 3.0:
            out.append(
                f"harmonic {n} is resolved by only {ppw:.2f} points per wavelength "
                f"(need >= 3): its amplitude is under-resolved — use "
                f"--dx {grid.dx * 1e3 * ppw / 3.0:.2f} or record fewer harmonics."
            )
    return out


def plan(setup: Setup, gpu: str, measure: bool) -> tuple[dict, str]:
    """Planner verdicts: measured-here (if asked) plus a datasheet GPU estimate."""
    common = dict(
        solver=setup.knobs.solver,
        harmonics=setup.knobs.harmonics,
        reference_point=setup.focus_vox,
    )
    est_gpu = planner.estimate(
        setup.grid, setup.medium, setup.source, setup.spec, gpu=gpu, **common
    )
    est_here = (
        planner.estimate(
            setup.grid, setup.medium, setup.source, setup.spec, gpu=gpu, measure=True, **common
        )
        if measure
        else est_gpu
    )
    text = "\n".join(
        [
            "--- planner ------------------------------------------------------",
            f"this machine [{est_here.source}]: t_step={est_here.t_step_s * 1e3:.1f} ms, "
            f"expected {est_here.t_expected_s:.0f} s, worst case {est_here.t_worst_s:.0f} s "
            f"({est_here.steps_expected}-{est_here.steps_worst} steps, spp={est_here.spp})",
            f"memory for this run: {est_here.vram_gib:.2f} GiB",
            est_gpu.summary(),
        ]
    )
    payload = {
        "source": est_here.source,
        "t_step_ms": round(est_here.t_step_s * 1e3, 3),
        "t_expected_s": est_here.t_expected_s,
        "t_worst_s": est_here.t_worst_s,
        "vram_gib": est_here.vram_gib,
        "vram_breakdown_bytes": est_here.vram_breakdown,
        "gpu": gpu,
        "gpu_t_expected_s": est_gpu.t_expected_s,
        "gpu_fits": est_gpu.fits,
        "warnings": list(est_here.warnings),
        "advice": list(est_gpu.advice),
    }
    return payload, text


def save_fields(outdir: Path, setup: Setup, result) -> Path:
    """Raw arrays as .npz — the library has no HDF5 contract yet (M10)."""
    org = analysis.region_origin(result)
    dx, apex = setup.grid.dx, setup.apex_vox
    arrays = {
        "amp_f0_pa": result.amp,
        "phase_f0_rad": result.phase,
        "p_max_pa": result.p_max,
        "x_mm": (np.arange(result.amp.shape[0]) + org[0] - apex[0]) * dx * 1e3,
        "y_mm": (np.arange(result.amp.shape[1]) + org[1] - apex[1]) * dx * 1e3,
        "z_mm": (np.arange(result.amp.shape[2]) + org[2] - apex[2]) * dx * 1e3,
        "sound_speed_m_s": setup.medium.c,
        "source_indices": setup.source.indices,
        "source_phases_rad": setup.source.phases,
    }
    for n in sorted(result.phasors):
        arrays[f"amp_h{n}_pa"] = result.harmonic_amp(n)
    if setup.labels is not None:
        arrays["tissue_labels"] = setup.labels
    path = outdir / "fields.npz"
    np.savez_compressed(path, **arrays)
    return path


def main(argv: list[str] | None = None) -> int:
    # The planner and the report both emit non-ASCII (±, µ); the default
    # Windows console codepage would mangle them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    if args.list:
        print("scenarios:")
        for name, fn in sorted(scenarios.BUILDERS.items()):
            head = (fn.__doc__ or "").strip().splitlines()[0]
            print(f"  {name:<16} {head}")
        return 0

    setup = make_setup(args)
    geo = setup.geometry_summary
    print(f"=== focus_study: {setup.title} ===")
    print(setup.description)
    print(
        f"grid {'x'.join(map(str, geo['shape']))} @ dx={geo['dx_mm']} mm "
        f"({geo['n_voxels']:,} voxels), ppw={geo['ppw_at_f0']} at f0, "
        f"{geo['source_voxels']:,} source voxels"
    )
    for note in setup.notes:
        print(f"  ! {note}")
    for warning in resolution_warnings(setup):
        print(f"  ! {warning}")

    plan_payload, plan_text = plan(setup, args.gpu, measure=not args.no_measure)
    print(plan_text)
    if args.dry_run:
        print("\n(dry run — nothing solved)")
        return 0

    outdir = args.out or DEFAULT_OUT_ROOT / f"{setup.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plan.txt").write_text(plan_text, encoding="utf-8")

    print(f"\nsolving ({setup.knobs.solver}, numpy backend) -> {outdir}", flush=True)
    t0 = time.perf_counter()
    result = solvers.get(setup.knobs.solver)().run(
        setup.grid,
        setup.medium,
        setup.source,
        setup.spec,
        backend="numpy",
        reference_point=setup.focus_vox,
        harmonics=setup.knobs.harmonics,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"done in {elapsed:.1f} s — {result.steps_total:,} steps, "
        f"converged at period {result.converged_period}"
        f"{' (SETTLE CAP HIT)' if result.settle_capped else ''}"
    )

    metrics = analysis.analyze(setup, result)
    prof = analysis.profiles(setup, result)
    figs = figures.make_all(setup, result, prof, outdir)
    save_fields(outdir, setup, result)
    report.write_metrics(
        outdir,
        {
            "app": f"focus_study {__version__}",
            "caustica": hs.__version__,
            "scenario": setup.name,
            "command": " ".join(sys.argv[1:]),
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(elapsed, 2),
            "setup": geo,
            "planner": plan_payload,
            **metrics,
        },
    )
    report.write_markdown(outdir, setup, metrics, plan_payload, figs, elapsed)
    html = report.write_html(outdir, setup, metrics, plan_payload, figs, elapsed)

    pk = metrics["peak"]
    print(
        f"peak {pk['p_mpa']} MPa at z={pk['position_mm_from_apex']['z']} mm "
        f"(gain x{pk['gain_vs_source']}), -6 dB spot "
        f"{metrics['focal_spot']['lateral_x_6db']['width_mm']} x "
        f"{metrics['focal_spot']['axial_6db']['width_mm']} mm (lateral x axial)"
    )
    if "vs_oneill" in metrics:
        o = metrics["vs_oneill"]
        print(
            f"vs O'Neil: r = {o['pearson_r_focal_region']} in the focal window, "
            f"{o['pearson_r_axial_shape']} over the whole axis"
        )
    print(f"report: {html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
