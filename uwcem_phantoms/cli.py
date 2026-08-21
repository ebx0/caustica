"""Command line for the phantom submodule.

    python -m uwcem_phantoms list
    python -m uwcem_phantoms fetch --all
    python -m uwcem_phantoms tissues --model detailed --f0 1.5
    python -m uwcem_phantoms build 012304 --dx 0.4 --standoff 15 --noise 2 --pval
    python -m uwcem_phantoms dataset            # all nine, one grid -> data/phantoms/
    python -m uwcem_phantoms dataset --verify   # re-check what is on disk
    python -m uwcem_phantoms setup              # nine run-ready setups -> data/setups/
    python -m uwcem_phantoms setup --list       # what is stored, in words
    python -m uwcem_phantoms info _data/exports/....npz

Everything the GUI can do is reachable from here, because a headless Colab
session cannot open a browser window and a study script should not have to.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from uwcem_phantoms import catalog
from uwcem_phantoms.asset import PhantomAsset
from uwcem_phantoms.builder import build
from uwcem_phantoms.spec import PhantomSpec
from uwcem_phantoms.tissue import tissue_table


def _cmd_list(args) -> int:
    rows = catalog.status()
    print(f"{'id':<11} {'ACR':>3}  {'grid':<16} {'extent [mm]':<20} {'local':<12} description")
    print("-" * 104)
    for r in rows:
        have = ("mtype" if r["has_mtype"] else "") + ("+pval" if r["has_pval"] else "")
        grid = "x".join(str(n) for n in r["shape"])
        ext = "x".join(f"{e:g}" for e in r["extent_mm"])
        print(
            f"{r['id']:<11} {r['acr_class']:>3}  {grid:<16} {ext:<20} "
            f"{have or '-':<12} {r['acr_description']}"
        )
    print()
    print(catalog.CITATION)
    return 0


def _cmd_fetch(args) -> int:
    ids = list(catalog.PHANTOM_IDS) if args.all else args.phantom_id
    if not ids:
        print("give phantom ids or --all", file=sys.stderr)
        return 2
    last = [""]

    def progress(name: str, done: int, total: int) -> None:
        tag = f"{name} {done / 1e6:6.1f}/{total / 1e6:6.1f} MB"
        if tag != last[0]:
            print(f"\r  {tag}", end="", flush=True)
            last[0] = tag

    for pid in ids:
        print(f"{pid}:")
        catalog.fetch(pid, with_pval=not args.no_pval, force=args.force, progress=progress)
        print("\r  done" + " " * 40)
    return 0


def _cmd_tissues(args) -> int:
    table = tissue_table(args.model, f0=args.f0 * 1e6)
    if args.json:
        print(json.dumps(table.describe(), indent=2))
        return 0
    print(f"tissue model {args.model!r} at f0 = {args.f0} MHz")
    print(
        f"{'id':>3}  {'name':<38} {'c [m/s]':>9} {'rho':>8} {'a [Np/m]':>9} "
        f"{'a [dB/cm]':>10} {'beta':>6}  src"
    )
    print("-" * 118)
    for r in table.describe():
        flag = "~" if r["interpolated"] else " "
        print(
            f"{r['id']:>3}{flag} {r['name']:<38} {r['c_mid']:>9.1f} {r['rho_mid']:>8.1f} "
            f"{r['alpha_np_m']:>9.2f} {r['alpha_db_cm']:>10.3f} {r['beta_mid']:>6.2f}  "
            f"{r['source'][:40]}"
        )
    print("\n'~' marks a row interpolated along the UWCEM water-content ordering.")
    return 0


def spec_from_args(args) -> PhantomSpec:
    """The :class:`PhantomSpec` a parsed ``build`` command line describes.

    Split out of :func:`_cmd_build` so it can be checked against the launcher's
    "same thing from the command line" line without running a build: a printed
    command that does not reproduce the spec it claims to is worse than no
    command at all.
    """
    return PhantomSpec(
        phantom_id=args.phantom_id,
        f0_mhz=args.f0,
        simplify={
            "tissue_model": args.model,
            "drop_muscle": args.drop_muscle,
            "drop_skin": args.drop_skin,
            "remove_islands_vox": args.remove_islands,
            "fill_holes": args.fill_holes,
            "smooth_iterations": args.smooth,
        },
        resolution={"dx_mm": args.dx, "method": args.resample},
        crop={"mode": args.crop, "margin_mm": args.margin},
        domain={
            "standoff_mm": args.standoff,
            "backing_mm": args.backing,
            "lateral_margin_mm": args.lateral,
            "fft_friendly": not args.no_fft_pad,
            "max_voxels": args.max_voxels,
        },
        heterogeneity={
            "use_pval": args.pval,
            "noise_pct": args.noise,
            "correlation_mm": args.correlation,
            "seed": args.seed,
        },
        name=args.name,
        note=args.note,
    )


def _cmd_build(args) -> int:
    spec = spec_from_args(args)
    if args.dry_run:
        print(spec.model_dump_json(indent=2))
        return 0

    asset = build(spec, progress=lambda m, f: print(f"[{f:4.0%}] {m}"))
    print()
    print(asset.summary())
    for w in asset.meta["warnings"]:
        print(f"WARNING: {w}")
    path = asset.save(args.out, store_properties=args.store_properties)
    print(f"\nwrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    print("import it with:  uwcem_phantoms.load_phantom(path).to_medium()")
    return 0


def _cmd_setup(args) -> int:
    from uwcem_phantoms.setup import (
        SetupError,
        build_setups,
        load_setup,
        setup_names,
        verify_setups,
    )

    try:
        if args.list:
            names = setup_names(args.out)
            if args.json:
                print(json.dumps(names, indent=2))
                return 0
            if not names:
                print("no setups stored; run `setup` to build them", file=sys.stderr)
                return 1
            for n in names:
                print(load_setup(n, out_dir=args.out, with_medium=False).summary())
                print()
            return 0
        if args.verify:
            report = verify_setups(
                args.out, args.data, progress=lambda m: print(m, file=sys.stderr)
            )
            if args.json:
                print(json.dumps(report, indent=2))
                return 0
            print(f"OK: {report['count']} setup(s) passed every check")
            return 0
        manifest = build_setups(
            args.phantom_id or None,
            out_dir=args.out,
            data_dir=args.data,
            amplitude_pa=args.amplitude * 1e6,
            pml_mm=args.pml,
        )
        for e in manifest["setups"]:
            print(
                f"{e['name']:<16} water {e['water_path_mm']:6.2f} mm  "
                f"clearance {e['min_clearance_mm']:6.2f} mm  "
                f"focus {e['focus_below_skin_mm']:6.2f} mm below skin (class {e['focus_class']})"
            )
        print(f"\n{len(manifest['setups'])} setup(s) written")
    except SetupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_dataset(args) -> int:
    from uwcem_phantoms.dataset import (
        DatasetError,
        build_dataset,
        plan_dataset,
        verify_dataset,
    )

    depth = args.depth if args.depth else None  # --depth 0 means "no ceiling"
    try:
        if args.verify:
            # verify checks what the manifest recorded; a subset of ids or a
            # build flag has nothing to attach to — dropping them silently
            # would answer a different question than the one asked.
            if args.phantom_id or (
                args.dx,
                args.f0,
                args.margin,
                args.front_gap,
                args.depth,
            ) != (
                0.25,
                1.0,
                5.0,
                20.0,
                120.0,
            ):
                print(
                    "--verify checks the existing dataset against its manifest; "
                    "phantom ids and build flags do not apply",
                    file=sys.stderr,
                )
                return 2
            if args.json:
                report = verify_dataset(args.out, progress=lambda m: print(m, file=sys.stderr))
                print(json.dumps(report, indent=2))
                return 0
            report = verify_dataset(args.out, progress=print)
            print(f"OK: {len(report['files'])} file(s) passed every check")
            return 0
        if args.dry_run:
            plan = plan_dataset(
                args.phantom_id or None,
                dx_mm=args.dx,
                f0_mhz=args.f0,
                margin_mm=args.margin,
                front_gap_mm=args.front_gap,
                depth_limit_mm=depth,
            )
            print(plan.summary())
            return 0
        build_dataset(
            args.phantom_id or None,
            dx_mm=args.dx,
            f0_mhz=args.f0,
            margin_mm=args.margin,
            front_gap_mm=args.front_gap,
            depth_limit_mm=depth,
            out_dir=args.out,
            progress=print,
            force=args.force,
        )
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_info(args) -> int:
    asset = PhantomAsset.load(args.path)
    print(asset.summary())
    meta = asset.meta
    if args.log:
        print("\nbuild log:")
        for line in meta.get("log", []):
            print(f"  - {line}")
    if meta.get("warnings"):
        print("\nwarnings:")
        for w in meta["warnings"]:
            print(f"  ! {w}")
    if args.json:
        print(json.dumps(meta, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m uwcem_phantoms",
        description="UWCEM numerical breast phantoms -> simulation-ready media",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="show the phantom catalog and what is downloaded")
    sp.set_defaults(func=_cmd_list)

    sp = sub.add_parser("fetch", help="download raw phantom files")
    sp.add_argument("phantom_id", nargs="*")
    sp.add_argument("--all", action="store_true", help="every phantom in the catalog")
    sp.add_argument("--no-pval", action="store_true", help="skip the large pval archives")
    sp.add_argument("--force", action="store_true", help="re-download even if present")
    sp.set_defaults(func=_cmd_fetch)

    sp = sub.add_parser("tissues", help="print the acoustic property table")
    sp.add_argument("--model", default="detailed", choices=("detailed", "grouped", "simple"))
    sp.add_argument("--f0", type=float, default=1.0, help="frequency [MHz]")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_tissues)

    sp = sub.add_parser("build", help="build and export one phantom")
    sp.add_argument("phantom_id")
    sp.add_argument("--f0", type=float, default=1.0, help="frequency [MHz] for the alpha power law")
    sp.add_argument("--model", default="detailed", choices=("detailed", "grouped", "simple"))
    sp.add_argument("--dx", type=float, default=None, help="target spacing [mm] (default 0.5)")
    sp.add_argument("--resample", default="smooth", choices=("nearest", "smooth"))
    sp.add_argument("--crop", default="breast", choices=("none", "tissue", "breast", "manual"))
    sp.add_argument("--margin", type=float, default=5.0, help="crop margin [mm]")
    sp.add_argument("--standoff", type=float, default=0.0, help="coupling medium in front [mm]")
    sp.add_argument("--backing", type=float, default=0.0, help="coupling medium behind [mm]")
    sp.add_argument("--lateral", type=float, default=0.0, help="transverse margin added [mm]")
    sp.add_argument("--no-fft-pad", action="store_true", help="do not pad to 2/3/5/7-smooth sizes")
    sp.add_argument("--max-voxels", type=int, default=None)
    sp.add_argument("--drop-muscle", action="store_true")
    sp.add_argument("--drop-skin", action="store_true")
    sp.add_argument("--fill-holes", action="store_true")
    sp.add_argument("--remove-islands", type=int, default=0, metavar="VOX")
    sp.add_argument("--smooth", type=int, default=0, metavar="N", help="majority-filter passes")
    sp.add_argument("--pval", action="store_true", help="use the repository's per-voxel p values")
    sp.add_argument("--noise", type=float, default=0.0, help="scatterer noise [%%]")
    sp.add_argument("--correlation", type=float, default=0.0, help="noise correlation length [mm]")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--name", default=None)
    sp.add_argument("--note", default="")
    sp.add_argument("--out", default=None, help="output path (default: _data/exports/<name>.npz)")
    sp.add_argument("--store-properties", default="auto", choices=("auto", "always", "never"))
    sp.add_argument("--dry-run", action="store_true", help="print the spec JSON and stop")
    sp.set_defaults(func=_cmd_build)

    sp = sub.add_parser(
        "dataset",
        help="build (or verify) the standard aligned dataset: every phantom on ONE "
        "grid, same dx, front faces and transverse centres aligned, pval on",
    )
    sp.add_argument("phantom_id", nargs="*", help="subset of ids (default: all nine)")
    sp.add_argument("--dx", type=float, default=0.25, help="target spacing [mm]")
    sp.add_argument("--f0", type=float, default=1.0, help="frequency [MHz] for the alpha power law")
    sp.add_argument("--margin", type=float, default=5.0, help="crop margin [mm]")
    sp.add_argument(
        "--front-gap",
        type=float,
        default=20.0,
        help="water in front of the skin [mm], all phantoms — this is the transducer "
        "budget: the PML sits inside it and a focused bowl needs room for its shell",
    )
    sp.add_argument(
        "--depth",
        type=float,
        default=120.0,
        help="hard ceiling on the propagation axis [mm]; deeper tissue is CUT off "
        "the back and the loss is recorded per phantom (0 = no limit)",
    )
    sp.add_argument("--out", default=None, help="output directory (default: data/phantoms)")
    sp.add_argument(
        "--verify", action="store_true", help="check existing files instead of building"
    )
    sp.add_argument("--json", action="store_true", help="with --verify: dump the report as JSON")
    sp.add_argument("--dry-run", action="store_true", help="survey and print the common grid, stop")
    sp.add_argument(
        "--force",
        action="store_true",
        help="skip the free-RAM rail, and allow overwriting a directory whose "
        "manifest was built with different parameters",
    )
    sp.set_defaults(func=_cmd_dataset)

    sp = sub.add_parser(
        "setup",
        help="build (or verify, or list) the stored simulation-ready setups: "
        "one phantom + one transducer + one focus per file, in data/setups",
    )
    sp.add_argument("phantom_id", nargs="*", help="subset of ids (default: all nine)")
    sp.add_argument("--out", default=None, help="setup directory (default: data/setups)")
    sp.add_argument("--data", default=None, help="phantom dataset directory")
    sp.add_argument(
        "--amplitude", type=float, default=0.1, help="source pressure at the elements [MPa]"
    )
    sp.add_argument("--pml", type=float, default=5.0, help="PML thickness [mm]")
    sp.add_argument("--list", action="store_true", help="list what is stored and stop")
    sp.add_argument("--verify", action="store_true", help="re-check stored setups and stop")
    sp.add_argument("--json", action="store_true", help="with --verify/--list: dump JSON")
    sp.set_defaults(func=_cmd_setup)

    sp = sub.add_parser("info", help="describe an exported phantom file")
    sp.add_argument("path", type=Path)
    sp.add_argument("--log", action="store_true", help="print the build log")
    sp.add_argument("--json", action="store_true", help="dump the full metadata")
    sp.set_defaults(func=_cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
