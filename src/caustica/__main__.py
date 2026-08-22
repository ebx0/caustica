"""Library command line: ``caustica <command>`` (also ``python -m caustica``).

M10b ships ``validate`` — every check that does not require solving, so a
typo'd or impossible job dies HERE, on the local machine, instead of after a
Colab session has been booked and a dataset staged. The M10c runner adds
``run`` on top of the same job contract; M10d adds ``report`` — local HTML +
figures from a run's output folder (or from its preview package alone).
M10h adds the ``caustica`` console entry point and ``example`` — packaged,
zero-data jobs copied *out* of the install before running (running them in
place would write into site-packages, see ``caustica.examples``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import caustica


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="caustica",
        description="caustica job tools (job format: caustica-job/1)",
    )
    p.add_argument("--version", action="version", version=f"caustica {caustica.__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser(
        "validate",
        help="check a job file without solving (schema, files, geometry, PML, focus, ppw)",
    )
    v.add_argument("job", type=Path, help="path to a caustica-job/1 JSON file")
    v.add_argument(
        "--fast",
        action="store_true",
        help="skip medium construction (defers the solver-capability check to run time)",
    )

    r = sub.add_parser(
        "run",
        help="execute a job file: plan first, refuse OOM, solve, stamp (exit codes: "
        "0 ok, 2 config, 3 OOM, 4 solver, 5 interrupted-resumable)",
    )
    r.add_argument("job", type=Path, help="path to a caustica-job/1 JSON file")
    r.add_argument("--out", type=Path, default=None, help="output folder (default: runs/<name>)")
    r.add_argument(
        "--backend",
        choices=("auto", "numpy", "cupy"),
        default=None,
        help="override the job's backend field",
    )
    r.add_argument("--gpu", default="A100", help="GPU for the datasheet estimate")
    r.add_argument("--no-measure", action="store_true", help="skip the ~20-step timing probe")
    r.add_argument("--dry-run", action="store_true", help="plan only: no solve, no result")
    r.add_argument("--resume", action="store_true", help="continue from an existing checkpoint.npz")
    r.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="stop gracefully after this many hours, leaving a resumable checkpoint",
    )
    r.add_argument(
        "--checkpoint-every",
        type=int,
        default=8,
        help="checkpoint cadence in acoustic periods",
    )
    r.add_argument(
        "--status-interval",
        type=float,
        default=30.0,
        help="status.json heartbeat interval [s]",
    )
    r.add_argument(
        "--vram-limit-gib",
        type=float,
        default=None,
        help="refuse to start if the planner needs more than this (default: device VRAM on cupy)",
    )
    r.add_argument(
        "--allow-slow-cpu",
        action="store_true",
        help="accept a CPU run over the CAUSTICA_CPU_LIMIT_MIN (default 5 min) estimate gate",
    )
    r.add_argument(
        "--preview-only",
        action="store_true",
        help="write only the <=10 MB preview package + metrics, no result.h5 "
        "(disk-pressure opt-in; the full field is unrecoverable and a rerun solves again)",
    )
    r.add_argument("--stop-after-periods", type=int, default=None, help=argparse.SUPPRESS)

    ex = sub.add_parser(
        "example",
        help="copy a packaged zero-data example job into your directory "
        "(run the copy, never the packaged file); no name lists what ships",
    )
    ex.add_argument("name", nargs="?", default=None, help="example name (omit to list)")
    ex.add_argument(
        "--to",
        type=Path,
        default=Path("."),
        help="destination directory (default: current directory)",
    )

    rep = sub.add_parser(
        "report",
        help="render REPORT.md + index.html (+ figures) for a run output folder, "
        "from result.h5 or — with --preview — from the <=10 MB preview package alone",
    )
    rep.add_argument("outdir", type=Path, help="a runner output folder")
    rep.add_argument(
        "--preview",
        action="store_true",
        help="use preview.npz even if result.h5 is present (fast look, no full field read)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    # The reports use non-ASCII; the default Windows console codepage mangles it.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        from caustica.config.job import validate_job

        report = validate_job(args.job, fast=args.fast)
        print(report.render())
        return 0 if report.ok else 2
    if args.command == "run":
        import logging

        from caustica.runner import RunnerOptions, run_job_file

        # D33: the LIBRARY installs no logging handler on import; the CLI is
        # an application and turns logging on at its entry point — scoped to
        # the caustica logger so a notebook calling main() does not suddenly
        # see every third-party INFO record (review, 2026-08-22).
        clog = logging.getLogger("caustica")
        if not clog.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
            clog.addHandler(handler)
            clog.setLevel(logging.INFO)

        return run_job_file(
            args.job,
            RunnerOptions(
                out=args.out,
                backend=args.backend,
                gpu=args.gpu,
                measure=not args.no_measure,
                dry_run=args.dry_run,
                resume=args.resume,
                max_hours=args.max_hours,
                checkpoint_every=args.checkpoint_every,
                status_interval_s=args.status_interval,
                vram_limit_gib=args.vram_limit_gib,
                stop_after_periods=args.stop_after_periods,
                allow_slow_cpu=args.allow_slow_cpu,
                preview_only=args.preview_only,
            ),
        )
    if args.command == "example":
        from caustica import examples

        if args.name is None:
            for name in examples.available():
                print(name)
            return 0
        try:
            target = examples.copy(args.name, args.to)
        except (KeyError, OSError) as exc:
            # str(KeyError) is the repr of its message (adds quotes); OSError
            # covers FileExistsError plus the POSIX-only NotADirectoryError /
            # PermissionError a bad --to produces — no raw tracebacks here.
            msg = exc.args[0] if isinstance(exc, KeyError) else exc
            print(f"EXAMPLE ERROR: {msg}", file=sys.stderr)
            return 2
        print(f"copied: {target}")
        print(f"next:   caustica validate {target}")
        print(f"        caustica run {target}")
        return 0
    if args.command == "report":
        from caustica.report.run_report import report_out_dir

        try:
            html = report_out_dir(args.outdir, preview_only=args.preview)
        except Exception as exc:
            # A half-synced preview.npz or torn result.h5 is the NORMAL
            # failure mode on a Drive mount — the report command must say
            # what is wrong, not dump a traceback (exit 2, like validate).
            print(f"REPORT ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        print(f"report: {html}")
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
