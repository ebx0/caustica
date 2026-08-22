"""``python -m caustica.validation <suite>`` — run a milestone gate suite.

Separate from ``caustica``'s own CLI on purpose: ``caustica run`` is what a
user does with the library, this is what the PROJECT does to itself. Keeping
the two apart means a validation suite can be as opinionated as it likes
(it books a whole GPU for minutes) without that opinion leaking into the
tool people install.
"""

from __future__ import annotations

import argparse
import sys

from caustica.validation.gpu_gates import DEFAULT_TARGETS_GIB, EXIT_ENV


def _targets(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part) for part in text.replace(",", " ").split())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a list of GiB numbers: {text!r}") from exc
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"targets must be positive GiB values, got {text!r}")
    return values


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m caustica.validation",
        description="milestone gate suites (they measure, they do not simulate for you)",
    )
    sub = p.add_subparsers(dest="suite", required=True)

    g = sub.add_parser(
        "gpu-gates",
        help="close M7's and M8's on-device criteria in one run: calibrate, walk a "
        "VRAM ladder, refuse the rung above the device, compare numpy vs cupy, "
        "and write a stamped PASS/FAIL report (exit: 0 all gates pass, "
        "2 no usable GPU, 4 a gate failed or is incomplete)",
    )
    g.add_argument(
        "--out",
        default=None,
        help="report root (default: benchmarks/reports/gpu_gates); one timestamped "
        "subfolder per run, holding the report AND every rung's output folder",
    )
    g.add_argument(
        "--max-vram-gib",
        type=float,
        default=None,
        help="pretend the device has at most this much free VRAM (shrinks the ladder)",
    )
    g.add_argument(
        "--targets",
        type=_targets,
        default=DEFAULT_TARGETS_GIB,
        help="ladder targets in GiB (default: 2,8,14,28; clipped to the device)",
    )
    g.add_argument("--dx-mm", type=float, default=0.30, help="voxel size (M7 says 0.30)")
    g.add_argument("--solver", default="westervelt", choices=("westervelt", "linear"))
    g.add_argument(
        "--settle-periods",
        type=int,
        default=2,
        help="settling periods per rung; min == max, so the step count is deterministic "
        "and the time gate measures the timing model, not the convergence heuristic",
    )
    g.add_argument(
        "--no-oom-rung",
        action="store_true",
        help="skip the deliberately-too-large rung (leaves M8's refusal gate INCOMPLETE)",
    )
    g.add_argument(
        "--no-parity",
        action="store_true",
        help="skip the numpy-vs-cupy comparison (leaves M7's parity gate INCOMPLETE)",
    )
    g.add_argument(
        "--gpu",
        default=None,
        help="datasheet key from gpu_db.json; detected from the device by default",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    if args.suite == "gpu-gates":
        from caustica.validation.gpu_gates import gpu_gates

        code, _ = gpu_gates(
            out=args.out,
            max_vram_gib=args.max_vram_gib,
            targets_gib=tuple(args.targets),
            dx_mm=args.dx_mm,
            solver=args.solver,
            settle_periods=args.settle_periods,
            oom_rung=not args.no_oom_rung,
            parity=not args.no_parity,
            gpu_key=args.gpu,
        )
        return code
    return EXIT_ENV  # pragma: no cover - argparse rejects unknown suites first


if __name__ == "__main__":
    raise SystemExit(main())
