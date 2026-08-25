"""Everything the library can be graded on, in one run, on one big card.

The validation this project has accumulated is spread across a gate suite, a
benchmark suite and five measurement harnesses, and each of them was sized for
whatever machine was available when it was written. On a card with 80 GB this
is the wrong shape: the interesting rungs are the ones a laptop refused, and
the interesting question is whether they all still hold at once, on the same
commit, on the same device, before a dataset is generated from any of it.

So this runs them in one pass and writes one summary. Every stage is a
subprocess: a stage that dies takes its own exit code and its own tail into
the report and nothing else with it, because a five-hour run that loses its
last two hours to a crash in its third stage is worse than useless.

Stages, in the order they run (fast and cheap first, so a broken checkout is
found in seconds rather than in hours)::

    env         what device, what commit, what versions
    tests       the library's own suite
    analytic    the closed-form gates, full size, absolute amplitude included
    gpu-gates   the on-device VRAM ladder and the numpy/cupy parity check
    geometry    what was ordered against what was built
    invariants  ten properties the code must have whatever the physics says:
                phase, superposition, scaling, the stored file, resume,
                determinism, placement, the sponge, dt, steering
    hetero      layered media against the exact transfer matrix, and k-Wave
    array       the spiral driven by YOUR phase vector
    nonlinear   harmonics: the scaling law, convergence, k-Wave, the
                absorption model's cost at 2f0
    resolution  fine spacing, small volumes, and one attribution
    itrusst     the published intercomparison, native and k-Wave
    compare     the same job on every registered engine

Run it::

    python scripts/colab_validation.py --phases my_phases.npy
    python scripts/colab_validation.py --only array,nonlinear --phases das:3,0,62
    python scripts/colab_validation.py --profile quick     # a shakedown, ~20 min

``--profile full`` is the default and is sized for a card with 80 GB. It is
long: expect hours, most of it in ``itrusst`` and ``nonlinear``. Nothing here
needs babysitting -- the summary is rewritten after every stage, so it can be
read while the rest is still running.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


# --------------------------------------------------------------------------
# stage definitions
# --------------------------------------------------------------------------


def stages(args: argparse.Namespace, out: Path) -> list[tuple[str, str, list[str], int]]:
    """``(id, what it answers, argv, timeout_s)``, cheapest first.

    The ladders differ between profiles and nowhere else: ``quick`` is the
    same questions at spacings a shakedown can afford, so a green quick run
    means the plumbing works and says nothing about the physics.
    """
    py = sys.executable
    full = args.profile == "full"
    rep = out / "reports"

    def sc(name: str) -> list[str]:
        return [py, str(REPO / "scripts" / name)]

    def val(sub: str) -> list[str]:
        return [py, "-m", "caustica.validation", sub]

    items: list[tuple[str, str, list[str], int]] = [
        (
            "tests",
            "does the library's own suite still pass on this machine?",
            [py, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]
            + ([] if full else ["-m", "not slow"]),
            7200,
        ),
        (
            "analytic",
            "do the solvers still match the closed forms, absolute amplitude included?",
            val("run-analytic") + ["--out", str(rep / "analytic"), "--size", "full"],
            7200,
        ),
        (
            "gpu-gates",
            "how far up a VRAM ladder does this card go, and do numpy and cupy agree?",
            val("gpu-gates") + ["--out", str(rep / "gpu_gates"), "--targets", args.vram_targets],
            21600,
        ),
        (
            "geometry",
            "was the geometry that got built the geometry that was ordered?",
            sc("dev_geometry.py") + ["--out", str(rep / "geometry")],
            3600,
        ),
        (
            "invariants",
            "the phase, superposition, the file, resume, determinism, the sponge",
            sc("dev_invariants.py")
            + [
                "--out",
                str(rep / "invariants"),
                "--ppw",
                "12" if full else "5",
                "--ladder",
                "6,10,16,20" if full else "5,8",
                "--pml-ladder",
                "2,4,6,8,12,16" if full else "2,6",
                "--cfl-ladder",
                "0.48,0.36,0.24,0.12" if full else "0.48,0.24",
            ],
            21600,
        ),
        (
            "hetero",
            "does a layered medium match the exact transfer matrix, and k-Wave?",
            sc("dev_hetero.py")
            + [
                "--out",
                str(rep / "hetero"),
                "--h4-ppw",
                "12" if full else "6",
                "--ladder",
                "8,12,16,24,32,48" if full else "8,16",
            ],
            21600,
        ),
        (
            "array",
            "does the spiral do what your phase vector says?",
            sc("dev_array_phases.py")
            + [
                "--out",
                str(rep / "array_phases"),
                "--phases",
                args.phases,
                "--f0",
                str(args.f0),
                "--amplitude",
                str(args.amplitude),
                "--ppw",
                "12" if full else "5",
                "--ppw-cross",
                "10" if full else "4",
                "--ladder",
                "6,8,10,12" if full else "4,5",
            ],
            43200,
        ),
        (
            "nonlinear",
            "is 2f0 a measurement, and at what spacing does it stop moving?",
            sc("dev_nonlinear.py")
            + [
                "--out",
                str(rep / "nonlinear"),
                "--ppw",
                "16" if full else "8",
                "--ladder",
                "6,8,10,12,14,16,20" if full else "6,8",
                "--drives",
                "25e3,50e3,100e3,200e3,400e3,800e3" if full else "50e3,200e3",
            ],
            43200,
        ),
        (
            "resolution",
            "does the answer converge at fine spacing in a small volume?",
            sc("dev_resolution.py")
            + ["--out", str(rep / "resolution")]
            + (
                ["--ladder", "0.2,0.1,0.05,0.025"]
                if full
                else ["--only", "R1", "--ladder", "0.2,0.1"]
            ),
            21600,
        ),
        (
            "itrusst-native",
            "the published intercomparison, native, at and beyond the paper's spacing",
            val("itrusst") + ["--out", str(rep / "itrusst_native"), "--dx-mm", args.itrusst_dx],
            43200,
        ),
        (
            "itrusst-kwave",
            "the same benchmark through k-Wave's CUDA binary",
            val("itrusst")
            + [
                "--out",
                str(rep / "itrusst_kwave"),
                "--solver",
                "kwave",
                "--gpu-binary",
                "--dx-mm",
                args.itrusst_dx,
            ],
            43200,
        ),
        (
            "compare",
            "what do the registered engines disagree about on one job?",
            val("compare") + ["--example", "water_bowl_mini", "--out", str(rep / "compare")],
            21600,
        ),
    ]
    return items


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------


def environment() -> dict:
    env: dict[str, Any] = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import caustica

        env["caustica"] = caustica.__version__
    except Exception as exc:
        env["caustica"] = f"IMPORT FAILED: {exc}"
    for cmd, key in (
        (["git", "rev-parse", "HEAD"], "commit"),
        (["git", "status", "--short"], "dirty"),
    ):
        try:
            env[key] = subprocess.run(
                cmd, cwd=REPO, capture_output=True, text=True, timeout=30
            ).stdout.strip()
        except Exception:
            env[key] = "(unavailable)"
    try:
        import numpy

        env["numpy"] = numpy.__version__
    except Exception:
        pass
    try:
        import cupy

        env["cupy"] = cupy.__version__
        props = cupy.cuda.runtime.getDeviceProperties(0)
        free, total = cupy.cuda.runtime.memGetInfo()
        env["gpu"] = props["name"].decode()
        env["compute_capability"] = f"{props['major']}.{props['minor']}"
        env["vram_total_gib"] = round(total / 2**30, 1)
        env["vram_free_gib"] = round(free / 2**30, 1)
        env["cuda_runtime"] = cupy.cuda.runtime.runtimeGetVersion()
    except Exception as exc:
        env["cupy"] = f"UNAVAILABLE: {exc}"
    try:
        import kwave

        env["kwave_python"] = getattr(kwave, "__version__", "(no __version__)")
        binaries = Path(kwave.__file__).parent / "bin"
        env["kwave_binaries"] = sorted(p.name for p in binaries.rglob("kspaceFirstOrder*"))[:6]
    except Exception as exc:
        env["kwave_python"] = f"UNAVAILABLE: {exc}"
    try:
        env["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception:
        env["nvidia_smi"] = "(unavailable)"
    return env


# --------------------------------------------------------------------------
# running one stage
# --------------------------------------------------------------------------


def tail(text: str, n: int = 40) -> str:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def run_stage(sid: str, question: str, argv: list[str], timeout: int, log_dir: Path) -> dict:
    """One subprocess, everything captured, nothing allowed to escape.

    Unbuffered and UTF-8 by force: a stage that streams its progress is worth
    watching live in Colab, and a Windows-default codepage turns the first
    non-ASCII character in a report into a crash three hours in.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{sid}.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    t0 = time.perf_counter()
    print(f"\n{'=' * 78}\n[{sid}] {question}\n{' '.join(argv)}\n{'=' * 78}", flush=True)
    try:
        proc = subprocess.run(
            argv,
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        code, out = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        code = 124
        out = (exc.stdout or "") + (exc.stderr or "") + f"\n\nTIMED OUT after {timeout} s"
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
    except Exception as exc:
        code, out = 127, f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - t0
    log_path.write_text(out, encoding="utf-8")
    print(tail(out, 25), flush=True)
    print(f"[{sid}] exit {code} in {elapsed / 60:.1f} min -> {log_path}", flush=True)
    return {
        "id": sid,
        "question": question,
        "argv": argv,
        "exit_code": code,
        "elapsed_s": round(elapsed, 1),
        "log": str(log_path.relative_to(log_path.parents[1])),
        "tail": tail(out, 40),
    }


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def gate_verdicts(root: Path) -> dict[str, str]:
    """Pull every gate id and verdict out of whatever reports were written.

    Deliberately tolerant: the suites do not share one schema, and a summary
    that refused to report anything because one file had a different shape
    would be the least useful possible behaviour.
    """
    found: dict[str, str] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for gate in doc.get("gates") or []:
            gid = gate.get("id")
            if gid:
                found[str(gid)] = str(gate.get("verdict", "?"))
        for chk in doc.get("checks") or []:
            cid = chk.get("id")
            if cid:
                found[f"{path.parent.name}/{cid}"] = str(chk.get("status", "?"))
    return found


def write_summary(out: Path, env: dict, results: list[dict]) -> None:
    lines = [
        "# caustica: full validation on one device",
        "",
        f"`{env.get('gpu', 'no GPU detected')}` · commit `{(env.get('commit') or '')[:12]}`"
        f" · {env.get('generated', '')}",
        "",
    ]
    if env.get("dirty"):
        lines += [
            "> The checkout is **not clean**. Every number below was produced by code that",
            "> differs from the commit named above:",
            "",
            "```",
            str(env["dirty"]),
            "```",
            "",
        ]
    lines += ["| stage | question | exit | minutes |", "|---|---|---|---|"]
    for r in results:
        mark = "ok" if r["exit_code"] == 0 else f"**{r['exit_code']}**"
        lines.append(f"| `{r['id']}` | {r['question']} | {mark} | {r['elapsed_s'] / 60:.1f} |")
    gates = gate_verdicts(out / "reports")
    if gates:
        bad = {k: v for k, v in gates.items() if v.upper() not in ("PASS", "OK")}
        lines += [
            "",
            f"## Gates: {len(gates) - len(bad)}/{len(gates)} pass",
            "",
            "| gate | verdict |",
            "|---|---|",
        ]
        for k in sorted(gates):
            lines.append(f"| `{k}` | {gates[k]} |")
    lines += ["", "## Environment", "", "```json", json.dumps(env, indent=2), "```"]
    for r in results:
        lines += [
            "",
            f"## `{r['id']}` — exit {r['exit_code']}",
            "",
            f"```\n{' '.join(r['argv'])}\n```",
            "",
            "```",
            r["tail"],
            "```",
        ]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps({"environment": env, "stages": results, "gates": gates}, indent=2),
        encoding="utf-8",
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="validation_out", help="where reports and logs land")
    ap.add_argument("--profile", choices=("full", "quick"), default="full")
    ap.add_argument("--only", default="", help="comma-separated stage ids")
    ap.add_argument("--skip", default="")
    ap.add_argument(
        "--phases",
        default="das",
        help="the array's per-element drive: a .npy/.txt of 64 radians, "
        "'das', 'das:x,y,z' in mm, or 'zeros'",
    )
    ap.add_argument("--f0", type=float, default=1.0e6)
    ap.add_argument("--amplitude", type=float, default=1.0e5)
    ap.add_argument("--itrusst-dx", default="0.25", help="mm; the paper's own is 0.5")
    ap.add_argument("--vram-targets", default="4,16,32,64", help="GiB ladder for gpu-gates")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    env = environment()
    print(json.dumps(env, indent=2), flush=True)
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    if str(env.get("cupy", "")).startswith("UNAVAILABLE"):
        print(
            "\n!! CuPy is unavailable, so every native stage will fall back to numpy "
            "and this run will take days rather than hours. Fix the GPU first.\n",
            flush=True,
        )

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    results: list[dict] = []
    for sid, question, cmd, timeout in stages(args, out):
        if (only and sid not in only) or sid in skip:
            continue
        results.append(run_stage(sid, question, cmd, timeout, out / "logs"))
        # Rewritten after every stage: a run this long has to be readable
        # while it is still going.
        write_summary(out, env, results)

    bad = [r["id"] for r in results if r["exit_code"] != 0]
    print(f"\n{'=' * 78}")
    print(
        f"{len(results) - len(bad)}/{len(results)} stages exited 0"
        + (f"; check {bad}" if bad else "")
    )
    print(f"summary: {out / 'SUMMARY.md'}")
    if not args.no_zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"archive: {archive} ({Path(archive).stat().st_size / 1e6:.1f} MB)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
