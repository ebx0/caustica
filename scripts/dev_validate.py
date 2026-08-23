#!/usr/bin/env python3
"""Measure every currently-open unknown in ONE run. Dev tooling, not library surface.

Two profiles, one script:

* ``--profile local`` — the CPU stages (L1..L5). Small grids only, a couple of
  minutes on a laptop, no GPU anywhere. This is the rehearsal: it proves the
  mechanics the Colab stages depend on (the subprocess harness, the divergence
  guard, the planner's pure arithmetic) before a GPU hour is spent on them.
* ``--profile colab`` — the GPU stages (U1..U6). Nothing about the device is
  hardcoded: the card, its VRAM and every shape are read at runtime, so an
  A100, an H100, a T4 or a card the database has never heard of all self-size.

Every stage runs inside its own ``try``/``except``: one failure is a row in the
table, never the end of the session. The console table and the stamped
``dev_validate.json`` in ``--out`` say the same thing.

Colab cell — paste as ONE cell (a GPU runtime already ships cupy, so do NOT
install the ``[gpu]`` extra; it would reinstall CUDA wheels for ten minutes)::

    # 1. the library, from the branch under test
    !pip -q install "git+https://github.com/ebx0/caustica@library-first"

    # 2. this script (clone: also gives you the repo for reference)
    !rm -rf /content/caustica-src
    !git clone -q --depth 1 -b library-first \
        https://github.com/ebx0/caustica /content/caustica-src

    # 3. run it (U1 first; ~30-60 min total, U2 dominates)
    %cd /content
    !python /content/caustica-src/scripts/dev_validate.py --profile colab

    # no-clone variant of step 2 — fetch just this file:
    # !curl -sLo /content/dev_validate.py https://raw.githubusercontent.com/\
    # ebx0/caustica/library-first/scripts/dev_validate.py
    # !python /content/dev_validate.py --profile colab

    # afterwards: keep the evidence
    # from google.colab import files; files.download(
    #     '/content/dev_validate_reports/<the folder printed at the end>/dev_validate.json')

Stage selection works on either profile::

    python scripts/dev_validate.py --profile local  --stages L3
    python scripts/dev_validate.py --profile colab --stages U1,U6 --out /content/probe

Exit codes: 0 every stage PASSed (or was purely informational), 2 the profile
cannot run here (no GPU for ``--profile colab``, or a bad ``--stages``), 4 at
least one stage FAILED or raised.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCRIPT = Path(__file__).resolve()

GIB = 2**30
FORMAT = "caustica-dev-validate/1"

EXIT_OK = 0
EXIT_ENV = 2
EXIT_FAILED = 4

#: The runner's own exit code for "the solver diverged" (caustica.runner.EXIT_SOLVER).
#: Duplicated as a literal on purpose: the parent must be able to judge a child
#: that ran a DIFFERENT caustica than the one this parent imported.
RUNNER_EXIT_SOLVER = 4

PROFILE_STAGES = {
    "local": ("L1", "L2", "L3", "L4", "L5"),
    # U1 first and loudest: it is the open mystery everything else is waiting on.
    "colab": ("U1", "U2", "U3", "U4", "U5", "U6"),
}


def _utf8_streams() -> None:
    """utf-8 + line buffering on stdout/stderr.

    utf-8 because the runner prints em dashes and a cp1252 console would kill a
    solve over punctuation. Line buffering because several stages hand their
    stdout straight to a child (``run_job_subprocess`` does not capture): with
    the default block buffering on a redirected stream, the child's output
    lands in the log BEFORE the parent's header for the stage that spawned it.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


# --------------------------------------------------------------- small helpers


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _newest(root: Path, pattern: str, newer_than: float = 0.0) -> Path | None:
    """Newest file matching ``pattern`` under ``root``, ignoring stale ones.

    ``newer_than`` is what stops a run that died before writing anything from
    being graded on the report of a previous session.
    """
    try:
        hits = [p for p in Path(root).glob(pattern) if p.stat().st_mtime >= newer_than]
    except OSError:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def _dev_pct(predicted: float | None, actual: float | None) -> float | None:
    """Signed deviation of ``actual`` from ``predicted``, in percent."""
    if predicted in (None, 0) or actual is None:
        return None
    return (float(actual) - float(predicted)) / float(predicted) * 100.0


def _num(value, fmt: str = ".3g") -> str:
    return "--" if value is None else format(float(value), fmt)


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


# ------------------------------------------------------------------- job dicts


def rung_job_dict(side: int, *, dx_mm: float = 0.30, apex_frac: float | None = None) -> dict:
    """A gpu-gates LADDER rung as a job dict, optionally with the apex moved.

    The shape comes from the library's own ``rung_job`` so the probe measures
    the thing the gate suite measures, not a lookalike. ``apex_frac`` overrides
    the default 0.06 x extent standoff — at 0.30 mm and 256 voxels that puts
    the bowl 4.6 mm from the face, inside a 7.7 mm PML band, which is one of
    the two hypotheses U1 exists to separate.
    """
    from caustica.validation.gpu_gates import RungSpec, rung_job

    spec = RungSpec(
        name=f"rung{side}",
        shape=(side,) * 3,
        dx_mm=dx_mm,
        solver="westervelt",
        target_gib=0.0,
        predicted_gib=0.0,
        preview_only=True,
        expect_fit=True,
    )
    job = rung_job(spec)
    if apex_frac is not None:
        extent = spec.extent_mm
        job["source"]["apex_mm"] = [
            round(0.5 * extent, 4),
            round(0.5 * extent, 4),
            round(apex_frac * extent, 4),
        ]
    return job


#: 24^3 at an amplitude float32 cannot carry. Same scene as
#: tests/test_divergence_guard.py, so a green stage here means the guard
#: behaves off the test bench exactly as it does on it.
DIVERGE_JOB = {
    "format": "caustica-job/1",
    "kind": "explicit",
    "name": "diverge-24",
    "medium": {"kind": "homogeneous"},
    "grid": {"ndim": 3, "dx_mm": 0.5, "size_mm": [12.0, 12.0, 12.0], "pml": {"thickness_mm": 1.0}},
    "source": {
        "kind": "array",
        "array": {"kind": "bowl", "d_outer_mm": 4.0, "roc_mm": 5.0},
        "apex_mm": [6.0, 6.0, 3.0],
    },
    "drive": {"f0_mhz": 1.0, "amplitude_kpa": 1e35},
    "run": {"spec": {"min_settle_periods": 1, "max_settle_periods": 2, "n_record_periods": 1}},
    "solver": "linear",
}

#: The 18x18x24 mm bowl from tests/test_runner.py's ``mini_job``.
MINI_JOB = {
    "format": "caustica-job/1",
    "kind": "explicit",
    "name": "mini",
    "medium": {"kind": "homogeneous"},
    "grid": {"ndim": 3, "dx_mm": 0.75, "size_mm": [18, 18, 24], "pml": {"thickness_mm": 3.0}},
    "source": {
        "kind": "array",
        "array": {"kind": "bowl", "d_outer_mm": 10.0, "roc_mm": 12.0},
        "apex_mm": [9, 9, 6.0],
    },
    "drive": {"f0_mhz": 1.0, "amplitude_kpa": 100.0},
    "run": {"spec": {"min_settle_periods": 2, "max_settle_periods": 6}, "harmonics": [1]},
    "solver": "linear",
}


# ------------------------------------------------------- the child (one solve)
#
# Every solve this script grades happens in a FRESH process. cupy's memory pool
# is process-wide and monotone, calibration mutates process state, and the whole
# point of U1 is to tell those apart from the shape -- none of which survives an
# in-process loop. The child re-invokes THIS file with --child.


def _solve(
    job: dict,
    out: Path,
    *,
    backend: str,
    tag: str,
    side,
    apex,
    preview_only: bool = True,
    allow_slow_cpu: bool = False,
) -> int:
    """Run one job and frame its output with markers the parent can parse."""
    from caustica.runner import RunnerOptions, run_job_file

    out.mkdir(parents=True, exist_ok=True)
    job_path = out / "job_input.json"
    job_path.write_text(json.dumps(job, indent=1), encoding="utf-8")
    print(f"### CASE {tag} side={side} apex={apex} backend={backend}", flush=True)
    code = run_job_file(
        job_path,
        RunnerOptions(
            out=out,
            backend=backend,
            measure=False,
            preview_only=preview_only,
            status_interval_s=0.0,
            progress="plain",
            allow_slow_cpu=allow_slow_cpu,
        ),
    )
    print(f"### EXIT {tag} {code}", flush=True)
    return int(code)


def child_main(mode: str, out: Path, backend: str) -> int:
    _utf8_streams()
    if mode == "u1-bare256":  # 256^3 alone, in a virgin process
        return _solve(
            rung_job_dict(256), out / "bare256", backend="cupy", tag="bare256", side=256, apex=0.06
        )
    if mode == "u1-cal256":  # planner.calibrate() first, SAME process
        from caustica import planner

        planner.calibrate(backend="cupy")
        return _solve(
            rung_job_dict(256), out / "cal256", backend="cupy", tag="cal256", side=256, apex=0.06
        )
    if mode == "u1-warm256":  # a clean 400^3 solve first, then 256^3
        _solve(
            rung_job_dict(400), out / "warm400", backend="cupy", tag="warm400", side=400, apex=0.06
        )
        return _solve(
            rung_job_dict(256), out / "warm256", backend="cupy", tag="warm256", side=256, apex=0.06
        )
    if mode == "u1-apex256":  # 256^3 with the source pushed clear of the PML
        return _solve(
            rung_job_dict(256, apex_frac=0.15),
            out / "apex256",
            backend="cupy",
            tag="apex256",
            side=256,
            apex=0.15,
        )
    if mode == "u1-bare400":  # control: a bigger shape, same everything else
        return _solve(
            rung_job_dict(400), out / "bare400", backend="cupy", tag="bare400", side=400, apex=0.06
        )
    if mode == "diverge":
        # preview_only=False on purpose: "must not leave result.h5" is vacuous
        # in a mode that never writes one.
        return _solve(
            DIVERGE_JOB,
            out,
            backend=backend,
            tag="diverge",
            side=24,
            apex="n/a",
            preview_only=False,
        )
    if mode == "sentinel":
        return _solve(
            rung_job_dict(96, apex_frac=0.25),
            out,
            backend="numpy",
            tag="sentinel",
            side=96,
            apex=0.25,
            allow_slow_cpu=True,
        )
    if mode.startswith("warmup-"):
        side = int(mode.split("-", 1)[1])
        return _solve(
            rung_job_dict(side), out, backend="cupy", tag=f"warmup{side}", side=side, apex=0.06
        )
    print(f"unknown child mode {mode!r}", file=sys.stderr)
    return EXIT_ENV


# ----------------------------------------------------- parsing a child's output

_CASE_RE = re.compile(r"^### CASE (\S+) side=(\S+) apex=(\S+) backend=(\S+)", re.M)
_EXIT_RE = re.compile(r"^### EXIT (\S+) (-?\d+)", re.M)
#: The progress writer's first period boundary: "period 1/N  step k/m  peak 1.234 MPa".
_PEAK1_RE = re.compile(r"period 1/\d+\s+step \d+/\d+\s+peak (\S+ \S+)")
#: The runner's completion line.
_DONE_PEAK_RE = re.compile(r"peak \|P\| = (\S+) MPa")
#: PRE-F1 textual divergence. Anchored to the words that carry a number --
#: never a bare "nan", which a probe whose own output folder was called
#: "nanprobe" once matched in every single case (2026-08-22).
_TEXT_NAN_RE = re.compile(r"peak\s+nan\b|peak \|P\| = nan\b|d=nan\b", re.I)


def _split_cases(text: str) -> list[dict]:
    """One record per ``### CASE`` marker, carrying that case's slice of output."""
    marks = list(_CASE_RE.finditer(text))
    exits = {m.group(1): int(m.group(2)) for m in _EXIT_RE.finditer(text)}
    rows = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end() : end]
        peak1 = _PEAK1_RE.search(body)
        done = _DONE_PEAK_RE.search(body)
        rows.append(
            {
                "case": m.group(1),
                "side": m.group(2),
                "apex": m.group(3),
                "backend": m.group(4),
                "exit": exits.get(m.group(1)),
                "period1_peak": peak1.group(1) if peak1 else None,
                "final_peak_mpa": done.group(1) if done else None,
                "text_nan": bool(_TEXT_NAN_RE.search(body)),
                "body_tail": _flat(body[-400:]),
            }
        )
    return rows


def _diverged(case_dir: Path, exit_code: int | None, text_nan: bool) -> tuple[bool | None, str]:
    """Did this case diverge? ``True`` / ``False`` / ``None`` = could not tell.

    Reads BOTH signals. Post-F1 the library raises ``SolverDivergedError`` at
    the first non-finite period boundary -- BEFORE the progress callback for
    that period fires -- so a diverged run prints no ``peak nan`` line at all
    and shows up purely as exit 4 plus an ``error.json``. On a pre-F1 commit it
    is the other way round. Reading both keeps the probe valid on either.

    The third state is the one that matters for U1's conclusion: a case that
    died of something ELSE (a VRAM refusal, a config error, a crash) has NOT
    shown that the shape is clean, and folding it into ``False`` would let the
    probe announce "not reproduced" on the strength of five runs that never
    happened.
    """
    err = _read_json(case_dir / "error.json")
    if exit_code == RUNNER_EXIT_SOLVER and err.get("error_class") == "SolverDivergedError":
        return True, "exit4+error.json"
    if text_nan:
        return True, "text(pre-F1)"
    if exit_code == 0:
        return False, "clean"
    return None, f"exit {exit_code}: {err.get('error_class') or 'no error.json'}"


def _run_child(mode: str, out: Path, *, backend: str = "numpy") -> subprocess.CompletedProcess:
    out.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--child",
            mode,
            "--child-out",
            str(out),
            "--child-backend",
            backend,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


# ----------------------------------------------------------------- stage plumbing


@dataclass
class Ctx:
    profile: str
    root: Path
    log: object = print

    def dir(self, stage: str) -> Path:
        d = self.root / stage
        d.mkdir(parents=True, exist_ok=True)
        return d


@dataclass
class Outcome:
    """One stage's verdict. ``numbers`` is the table's third column."""

    verdict: str = "INFO"  # PASS | FAIL | WARN | INFO | ERROR
    numbers: str = ""
    note: str = ""
    data: dict = field(default_factory=dict)


def _verdict_from(checks: list[tuple[str, bool, str]]) -> tuple[str, str]:
    """Fold a list of ``(name, ok, detail)`` into a verdict + a failure note."""
    bad = [name for name, ok, _ in checks if not ok]
    return ("PASS" if not bad else "FAIL"), ("" if not bad else "failed: " + ", ".join(bad))


# ------------------------------------------------------------------ LOCAL stages


def stage_l1(ctx: Ctx) -> Outcome:
    """L1 — the analytic suite at its 'quick' size, as a subprocess."""
    out = ctx.dir("L1")
    started = time.time() - 2.0
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "caustica.validation",
            "run-analytic",
            "--size",
            "quick",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    report = _newest(out, "*/analytic.json", started)
    payload = _read_json(report) if report else {}
    gates = payload.get("gates", [])
    passed = [g for g in gates if g.get("verdict") == "PASS"]
    failed = [g["id"] for g in gates if g.get("verdict") == "FAIL"]
    incomplete = [g["id"] for g in gates if g.get("verdict") == "INCOMPLETE"]
    overall = payload.get("verdict", "(no report)")
    numbers = f"exit {proc.returncode}; gates {len(passed)}/{len(gates)} PASS; overall {overall}"
    if not report:
        return Outcome(
            "FAIL",
            numbers,
            f"no analytic.json under {out}; stderr tail: {_flat(proc.stderr[-300:])}",
        )
    ok = proc.returncode == 0 and overall == "PASS"
    note = f"report {report}"
    if failed:
        note = "FAILED gates: " + ", ".join(failed)
    elif incomplete:
        note = "INCOMPLETE gates: " + ", ".join(incomplete)
    return Outcome(
        "PASS" if ok else "FAIL",
        numbers,
        note,
        {
            "exit_code": proc.returncode,
            "report": str(report),
            "overall": overall,
            "elapsed_s": payload.get("elapsed_s"),
            "gates": [
                {
                    "id": g.get("id"),
                    "verdict": g.get("verdict"),
                    "n_pass": g.get("n_pass"),
                    "required": g.get("required"),
                }
                for g in gates
            ],
        },
    )


def _divergence_stage(ctx: Ctx, stage: str, backend: str) -> Outcome:
    """L2/U6 — a job that cannot help but overflow MUST fail loudly."""
    out = ctx.dir(stage)
    proc = _run_child("diverge", out, backend=backend)
    cases = _split_cases(proc.stdout + "\n" + proc.stderr)
    marker_exit = cases[0]["exit"] if cases else None
    exit_code = marker_exit if marker_exit is not None else proc.returncode
    err = _read_json(out / "error.json")
    result_exists = (out / "result.h5").exists()
    checks = [
        ("exit==4", exit_code == RUNNER_EXIT_SOLVER, f"got {exit_code}"),
        ("error.json written", bool(err), str(sorted(err)[:4])),
        (
            "error_class",
            err.get("error_class") == "SolverDivergedError",
            str(err.get("error_class")),
        ),
        ("no result.h5", not result_exists, f"exists={result_exists}"),
    ]
    verdict, note = _verdict_from(checks)
    if not cases and proc.returncode not in (0, RUNNER_EXIT_SOLVER):
        note = (note + "; " if note else "") + f"child crashed: {_flat(proc.stderr[-300:])}"
    numbers = (
        f"backend {backend}; exit {exit_code}; "
        f"error_class {err.get('error_class')}; result.h5 {result_exists}"
    )
    return Outcome(
        verdict,
        numbers,
        note,
        {
            "backend": backend,
            "exit_code": exit_code,
            "child_returncode": proc.returncode,
            "error": {k: err.get(k) for k in ("stage", "error_class", "exit_code", "message")},
            "result_h5_exists": result_exists,
            "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
        },
    )


def stage_l2(ctx: Ctx) -> Outcome:
    return _divergence_stage(ctx, "L2", "numpy")


def stage_l3(ctx: Ctx) -> Outcome:
    """L3 — the planner's pure arithmetic. No solves, no hardware, no disk."""
    from caustica import planner
    from caustica.planner import calibration as cal
    from caustica.planner import model

    checks: list[tuple[str, bool, str]] = []
    data: dict = {}

    # --- 1. probe shapes for three real cards -------------------------------
    probes = {}
    for name, free_gib in (("T4", 14.6), ("A100-40", 39.6), ("A100-80", 79.2)):
        shapes = cal.probe_shapes_for_budget(int(free_gib * GIB))
        sides = [s[0] for s in shapes]
        biggest = sides[-1] ** 3 if sides else 0
        probes[name] = {"free_gib": free_gib, "sides": sides, "largest_elems": biggest}
        checks += [
            (f"{name}: >=2 probe shapes", len(sides) >= 2, str(sides)),
            (
                f"{name}: ascending",
                sides == sorted(sides) and len(set(sides)) == len(sides),
                str(sides),
            ),
            (
                f"{name}: largest >= MIN_GPU_PROBE_ELEMS",
                biggest >= cal.MIN_GPU_PROBE_ELEMS,
                f"{biggest:,} vs {cal.MIN_GPU_PROBE_ELEMS:,}",
            ),
        ]
    data["probe_shapes"] = probes

    # --- 2. the fit residual flags the A100 launch-latency samples ----------
    # 48^3 and 72^3, both ~1.0 ms/step because neither saturates an A100: the
    # samples that produced a=0 and a 6.8x over-prediction (2026-08-22).
    bad = [(110592, 1.0e-3), (373248, 1.0e-3)]
    a, b = cal.fit_time_model(bad)
    residual = cal.fit_max_rel_residual(bad, a, b)
    good = [(2_097_152, 6.0e-3), (16_777_216, 55.0e-3)]
    ga, gb = cal.fit_time_model(good)
    good_residual = cal.fit_max_rel_residual(good, ga, gb)
    checks += [
        (
            "bad-fit samples flagged",
            residual > cal.FIT_RESIDUAL_LIMIT,
            f"residual {residual:.3f} > {cal.FIT_RESIDUAL_LIMIT}",
        ),
        (
            "healthy samples not flagged",
            good_residual <= cal.FIT_RESIDUAL_LIMIT,
            f"residual {good_residual:.3g}",
        ),
    ]
    data["fit"] = {
        "bad_samples": bad,
        "bad_a": a,
        "bad_b": b,
        "bad_residual": residual,
        "good_residual": good_residual,
        "limit": cal.FIT_RESIDUAL_LIMIT,
    }

    # --- 3. spec_for_device: the known row and the unknown card -------------
    devices, _aliases = planner.load_gpu_db()
    known = planner.spec_for_device("NVIDIA A100-SXM4-40GB", int(39.6 * GIB))
    unknown_name = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    unknown = planner.spec_for_device(unknown_name, int(95.0 * GIB))
    checks += [
        ("known card -> gpu_db row", known.key in devices, known.key),
        (
            "known card carries device_name",
            known.device_name == "NVIDIA A100-SXM4-40GB",
            str(known.device_name),
        ),
        ("known card source=db", known.source == "db", known.source),
        ("unknown card -> unknown: key", unknown.key.startswith("unknown:"), unknown.key),
        (
            "unknown card VRAM from the device",
            abs(unknown.vram_gib - 95.0) < 0.5,
            f"{unknown.vram_gib} GiB",
        ),
        ("unknown card source=device", unknown.source == "device", unknown.source),
        ("unknown card is not silently an A100", unknown.key not in devices, unknown.key),
    ]
    data["spec_for_device"] = {
        "known": {"key": known.key, "vram_gib": known.vram_gib, "source": known.source},
        "unknown": {"key": unknown.key, "vram_gib": unknown.vram_gib, "source": unknown.source},
    }

    # --- 4. calibrated_warmup: two-term, flat, and neither ------------------
    p_elems = 16_777_216
    two_term = {"warmup_context_s": 2.0, "warmup_per_elem_s": 1.0e-6}
    flat = {"warmup_s": 7.5}
    expect_two = 2.0 + 1.0e-6 * p_elems
    checks += [
        (
            "calibrated_warmup two-term",
            math.isclose(cal.calibrated_warmup(two_term, p_elems), expect_two, rel_tol=1e-9),
            f"{cal.calibrated_warmup(two_term, p_elems):.3f} s",
        ),
        (
            "calibrated_warmup two-term scales with P",
            cal.calibrated_warmup(two_term, 2 * p_elems) > cal.calibrated_warmup(two_term, p_elems),
            "monotone in P",
        ),
        (
            "calibrated_warmup flat fallback",
            cal.calibrated_warmup(flat, p_elems) == 7.5,
            f"{cal.calibrated_warmup(flat, p_elems)} s",
        ),
        (
            "calibrated_warmup default when empty",
            cal.calibrated_warmup({}, p_elems, default=model.GPU_WARMUP_S) == model.GPU_WARMUP_S,
            f"{model.GPU_WARMUP_S} s",
        ),
    ]
    data["calibrated_warmup"] = {
        "two_term_s": cal.calibrated_warmup(two_term, p_elems),
        "flat_s": cal.calibrated_warmup(flat, p_elems),
        "default_s": cal.calibrated_warmup({}, p_elems, default=model.GPU_WARMUP_S),
    }

    verdict, note = _verdict_from(checks)
    data["checks"] = [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks]
    n_ok = sum(1 for _, ok, _ in checks if ok)
    numbers = (
        f"{n_ok}/{len(checks)} checks; bad-fit residual {residual:.2f}; "
        f"unknown card -> source={unknown.source}, {unknown.vram_gib:.0f} GiB"
    )
    return Outcome(verdict, numbers, note or "planner arithmetic only — nothing was solved", data)


def stage_l4(ctx: Ctx) -> Outcome:
    """L4 — the subprocess harness the GPU stages are built on, on numpy."""
    from caustica.runner import RunnerOptions
    from caustica.validation.gpu_gates import run_job_subprocess

    out = ctx.dir("L4")
    job_path = out / "mini.json"
    job_path.write_text(json.dumps(MINI_JOB, indent=1), encoding="utf-8")
    t0 = time.perf_counter()
    code = run_job_subprocess(
        job_path,
        RunnerOptions(
            out=out,
            backend="numpy",
            measure=False,
            preview_only=True,
            status_interval_s=0.0,
            progress=None,  # -> --no-progress: this stage is about plumbing
        ),
    )
    elapsed = time.perf_counter() - t0
    meta = _read_json(out / "run_meta.json")
    preview = out / "preview.npz"
    result = out / "result.h5"
    checks = [
        ("exit==0", code == 0, f"got {code}"),
        ("preview.npz written", preview.exists(), str(preview)),
        ("no result.h5 (preview-only)", not result.exists(), f"exists={result.exists()}"),
        ("run_meta stamped", bool(meta.get("actual")), str(meta.get("backend"))),
        (
            "meta says preview_only",
            meta.get("actual", {}).get("preview_only") is True,
            str(meta.get("actual", {}).get("preview_only")),
        ),
    ]
    verdict, note = _verdict_from(checks)
    numbers = (
        f"exit {code}; {elapsed:.1f} s wall; steps {meta.get('actual', {}).get('steps_total')}"
    )
    return Outcome(
        verdict,
        numbers,
        note or "run_job_subprocess is the U-stage harness",
        {
            "exit_code": code,
            "elapsed_s": round(elapsed, 2),
            "backend": meta.get("backend"),
            "actual": meta.get("actual"),
            "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
        },
    )


def stage_l5(ctx: Ctx) -> Outcome:
    """L5 — the light stand-in for U1's control leg: 96^3 rung geometry on numpy.

    Same job family as the 256^3 case, at a size a laptop can finish, with the
    apex pushed well clear of the PML. It must come back with a FINITE peak: if
    the rung scene itself were unsound, this is where it would show without
    booking a GPU. 256^3 is never run here.
    """
    out = ctx.dir("L5")
    t0 = time.perf_counter()
    proc = _run_child("sentinel", out, backend="numpy")
    elapsed = time.perf_counter() - t0
    cases = _split_cases(proc.stdout + "\n" + proc.stderr)
    row = cases[0] if cases else {}
    exit_code = row.get("exit", proc.returncode)
    peak_txt = row.get("final_peak_mpa")
    peak = None
    try:
        peak = float(peak_txt) if peak_txt is not None else None
    except ValueError:
        peak = None
    nan, signal = _diverged(out, exit_code, bool(row.get("text_nan")))
    checks = [
        ("exit==0", exit_code == 0, f"got {exit_code}"),
        ("a peak was reported", peak is not None, str(peak_txt)),
        (
            "peak is finite and > 0",
            peak is not None and math.isfinite(peak) and peak > 0.0,
            str(peak),
        ),
        ("no divergence signal", nan is False, signal or "none"),
    ]
    verdict, note = _verdict_from(checks)
    numbers = (
        f"exit {exit_code}; peak {_num(peak, '.3f')} MPa; period-1 "
        f"{row.get('period1_peak') or '--'}; {elapsed:.0f} s"
    )
    if not cases:
        note = f"child produced no CASE marker: {_flat(proc.stderr[-300:])}"
    return Outcome(
        verdict,
        numbers,
        note or "96^3 westervelt, apex at 0.25 x extent, CPU",
        {
            "exit_code": exit_code,
            "elapsed_s": round(elapsed, 1),
            "final_peak_mpa": peak,
            "period1_peak": row.get("period1_peak"),
            "diverged": nan,
            "divergence_signal": signal,
            "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
        },
    )


# ------------------------------------------------------------------ COLAB stages

#: How the tri-state divergence verdict prints. ``?`` is "the case did not run
#: to a verdict" -- never quietly a clean run.
_NAN_CELL = {True: "NaN", False: "clean", None: "?"}

U1_CASES = (
    ("bare256", "u1-bare256", 256, "0.06", "256^3 alone in a virgin process"),
    ("cal256", "u1-cal256", 256, "0.06", "planner.calibrate() first, same process"),
    ("warm256", "u1-warm256", 256, "0.06", "a 400^3 solve first, same process"),
    ("apex256", "u1-apex256", 256, "0.15", "256^3 with the apex clear of the PML"),
    ("bare400", "u1-bare400", 400, "0.06", "control: a larger shape, alone"),
)


def _u1_interpret(nan: dict[str, bool | None]) -> dict:
    """Which hypothesis survives the five cases. Pure — reads only the verdicts."""
    b256, b400 = nan.get("bare256"), nan.get("bare400")
    cal, warm, apex = nan.get("cal256"), nan.get("warm256"), nan.get("apex256")

    def tri(known: bool, supported: bool) -> str:
        return ("supported" if supported else "refuted") if known else "undetermined"

    hyps = {
        "not_reproduced": {
            "claim": "the 256^3 rung does not diverge on this device/commit at all",
            "verdict": tri(b256 is not None, b256 is False),
        },
        "not_shape_specific": {
            "claim": "the rung scene diverges at 400^3 too — this is not about 256^3",
            "verdict": tri(b400 is not None, bool(b400)),
        },
        "geometry_x_backend": {
            "claim": "the default apex (0.06 x extent) sits in/against the PML; "
            "moving it to 0.15 x extent fixes it",
            "verdict": tri(b256 is not None and apex is not None, bool(b256) and apex is False),
        },
        "process_state": {
            "claim": "whether 256^3 diverges depends on what ran before it IN THE SAME "
            "process (calibration / a previous solve)",
            "verdict": tri(
                b256 is not None and (cal is not None or warm is not None),
                b256 is not None
                and ((cal is not None and cal != b256) or (warm is not None and warm != b256)),
            ),
        },
        "shape_itself": {
            "claim": "the 256^3 shape diverges however it is reached, and 400^3 does not",
            "verdict": tri(
                None not in (b256, cal, warm, apex, b400),
                bool(b256) and bool(cal) and bool(warm) and bool(apex) and b400 is False,
            ),
        },
    }
    order = (
        "not_reproduced",
        "not_shape_specific",
        "geometry_x_backend",
        "process_state",
        "shape_itself",
    )
    surviving = next((k for k in order if hyps[k]["verdict"] == "supported"), "inconclusive")
    return {"surviving": surviving, "hypotheses": hyps}


def stage_u1(ctx: Ctx) -> Outcome:
    """U1 — five fresh processes that tell the 256^3 NaN's three explanations apart."""
    out = ctx.dir("U1")
    rows: list[dict] = []
    nan: dict[str, bool | None] = {}
    for tag, mode, side, apex, why in U1_CASES:
        print(f"  U1 case {tag}: {why} ...", flush=True)
        t0 = time.perf_counter()
        proc = _run_child(mode, out)
        elapsed = time.perf_counter() - t0
        parsed = {c["case"]: c for c in _split_cases(proc.stdout + "\n" + proc.stderr)}
        case = parsed.get(tag, {})
        case_dir = out / tag
        exit_code = case.get("exit", proc.returncode)
        diverged, signal = _diverged(case_dir, exit_code, bool(case.get("text_nan")))
        err = _read_json(case_dir / "error.json")
        nan[tag] = diverged
        rows.append(
            {
                "case": tag,
                "side": side,
                "apex_frac": apex,
                "why": why,
                "exit": exit_code,
                "child_returncode": proc.returncode,
                "period1_peak": case.get("period1_peak"),
                "final_peak_mpa": case.get("final_peak_mpa"),
                "diverged": diverged,
                "signal": signal,
                "error_class": err.get("error_class"),
                "error_message": _flat(err.get("message", ""))[:300],
                "elapsed_s": round(elapsed, 1),
                "saw_marker": tag in parsed,
                "stderr_tail": "" if tag in parsed else _flat(proc.stderr[-300:]),
            }
        )

    interp = _u1_interpret(nan)
    header = (
        f"{'case':<9} {'side':>5} {'apex':>6} {'exit':>5} {'period-1 peak':>16} "
        f"{'final MPa':>10} {'NaN':>6} {'signal':<24} {'s':>6}"
    )
    lines = [
        "",
        "=" * 96,
        "U1 - 256^3 NaN discrimination probe (one FRESH process per case)",
        "=" * 96,
        header,
        "-" * len(header),
    ]
    for r in rows:
        lines.append(
            f"{r['case']:<9} {r['side']:>5} {r['apex_frac']:>6} {str(r['exit']):>5} "
            f"{(r['period1_peak'] or '(no solve)'):>16} {(r['final_peak_mpa'] or '--'):>10} "
            f"{_NAN_CELL[r['diverged']]:>6} {(r['signal'] or '-'):<24} {r['elapsed_s']:>6.0f}"
        )
    lines += ["-" * len(header), f"surviving hypothesis: {interp['surviving']}"]
    for key, h in interp["hypotheses"].items():
        lines.append(f"  {h['verdict']:<13} {key}: {h['claim']}")
    lines.append("=" * 96)
    for line in lines:
        print(line, flush=True)

    unseen = [r["case"] for r in rows if not r["saw_marker"]]
    unknown = [r["case"] for r in rows if r["diverged"] is None]
    verdict = "WARN" if (unseen or unknown) else "INFO"
    numbers = ", ".join(f"{r['case']}={_NAN_CELL[r['diverged']]}" for r in rows)
    note = f"surviving: {interp['surviving']}"
    if unknown:
        note += f"; UNDETERMINED (did not diverge and did not complete): {', '.join(unknown)}"
    if unseen:
        note += f"; no output from {', '.join(unseen)}"
    return Outcome(
        verdict,
        numbers,
        note,
        {
            "cases": rows,
            "interpretation": interp,
            "table": lines,
        },
    )


def stage_u2(ctx: Ctx) -> Outcome:
    """U2 — rerun the gpu-gates suite and read its report back."""
    root = Path("benchmarks/reports/gpu_gates")
    started = time.time() - 2.0
    print("  (gpu-gates output streams below; this stage takes tens of minutes)", flush=True)
    proc = subprocess.run([sys.executable, "-m", "caustica.validation", "gpu-gates"], check=False)
    report = _newest(root, "*/gpu_gates.json", started)
    if report is None:
        return Outcome(
            "FAIL",
            f"exit {proc.returncode}; no report",
            f"nothing newer than this stage under {root.resolve()}",
            {"exit_code": proc.returncode},
        )
    payload = _read_json(report)

    rungs = []
    for r in payload.get("rungs", []):
        plan, actual = r.get("plan") or {}, r.get("actual") or {}
        rungs.append(
            {
                "name": r.get("name"),
                "shape": r.get("shape"),
                "exit_code": r.get("exit_code"),
                "expect_fit": r.get("expect_fit"),
                "vram_plan_gib": plan.get("vram_gib"),
                "vram_actual_gib": actual.get("vram_pool_peak_gib"),
                "vram_dev_pct": _dev_pct(plan.get("vram_gib"), actual.get("vram_pool_peak_gib")),
                "t_plan_s": plan.get("t_expected_s"),
                "t_actual_s": actual.get("elapsed_solve_s"),
                "t_dev_pct": _dev_pct(plan.get("t_expected_s"), actual.get("elapsed_solve_s")),
                "plan_source": plan.get("source"),
                "warmup_s": actual.get("warmup_s"),
                "error_class": (r.get("error") or {}).get("error_class"),
            }
        )
    gates = {g.get("id"): g.get("verdict") for g in payload.get("gates", [])}
    time_checks = [
        {"name": c.get("name"), "verdict": c.get("verdict"), "detail": c.get("detail")}
        for g in payload.get("gates", [])
        if g.get("id") == "M8.time"
        for c in g.get("checks", [])
    ]
    diverged = [
        r["name"] for r in rungs if r["error_class"] == "SolverDivergedError" or r["exit_code"] == 4
    ]
    worst_vram = max(
        (abs(r["vram_dev_pct"]) for r in rungs if r["vram_dev_pct"] is not None), default=None
    )

    overall = payload.get("verdict")
    verdict = (
        "PASS"
        if proc.returncode == 0 and overall == "PASS"
        else ("WARN" if overall == "INCOMPLETE" else "FAIL")
    )
    note_bits = [f"gates: {', '.join(f'{k}={v}' for k, v in gates.items())}"]
    if diverged:
        # Reported, never hidden: until U1's root cause is fixed this is the
        # expected shape of the 256^3 rung, and a stage that swallowed it
        # would make the next session rediscover it.
        note_bits.append(f"DIVERGED rungs (exit 4): {', '.join(diverged)}")
    return Outcome(
        verdict,
        (
            f"exit {proc.returncode}; overall {overall}; "
            f"{len(rungs)} rungs; worst |VRAM dev| {_num(worst_vram, '.1f')}%"
        ),
        "; ".join(note_bits),
        {
            "exit_code": proc.returncode,
            "report": str(report),
            "overall": overall,
            "device": payload.get("device"),
            "gpu_key": payload.get("gpu_key"),
            "free_vram_gib_at_start": payload.get("free_vram_gib_at_start"),
            "rungs": rungs,
            "gates": gates,
            "m8_time_checks": time_checks,
            "diverged_rungs": diverged,
            "notes": payload.get("notes", []),
        },
    )


def _device_calibration() -> tuple[str, dict | None, Path]:
    from caustica.planner import calibration as cal

    path = cal.default_calibration_path()
    device = cal.device_name("cupy")
    entry = cal.load_calibration(path).get("devices", {}).get(device)
    if entry is None:
        entry = cal.find_calibration_for("unknown", path, device_name=device)
    return device, entry, path


def _pick_side(free_bytes: int | None, candidates: tuple[int, ...], fraction: float) -> int:
    """Largest candidate side whose predicted VRAM fits ``fraction`` of free."""
    from caustica.planner import model

    if not free_bytes:
        return candidates[-1]
    budget = free_bytes * fraction
    for side in candidates:
        mem = model.kspace_memory((side,) * 3, False, 1, rec_elems=side**3)
        if mem.total_bytes <= budget:
            return side
    return candidates[-1]


def stage_u3(ctx: Ctx) -> Outcome:
    """U3 — is the calibration this device just wrote any good?"""
    import caustica as hs
    from caustica import planner
    from caustica.materials import water
    from caustica.medium import Medium
    from caustica.planner import calibration as cal
    from caustica.planner import model
    from caustica.sources import plane_cw_source

    device, entry, path = _device_calibration()
    if entry is None:
        return Outcome(
            "FAIL",
            f"device {device}",
            f"no calibration entry in {path}",
            {"device": device, "path": str(path)},
        )

    residual = entry.get("fit_max_rel_residual")
    checks = [
        ("probe shapes recorded", bool(entry.get("shapes")), str(entry.get("shapes"))),
        (">=2 probe shapes", len(entry.get("shapes") or []) >= 2, str(entry.get("shapes"))),
        ("fit_mode recorded", bool(entry.get("fit_mode")), str(entry.get("fit_mode"))),
        (
            "fit_max_rel_residual < 0.10",
            residual is not None and float(residual) < cal.FIT_RESIDUAL_LIMIT,
            _num(residual),
        ),
        (
            "warmup_context_s present",
            entry.get("warmup_context_s") is not None,
            _num(entry.get("warmup_context_s")),
        ),
        (
            "warmup_per_elem_s present",
            entry.get("warmup_per_elem_s") is not None,
            _num(entry.get("warmup_per_elem_s"), ".3e"),
        ),
    ]

    # --- spot-check: planner.estimate vs ~20 real steps at a mid size -------
    free = cal.free_device_bytes("cupy")
    side = _pick_side(free, (320, 288, 256, 240, 216, 192, 160), 0.45)
    shape = (side,) * 3
    spot: dict = {"shape": list(shape), "free_bytes": free}
    try:
        import cupy as cp  # noqa: PLC0415 (GPU-only stage)

        total = int(cp.cuda.Device().mem_info[1])
        gpu_spec = planner.spec_for_device(device, total)
        grid = hs.Grid(shape=shape, dx=0.30e-3, pml=hs.PMLSpec(thickness=3.0e-3))
        # water() defaults to beta=0 -> a LINEAR medium, and solver="linear"
        # keeps nonlinear=False: exactly the regime calibrate() probed, so the
        # comparison is about the timing model rather than about buffer counts.
        med = Medium.homogeneous(shape, water())
        src = plane_cw_source(grid, f0=0.5e6, amplitude=1e5)
        est = planner.estimate(grid, med, src, solver="linear", gpu=gpu_spec)
        del med, src
        run = cal.measure_step_time(shape, nonlinear=False, backend="cupy", n_steps=20)
        dev = _dev_pct(est.t_step_s, run["t_step_s"])
        spot.update(
            {
                "estimate_source": est.source,
                "t_step_predicted_s": est.t_step_s,
                "t_step_measured_s": run["t_step_s"],
                "dev_pct": dev,
                "p_elems": model.fft_sizes(shape)[1],
            }
        )
        checks += [
            ("spot-check plan is 'calibrated'", est.source == "calibrated", est.source),
            (
                "|t_step deviation| <= 25%",
                dev is not None and abs(dev) <= 25.0,
                f"{_num(dev, '.1f')}%",
            ),
        ]
    except Exception as exc:  # noqa: BLE001 - the entry audit above still stands
        spot["error"] = f"{type(exc).__name__}: {exc}"
        checks.append(("spot-check ran", False, spot["error"]))

    verdict, note = _verdict_from(checks)
    numbers = (
        f"{device}; fit {entry.get('fit_mode')} residual {_num(residual, '.3f')}; "
        f"{side}^3 t_step dev {_num(spot.get('dev_pct'), '.1f')}%"
    )
    return Outcome(
        verdict,
        numbers,
        note or f"calibration read from {path}",
        {
            "device": device,
            "path": str(path),
            "entry": {
                k: entry.get(k)
                for k in (
                    "a",
                    "b",
                    "fit_mode",
                    "fit_max_rel_residual",
                    "p_max_probe",
                    "shapes",
                    "warmup_s",
                    "warmup_source",
                    "warmup_context_s",
                    "warmup_per_elem_s",
                    "backend",
                    "nonlinear",
                    "calibrated_at",
                )
            },
            "spot_check": spot,
            "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
        },
    )


def stage_u4(ctx: Ctx) -> Outcome:
    """U4 — which branch of spec_for_device this card takes, and can it find itself."""
    import cupy as cp

    from caustica import planner
    from caustica.planner import calibration as cal

    device = cal.device_name("cupy")
    total = int(cp.cuda.Device().mem_info[1])
    spec = planner.spec_for_device(device, total)
    devices, _aliases = planner.load_gpu_db()
    unknown_branch = spec.key.startswith("unknown:")
    branch = "unknown" if unknown_branch else "known (gpu_db row)"

    by_name = cal.find_calibration_for(spec.key, device_name=spec.device_name)
    by_key = cal.find_calibration_for(spec.key)  # no device_name: the old path

    checks = [("device_name attached", spec.device_name == device, str(spec.device_name))]
    if unknown_branch:
        checks += [
            ("key starts with 'unknown:'", True, spec.key),
            ("key is not a gpu_db row", spec.key not in devices, spec.key),
            (
                "vram matches the card",
                abs(spec.vram_gib - total / GIB) < 0.5,
                f"{spec.vram_gib:.2f} vs {total / GIB:.2f}",
            ),
            ("source == 'device'", spec.source == "device", spec.source),
        ]
    else:
        checks += [
            ("key is a gpu_db row", spec.key in devices, spec.key),
            ("source == 'db'", spec.source == "db", spec.source),
        ]
    checks.append(
        (
            "find_calibration_for(device_name=...) reaches the entry",
            by_name is not None,
            "found" if by_name else "MISSING — U2's calibration is unreachable",
        )
    )

    verdict, note = _verdict_from(checks)
    numbers = (
        f"{device}; branch {branch}; key {spec.key}; vram {spec.vram_gib:.1f}/{total / GIB:.1f} GiB"
    )
    extra = (
        "token lookup alone also finds it"
        if by_key is not None
        else "token lookup alone does NOT find it — device_name is load-bearing here"
    )
    return Outcome(
        verdict,
        numbers,
        note or extra,
        {
            "device": device,
            "branch": branch,
            "total_bytes": total,
            "spec": {
                "key": spec.key,
                "vram_gib": spec.vram_gib,
                "mem_bw_gbs": spec.mem_bw_gbs,
                "fp32_tflops": spec.fp32_tflops,
                "source": spec.source,
                "device_name": spec.device_name,
                "notes": spec.notes,
            },
            "calibration_found_by_device_name": by_name is not None,
            "calibration_found_by_key_alone": by_key is not None,
            "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
        },
    )


def stage_u5(ctx: Ctx) -> Outcome:
    """U5 — does warmup = context + per_elem * P predict a real run's warmup?"""
    from caustica.planner import calibration as cal
    from caustica.planner import model

    out = ctx.dir("U5")
    device, entry, path = _device_calibration()
    if entry is None:
        return Outcome(
            "FAIL", f"device {device}", f"no calibration entry in {path}", {"device": device}
        )
    ctx_s, per_s = entry.get("warmup_context_s"), entry.get("warmup_per_elem_s")
    if ctx_s is None or per_s is None:
        return Outcome(
            "FAIL",
            f"context={_num(ctx_s)} per_elem={_num(per_s)}",
            "the entry carries no two-term warmup model (pre-2026-08-23 entry)",
            {"device": device, "entry_keys": sorted(entry)},
        )

    free = cal.free_device_bytes("cupy")
    side = _pick_side(free, (192, 160, 128, 96), 0.25)
    p_elems = int(model.fft_sizes((side,) * 3)[1])
    predicted = cal.calibrated_warmup(entry, p_elems)

    t0 = time.perf_counter()
    proc = _run_child(f"warmup-{side}", out)
    elapsed = time.perf_counter() - t0
    meta = _read_json(out / "run_meta.json")
    actual = (meta.get("actual") or {}).get("warmup_s")
    cases = _split_cases(proc.stdout + "\n" + proc.stderr)
    exit_code = cases[0]["exit"] if cases else proc.returncode
    ratio = (float(actual) / predicted) if (actual is not None and predicted > 0) else None

    checks = [
        ("run completed", exit_code == 0, f"exit {exit_code}"),
        ("run_meta carries a warmup", actual is not None, _num(actual)),
        ("prediction within +/-40%", ratio is not None and 0.6 <= ratio <= 1.4, _num(ratio, ".2f")),
    ]
    verdict, note = _verdict_from(checks)
    if verdict == "FAIL" and exit_code == 0 and actual is not None:
        # A model check, not a gate: a miss is information, not a broken build.
        verdict, note = "WARN", note + " (model check only — not a release gate)"
    numbers = (
        f"{side}^3, P={p_elems:,}; predicted {_num(predicted, '.2f')} s, "
        f"actual {_num(actual, '.2f')} s, ratio {_num(ratio, '.2f')}"
    )
    return Outcome(
        verdict,
        numbers,
        note or f"context {_num(ctx_s, '.2f')} s + {_num(per_s, '.2e')} s/elem",
        {
            "device": device,
            "side": side,
            "p_elems": p_elems,
            "warmup_context_s": ctx_s,
            "warmup_per_elem_s": per_s,
            "predicted_s": predicted,
            "actual_s": actual,
            "ratio": ratio,
            "exit_code": exit_code,
            "elapsed_s": round(elapsed, 1),
            "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
        },
    )


def stage_u6(ctx: Ctx) -> Outcome:
    return _divergence_stage(ctx, "U6", "cupy")


STAGES = {
    "L1": ("analytic suite, 'quick' size (CPU)", stage_l1),
    "L2": ("divergence guard on numpy", stage_l2),
    "L3": ("planner arithmetic: probes, fit, spec_for_device, warmup", stage_l3),
    "L4": ("subprocess runner mechanics (run_job_subprocess, numpy)", stage_l4),
    "L5": ("96^3 NaN sentinel on numpy (light stand-in for U1)", stage_l5),
    "U1": ("256^3 NaN discrimination probe (5 fresh processes)", stage_u1),
    "U2": ("gpu-gates rerun + report parse", stage_u2),
    "U3": ("calibration quality + step-time spot check", stage_u3),
    "U4": ("unknown-GPU path through spec_for_device", stage_u4),
    "U5": ("two-term warmup model vs a real run", stage_u5),
    "U6": ("divergence guard on cupy", stage_u6),
}


# ------------------------------------------------------------------- the runner


def _print_table(rows: list[dict]) -> None:
    head = ("stage", "verdict", "s", "key numbers", "note")
    body = [
        (r["stage"], r["verdict"], f"{r['elapsed_s']:.0f}", _flat(r["numbers"]), _flat(r["note"]))
        for r in rows
    ]
    widths = [
        max(len(head[i]), *(len(b[i]) for b in body)) if body else len(head[i]) for i in range(4)
    ]
    line = "  ".join("-" * w for w in widths) + "  " + "-" * 4
    print("")
    print("=" * 100)
    print("dev_validate summary")
    print("=" * 100)
    print("  ".join(h.ljust(w) for h, w in zip(head[:4], widths, strict=True)) + "  " + head[4])
    print(line)
    for b in body:
        print("  ".join(b[i].ljust(widths[i]) for i in range(4)) + "  " + b[4])
    print("=" * 100)


def _stamp(profile: str, rows: list[dict], root: Path, elapsed: float) -> dict:
    try:
        from caustica import __version__ as version
        from caustica.env import env_report, git_commit

        env, commit = env_report(), git_commit()
    except Exception as exc:  # pragma: no cover - a broken install still reports
        version, env, commit = None, {"stamp_error": f"{type(exc).__name__}: {exc}"}, "unknown"
    return {
        "format": FORMAT,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": profile,
        "caustica": version,
        "git_commit": commit,
        "python": sys.version.split()[0],
        "argv": sys.argv[1:],
        "cwd": str(Path.cwd()),
        "report_dir": str(root),
        "environment": env,
        "elapsed_s": round(elapsed, 1),
        "stages": rows,
    }


def _require_profile_env(profile: str) -> None:
    """``--profile colab`` on a CPU box exits 2 with the fix, never a traceback."""
    if profile != "colab":
        return
    try:
        from caustica.env import env_report, require_gpu
    except Exception as exc:
        print(
            f"dev_validate: cannot import caustica ({type(exc).__name__}: {exc}). "
            f"Install it first: pip install -e . (or the git+https URL in this "
            f"script's docstring).",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_ENV) from None
    try:
        require_gpu("the colab profile measures a real GPU (U1-U6)")
    except Exception as exc:
        resolved = env_report().get("resolved_backend")
        print("", file=sys.stderr)
        print(
            f"dev_validate: --profile colab cannot run here (resolved backend: {resolved}).",
            file=sys.stderr,
        )
        print(f"  {exc}", file=sys.stderr)
        print("  Run `--profile local` for the CPU stages (L1-L5).", file=sys.stderr)
        raise SystemExit(EXIT_ENV) from None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dev_validate.py",
        description="measure every open unknown in one run (dev tooling, not library surface)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="stages — " + "; ".join(f"{k}: {v[0]}" for k, v in STAGES.items()),
    )
    p.add_argument(
        "--profile",
        choices=tuple(PROFILE_STAGES),
        default=None,
        help="local: the CPU stages L1-L5. colab: the GPU stages U1-U6.",
    )
    p.add_argument(
        "--stages",
        default=None,
        help="comma-separated subset of the profile's stages (default: all of them)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="report folder (default: dev_validate_reports/<profile>-<timestamp>)",
    )
    # The child protocol: this script re-invokes itself so every solve gets a
    # fresh process (cupy's memory pool and calibration are process state).
    p.add_argument("--child", default=None, help=argparse.SUPPRESS)
    p.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    p.add_argument("--child-backend", default="numpy", help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # utf-8 for every child too: the runner prints em dashes, and a cp1252
    # pipe would kill a solve over a punctuation mark.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8:replace")
    _utf8_streams()

    if args.child:
        if not args.child_out:
            print("--child requires --child-out", file=sys.stderr)
            return EXIT_ENV
        return child_main(args.child, Path(args.child_out), args.child_backend)

    if not args.profile:
        print("dev_validate: --profile local|colab is required", file=sys.stderr)
        return EXIT_ENV

    available = PROFILE_STAGES[args.profile]
    if args.stages:
        wanted = [s.strip().upper() for s in args.stages.replace(",", " ").split()]
        bad = [s for s in wanted if s not in available]
        if bad:
            print(
                f"dev_validate: {', '.join(bad)} is not in the {args.profile} profile "
                f"({', '.join(available)})",
                file=sys.stderr,
            )
            return EXIT_ENV
        selected = [s for s in available if s in wanted]
    else:
        selected = list(available)

    _require_profile_env(args.profile)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(args.out) if args.out else Path("dev_validate_reports") / f"{args.profile}-{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(profile=args.profile, root=root)

    print(f"dev_validate — profile {args.profile}, stages {', '.join(selected)}")
    print(f"report folder: {root.resolve()}")

    rows: list[dict] = []
    t_all = time.perf_counter()
    for stage in selected:
        title, fn = STAGES[stage]
        print(f"\n--- {stage}: {title} " + "-" * max(0, 60 - len(title)))
        t0 = time.perf_counter()
        try:
            outcome = fn(ctx)
        except Exception as exc:  # noqa: BLE001 - one bad stage must not lose the rest
            outcome = Outcome(
                "ERROR",
                f"{type(exc).__name__}",
                _flat(str(exc))[:200],
                {"traceback": traceback.format_exc()},
            )
            print(f"  {stage} raised {type(exc).__name__}: {exc}")
        elapsed = time.perf_counter() - t0
        row = {
            "stage": stage,
            "title": title,
            "verdict": outcome.verdict,
            "numbers": outcome.numbers,
            "note": outcome.note,
            "elapsed_s": round(elapsed, 1),
            "data": outcome.data,
        }
        rows.append(row)
        print(f"  -> {outcome.verdict}  {outcome.numbers}  ({elapsed:.1f} s)")
        if outcome.note:
            print(f"     {outcome.note}")

    elapsed_all = time.perf_counter() - t_all
    _print_table(rows)

    payload = _stamp(args.profile, rows, root, elapsed_all)
    report = root / "dev_validate.json"
    report.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\njson: {report.resolve()}")
    print(f"total: {elapsed_all:.1f} s")

    return EXIT_FAILED if any(r["verdict"] in ("FAIL", "ERROR") for r in rows) else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
