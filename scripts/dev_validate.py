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

Stage selection works on either profile (case-insensitive)::

    python scripts/dev_validate.py --profile local  --stages L3
    python scripts/dev_validate.py --profile colab --stages U1,U6 --out /content/probe
    python scripts/dev_validate.py --profile colab --stages U1b   # the shape hunt

U1b's step-by-step machinery has a GPU-less rehearsal — two numpy mirrors of
the same job, which must agree bit for bit at every step and every
intermediate. It needs no profile and no GPU::

    python scripts/dev_validate.py --u1b-rehearse

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
    # U1b is its follow-up (which shapes/settings, and which operator).
    "colab": ("U1", "U1b", "U2", "U3", "U4", "U5", "U6"),
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


def _jsonsafe(obj):
    """Recursively replace non-finite floats with ``'inf'`` / ``'nan'`` strings.

    Python's ``json`` happily writes bare ``Infinity`` and ``NaN`` tokens, which
    no other JSON reader accepts — and this probe's whole subject is fields that
    stop being finite, so the report would be unreadable by anything but Python
    exactly when it matters most.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else repr(obj)
    if isinstance(obj, dict):
        return {k: _jsonsafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonsafe(v) for v in obj]
    return obj


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


#: Settling cap for every U1b matrix case. It is NOT what makes the matrix
#: short: the engine's effective settling floor is
#: ``tof_periods + max(min_settle_periods, ceil(ramp_periods) + 1)``, and with
#: the rung scene's focus-referenced time of flight (~8 periods at 256^3, ~12
#: at 400^3) plus the 3-period ramp that floor already exceeds 3. The override
#: is here so the CAP can never be the reason a case runs long -- every case
#: lands on ~13 (256-scene) / ~17 (400-scene) periods x spp=20 steps.
U1B_MAX_SETTLE = 3

#: The U1b shape/setting matrix. ``letter`` is the case's label in the task
#: that commissioned it; ``over`` is the complete set of deviations from
#: ``rung_job(RungSpec(shape=(side,)*3))``. Two controls (``-``) bracket it:
#: without them a matrix in which nothing diverged could not tell "the shape is
#: innocent" from "this stage's settings changed the run".
U1B_CASES = (
    (
        "ref256",
        "-",
        {"side": 256},
        "control: the real 256^3 rung, at this stage's settling cap",
    ),
    ("a240", "a", {"side": 240}, "240^3 — smooth (2^4*3*5), just under 256"),
    ("b250", "b", {"side": 250}, "250^3 = 2*5^4 — smooth, no padding, between 240 and 256"),
    (
        "c256lin",
        "c",
        {"side": 256, "solver": "linear"},
        "256^3 without the Westervelt term — does the nonlinear term matter?",
    ),
    (
        "d256amp2",
        "d",
        {"side": 256, "amplitude_kpa": 2.0},
        "256^3 driven at 2 kPa (1/100 of the rung) — is it amplitude-dependent?",
    ),
    (
        "e256x400",
        "e",
        {"side": 256, "size_mm": (76.8, 76.8, 120.0)},
        "256x256x400: 256 on the two transverse (c2c) axes only",
    ),
    (
        "f400x256",
        "f",
        {"side": 400, "size_mm": (120.0, 120.0, 76.8)},
        "400x400x256: 256 on the beam axis only (the r2c/last FFT axis)",
    ),
    ("g288", "g", {"side": 288}, "288^3 — control just above 256"),
    ("ref400", "-", {"side": 400}, "control: the healthy 400^3 rung"),
)
_U1B_BY_TAG = {c[0]: c for c in U1B_CASES}

#: The rehearsal scene: the same rung geometry at a size a laptop can step,
#: with the apex at 0.25 x extent so the bowl clears the PML at 64^3 (exactly
#: what L5's sentinel does at 96^3).
U1B_REHEARSE_SIDE = 64
U1B_REHEARSE_APEX = 0.25


def u1b_matrix_job(tag: str) -> tuple[dict, dict]:
    """One U1b matrix case as ``(job dict, meta)``.

    ``meta['overrides']`` is the audit trail: every deviation from the plain
    rung job, in words, so the report says what was run without anyone having
    to re-derive it from a shape.

    The two mixed shapes deserve their own sentence, because the ONLY thing
    they change is the FFT size. ``rung_job`` derives the whole scene (bowl
    diameter, radius of curvature, apex standoff) from ``shape[-1] * dx``, so
    building them by side alone would move the transducer as well as the
    transform. Instead the scene is built at its own side and only
    ``grid.size_mm`` is overridden per axis:

    * ``e256x400`` — the 256-rung SCENE (76.8 mm bowl geometry) on a
      76.8 x 76.8 x 120.0 mm grid => shape 256 x 256 x 400.
    * ``f400x256`` — the 400-rung SCENE (120 mm bowl geometry) on a
      120.0 x 120.0 x 76.8 mm grid => shape 400 x 400 x 256.
    """
    if tag not in _U1B_BY_TAG:
        raise KeyError(f"unknown U1b case {tag!r}; known: {', '.join(_U1B_BY_TAG)}")
    _tag, letter, over, why = _U1B_BY_TAG[tag]
    side = int(over["side"])
    job = rung_job_dict(side)
    job["name"] = f"u1b-{tag}"
    dx_mm = float(job["grid"]["dx_mm"])
    notes = [f"rung_job(shape=({side},)*3, dx={dx_mm} mm) — scene extent {side * dx_mm:g} mm"]
    job["run"]["spec"]["max_settle_periods"] = U1B_MAX_SETTLE
    notes.append(f"run.spec.max_settle_periods -> {U1B_MAX_SETTLE}")
    if "solver" in over:
        job["solver"] = over["solver"]
        notes.append(f"solver -> {over['solver']!r} (was 'westervelt')")
    if "amplitude_kpa" in over:
        job["drive"]["amplitude_kpa"] = float(over["amplitude_kpa"])
        notes.append(f"drive.amplitude_kpa -> {over['amplitude_kpa']} (was 200.0)")
    if "size_mm" in over:
        job["grid"]["size_mm"] = [float(s) for s in over["size_mm"]]
        notes.append(
            f"grid.size_mm -> {job['grid']['size_mm']} — the SCENE stays the "
            f"{side}-rung scene, only the FFT size changes"
        )
    shape = tuple(int(round(s / dx_mm)) for s in job["grid"]["size_mm"])
    meta = {
        "tag": tag,
        "letter": letter,
        "why": why,
        "scene_side": side,
        "scene_extent_mm": round(side * dx_mm, 4),
        "shape": list(shape),
        "shape_str": "x".join(str(n) for n in shape),
        "solver": job["solver"],
        "amplitude_kpa": job["drive"]["amplitude_kpa"],
        "size_mm": list(job["grid"]["size_mm"]),
        "apex_mm": list(job["source"]["apex_mm"]),
        "overrides": notes,
    }
    try:  # the padded shape is evidence in its own right (2,3,5-smooth => equal)
        from caustica.solvers.kspace.operators import pad_shape

        padded = tuple(pad_shape(shape))
        meta["padded"] = list(padded)
        meta["padding_free"] = padded == shape
    except Exception as exc:  # noqa: BLE001 - a missing import must not lose the case
        meta["padded"] = None
        meta["padding_free"] = None
        meta["padded_error"] = f"{type(exc).__name__}: {exc}"
    return job, meta


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
    if mode.startswith("u1b-") and mode[len("u1b-") :] in _U1B_BY_TAG:
        tag = mode[len("u1b-") :]  # one U1b matrix case, preview-only
        job, meta = u1b_matrix_job(tag)
        return _solve(
            job,
            out / tag,
            backend=backend,
            tag=tag,
            side=meta["shape_str"],
            apex="0.06",
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
    #: U1b-stepdiff knobs (hidden CLI flags). The defaults are the task's:
    #: 40 steps on the real 256^3 rung and on the 400^3 control. The CPU leg
    #: dominates the cost (~8 FFTs per step over the whole volume), so a
    #: memory- or time-constrained session can shrink either.
    u1b_steps: int = 40
    u1b_shapes: tuple[int, ...] = (256, 400)

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


# ------------------------------------------------- U1b: which shape, which op
#
# U1 left exactly one hypothesis standing: "the 256^3 shape diverges however it
# is reached". U1b attacks that from two sides in one stage.
#
#   U1b-matrix    — nine fresh subprocesses that ask WHICH shapes and settings
#                   carry the failure (240/250/288 around it, linear vs
#                   westervelt, 200 kPa vs 2 kPa, and the two mixed shapes that
#                   split "256 on the transverse axes" from "256 on the beam
#                   axis"). Same divergence detection as U1, same tri-state.
#   U1b-stepdiff  — ONE process, both backends, the engine's own step
#                   composition rebuilt around the REAL maps of the real rung
#                   job. It answers the two questions the matrix cannot: which
#                   STEP first disagrees, and inside that step, which OPERATOR.


def _u1b_matrix_interpret(nan: dict[str, bool | None]) -> dict:
    """What the nine verdicts allow one to say. Pure — reads only the verdicts."""

    def tri(known: bool, supported: bool) -> str:
        return ("supported" if supported else "refuted") if known else "undetermined"

    ref256, ref400 = nan.get("ref256"), nan.get("ref400")
    a240, b250, g288 = nan.get("a240"), nan.get("b250"), nan.get("g288")
    c_lin, d_amp = nan.get("c256lin"), nan.get("d256amp2")
    e_tr, f_beam = nan.get("e256x400"), nan.get("f400x256")

    hyps = {
        "reproduced_here": {
            "claim": "the 256^3 rung still diverges under this stage's settings "
            "(max_settle_periods=3) — the matrix's own positive control",
            "verdict": tri(ref256 is not None, bool(ref256)),
        },
        "not_shape_specific": {
            "claim": "the 400^3 control diverges too, so this matrix is measuring "
            "its own settings rather than the shape",
            "verdict": tri(ref400 is not None, bool(ref400)),
        },
        "isolated_at_256": {
            "claim": "240^3, 250^3 and 288^3 are all clean while 256^3 diverges: the "
            "failure is pinned to the SIZE 256, not to a size range",
            "verdict": tri(
                None not in (ref256, a240, b250, g288),
                bool(ref256) and a240 is False and b250 is False and g288 is False,
            ),
        },
        "needs_the_nonlinear_term": {
            "claim": "256^3 on solver 'linear' is clean: the Westervelt term is part "
            "of the mechanism",
            "verdict": tri(None not in (ref256, c_lin), bool(ref256) and c_lin is False),
        },
        "amplitude_dependent": {
            "claim": "256^3 at a 2 kPa drive is clean: the blow-up scales with the "
            "drive, so it grows out of the field rather than out of an index",
            "verdict": tri(None not in (ref256, d_amp), bool(ref256) and d_amp is False),
        },
        "beam_axis_carries_it": {
            "claim": "only the LAST axis (the real-to-complex FFT axis) being 256 "
            "matters: 400x400x256 diverges, 256x256x400 does not",
            "verdict": tri(None not in (e_tr, f_beam), bool(f_beam) and e_tr is False),
        },
        "transverse_axes_carry_it": {
            "claim": "only the two complex-to-complex axes being 256 matter: "
            "256x256x400 diverges, 400x400x256 does not",
            "verdict": tri(None not in (e_tr, f_beam), bool(e_tr) and f_beam is False),
        },
        "any_256_axis": {
            "claim": "a 256 on ANY axis is enough — both mixed shapes diverge",
            "verdict": tri(None not in (e_tr, f_beam), bool(e_tr) and bool(f_beam)),
        },
        "cubic_256_only": {
            "claim": "neither mixed shape diverges: it takes 256 on ALL THREE axes "
            "at once (a shape-triple effect, not a per-axis transform effect)",
            "verdict": tri(
                None not in (ref256, e_tr, f_beam),
                bool(ref256) and e_tr is False and f_beam is False,
            ),
        },
    }
    axis = "undetermined (a mixed-shape case did not reach a verdict)"
    if None not in (e_tr, f_beam):
        axis = {
            (True, True): "any axis of size 256",
            (True, False): "the transverse (complex-to-complex) axes",
            (False, True): "the beam axis (the real-to-complex / last FFT axis)",
            (False, False): "no single axis — only the fully cubic 256^3",
        }[(bool(e_tr), bool(f_beam))]
    return {
        "supported": [k for k, h in hyps.items() if h["verdict"] == "supported"],
        "axis_verdict": axis,
        "hypotheses": hyps,
    }


def _u1b_matrix(out: Path) -> dict:
    """Run the nine matrix cases, one FRESH subprocess each, and grade them."""
    rows: list[dict] = []
    nan: dict[str, bool | None] = {}
    for tag, letter, _over, why in U1B_CASES:
        _job, meta = u1b_matrix_job(tag)
        print(f"  U1b-matrix {letter} {tag} [{meta['shape_str']}]: {why} ...", flush=True)
        t0 = time.perf_counter()
        proc = _run_child(f"u1b-{tag}", out, backend="cupy")
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
                "letter": letter,
                "why": why,
                "meta": meta,
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

    interp = _u1b_matrix_interpret(nan)
    header = (
        f"{'case':<9} {'shape':>13} {'solver':>10} {'kPa':>6} {'exit':>5} "
        f"{'period-1 peak':>16} {'final MPa':>10} {'NaN':>6} {'signal':<20} {'s':>5}"
    )
    lines = [
        "",
        "=" * len(header),
        "U1b-matrix — which shapes/settings diverge (one FRESH process per case)",
        "=" * len(header),
        header,
        "-" * len(header),
    ]
    for r in rows:
        m = r["meta"]
        lines.append(
            f"{r['case']:<9} {m['shape_str']:>13} {m['solver']:>10} "
            f"{m['amplitude_kpa']:>6g} {str(r['exit']):>5} "
            f"{(r['period1_peak'] or '(no solve)'):>16} {(r['final_peak_mpa'] or '--'):>10} "
            f"{_NAN_CELL[r['diverged']]:>6} {(r['signal'] or '-'):<20} {r['elapsed_s']:>5.0f}"
        )
    lines += [
        "-" * len(header),
        f"axis verdict: {interp['axis_verdict']}",
        "interpretation:",
    ]
    for key, h in interp["hypotheses"].items():
        lines.append(f"  {h['verdict']:<13} {key}: {h['claim']}")
    lines.append("=" * len(header))
    return {"cases": rows, "verdicts": nan, "interpretation": interp, "table": lines}


# ------------------------------------------------------------ U1b-stepdiff
#
# The engine's step is mirrored here, not called: run_cw_kspace_pstd owns its
# whole loop and reports only per-period peaks, which is three orders of
# magnitude too coarse to name an operator. planner.calibration.measure_step_time
# already mirrors the same composition with SYNTHETIC maps for timing; this is
# that mirror with the job's REAL maps, which is the point -- a probe hunting a
# backend-and-shape-specific fault may not approximate any input.


class _StepMirror:
    """One backend's private rebuild of the engine's per-step composition.

    Mirrors :func:`caustica.solvers.kspace.engine.run_cw_kspace_pstd` line for
    line: the padded shape, the four/five property maps, ``kappa``, the
    per-axis derivative factors, the sponge, and the source scatter — then
    ``step()`` applies them in the engine's exact order (u first from a single
    ``rfftn(p)``, absorb then sponge, the pressure update, absorb then sponge
    again, and the ramped source LAST).

    Every array is built through THIS instance's ``xp``. That is not a style
    point: a numpy array handed to a cupy op raises, and the one thing this
    probe must never do is compare a program against a slightly different
    program.
    """

    def __init__(self, backend: str, built: object, harmonics: tuple[int, ...] = (1,)) -> None:
        import numpy as np

        from caustica.core.backend import get_backend
        from caustica.solvers.kspace import operators as ops
        from caustica.solvers.kspace.engine import cw_discretization

        self.backend_name = backend
        b = self.b = get_backend(backend)
        xp = self.xp = b.xp
        self.fft = b.fft

        grid, medium, source, spec = built.grid, built.medium, built.source, built.spec
        if medium is None:
            raise ValueError("_StepMirror needs a built job WITH its medium (with_medium=True)")
        self.nd = nd = grid.ndim
        self.dx = dx = grid.dx
        self.period = 1.0 / source.f0
        self.omega = 2.0 * np.pi * source.f0
        self.ramp_periods = source.ramp_periods
        # engine: nonlinear=True is passed by the westervelt solver, False by
        # linear; beta2_dt then survives only if the medium is not linear.
        self.nonlinear = built.solver == "westervelt"

        self.spp, self.dt = cw_discretization(grid, medium, spec, source.f0, harmonics)
        dt = self.dt
        self.padded = padded = ops.pad_shape(grid.shape)
        self.active_shape = tuple(grid.shape)

        self.dt_over_rho = xp.asarray(dt / ops.pad_volume(medium.rho, padded), dtype=xp.float32)
        self.rhoc2_dt = xp.asarray(
            ops.pad_volume(medium.rho * medium.c.astype(np.float64) ** 2, padded) * dt,
            dtype=xp.float32,
        )
        self.absorb = xp.asarray(
            np.exp(-ops.pad_volume(medium.alpha * medium.c, padded) * dt), dtype=xp.float32
        )
        self.beta2_dt = None
        if self.nonlinear and not medium.is_linear:
            self.beta2_dt = xp.asarray(
                ops.pad_volume(medium.beta, padded) * (2.0 * dt), dtype=xp.float32
            )

        ks = ops.k_vectors(padded, dx, xp)
        kappa = ops.kappa_sinc(ks, c_ref=medium.c_max, dt=dt, xp=xp)
        self.deriv = ops.spectral_derivative_factors(ks, kappa, xp)
        # The engine deletes ks and kappa here; keep only the scalars the
        # sanity dump asks for, so this mirror does not carry a shadow copy.
        self.max_kappa = float(abs(kappa).max())
        self.min_kappa = float(abs(kappa).min())
        self.max_k = [float(abs(k).max()) for k in ks]
        del ks, kappa

        pml_edge = grid.pml.edge if grid.pml is not None else 2.0
        self.pml_vox = grid.pml_vox
        self.sponge = ops.sponge_volume(padded, grid.pml_vox, pml_edge, xp)

        self.p = xp.zeros(padded, dtype=xp.float32)
        self.u = [xp.zeros(padded, dtype=xp.float32) for _ in range(nd)]

        self.src_idx = tuple(xp.asarray(source.indices[:, d]) for d in range(nd))
        self.src_ph = xp.asarray(source.phases, dtype=xp.float32)
        self.amp = float(source.amplitude)
        c_src = medium.c[tuple(source.indices[:, d] for d in range(nd))]
        self.src_scale = xp.asarray(2.0 * c_src.astype(np.float64) * dt / dx, dtype=xp.float32)
        self.n_src = int(source.indices.shape[0])

    # ---- state handling ---------------------------------------------------

    def state(self) -> list:
        """Host float32 COPIES of ``(p, u0..u_{nd-1})``."""

        def host(arr):
            h = self.b.to_numpy(arr)
            # On numpy ``to_numpy`` hands back the LIVE array and the next step
            # would rewrite the "snapshot" in place -- the copy is load-bearing.
            # On cupy ``asnumpy`` already produced a fresh host array, so a
            # second copy would be a pure waste of hundreds of MB.
            return h if self.b.is_gpu else h.copy()

        return [host(self.p)] + [host(x) for x in self.u]

    def load_state(self, arrays: list) -> None:
        """Adopt a host state. ``xp.array`` copies on BOTH backends."""
        xp = self.xp
        self.p = xp.array(arrays[0], dtype=xp.float32)
        self.u = [xp.array(a, dtype=xp.float32) for a in arrays[1:]]

    def sync(self) -> None:
        self.b.synchronize()

    def maps(self) -> dict:
        """Every input the step reads, by name (``None`` where inactive)."""
        out = {
            name: getattr(self, name)
            for name in ("dt_over_rho", "rhoc2_dt", "absorb", "sponge", "beta2_dt", "src_scale")
        }
        out.update({f"deriv{i}": self.deriv[i] for i in range(self.nd)})
        return out

    # ---- the step ---------------------------------------------------------

    def _scatter(self, n: int) -> None:
        """The engine's source injection: ramp x mass-source scale x sin(wt-phi)."""
        from caustica.sources import ramp_envelope

        xp = self.xp
        t = n * self.dt
        env = ramp_envelope(t, self.period, self.ramp_periods)
        self.p[self.src_idx] += (xp.float32(self.amp * env) * self.src_scale) * xp.sin(
            xp.float32(self.omega * t) - self.src_ph
        )

    def step(self, n: int) -> None:
        fft = self.fft
        p, u, padded, deriv = self.p, self.u, self.padded, self.deriv
        pk = fft.rfftn(p)
        for i in range(self.nd):
            grad_i = fft.irfftn(deriv[i] * pk, s=padded)
            u[i] -= self.dt_over_rho * grad_i
            u[i] *= self.absorb
            u[i] *= self.sponge
        acc = None
        for i in range(self.nd):
            term = deriv[i] * fft.rfftn(u[i])
            acc = term if acc is None else acc + term
        divu = fft.irfftn(acc, s=padded)
        if self.beta2_dt is None:
            p -= self.rhoc2_dt * divu
        else:
            p -= (self.rhoc2_dt + self.beta2_dt * p) * divu
        p *= self.absorb
        p *= self.sponge
        self._scatter(n)

    def step_traced(self, n: int):
        """The SAME step, yielding ``(name, array)`` at every intermediate.

        A generator so the two backends can be walked in lock-step and each
        intermediate compared and released before the next one is built —
        holding all of them at 400^3 would be gigabytes per side.
        """
        fft = self.fft
        p, u, padded, deriv = self.p, self.u, self.padded, self.deriv
        pk = fft.rfftn(p)
        yield "pk = rfftn(p)", pk
        for i in range(self.nd):
            dpk = deriv[i] * pk
            yield f"deriv{i} * pk", dpk
            grad_i = fft.irfftn(dpk, s=padded)
            del dpk
            yield f"grad{i} = irfftn(deriv{i}*pk)", grad_i
            u[i] -= self.dt_over_rho * grad_i
            u[i] *= self.absorb
            u[i] *= self.sponge
            yield f"u{i} after update", u[i]
        acc = None
        for i in range(self.nd):
            uk = fft.rfftn(u[i])
            yield f"rfftn(u{i})", uk
            term = deriv[i] * uk
            del uk
            yield f"deriv{i} * rfftn(u{i})", term
            acc = term if acc is None else acc + term
        divu = fft.irfftn(acc, s=padded)
        del acc
        yield "divu = irfftn(sum)", divu
        if self.beta2_dt is None:
            p -= self.rhoc2_dt * divu
        else:
            p -= (self.rhoc2_dt + self.beta2_dt * p) * divu
        yield "p after pressure update", p
        p *= self.absorb
        p *= self.sponge
        yield "p after absorb+sponge", p
        self._scatter(n)
        yield "p after source scatter", p


def _host64(mirror: _StepMirror, arr):
    """One device->host copy, promoted to float64/complex128 for comparison."""
    import numpy as np

    h = mirror.b.to_numpy(arr)
    return h.astype(np.complex128 if np.iscomplexobj(h) else np.float64)


def _rel_linf(a, c) -> dict:
    """Relative L-infinity difference of two host arrays, non-finite aware."""
    import numpy as np

    with np.errstate(invalid="ignore"):
        a_max = float(np.abs(a).max()) if a.size else 0.0
        c_max = float(np.abs(c).max()) if c.size else 0.0
        a_fin = bool(np.isfinite(a).all())
        c_fin = bool(np.isfinite(c).all())
        if not (a_fin and c_fin):
            rel = float("inf")
        else:
            diff = float(np.abs(a - c).max())
            den = max(a_max, c_max)
            rel = (diff / den) if den > 0.0 else 0.0
    return {
        "rel": rel,
        "a_max": a_max,
        "b_max": c_max,
        "a_finite": a_fin,
        "b_finite": c_fin,
    }


def _u1b_input_report(ma: _StepMirror, mb: _StepMirror) -> dict:
    """Cheap sanity: do the two sides' INPUTS agree before a single step runs?

    Both sides' stats AND their cross difference come from ONE host copy per
    map per side -- dtype and shape are read off the device array directly, so
    a 256^3 run pays nine transfers, not eighteen.
    """
    a_maps, b_maps = ma.maps(), mb.maps()
    per_side: dict[str, dict] = {"a": {}, "b": {}}
    cross: dict[str, dict | None] = {}
    for name, arr_a in a_maps.items():
        arr_b = b_maps.get(name)
        if arr_a is None or arr_b is None:
            per_side["a"][name] = per_side["b"][name] = cross[name] = None
            continue
        d = _rel_linf(_host64(ma, arr_a), _host64(mb, arr_b))
        per_side["a"][name] = {
            "dtype": str(arr_a.dtype),
            "shape": list(arr_a.shape),
            "max_abs": d["a_max"],
            "all_finite": d["a_finite"],
        }
        per_side["b"][name] = {
            "dtype": str(arr_b.dtype),
            "shape": list(arr_b.shape),
            "max_abs": d["b_max"],
            "all_finite": d["b_finite"],
        }
        cross[name] = d

    sides = {}
    for tag, m in (("a", ma), ("b", mb)):
        sides[tag] = {
            "backend": m.backend_name,
            "padded": list(m.padded),
            "active": list(m.active_shape),
            "dt": m.dt,
            "spp": m.spp,
            "pml_vox": m.pml_vox,
            "n_source_voxels": m.n_src,
            "nonlinear": m.nonlinear,
            "beta2_dt_active": m.beta2_dt is not None,
            "max_kappa": m.max_kappa,
            "min_kappa": m.min_kappa,
            "max_k_per_axis": m.max_k,
            "p_dtype": str(m.p.dtype),
            "u_dtype": str(m.u[0].dtype),
            "src_idx_dtype": str(m.src_idx[0].dtype),
            "src_ph_dtype": str(m.src_ph.dtype),
            "maps": per_side[tag],
        }
    nonfinite = [
        f"{side}:{name}"
        for side, info in sides.items()
        for name, m in info["maps"].items()
        if m is not None and not m["all_finite"]
    ]
    worst = max((v["rel"] for v in cross.values() if v is not None), default=0.0)
    return {
        "sides": sides,
        "cross": cross,
        "worst_map_rel": worst,
        "nonfinite_maps": nonfinite,
        "shapes_agree": sides["a"]["padded"] == sides["b"]["padded"],
        "dt_agrees": sides["a"]["dt"] == sides["b"]["dt"],
    }


def _u1b_stepdiff_one(
    job: dict,
    *,
    label: str,
    backends: tuple[str, str],
    n_steps: int,
    tol: float,
    force_bad_step: int | None = None,
) -> dict:
    """Step-by-step, then operator-by-operator, for ONE job on TWO backends.

    ``force_bad_step`` is for the rehearsal ONLY: it declares a step "the first
    bad one" whether or not it exceeded ``tol``, so the operator drill can be
    exercised on a machine where the two sides agree exactly. It never fires
    on the real stage (which passes ``None``), and the result carries the flag
    so no reader can mistake a forced drill for a real divergence.
    """
    from caustica.config.job import build_job, parse_job

    t0 = time.perf_counter()
    built = build_job(parse_job(job), None, with_medium=True)
    solver_name = built.solver
    ma = _StepMirror(backends[0], built, tuple(built.harmonics))
    mb = _StepMirror(backends[1], built, tuple(built.harmonics))
    # The medium's four property volumes are ~4 x P float32 on the host (1 GiB
    # at 400^3) and both mirrors have already absorbed them into their own
    # maps. Nothing below reads `built` again.
    del built
    inputs = _u1b_input_report(ma, mb)

    rows: list[dict] = []
    first_bad: int | None = None
    last_good: list | None = None
    for n in range(n_steps):
        # The rolling snapshot is the drill's starting point; once the first
        # bad step is known it costs memory and buys nothing, so it stops.
        prev = ma.state() if first_bad is None else None
        ma.step(n)
        ma.sync()
        mb.step(n)
        mb.sync()
        fields = {"p": _rel_linf(_host64(ma, ma.p), _host64(mb, mb.p))}
        for i in range(ma.nd):
            fields[f"u{i}"] = _rel_linf(_host64(ma, ma.u[i]), _host64(mb, mb.u[i]))
        worst_field = max(fields, key=lambda k: fields[k]["rel"])
        worst = fields[worst_field]["rel"]
        rows.append(
            {
                "step": n,
                "worst_rel": worst,
                "worst_field": worst_field,
                "rel": {k: v["rel"] for k, v in fields.items()},
                "p_max_a": fields["p"]["a_max"],
                "p_max_b": fields["p"]["b_max"],
                "finite_a": all(v["a_finite"] for v in fields.values()),
                "finite_b": all(v["b_finite"] for v in fields.values()),
            }
        )
        if first_bad is None and (worst > tol or n == force_bad_step):
            first_bad, last_good = n, prev
        del prev

    drill: list[dict] = []
    guilty: str | None = None
    drill_error = None
    if first_bad is not None and last_good is not None:
        try:
            # Both sides start from the SAME state — side A's, the reference —
            # so the one step that follows is the only difference between them.
            ma.load_state(last_good)
            mb.load_state(last_good)
            ga, gb = ma.step_traced(first_bad), mb.step_traced(first_bad)
            for (na, aa), (nb, bb) in zip(ga, gb, strict=True):
                if na != nb:
                    raise RuntimeError(f"traced steps desynchronized: {na!r} vs {nb!r}")
                d = _rel_linf(_host64(ma, aa), _host64(mb, bb))
                drill.append({"op": na, **d})
                if guilty is None and d["rel"] > tol:
                    guilty = na
                del aa, bb
        except Exception as exc:  # noqa: BLE001 - the step table still stands
            drill_error = f"{type(exc).__name__}: {exc}"
    del last_good

    return {
        "label": label,
        "backends": list(backends),
        "n_steps": n_steps,
        "tol": tol,
        "padded": list(ma.padded),
        "active": list(ma.active_shape),
        "solver": solver_name,
        "dt": ma.dt,
        "spp": ma.spp,
        "nonlinear": ma.nonlinear,
        "inputs": inputs,
        "steps": rows,
        "first_bad_step": first_bad,
        "forced_bad_step": force_bad_step,
        "worst_rel_overall": max((r["worst_rel"] for r in rows), default=0.0),
        "all_zero": all(r["worst_rel"] == 0.0 for r in rows),
        "drill": drill,
        "guilty_operator": guilty,
        "drill_error": drill_error,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def _u1b_stepdiff_lines(res: dict) -> list[str]:
    """The console rendering of one stepdiff result."""
    inp = res["inputs"]
    a, b = inp["sides"]["a"], inp["sides"]["b"]
    width = 92
    lines = [
        "",
        "=" * width,
        f"U1b-stepdiff — {res['label']}: {a['backend']} vs {b['backend']}",
        "=" * width,
        f"  padded {'x'.join(str(n) for n in res['padded'])}  active "
        f"{'x'.join(str(n) for n in res['active'])}  solver {res['solver']}  "
        f"nonlinear {res['nonlinear']}",
        f"  dt {res['dt']:.6e} s  spp {res['spp']}  steps {res['n_steps']}  tol {res['tol']:g}",
        f"  max|kappa| a={a['max_kappa']:.9g} b={b['max_kappa']:.9g}  "
        f"min|kappa| a={a['min_kappa']:.9g} b={b['min_kappa']:.9g}",
        f"  max|k| per axis a={[float(f'{v:.6g}') for v in a['max_k_per_axis']]}",
        f"  max|k| per axis b={[float(f'{v:.6g}') for v in b['max_k_per_axis']]}",
        f"  state dtypes  p {a['p_dtype']}/{b['p_dtype']}  u {a['u_dtype']}/{b['u_dtype']}  "
        f"src idx {a['src_idx_dtype']}/{b['src_idx_dtype']}  "
        f"phase {a['src_ph_dtype']}/{b['src_ph_dtype']}  "
        f"({a['n_source_voxels']} source voxels, pml {a['pml_vox']} vox)",
    ]
    for name, m in a["maps"].items():
        mb_ = b["maps"].get(name)
        cross = inp["cross"].get(name)
        if m is None and mb_ is None:
            lines.append(f"    map {name:<12} inactive on both sides")
            continue
        rel = "--" if cross is None else f"{cross['rel']:.3e}"
        lines.append(
            f"    map {name:<12} a[{m['dtype']}, max {m['max_abs']:.6g}, finite "
            f"{m['all_finite']}]  b[{mb_['dtype']}, max {mb_['max_abs']:.6g}, finite "
            f"{mb_['all_finite']}]  rel {rel}"
        )
    lines.append(
        f"  INPUTS: shapes agree {inp['shapes_agree']}, dt agrees {inp['dt_agrees']}, "
        f"worst map rel {inp['worst_map_rel']:.3e}, non-finite maps "
        f"{inp['nonfinite_maps'] or 'none'}"
    )
    nd = len(res["steps"][0]["rel"]) - 1 if res["steps"] else 3
    head = f"  {'step':>5} " + " ".join(
        f"{('rel ' + k):>12}" for k in ["p"] + [f"u{i}" for i in range(nd)]
    )
    head += f" {'worst':>12} {'field':>6} {'fin a/b':>9}"
    lines += [head, "  " + "-" * (len(head) - 2)]
    for r in res["steps"]:
        cells = " ".join(f"{r['rel'][k]:>12.3e}" for k in ["p"] + [f"u{i}" for i in range(nd)])
        lines.append(
            f"  {r['step']:>5} {cells} {r['worst_rel']:>12.3e} {r['worst_field']:>6} "
            f"{str(r['finite_a'])[0] + '/' + str(r['finite_b'])[0]:>9}"
        )
    lines.append("  " + "-" * (len(head) - 2))
    if res["first_bad_step"] is None:
        lines.append(
            f"  no step exceeded rel {res['tol']:g} in {res['n_steps']} steps "
            f"(worst {res['worst_rel_overall']:.3e}; all-zero: {res['all_zero']})"
        )
    else:
        forced = res.get("forced_bad_step") == res["first_bad_step"]
        lines.append(
            f"  {'FORCED (rehearsal) bad step' if forced else 'FIRST bad step'}: "
            f"{res['first_bad_step']} "
            f"(worst field {res['steps'][res['first_bad_step']]['worst_field']}, "
            f"rel {res['steps'][res['first_bad_step']]['worst_rel']:.3e})"
        )
        lines.append(
            f"  operator drill — both sides reloaded from side-A state after step "
            f"{res['first_bad_step'] - 1}, then ONE step:"
        )
        lines.append(f"    {'operator':<28} {'rel':>12} {'max a':>12} {'max b':>12} {'fin a/b':>9}")
        for d in res["drill"]:
            lines.append(
                f"    {d['op']:<28} {d['rel']:>12.3e} {d['a_max']:>12.6g} "
                f"{d['b_max']:>12.6g} "
                f"{str(d['a_finite'])[0] + '/' + str(d['b_finite'])[0]:>9}"
            )
        if res["drill_error"]:
            lines.append(f"    drill FAILED: {res['drill_error']}")
        lines.append(f"  FIRST guilty operator: {res['guilty_operator'] or '(none exceeded tol)'}")
    lines.append(f"  {res['elapsed_s']:.1f} s")
    lines.append("=" * width)
    return lines


#: What "the same" means between two backends' float32 states, per the task.
U1B_TOL = 1e-3


def _u1b_stepdiff(ctx: Ctx, backends: tuple[str, str] = ("numpy", "cupy")) -> dict:
    """Both stepdiff shapes, in THIS process, with per-shape error containment."""
    results = []
    for side in ctx.u1b_shapes:
        label = f"rung{side} ({side}^3)"
        # ~8 whole-volume transforms per step, and the numpy leg pays them on
        # one CPU core: minutes at 256^3, tens of minutes at 400^3. Said out
        # loud so a silent hour is never mistaken for a hang.
        print(
            f"  U1b-stepdiff {label}: {ctx.u1b_steps} steps on "
            f"{backends[0]} and {backends[1]} — the {backends[0]} leg is the slow "
            f"one (~8 full {side}^3 transforms per step) ...",
            flush=True,
        )
        try:
            res = _u1b_stepdiff_one(
                rung_job_dict(side),
                label=label,
                backends=backends,
                n_steps=ctx.u1b_steps,
                tol=U1B_TOL,
            )
        except Exception as exc:  # noqa: BLE001 - one shape must not lose the other
            results.append(
                {
                    "label": label,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"    {label} raised {type(exc).__name__}: {exc}", flush=True)
            continue
        for line in _u1b_stepdiff_lines(res):
            print(line, flush=True)
        results.append(res)
    return {"shapes": results}


def stage_u1b(ctx: Ctx) -> Outcome:
    """U1b — which shapes/settings diverge, and which operator does it first."""
    out = ctx.dir("U1b")
    matrix = _u1b_matrix(out)
    for line in matrix["table"]:
        print(line, flush=True)
    step = _u1b_stepdiff(ctx)

    rows = matrix["cases"]
    unknown = [r["case"] for r in rows if r["diverged"] is None]
    unseen = [r["case"] for r in rows if not r["saw_marker"]]
    broken = [s["label"] for s in step["shapes"] if s.get("error")]
    good = [s for s in step["shapes"] if not s.get("error")]

    numbers = ", ".join(f"{r['case']}={_NAN_CELL[r['diverged']]}" for r in rows)
    if good:
        numbers += "; " + ", ".join(
            f"{s['label'].split()[0]} first-bad-step "
            f"{'none' if s['first_bad_step'] is None else s['first_bad_step']}"
            for s in good
        )
    note_bits = [f"axis verdict: {matrix['interpretation']['axis_verdict']}"]
    supported = matrix["interpretation"]["supported"]
    note_bits.append("supported: " + (", ".join(supported) if supported else "(none)"))
    for s in good:
        if s["guilty_operator"]:
            note_bits.append(f"{s['label'].split()[0]} first guilty op: {s['guilty_operator']}")
    if unknown:
        note_bits.append(f"UNDETERMINED cases: {', '.join(unknown)}")
    if unseen:
        note_bits.append(f"no output from: {', '.join(unseen)}")
    if broken:
        note_bits.append(f"stepdiff FAILED for: {', '.join(broken)}")

    verdict = "WARN" if (unknown or unseen or broken) else "INFO"
    return Outcome(
        verdict,
        numbers,
        "; ".join(note_bits),
        # Sanitized AFTER the console tables were rendered from the real floats:
        # a diverged cupy leg puts inf in half these cells.
        _jsonsafe({"matrix": matrix, "stepdiff": step}),
    )


#: How many intermediates ``_StepMirror.step_traced`` yields in 3-D:
#: pk, then 3 x (deriv*pk, grad, u updated), then 3 x (rfftn(u), deriv*rfftn(u)),
#: then divu and the three pressure stages.
U1B_TRACE_OPS_3D = 1 + 3 * 3 + 3 * 2 + 1 + 3


def u1b_rehearse(n_steps: int = 8) -> int:
    """The GPU-less plumbing test: stepdiff numpy-vs-numpy on a mini scene.

    Two numpy mirrors of the SAME job must agree bit for bit at every step and
    at every intermediate — every relative difference exactly ``0.0``. Anything
    else is a bug in this file (a shared array where two were meant, a mirror
    that is not the engine's step, a comparison that lies), found on a laptop
    instead of on a GPU hour.

    Two passes, because "no divergence" would leave the more intricate half of
    the machinery unexecuted:

    1. the plain run — every step must be exactly 0.0 and NO step may be
       flagged;
    2. the same run with the drill FORCED at a mid-trajectory step, so
       ``load_state`` on both sides, the two lock-stepped ``step_traced``
       generators, the name check and the operator table all actually run —
       against a non-trivial field, and still at exactly 0.0.
    """
    _utf8_streams()
    job = rung_job_dict(U1B_REHEARSE_SIDE, apex_frac=U1B_REHEARSE_APEX)
    job["name"] = "u1b-rehearse"
    job["run"]["spec"]["max_settle_periods"] = U1B_MAX_SETTLE
    label = f"rehearse{U1B_REHEARSE_SIDE} ({U1B_REHEARSE_SIDE}^3)"
    print(
        f"U1b rehearsal — {U1B_REHEARSE_SIDE}^3 rung scene (apex at "
        f"{U1B_REHEARSE_APEX} x extent), numpy vs numpy, {n_steps} steps."
    )

    print("\n--- pass 1/2: identity (no step may be flagged) ---")
    res = _u1b_stepdiff_one(
        job, label=label, backends=("numpy", "numpy"), n_steps=n_steps, tol=U1B_TOL
    )
    for line in _u1b_stepdiff_lines(res):
        print(line)

    forced = max(0, n_steps - 3)
    print(f"\n--- pass 2/2: the operator drill, FORCED at step {forced} ---")
    drilled = _u1b_stepdiff_one(
        job,
        label=label + " [forced drill]",
        backends=("numpy", "numpy"),
        n_steps=n_steps,
        tol=U1B_TOL,
        force_bad_step=forced,
    )
    for line in _u1b_stepdiff_lines(drilled):
        print(line)

    checks = [
        (
            "every step rel diff is exactly 0.0",
            res["all_zero"],
            f"worst {res['worst_rel_overall']}",
        ),
        ("no step flagged bad", res["first_bad_step"] is None, str(res["first_bad_step"])),
        (
            "every map agrees exactly",
            res["inputs"]["worst_map_rel"] == 0.0,
            f"worst map rel {res['inputs']['worst_map_rel']}",
        ),
        ("no non-finite map", not res["inputs"]["nonfinite_maps"], "ok"),
        (
            "every field stayed finite",
            all(r["finite_a"] and r["finite_b"] for r in res["steps"]),
            "ok",
        ),
        (
            "the forced drill ran every intermediate",
            len(drilled["drill"]) == U1B_TRACE_OPS_3D and not drilled["drill_error"],
            f"{len(drilled['drill'])}/{U1B_TRACE_OPS_3D} ops, error {drilled['drill_error']}",
        ),
        (
            "every intermediate is exactly 0.0",
            all(d["rel"] == 0.0 for d in drilled["drill"]),
            f"worst {max((d['rel'] for d in drilled['drill']), default=None)}",
        ),
        (
            "the drill named no guilty operator",
            drilled["guilty_operator"] is None,
            str(drilled["guilty_operator"]),
        ),
        (
            "the drill saw a NON-trivial field (max|p| > 0)",
            any(d["a_max"] > 0.0 for d in drilled["drill"]),
            f"max over ops {max((d['a_max'] for d in drilled['drill']), default=0.0):.6g}",
        ),
    ]
    print("")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
    bad = [n for n, ok, _ in checks if not ok]
    if bad:
        print(f"\nREHEARSAL FAILED: {', '.join(bad)}")
        return EXIT_FAILED
    print("\nREHEARSAL PASSED — the stepdiff machinery is wired correctly.")
    return EXIT_OK


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
    # The honest quality number depends on what actually predicts: an
    # interpolated entry is graded by leave-one-out (its on-sample residual is
    # 0 by construction, and the GLOBAL fit's residual only says the fallback
    # would have been bad - on a curved device it reads 0.705 forever).
    loo = entry.get("interp_loo_rel_err")
    interpolated = entry.get("fit_mode") == "interpolated" and loo is not None
    quality_name = "interp_loo_rel_err < 0.10" if interpolated else "fit_max_rel_residual < 0.10"
    quality_val = loo if interpolated else residual
    warm_loo = entry.get("warmup_loo_rel_err")
    checks = [
        ("probe shapes recorded", bool(entry.get("shapes")), str(entry.get("shapes"))),
        (">=2 probe shapes", len(entry.get("shapes") or []) >= 2, str(entry.get("shapes"))),
        ("fit_mode recorded", bool(entry.get("fit_mode")), str(entry.get("fit_mode"))),
        (
            quality_name,
            quality_val is not None and float(quality_val) < cal.FIT_RESIDUAL_LIMIT,
            _num(quality_val),
        ),
        (
            "warmup_loo_rel_err recorded (info)",
            True,  # informational: absent on entries without >=3 warmup samples
            _num(warm_loo),
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
        f"{device}; fit {entry.get('fit_mode')} "
        f"{'loo' if interpolated else 'residual'} {_num(quality_val, '.3f')}; "
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
    "U1b": ("256^3 shape matrix + per-step/per-operator backend diff", stage_u1b),
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
    # U1b dev flags. --u1b-rehearse is the GPU-less proof that the stepdiff
    # machinery works at all (numpy vs numpy must be bit-identical); the other
    # two size the real stage.
    p.add_argument("--u1b-rehearse", action="store_true", help=argparse.SUPPRESS)
    # None -> the context's own default: 40 steps for the stage, 8 for the
    # rehearsal (where the point is the plumbing, not the trajectory).
    p.add_argument("--u1b-steps", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--u1b-shapes", default="256,400", help=argparse.SUPPRESS)
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

    if args.u1b_rehearse:
        # Deliberately BEFORE the profile check: the whole point is that it
        # runs on a machine with no GPU at all.
        return u1b_rehearse(max(1, int(args.u1b_steps)) if args.u1b_steps else 8)

    if not args.profile:
        print("dev_validate: --profile local|colab is required", file=sys.stderr)
        return EXIT_ENV

    available = PROFILE_STAGES[args.profile]
    if args.stages:
        # Case-insensitive so "--stages u1b", "U1B" and "U1b" all name the
        # stage whose canonical label carries a lower-case suffix.
        canon = {s.upper(): s for s in available}
        wanted = [s.strip().upper() for s in args.stages.replace(",", " ").split()]
        bad = [s for s in wanted if s not in canon]
        if bad:
            print(
                f"dev_validate: {', '.join(bad)} is not in the {args.profile} profile "
                f"({', '.join(available)})",
                file=sys.stderr,
            )
            return EXIT_ENV
        selected = [s for s in available if s.upper() in wanted]
    else:
        selected = list(available)

    _require_profile_env(args.profile)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(args.out) if args.out else Path("dev_validate_reports") / f"{args.profile}-{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        shapes = tuple(int(s) for s in str(args.u1b_shapes).replace(",", " ").split())
    except ValueError:
        print(
            f"dev_validate: --u1b-shapes {args.u1b_shapes!r} is not a list of ints", file=sys.stderr
        )
        return EXIT_ENV
    ctx = Ctx(
        profile=args.profile,
        root=root,
        u1b_steps=max(1, int(args.u1b_steps)) if args.u1b_steps else Ctx.u1b_steps,
        u1b_shapes=shapes or Ctx.u1b_shapes,
    )

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
