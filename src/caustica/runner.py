"""The runner (M10c): execute ONE ``caustica-job/1`` file, plan-first and stamped.

This is the single entry point the Colab notebook cell calls — and the same
command runs locally on numpy. Design rule inherited from ``focus_study``:
NOTHING expensive happens before the planner has spoken. The plan is printed
and saved first, a run that cannot fit is refused with actionable advice
BEFORE it starts, and everything that later needs auditing (job copy, git
commit, environment, planner-vs-actual) is stamped into the output folder.

Output folder layout (deterministic — resume depends on it)::

    <out>/
      job.json          normalized copy of the job that ran
      plan.json/.txt    planner verdict, written BEFORE solving
      status.json       heartbeat: {state, step k/N, ETA} — watch it over
                        Drive sync without opening the Colab tab
      checkpoint.npz    in-run state (M10); removed on success
      result.h5         caustica-result/1 (M10 store)
      preview.npz       <=10 MB quick-look package (M10d): peak slices +
                        coarse amp volume — `caustica report` renders it
                        without touching result.h5
      metrics.json      focal metrics (caustica.report.metrics — the same
                        definitions focus_study uses)
      run_meta.json     the stamp: env + git + planner vs actual + derived
                        geometry (M8's two Colab gates measure themselves
                        from this file)
      error.json        why a FAILED run failed (M10l): the structured twin
                        of the stderr message, written even for failures
                        that happen before solving starts
      cancel            INPUT, not output (M10l): a caller creates this file
                        to ask a running solve to stop; see below

Exit codes are DISJOINT so a queue can react without parsing text:
0 success (or already complete) · 2 config error · 3 OOM refusal ·
4 solver error · 5 interrupted-but-resumable (``--max-hours``, or a
``cancel`` file).

Cancel protocol (M10l — the GUI's "Stop" button, and the reason killing the
process is not the only way out): create an (empty) file named ``cancel`` in
the output folder. The next PERIOD BOUNDARY sees it — one ``stat`` per
period, never per step — writes a checkpoint and exits 5. The runner then
removes the file, so a stopped folder never advertises a stop nobody will
honor; the ``--resume`` is carried by the separate clear at the start of
every real run. A run that completed with ``--resume`` is BIT-IDENTICAL to
the uninterrupted one. Only the native engine takes checkpoints, so only native
solvers can be cancelled this way; a ``kwave`` job ignores the file, because
stopping it would lose the whole run rather than pause it.

The whole surface a GUI may rely on — this folder, the exit codes,
``error.json``, ``cancel`` — is written down in ``docs/gui_contract.md``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

import caustica
from caustica.config.job import BuiltJob, build_job, dump_job, load_job
from caustica.core.backend import CausticaWarning, check_backend_name, get_backend
from caustica.env import env_report, git_commit, gpu_environment
from caustica.io.atomic import atomic_write
from caustica.io.checkpoint import CheckpointSpec, RunInterrupted
from caustica.io.store import (
    ensure_dir_verified,
    probe_writable,
    save_result,
    validate_result_file,
)
from caustica.progress import chain as progress_chain
from caustica.progress import close as progress_close
from caustica.progress import resolve as progress_resolve

log = logging.getLogger("caustica")

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_OOM = 3
EXIT_SOLVER = 4
EXIT_INTERRUPTED = 5

#: Structured failure record (M10l). Every non-zero exit that has an output
#: folder writes one; a successful run writes none and a new attempt deletes
#: the previous one, so its presence always means "this folder failed".
ERROR_FORMAT = "caustica-error/1"
ERROR_FILE = "error.json"
#: The keys of ``error.json`` — the contract, in the order written.
ERROR_KEYS = ("format", "stage", "exit_code", "error_class", "message", "advice", "written_at")
#: Where the failure happened. ``stage`` is coarse on purpose: a GUI routes
#: on it (retry / edit the job / pick a bigger GPU), it does not parse it.
ERROR_STAGES = ("config", "plan", "gate", "checkpoint", "solve", "store")

#: Cancel request file (M10l). A caller creates it in the output folder; the
#: solve polls for it ONCE PER PERIOD BOUNDARY and stops resumably (exit 5).
CANCEL_FILE = "cancel"

#: Solvers the planner models (the native k-space engine); anything else
#: (e.g. the external k-Wave binary) runs without a plan or a checkpoint.
_NATIVE_SOLVERS = ("linear", "westervelt")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict) -> None:
    with atomic_write(path) as tmp:
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_error_json(
    outdir: Path | None,
    *,
    stage: str,
    exit_code: int,
    error_class: str,
    message: str,
    advice: tuple[str, ...] | list[str] = (),
) -> None:
    """Record WHY this run failed, as data instead of stderr prose (M10l).

    Best effort by construction, and deliberately so: this is an ADDITION to
    the existing failure contract, never a replacement. The exit code and the
    stderr text are unchanged and stay authoritative — if the folder does not
    exist, cannot be created, or the write fails, the run still exits with the
    same code and the same message.

    ``outdir`` is None exactly when there is nowhere honest to write: the run
    failed before an output folder could be established (a job that would not
    load AND no explicit ``--out``, so the folder's own name was never known),
    or creating it is what failed. No path is invented in that case.
    """
    if outdir is None:
        return
    try:
        payload = {
            "format": ERROR_FORMAT,
            "stage": stage,
            "exit_code": int(exit_code),
            "error_class": error_class,
            "message": message,
            "advice": list(advice),
            "written_at": _now_iso(),
        }
        _write_json(Path(outdir) / ERROR_FILE, payload)
    except Exception as exc:  # noqa: BLE001 - must never mask the real failure
        log.warning("error.json write failed: %s", exc)


def _error_outdir(established: Path | None, opts: RunnerOptions) -> Path | None:
    """The folder ``error.json`` may go in when the run died before setup.

    An explicit ``--out`` names the folder without reading the job, which is
    the case a GUI always has — so a job that will not even parse still leaves
    a machine-readable reason behind. Without ``--out`` the folder name comes
    FROM the job (``output.folder`` or ``runs/<name>``), so a job that failed
    to load leaves nowhere to write and this returns None rather than guessing.
    """
    if established is not None:
        return established
    if opts.out is None:
        return None
    try:
        # Through ensure_dir_verified, like every other folder creation here:
        # a bare mkdir is not proof on a Drive FUSE mount, and a phantom
        # directory that the rest of the runner would have rejected is not
        # something a failure path should leave behind (review, 2026-08-22).
        return ensure_dir_verified(Path(opts.out))
    except Exception:  # noqa: BLE001 - an unwritable --out is not a new failure
        return None


def _cancel_requested(path: Path) -> bool:
    """The ONE place the ``cancel`` file is polled — deliberately a function.

    The poll's cost contract (one ``stat`` per acoustic period boundary, never
    per step) is only TESTABLE if there is a single call site to count, and
    counting a filesystem primitive instead is not enough: on Windows/py3.12
    ``os.path.exists`` is ``nt._path_exists``, a C shortcut that never reaches
    ``os.stat``, so a per-step poll written that way is invisible to a test
    that instruments ``Path.exists`` or ``os.stat`` (mutation review,
    2026-08-22). ``tests/test_runner.py`` counts calls to THIS function, and
    separately watches every other spelling for a poll that bypassed it.

    ``is_file``, not ``exists``: a *directory* named ``cancel`` can never be
    unlinked, so treating it as a request would stop every run in that folder
    at period 1 — ``--resume`` included, forever.
    """
    return path.is_file()


def _clear_stale(path: Path) -> None:
    """Remove a leftover ``error.json``/``cancel`` from a previous attempt."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - locked file on Windows
        log.warning("could not remove stale %s: %s", path.name, exc)


# Promoted to caustica.env (fix A1) — and no longer git-only: a wheel install
# has no checkout, so the commit now falls back to the stamp the build froze
# into the package. Colab installs from a wheel, which is exactly where every
# run used to be stamped "unknown".
_git_commit = git_commit


# Promoted to caustica.env (M10i) — the runner keeps calling the same
# function, so the run_meta stamp and a notebook's env_report() cannot
# disagree. The alias stays because the VRAM block below uses it.
_gpu_environment = gpu_environment


def _vram_pool_peak_gib(backend_name: str) -> float | None:
    if backend_name != "cupy":
        return None
    try:  # pragma: no cover - requires a GPU
        import cupy

        return round(cupy.get_default_memory_pool().total_bytes() / 2**30, 3)
    except Exception:
        return None


class _StepTiming:
    """Split a solve's wall time into one-time warmup and steady per-step cost.

    ``t_step_measured_s`` (``elapsed / steps``) is honest arithmetic and a
    misleading number. On the first real GPU session (A100-SXM4-40GB, Colab,
    2026-08-22) it read 26.6 ms/step against a 1.03 ms/step probe — a 25.9x
    "miss" that was almost entirely a ~2.7 s one-time CUDA/JIT/cuFFT-plan cost
    smeared over a 104-step, 2.77 s run. The SAME job on the CPU came out at
    0.96x. The accounting was right; the scale hid a constant.

    So this consumes the engine's period-boundary payload — no new
    instrumentation, no extra device sync, nothing the solve can feel — and
    reads the pairs ``(step, elapsed_s)`` it already carries. The steady cost
    is the MEDIAN of the per-step rates BETWEEN boundaries; the interval from
    the solve's start to the first boundary is never one of them, because that
    is exactly the interval that pays for warmup. Warmup is then what the
    total does not explain.

    Deliberately conservative: fewer than three boundaries means fewer than
    two intervals to take a median over, and the answer is ``None`` — a stamp
    that says "not measured" beats one that says a number it cannot support.
    The historical keys are untouched; these are ADDITIONS (M8's Colab gates
    and ``docs/gui_contract.md`` read the old ones).
    """

    def __init__(self) -> None:
        self.samples: list[tuple[int, float]] = []

    def __call__(self, ev: dict) -> None:
        try:
            self.samples.append((int(ev["step"]), float(ev["elapsed_s"])))
        except Exception:  # noqa: BLE001 - telemetry must never fail a solve
            pass

    def steady_step_s(self) -> float | None:
        """Median seconds/step between period boundaries, or None."""
        if len(self.samples) < 3:
            return None
        rates = [
            (t1 - t0) / (s1 - s0)
            for (s0, t0), (s1, t1) in zip(self.samples, self.samples[1:], strict=False)
            # The settle->record transition emits at the SAME step count as
            # the boundary before it; a zero-step interval is not a rate.
            if s1 > s0 and t1 > t0
        ]
        if len(rates) < 2:
            return None
        return float(median(rates))

    def split(self, elapsed_s: float, session_steps: int) -> dict:
        """``{warmup_s, t_step_steady_s, steady_samples}`` for the stamp."""
        steady = self.steady_step_s()
        return {
            "warmup_s": (
                None if steady is None else round(max(0.0, elapsed_s - session_steps * steady), 3)
            ),
            "t_step_steady_s": None if steady is None else round(steady, 6),
            "steady_samples": len(self.samples),
        }


class _Heartbeat:
    """status.json writer: one tick per acoustic period, throttled to disk.

    Since M10j the heartbeat is a CONSUMER of the engine's progress payload
    (``__call__``), not a second instrumentation of the same boundary: the
    engine emits one dict per period, this writes the subset ``status.json``
    has always carried. Step counts stay DERIVED (periods * spp) and the ETA
    stays measured from THIS run's cadence, not the planner's guess.

    Known (accepted) imprecision, unchanged on purpose: the boundary fires
    once more just before the record window, so a run that reaches recording
    reports periods_done one higher than the settle count. Status numbers are
    progress telemetry, not provenance — the checkpoint meta and run_meta
    carry the exact counters, and M8's Colab gates measure themselves from
    this file, so its numbers do not move for cosmetics.
    """

    def __init__(
        self,
        path: Path,
        base: dict,
        spp: int,
        steps_expected: int | None,
        steps_worst: int | None,
        interval_s: float,
        offset_periods: int = 0,
    ):
        self.path = path
        self.base = base
        self.spp = spp
        self.steps_expected = steps_expected
        self.steps_worst = steps_worst
        self.interval_s = interval_s
        self.periods = offset_periods
        self._session_periods = 0
        self._t0 = time.monotonic()
        self._last_write = 0.0

    @property
    def session_periods(self) -> int:
        return self._session_periods

    def __call__(self, ev: dict) -> None:
        """Consume one progress payload (the engine's period boundary).

        Counts events rather than reading ``ev["period"]``: the engine knows
        the true period number, but status.json has always reported the
        boundary count (see the class docstring), and that is the number the
        Colab gates were calibrated against.
        """
        self.periods += 1
        self._session_periods += 1
        now = time.monotonic()
        if now - self._last_write >= self.interval_s:
            self.write("solving")

    def write(self, state: str, **extra) -> None:
        elapsed = time.monotonic() - self._t0
        steps_done = self.periods * self.spp
        eta = None
        if self.steps_expected and self._session_periods > 0:
            t_step = elapsed / (self._session_periods * self.spp)
            eta = round(max(0.0, (self.steps_expected - steps_done) * t_step), 1)
        payload = {
            **self.base,
            "state": state,
            "periods_done": self.periods,
            "steps_done": steps_done,
            "steps_expected": self.steps_expected,
            "steps_worst": self.steps_worst,
            "eta_s": eta,
            "elapsed_s": round(elapsed, 1),
            "written_at": _now_iso(),
            **extra,
        }
        try:
            _write_json(self.path, payload)
            self._last_write = time.monotonic()
        except OSError as exc:  # a flaky Drive mount must not kill the solve
            log.warning("status.json write failed: %s", exc)


@dataclass
class RunnerOptions:
    """Everything ``python -m caustica run`` can set (defaults = the CLI's)."""

    out: str | Path | None = None
    backend: str | None = None  # None -> the job's backend field
    gpu: str = "A100"  # datasheet estimate target (always reported)
    measure: bool = True  # ~20-step local timing probe
    dry_run: bool = False
    resume: bool = False
    max_hours: float | None = None
    checkpoint_every: int = 8
    status_interval_s: float = 30.0
    vram_limit_gib: float | None = None  # None -> actual device VRAM on cupy
    stop_after_periods: int | None = None  # deterministic stop (tests/ops)
    allow_slow_cpu: bool = False  # M10i/D20: override the CPU time gate
    preview_only: bool = False  # M10i/D34: skip result.h5, keep the preview
    #: M10j progress display: None (silent — the library default, so a test
    #: or an embedding app gets no surprise output), "auto"/"plain", or any
    #: callable taking the payload dict. status.json is written either way:
    #: the heartbeat is always a consumer, this only adds a second one.
    progress: Any = None


def _cpu_limit_min() -> float:
    """CPU refusal threshold in minutes (``CAUSTICA_CPU_LIMIT_MIN``, default 5)."""
    try:
        return float(os.environ.get("CAUSTICA_CPU_LIMIT_MIN", "5"))
    except ValueError:
        return 5.0


def _cpu_time_estimate(est, grid_shape: tuple[int, ...], opts: RunnerOptions):
    """CPU wall-time estimate feeding the D20 gate: ``(seconds, source)``.

    With the default ``measure=True`` the planner already timed ~20 real
    steps on THIS machine (source ``"measured"``) — trustworthy. With
    ``--no-measure`` the plan's number targets the ``--gpu`` datasheet, so
    gating on it would let a 10-hour CPU job through; rescale through the
    calibrated cpu entry instead. ``None`` when neither exists — the gate
    then cannot judge and says so instead of pretending.
    """
    if opts.measure:
        return est.t_expected_s, est.source
    from caustica.planner.calibration import find_calibration_for  # noqa: PLC0415
    from caustica.planner.model import fft_sizes, step_time  # noqa: PLC0415

    entry = find_calibration_for("cpu")
    if entry is None:
        return None
    _, p_elems, _ = fft_sizes(tuple(grid_shape))
    return est.steps_expected * step_time(entry["a"], entry["b"], p_elems), "calibrated"


def _ppw_warnings_for(built: BuiltJob) -> list[str]:
    """Low-ppw warnings for a built job (single source: config.job)."""
    from caustica.config.job import _APPROX_C_MIN, low_ppw_warnings  # noqa: PLC0415

    c_min = built.c_min_hint
    approx = ""
    if c_min is None and built.medium is not None:
        c_min = float(built.medium.c_min)
    if c_min is None:
        c_min, approx = _APPROX_C_MIN, " (approx. c_min)"
    return low_ppw_warnings(built.grid, built.source.f0, built.harmonics, c_min, approx)


def _plan(built: BuiltJob, backend_name: str, opts: RunnerOptions):
    """Planner verdicts: datasheet GPU + (optionally) measured-here."""
    from caustica import planner  # noqa: PLC0415 (keep import caustica light)

    common = dict(
        solver=built.solver,
        harmonics=built.harmonics,
        record_region=built.record_region,
        reference_point=built.focus_vox,
    )
    est_gpu = planner.estimate(
        built.grid, built.medium, built.source, built.spec, gpu=opts.gpu, **common
    )
    est_here = (
        planner.estimate(
            built.grid,
            built.medium,
            built.source,
            built.spec,
            gpu=opts.gpu,
            measure=True,
            # The probe must time the backend the run will USE — "auto" on a
            # GPU machine forced to numpy would time cuFFT and feed the CPU
            # gate an hours-wrong "measured" number (review, 2026-08-22).
            measure_backend=backend_name,
            **common,
        )
        if opts.measure
        else est_gpu
    )
    # Expected result.h5 size (M10i/D34): the disk cost of a run is a choice
    # the user must SEE before a multi-GB file lands on a Drive mount.
    rec = built.record_region
    if rec is None:
        n_rec = int(np.prod(built.grid.shape))
    else:
        n_rec = int(np.prod([sl.stop - sl.start for sl in rec]))
    nh = len(built.harmonics)
    if built.output.quantize:  # float16 pairs per phasor + float16 p_max
        result_bytes = n_rec * (nh * 4 + 2)
    else:  # complex64 phasors + float32 p_max
        result_bytes = n_rec * (nh * 8 + 4)
    result_mb = result_bytes / 2**20

    payload = {
        "source": est_here.source,
        "spp": est_here.spp,
        "dt_s": est_here.dt,
        "t_step_s": est_here.t_step_s,
        # One-time, NOT per step (fix A2): t_expected_s = warmup_s +
        # steps_expected * t_step_s. Reported separately so a caller can tell
        # a slow device from a short run that is mostly warmup.
        "warmup_s": round(est_here.warmup_s, 3),
        "steps_expected": est_here.steps_expected,
        "steps_worst": est_here.steps_worst,
        "t_expected_s": round(est_here.t_expected_s, 1),
        "t_worst_s": round(est_here.t_worst_s, 1),
        "vram_gib": round(est_here.vram_gib, 3),
        "vram_breakdown_bytes": dict(est_here.vram_breakdown),
        "result_size_mb_expected": round(result_mb, 1),
        "gpu": opts.gpu,
        "gpu_t_expected_s": round(est_gpu.t_expected_s, 1),
        "gpu_fits": est_gpu.fits,
        "warnings": list(est_here.warnings),
        "advice": list(est_gpu.advice),
    }
    text = "\n".join(
        [
            "--- planner ------------------------------------------------------",
            f"this machine [{est_here.source}]: t_step={est_here.t_step_s * 1e3:.2f} ms, "
            f"expected {est_here.t_expected_s:.0f} s "
            f"({est_here.steps_expected}-{est_here.steps_worst} steps, spp={est_here.spp})",
            f"memory for this run: {est_here.vram_gib:.2f} GiB",
            f"expected result.h5 size: ~{result_mb:,.1f} MB "
            f"({'float16-quantized' if built.output.quantize else 'float32'}, "
            f"{nh} harmonic{'s' if nh > 1 else ''}; --preview-only skips it)",
            est_gpu.summary(),
        ]
    )
    return est_here, payload, text


@dataclass(frozen=True)
class Refusal:
    """A pre-run gate said no: the exit code, the message, and why.

    One object, two presentations: ``run_job_file`` prints ``lines`` to stderr
    and returns ``exit_code``; :func:`caustica.simulate` raises
    :class:`~caustica.facade.SimulationError` carrying the same text and the
    same code in ``.exit_code`` (there is no separate refusal class — the
    number is the classification). The gates themselves exist exactly ONCE — an in-memory run that
    skipped them would be the "works on my laptop, dies on Colab" bug the
    plan-first discipline exists to prevent.

    Since M10l the advice is stored as a LIST, not baked into the printed
    lines: the same strings feed ``error.json``'s ``advice[]``, and a copy
    kept only for the file would drift from the one shown on screen.
    """

    kind: str  # "vram" | "cpu"
    exit_code: int
    headline: str
    advice: tuple[str, ...] = ()
    note: str = ""  # what --dry-run prints instead of refusing

    #: ``kind`` -> the name a GUI switches on in ``error.json``.
    _CLASSES = {"vram": "VramRefusal", "cpu": "CpuTimeRefusal"}

    @property
    def lines(self) -> tuple[str, ...]:
        """The stderr rendering: headline, then one indented arrow per advice."""
        return (self.headline, *(f"  -> {a}" for a in self.advice))

    @property
    def message(self) -> str:
        return "\n".join(self.lines)

    @property
    def error_class(self) -> str:
        return self._CLASSES.get(self.kind, "Refusal")


def check_gates(
    built: BuiltJob,
    est,
    backend_name: str,
    opts: RunnerOptions,
    gpu_env: dict,
    ck_exists: bool = False,
) -> Refusal | None:
    """The two pre-run gates (M10i): device memory, then CPU wall time.

    Returns ``None`` to proceed. Warnings that do NOT block (an accepted slow
    CPU run, an unjudgeable one, a plain numpy notice) are raised here so both
    callers emit them identically.
    """
    limit_gib = opts.vram_limit_gib
    limit_label = "requested limit (--vram-limit-gib)"
    if limit_gib is None and backend_name == "cupy":
        gpu_name = gpu_env.get("gpu_name", "unknown GPU")
        # FREE VRAM, not total (M10i): the CUDA context alone eats
        # 0.8-1.5 GB on Colab — gating on the total says "fits" and then
        # dies OOM mid-run. The message names which limit was used.
        # (gpu_env is the PRE-probe snapshot — see the caller.)
        limit_gib = gpu_env.get("vram_free_gib")
        limit_label = f"free device VRAM ({gpu_name})"
        if limit_gib is None:
            limit_gib = gpu_env.get("vram_total_gib")
            limit_label = f"total device VRAM ({gpu_name}; free VRAM unavailable)"
    if limit_gib is not None and est.vram_gib > limit_gib:
        return Refusal(
            kind="vram",
            exit_code=EXIT_OOM,
            headline=(
                f"REFUSED before solving: this run needs {est.vram_gib:.2f} GiB but "
                f"{limit_label} is {limit_gib:.2f} GiB."
            ),
            # The planner's own advice, verbatim — printed AND written to
            # error.json (M10l), where it is the actionable part for a GUI.
            advice=tuple(
                est.advice
                or ("coarsen dx, shrink the record region, or switch to the linear solver",)
            ),
        )

    # ---- CPU gate (M10i/D20): refuse an hours-long numpy run BEFORE
    # paying for it; the message names its own escapes. Reuses
    # EXIT_CONFIG — the exit-code set is the queue's API (no sixth code).
    if backend_name == "numpy" and opts.resume and ck_exists:
        # An explicit --resume of an existing checkpoint is its own
        # acceptance: the sunk periods are already paid for, and a
        # refusal here would strand pre-gate checkpoints forever
        # (review finding, 2026-08-22 — reproduced).
        warnings.warn(
            "resuming an interrupted CPU run: the slow-CPU gate is bypassed for "
            "an explicit --resume of an existing checkpoint.",
            CausticaWarning,
            stacklevel=2,
        )
    elif backend_name == "numpy":
        limit_min = _cpu_limit_min()
        cpu_est = _cpu_time_estimate(est, built.grid.shape, opts)
        if cpu_est is None:
            warnings.warn(
                "running on the numpy (CPU) backend with NO wall-time estimate "
                "(--no-measure and no cpu calibration): the slow-CPU gate cannot "
                "judge this run. Drop --no-measure or run planner.calibrate() once.",
                CausticaWarning,
                stacklevel=2,
            )
        else:
            t_cpu, cpu_src = cpu_est
            if t_cpu > limit_min * 60.0 and not opts.allow_slow_cpu:
                return Refusal(
                    kind="cpu",
                    exit_code=EXIT_CONFIG,
                    headline=(
                        f"REFUSED before solving: estimated wall time on the numpy (CPU) "
                        f"backend is ~{t_cpu:.0f} s (~{t_cpu / 3600:.1f} h, estimate "
                        f"source: {cpu_src}), over the {limit_min:g} min CPU limit."
                    ),
                    advice=(
                        "run on a GPU backend (--backend cupy / backend='cupy'), or",
                        "accept the wait: --allow-slow-cpu (CLI) / "
                        "allow_slow_cpu=True; the threshold itself is "
                        "CAUSTICA_CPU_LIMIT_MIN (minutes).",
                    ),
                    note=(
                        f"NOTE: a REAL run would be refused here — estimated "
                        f"~{t_cpu:.0f} s (~{t_cpu / 3600:.1f} h, source: {cpu_src}) on "
                        f"the numpy backend, over the {limit_min:g} min CPU limit "
                        f"(escapes: --backend cupy, --allow-slow-cpu)."
                    ),
                )
            if t_cpu > limit_min * 60.0:
                warnings.warn(
                    f"slow CPU run ACCEPTED via allow_slow_cpu: estimated ~{t_cpu:.1f} s "
                    f"(~{t_cpu / 3600:.1f} h, source: {cpu_src}) on the numpy backend.",
                    CausticaWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"running on the numpy (CPU) backend: estimated ~{t_cpu:.1f} s "
                    f"(source: {cpu_src}). A GPU backend would be faster for real runs.",
                    CausticaWarning,
                    stacklevel=2,
                )
    return None


def _write_preview_package(built: BuiltJob, result, outdir: Path, apex_vox: tuple) -> None:
    """The M10d preview next to the result: metrics.json + preview.npz.

    Called only AFTER the result is safely stored; any failure here must
    never turn a successful run into a failed one — the caller warns.
    """
    from caustica.report import preview as _preview  # noqa: PLC0415
    from caustica.report.metrics import focus_metrics  # noqa: PLC0415

    metrics = focus_metrics(
        result,
        dx=built.grid.dx,
        grid_shape=built.grid.shape,
        pml_vox=built.grid.pml_vox,
        apex_vox=apex_vox,
        focus_vox=built.focus_vox,
        source_amplitude=built.source.amplitude,
        medium=built.medium,
        solver=built.solver,
    )
    _preview.write_preview(
        outdir,
        result,
        dx=built.grid.dx,
        grid_shape=built.grid.shape,
        pml_vox=built.grid.pml_vox,
        apex_vox=apex_vox,
        focus_vox=built.focus_vox,
        metrics={
            "format": "caustica-metrics/1",
            "job": built.name,
            "generated": _now_iso(),
            **metrics,
        },
    )


def run_job_file(job_path: str | Path, opts: RunnerOptions | None = None) -> int:
    """Execute one job file. Returns a disjoint exit code (module constants)."""
    opts = opts or RunnerOptions()
    t_start = time.perf_counter()
    outdir: Path | None = None

    def record_failure(where: Path | None, **fields) -> None:
        """Write ``error.json`` — unless this is a ``--dry-run``.

        A dry run is a PROBE, not an attempt on the folder: it must not
        erase the failure record a GUI is displaying, and it must not
        invent one either. Its verdict is the exit code plus ``plan.json``,
        which already carries the planner's advice (review, 2026-08-22).
        """
        if not opts.dry_run:
            _write_error_json(where, **fields)

    # ---- everything before the solve is, by definition, a config problem ----
    try:
        # BEFORE the medium is built: `--backend` used to be an argparse
        # `choices=`, so a typo was refused instantly. Since M10n opened the
        # name to the registry, refusing it here keeps that — otherwise a
        # misspelled backend costs a multi-GB medium build first.
        if opts.backend is not None:
            check_backend_name(opts.backend)
        job, base_dir = load_job(job_path)
        built = build_job(job, base_dir=base_dir, with_medium=True)
        backend_name = get_backend(opts.backend or built.backend).name
        # Relative output paths resolve against the JOB FILE, like every
        # other relative path in a job — resolving against the CWD would make
        # --resume from a different CWD silently restart from period 0
        # (adversarial review, 2026-08-19). An explicit --out stays CWD-based
        # (normal CLI semantics).
        if opts.out is not None:
            outdir_raw = Path(opts.out)
        else:
            outdir_raw = (
                Path(built.output.folder) if built.output.folder else Path("runs") / built.name
            )
            if not outdir_raw.is_absolute():
                outdir_raw = base_dir / outdir_raw
        outdir = ensure_dir_verified(outdir_raw)
        probe_writable(outdir)
    except Exception as exc:
        print(f"CONFIG ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        record_failure(
            _error_outdir(outdir, opts),
            stage="config",
            exit_code=EXIT_CONFIG,
            error_class=type(exc).__name__,
            message=f"{type(exc).__name__}: {exc}",
            advice=(
                f"run `caustica validate {job_path}` for the full list of problems",
                "`caustica schema` prints the caustica-job/1 JSON Schema",
            ),
        )
        return EXIT_CONFIG

    result_path = outdir / "result.h5"
    ck_path = outdir / "checkpoint.npz"
    status_path = outdir / "status.json"
    cancel_path = outdir / CANCEL_FILE
    error_path = outdir / ERROR_FILE

    # ---- file-level resume: a complete result is never produced twice ----
    if result_path.exists() and validate_result_file(result_path):
        # A folder holding a complete result did not fail, so an old failure
        # record is stale. `cancel` is deliberately NOT touched here: it may
        # belong to a run still in flight in another process.
        _clear_stale(error_path)
        print(f"already complete: {result_path} (skip-guard; delete it to regenerate)")
        return EXIT_OK

    # ---- plan FIRST; refuse before paying, not after ----
    # Still config territory: an unknown --gpu name or a Drive hiccup on the
    # plan writes must exit 2, not leak a raw traceback with exit 1.
    native = built.solver in _NATIVE_SOLVERS
    est = plan_payload = None
    # Low ppw is loud in four places (M10i/D31): plan, status.json,
    # run_meta.json and the report head. Ignorable — never a block.
    ppw_warns = _ppw_warnings_for(built)
    # GPU facts are snapshotted BEFORE the plan: the measure probe fills the
    # cupy memory pool with ~the run's own footprint and the pool keeps its
    # blocks, so free VRAM read AFTER the probe would falsely refuse any job
    # over ~half the free VRAM (review finding, 2026-08-22).
    gpu_env = _gpu_environment(backend_name) if backend_name == "cupy" else {}
    try:
        dump_job(job, outdir / "job.json")
        if native:
            est, plan_payload, plan_text = _plan(built, backend_name, opts)
            if ppw_warns:
                plan_text += "\n" + "\n".join(f"  ! WARNING: {w}" for w in ppw_warns)
            plan_payload["ppw_warnings"] = ppw_warns
            print(plan_text)
            _write_json(outdir / "plan.json", plan_payload)
            with atomic_write(outdir / "plan.txt") as tmp:
                tmp.write_text(plan_text, encoding="utf-8")
    except Exception as exc:
        print(f"CONFIG ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        record_failure(
            outdir,
            stage="plan",
            exit_code=EXIT_CONFIG,
            error_class=type(exc).__name__,
            message=f"{type(exc).__name__}: {exc}",
            advice=("re-run with --dry-run to see how far the plan gets",),
        )
        return EXIT_CONFIG
    if ppw_warns:
        warnings.warn(
            "low spatial resolution: " + " | ".join(ppw_warns),
            CausticaWarning,
            stacklevel=2,
        )

    if native:
        refusal = check_gates(built, est, backend_name, opts, gpu_env, ck_exists=ck_path.exists())
        if refusal is not None:
            if refusal.kind == "cpu" and opts.dry_run:
                # A dry run pays nothing, and planning a Colab-bound job on a
                # CPU box is a legitimate flow — preview the verdict without
                # breaking the dry-run exit-0 contract (review, 2026-08-22).
                print(refusal.note)
            else:
                for line in refusal.lines:
                    print(line, file=sys.stderr)
                record_failure(
                    outdir,
                    stage="gate",
                    exit_code=refusal.exit_code,
                    error_class=refusal.error_class,
                    message=refusal.headline,
                    advice=refusal.advice,
                )
                return refusal.exit_code
    else:
        print(f"(planner models the native engine only; '{built.solver}' runs unplanned)")
        # Honest about the gap rather than silently ignoring the file: no
        # checkpoint means no period boundary to stop AT, so a cancel could
        # only kill the run, which is what the file exists to avoid (M10l).
        print(f"(no checkpoints for '{built.solver}': a '{CANCEL_FILE}' file has no effect)")

    if opts.dry_run:
        print("\n(dry run — nothing solved, nothing written beyond the plan)")
        return EXIT_OK

    # From here a REAL run owns the folder: a failure record and a stop
    # request left by the PREVIOUS attempt are stale by definition. Clearing
    # `cancel` also stops a process killed between "cancel seen" and "cancel
    # honored" from cancelling every resume that follows, forever (M10l).
    _clear_stale(error_path)
    _clear_stale(cancel_path)

    # ---- checkpoint policy: resuming is EXPLICIT ----
    offset_periods = 0
    if ck_path.exists():
        if not opts.resume:
            print(
                f"CONFIG ERROR: {ck_path} exists — a previous run was interrupted. "
                f"Rerun with --resume to continue it, or delete the checkpoint to restart.",
                file=sys.stderr,
            )
            record_failure(
                outdir,
                stage="checkpoint",
                exit_code=EXIT_CONFIG,
                error_class="CheckpointConflict",
                message=f"{ck_path} exists — a previous run was interrupted.",
                advice=(
                    "rerun with --resume to continue it, or",
                    f"delete {ck_path} to restart this run from scratch",
                ),
            )
            return EXIT_CONFIG
        try:
            with np.load(ck_path, allow_pickle=False) as npz:
                offset_periods = int(json.loads(str(npz["meta_json"]))["periods_done"])
        except Exception:
            offset_periods = 0
    elif opts.resume:
        # Loud on purpose: --resume against the wrong folder (CWD drift,
        # renamed output) would otherwise silently redo hours of solving.
        print(
            f"NOTE: --resume requested but no checkpoint at {ck_path} — starting fresh. "
            f"If a previous run WAS interrupted, check that the output folder matches."
        )

    stamp_base = {
        "job": built.name,
        "solver": built.solver,
        "backend": backend_name,
        "pid": os.getpid(),
        "ppw_warnings": ppw_warns,  # D31: visible in every status heartbeat
    }
    hb = _Heartbeat(
        status_path,
        base=stamp_base,
        spp=est.spp if est else 1,
        steps_expected=est.steps_expected if est else None,
        steps_worst=est.steps_worst if est else None,
        interval_s=opts.status_interval_s,
        offset_periods=offset_periods,
    )

    # `is not None`, not truthiness: --max-hours 0 means "checkpoint and stop
    # at the first period boundary", which is a legitimate drain request.
    deadline = time.monotonic() + opts.max_hours * 3600.0 if opts.max_hours is not None else None

    cancelled = False

    def stop_when() -> bool:
        # The heartbeat is no longer ticked HERE (M10j): it consumes the
        # engine's progress payload, which the boundary emits just before
        # this poll — same call site, same order, same counters, one
        # instrumentation instead of two.
        nonlocal cancelled
        if opts.stop_after_periods is not None and hb.session_periods >= opts.stop_after_periods:
            return True
        # The cancel poll (M10l) is ONE stat, at the period boundary, which
        # is the only place this hook is called from — a per-step poll would
        # put a filesystem round-trip between GPU kernels and is exactly what
        # the period-boundary discipline exists to prevent. It goes through
        # `_cancel_requested` and nowhere else, so the cost claim can be
        # counted rather than believed (see that helper).
        if _cancel_requested(cancel_path):
            cancelled = True
            return True
        return deadline is not None and time.monotonic() >= deadline

    hb.write("solving")
    run_kwargs: dict = dict(
        record_region=built.record_region,
        reference_point=built.focus_vox,
        harmonics=built.harmonics,
    )
    display = None
    timing = _StepTiming()
    if native:
        # backend=, checkpoint= and progress= are NATIVE-engine options; the
        # kwave adapter rejects unknown kwargs by contract, so passing them
        # would crash every kwave job (adversarial review, 2026-08-19; the
        # same trap catches progress= — T3).
        run_kwargs["backend"] = backend_name
        # keep_on_success: the checkpoint outlives the solve until the result
        # is SAFELY stored — a Drive failure during save stays resumable from
        # the pre-record snapshot instead of discarding the whole solve.
        run_kwargs["checkpoint"] = CheckpointSpec(
            path=ck_path,
            every_periods=opts.checkpoint_every,
            stop_when=stop_when,
            keep_on_success=True,
        )
        display = progress_resolve(opts.progress, label=built.name)
        # The timer is NOT a display: it is attached whether or not progress
        # output is enabled, because the warmup split is part of the stamp.
        run_kwargs["progress"] = progress_chain(hb, timing, display)

    import caustica.solvers as solvers  # noqa: PLC0415

    print(f"\nsolving ({built.solver}, {backend_name}) -> {outdir}", flush=True)
    t_solve = time.perf_counter()
    try:
        result = solvers.get(built.solver)().run(
            built.grid, built.medium, built.source, built.spec, **run_kwargs
        )
    except RunInterrupted as exc:
        hb.write("interrupted", detail=str(exc))
        if cancelled:
            # Consume the request so a stopped folder does not advertise a
            # stop nobody will honor: once this process exits, `cancel` is
            # gone and a GUI polling the folder sees a settled state.
            #
            # This is BELT-AND-BRACES, not the load-bearing part, though the
            # comment here used to claim otherwise ("leaving it would cancel
            # the --resume too, forever"). That is false: the clear at the
            # top of every real run above is what carries the resume, and
            # with this line deleted the resume still completes bit-identical
            # (measured, mutation review 2026-08-22). Both halves are pinned
            # in tests/test_runner.py.
            _clear_stale(cancel_path)
            print(f"\nCANCELLED on request ({CANCEL_FILE} file): {exc}")
        else:
            print(f"\nINTERRUPTED (resumable): {exc}")
        print(f"rerun with --resume to continue; state: {ck_path}")
        # No error.json: an interruption is not a failure. status.json says
        # "interrupted" and the exit code is 5 — that IS the contract.
        return EXIT_INTERRUPTED
    except Exception as exc:
        hb.write("failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        _write_error_json(
            outdir,
            stage="solve",
            exit_code=EXIT_SOLVER,
            error_class=type(exc).__name__,
            message=f"{type(exc).__name__}: {exc}",
            advice=(
                (f"a checkpoint survives at {ck_path}: rerun with --resume",)
                if ck_path.exists()
                else ()
            ),
        )
        return EXIT_SOLVER
    finally:
        progress_close(display)  # a live tqdm bar must not outlive the solve
    elapsed_solve = time.perf_counter() - t_solve

    # ---- store (M10 contract) + the stamp ----
    # A store failure here must NOT lose the solve: the checkpoint is still
    # on disk (keep_on_success), so we report a classified failure and the
    # user resumes — redoing only the short record window, not the hours.
    try:
        hb.write("writing")
        git_commit = _git_commit()
        apex_vox = tuple(int(v) for v in built.derived.get("apex_vox", (0, 0, 0)))
        # Re-verify at WRITE time: minutes of solving separate this moment
        # from the startup probe, and a Drive FUSE mount can lose the
        # directory in between (v12.3 lesson).
        ensure_dir_verified(outdir)
        if opts.preview_only:
            # D34 opt-in: the field is deliberately discarded — the preview
            # package IS the output here, so its failure is a real failure
            # (the post-store preview below is best-effort only because
            # result.h5 is already safe). No result.h5 also means no
            # skip-guard: a rerun of this folder solves again.
            _write_preview_package(built, result, outdir, apex_vox)
        else:
            save_result(
                result_path,
                result,
                built.source,
                dx=built.grid.dx,
                grid_shape=built.grid.shape,
                pml_vox=built.grid.pml_vox,
                quantize=built.output.quantize,
                max_norm_err=built.output.max_norm_err,
                extra_attrs={
                    "job_name": built.name,
                    "job_kind": built.job.kind,
                    "git_commit": git_commit,
                    "runner": f"caustica {caustica.__version__}",
                    # Geometry stamp so `caustica report` can place the field
                    # in mm-from-apex without the job/medium (M10d).
                    "apex_vox": list(apex_vox),
                    "focus_vox": [int(v) for v in built.focus_vox],
                },
            )
    except Exception as exc:
        hb.write("failed", error=f"store: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        if native:
            print(
                f"the finished solve is NOT lost: checkpoint retained at {ck_path}. "
                f"Fix the storage problem and rerun with --resume — only the record "
                f"window is redone.",
                file=sys.stderr,
            )
        _write_error_json(
            outdir,
            stage="store",
            exit_code=EXIT_SOLVER,
            error_class=type(exc).__name__,
            message=f"store: {type(exc).__name__}: {exc}",
            advice=(
                (
                    f"the solve is NOT lost: fix the storage problem and rerun with "
                    f"--resume (checkpoint retained at {ck_path}); only the record "
                    f"window is redone",
                )
                if native and ck_path.exists()
                else ()
            ),
        )
        return EXIT_SOLVER

    # ---- preview package (M10d): <=10 MB answer to "did the run work?" ----
    # The result is already safe on disk; a preview failure must never turn
    # a successful run into a failed one — warn and move on. (In
    # --preview-only mode the package was already written above, fatally.)
    if not opts.preview_only:
        try:
            _write_preview_package(built, result, outdir, apex_vox)
        except Exception as exc:
            log.warning("preview package failed (the result itself is safe): %s", exc)

    # This session's steps: a --resume run's elapsed_solve covers only what
    # THIS process did, so charging it with the resumed run's total steps
    # would report a negative warmup.
    prior_steps = offset_periods * est.spp if (est is not None and offset_periods) else 0
    session_steps = max(result.steps_total - prior_steps, 1)
    actual = {
        "elapsed_solve_s": round(elapsed_solve, 2),
        "elapsed_total_s": round(time.perf_counter() - t_start, 2),
        "steps_total": result.steps_total,
        # Kept verbatim: M8's Colab gates and the GUI contract read it. It
        # bundles the one-time warmup into a per-step average, which is what
        # the three keys below take apart (fix A2).
        "t_step_measured_s": round(elapsed_solve / max(result.steps_total, 1), 6),
        **timing.split(elapsed_solve, session_steps),
        "converged_period": result.converged_period,
        "settle_capped": result.settle_capped,
        "resumed_from_period": offset_periods or None,
        "vram_pool_peak_gib": _vram_pool_peak_gib(backend_name),
        "preview_only": opts.preview_only,
    }
    meta = {
        "format": "caustica-run-meta/1",
        "job": built.name,
        "job_kind": built.job.kind,
        "solver": built.solver,
        "backend": backend_name,
        "generated": _now_iso(),
        "git_commit": git_commit,
        # env_report keeps the historical key names (M8's Colab gates read
        # them) and only ADDS facts — see caustica.env.
        "environment": env_report(backend_name),
        "ppw_warnings": ppw_warns,  # D31: the report head re-reads these
        "planner": plan_payload,  # None for non-native solvers
        "actual": actual,  # planner-vs-actual: M8's Colab gates read this
        "derived": built.derived,  # re-derivable geometry (check_derived contract)
    }
    try:
        _write_json(outdir / "run_meta.json", meta)
    except OSError as exc:
        # The result itself is safe; a lost stamp is a warning, not a failure.
        log.warning("run_meta.json write failed: %s", exc)
    if native:
        ck_path.unlink(missing_ok=True)  # only now is it safe to forget the run
    final_artifact = outdir / "preview.npz" if opts.preview_only else result_path
    hb.write("done", result=str(final_artifact))

    pk = float(np.abs(result.phasor).max())
    print(
        f"done in {elapsed_solve:.1f} s — {result.steps_total:,} steps, "
        f"converged at period {result.converged_period}"
        f"{' (SETTLE CAP HIT)' if result.settle_capped else ''}; "
        f"peak |P| = {pk / 1e6:.3f} MPa"
    )
    if opts.preview_only:
        print(f"preview only (no result.h5, --preview-only): {final_artifact}")
    else:
        print(f"result: {final_artifact}")
    return EXIT_OK
