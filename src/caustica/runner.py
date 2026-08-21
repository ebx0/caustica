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

Exit codes are DISJOINT so a queue can react without parsing text:
0 success (or already complete) · 2 config error · 3 OOM refusal ·
4 solver error · 5 interrupted-but-resumable (``--max-hours`` stop).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import caustica
from caustica.config.job import BuiltJob, build_job, dump_job, load_job
from caustica.core.backend import CausticaWarning, get_backend
from caustica.io.atomic import atomic_write
from caustica.io.checkpoint import CheckpointSpec, RunInterrupted
from caustica.io.store import (
    ensure_dir_verified,
    probe_writable,
    save_result,
    validate_result_file,
)

log = logging.getLogger("caustica")

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_OOM = 3
EXIT_SOLVER = 4
EXIT_INTERRUPTED = 5

#: Solvers the planner models (the native k-space engine); anything else
#: (e.g. the external k-Wave binary) runs without a plan or a checkpoint.
_NATIVE_SOLVERS = ("linear", "westervelt")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict) -> None:
    with atomic_write(path) as tmp:
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _git_commit() -> str:
    """Best-effort commit hash of the caustica checkout ('unknown' when not a repo)."""
    try:
        root = Path(caustica.__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # pragma: no cover - git missing
        return "unknown"


def _gpu_environment(backend_name: str) -> dict:
    """Device/driver/cupy stamp when running on CUDA; empty otherwise."""
    if backend_name != "cupy":
        return {}
    try:  # pragma: no cover - requires a GPU
        import cupy

        dev = cupy.cuda.Device()
        props = cupy.cuda.runtime.getDeviceProperties(dev.id)
        free_b, total_b = dev.mem_info
        return {
            "gpu_name": props["name"].decode()
            if isinstance(props["name"], bytes)
            else str(props["name"]),
            "driver_version": cupy.cuda.runtime.driverGetVersion(),
            "cuda_runtime_version": cupy.cuda.runtime.runtimeGetVersion(),
            "cupy_version": cupy.__version__,
            "vram_total_gib": round(total_b / 2**30, 3),
            "vram_free_gib": round(free_b / 2**30, 3),
        }
    except Exception as exc:
        return {"gpu_probe_error": f"{type(exc).__name__}: {exc}"}


def _vram_pool_peak_gib(backend_name: str) -> float | None:
    if backend_name != "cupy":
        return None
    try:  # pragma: no cover - requires a GPU
        import cupy

        return round(cupy.get_default_memory_pool().total_bytes() / 2**30, 3)
    except Exception:
        return None


class _Heartbeat:
    """status.json writer: one tick per acoustic period, throttled to disk.

    The engine polls ``stop_when`` once per period boundary (M10); the
    heartbeat counts those polls, so step counts are DERIVED (periods * spp),
    not instrumented — no engine changes, no per-step overhead. ETA uses the
    measured cadence of THIS run, not the planner's guess.

    Known (accepted) imprecision: the engine polls once more just before the
    record window, so a run that reaches recording reports periods_done one
    higher than the settle count. Status numbers are progress telemetry, not
    provenance — the checkpoint meta and run_meta carry the exact counters.
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

    def tick(self) -> None:
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
            **common,
        )
        if opts.measure
        else est_gpu
    )
    payload = {
        "source": est_here.source,
        "spp": est_here.spp,
        "dt_s": est_here.dt,
        "t_step_s": est_here.t_step_s,
        "steps_expected": est_here.steps_expected,
        "steps_worst": est_here.steps_worst,
        "t_expected_s": round(est_here.t_expected_s, 1),
        "t_worst_s": round(est_here.t_worst_s, 1),
        "vram_gib": round(est_here.vram_gib, 3),
        "vram_breakdown_bytes": dict(est_here.vram_breakdown),
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
            est_gpu.summary(),
        ]
    )
    return est_here, payload, text


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

    # ---- everything before the solve is, by definition, a config problem ----
    try:
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
        return EXIT_CONFIG

    result_path = outdir / "result.h5"
    ck_path = outdir / "checkpoint.npz"
    status_path = outdir / "status.json"

    # ---- file-level resume: a complete result is never produced twice ----
    if result_path.exists() and validate_result_file(result_path):
        print(f"already complete: {result_path} (skip-guard; delete it to regenerate)")
        return EXIT_OK

    # ---- plan FIRST; refuse before paying, not after ----
    # Still config territory: an unknown --gpu name or a Drive hiccup on the
    # plan writes must exit 2, not leak a raw traceback with exit 1.
    native = built.solver in _NATIVE_SOLVERS
    est = plan_payload = None
    try:
        dump_job(job, outdir / "job.json")
        if native:
            est, plan_payload, plan_text = _plan(built, backend_name, opts)
            print(plan_text)
            _write_json(outdir / "plan.json", plan_payload)
            with atomic_write(outdir / "plan.txt") as tmp:
                tmp.write_text(plan_text, encoding="utf-8")
    except Exception as exc:
        print(f"CONFIG ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    if native:
        limit_gib = opts.vram_limit_gib
        limit_label = "requested limit"
        if limit_gib is None and backend_name == "cupy":
            env = _gpu_environment(backend_name)
            limit_gib = env.get("vram_total_gib")
            limit_label = f"device VRAM ({env.get('gpu_name', 'unknown GPU')})"
        if limit_gib is not None and est.vram_gib > limit_gib:
            print(
                f"REFUSED before solving: this run needs {est.vram_gib:.2f} GiB but "
                f"{limit_label} is {limit_gib:.2f} GiB.",
                file=sys.stderr,
            )
            for a in est.advice or (
                "coarsen dx, shrink the record region, or switch to the linear solver",
            ):
                print(f"  -> {a}", file=sys.stderr)
            return EXIT_OOM

        # ---- CPU gate (M10i/D20): refuse an hours-long numpy run BEFORE
        # paying for it; the message names its own escapes. Reuses
        # EXIT_CONFIG — the exit-code set is the queue's API (no sixth code).
        if backend_name == "numpy":
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
                    print(
                        f"REFUSED before solving: estimated wall time on the numpy (CPU) "
                        f"backend is ~{t_cpu:.0f} s (~{t_cpu / 3600:.1f} h, estimate source: "
                        f"{cpu_src}), over the {limit_min:g} min CPU limit.",
                        file=sys.stderr,
                    )
                    print(
                        "  -> run on a GPU backend (--backend cupy / backend='cupy'), or",
                        file=sys.stderr,
                    )
                    print(
                        "  -> accept the wait: --allow-slow-cpu (CLI) / allow_slow_cpu=True; "
                        "the threshold itself is CAUSTICA_CPU_LIMIT_MIN (minutes).",
                        file=sys.stderr,
                    )
                    return EXIT_CONFIG
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
    else:
        print(f"(planner models the native engine only; '{built.solver}' runs unplanned)")

    if opts.dry_run:
        print("\n(dry run — nothing solved, nothing written beyond the plan)")
        return EXIT_OK

    # ---- checkpoint policy: resuming is EXPLICIT ----
    offset_periods = 0
    if ck_path.exists():
        if not opts.resume:
            print(
                f"CONFIG ERROR: {ck_path} exists — a previous run was interrupted. "
                f"Rerun with --resume to continue it, or delete the checkpoint to restart.",
                file=sys.stderr,
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

    def stop_when() -> bool:
        hb.tick()
        if opts.stop_after_periods is not None and hb.session_periods >= opts.stop_after_periods:
            return True
        return deadline is not None and time.monotonic() >= deadline

    hb.write("solving")
    run_kwargs: dict = dict(
        record_region=built.record_region,
        reference_point=built.focus_vox,
        harmonics=built.harmonics,
    )
    if native:
        # backend= and checkpoint= are NATIVE-engine options; the kwave
        # adapter rejects unknown kwargs by contract, so passing them would
        # crash every kwave job (adversarial review, 2026-08-19).
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

    import caustica.solvers as solvers  # noqa: PLC0415

    print(f"\nsolving ({built.solver}, {backend_name}) -> {outdir}", flush=True)
    t_solve = time.perf_counter()
    try:
        result = solvers.get(built.solver)().run(
            built.grid, built.medium, built.source, built.spec, **run_kwargs
        )
    except RunInterrupted as exc:
        hb.write("interrupted", detail=str(exc))
        print(f"\nINTERRUPTED (resumable): {exc}")
        print(f"rerun with --resume to continue; state: {ck_path}")
        return EXIT_INTERRUPTED
    except Exception as exc:
        hb.write("failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return EXIT_SOLVER
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
                # Geometry stamp so `caustica report` can place the field in
                # mm-from-apex without the job/medium (M10d).
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
        return EXIT_SOLVER

    # ---- preview package (M10d): <=10 MB answer to "did the run work?" ----
    # The result is already safe on disk; a preview failure must never turn
    # a successful run into a failed one — warn and move on.
    try:
        _write_preview_package(built, result, outdir, apex_vox)
    except Exception as exc:
        log.warning("preview package failed (the result itself is safe): %s", exc)

    actual = {
        "elapsed_solve_s": round(elapsed_solve, 2),
        "elapsed_total_s": round(time.perf_counter() - t_start, 2),
        "steps_total": result.steps_total,
        "t_step_measured_s": round(elapsed_solve / max(result.steps_total, 1), 6),
        "converged_period": result.converged_period,
        "settle_capped": result.settle_capped,
        "resumed_from_period": offset_periods or None,
        "vram_pool_peak_gib": _vram_pool_peak_gib(backend_name),
    }
    meta = {
        "format": "caustica-run-meta/1",
        "job": built.name,
        "job_kind": built.job.kind,
        "solver": built.solver,
        "backend": backend_name,
        "generated": _now_iso(),
        "git_commit": git_commit,
        "environment": {
            "caustica": caustica.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            **_gpu_environment(backend_name),
        },
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
    hb.write("done", result=str(result_path))

    pk = float(np.abs(result.phasor).max())
    print(
        f"done in {elapsed_solve:.1f} s — {result.steps_total:,} steps, "
        f"converged at period {result.converged_period}"
        f"{' (SETTLE CAP HIT)' if result.settle_capped else ''}; "
        f"peak |P| = {pk / 1e6:.3f} MPa"
    )
    print(f"result: {result_path}")
    return EXIT_OK
