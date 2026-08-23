"""``caustica.simulate`` — one call for a notebook, the same machinery (M10j / W3).

This module adds NO physics and NO second way to build a run. Every accepted
input is normalised into a ``caustica-job/1`` config and handed to
:func:`caustica.config.job.build_job`; the plan, the two pre-run gates, the
metric definitions and the preview package are the ones the CLI, the queue
and a future GUI use. That is the whole point: the facade must describe the
same world as ``caustica run``, or a notebook result and a queue result stop
being comparable.

Accepted ``setup=`` forms — a short, closed list:

1. **a path** to a ``caustica-job/1`` JSON file (``str`` or ``os.PathLike``);
2. **a dict** in the job format;
3. an **:class:`~caustica.config.job.ExplicitJobConfig`** instance;
4. a **:class:`~caustica.config.job.BuiltJob`** — what ``build_job`` returns,
   i.e. the already-constructed grid / medium / source with the job that
   produced them still attached.

Anything else (a bare ``Grid``, ``Medium`` or ``CWSource``) raises
:class:`TypeError` naming these four. Those objects cannot be turned back
into a job — a voxelised source is not an array description — so accepting
them would mean a second construction path, which is exactly what this
module exists to avoid. Drive them through the object API instead::

    caustica.solvers.get("westervelt")().run(grid, medium, source, spec)

Two output modes, one preflight:

* ``out=None`` — nothing is written anywhere. The planner still speaks and
  both M10i gates (VRAM, the 5-minute CPU limit) still apply: an in-memory
  run that skipped them would be the "fine on my laptop, dies on Colab"
  failure the plan-first discipline exists to prevent.
* ``out=<path>`` — delegated verbatim to :func:`caustica.runner.run_job_file`,
  which owns the output folder, the stamp and resume. No copy of that logic
  lives here, and a job read from a file keeps resolving its relative paths
  against the job file (trap T4).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from caustica.config.job import (
    BuiltJob,
    ExplicitJobConfig,
    build_job,
    dump_job,
    load_job,
    parse_job,
)
from caustica.core.backend import CausticaWarning, check_backend_name, get_backend
from caustica.env import gpu_environment
from caustica.progress import close as progress_close
from caustica.progress import resolve as progress_resolve
from caustica.runner import (
    _NATIVE_SOLVERS,
    EXIT_CONFIG,
    EXIT_OK,
    EXIT_SOLVER,
    RunnerOptions,
    _linear_solve_warnings_for,
    _now_iso,
    _plan,
    _ppw_warnings_for,
    check_gates,
    run_job_file,
)
from caustica.solvers.base import SolverResult

__all__ = ["SimulationError", "SimulationRun", "simulate"]

#: Quoted verbatim by the TypeError, so the message teaches the fix.
SETUP_FORMS = (
    "a path to a caustica-job/1 JSON file",
    "a dict in the job format",
    "an ExplicitJobConfig",
    "a BuiltJob (what caustica.config.job.build_job returns)",
)


#: Sentinel for "the caller did not mention this argument", so a value set
#: only through ``options=`` is not overwritten by the default.
_KEEP = object()

#: RunnerOptions that need an output folder to mean anything. Refused rather
#: than ignored: a notebook that asks for a 2-hour budget and silently gets
#: none is the failure mode PLAN section 8 opens with.
_NEEDS_A_FOLDER = ("resume", "max_hours", "stop_after_periods", "preview_only", "dry_run")


def _refuse_options_without_a_folder(opts: RunnerOptions) -> None:
    default = RunnerOptions()
    named = [n for n in _NEEDS_A_FOLDER if getattr(opts, n) != getattr(default, n)]
    if named:
        raise ValueError(
            f"simulate(out=None) runs in memory, so {named} cannot do anything: "
            f"resuming, a checkpointed time budget, a deterministic stop, a "
            f"preview-only artifact and a plan-only run all need an output "
            f"folder. Pass out=<path>, or drop these options."
        )


class SimulationError(RuntimeError):
    """A run that produced no result, carrying the runner's exit code.

    ``exit_code`` is from the SAME disjoint set the CLI and the queue use
    (2 config, 3 OOM, 4 solver, 5 interrupted-resumable), so a notebook, a
    shell script and a GUI classify the same failure the same way.
    """

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass
class SimulationRun:
    """What :func:`simulate` returns: a thin wrapper, never a reimplementation.

    ``result`` is the solver's own :class:`~caustica.solvers.base.SolverResult`
    (loaded from ``result.h5`` when a run wrote one). ``metrics`` are the
    definitions in :mod:`caustica.report.metrics` — the ones ``focus_study``
    and every REPORT.md quote. ``preview()`` is the <=10 MB package the runner
    writes next to a result, built in memory here.
    """

    job: ExplicitJobConfig
    outdir: Path | None
    exit_code: int
    plan: dict | None
    warnings: tuple[str, ...] = ()
    _geometry: dict | None = None
    _base_dir: Path | None = None
    _result: SolverResult | None = None
    _metrics: dict | None = None
    _medium: Any = field(default=None, repr=False)

    @property
    def geometry(self) -> dict:
        """dx / shape / PML / apex / focus / source for this run.

        Rebuilt from the job on demand for a delegated run, and only if
        something actually needs it: a written folder already carries
        ``metrics.json`` and ``preview.npz``, so the common path never pays
        for a second geometry build.
        """
        if self._geometry is None:
            self._geometry = _geometry_of(
                build_job(self.job, base_dir=self._base_dir, with_medium=False)
            )
        return self._geometry

    # -- the field ------------------------------------------------------
    @property
    def result(self) -> SolverResult:
        """The solve. Read back from ``result.h5`` for a run that wrote one."""
        if self._result is None:
            from caustica.io.store import load_result  # noqa: PLC0415

            path = None if self.outdir is None else self.outdir / "result.h5"
            if path is None or not path.exists():
                raise SimulationError(
                    "this run has no field to hand back: it was written with "
                    "preview_only=True, so result.h5 was deliberately skipped. "
                    "Use .preview() / .metrics, or rerun without preview_only.",
                    EXIT_CONFIG,
                )
            self._result = load_result(path)
        return self._result

    @property
    def phasor(self) -> np.ndarray:
        return self.result.phasor

    @property
    def p_max(self) -> np.ndarray:
        return self.result.p_max

    # -- the numbers ----------------------------------------------------
    @property
    def metrics(self) -> dict:
        """Focal metrics — :func:`caustica.report.metrics.focus_metrics`.

        Read from ``metrics.json`` when the runner already wrote it, so a
        folder and the object in your hand can never quote different numbers.
        """
        if self._metrics is None:
            path = None if self.outdir is None else self.outdir / "metrics.json"
            if path is not None and path.exists():
                self._metrics = json.loads(path.read_text(encoding="utf-8"))
            else:
                self._metrics = self._compute_metrics(self.result)
        return self._metrics

    def _compute_metrics(self, result: SolverResult) -> dict:
        from caustica.report.metrics import FieldFrame, focus_metrics  # noqa: PLC0415

        g = self.geometry
        return {
            "format": "caustica-metrics/1",
            "job": self.job.name,
            "generated": _now_iso(),
            **focus_metrics(
                result,
                FieldFrame.from_geometry(g),
                source_amplitude=g["source_amplitude"],
                medium=self._medium,
                solver=self.job.solver,
            ),
        }

    # -- the <=10 MB package --------------------------------------------
    def preview(self) -> dict:
        """The ``caustica-preview/1`` package, in memory (arrays + ``meta``).

        Identical in content to the ``preview.npz`` a written run produces —
        and read back from that file when there is one.
        """
        from caustica.report.metrics import FieldFrame  # noqa: PLC0415
        from caustica.report.preview import (  # noqa: PLC0415
            build_preview,
            decode_preview,
            load_preview,
        )

        path = None if self.outdir is None else self.outdir / "preview.npz"
        if path is not None and path.exists():
            return load_preview(path)
        return decode_preview(build_preview(self.result, FieldFrame.from_geometry(self.geometry)))

    # -- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write the field as ``caustica-result/1``.

        A run that already produced a ``result.h5`` is COPIED, not re-encoded:
        re-quantizing an already-quantized field would add a second lossy
        pass for nothing.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = None if self.outdir is None else self.outdir / "result.h5"
        if existing is not None and existing.exists():
            if existing.resolve() == path.resolve():
                return path
            import shutil  # noqa: PLC0415

            shutil.copyfile(existing, path)
            return path

        from caustica.io.store import save_result  # noqa: PLC0415

        g = self.geometry
        return save_result(
            path,
            self.result,
            g["source"],
            dx=g["dx"],
            grid_shape=g["grid_shape"],
            pml_vox=g["pml_vox"],
            quantize=self.job.output.quantize,
            max_norm_err=self.job.output.max_norm_err,
            extra_attrs={
                "job_name": self.job.name,
                "job_kind": self.job.kind,
                "runner": "caustica.simulate",
                "apex_vox": list(g["apex_vox"]),
                "focus_vox": [int(v) for v in g["focus_vox"]],
            },
        )


# --------------------------------------------------------------- normalise


def _as_job(setup: Any) -> tuple[ExplicitJobConfig, Path | None, Path | None, BuiltJob | None]:
    """``(job, base_dir, job_path, built)`` for every accepted setup form."""
    if isinstance(setup, BuiltJob):
        # Already through build_job — the caller did that half themselves.
        # `setup.job` is the ORIGINAL config, relative paths and all (the
        # build resolves into local copies), so the base directory has to
        # come off the BuiltJob or T4 breaks: a re-dump would resolve the
        # medium file against a temp directory, or a rebuild against the CWD
        # (adversarial review, 2026-08-22 — reproduced both).
        return setup.job, setup.base_dir, None, setup
    if isinstance(setup, ExplicitJobConfig):
        return setup, Path.cwd(), None, None
    if isinstance(setup, Mapping):
        return parse_job(dict(setup)), Path.cwd(), None, None
    if isinstance(setup, (str, os.PathLike)):
        job, base_dir = load_job(setup)
        return job, base_dir, Path(setup), None
    raise TypeError(
        f"simulate(setup=...) does not accept {type(setup).__name__}. Pass one of: "
        + "; ".join(SETUP_FORMS)
        + ". Already-built Grid/Medium/CWSource objects cannot be described as a "
        "job (a voxelised source is not an array description) — run those through "
        "the object API: caustica.solvers.get(name)().run(grid, medium, source, spec)."
    )


def _override(job: ExplicitJobConfig, solver, harmonics) -> ExplicitJobConfig:
    """Apply the facade's job-level overrides, RE-VALIDATED.

    Round-tripping through the parser rather than ``model_copy`` is
    deliberate: ``harmonics=(2, 3)`` must fail with the model's own message,
    not solve something nobody asked for.
    """
    if solver is None and harmonics is None:
        return job  # untouched: the bytes that ran are the bytes you passed
    data = json.loads(job.model_dump_json())
    if solver is not None:
        data["solver"] = solver
    if harmonics is not None:
        data["run"]["harmonics"] = [int(h) for h in harmonics]
    return parse_job(data, "simulate(...) overrides")


def _geometry_of(built: BuiltJob) -> dict:
    return {
        "dx": built.grid.dx,
        "grid_shape": built.grid.shape,
        "pml_vox": built.grid.pml_vox,
        "apex_vox": tuple(int(v) for v in built.derived.get("apex_vox", (0, 0, 0))),
        "focus_vox": built.focus_vox,
        "source": built.source,
        "source_amplitude": built.source.amplitude,
    }


def _enable_logging() -> None:
    """Turn caustica's own log records on at the facade's door (PLAN section 8).

    The LIBRARY installs no handler on import (D33); ``simulate()`` is an
    entry point, like the CLI, and a notebook that never sees "falling back
    to numpy" is the silent-failure case the policy exists to prevent. Scoped
    to the caustica logger, and never added twice.
    """
    clog = logging.getLogger("caustica")
    if not clog.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        clog.addHandler(handler)
        clog.setLevel(logging.INFO)


# ------------------------------------------------------------------- entry


def simulate(
    setup: Any,
    *,
    solver: str | None = None,
    backend: str | None = None,
    harmonics: tuple[int, ...] | None = None,
    out: str | Path | None = None,
    progress: Any = _KEEP,
    allow_slow_cpu: bool = False,
    options: RunnerOptions | None = None,
) -> SimulationRun:
    """Run one simulation and hand back its result, metrics and preview.

    Parameters
    ----------
    setup:
        One of the four forms in the module docstring.
    solver, backend, harmonics:
        Overrides for the job's own fields (``None`` leaves each alone).
        ``backend`` is the runner's override, so ``"auto"`` still means
        "cupy if this machine has a usable one".
    out:
        ``None`` writes nothing at all; a path gets the full runner output
        folder (job copy, plan, status, result, preview, stamp).
    progress:
        ``"auto"`` (the default when neither this nor ``options.progress`` is
        given) prints a progress line per acoustic period and a coarse focal
        preview every 8 periods on stderr; ``"plain"`` never uses a tqdm bar;
        ``None`` is silent; a callable receives the raw payload.
    allow_slow_cpu:
        Accept a numpy run the planner expects to take longer than the
        5-minute CPU limit (``CAUSTICA_CPU_LIMIT_MIN``).
    options:
        Everything else ``caustica run`` can set (resume, max_hours,
        checkpoint cadence, preview_only, ...). The named arguments above win
        over the matching fields. Only the ``out=<path>`` mode uses the
        file-bound ones — an in-memory run has no folder to resume from.
    """
    _enable_logging()
    job, base_dir, job_path, built = _as_job(setup)
    job = _override(job, solver, harmonics)
    if built is not None and job is not built.job:
        built = None  # overrides invalidate a caller-built job; rebuild below

    opts = RunnerOptions(**vars(options)) if options is not None else RunnerOptions()
    # A named argument wins, but an option set ONLY through `options=` must
    # still be honoured -- silently dropping a requested output folder (or a
    # requested renderer) is the exact "looks fine, does something else"
    # failure the environment policy exists to prevent.
    if out is not None:
        opts.out = out
    if progress is not _KEEP:
        opts.progress = progress  # includes an explicit progress=None (silence)
    elif opts.progress is None:
        # D21/K11: the FACADE's default is progress on, while the library-level
        # RunnerOptions default stays silent. An `options=` that names a
        # renderer keeps it.
        opts.progress = "auto"
    opts.allow_slow_cpu = allow_slow_cpu or opts.allow_slow_cpu
    if backend is not None:
        opts.backend = backend

    if opts.out is not None:
        return _run_via_runner(job, job_path, base_dir, opts)
    _refuse_options_without_a_folder(opts)
    return _run_in_memory(job, base_dir, built, opts)


def _run_via_runner(
    job: ExplicitJobConfig,
    job_path: Path | None,
    base_dir: Path | None,
    opts: RunnerOptions,
) -> SimulationRun:
    """Delegate to ``run_job_file`` — output, stamp and resume live there."""
    outdir = Path(opts.out)  # type: ignore[arg-type]
    if job_path is not None:
        # The user's own file goes to the runner untouched, so T4 applies to
        # it exactly as it does for `caustica run`.
        code = run_job_file(job_path, opts)
        base_after = Path(job_path).parent
    else:
        # A job that was never a file has no file to resolve against, so its
        # relative paths are pinned to `base_dir` *before* it is handed over.
        # Afterwards every path in it is absolute, which is why the run object
        # gets base_dir=None: the temp directory is gone by then, and handing
        # out a path that no longer exists is how a lazily-resolved field
        # would later resolve against nothing (review, 2026-08-22).
        job = job.model_copy(
            update={
                "medium": job.medium.resolve_paths(base_dir),
                "source": job.source.resolve_paths(base_dir),
            }
        )
        with tempfile.TemporaryDirectory(prefix="caustica-job-") as tmp:
            code = run_job_file(dump_job(job, Path(tmp) / "job.json"), opts)
        base_after = None
    if code != EXIT_OK:
        raise SimulationError(
            f"caustica run exited {code} (0 ok, 2 config, 3 OOM, 4 solver, "
            f"5 interrupted-resumable); the diagnosis was printed above.",
            code,
        )
    plan_path = outdir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None
    meta_path = outdir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return SimulationRun(
        job=job,
        outdir=outdir,
        exit_code=code,
        plan=plan,
        warnings=tuple(meta.get("ppw_warnings", ())),
        _base_dir=base_after,
    )


def _run_in_memory(
    job: ExplicitJobConfig,
    base_dir: Path | None,
    built: BuiltJob | None,
    opts: RunnerOptions,
) -> SimulationRun:
    """Plan, gate, solve — writing nothing. Same helpers as ``run_job_file``."""
    import caustica.solvers as solvers  # noqa: PLC0415

    # Everything before the solve is, by definition, a config problem -- the
    # same classification `run_job_file` applies when it turns a preflight
    # exception into EXIT_CONFIG. Without this the two output modes would
    # report the SAME broken job differently: exit code 2 through a folder,
    # a bare JobError in memory (adversarial review, 2026-08-22). The cause
    # is chained, so the original message and type are still right there.
    try:
        # BEFORE the medium is built, for the reason runner.run_job_file gives
        # at its own copy of this line: a misspelled backend must not cost a
        # multi-GB medium build first.
        if opts.backend is not None:
            check_backend_name(opts.backend)
        if built is None:
            built = build_job(job, base_dir=base_dir, with_medium=True)
        backend_name = get_backend(opts.backend or built.backend).name
    except Exception as exc:
        raise SimulationError(f"{type(exc).__name__}: {exc}", EXIT_CONFIG) from exc
    native = built.solver in _NATIVE_SOLVERS
    ppw_warns = _ppw_warnings_for(built)
    # Snapshot GPU facts BEFORE the measure probe fills the cupy pool — the
    # same ordering hazard the runner documents.
    gpu_env = gpu_environment(backend_name) if backend_name == "cupy" else {}

    est = plan_payload = None
    if native:
        try:
            est, plan_payload, plan_text = _plan(built, backend_name, opts)
        except Exception as exc:
            raise SimulationError(f"{type(exc).__name__}: {exc}", EXIT_CONFIG) from exc
        # Unconditionally, exactly as the runner writes it into plan.json: a
        # consumer must be able to tell "no warnings" from "this plan does not
        # carry the field" (D31).
        plan_payload["ppw_warnings"] = ppw_warns
        if ppw_warns:
            plan_text += "\n" + "\n".join(f"  ! WARNING: {w}" for w in ppw_warns)
        print(plan_text)
    else:
        print(f"(planner models the native engine only; '{built.solver}' runs unplanned)")
    if ppw_warns:
        warnings.warn(
            "low spatial resolution: " + " | ".join(ppw_warns),
            CausticaWarning,
            stacklevel=2,
        )
    for warn_text in _linear_solve_warnings_for(built):
        warnings.warn(warn_text, CausticaWarning, stacklevel=2)
    if native:
        refusal = check_gates(built, est, backend_name, opts, gpu_env)
        if refusal is not None:
            print(refusal.message, file=sys.stderr)
            raise SimulationError(refusal.message, refusal.exit_code)

    run_kwargs: dict = dict(
        record_region=built.record_region,
        reference_point=built.focus_vox,
        harmonics=built.harmonics,
    )
    display = None
    if native:
        # T3, again: backend= and progress= are native-engine options and the
        # kwave adapter rejects unknown kwargs by contract. No checkpoint —
        # an in-memory run has nowhere to put one, which is exactly the case
        # trap T1 was about.
        run_kwargs["backend"] = backend_name
        display = progress_resolve(opts.progress, label=built.name)
        run_kwargs["progress"] = display
    print(f"solving ({built.solver}, {backend_name}) -> memory", flush=True)
    try:
        result = solvers.get(built.solver)().run(
            built.grid, built.medium, built.source, built.spec, **run_kwargs
        )
    except SimulationError:
        raise
    except Exception as exc:
        # Exit code 4, like the runner's classified solver failure.
        raise SimulationError(f"{type(exc).__name__}: {exc}", EXIT_SOLVER) from exc
    finally:
        progress_close(display)

    run = SimulationRun(
        job=job,
        outdir=None,
        exit_code=EXIT_OK,
        plan=plan_payload,
        warnings=tuple(ppw_warns),
        _geometry=_geometry_of(built),
        _result=result,
        _medium=built.medium,
    )
    # Metrics are computed NOW, while the medium is still alive, then the
    # reference is dropped: keeping a multi-GB property volume pinned to the
    # returned object for the sake of one intensity number is not a trade a
    # notebook should have to think about.
    run._metrics = run._compute_metrics(result)
    run._medium = None
    pk = float(np.abs(result.phasor).max())
    print(
        f"done — {result.steps_total:,} steps, converged at period "
        f"{result.converged_period}"
        f"{' (SETTLE CAP HIT)' if result.settle_capped else ''}; "
        f"peak |P| = {pk / 1e6:.3f} MPa (in memory; nothing written)"
    )
    return run
