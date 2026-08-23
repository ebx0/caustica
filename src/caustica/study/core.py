"""The Study object: a base job, its variants, their runs and their numbers.

Three rules shaped this module, and every design choice below follows from
one of them.

**1. A Study is orchestration, not a solver.** It never touches a grid, a
medium or a source. Every run leaves through :func:`caustica.simulate`, so a
study run and a ``caustica run`` of the same job are the same code path with
the same plan, the same two pre-run gates and the same metric definitions.
If a study could produce a number the CLI cannot, the two would stop being
comparable — which is the whole reason to have a study at all.

**2. A variant is a job, re-validated.** ``"drive.amplitude_kpa"`` addresses
a field of the ``caustica-job/1`` JSON, the override is applied to the dumped
mapping, and the result goes back through :func:`caustica.config.job.parse_job`.
So a sweep over an illegal value fails with the model's own message rather
than solving something nobody asked for — the same trick the facade plays
with ``solver=``/``harmonics=``. And it fails BEFORE anything runs: a sweep
builds all of its jobs first, so a typo costs zero solves, not two.

**3. A prediction that is never compared is a decoration.** Every run
records the planner's estimate AND what actually happened, in the same
object, so the report can print them next to each other. This is the same
``planner``/``actual`` pair the runner writes into ``run_meta.json``; the
study reads that file when there is one rather than re-deriving it.

What a *failed* run does is deliberately split: a **config** error (bad
override, illegal value) raises immediately and nothing runs, while a
**runtime** failure (OOM, a solver blowing up) is recorded on its run and the
sweep continues. Losing four good runs because the fifth ran out of memory
is the behaviour a long sweep can least afford, and the report says FAILED in
the row where it happened.
"""

from __future__ import annotations

import copy
import json
import platform
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from caustica.config.job import ExplicitJobConfig, parse_job
from caustica.core.backend import check_backend_name
from caustica.facade import SimulationError, SimulationRun, simulate
from caustica.runner import RunnerOptions, _now_iso

__all__ = [
    "FORMAT",
    "Study",
    "StudyError",
    "StudyRun",
    "StudySweep",
    "get_by_path",
    "set_by_path",
]

#: The payload contract ``study.json`` declares. Bumped, never redefined.
FORMAT = "caustica-study/1"

#: Planner keys copied into a run's ``expected``. A subset of ``plan.json``:
#: the ones a reader can compare against something that actually happened.
_EXPECTED_KEYS = (
    "source",
    "spp",
    "dt_s",
    "t_step_s",
    "warmup_s",
    "steps_expected",
    "steps_worst",
    "t_expected_s",
    "t_worst_s",
    "vram_gib",
    "result_size_mb_expected",
)


class StudyError(ValueError):
    """A study was asked for something it cannot describe.

    A ``ValueError`` like :class:`~caustica.config.job.JobError`, because
    that is what it is: the *configuration* is wrong, not the machine. A run
    that starts and then fails raises the facade's
    :class:`~caustica.facade.SimulationError` instead, carrying the runner's
    exit code.
    """


# --------------------------------------------------------- parameter paths


def _split_path(path: str) -> list[str]:
    if not isinstance(path, str) or not path.strip():
        raise StudyError(
            "a parameter path must be a non-empty string addressing a job field, "
            "e.g. 'drive.amplitude_kpa' or 'source.apex_mm.2'"
        )
    parts = path.split(".")
    if any(not p for p in parts):
        raise StudyError(f"parameter path {path!r} has an empty segment (a stray '.')")
    return parts


def _step(node: Any, seg: str, path: str, walked: Sequence[str]) -> Any:
    """One segment of a path, or a :class:`StudyError` that names the options.

    The error is the whole point of doing this by hand instead of with
    ``functools.reduce``: a mistyped field must print the fields that DO
    exist at that level, because the job schema is deep enough that guessing
    is hopeless.
    """
    here = ".".join(walked) or "<root>"
    if isinstance(node, Mapping):
        if seg not in node:
            raise StudyError(
                f"parameter path {path!r}: {seg!r} is not a field of {here}. "
                f"Available: {', '.join(sorted(map(str, node)))}"
            )
        return node[seg]
    if isinstance(node, list):
        try:
            idx = int(seg)
        except ValueError:
            raise StudyError(
                f"parameter path {path!r}: {here} is a list of {len(node)}, so "
                f"{seg!r} must be an integer index (0..{len(node) - 1})"
            ) from None
        if not -len(node) <= idx < len(node):
            raise StudyError(
                f"parameter path {path!r}: index {idx} is out of range for {here} "
                f"(length {len(node)})"
            )
        return node[idx]
    raise StudyError(
        f"parameter path {path!r}: {here} is a {type(node).__name__}, not a field "
        f"container — it has no {seg!r}"
    )


def get_by_path(data: Mapping[str, Any], path: str) -> Any:
    """Read the job-mapping field ``path`` addresses (``"drive.f0_mhz"``)."""
    node: Any = data
    parts = _split_path(path)
    for i, seg in enumerate(parts):
        node = _step(node, seg, path, parts[:i])
    return node


def set_by_path(data: Mapping[str, Any], path: str, value: Any) -> dict:
    """A DEEP COPY of ``data`` with ``path`` set to ``value``.

    Copying rather than mutating is what lets one base job feed a whole
    sweep: every variant is built from the same pristine mapping, so run 3
    cannot inherit run 2's override.
    """
    out = copy.deepcopy(dict(data))
    parts = _split_path(path)
    node: Any = out
    for i, seg in enumerate(parts[:-1]):
        node = _step(node, seg, path, parts[:i])
    last = parts[-1]
    _step(node, last, path, parts[:-1])  # existence check, with the good message
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value
    return out


# ---------------------------------------------------------------- plumbing


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", str(text)).strip("-").lower() or "x"


def _fmt_value(value: Any) -> str:
    """A swept value as it appears in a label: ``50``, not ``50.0``."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def job_hash(job: ExplicitJobConfig) -> str:
    """sha256 (16 hex) of the job's canonical JSON — its identity.

    Canonical means "what the model dumps", so the SAME job reaches the same
    hash whether it came from a file, a dict or a study override. Two rows of
    a sweep report sharing a hash would mean the sweep swept nothing.
    """
    import hashlib  # noqa: PLC0415 (only needed when a run is described)

    return hashlib.sha256(job.model_dump_json().encode("utf-8")).hexdigest()[:16]


def stamp(backend_name: str | None = None) -> dict:
    """Who ran this, where, with which caustica — never raises.

    Deliberately the SAME composition the runner puts in ``run_meta.json``:
    :func:`caustica.env.env_report` (versions, resolved backend, GPU name and
    VRAM) plus :func:`caustica.env.git_commit`. A study stamp and a run stamp
    that could disagree would make a report unciteable.
    """
    from caustica import __version__  # noqa: PLC0415 (cycle-safe at call time)
    from caustica.env import env_report, git_commit  # noqa: PLC0415

    return {
        "generated": _now_iso(),
        "caustica": __version__,
        "git_commit": git_commit(),
        "host": platform.node(),
        "environment": env_report(backend_name),
    }


def _expected_from_plan(plan: Mapping[str, Any] | None) -> dict:
    """The planner's promise, as the report will quote it back."""
    if not plan:
        return {}
    return {k: plan[k] for k in _EXPECTED_KEYS if k in plan}


def _actual_from(sim: SimulationRun, wall_s: float) -> dict:
    """What actually happened — from ``run_meta.json`` when there is one.

    The metrics dict is the fallback and the floor: it always carries
    ``steps_total`` / ``converged_period``, and reading them from there costs
    no h5py. ``run_meta.json`` adds the timings the runner measured, which an
    in-memory run has no way to produce — for that case the study's own wall
    clock stands in, and ``source`` says so rather than pretending.
    """
    m_run = (sim.metrics or {}).get("run", {})
    actual: dict = {
        "steps_total": m_run.get("steps_total"),
        "converged_period": m_run.get("converged_period"),
        "settle_capped": m_run.get("settle_capped"),
        "wall_s": round(wall_s, 2),
    }
    meta_path = None if sim.outdir is None else sim.outdir / "run_meta.json"
    if meta_path is not None and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
        recorded = meta.get("actual") or {}
        if recorded:
            actual.update(recorded)
            actual["wall_s"] = round(wall_s, 2)
            actual["source"] = "run_meta.json"
            return actual
    steps = actual["steps_total"] or 0
    actual["elapsed_solve_s"] = round(wall_s, 2)
    actual["t_step_measured_s"] = round(wall_s / steps, 6) if steps else None
    actual["source"] = "study wall clock (nothing was written)"
    return actual


# ------------------------------------------------------------------ result


@dataclass
class StudyRun:
    """One run of a study: which job, what was predicted, what happened.

    Everything a report needs is captured HERE, at run time, in JSON-ready
    form: the field itself stays behind ``simulation`` and is never needed to
    write a report. That is what lets a 40-run sweep report be written
    without reopening a single ``result.h5``.
    """

    study: str
    label: str
    index: int
    overrides: dict[str, Any]
    job: ExplicitJobConfig
    job_hash: str
    outdir: Path | None
    stamp: dict
    expected: dict = field(default_factory=dict)
    actual: dict = field(default_factory=dict)
    metrics: dict | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None
    exit_code: int = 0
    simulation: SimulationRun | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def peak_pa(self) -> float | None:
        """Peak fundamental pressure [Pa], or ``None`` for a failed run."""
        if not self.metrics:
            return None
        return self.metrics.get("peak", {}).get("p_pa")

    @property
    def result(self):
        """The solve — see :attr:`caustica.facade.SimulationRun.result`."""
        if self.simulation is None:
            raise StudyError(f"run {self.label!r} produced no result: {self.error}")
        return self.simulation.result

    def as_dict(self) -> dict:
        """The run's row in ``study.json``."""
        return {
            "index": self.index,
            "label": self.label,
            "overrides": dict(self.overrides),
            "job_name": self.job.name,
            "job_hash": self.job_hash,
            "solver": self.job.solver,
            "backend": self.job.backend,
            "outdir": None if self.outdir is None else str(self.outdir),
            "expected": self.expected,
            "actual": self.actual,
            "metrics": self.metrics,
            "warnings": list(self.warnings),
            "error": self.error,
            "exit_code": self.exit_code,
        }

    def figures(self) -> Path:
        """Render this run's full figure set with ``caustica report``.

        Delegation, not a second renderer: the study owns the comparison
        between runs, ``caustica.report`` owns what one run looks like.
        Needs a run that wrote a folder (and matplotlib).
        """
        if self.outdir is None:
            raise StudyError(
                f"run {self.label!r} wrote nothing, so there is no folder to render. "
                f"Give the Study an `out=` folder (or pass out= to run/sweep)."
            )
        from caustica.report.run_report import report_out_dir  # noqa: PLC0415 (matplotlib)

        return report_out_dir(self.outdir)

    def report(self, outdir: str | Path | None = None) -> Path:
        """Write ``STUDY.md`` + ``study.json`` for this run; returns the folder.

        Defaults to the run's own output folder. The names do not collide
        with ``caustica report``'s ``REPORT.md`` / ``index.html``, so a run
        folder can carry both.
        """
        from caustica.study.report import write_run_report  # noqa: PLC0415

        target = Path(outdir) if outdir is not None else self.outdir
        if target is None:
            raise StudyError(
                f"run {self.label!r} has no output folder, so report() needs one: "
                f"run.report(outdir)."
            )
        return write_run_report(self, target)


@dataclass
class StudySweep:
    """One parameter, several values, and the runs they produced.

    The combined report is the point. A sweep whose runs each got their own
    report and nothing else would leave the comparison — the only question a
    sweep is asked — to the reader's eye.
    """

    study: str
    param_path: str
    values: tuple[Any, ...]
    runs: list[StudyRun]
    root: Path | None
    stamp: dict
    common: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.runs)

    @property
    def failures(self) -> list[StudyRun]:
        return [r for r in self.runs if not r.ok]

    def peaks_mpa(self) -> list[float | None]:
        """Peak fundamental pressure per run [MPa]; ``None`` where it failed."""
        return [None if r.peak_pa is None else r.peak_pa / 1e6 for r in self.runs]

    def report(self, outdir: str | Path | None = None, *, figures: bool = True) -> Path:
        """Write the COMBINED ``STUDY.md`` + ``study.json`` (+ figure); returns the folder."""
        from caustica.study.report import write_sweep_report  # noqa: PLC0415

        target = Path(outdir) if outdir is not None else self.root
        if target is None:
            raise StudyError(
                "this sweep ran in memory, so report() needs somewhere to write: "
                "sweep.report(outdir), or give the Study an out= folder."
            )
        return write_sweep_report(self, target, figures=figures)


# ------------------------------------------------------------------- study


class Study:
    """A named base job plus the machinery to run variants of it.

    Parameters
    ----------
    name:
        Identifies the study in every report it writes.
    setup:
        The base job, in any of the four forms :func:`caustica.simulate`
        accepts (path / dict / ``ExplicitJobConfig`` / ``BuiltJob``). It is
        parsed and its relative paths are resolved AT CONSTRUCTION, so a
        broken setup fails here rather than three runs into a sweep, and a
        job loaded from a file keeps resolving its medium against the file's
        own directory (trap T4) even after an override rebuilds it.
    out:
        Root output folder. ``None`` runs in memory: fast and test-friendly,
        but there is then no folder for :meth:`StudyRun.report` to default to.
    solver, backend, harmonics:
        Study-wide overrides of the job's own fields, re-validated exactly
        like the facade's.
    progress, allow_slow_cpu, options:
        Passed to :func:`caustica.simulate` for every run. ``progress``
        defaults to ``None`` (silent), the LIBRARY default rather than the
        facade's ``"auto"`` — a sweep that prints a progress bar per run is
        the one thing a notebook cell cannot scroll past.

    A ``BuiltJob`` handed in as ``setup`` is used for its job description
    only; the medium is rebuilt per run, because a variant may change the
    grid it was built on. One code path, at the price of one build.
    """

    def __init__(
        self,
        name: str,
        setup: Any,
        *,
        out: str | Path | None = None,
        solver: str | None = None,
        backend: str | None = None,
        harmonics: Sequence[int] | None = None,
        progress: Any = None,
        allow_slow_cpu: bool = False,
        options: RunnerOptions | None = None,
    ) -> None:
        from caustica.facade import _as_job, _override  # noqa: PLC0415 (same-package seam)

        if not isinstance(name, str) or not name.strip():
            raise StudyError("a Study needs a non-empty name — it labels every report it writes")
        if backend is not None:
            # Before anything else, for the reason the runner refuses a
            # misspelled backend first: a typo must not cost a medium build.
            check_backend_name(backend)
        job, base_dir, _job_path, _built = _as_job(setup)
        job = _override(job, solver, None if harmonics is None else tuple(harmonics))
        # Pin every relative path to the setup's own directory NOW. After an
        # override the job is rebuilt from a mapping with no file behind it,
        # and a lazily-resolved medium path would then resolve against the
        # CWD — the exact failure the facade documents at its own copy.
        job = job.model_copy(
            update={
                "medium": job.medium.resolve_paths(base_dir),
                "source": job.source.resolve_paths(base_dir),
            }
        )
        self.name = name
        self.out = None if out is None else Path(out)
        self.backend = backend
        self.progress = progress
        self.allow_slow_cpu = allow_slow_cpu
        self.options = options
        self.base_job: ExplicitJobConfig = job
        self._base_data: dict = json.loads(job.model_dump_json())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Study(name={self.name!r}, job={self.base_job.name!r}, out={self.out})"

    # -- configuration ---------------------------------------------------
    def job_for(self, overrides: Mapping[str, Any] | None = None) -> ExplicitJobConfig:
        """The base job with ``overrides`` applied, RE-VALIDATED.

        The round trip through :func:`~caustica.config.job.parse_job` is the
        contract: ``{"drive.amplitude_kpa": -5}`` fails with the model's own
        "greater than 0" message instead of solving a negative drive.
        """
        data = self._base_data
        for path, value in (overrides or {}).items():
            data = set_by_path(data, path, value)
        return parse_job(data, f"study {self.name!r} overrides")

    def peek(self, path: str) -> Any:
        """The base job's current value at ``path`` — what a sweep varies from."""
        return get_by_path(self._base_data, path)

    # -- one run ---------------------------------------------------------
    def run(
        self,
        overrides: Mapping[str, Any] | None = None,
        *,
        label: str | None = None,
        out: str | Path | None = None,
        **kwargs: Any,
    ) -> StudyRun:
        """Run one variant. ``overrides`` and ``**kwargs`` both address job fields.

        Dotted addresses cannot be Python keywords, so a mapping is the
        general form (``study.run({"drive.amplitude_kpa": 200})``); plain
        top-level fields work as keywords (``study.run(solver="linear")``).
        """
        merged = dict(overrides or {})
        clash = set(merged) & set(kwargs)
        if clash:
            raise StudyError(f"override given twice: {sorted(clash)}")
        merged.update(kwargs)
        label = label or "run"
        job = self.job_for(merged)  # config errors raise BEFORE anything runs
        target = self._outdir_for(label, out)
        return self._execute(job, merged, label=label, index=0, outdir=target, shared_stamp=None)

    # -- many runs -------------------------------------------------------
    def sweep(
        self,
        param_path: str,
        values: Sequence[Any],
        *,
        out: str | Path | None = None,
        labels: Sequence[str] | None = None,
        **common: Any,
    ) -> StudySweep:
        """Run one job once per value of ``param_path``; returns a :class:`StudySweep`.

        ``**common`` are overrides applied to EVERY run (so a sweep can pin
        ``solver="linear"`` while varying the drive). Every variant is built
        and validated before the first solve starts: a sweep that dies on
        value 3 because of a typo would have wasted values 1 and 2.
        """
        values = tuple(values)
        if not values:
            raise StudyError(f"sweep of {param_path!r} needs at least one value")
        if param_path in common:
            raise StudyError(
                f"{param_path!r} is both the swept parameter and a fixed override — "
                f"the fixed one would win for every run"
            )
        self.peek(param_path)  # address check, before any job is built
        if labels is not None and len(labels) != len(values):
            raise StudyError(f"got {len(labels)} labels for {len(values)} values")

        leaf = param_path.split(".")[-1]
        plan: list[tuple[str, dict, ExplicitJobConfig]] = []
        for i, value in enumerate(values):
            over = {**common, param_path: value}
            name = labels[i] if labels is not None else f"{i:02d}-{leaf}-{_fmt_value(value)}"
            plan.append((_slug(name), over, self.job_for(over)))

        root = Path(out) if out is not None else self.out
        shared = stamp(self.backend)
        started = time.perf_counter()
        runs = [
            self._execute(
                job,
                over,
                label=name,
                index=i,
                outdir=None if root is None else root / "runs" / name,
                shared_stamp=shared,
            )
            for i, (name, over, job) in enumerate(plan)
        ]
        return StudySweep(
            study=self.name,
            param_path=param_path,
            values=values,
            runs=runs,
            root=root,
            stamp=shared,
            common=dict(common),
            elapsed_s=round(time.perf_counter() - started, 2),
        )

    # -- internals -------------------------------------------------------
    def _outdir_for(self, label: str, out: str | Path | None) -> Path | None:
        if out is not None:
            return Path(out)
        return None if self.out is None else self.out / "runs" / _slug(label)

    def _options(self) -> RunnerOptions:
        """A fresh copy per run, with ``out`` cleared — the study owns the folder."""
        opts = RunnerOptions(**vars(self.options)) if self.options is not None else RunnerOptions()
        opts.out = None
        return opts

    def _execute(
        self,
        job: ExplicitJobConfig,
        overrides: Mapping[str, Any],
        *,
        label: str,
        index: int,
        outdir: Path | None,
        shared_stamp: dict | None,
    ) -> StudyRun:
        run = StudyRun(
            study=self.name,
            label=label,
            index=index,
            overrides=dict(overrides),
            job=job,
            job_hash=job_hash(job),
            outdir=outdir,
            stamp=shared_stamp or stamp(self.backend),
        )
        t0 = time.perf_counter()
        try:
            sim = simulate(
                job,
                backend=self.backend,
                out=outdir,
                progress=self.progress,
                allow_slow_cpu=self.allow_slow_cpu,
                options=self._options(),
            )
        except SimulationError as exc:
            # Recorded, never swallowed: the row says FAILED, the sweep goes
            # on, and the exit code stays the runner's own classification.
            run.error = str(exc)
            run.exit_code = exc.exit_code
            run.actual = {"wall_s": round(time.perf_counter() - t0, 2)}
            return run
        wall = time.perf_counter() - t0
        run.simulation = sim
        run.warnings = sim.warnings
        run.metrics = sim.metrics
        run.expected = _expected_from_plan(sim.plan)
        run.actual = _actual_from(sim, wall)
        return run
