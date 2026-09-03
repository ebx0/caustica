"""The Colab bridge: one call from a notebook cell, no logic in the notebook.

This is the one module in caustica that is allowed an opinion about *where*
it is running. Everything below it — the job format, the planner, the two
pre-run gates, the runner's output folder, its exit codes, ``error.json``
and the ``cancel`` file — stays environment-blind and exactly as
the GUI contract page freezes it. The bridge does not re-implement any of
it; it adds the three things a notebook would otherwise have to carry itself:

1. **An environment verdict before anything is prepared.** :func:`preflight`
   prints :func:`caustica.env.env_report` and then *requires* a GPU. Nothing
   is downloaded, no folder is created and no medium is built until it has
   passed, because an unusable runtime should cost a message, not a
   multi-GB build. caustica never installs anything for you, so
   the refusal names the fix for the machine you are actually on — and it
   keeps apart the two failures a single "no GPU" line would blur (see
   :func:`require_gpu_here`).
2. **A default output folder under** ``/content``. Colab's session disk.
3. **A readable verdict for a run that did not finish**, assembled from the
   runner's own ``error.json`` — the same message, advice, ``stage`` and
   ``error_class`` a GUI would route on, never a second diagnosis invented
   here. The two runs that write no failure record by contract (an
   interrupted one, a dry one) are told apart and said so, rather than
   pointed at a file that will not be there.

The VRAM gate is deliberately NOT repeated here. It lives once, in
``caustica.runner.check_gates``, where it runs plan-first against *free*
device VRAM before anything expensive happens; a copy in the bridge could
only drift from it. What the bridge adds is the check the runner
deliberately does not make: the runner's ``backend="auto"`` falls back to
numpy on a GPU-less machine, which on Colab means an hours-long CPU run
nobody asked for.

**No Google Drive, anywhere.** This module does not mount Drive,
does not know any Drive path, and carries no Drive-specific retry logic. If
you want a run to outlive the session, mount Drive yourself in a cell and
pass ``out=<that path>`` — the runner writes wherever it is pointed, and always
has. The accepted risk is written down: ``/content`` survives a
runtime restart (so ``resume=True`` works) but not a VM teardown.

``google.colab`` is never imported either. The bridge only *asks* whether it
is already loaded, through the single probe :mod:`caustica.env` uses for its
own messages.

Typical use is two lines in a notebook cell::

    from caustica.colab import run_job, show

    outdir = run_job("my_job.json")   # or an http(s) URL to one
    show(outdir)
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from caustica.core.backend import cupy_available
from caustica.env import _on_colab, env_report, require_gpu
from caustica.facade import SimulationError
from caustica.io.atomic import atomic_write
from caustica.runner import (
    ERROR_FILE,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_OOM,
    RunnerOptions,
    run_job_file,
)

__all__ = [
    "CONTENT_ROOT",
    "default_out",
    "env_summary",
    "on_colab",
    "preflight",
    "require_gpu_here",
    "run_job",
    "session_root",
    "show",
    "summary",
]

#: Colab's session-local disk. This is NOT Google Drive: it is the scratch
#: space every Colab runtime already has, it needs no mount, and it is gone
#: when the VM goes away (an accepted risk).
CONTENT_ROOT = Path("/content")

#: Where a job given as a URL is downloaded to, under the session root.
JOBS_DIRNAME = "jobs"

#: The install sentence :func:`caustica.env.require_gpu` uses off Colab. The
#: bridge needs the same words for the case env policy cannot see (cupy simply
#: absent), and two hand-written copies would drift, so there is one string and
#: a test asserting env's message still contains it.
INSTALL_ADVICE = (
    "If it has an NVIDIA GPU with CUDA 12: pip install cupy-cuda12x (the "
    "caustica[gpu] extra). caustica never installs it for you."
)

#: What a non-zero runner exit means, quoted from the exit-code table in
#: the GUI contract page. The set is closed and the numbers are the queue's
#: API — reproduced here, never extended here, and pinned against the page by
#: ``tests/test_colab.py::test_the_exit_code_glosses_are_the_documented_ones``
#: so this copy cannot drift from the one a GUI reads.
_EXIT_MEANING = {
    2: "config error: bad job, unknown backend/GPU, checkpoint conflict, CPU-time refusal",
    3: "refused before solving: the run does not fit in memory",
    4: "the solve or the store failed",
    5: "stopped cleanly and resumably (`--max-hours`, or a `cancel` file)",
}


# --------------------------------------------------------------- where am I


def on_colab() -> bool:
    """True when this process is a Google Colab runtime.

    Detection only, and through the ONE probe caustica already has
    (:func:`caustica.env._on_colab`: a ``sys.modules`` / env-var look). A
    second definition here could disagree with the message
    :func:`caustica.env.require_gpu` picks for the same machine, and two
    modules answering "are we on Colab?" differently is exactly the drift
    this bridge exists to avoid. ``google.colab`` itself is never imported —
    importing a thing to ask whether it exists is how a library ends up
    needing Colab in order to run.
    """
    return _on_colab()


def session_root() -> Path:
    """``/content`` on Colab; the current directory anywhere else.

    The local fallback is what makes the notebook's cells runnable on a
    laptop without editing them: same call, sane place.
    """
    return CONTENT_ROOT if on_colab() else Path.cwd()


def _is_url(spec: str) -> bool:
    return spec.lower().startswith(("http://", "https://"))


def default_out(job_path: str | os.PathLike[str]) -> Path:
    """``<session root>/runs/<job file stem>`` — where a run lands by default.

    Derived from the file NAME, deliberately not from the job's ``name``
    field: reading that field would mean parsing the job *here*, and a job
    that does not parse must fail through the runner (exit 2 plus an
    ``error.json`` a GUI can read), not as a traceback out of the bridge.

    Because :func:`run_job` therefore always passes an explicit ``out``, the
    runner's own default (``<job dir>/runs/<job name>``) is never reached
    from this module — there is one rule in play per call, not two.
    """
    spec = str(job_path)
    name = urllib.parse.urlparse(spec).path if _is_url(spec) else spec
    return session_root() / "runs" / (Path(name).stem or "run")


# ------------------------------------------------------------ the GPU gate


def _cupy_installed() -> bool:
    """Is there a ``cupy`` for Python to import? (Nothing is imported.)"""
    try:
        return importlib.util.find_spec("cupy") is not None
    except (ImportError, ValueError):  # namespace/partial installs
        return False


def require_gpu_here(reason: str = "caustica.colab.run_job") -> None:
    """Raise unless a usable CUDA GPU is present. Never installs anything.

    Two failures, two messages, because they have two different fixes and a
    single "no GPU" line sends half of its readers to the wrong one:

    * **cupy is not installed.** Nothing is wrong with the *device* yet. On
      Colab this almost always means the runtime is not a GPU one, since a
      Colab GPU runtime ships cupy preinstalled — so the fix is the Runtime
      menu, and the pip command is named only as the fallback for a custom
      image. caustica will not run it for you.
    * **cupy is installed, but no CUDA device answers.** The runtime itself
      has no GPU. That sentence is :func:`caustica.env.require_gpu`'s, and it
      is *called*, not restated — the device wording exists once.

    Only the first case is the bridge's own, because it is the one
    ``env.require_gpu`` cannot see: it asks the backend "is a GPU usable?",
    gets one ``False`` for both causes, and then picks its message from the
    *machine* rather than the cause. On Colab the cupy-missing case is the
    common one, so this branch is what a user usually meets; the pip command
    it names is the sentence env policy uses off Colab, kept identical on
    purpose and pinned by a test.

    Known and accepted: a *broken* cupy install (importable metadata, failing
    import) lands in the second message, because the device probe is what
    fails first. On Colab that message is the CPU-runtime one, so a GPU
    runtime with a broken CUDA stack is told to switch to the runtime it is
    already on, and it names no install command — the honest fix for that
    case is env policy's to make, and env policy is not this module's to
    edit: a known wart, recorded rather than papered over here with a
    third message that guesses.
    """
    if cupy_available():
        return
    if not _cupy_installed():
        if on_colab():
            raise RuntimeError(
                f"cupy is not installed in this Colab runtime, so {reason} cannot "
                f"start. A Colab GPU runtime ships cupy preinstalled, so this is "
                f"almost certainly a CPU runtime: Runtime -> Change runtime type -> "
                f"Hardware accelerator: GPU, then reconnect and re-run this cell. If "
                f"it really is a GPU runtime (a custom image), install it yourself: "
                f"pip install cupy-cuda12x. caustica never installs anything for you."
            )
        raise RuntimeError(
            f"cupy is not installed on this machine, so {reason} cannot start. {INSTALL_ADVICE}"
        )
    require_gpu(reason)  # cupy imports fine -> the DEVICE is what is missing


def env_summary(report: dict | None = None) -> str:
    """Two or three lines describing this machine. Never raises.

    Formats :func:`caustica.env.env_report` — the same dict the runner
    stamps into ``run_meta.json``, so the notebook printout and the audit
    trail cannot disagree about what ran where.
    """
    r = env_report() if report is None else report
    lines = [
        f"caustica {r.get('caustica')} | python {r.get('python')} | "
        f"numpy {r.get('numpy')} | backend {r.get('resolved_backend')} | "
        f"{'Colab' if on_colab() else 'local'}"
    ]
    if r.get("gpu_name"):
        lines.append(
            f"GPU {r.get('gpu_name')} | VRAM {r.get('vram_free_gib')} of "
            f"{r.get('vram_total_gib')} GiB free | cupy {r.get('cupy_version')} | "
            f"driver {r.get('driver_version')}"
        )
    elif r.get("gpu_probe_error"):
        lines.append(f"GPU probe failed: {r.get('gpu_probe_error')}")
    lines.append(f"platform {r.get('platform')}")
    return "\n".join(lines)


def preflight(*, verbose: bool = True) -> dict:
    """Print what machine this is, then require a GPU. Writes nothing to disk.

    :func:`run_job` calls this first and on purpose: an unusable runtime
    costs a printout and one actionable message, never a download, a folder
    or a medium build. Returns the ``env_report()`` dict it printed.

    "Writes nothing" is the exact claim, and it is narrower than "does
    nothing" — asking the environment what it is has three effects worth
    knowing, all of them the ones ``caustica run`` already has because it
    calls the same function:

    * resolving ``auto`` emits the once-per-process "falling back to numpy"
      warning on a GPU-less machine, so on a CPU box that warning arrives
      just before the refusal that follows it;
    * the cupy probe result is cached for the process lifetime;
    * on a machine that *has* a GPU, reading free VRAM initializes the CUDA
      context (0.8-1.5 GiB on Colab). The runner does this too, before its
      own pre-plan snapshot, so the memory gate sees the same picture either
      way.
    """
    report = env_report()
    if verbose:
        print(env_summary(report))
    require_gpu_here()
    return report


# ----------------------------------------------------------------- the run


def _fetch(url: str, dest_dir: Path) -> Path:
    """Download a job document into ``dest_dir``; return the local path.

    stdlib only (the bridge adds no runtime dependency) and deliberately
    dumb: it saves bytes, it does not look at them. A job that is not a job
    must still fail through the runner, with the runner's exit code.

    A job fetched from a URL has to be SELF-CONTAINED. Relative paths inside
    a job resolve against the job file (trap T4), which after a download is
    this folder — not wherever the URL lived.

    The write goes through :func:`caustica.io.atomic.atomic_write`, like every
    other file caustica creates: a half-downloaded job that still parses is
    a worse failure than no job at all, and re-running the cell must replace
    the previous download rather than tear it.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / (Path(urllib.parse.urlparse(url).path).name or "job.json")
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    with atomic_write(target) as tmp:
        tmp.write_bytes(payload)
    print(f"downloaded {url}\n       -> {target}")
    return target


def _runner_option_names() -> list[str]:
    return sorted(f.name for f in dataclasses.fields(RunnerOptions) if f.name not in ("out", "gpu"))


def _read_json(path: Path) -> dict | None:
    """Read a JSON file if it is there and readable; never raise."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _failure_message(outdir: Path, code: int) -> str:
    """Explain a non-zero exit, using the runner's own ``error.json``.

    The bridge invents no second diagnosis: the message and the advice here
    are the strings the runner already wrote and printed, and the last line
    quotes the two fields a program routes on (``stage`` / ``error_class``)
    rather than paraphrasing them. Only the Colab-specific lever — the
    runtime's GPU size, the fact that ``/content`` survives a restart but not
    a teardown — is added.

    ``error.json`` is named only when it is actually there. Two documented
    cases produce none: an interrupted run (stopping is not failing) and a
    ``--dry-run`` (a probe is not an attempt). Pointing a user at a file the
    contract promises will be absent is the one thing this message must not
    do.
    """
    lines = [f"caustica run exited {code}: {_EXIT_MEANING.get(code, 'unclassified failure')}."]
    record = _read_json(outdir / ERROR_FILE) or {}
    message = str(record.get("message", "")).strip()
    if message:
        lines.append(message)
    lines += [f"  -> {advice}" for advice in record.get("advice", ())]
    if code == EXIT_OOM:
        lines.append(
            "  -> on Colab the other lever is the runtime itself: Runtime -> Change "
            "runtime type offers bigger GPUs (T4 ~15 GB < L4 ~22 GB < A100 ~40 GB)."
        )
    if code == EXIT_INTERRUPTED:
        lines.append(
            f"  -> nothing is lost: the checkpoint is in {outdir}. Continue with "
            f"run_job(..., out={str(outdir)!r}, resume=True) in the SAME session — "
            f"/content survives a runtime restart, not a VM teardown."
        )
    if record:
        lines.append(
            f"run folder: {outdir} — error.json says stage={record.get('stage')!r}, "
            f"error_class={record.get('error_class')!r} (those two are what a program "
            f"routes on)."
        )
    else:
        # No failure record, by contract — so the runner's own diagnosis went
        # to stderr and nowhere else. plan.json is the one file a dry run DOES
        # write, and it carries the same `advice` strings error.json would
        # have; quoting them keeps the exception useful instead of bare.
        for advice in (_read_json(outdir / "plan.json") or {}).get("advice", ()):
            lines.append(f"  -> {advice}")
        lines.append(
            f"run folder: {outdir} — no error.json here, by contract: an interrupted "
            f"run and a dry run both write none "
            f"(https://ebx0.github.io/caustica/gui_contract/). "
            f"The runner printed its full diagnosis to stderr above."
        )
    return "\n".join(lines)


def _announce_folder_override(job_file: Path, outdir: Path) -> None:
    """Say it out loud when a job's own ``output.folder`` is not what we use.

    :func:`run_job` ALWAYS passes an explicit output folder, so the runner's
    own rule (``output.folder`` if the job names one, else
    ``<job dir>/runs/<job name>``) never applies through the bridge. That is
    deliberate — a folder chosen while authoring a job on a laptop is rarely
    where a Colab VM should write, and the caller needs a path handed back —
    but it must not be *silent*: the same job through ``caustica run`` would
    land somewhere else.

    Nothing is parsed strictly and nothing is resolved: a job that will not
    load is the runner's to refuse (exit 2 plus ``error.json``), so every
    failure here is swallowed on purpose. This function only ever prints.
    """
    try:
        declared = json.loads(job_file.read_text(encoding="utf-8"))["output"]["folder"]
    except Exception:
        return
    if declared:
        print(
            f"note: this job asks for output folder {declared!r}; the Colab bridge "
            f"writes to {outdir} instead. Pass out=... to choose the folder yourself."
        )


def run_job(
    job_path: str | os.PathLike[str],
    out: str | os.PathLike[str] | None = None,
    gpu: str | None = None,
    **runner_opts: Any,
) -> Path:
    """Run ONE ``caustica-job/1`` file on this runtime's GPU; return its folder.

    Order matters and is the point of the function: environment first, then
    the keywords, then the job. Nothing is fetched, created or built until
    the runtime has been judged fit AND the options have been accepted — a
    typo'd keyword must not cost a download either.

    Parameters
    ----------
    job_path:
        A local path to a ``caustica-job/1`` file, or an ``http(s)://`` URL
        to one (downloaded into ``<session root>/jobs/`` first). A URL job
        must be self-contained — see :func:`_fetch`.
    out:
        Where to write. Defaults to :func:`default_out`, i.e. under
        ``/content`` on Colab. Any path works, **including one inside a Drive
        folder you mounted yourself** — this module neither mounts nor knows
        about Drive. Because an explicit folder is always passed
        on, a job's own ``output.folder`` field does not take effect here;
        when a job names one, that is said out loud rather than swallowed.
    gpu:
        The datasheet GPU the planner's second estimate targets. Does not
        select hardware; the run uses the device this runtime has.
    **runner_opts:
        Passed straight through to :class:`caustica.runner.RunnerOptions`
        (``resume``, ``max_hours``, ``backend``, ``preview_only``,
        ``dry_run``, ...). ``progress`` defaults to ``"auto"`` here — this is
        an entry point for a human watching a cell, like
        :func:`caustica.simulate`, while the library default stays silent.

    Returns
    -------
    Path
        The output folder, holding whatever the runner writes for the options
        it was given: the full set (``job.json``, ``plan.json``,
        ``status.json``, ``result.h5``, ``preview.npz``, ``metrics.json``,
        ``run_meta.json``) for a default run, and the documented subsets
        otherwise — ``preview_only`` skips ``result.h5``, ``dry_run`` writes
        the plan and nothing else, and a non-native solver has no plan to
        write.

    Raises
    ------
    RuntimeError
        Before the job is even read: this runtime cannot run GPU work. It
        carries NO ``exit_code`` — nothing ran, so there is no run to
        classify. Note that :class:`caustica.SimulationError` subclasses
        ``RuntimeError``, so an ``except RuntimeError`` that reads
        ``.exit_code`` must catch ``SimulationError`` first.
    caustica.SimulationError
        The run itself did not succeed. ``.exit_code`` is the runner's own
        disjoint code (2 config, 3 OOM, 4 solver, 5 interrupted-resumable),
        the same number the CLI returns and a queue routes on.

    Notes
    -----
    The GPU gate guarantees a usable device *exists*; it does not overrule
    the job. On a machine that has one, a job whose ``backend`` field says
    ``numpy`` still runs on the CPU and is still judged by the runner's
    CPU-time gate. On a machine that has none, this function refuses first,
    so a deliberately-numpy job belongs in ``caustica run`` or
    :func:`caustica.simulate` — the bridge means "run this on the GPU".
    """
    preflight()
    spec = str(job_path)
    outdir = Path(out) if out is not None else default_out(spec)
    runner_opts.setdefault("progress", "auto")
    try:
        opts = RunnerOptions(out=outdir, gpu=gpu, **runner_opts)
    except TypeError as exc:
        raise TypeError(
            f"{exc}. run_job hands its extra keywords straight to RunnerOptions; "
            f"the accepted names are: {', '.join(_runner_option_names())}."
        ) from None

    job_file = _fetch(spec, session_root() / JOBS_DIRNAME) if _is_url(spec) else Path(job_path)
    if out is None:
        _announce_folder_override(job_file, outdir)
    code = run_job_file(job_file, opts)
    if code != EXIT_OK:
        raise SimulationError(_failure_message(outdir, code), code)
    print(f"\noutput folder: {outdir}")
    print(f"next: show({str(outdir)!r})   # summary + figures, right here")
    return outdir


# --------------------------------------------------------------- the result


def _fmt(value: Any, unit: str = "") -> str:
    return "n/a" if value is None else f"{value}{unit}"


def summary(outdir: str | os.PathLike[str]) -> str:
    """A short text report of a finished run, read from the files it wrote.

    Every number comes out of ``metrics.json`` and ``run_meta.json``, so this
    quotes the run's own single-source metrics rather than recomputing
    anything. A folder missing those files still produces useful lines.
    """
    outdir = Path(outdir)
    metrics = _read_json(outdir / "metrics.json") or {}
    meta = _read_json(outdir / "run_meta.json") or {}
    peak = metrics.get("peak", {})
    spot = metrics.get("focal_spot", {})
    target = metrics.get("target", {})
    actual = meta.get("actual", {})
    planner = meta.get("planner") or {}
    environment = meta.get("environment", {})

    lines = [f"run: {outdir}"]
    if meta:
        lines.append(
            f"job {meta.get('job')} | solver {meta.get('solver')} | "
            f"backend {meta.get('backend')} | "
            f"{environment.get('gpu_name', 'no GPU stamped')}"
        )
    for warning in meta.get("ppw_warnings", ()):
        lines.append(f"  ! {warning}")
    if peak:
        position = peak.get("position_mm_from_apex", {})
        lines.append(
            f"peak {_fmt(peak.get('p_mpa'))} MPa at "
            f"({position.get('x')}, {position.get('y')}, {position.get('z')}) mm "
            f"from the apex | gain x{_fmt(peak.get('gain_vs_source'))} | "
            f"I_sppa {_fmt(peak.get('isppa_w_cm2'))} W/cm^2"
        )
    if spot:
        lines.append(
            f"-6 dB spot: axial {_fmt(spot.get('axial_6db', {}).get('width_mm'))} mm | "
            f"lateral {_fmt(spot.get('lateral_x_6db', {}).get('width_mm'))} x "
            f"{_fmt(spot.get('lateral_y_6db', {}).get('width_mm'))} mm | "
            f"volume {_fmt(spot.get('volume_above_6db_mm3'))} mm^3"
        )
    if target:
        lines.append(f"target miss: {_fmt(target.get('displacement_norm_mm'))} mm")
    if actual:
        lines.append(
            f"wall time {_fmt(actual.get('elapsed_solve_s'))} s solve / "
            f"{_fmt(actual.get('elapsed_total_s'))} s total"
            + (
                f" (planner expected {planner.get('t_expected_s')} s)"
                if planner.get("t_expected_s") is not None
                else ""
            )
        )
        lines.append(
            f"VRAM peak {_fmt(actual.get('vram_pool_peak_gib'))} GiB"
            + (
                f" (planner expected {planner.get('vram_gib')} GiB)"
                if planner.get("vram_gib") is not None
                else ""
            )
        )
    present = [p.name for p in sorted(outdir.glob("*")) if p.is_file()]
    if present:
        lines.append("files: " + ", ".join(present))
    return "\n".join(lines)


def _display(paths: list[Path]) -> bool:
    """Show images inline if we are in a notebook; return whether we did."""
    try:  # IPython is present in Colab and in Jupyter, and nowhere else
        from IPython.display import Image, display  # noqa: PLC0415 (optional, notebook-only)
    except ImportError:
        return False
    for path in paths:
        display(Image(filename=str(path)))
    return True


def show(
    outdir: str | os.PathLike[str],
    *,
    figures: bool = True,
    preview_only: bool = False,
) -> Path | None:
    """Print :func:`summary` and render the run's report inline.

    Rendering goes through ``caustica.report.render_report`` — the very same
    renderer ``caustica report <folder>`` uses, so the figures in the cell
    and the figures in the folder are one artifact, not two.

    Returns the path of the rendered HTML, or ``None`` for any reason the
    figures did not appear: ``figures=False``, matplotlib missing, or a
    renderer failure (a torn ``preview.npz``, an unreadable ``result.h5``).
    Those are NOT equivalent, and the difference is printed rather than
    hidden behind the return value — a viewer that killed a notebook after a
    successful run because a figure would not draw would be the wrong trade,
    but so would silence. ``caustica report <folder>`` treats the same
    failure harder: it exits 2. The numbers are printed either way, and they
    come from the run's own ``metrics.json``.

    ``preview_only=True`` renders from the <=10 MB ``preview.npz`` even when
    the full ``result.h5`` is there — the quick look, at quick-look cost.
    """
    outdir = Path(outdir)
    print(summary(outdir))
    if not figures:
        return None
    from caustica.report.renderers import render_report  # noqa: PLC0415 (matplotlib lazy)

    try:
        html = render_report(outdir, preview_only=preview_only)
    except Exception as exc:
        hint = (
            " Install the report extra: pip install 'caustica[report]'."
            if isinstance(exc, ImportError)
            else ""
        )
        print(
            f"\n(no figures — {type(exc).__name__}: {exc}.{hint} The numbers above "
            f"stand; `caustica report {outdir}` reports the same failure with exit 2.)"
        )
        return None
    if not _display(sorted(html.parent.glob("*.png"))):
        print(f"\nreport: {html}")
    return html
