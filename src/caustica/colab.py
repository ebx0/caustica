"""The Colab bridge (M10f): one call from a notebook cell, no logic in the notebook.

This is the one module in caustica that is allowed an opinion about *where*
it is running. Everything below it — the job format, the planner, the two
pre-run gates, the runner's output folder, its exit codes, ``error.json``
and the ``cancel`` file — stays environment-blind and exactly as
``docs/gui_contract.md`` freezes it. The bridge does not re-implement any of
it; it adds the three things a notebook would otherwise have to carry itself:

1. **An environment verdict before anything is prepared.** :func:`preflight`
   prints :func:`caustica.env.env_report` and then *requires* a GPU. Nothing
   is downloaded, no folder is created and no medium is built until it has
   passed, because an unusable runtime should cost a message, not a
   multi-GB build. caustica never installs anything for you (PLAN K6), so
   the refusal names the fix for the machine you are actually on — and it
   keeps apart the two failures a single "no GPU" line would blur (see
   :func:`require_gpu_here`).
2. **A default output folder under** ``/content``. Colab's session disk.
3. **A readable verdict for a run that did not finish**, assembled from the
   runner's own ``error.json`` — the same message and the same advice a GUI
   would route on, never a second diagnosis invented here.

The VRAM gate is deliberately NOT repeated here. It lives once, in
``caustica.runner.check_gates``, where it runs plan-first against *free*
device VRAM before anything expensive happens; a copy in the bridge could
only drift from it. What the bridge adds is the check the runner
deliberately does not make: the runner's ``backend="auto"`` falls back to
numpy on a GPU-less machine, which on Colab means an hours-long CPU run
nobody asked for.

**No Google Drive, anywhere (PLAN K12).** This module does not mount Drive,
does not know any Drive path, and carries no Drive-specific retry logic. If
you want a run to outlive the session, mount Drive yourself in a cell and
pass ``out=<that path>`` — the runner writes wherever it is pointed, and has
since M10c. The accepted risk is written down: ``/content`` survives a
runtime restart (so ``resume=True`` works) but not a VM teardown.

``google.colab`` is never imported either. The bridge only *asks* whether it
is already loaded, through the single probe :mod:`caustica.env` uses for its
own messages.

Typical use is two lines in a notebook cell::

    from caustica.colab import run_job, show

    outdir = run_job("my_job.json")   # or an https URL to one
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
#: when the VM goes away (PLAN K12's accepted risk).
CONTENT_ROOT = Path("/content")

#: Where a job given as a URL is downloaded to, under the session root.
JOBS_DIRNAME = "jobs"

#: What a non-zero runner exit means, in the words of docs/gui_contract.md.
#: The set is closed and the numbers are the queue's API — quoted here, never
#: extended here.
_EXIT_MEANING = {
    2: "config error — the job would not load or build",
    3: "refused before solving — the run does not fit in memory",
    4: "the solve or the store failed",
    5: "stopped cleanly and resumably — a checkpoint is on disk",
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
      image. caustica will not run it for you (PLAN K6/D6).
    * **cupy is installed, but no CUDA device answers.** The runtime itself
      has no GPU. :func:`caustica.env.require_gpu` owns that message and is
      called for it, so the wording exists once and cannot drift.

    Known and accepted: a *broken* cupy install (importable metadata, failing
    import) lands in the second message, because the device probe is what
    fails first. The install command is still one line above it.
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
            f"cupy is not installed on this machine, so {reason} cannot start. If it "
            f"has an NVIDIA GPU with CUDA 12: pip install cupy-cuda12x (the "
            f"caustica[gpu] extra). caustica never installs it for you."
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
    """Print what machine this is, then require a GPU. Prepares NOTHING.

    :func:`run_job` calls this first and on purpose: an unusable runtime
    costs a printout and one actionable message, never a download, a folder
    or a medium build. Returns the ``env_report()`` dict it printed.
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
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / (Path(urllib.parse.urlparse(url).path).name or "job.json")
    with urllib.request.urlopen(url, timeout=60) as response:
        target.write_bytes(response.read())
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
    are the strings the runner already wrote and printed. Only the
    Colab-specific lever — the runtime's GPU size, the fact that ``/content``
    survives a restart but not a teardown — is added.
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
    lines.append(f"run folder: {outdir} (error.json carries the same verdict).")
    return "\n".join(lines)


def run_job(
    job_path: str | os.PathLike[str],
    out: str | os.PathLike[str] | None = None,
    gpu: str = "A100",
    **runner_opts: Any,
) -> Path:
    """Run ONE ``caustica-job/1`` file on this runtime's GPU; return its folder.

    Order matters and is the point of the function: environment first, then
    the job. Nothing is fetched, created or built until the runtime has been
    judged fit.

    Parameters
    ----------
    job_path:
        A local path to a ``caustica-job/1`` file, or an ``https://`` URL to
        one (downloaded into ``<session root>/jobs/`` first). A URL job must
        be self-contained — see :func:`_fetch`.
    out:
        Where to write. Defaults to :func:`default_out`, i.e. under
        ``/content`` on Colab. Any path works, **including one inside a Drive
        folder you mounted yourself** — this module neither mounts nor knows
        about Drive (PLAN K12).
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
        The output folder, with the ordinary runner contents (``job.json``,
        ``plan.json``, ``status.json``, ``result.h5``, ``preview.npz``,
        ``metrics.json``, ``run_meta.json``).

    Raises
    ------
    RuntimeError
        Before the job is touched: this runtime cannot run GPU work. The
        message names the fix for this machine and never runs pip.
    caustica.SimulationError
        The run itself did not succeed. ``.exit_code`` is the runner's own
        disjoint code (2 config, 3 OOM, 4 solver, 5 interrupted-resumable),
        the same number the CLI returns and a queue routes on.

    Notes
    -----
    The GPU gate guarantees a usable device *exists*; it does not overrule
    the job. A job whose ``backend`` field says ``numpy`` still runs on the
    CPU, and is still judged by the runner's CPU-time gate.
    """
    preflight()
    spec = str(job_path)
    job_file = _fetch(spec, session_root() / JOBS_DIRNAME) if _is_url(spec) else Path(job_path)
    outdir = Path(out) if out is not None else default_out(spec)
    runner_opts.setdefault("progress", "auto")
    try:
        opts = RunnerOptions(out=outdir, gpu=gpu, **runner_opts)
    except TypeError as exc:
        raise TypeError(
            f"{exc}. run_job hands its extra keywords straight to RunnerOptions; "
            f"the accepted names are: {', '.join(_runner_option_names())}."
        ) from None

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
    and the figures in the folder are one artifact, not two. Returns the
    path of the rendered HTML, or ``None`` when figures were skipped or
    matplotlib is not installed (the numbers are printed either way).

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
    except ImportError as exc:
        print(f"\n(no figures: {exc}. Install the extra: pip install 'caustica[report]')")
        return None
    except Exception as exc:
        print(f"\n(no figures: {type(exc).__name__}: {exc})")
        return None
    if not _display(sorted(html.parent.glob("*.png"))):
        print(f"\nreport: {html}")
    return html
