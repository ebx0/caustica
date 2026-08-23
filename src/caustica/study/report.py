"""``STUDY.md`` + ``study.json``: the human half and the machine half.

Nothing in here computes a number. Focal metrics come from
:mod:`caustica.report.metrics` (already attached to each run), the
planner/actual pair comes from the runner's own stamp, and the environment
comes from :func:`caustica.env.env_report` — this module only decides which
of them go in which table. A report that recomputed anything would be a
second definition of it, and the whole point of ``caustica.report`` being the
single source is that there is not one.

Two shapes, one contract (``caustica-study/1``):

* **one run** — sections rendered by :func:`caustica.report.html.render_markdown`,
  reusing the very row builders ``caustica report`` and ``focus_study`` use,
  so a study's "peak pressure" line is character-for-character theirs.
* **a sweep** — wide comparison tables, which the row renderer cannot
  express, written here directly (the same choice
  :mod:`caustica.validation.analytic_suite` made for its gate tables).

Both write ``study.json`` through :func:`caustica.io.atomic.atomic_write`:
it is the machine half, something else will read it, and a half-written
report that merely *exists* is the failure atomic writes exist to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from caustica.io.atomic import atomic_write
from caustica.report import html as hrep
from caustica.study.core import FORMAT, StudyRun, StudySweep
from caustica.validation._verdict import fmt_dev_pct, fmt_num

#: The two files every study report writes, whatever its shape.
MD_NAME = "STUDY.md"
JSON_NAME = "study.json"


def _cell(text: Any) -> str:
    """One Markdown table cell: escape the pipes that would split the row."""
    return str(text).replace("|", "\\|")


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _write_payload(outdir: Path, payload: dict) -> None:
    """The machine half, atomically — something else reads this file."""
    with atomic_write(outdir / JSON_NAME) as tmp:
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write(outdir: Path, payload: dict, md: str) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_payload(outdir, payload)
    with atomic_write(outdir / MD_NAME) as tmp:
        tmp.write_text(md, encoding="utf-8")
    return outdir


def _rel(path: Path | str | None, base: Path) -> str:
    """A run folder as the report should link to it — relative when it can be."""
    if path is None:
        return "— (in memory)"
    p = Path(path)
    try:
        rel = str(p.resolve().relative_to(Path(base).resolve())).replace("\\", "/")
    except ValueError:
        return str(p)
    return "." if rel in ("", ".") else rel


# ------------------------------------------------------------ stamp tables


def stamp_rows(stamp: dict) -> list[tuple[str, str]]:
    """The environment/GPU/git provenance, as ``(label, value)`` pairs.

    One builder for both report shapes, so a sweep and a single run can never
    stamp themselves with different facts in different words.
    """
    env = stamp.get("environment", {})
    gpu = env.get("gpu_name")
    vram = env.get("vram_total_gib")
    return [
        ("generated", str(stamp.get("generated", "?"))),
        ("caustica", f"{stamp.get('caustica')} @ {str(stamp.get('git_commit', ''))[:12]}"),
        ("host", str(stamp.get("host", "?"))),
        (
            "python / numpy / scipy",
            f"{env.get('python')} / {env.get('numpy')} / {env.get('scipy')}",
        ),
        ("platform", str(env.get("platform", "?"))),
        ("resolved backend", str(env.get("resolved_backend", "?"))),
        ("GPU", f"{gpu} ({vram} GiB)" if gpu else "— (no CUDA device)"),
    ]


def expected_vs_actual(expected: dict, actual: dict) -> list[list[Any]]:
    """The planner-vs-reality rows: ``[quantity, expected, actual, deviation]``.

    ``--`` in the expected column is honest and common: the planner models
    the native engine only, so a k-Wave run has nothing to compare against.
    """
    exp_step = expected.get("t_step_s")
    act_step = actual.get("t_step_measured_s")
    return [
        [
            "solve time [s]",
            fmt_num(expected.get("t_expected_s")),
            fmt_num(actual.get("elapsed_solve_s")),
            fmt_dev_pct(expected.get("t_expected_s"), actual.get("elapsed_solve_s")),
        ],
        [
            "steps",
            fmt_num(expected.get("steps_expected")),
            fmt_num(actual.get("steps_total")),
            fmt_dev_pct(expected.get("steps_expected"), actual.get("steps_total")),
        ],
        [
            "per-step [ms]",
            fmt_num(None if exp_step is None else exp_step * 1e3),
            fmt_num(None if act_step is None else act_step * 1e3),
            fmt_dev_pct(exp_step, act_step),
        ],
        ["converged period", "--", fmt_num(actual.get("converged_period")), "--"],
        ["steps per period", fmt_num(expected.get("spp")), "--", "--"],
        ["memory [GiB]", fmt_num(expected.get("vram_gib")), "--", "--"],
    ]


# --------------------------------------------------------------- one run


def run_payload(run: StudyRun) -> dict:
    """``study.json`` for a single run."""
    return {
        "format": FORMAT,
        "kind": "run",
        "study": run.study,
        **run.stamp,
        "runs": [run.as_dict()],
    }


def _overrides_text(overrides: dict) -> str:
    if not overrides:
        return "— (the base job, unchanged)"
    return ", ".join(f"`{k}` = {v}" for k, v in overrides.items())


def write_run_report(run: StudyRun, outdir: str | Path) -> Path:
    """Write ``STUDY.md`` + ``study.json`` for one run; returns the folder."""
    outdir = Path(outdir)
    payload = run_payload(run)
    m = run.metrics or {}

    rows: list[hrep.Row] = [
        ("Study", "study", run.study),
        ("Study", "run", run.label),
        ("Study", "overrides", _overrides_text(run.overrides)),
        ("Study", "base job", f"`{run.job.name}` (`{run.job.kind}`)"),
        ("Study", "job hash", f"`{run.job_hash}`"),
        ("Study", "solver / backend", f"{run.job.solver} / {run.job.backend}"),
        ("Study", "output folder", "— (in memory)" if run.outdir is None else str(run.outdir)),
    ]
    rows += [("Environment", label, value) for label, value in stamp_rows(run.stamp)]
    if run.error is not None:
        rows.append(("Result", "FAILED", f"exit {run.exit_code}: {run.error}"))
    for quantity, exp, act, dev in expected_vs_actual(run.expected, run.actual):
        # A prediction with nothing to compare it against says so. Printing
        # "expected 4 -> actual --" invites the reader to score a deviation
        # that was never measurable.
        if act == "--":
            value = f"expected {exp} (nothing measured to compare)"
        elif exp == "--":
            value = f"actual {act} (the planner does not predict this)"
        else:
            value = f"expected {exp} → actual {act} ({dev})"
        rows.append(("Planner vs actual", quantity, value))
    rows.append(
        (
            "Planner vs actual",
            "estimate / measurement source",
            f"{run.expected.get('source', '--')} / {run.actual.get('source', '--')}",
        )
    )
    if m:
        rows += hrep.focus_rows(m) + hrep.focal_spot_rows(m)
        rows += hrep.run_rows(m) + hrep.harmonics_rows(m)

    notes = [f"resolution warning: {w}" for w in run.warnings]
    notes.append(
        "Every focal number above is `caustica.report.metrics` — the definitions "
        "`caustica report` and `apps/focus_study` quote."
    )
    files = [f"- `{JSON_NAME}` — this report, machine-readable (`{FORMAT}`)"]
    if run.outdir is not None:
        where = _rel(run.outdir, outdir)
        files.append(
            f"- `{where}` — {'this folder' if where == '.' else 'the run folder'}: result, "
            f"metrics, plan and stamp; `caustica report` renders its full figure set"
        )

    outdir.mkdir(parents=True, exist_ok=True)
    # The shared row renderer writes to whatever path it is handed, so it
    # renders straight into the atomic temp — one renderer, still atomic.
    with atomic_write(outdir / MD_NAME) as tmp:
        hrep.render_markdown(
            tmp,
            title=f"caustica study — {run.study} / {run.label}",
            description=(
                f"One run of study `{run.study}`, with the planner's prediction next to "
                f"what happened."
            ),
            lede="*Generated by `caustica.study`.*",
            rows=rows,
            notes=tuple(notes),
            files_lines=files,
        )
    _write_payload(outdir, payload)
    return outdir


# ----------------------------------------------------------------- sweep


def scaling_rows(payload: dict) -> list[list[Any]]:
    """How the peak tracked the swept value, normalised to the first run.

    Only meaningful for numeric values, and only stated where both the value
    and the peak exist — for a linear solver the last column is the sweep's
    own self-check, and for a nonlinear one it is the answer.
    """
    runs = payload.get("runs", [])
    values = payload.get("values", [])
    base = None
    out: list[list[Any]] = []
    for r, v in zip(runs, values, strict=False):
        peak = ((r.get("metrics") or {}).get("peak") or {}).get("p_mpa")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or peak is None:
            continue
        if base is None:
            if v == 0:
                continue
            base = (float(v), float(peak))
        v_ratio = float(v) / base[0]
        p_ratio = float(peak) / base[1]
        dev = "--" if v_ratio == 0 else f"{100.0 * (p_ratio / v_ratio - 1.0):+.2f}%"
        out.append([r.get("label"), fmt_num(v), f"{v_ratio:.4g}", f"{p_ratio:.4g}", dev])
    return out


def sweep_payload(sweep: StudySweep) -> dict:
    """``study.json`` for a whole sweep — the stamp once, the runs in order."""
    payload = {
        "format": FORMAT,
        "kind": "sweep",
        "study": sweep.study,
        "param_path": sweep.param_path,
        "values": list(sweep.values),
        "common_overrides": dict(sweep.common),
        **{k: v for k, v in sweep.stamp.items()},
        "elapsed_s": sweep.elapsed_s,
        "n_runs": len(sweep.runs),
        "n_ok": sum(r.ok for r in sweep.runs),
        "runs": [r.as_dict() for r in sweep.runs],
    }
    payload["scaling"] = scaling_rows(payload)
    return payload


def render_sweep_markdown(payload: dict, figs: dict[str, str], outdir: Path) -> str:
    """The human half of a sweep report. ``figs`` is ``{filename: caption}``."""
    param = payload["param_path"]
    leaf = param.split(".")[-1]
    runs = payload["runs"]
    ok = payload["n_ok"] == payload["n_runs"]

    head = [
        f"# caustica study — {payload['study']}",
        "",
        f"Sweep of `{param}` over {len(payload['values'])} values "
        f"(`{payload['format']}`), {payload['n_ok']} / {payload['n_runs']} runs completed"
        f"{'' if ok else ' — **some runs FAILED, see the table**'}.",
        "",
        "*Generated by `caustica.study`. Every focal number is "
        "`caustica.report.metrics`; the planner column is the same estimate "
        "`caustica run` prints before it starts.*",
        "",
    ]
    solvers = sorted({str(r.get("solver")) for r in runs})
    meta_rows = [
        ["study", payload["study"]],
        ["swept parameter", f"`{param}`"],
        ["values", ", ".join(str(v) for v in payload["values"])],
        [
            "fixed overrides",
            ", ".join(f"`{k}` = {v}" for k, v in (payload.get("common_overrides") or {}).items())
            or "—",
        ],
        ["solver(s)", ", ".join(solvers)],
        ["sweep runtime [s]", fmt_num(payload.get("elapsed_s"))],
    ] + [[label, value] for label, value in stamp_rows(payload)]
    head += _table(["", ""], meta_rows) + [""]

    metric_rows = []
    for r in runs:
        m = r.get("metrics") or {}
        peak = m.get("peak") or {}
        spot = m.get("focal_spot") or {}
        metric_rows.append(
            [
                r["index"],
                r["overrides"].get(param, "--"),
                f"`{r['job_hash']}`",
                "FAILED" if r.get("error") else fmt_num(peak.get("p_mpa")),
                fmt_num(peak.get("gain_vs_source")),
                fmt_num((spot.get("axial_6db") or {}).get("width_mm")),
                fmt_num((spot.get("lateral_x_6db") or {}).get("width_mm")),
                fmt_num(peak.get("isppa_w_cm2")),
            ]
        )
    body = [
        "## Runs — focal metrics",
        "",
        *_table(
            [
                "#",
                leaf,
                "job hash",
                "peak [MPa]",
                "gain ×p0",
                "-6 dB axial [mm]",
                "-6 dB lateral x [mm]",
                "I_sppa [W/cm²]",
            ],
            metric_rows,
        ),
        "",
        "## Planner vs actual",
        "",
    ]
    timing_rows = []
    for r in runs:
        exp, act = r.get("expected") or {}, r.get("actual") or {}
        timing_rows.append(
            [
                r["index"],
                r["overrides"].get(param, "--"),
                fmt_num(exp.get("t_expected_s")),
                fmt_num(act.get("elapsed_solve_s")),
                fmt_dev_pct(exp.get("t_expected_s"), act.get("elapsed_solve_s")),
                fmt_num(exp.get("steps_expected")),
                fmt_num(act.get("steps_total")),
                fmt_num(act.get("converged_period")),
            ]
        )
    body += _table(
        [
            "#",
            leaf,
            "expected [s]",
            "measured [s]",
            "deviation",
            "expected steps",
            "actual steps",
            "converged period",
        ],
        timing_rows,
    )
    body += [""]

    scaling = payload.get("scaling") or []
    if scaling:
        body += [
            "## Scaling",
            "",
            f"Peak pressure against `{leaf}`, both normalised to the first run. For a "
            f"linear solver the last column is zero to within discretisation; a "
            f"nonlinear one is expected to fall behind at the high end.",
            "",
            *_table(
                ["run", leaf, f"{leaf} / {leaf}₀", "peak / peak₀", "departure from proportional"],
                scaling,
            ),
            "",
        ]

    if figs:
        body += ["## Figures", ""]
        for name, caption in figs.items():
            body += [f"![{name}]({name})", "", f"*{caption}*", ""]

    notes: list[str] = []
    for r in runs:
        if r.get("error"):
            notes.append(f"run `{r['label']}` FAILED (exit {r.get('exit_code')}): {r['error']}")
        for w in r.get("warnings") or []:
            notes.append(f"run `{r['label']}` resolution warning: {w}")
    if notes:
        body += ["## Caveats", "", *[f"- {n}" for n in notes], ""]

    files = [f"- `{JSON_NAME}` — this report, machine-readable (`{FORMAT}`)"]
    files += [f"- `{name}` — {caption}" for name, caption in figs.items()]
    for r in runs:
        files.append(
            f"- `{_rel(r.get('outdir'), outdir)}` — run `{r['label']}` "
            f"(result, metrics, plan, stamp)"
        )
    body += ["## Files", "", *files, ""]
    return "\n".join(head + body)


def write_sweep_report(sweep: StudySweep, outdir: str | Path, *, figures: bool = True) -> Path:
    """Write the combined ``STUDY.md`` + ``study.json`` (+ figure); returns the folder."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = sweep_payload(sweep)
    figs: dict[str, str] = {}
    if figures:
        try:
            from caustica.study.figures import sweep_figures  # noqa: PLC0415 (matplotlib lazy)

            figs = sweep_figures(payload, outdir)
        except ImportError as exc:
            # A missing matplotlib must cost the picture, not the report —
            # the numbers are all in the tables and in study.json.
            payload.setdefault("notes", []).append(f"no sweep figure: {exc}")
    payload["figures"] = figs
    return _write(outdir, payload, render_sweep_markdown(payload, figs, outdir))
