"""``caustica report <out-dir>``: render a runner output folder locally.

Two speeds, one command:

* ``result.h5`` present  -> full figures (field maps, profiles, harmonics,
  convergence) from the stored fields, plus REPORT.md + index.html.
* only ``preview.npz``   -> a quick look from the <=10 MB package alone —
  the point of the preview: judge a Colab run before the full field syncs.

``metrics.json`` is trusted when present (the runner computed it WITH the
medium in memory, so it carries I_sppa); when absent it is recomputed from
the result file — minus the medium-dependent numbers, which a result file
honestly cannot provide.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from caustica.io.atomic import atomic_write
from caustica.report import html as hrep
from caustica.report.html import FIG_CAPTIONS
from caustica.report.metrics import FieldFrame, axial_profiles, focus_metrics


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _header_rows(name: str, meta: dict | None, geo: dict | None) -> list[hrep.Row]:
    rows: list[hrep.Row] = [("Job", "name", name)]
    if meta:
        rows += [
            ("Job", "solver / backend", f"{meta.get('solver', '?')} / {meta.get('backend', '?')}"),
            ("Job", "generated", str(meta.get("generated", "?"))),
            ("Job", "git commit", str(meta.get("git_commit", "?"))[:12]),
        ]
        env = meta.get("environment", {})
        if env.get("gpu_name"):
            rows.append(("Job", "GPU", str(env["gpu_name"])))
    elif geo:
        rows.append(("Job", "solver / backend", f"{geo['solver']} / {geo['backend']}"))
    if geo:
        shape = geo["grid_shape"]
        n_vox = int(np.prod(shape))
        rows.append(
            (
                "Job",
                "grid",
                f"{'x'.join(map(str, shape))} @ dx={geo['dx'] * 1e3:g} mm ({n_vox:,} voxels), "
                f"PML {geo['pml_vox']} vox",
            )
        )
        rows.append(
            (
                "Job",
                "drive",
                f"{geo['f0_hz'] / 1e6:g} MHz CW, {geo['amplitude_pa'] / 1e3:g} kPa at the source",
            )
        )
    return rows


def _metric_rows(m: dict, wall_time: str | None) -> list[hrep.Row]:
    rows = hrep.focus_rows(m) + hrep.focal_spot_rows(m)
    rows += hrep.run_rows(m, wall_time=wall_time)
    rows += hrep.harmonics_rows(m)
    return rows


def _wall_time(meta: dict | None) -> str | None:
    if not meta:
        return None
    actual = meta.get("actual") or {}
    t = actual.get("elapsed_solve_s")
    return None if t is None else f"{t:.1f} s (solve, from run_meta.json)"


def _with_run_warnings(description: str, meta: dict | None) -> str:
    """Prepend the run's ppw warnings to the report HEAD.

    A reader who opens only the report must meet the resolution warning
    before any number — it cannot live in a footnote."""
    warns = (meta or {}).get("ppw_warnings") or []
    if not warns:
        return description
    banner = "\n".join(f"⚠ **WARNING:** {w}" for w in warns)
    return f"{banner}\n\n{description}"


def report_out_dir(outdir: str | Path, *, preview_only: bool = False) -> Path:
    """Render REPORT.md + index.html (+ figures) into ``outdir``; returns the html."""
    outdir = Path(outdir)
    if not outdir.is_dir():
        raise FileNotFoundError(f"not a directory: {outdir}")
    result_path = outdir / "result.h5"
    preview_path = outdir / "preview.npz"
    metrics = _read_json(outdir / "metrics.json")
    meta = _read_json(outdir / "run_meta.json")

    if result_path.exists() and not preview_only:
        return _full_report(outdir, result_path, metrics, meta)
    if preview_path.exists():
        return _preview_report(outdir, preview_path, metrics, meta)
    if preview_only:
        raise FileNotFoundError(
            f"--preview requested but there is no preview.npz in {outdir}"
            + (" (result.h5 exists - run without --preview)" if result_path.exists() else "")
        )
    raise FileNotFoundError(f"nothing to report in {outdir}: no result.h5 and no preview.npz")


def _full_report(outdir: Path, result_path: Path, metrics: dict | None, meta: dict | None) -> Path:
    from caustica.io.store import load_result  # noqa: PLC0415 (h5py lazy)
    from caustica.report import figures as hfig  # noqa: PLC0415 (matplotlib lazy)

    # ONE open for the fields and the geometry: the parsing of the result
    # attrs belongs to the module that writes them (janitor ticket 02).
    result, geo = load_result(result_path, with_geometry=True)
    frame = FieldFrame.from_geometry(geo)
    name = geo["job_name"] or (metrics or {}).get("job") or outdir.name
    notes: list[str] = []
    if metrics is None:
        m = focus_metrics(
            result,
            frame,
            source_amplitude=geo["amplitude_pa"],
            medium=None,
        )
        metrics = {"format": "caustica-metrics/1", "job": name, **m}
        with atomic_write(outdir / "metrics.json") as tmp:
            tmp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        notes.append(
            "metrics.json was recomputed from result.h5; I_sppa needs the medium "
            "(rho, c at the peak) and is therefore not available."
        )
        if not geo["apex_known"]:
            notes.append(
                "the result file predates the apex_vox stamp — mm positions are "
                "measured from the grid origin, not the transducer apex."
            )

    ctx = hfig.FigureContext(
        frame=frame,
        title=f"caustica run — {name}",
        solver=geo["solver"] or metrics.get("run", {}).get("solver", ""),
        source_indices=geo["source_indices"],
    )
    prof = axial_profiles(result, frame)
    figs = hfig.make_all(ctx, result, prof, outdir)

    rows = _header_rows(name, meta, geo) + _metric_rows(metrics, _wall_time(meta))
    description = _with_run_warnings(
        f"Result of job `{name}` under the caustica-result/1 contract "
        f"({geo['solver']} solver, {geo['backend']} backend).",
        meta,
    )
    files_lines = [
        "- `result.h5` — full fields (caustica-result/1)",
        "- `metrics.json` — every number above, machine-readable",
        "- `preview.npz` — <=10 MB quick-look package (peak slices + coarse volume)",
        "- `run_meta.json` — environment, git commit, planner-vs-actual",
    ]
    hrep.render_markdown(
        outdir / "REPORT.md",
        title=f"caustica run — {name}",
        description=description,
        lede=f"*Generated by `caustica report` from `{result_path.name}`.*",
        rows=rows,
        notes=tuple(notes),
        figs=figs,
        captions=FIG_CAPTIONS,
        files_lines=files_lines,
    )
    return hrep.render_html(
        outdir / "index.html",
        page_title=f"caustica — {name}",
        title=f"caustica run — {name}",
        description=description,
        rows=rows,
        notes=tuple(notes),
        figs=figs,
        captions=FIG_CAPTIONS,
        foot_html=(
            "<p class='foot'>Generated by <code>caustica report</code>. Raw fields in "
            "<code>result.h5</code>, all numbers in <code>metrics.json</code>.</p>"
        ),
    )


def _preview_report(
    outdir: Path, preview_path: Path, metrics: dict | None, meta: dict | None
) -> Path:
    from caustica.report import figures as hfig  # noqa: PLC0415 (matplotlib lazy)
    from caustica.report.preview import load_preview  # noqa: PLC0415

    pre = load_preview(preview_path)
    name = (metrics or {}).get("job") or (meta or {}).get("job") or outdir.name
    figs = [hfig.preview_quicklook(pre, outdir)]
    notes = (
        "Rendered from the preview package only — the full result.h5 was not "
        "available. Slices pass through the realized peak; the coarse volume "
        f"is block-averaged (step {pre['meta']['coarse_step']}).",
    )
    rows = _header_rows(name, meta, None)
    if metrics is not None and "peak" in metrics:
        rows += _metric_rows(metrics, _wall_time(meta))
    description = _with_run_warnings(
        f"Quick look at job `{name}` from its preview package (result.h5 not read).", meta
    )
    hrep.render_markdown(
        outdir / "REPORT.md",
        title=f"caustica run — {name} (preview)",
        description=description,
        lede=f"*Generated by `caustica report` from `{preview_path.name}` only.*",
        rows=rows,
        notes=notes,
        figs=figs,
        captions=FIG_CAPTIONS,
        files_lines=["- `preview.npz` — quick-look package", "- `metrics.json` — metrics"],
    )
    return hrep.render_html(
        outdir / "index.html",
        page_title=f"caustica — {name} (preview)",
        title=f"caustica run — {name} (preview)",
        description=description,
        rows=rows,
        notes=notes,
        figs=figs,
        captions=FIG_CAPTIONS,
        foot_html=(
            "<p class='foot'>Generated by <code>caustica report</code> from the preview "
            "package. Numbers in <code>metrics.json</code>.</p>"
        ),
    )
