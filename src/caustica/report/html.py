"""Report rendering: REPORT.md + a self-contained index.html.

Extracted from ``apps/focus_study/report.py`` so the runner's ``caustica
report`` and the focus_study app share one renderer and one set of
metric-row definitions. Both renderers show the SAME numbers from the same
``(section, label, value)`` rows; the HTML exists because a figure-heavy
result set is much easier to read in a browser than in a terminal. No
external assets — the PNGs sit next to the page.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from caustica.report.metrics import A2_PML_MARGIN_WARN_VOX


def fmt(v: Any, unit: str = "", nd: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}g}{unit}"
    return f"{v}{unit}"


Row = tuple[str, str, str]  # (section, label, value)


#: One caption per canonical figure name — the single source both the
#: focus_study report and `caustica report` render from (extra keys are
#: harmless: captions are only looked up for figures actually produced).
FIG_CAPTIONS = {
    "fig_medium.png": "Medium: tissue labels and sound speed in the beam plane.",
    "fig_field.png": "Fundamental pressure amplitude: beam plane (x-z) and focal plane (x-y). "
    "Orange contour = -6 dB; grey dots = source voxels.",
    "fig_profiles.png": "Axial and lateral pressure profiles through the realized peak.",
    "fig_harmonics.png": "Second-harmonic field and its ratio to the fundamental.",
    "fig_convergence.png": "Steady-state settling: per-period peak and relative change.",
    "fig_preview.png": "Quick look from the preview package (peak slices + convergence); "
    "the full field was not read.",
}


# ------------------------------------------------------- shared metric rows


def focus_rows(m: dict) -> list[Row]:
    """The ``Focus`` section, straight from a :func:`focus_metrics` dict."""
    pk = m["peak"]
    pos = pk["position_mm_from_apex"]
    rows: list[Row] = []
    gain = pk["gain_vs_source"]
    rows.append(
        (
            "Focus",
            "peak pressure",
            f"{pk['p_mpa']} MPa (gain x{gain} over the source)"
            if gain is not None
            else f"{pk['p_mpa']} MPa",
        )
    )
    rows.append(
        ("Focus", "peak position (from apex)", f"x={pos['x']}, y={pos['y']}, z={pos['z']} mm")
    )
    if "target" in m:
        tg = m["target"]
        disp = tg["displacement_mm"]
        rows.append(
            (
                "Focus",
                "requested focus",
                f"voxel {tg['voxel_grid']}, hit ratio {fmt(tg['hit_ratio'])}",
            )
        )
        rows.append(
            (
                "Focus",
                "focus displacement",
                f"{tg['displacement_norm_mm']} mm "
                f"(dx={disp['x']}, dy={disp['y']}, dz={disp['z']} mm)",
            )
        )
    rows.append(
        (
            "Focus",
            "I_sppa at peak",
            f"{pk['isppa_w_cm2']:,.1f} W/cm² (plane-wave estimate)"
            if pk["isppa_w_cm2"] is not None
            else "— (medium not available)",
        )
    )
    rows.append(("Focus", "time-domain peak", f"{pk['p_max_time_domain_pa'] / 1e6:.4g} MPa"))
    return rows


def focal_spot_rows(m: dict) -> list[Row]:
    fs = m["focal_spot"]
    return [
        ("Focal spot", "-6 dB axial length", fmt(fs["axial_6db"]["width_mm"], " mm")),
        ("Focal spot", "-6 dB lateral (x)", fmt(fs["lateral_x_6db"]["width_mm"], " mm")),
        ("Focal spot", "-6 dB lateral (y)", fmt(fs["lateral_y_6db"]["width_mm"], " mm")),
        ("Focal spot", "volume above -6 dB", f"{fs['volume_above_6db_mm3']} mm³"),
    ]


def run_rows(m: dict, wall_time: str | None = None) -> list[Row]:
    run = m["run"]
    rows: list[Row] = [
        (
            "Run",
            "time step / steps per period",
            f"{run['dt_s'] * 1e9:.2f} ns / {run['steps_per_period']}",
        ),
        ("Run", "steps taken", f"{run['steps_total']:,} (t_end = {run['t_end_us']} µs)"),
        (
            "Run",
            "settling",
            f"converged at period {run['converged_period']} "
            f"(time-of-flight {run['tof_periods']}); "
            f"cap hit: {fmt(run['settle_capped'])}",
        ),
    ]
    if wall_time is not None:
        rows.append(("Run", "wall time", wall_time))
    return rows


def harmonics_rows(m: dict) -> list[Row]:
    if "harmonics" not in m:
        return []
    h = m["harmonics"]
    # The A2 MAXIMUM is a whole-interior argmax, so it can land on the PML's
    # own harmonic residue. The number is never edited (contract stability) —
    # the row says where it sits instead (janitor ticket 09). Absent in a
    # metrics.json written before that field existed: no distance, no caveat.
    edge = h.get("a2_peak_distance_to_pml_vox")
    caveat = (
        f" — {edge} voxels from the PML edge: likely an edge artifact, "
        f"read A2 at the fundamental peak instead"
        if edge is not None and edge < A2_PML_MARGIN_WARN_VOX
        else ""
    )
    return [
        (
            "Harmonics",
            "A2 at the fundamental peak",
            f"{h['a2_at_fundamental_peak_pa'] / 1e6:.4g} MPa",
        ),
        ("Harmonics", "A2 / A1 at the peak", f"{h['a2_over_a1_at_peak_pct']} %"),
        (
            "Harmonics",
            "A2 maximum",
            f"{h['a2_peak_pa'] / 1e6:.4g} MPa at voxel {h['a2_peak_voxel_grid']}{caveat}",
        ),
    ]


# ------------------------------------------------------------- renderers


def render_markdown(
    path: Path,
    *,
    title: str,
    description: str,
    lede: str,
    rows: Sequence[Row],
    notes: Sequence[str] = (),
    figs: Sequence[str] = (),
    captions: dict[str, str] | None = None,
    files_lines: Sequence[str] = (),
) -> Path:
    captions = captions or {}
    lines = [f"# {title}", "", description, "", lede, ""]
    section = None
    for sec, label, value in rows:
        if sec != section:
            if section is not None:
                lines.append("")
            section = sec
            lines += [f"## {sec}", "", "| | |", "|---|---|"]
        lines.append(f"| {label} | {value} |")
    lines.append("")
    if notes:
        lines += ["## Caveats", ""] + [f"- {n}" for n in notes] + [""]
    if figs:
        lines += ["## Figures", ""]
        for f in figs:
            lines += [f"![{f}]({f})", "", f"*{captions.get(f, f)}*", ""]
    if files_lines:
        lines += ["## Files", ""] + list(files_lines) + [""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


CSS = """
:root { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e2e1dd; --accent:#2a78d6;
        --card:#ffffff; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141414; --ink:#f2f1ee; --ink2:#a8a6a1; --line:#2e2d2b; --accent:#7fb3f0;
          --card:#1c1c1b; } }
* { box-sizing: border-box; }
body { margin:0; padding:2.2rem 1.2rem 4rem; background:var(--bg); color:var(--ink);
       font:15px/1.6 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif; }
main { max-width: 1080px; margin: 0 auto; }
h1 { font-size:1.65rem; margin:0 0 .3rem; letter-spacing:-.01em; }
h2 { font-size:1.05rem; margin:2.4rem 0 .7rem; color:var(--ink);
     border-bottom:1px solid var(--line); padding-bottom:.35rem; }
p.lede { color:var(--ink2); margin:.2rem 0 1.6rem; max-width:70ch; }
table { border-collapse:collapse; width:100%; margin:0 0 .4rem; font-size:14px; }
td { padding:.42rem .6rem; border-bottom:1px solid var(--line); vertical-align:top; }
td:first-child { color:var(--ink2); width:34%; }
td:last-child { font-variant-numeric: tabular-nums; }
figure { margin:1.4rem 0; background:var(--card); border:1px solid var(--line);
         border-radius:10px; padding:.8rem; overflow-x:auto; }
figure img { display:block; width:100%; height:auto; border-radius:4px; }
figcaption { color:var(--ink2); font-size:13px; margin-top:.6rem; }
ul { color:var(--ink2); max-width:75ch; }
code { background:var(--card); border:1px solid var(--line); border-radius:4px;
       padding:.05rem .3rem; font-size:13px; }
.foot { margin-top:3rem; color:var(--ink2); font-size:13px; }
"""


def render_html(
    path: Path,
    *,
    page_title: str,
    title: str,
    description: str,
    rows: Sequence[Row],
    notes: Sequence[str] = (),
    figs: Sequence[str] = (),
    captions: dict[str, str] | None = None,
    foot_html: str = "",
) -> Path:
    captions = captions or {}
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{page_title}</title><style>{CSS}</style></head><body><main>",
        f"<h1>{title}</h1>",
        f"<p class='lede'>{description}</p>",
    ]
    section = None
    for sec, label, value in rows:
        if sec != section:
            if section is not None:
                parts.append("</table>")
            section = sec
            parts.append(f"<h2>{sec}</h2><table>")
        parts.append(f"<tr><td>{label}</td><td>{value}</td></tr>")
    if section is not None:
        parts.append("</table>")
    if notes:
        items = "".join(f"<li>{n}</li>" for n in notes)
        parts.append(f"<h2>Caveats</h2><ul>{items}</ul>")
    if figs:
        parts.append("<h2>Figures</h2>")
        for f in figs:
            parts.append(
                f"<figure><img src='{f}' alt='{f}'>"
                f"<figcaption>{captions.get(f, f)}</figcaption></figure>"
            )
    if foot_html:
        parts.append(foot_html)
    parts.append("</main></body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
