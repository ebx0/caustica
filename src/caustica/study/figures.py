"""The combined sweep figure — the one picture a sweep exists to produce.

Style is NOT redefined here. Importing :mod:`caustica.report.figures` applies
the project's dataviz rcParams (surface, ink, categorical order, thin marks,
recessive grid) and hands over the same ``_save`` the run figures use, so a
sweep panel and a field map come out of the same design system. matplotlib is
therefore a hard requirement of this module, exactly as it is of
``caustica.report.figures`` — which is why the study report imports it lazily
and degrades to a note when it is missing.

Captions travel WITH the figures: :func:`sweep_figures` returns
``{filename: caption}``, and the report renders both the image list and the
caption list from that one mapping. A caption cannot drift from the figure it
describes if there is only one place that states it (the M10d rule).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from caustica.report import figures as hfig

plt = hfig.plt

#: File name of the combined figure. One constant, used by the writer and
#: quoted in the caption mapping it returns.
SWEEP_FIG = "fig_sweep"


def _numeric(values: list[Any]) -> list[float] | None:
    """The values as floats, or ``None`` if any of them is not a number.

    A sweep over solver names is legitimate and must still draw; it just
    gets categorical positions and no proportionality reference.
    """
    out: list[float] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        out.append(float(v))
    return out


def _series(payload: dict) -> dict:
    """Everything the panels plot, pulled out of the report payload once."""
    runs = payload.get("runs", [])
    values = list(payload.get("values", []))
    peaks: list[float | None] = []
    expected: list[float | None] = []
    actual: list[float | None] = []
    for r in runs:
        m = r.get("metrics") or {}
        peak = (m.get("peak") or {}).get("p_mpa")
        peaks.append(None if peak is None else float(peak))
        exp = (r.get("expected") or {}).get("t_expected_s")
        expected.append(None if exp is None else float(exp))
        act = (r.get("actual") or {}).get("elapsed_solve_s")
        actual.append(None if act is None else float(act))
    return {
        "values": values,
        "labels": [str(r.get("label", i)) for i, r in enumerate(runs)],
        "peaks": peaks,
        "expected": expected,
        "actual": actual,
    }


def _plot_points(ax, x, ys, color, label, marker="o"):
    """Plot only the points that exist; returns whether anything was drawn."""
    pairs = [(xi, yi) for xi, yi in zip(x, ys, strict=True) if yi is not None]
    if not pairs:
        return False
    ax.plot(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        marker=marker,
        ms=5,
        lw=1.4,
        color=color,
        label=label,
    )
    return True


def sweep_figures(payload: dict, outdir: Path) -> dict[str, str]:
    """Write the combined figure into ``outdir``; return ``{name: caption}``.

    Two panels, because a sweep asks two questions: what did the physics do
    (peak focal pressure per value, against strict proportionality — the
    expectation a linear solver must meet) and was the planner right (its
    estimate against the measured solve time, per value).
    """
    s = _series(payload)
    param = payload.get("param_path", "parameter")
    nums = _numeric(s["values"])
    x = nums if nums is not None else list(range(len(s["values"])))
    xlabel = param if nums is not None else f"{param} (categorical)"

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    drew = _plot_points(ax, x, s["peaks"], hfig.CAT[0], "peak |p| at f0")
    ref_drawn = False
    if nums is not None and drew:
        first = next(
            ((xi, yi) for xi, yi in zip(x, s["peaks"], strict=True) if yi is not None), None
        )
        if first is not None and first[0] != 0.0:
            slope = first[1] / first[0]
            xs = np.asarray(sorted(x), dtype=float)
            ax.plot(
                xs,
                slope * xs,
                ls="--",
                lw=1.1,
                color=hfig.INK2,
                label="strict proportionality",
            )
            ref_drawn = True
    ax.set(xlabel=xlabel, ylabel="peak |p| [MPa]", title="Focal pressure across the sweep")
    if not nums:
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in s["values"]], rotation=20, ha="right")
    ax.legend(frameon=False, loc="best")

    ax = axes[1]
    any_time = _plot_points(ax, x, s["expected"], hfig.CAT[1], "planner expected", marker="s")
    any_time |= _plot_points(ax, x, s["actual"], hfig.CAT[2], "measured solve")
    ax.set(xlabel=xlabel, ylabel="solve time [s]", title="Planner estimate vs measured")
    if not nums:
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in s["values"]], rotation=20, ha="right")
    if any_time:
        ax.legend(frameon=False, loc="best")
    else:
        ax.text(0.5, 0.5, "no timings recorded", ha="center", va="center", color=hfig.INK2)

    name = hfig._save(fig, Path(outdir), SWEEP_FIG)
    caption = (
        f"Sweep of `{param}`: peak fundamental pressure at each value (left"
        + (
            "; dashed line = strict proportionality through the first point, which a "
            "linear solver must follow"
            if ref_drawn
            else ""
        )
        + ") and the planner's estimate against the measured solve time (right)."
    )
    return {name: caption}
