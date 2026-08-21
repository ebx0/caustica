"""Figure set for one CW solve (M10d — extracted from ``apps/focus_study``).

Style follows the project's dataviz rules (as in scripts/gen_validation_report.py):
a sequential single-hue ramp for magnitude fields, a diverging ramp for signed
quantities, a fixed categorical order for lines, thin marks, recessive grid,
no rainbow colormaps.

matplotlib is NOT a caustica dependency — this module imports it at module
load, so importing ``caustica.report.figures`` (or calling ``caustica report``)
without matplotlib fails with an actionable message while the rest of
``caustica.report`` (metrics, preview) stays importable on a bare install.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import matplotlib
except ImportError as exc:  # pragma: no cover - matplotlib present in dev env
    raise ImportError(
        "caustica report figures need matplotlib: pip install matplotlib "
        "(or `pip install caustica[report]`)"
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
)

from caustica.report.metrics import (  # noqa: E402
    HALF_PRESSURE,
    interior_slices,
    mm_axes,
    region_origin,
)
from caustica.solvers.base import SolverResult  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQ_RAMP = [
    "#fcfcfb",
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
CMAP_SEQ = LinearSegmentedColormap.from_list("caustica_seq", SEQ_RAMP)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 190,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.color": "#e8e7e3",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    }
)


@dataclass
class FigureContext:
    """The geometry/labeling a figure needs, decoupled from any app's Setup.

    focus_study builds one from its ``Setup``; ``caustica report`` builds one
    from the ``caustica-result/1`` attrs. Optional fields simply drop the
    corresponding decoration (no source dots, no medium figure) instead of
    failing.
    """

    dx: float
    grid_shape: tuple[int, ...]
    pml_vox: int
    apex_vox: tuple[int, int, int]
    title: str
    solver: str = ""
    focus_vox: tuple[int, int, int] | None = None
    source_indices: np.ndarray | None = None  # (n, 3) grid-frame voxels
    labels: np.ndarray | None = None  # integer label map (full grid)
    label_names: dict[int, str] = field(default_factory=dict)
    sound_speed: np.ndarray | None = None  # full-grid c map, for the medium figure


def _save(fig, outdir: Path, name: str) -> str:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    return f"{name}.png"


def _extent(h_mm: np.ndarray, v_mm: np.ndarray) -> list[float]:
    """imshow extent for an (n_v, n_h) image drawn as v (rows) vs h (cols)."""
    return [float(h_mm[0]), float(h_mm[-1]), float(v_mm[-1]), float(v_mm[0])]


def _mm_axes(ctx: FigureContext, result: SolverResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return mm_axes(result, dx=ctx.dx, apex_vox=ctx.apex_vox)


def field_maps(ctx: FigureContext, result: SolverResult, prof: dict, outdir: Path) -> str:
    """Fundamental |p| through the beam axis (x-z) and across it (x-y)."""
    amp = result.amp / 1e6  # MPa
    x, y, z = _mm_axes(ctx, result)
    pk = tuple(int(i) for i in prof["peak_idx"])
    vmax = float(amp[pk])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), width_ratios=[1.55, 1.0])
    im = axes[0].imshow(
        amp[:, pk[1], :], cmap=CMAP_SEQ, extent=_extent(z, x), vmin=0.0, vmax=vmax, aspect="equal"
    )
    axes[0].contour(
        z, x, amp[:, pk[1], :], levels=[HALF_PRESSURE * vmax], colors=[CAT[1]], linewidths=1.0
    )
    if ctx.source_indices is not None:
        src = np.asarray(ctx.source_indices)
        sel = np.abs(src[:, 1] - (pk[1] + region_origin(result)[1])) <= 1
        axes[0].plot(
            (src[sel, 2] - ctx.apex_vox[2]) * ctx.dx * 1e3,
            (src[sel, 0] - ctx.apex_vox[0]) * ctx.dx * 1e3,
            ".",
            ms=1.6,
            color=INK2,
            alpha=0.75,
        )
    if ctx.focus_vox is not None:
        tz = (ctx.focus_vox[2] - ctx.apex_vox[2]) * ctx.dx * 1e3
        tx = (ctx.focus_vox[0] - ctx.apex_vox[0]) * ctx.dx * 1e3
        axes[0].plot([tz], [tx], "+", ms=11, mew=1.4, color=CAT[1], label="requested focus")
    axes[0].plot([z[pk[2]]], [x[pk[0]]], "x", ms=9, mew=1.4, color=INK, label="realized peak")
    axes[0].set(xlabel="z from apex [mm]", ylabel="x [mm]", title="|p| at f0 — beam plane (x-z)")
    axes[0].grid(False)
    axes[0].legend(frameon=False, loc="upper left")
    fig.colorbar(im, ax=axes[0], shrink=0.82, pad=0.02, label="MPa")

    im2 = axes[1].imshow(
        amp[:, :, pk[2]], cmap=CMAP_SEQ, extent=_extent(y, x), vmin=0.0, vmax=vmax, aspect="equal"
    )
    axes[1].contour(
        y, x, amp[:, :, pk[2]], levels=[HALF_PRESSURE * vmax], colors=[CAT[1]], linewidths=1.0
    )
    axes[1].set(xlabel="y [mm]", ylabel="x [mm]", title=f"focal plane (x-y) at z={z[pk[2]]:.1f} mm")
    axes[1].grid(False)
    fig.colorbar(im2, ax=axes[1], shrink=0.82, pad=0.02, label="MPa")
    fig.suptitle(ctx.title, y=1.02, fontsize=11)
    return _save(fig, outdir, "fig_field")


def profile_plot(ctx: FigureContext, prof: dict, outdir: Path) -> str:
    """Axial and lateral profiles, with the analytic reference when it exists."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.8))
    z, ax_p = prof["z_mm"], prof["axial"] / 1e6
    axes[0].plot(z, ax_p, color=CAT[0], label=f"caustica {ctx.solver}")
    if "axial_oneill" in prof:
        on = prof["axial_oneill"]
        scaled = on / on.max() * ax_p.max()
        axes[0].plot(z, scaled, color=INK2, ls=":", lw=1.5, label="O'Neil (scaled to peak)")
    if "axial_h2" in prof:
        axes[0].plot(z, prof["axial_h2"] / 1e6, color=CAT[2], lw=1.3, label="2nd harmonic")
    if ctx.focus_vox is not None:
        axes[0].axvline(
            (ctx.focus_vox[2] - ctx.apex_vox[2]) * ctx.dx * 1e3,
            color=CAT[1],
            lw=1.0,
            ls="--",
            label="geometric focus",
        )
    pml_mm = float(prof["pml_mm"][0])
    axes[0].axvspan(z[-1] - pml_mm, z[-1], color=INK2, alpha=0.08, lw=0, label="PML")
    axes[0].set(xlabel="z from apex [mm]", ylabel="|p| [MPa]", title="Axial profile through peak")
    axes[0].legend(frameon=False)

    x, lat = prof["x_mm"], prof["lateral"] / 1e6
    axes[1].plot(x, lat, color=CAT[0])
    axes[1].axhline(HALF_PRESSURE * lat.max(), color=CAT[1], lw=1.0, ls="--", label="-6 dB")
    axes[1].set(xlabel="x [mm]", ylabel="|p| [MPa]", title="Lateral profile at the focal plane")
    axes[1].legend(frameon=False)
    return _save(fig, outdir, "fig_profiles")


def harmonic_plot(ctx: FigureContext, result: SolverResult, prof: dict, outdir: Path) -> str | None:
    """Second-harmonic field and its ratio to the fundamental along the axis."""
    if 2 not in result.phasors:
        return None
    a1, a2 = result.amp, result.harmonic_amp(2)
    x, y, z = _mm_axes(ctx, result)
    pk = tuple(int(i) for i in prof["peak_idx"])
    interior = interior_slices(result, grid_shape=ctx.grid_shape, pml_vox=ctx.pml_vox)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.9), width_ratios=[1.3, 1.0])
    # Scale to the INTERIOR maximum: the sponge edge can hold a numerical
    # 2f0 artefact that would otherwise set the color scale.
    vmax2 = float(a2[interior].max()) / 1e6
    im = axes[0].imshow(
        a2[:, pk[1], :] / 1e6,
        cmap=CMAP_SEQ,
        extent=_extent(z, x),
        vmin=0.0,
        vmax=vmax2,
        aspect="equal",
    )
    axes[0].set(xlabel="z from apex [mm]", ylabel="x [mm]", title="|p| at 2f0 — beam plane")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], shrink=0.82, pad=0.02, label="MPa")

    ratio = 100.0 * a2[pk[0], pk[1], :] / np.maximum(a1[pk[0], pk[1], :], 1e-9)
    # A ratio is only meaningful where BOTH the field is strong and the
    # sponge is not eating the fundamental — inside the PML it diverges.
    z_int = interior[2]
    keep = a1[pk[0], pk[1], :] > 0.02 * float(a1[tuple(pk)])
    keep[: z_int.start] = False
    keep[z_int.stop :] = False
    axes[1].plot(z[keep], ratio[keep], color=CAT[2])
    axes[1].axvline(z[pk[2]], color=INK2, lw=1.0, ls=":", label="focus")
    axes[1].set(
        xlabel="z from apex [mm]",
        ylabel="A2 / A1 [%]",
        title="Harmonic content along the axis (|p| > 2% of max)",
    )
    axes[1].legend(frameon=False)
    return _save(fig, outdir, "fig_harmonics")


def convergence_plot(result: SolverResult, outdir: Path) -> str | None:
    """Steady-state settling history — the solver's own honesty check."""
    if not result.convergence_history:
        return None
    hist = np.array(result.convergence_history, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.4))
    axes[0].plot(hist[:, 0], hist[:, 1] / 1e6, color=CAT[0], marker="o", ms=2.5)
    axes[0].axvline(result.tof_periods, color=INK2, ls=":", lw=1.0, label="time-of-flight")
    axes[0].set(xlabel="period", ylabel="peak |p| [MPa]", title="Per-period peak")
    axes[0].legend(frameon=False)
    axes[1].semilogy(hist[:, 0], np.maximum(hist[:, 2], 1e-12), color=CAT[1], marker="o", ms=2.5)
    axes[1].set(
        xlabel="period",
        ylabel="relative change",
        title=f"Convergence (stopped at period {result.converged_period})",
    )
    return _save(fig, outdir, "fig_convergence")


def medium_plot(ctx: FigureContext, outdir: Path) -> str | None:
    """Label map + sound speed, for heterogeneous scenarios only."""
    if ctx.labels is None:
        return None
    dx, apex = ctx.dx, ctx.apex_vox
    lab = ctx.labels[:, apex[1], :]
    x = (np.arange(ctx.grid_shape[0]) - apex[0]) * dx * 1e3
    z = (np.arange(ctx.grid_shape[2]) - apex[2]) * dx * 1e3
    ids = sorted(ctx.label_names)
    cmap = ListedColormap([CAT[i % len(CAT)] for i in range(len(ids))])
    norm = BoundaryNorm([*[i - 0.5 for i in range(len(ids))], len(ids) - 0.5], cmap.N)
    remap = np.zeros_like(lab)
    for k, i in enumerate(ids):
        remap[lab == i] = k

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.9))
    axes[0].imshow(remap, cmap=cmap, norm=norm, extent=_extent(z, x), aspect="equal")
    axes[0].set(xlabel="z from apex [mm]", ylabel="x [mm]", title="Tissue labels (beam plane)")
    axes[0].grid(False)
    handles = [
        plt.Line2D([], [], marker="s", ls="", color=cmap(k), label=f"{i}: {ctx.label_names[i]}")
        for k, i in enumerate(ids)
    ]
    axes[0].legend(handles=handles, frameon=False, loc="upper right", fontsize=7)

    if ctx.sound_speed is None:
        return _save(fig, outdir, "fig_medium")
    c_map = ctx.sound_speed[:, apex[1], :]
    # Keep the slowest material visibly tinted: the ramp starts at the page
    # colour, so vmin=c.min() would render that tissue as blank paper.
    lo, hi = float(c_map.min()), float(c_map.max())
    im = axes[1].imshow(
        c_map,
        cmap=CMAP_SEQ,
        extent=_extent(z, x),
        aspect="equal",
        vmin=lo - 0.12 * (hi - lo or 1.0),
        vmax=hi,
    )
    axes[1].set(xlabel="z from apex [mm]", ylabel="x [mm]", title="Sound speed")
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], shrink=0.82, pad=0.02, label="m/s")
    return _save(fig, outdir, "fig_medium")


def make_all(ctx: FigureContext, result: SolverResult, prof: dict, outdir: Path) -> list[str]:
    figs = [
        medium_plot(ctx, outdir),
        field_maps(ctx, result, prof, outdir),
        profile_plot(ctx, prof, outdir),
        harmonic_plot(ctx, result, prof, outdir),
        convergence_plot(result, outdir),
    ]
    return [f for f in figs if f]


def preview_quicklook(pre: dict, outdir: Path) -> str:
    """One figure straight from a preview package — no result.h5 needed.

    Beam plane + focal plane from the stored peak slices, the axial profile
    read out of the beam-plane slice, and the convergence history.
    """
    meta = pre["meta"]
    beam = pre["amp_h1_slice_y"] / 1e6  # (x, z) plane through the peak
    focal = pre["amp_h1_slice_z"] / 1e6  # (x, y) plane at the peak
    x, y, z = pre["x_mm"], pre["y_mm"], pre["z_mm"]
    pk = [
        int(i) - int(o) for i, o in zip(meta["peak_voxel_grid"], meta["region_start"], strict=True)
    ]
    vmax = float(beam.max())

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), width_ratios=[1.55, 1.0])
    im = axes[0, 0].imshow(
        beam, cmap=CMAP_SEQ, extent=_extent(z, x), vmin=0.0, vmax=vmax, aspect="equal"
    )
    axes[0, 0].plot([z[pk[2]]], [x[pk[0]]], "x", ms=9, mew=1.4, color=INK, label="peak")
    axes[0, 0].set(
        xlabel="z from apex [mm]", ylabel="x [mm]", title="|p| at f0 — beam plane (preview)"
    )
    axes[0, 0].grid(False)
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.82, pad=0.02, label="MPa")

    im2 = axes[0, 1].imshow(
        focal, cmap=CMAP_SEQ, extent=_extent(y, x), vmin=0.0, vmax=vmax, aspect="equal"
    )
    axes[0, 1].set(xlabel="y [mm]", ylabel="x [mm]", title=f"focal plane at z={z[pk[2]]:.1f} mm")
    axes[0, 1].grid(False)
    fig.colorbar(im2, ax=axes[0, 1], shrink=0.82, pad=0.02, label="MPa")

    axes[1, 0].plot(z, beam[pk[0], :], color=CAT[0])
    axes[1, 0].set(
        xlabel="z from apex [mm]", ylabel="|p| [MPa]", title="Axial profile through peak"
    )

    hist = np.asarray(pre.get("convergence_history", np.empty((0, 3))), dtype=float)
    if hist.size:
        axes[1, 1].semilogy(
            hist[:, 0], np.maximum(hist[:, 2], 1e-12), color=CAT[1], marker="o", ms=2.5
        )
        axes[1, 1].set(xlabel="period", ylabel="relative change", title="Convergence")
    else:
        axes[1, 1].axis("off")
    return _save(fig, outdir, "fig_preview")
