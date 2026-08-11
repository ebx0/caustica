"""Validation-report generator: runs scenarios, writes PNG figures + metrics.

Usage:
    python scripts/gen_validation_report.py --outdir benchmarks/reports/<date> \
        --scenario all | kwave2d_linear | kwave2d_hetero | kwave2d_nonlinear |
                   kwave3d_bowl | oneill3d | fubini | planewave | spiral

Each scenario is independent; kwave scenarios must NOT start within the same
wall-clock second (k-wave-python names its temp .h5 by timestamp). Each writes
its figure(s) and a ``metrics_<name>.json`` fragment into --outdir. A final
``--assemble`` pass merges fragments into metrics.json + REPORT.md.

Figure style follows the project dataviz rules: sequential single-hue ramp
for magnitude fields, diverging blue/red for signed differences, categorical
colors in fixed order, thin marks, recessive grid, no rainbow colormaps.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import hifusim.solvers as solvers
from hifusim import Grid, Medium, PMLSpec
from hifusim.analytic import (
    axial_pressure,
    fubini_harmonic,
    rayleigh_pressure,
    shock_distance,
    spherical_cap_points,
)
from hifusim.arrays import archimedean_spiral
from hifusim.materials import Material, water
from hifusim.medium import Medium as Med
from hifusim.solvers import CWRunSpec
from hifusim.sources import CWSource, bowl_cw_source, plane_cw_source

# ---------------------------------------------------------------------------
# Style (validated reference palette; see docs/devlog.md session 5)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed order
SEQ_RAMP = [
    "#fcfcfb", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
    "#104281", "#0d366b",
]
CMAP_SEQ = LinearSegmentedColormap.from_list("hifusim_seq", SEQ_RAMP)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "hifusim_div", ["#2a78d6", "#f0efec", "#e34948"]
)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.grid": True, "grid.color": "#e8e7e3",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

F0, C0 = 1.0e6, 1500.0


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def field_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Normalized-field agreement metrics between two amplitude fields."""
    an = a / a.max()
    bn = b / b.max()
    rel_l2 = float(np.linalg.norm(an - bn) / np.linalg.norm(bn))
    r = float(np.corrcoef(an.ravel(), bn.ravel())[0, 1])
    pa = np.unravel_index(np.argmax(an), an.shape)
    pb = np.unravel_index(np.argmax(bn), bn.shape)
    return {
        "rel_l2_pct": round(rel_l2 * 100, 2),
        "pearson_r": round(r, 5),
        "peak_offset_vox": [int(x - y) for x, y in zip(pa, pb, strict=True)],
        "peak_amp_ratio": round(float(a.max() / b.max()), 4),
    }


def save_fragment(outdir: Path, name: str, metrics: dict, elapsed: float) -> None:
    metrics = {"scenario": name, "elapsed_s": round(elapsed, 1), **metrics}
    (outdir / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2))
    print(f"[{name}] done in {elapsed:.1f}s")


def savefig(fig, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def imshow_mm(ax, img, dx_mm, cmap=CMAP_SEQ, vmax=None, vmin=0.0):
    nh, nv = img.shape
    ext = [0, nv * dx_mm, nh * dx_mm, 0]
    im = ax.imshow(img, cmap=cmap, extent=ext, vmax=vmax, vmin=vmin, aspect="equal")
    ax.grid(False)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("x [mm]")
    return im


def _disc_source_2d(center, radius_vox, amplitude, f0=F0):
    o = np.arange(-radius_vox, radius_vox + 1)
    mx, my = np.meshgrid(o, o, indexing="ij")
    m = mx**2 + my**2 <= radius_vox**2
    pts = np.stack([mx[m], my[m]], 1) + np.asarray(center)
    return CWSource(
        indices=pts.astype(np.int64),
        phases=np.zeros(pts.shape[0], np.float32),
        amplitude=amplitude,
        f0=f0,
        label="disc",
    )


def _compare_panels(outdir, name, title, ours, kwave, dx_mm, inner, label_a="hifusim"):
    """Three-panel |amp| comparison: ours | k-Wave | signed difference."""
    a = ours[inner] / ours[inner].max()
    b = kwave[inner] / kwave[inner].max()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    im0 = imshow_mm(axes[0], a, dx_mm, vmax=1.0)
    axes[0].set_title(f"{label_a} |p| (norm)")
    im1 = imshow_mm(axes[1], b, dx_mm, vmax=1.0)
    axes[1].set_title("k-Wave |p| (norm)")
    d = a - b
    lim = max(1e-6, np.abs(d).max())
    im2 = imshow_mm(axes[2], d, dx_mm, cmap=CMAP_DIV, vmin=-lim, vmax=lim)
    axes[2].set_title(f"difference (max |Δ| = {lim:.3f})")
    for im, ax in ((im0, axes[0]), (im1, axes[1]), (im2, axes[2])):
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    fig.suptitle(title, fontsize=11, y=1.02)
    savefig(fig, outdir, name)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def scenario_kwave2d_linear(outdir: Path) -> None:
    t0 = time.perf_counter()
    grid = Grid(shape=(128, 128), dx=0.5e-3, pml=PMLSpec(thickness=10e-3))
    med = Medium.homogeneous(grid.shape, water(c=C0))
    src = _disc_source_2d((40, 64), 3, 1e5, f0=0.5e6)
    spec = CWRunSpec(min_settle_periods=10)
    r_us = solvers.get("linear")().run(grid, med, src, spec, backend="numpy")
    r_kw = solvers.get("kwave")().run(grid, med, src, spec)
    inner = (slice(26, 102), slice(26, 102))
    m = field_metrics(r_us.amp[inner], r_kw.amp[inner])
    _compare_panels(
        outdir, "fig_kwave2d_linear", "2-D linear water: hifusim `linear` vs k-Wave (OMP)",
        r_us.amp, r_kw.amp, 0.5, inner,
    )
    save_fragment(outdir, "kwave2d_linear", m, time.perf_counter() - t0)


def scenario_kwave2d_hetero(outdir: Path) -> None:
    t0 = time.perf_counter()
    grid = Grid(shape=(128, 128), dx=0.5e-3, pml=PMLSpec(thickness=10e-3))
    id_map = np.zeros(grid.shape, dtype=np.uint8)
    id_map[:, 64:] = 1  # far half: fat-like slab
    from hifusim.materials import MaterialDB

    db = MaterialDB(
        materials={
            0: water(c=C0),
            1: Material(name="fat", alpha_np_m=6.0, rho=932.0, c=1450.0, beta=0.0),
        }
    )
    med = Med.from_id_map(id_map, db)
    src = _disc_source_2d((40, 40), 3, 1e5, f0=0.5e6)
    spec = CWRunSpec(min_settle_periods=14)
    r_us = solvers.get("linear")().run(grid, med, src, spec, backend="numpy")
    r_kw = solvers.get("kwave")().run(grid, med, src, spec)
    inner = (slice(26, 102), slice(26, 102))
    m = field_metrics(r_us.amp[inner], r_kw.amp[inner])
    _compare_panels(
        outdir, "fig_kwave2d_hetero",
        "2-D heterogeneous (water | fat slab, alpha=6 Np/m): hifusim vs k-Wave",
        r_us.amp, r_kw.amp, 0.5, inner,
    )
    save_fragment(outdir, "kwave2d_hetero", m, time.perf_counter() - t0)


def scenario_kwave2d_nonlinear(outdir: Path) -> None:
    """Nonlinear cross-check with amplitude matching.

    The two codes scale injected sources differently, and nonlinear growth
    depends on the REALIZED amplitude — so we first match the realized f0
    peak (one cheap re-run with a scaled source), then compare f0 and 2f0.
    """
    t0 = time.perf_counter()
    grid = Grid(shape=(128, 128), dx=0.5e-3, pml=PMLSpec(thickness=10e-3))
    med = Medium.homogeneous(grid.shape, water(c=C0, beta=3.5))
    spec = CWRunSpec(min_settle_periods=10)
    inner = (slice(26, 102), slice(26, 102))
    amp0 = 2.0e6
    src = _disc_source_2d((40, 64), 3, amp0, f0=0.5e6)
    r_kw = solvers.get("kwave")().run(grid, med, src, spec, harmonics=(1, 2))
    r_us0 = solvers.get("westervelt")().run(
        grid, med, src, spec, backend="numpy", harmonics=(1, 2)
    )
    scale = float(r_kw.amp[inner].max() / r_us0.amp[inner].max())
    src2 = _disc_source_2d((40, 64), 3, amp0 * scale, f0=0.5e6)
    r_us = solvers.get("westervelt")().run(
        grid, med, src2, spec, backend="numpy", harmonics=(1, 2)
    )

    m1 = field_metrics(r_us.amp[inner], r_kw.amp[inner])
    a2_us, a2_kw = r_us.harmonic_amp(2), r_kw.harmonic_amp(2)
    m2 = field_metrics(a2_us[inner], a2_kw[inner])
    ratio_us = float(a2_us[inner].max() / r_us.amp[inner].max())
    ratio_kw = float(a2_kw[inner].max() / r_kw.amp[inner].max())
    _compare_panels(
        outdir, "fig_kwave2d_nl_f0",
        "2-D nonlinear water (beta=3.5), f0 field: hifusim `westervelt` vs k-Wave",
        r_us.amp, r_kw.amp, 0.5, inner,
    )
    _compare_panels(
        outdir, "fig_kwave2d_nl_2f0",
        "2-D nonlinear water, SECOND HARMONIC (2f0): hifusim vs k-Wave",
        a2_us, a2_kw, 0.5, inner,
    )
    save_fragment(
        outdir, "kwave2d_nonlinear",
        {
            "f0": m1, "second_harmonic": m2,
            "a2_over_a1_hifusim": round(ratio_us, 4),
            "a2_over_a1_kwave": round(ratio_kw, 4),
            "amplitude_match_scale": round(scale, 4),
        },
        time.perf_counter() - t0,
    )


def scenario_kwave3d_bowl(outdir: Path) -> None:
    t0 = time.perf_counter()
    dx = 0.375e-3
    grid = Grid(shape=(80, 80, 120), dx=dx, pml=PMLSpec(thickness=5e-3))
    med = Medium.homogeneous(grid.shape, water(c=C0))
    apex, a, roc = (40, 40, 16), 7.5e-3, 18e-3
    src = bowl_cw_source(grid, f0=F0, amplitude=1e5, aperture_radius=a, roc=roc, apex_vox=apex)
    spec = CWRunSpec(min_settle_periods=8, max_settle_periods=30)
    focus = (40, 40, 64)
    r_us = solvers.get("linear")().run(
        grid, med, src, spec, backend="numpy", reference_point=focus
    )
    r_kw = solvers.get("kwave")().run(grid, med, src, spec, reference_point=focus)

    inner = (slice(16, 64), slice(16, 64), slice(24, 100))
    m = field_metrics(r_us.amp[inner], r_kw.amp[inner])

    # Three-way axial overlay: hifusim, k-Wave, O'Neil closed form.
    z_phys = (np.arange(120) - apex[2]) * dx
    sel = slice(28, 100)
    ax_us = r_us.amp[40, 40, sel]
    ax_kw = r_kw.amp[40, 40, sel]
    ax_on = np.abs(axial_pressure(z_phys[sel], a, roc, F0, C0))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    z_mm = z_phys[sel] * 1e3
    axes[0].plot(z_mm, ax_us / ax_us.max(), color=CAT[0], label="hifusim linear")
    axes[0].plot(z_mm, ax_kw / ax_kw.max(), color=CAT[1], ls="--", label="k-Wave 3D (OMP)")
    axes[0].plot(z_mm, ax_on / ax_on.max(), color=INK2, ls=":", lw=1.6, label="O'Neil (analytic)")
    axes[0].set_xlabel("z from apex [mm]")
    axes[0].set_ylabel("|p| / max")
    axes[0].set_title("3-D focused bowl, axial profile (three-way)")
    axes[0].legend(frameon=False)
    mid = imshow_mm(
        axes[1], r_us.amp[:, 40, :] / r_us.amp.max(), dx * 1e3, vmax=1.0
    )
    axes[1].set_title("hifusim |p| mid-plane (x-z)")
    fig.colorbar(mid, ax=axes[1], shrink=0.85, pad=0.02)
    savefig(fig, outdir, "fig_kwave3d_bowl")

    on_metrics = field_metrics(ax_us[:, None], ax_on[:, None])
    save_fragment(
        outdir, "kwave3d_bowl",
        {"vs_kwave": m, "axial_vs_oneill": on_metrics},
        time.perf_counter() - t0,
    )


def scenario_oneill3d(outdir: Path) -> None:
    t0 = time.perf_counter()
    dx = 0.375e-3
    grid = Grid(shape=(80, 80, 120), dx=dx, pml=PMLSpec(thickness=5e-3))
    med = Medium.homogeneous(grid.shape, water(c=C0))
    apex, a, roc = (40, 40, 16), 7.5e-3, 18e-3
    src = bowl_cw_source(grid, f0=F0, amplitude=1e5, aperture_radius=a, roc=roc, apex_vox=apex)
    spec = CWRunSpec(min_settle_periods=8, max_settle_periods=30)
    res = solvers.get("linear")().run(
        grid, med, src, spec, backend="numpy", reference_point=(40, 40, 64)
    )
    z_phys = (np.arange(120) - apex[2]) * dx
    sel = slice(28, 100)
    axial = res.amp[40, 40, sel]
    on = np.abs(axial_pressure(z_phys[sel], a, roc, F0, C0))

    z_pk = int(np.argmax(res.amp[40, 40, :]))
    lateral = res.amp[:, 40, z_pk]
    x_phys = (np.arange(80) - 40) * dx
    pts, _n, areas = spherical_cap_points(a, roc, spacing=C0 / F0 / 5)
    targets = np.column_stack(
        [x_phys, np.zeros_like(x_phys), np.full_like(x_phys, (z_pk - apex[2]) * dx)]
    )
    ray = np.abs(rayleigh_pressure(pts, areas, 1.0, targets, k=2 * np.pi * F0 / C0, c=C0))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    axes[0].plot(z_phys[sel] * 1e3, axial / axial.max(), color=CAT[0], label="hifusim linear")
    axes[0].plot(z_phys[sel] * 1e3, on / on.max(), color=INK2, ls=":", lw=1.6, label="O'Neil")
    axes[0].set_xlabel("z from apex [mm]")
    axes[0].set_ylabel("|p| / max")
    axes[0].set_title("Axial: solver vs O'Neil closed form")
    axes[0].legend(frameon=False)
    axes[1].plot(x_phys * 1e3, lateral / lateral.max(), color=CAT[0], label="hifusim linear")
    axes[1].plot(x_phys * 1e3, ray / ray.max(), color=INK2, ls=":", lw=1.6, label="Rayleigh")
    axes[1].set_xlabel("x at focal plane [mm]")
    axes[1].set_title("Lateral: solver vs Rayleigh integral")
    axes[1].legend(frameon=False)
    savefig(fig, outdir, "fig_oneill3d")

    r_ax = float(np.corrcoef(axial / axial.max(), on / on.max())[0, 1])
    r_lat = float(np.corrcoef(lateral / lateral.max(), ray / ray.max())[0, 1])
    save_fragment(
        outdir, "oneill3d",
        {"axial_r": round(r_ax, 5), "lateral_r": round(r_lat, 5)},
        time.perf_counter() - t0,
    )


def scenario_fubini(outdir: Path) -> None:
    t0 = time.perf_counter()
    ppw = 16.0
    dx = C0 / (F0 * ppw)
    grid = Grid(shape=(600,), dx=dx, pml=PMLSpec(thickness=40 * dx))
    med = Medium.homogeneous(grid.shape, water(c=C0, beta=3.5))
    src = plane_cw_source(grid, f0=F0, amplitude=2.0e6, position_vox=60)
    spec = CWRunSpec(min_settle_periods=45, max_settle_periods=100, convergence_tol=0.003)
    res = solvers.get("westervelt")().run(
        grid, med, src, spec, backend="numpy", harmonics=(1, 2, 3)
    )
    a1, a2, a3 = (res.harmonic_amp(k) for k in (1, 2, 3))
    i0 = 90
    p0 = a1[i0] / fubini_harmonic(
        1, ((i0 - 60) * dx) / shock_distance(a1[i0], F0, C0, beta=3.5)
    )
    x_sh = shock_distance(p0, F0, C0, beta=3.5)
    iv = np.arange(100, 545, 15)
    sig = (iv - 60) * dx / x_sh
    r2_meas = a2[iv] / a1[iv]
    r3_meas = a3[iv] / a1[iv]
    s_fine = np.linspace(0.02, min(0.95, sig.max()), 200)
    r2_an = fubini_harmonic(2, s_fine) / fubini_harmonic(1, s_fine)
    r3_an = fubini_harmonic(3, s_fine) / fubini_harmonic(1, s_fine)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(s_fine, r2_an, color=INK2, lw=1.4, label="Fubini analytic")
    ax.plot(s_fine, r3_an, color=INK2, lw=1.4, ls=":")
    ax.plot(sig, r2_meas, "o", ms=5, color=CAT[0], label="A2/A1 westervelt")
    ax.plot(sig, r3_meas, "s", ms=5, color=CAT[1], label="A3/A1 westervelt")
    ax.set_xlabel("sigma = x / x_shock")
    ax.set_ylabel("harmonic ratio")
    ax.set_title("Nonlinear harmonic growth vs Fubini solution (plane wave, 16 ppw)")
    ax.legend(frameon=False)
    savefig(fig, outdir, "fig_fubini")

    sel = (sig > 0.05) & (sig < 0.95)
    err2 = np.abs(
        r2_meas[sel] - fubini_harmonic(2, sig[sel]) / fubini_harmonic(1, sig[sel])
    ) / (fubini_harmonic(2, sig[sel]) / fubini_harmonic(1, sig[sel]))
    save_fragment(
        outdir, "fubini",
        {
            "p0_effective_mpa": round(float(p0) / 1e6, 3),
            "sigma_range": [round(float(sig[sel].min()), 3), round(float(sig[sel].max()), 3)],
            "a2_a1_max_err_pct": round(float(err2.max() * 100), 2),
            "a2_a1_mean_err_pct": round(float(err2.mean() * 100), 2),
        },
        time.perf_counter() - t0,
    )


def scenario_planewave(outdir: Path) -> None:
    t0 = time.perf_counter()
    ppw = 4.0
    dx = C0 / (F0 * ppw)
    n = 288
    grid = Grid(shape=(n,), dx=dx, pml=PMLSpec(thickness=32 * dx))
    alpha = 30.0
    med = Medium.homogeneous(grid.shape, water(alpha_np_m=alpha, c=C0))
    src = plane_cw_source(grid, f0=F0, amplitude=1e5, position_vox=40)
    spec = CWRunSpec(min_settle_periods=45, max_settle_periods=95, convergence_tol=0.005)
    res = solvers.get("linear")().run(grid, med, src, spec, backend="numpy")
    span = slice(80, 200)
    x = np.arange(*span.indices(n)) * dx
    amp = res.amp[span]
    slope = np.polyfit(x, np.log(amp), 1)[0]
    alpha_meas = -slope

    # Dispersion is measured on a LOSSLESS run (mixing it into the absorbing
    # run contaminates the phase-gradient estimate with decay ripple).
    med0 = Medium.homogeneous(grid.shape, water(c=C0))
    res0 = solvers.get("linear")().run(grid, med0, src, spec, backend="numpy")
    ph = res0.phasor[span]
    k_num = float(-np.angle(ph[1:] * np.conj(ph[:-1])).mean() / dx)
    k0 = 2 * np.pi * F0 / C0
    disp_err = abs(k_num - k0) / k0

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.semilogy(x * 1e3, amp / amp[0], color=CAT[0], label="hifusim linear |p|")
    ax.semilogy(
        x * 1e3, np.exp(-alpha * (x - x[0])), color=INK2, ls=":", lw=1.6,
        label=f"exp(-{alpha:.0f} x) analytic",
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("|p| / |p(x0)| (log)")
    ax.set_title(
        f"Plane-wave absorption: measured alpha = {alpha_meas:.2f} Np/m "
        f"(target {alpha:.0f}, err {abs(alpha_meas - alpha) / alpha * 100:.2f}%)"
    )
    ax.legend(frameon=False)
    savefig(fig, outdir, "fig_absorption")

    save_fragment(
        outdir, "planewave",
        {
            "alpha_target_np_m": alpha,
            "alpha_measured_np_m": round(float(alpha_meas), 3),
            "alpha_err_pct": round(float(abs(alpha_meas - alpha) / alpha * 100), 3),
            "dispersion_err_pct": round(float(disp_err * 100), 5),
        },
        time.perf_counter() - t0,
    )


def scenario_spiral(outdir: Path) -> None:
    t0 = time.perf_counter()
    arr = archimedean_spiral()  # production 128-element geometry
    target = np.array([8e-3, 0.0, 0.085])  # steer 8 mm laterally, 15 mm proximal
    phases = arr.das_phases(target, f0=1.1e6, c0=C0)

    # Rayleigh beam preview on the x-z mid-plane.
    x = np.linspace(-30e-3, 30e-3, 181)
    z = np.linspace(0.03, 0.14, 221)
    gx, gz = np.meshgrid(x, z, indexing="ij")
    pts = np.column_stack([gx.ravel(), np.zeros(gx.size), gz.ravel()])
    p = np.abs(
        rayleigh_pressure(
            arr.positions, np.full(arr.n_elements, np.pi * arr.elem_radius**2),
            np.exp(1j * phases.astype(np.float64)), pts, k=2 * np.pi * 1.1e6 / C0, c=C0,
        )
    ).reshape(gx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    sc = axes[0].scatter(
        arr.positions[:, 0] * 1e3, arr.positions[:, 1] * 1e3,
        c=phases, cmap=CMAP_SEQ, s=42, edgecolors=INK2, linewidths=0.4,
    )
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    axes[0].set_title("128-element spiral, DAS phases (steered focus)")
    fig.colorbar(sc, ax=axes[0], shrink=0.85, pad=0.02, label="phase [rad]")
    im = axes[1].imshow(
        p / p.max(), cmap=CMAP_SEQ,
        extent=[z[0] * 1e3, z[-1] * 1e3, x[0] * 1e3, x[-1] * 1e3],
        aspect="auto", vmin=0, vmax=1,
    )
    axes[1].grid(False)
    axes[1].plot(target[2] * 1e3, target[0] * 1e3, "+", color=CAT[1], ms=14, mew=2.2)
    axes[1].set_xlabel("z [mm]")
    axes[1].set_ylabel("x [mm]")
    axes[1].set_title("Rayleigh beam preview (x-z), + = commanded target")
    fig.colorbar(im, ax=axes[1], shrink=0.85, pad=0.02)
    savefig(fig, outdir, "fig_spiral")

    pk = np.unravel_index(np.argmax(p), p.shape)
    peak_xz = np.array([x[pk[0]], z[pk[1]]])
    save_fragment(
        outdir, "spiral",
        {
            "n_elements": arr.n_elements,
            "elem_radius_mm": round(arr.elem_radius * 1e3, 3),
            "target_xz_mm": [round(target[0] * 1e3, 1), round(target[2] * 1e3, 1)],
            "peak_xz_mm": [round(float(peak_xz[0] * 1e3), 2), round(float(peak_xz[1] * 1e3), 2)],
        },
        time.perf_counter() - t0,
    )


SCENARIOS = {
    "kwave2d_linear": scenario_kwave2d_linear,
    "kwave2d_hetero": scenario_kwave2d_hetero,
    "kwave2d_nonlinear": scenario_kwave2d_nonlinear,
    "kwave3d_bowl": scenario_kwave3d_bowl,
    "oneill3d": scenario_oneill3d,
    "fubini": scenario_fubini,
    "planewave": scenario_planewave,
    "spiral": scenario_spiral,
}


def assemble(outdir: Path) -> None:
    merged = {}
    for f in sorted(outdir.glob("metrics_*.json")):
        frag = json.loads(f.read_text())
        merged[frag.pop("scenario")] = frag
    (outdir / "metrics.json").write_text(json.dumps(merged, indent=2))
    print(f"assembled {len(merged)} scenario metrics -> metrics.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.assemble:
        assemble(outdir)
        return
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in names:
        SCENARIOS[name](outdir)


if __name__ == "__main__":
    main()
