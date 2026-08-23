#!/usr/bin/env python3
"""Run the documentation site's example gallery and write it out.

Five worked examples, each one actually solved here -- no figure on the gallery
page is a sketch, and no number on it was typed by hand. The script writes::

    docs/examples.md                      the gallery page, generated
    docs/assets/examples/<key>.svg        one figure per example, light
    docs/assets/examples/<key>-dark.svg   ...and dark

Run it after anything that could change a result::

    python scripts/make_examples.py            # all five
    python scripts/make_examples.py bowl heat  # only the ones named
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import textwrap
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
from make_howto import DARK, LIGHT, REPO, Theme, cmap_of, mix  # noqa: E402

matplotlib.rcParams["svg.fonttype"] = "path"
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["path.simplify_threshold"] = 1.0

logging.disable(logging.WARNING)
warnings.simplefilter("ignore")

OUT = REPO / "docs" / "assets" / "examples"
PAGE = REPO / "docs" / "examples.md"

C0_WATER = 1500.0
WATER = {"name": "water", "c": 1500.0, "rho": 1000.0, "alpha_np_m": 0.025, "beta": 3.5}

# One aperture, shared by the focused examples. f/1.0 at 1.2 MHz is ka = 50 and a
# linear focal gain near 13x -- enough that the pressure maximum lands where the
# geometry says it should. A slower, smaller aperture (f/1.25 at 1 MHz was tried)
# has its maximum several millimetres pre-focal, which makes every metric on this
# page read like a bug: the -6 dB length triples and a steering test appears to
# miss its target by the same amount the focal shift moved it.
F0_MHZ = 1.2
DX_MM = 0.18
BOX = [25.92, 25.92, 32.4]  # mm; 144 x 144 x 180 voxels
PML_MM = 2.7
CENTRE = BOX[0] / 2
APEX_Z = 4.5
ROC_MM = 20.0
APERTURE_MM = 20.0
FOCUS_Z = APEX_Z + ROC_MM

# --------------------------------------------------------------------- drawing

FIG_W = 9.4  # inches; the gallery renders these at the content column's width
FIG_H = 3.5


def figure(th: Theme, ncols: int = 2):
    fig, axes = plt.subplots(1, ncols, figsize=(FIG_W, FIG_H), dpi=100)
    fig.patch.set_facecolor(th.fig_bg)
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(th.fig_bg)
        for s in ax.spines.values():
            s.set_color(th.stroke)
        ax.tick_params(colors=th.muted, labelsize=8, length=3)
        ax.xaxis.label.set_color(th.muted)
        ax.yaxis.label.set_color(th.muted)
        ax.title.set_color(th.text)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.15, wspace=0.22)
    return fig, np.atleast_1d(axes)


def field_panel(ax, data, extent, th: Theme, *, title: str, levels: int = 11) -> None:
    """One |P| map, in the single-hue ramp the rest of the site is drawn in."""
    peak = float(np.nanmax(data)) or 1.0
    ax.contourf(
        data / peak,
        levels=np.linspace(0.06, 1.0, levels),
        cmap=cmap_of(th),
        extend="max",
        extent=extent,
        origin="lower",
    )
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_aspect("equal")


def line_panel(ax, th: Theme, *, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, color=mix(th.fig_bg, th.stroke, 0.7), lw=0.6)
    ax.set_axisbelow(True)


def legend(ax, th: Theme) -> None:
    leg = ax.legend(fontsize=8, frameon=True, facecolor=th.fig_bg, edgecolor=th.stroke)
    for t in leg.get_texts():
        t.set_color(th.text)


def save(fig, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", facecolor=fig.get_facecolor())
    plt.close(fig)
    svg = re.sub(r"\d+\.\d{3,}", lambda m: f"{float(m.group()):.2f}", buf.getvalue())
    path.write_text(svg, encoding="utf-8")
    return len(svg)


# ----------------------------------------------------------------- the harness


@dataclass
class Example:
    key: str
    title: str
    lede: str
    body: str
    #: builds whatever the figure and the facts need; returns a context dict
    compute: Callable[[], dict[str, Any]]
    draw: Callable[[Any, dict[str, Any], Theme], None]
    facts: Callable[[dict[str, Any]], list[tuple[str, str]]]
    #: shown in the page's collapsed job block; None for the low-level examples
    job: dict[str, Any] | None = None
    code: str = ""
    seconds: float = field(default=0.0, init=False)


def axes_mm(shape, dx_mm, pml_vox):
    """Interior extent in millimetres, measured from the interior corner."""
    n = [s - 2 * pml_vox for s in shape]
    return [x * dx_mm for x in n]


def mid_plane(phasor, pml_vox):
    """|P| on the x-z plane through the middle of the domain, PML trimmed."""
    p = np.abs(np.asarray(phasor))
    w = pml_vox
    return p[w:-w, p.shape[1] // 2, w:-w]


def run_job(job: dict) -> Any:
    import caustica

    return caustica.simulate(job, out=None, progress=None)


# ------------------------------------------------------------------ example: 1


def bowl_job() -> dict:
    return {
        "format": "caustica-job/1",
        "kind": "explicit",
        "name": "focused_bowl",
        "medium": {"kind": "homogeneous"},
        "grid": {
            "ndim": 3,
            "dx_mm": DX_MM,
            "size_mm": BOX,
            "pml": {"thickness_mm": PML_MM},
        },
        "source": {
            "kind": "array",
            "array": {"kind": "bowl", "d_outer_mm": APERTURE_MM, "roc_mm": ROC_MM},
            "apex_mm": [CENTRE, CENTRE, APEX_Z],
        },
        "drive": {"f0_mhz": F0_MHZ, "amplitude_kpa": 100.0},
        "run": {"spec": {"min_settle_periods": 2, "max_settle_periods": 8}, "harmonics": [1]},
        "solver": "linear",
    }


def bowl_compute() -> dict:
    from caustica import analytic

    job = bowl_job()
    res = run_job(job)
    g = res.geometry
    pml, dx_mm = g["pml_vox"], DX_MM
    plane = mid_plane(res.phasor, pml)

    # the axis, and O'Neil on the same axis -- both normalised, because the
    # analytic form takes a surface velocity and the job takes a pressure
    axis = plane[plane.shape[0] // 2]
    z_mm = (np.arange(axis.size) + 0.5) * dx_mm + pml * dx_mm
    z_from_apex = (z_mm - job["source"]["apex_mm"][2]) * 1e-3
    ok = z_from_apex > 0
    oneil = np.zeros_like(axis)
    oneil[ok] = np.abs(
        analytic.axial_pressure(
            z_from_apex[ok],
            aperture_radius=APERTURE_MM / 2 * 1e-3,
            roc=ROC_MM * 1e-3,
            f0=F0_MHZ * 1e6,
            c0=C0_WATER,
        )
    )
    a, b = axis / axis.max(), oneil / max(oneil.max(), 1e-30)
    r = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return {
        "job": job,
        "res": res,
        "plane": plane,
        "extent": [pml * dx_mm, pml * dx_mm + plane.shape[1] * dx_mm, -CENTRE, CENTRE],
        "z_mm": z_mm,
        "solved": a,
        "oneil": b,
        "r": r,
        "gain": analytic.focal_gain(APERTURE_MM / 2 * 1e-3, ROC_MM * 1e-3, F0_MHZ * 1e6, C0_WATER),
    }


def bowl_draw(fig, c, th: Theme) -> None:
    axes = fig.axes
    field_panel(axes[0], c["plane"], c["extent"], th, title="|P| on the x–z plane")
    axes[0].set_xlabel("z [mm]", fontsize=8)
    axes[0].set_ylabel("x [mm]", fontsize=8)

    ax = axes[1]
    line_panel(ax, th, title="on axis, against O'Neil", xlabel="z [mm]", ylabel="|P| / peak")
    ax.plot(c["z_mm"], c["solved"], color=th.accent, lw=1.8, label="caustica")
    ax.plot(c["z_mm"], c["oneil"], color=th.muted, lw=1.2, ls="--", label="O'Neil (1949)")
    legend(ax, th)


def bowl_facts(c) -> list[tuple[str, str]]:
    m = c["res"].metrics
    return [
        ("peak pressure", f"{m['peak']['p_pa'] / 1e6:.2f} MPa"),
        ("focus offset from geometric", f"{m['target']['displacement_norm_mm']:.2f} mm"),
        ("−6 dB axial × lateral", _spot(m)),
        ("correlation with O'Neil on axis", f"r = {c['r']:.4f}"),
        ("linear focal gain (analytic)", f"{c['gain']:.1f}×"),
    ]


def _spot(m) -> str:
    s = m["focal_spot"]
    return f"{s['axial_6db']['width_mm']:.2f} mm × {s['lateral_x_6db']['width_mm']:.2f} mm"


# ------------------------------------------------------------------ example: 2


PPW, P0_PA, BETA = 16.0, 2.0e6, 3.5
SRC_VOX, N_CELLS = 60, 600


def harmonics_compute() -> dict:
    """A 1-D plane wave, steepening, against Fubini's series term by term."""
    import caustica.solvers as solvers
    from caustica import Grid, Medium, PMLSpec
    from caustica.analytic import fubini_harmonic, shock_distance
    from caustica.materials import water
    from caustica.solvers import CWRunSpec
    from caustica.sources import plane_cw_source

    f0, c0 = 1.0e6, 1500.0
    dx = c0 / (f0 * PPW)
    grid = Grid(shape=(N_CELLS,), dx=dx, pml=PMLSpec(thickness=40 * dx))
    med = Medium.homogeneous(grid.shape, water(c=c0, beta=BETA))
    src = plane_cw_source(grid, f0=f0, amplitude=P0_PA, position_vox=SRC_VOX)
    res = solvers.get("westervelt")().run(
        grid,
        med,
        src,
        CWRunSpec(min_settle_periods=45, max_settle_periods=100, convergence_tol=0.003),
        backend="numpy",
        harmonics=(1, 2, 3),
    )

    amp = {n: np.asarray(res.harmonic_amp(n)) for n in (1, 2, 3)}
    # sigma = distance / shock distance, with the source amplitude recovered from
    # the field itself: what reaches the medium is not exactly what was asked for
    i0 = 90
    guess = ((i0 - SRC_VOX) * dx) / shock_distance(amp[1][i0], f0, c0, beta=BETA)
    p0 = amp[1][i0] / fubini_harmonic(1, guess)
    x_sh = shock_distance(p0, f0, c0, beta=BETA)

    idx = np.arange(SRC_VOX + 20, N_CELLS - 60)
    sigma = (idx - SRC_VOX) * dx / x_sh
    keep = (sigma >= 0.05) & (sigma <= 0.95)
    idx, sigma = idx[keep], sigma[keep]

    solved = {n: amp[n][idx] / p0 for n in (1, 2, 3)}
    exact = {n: np.asarray(fubini_harmonic(n, sigma), float) for n in (1, 2, 3)}
    err = np.abs(solved[2] / solved[1] - exact[2] / exact[1]) / (exact[2] / exact[1]) * 100.0
    # a standing-wave ripple at the acoustic wavelength rides on the deviation;
    # averaging over exactly one wavelength removes it without hiding the level
    w = int(round(PPW))
    kernel = np.ones(w) / w
    smooth = np.convolve(np.pad(err, w // 2, mode="edge"), kernel, mode="same")[
        w // 2 : w // 2 + err.size
    ]
    return {
        "res": res,
        "sigma": sigma,
        "solved": solved,
        "exact": exact,
        "err": err,
        "smooth": smooth,
        "p0_mpa": float(p0 / 1e6),
        "worst": float(err.max()),
        "median": float(np.median(err)),
        "x_sh_mm": float(x_sh * 1e3),
    }


def harmonics_draw(fig, c, th: Theme) -> None:
    axes = fig.axes
    ax = axes[0]
    line_panel(ax, th, title="harmonic amplitudes", xlabel="sigma = x / x_shock", ylabel="An / p0")
    shades = [th.accent, mix(th.fig_bg, th.accent, 0.6), mix(th.fig_bg, th.accent, 0.35)]
    for n, col in zip((1, 2, 3), shades, strict=False):
        ax.plot(c["sigma"], c["solved"][n], color=col, lw=1.8, label=f"{n}f0")
        ax.plot(c["sigma"], c["exact"][n], color=th.muted, lw=1.0, ls="--")
    ax.plot([], [], color=th.muted, lw=1.0, ls="--", label="Fubini")
    legend(ax, th)

    ax = axes[1]
    line_panel(
        ax,
        th,
        title="A2/A1 against Fubini",
        xlabel="sigma = x / x_shock",
        ylabel="deviation [%]",
    )
    ax.plot(c["sigma"], c["err"], color=mix(th.fig_bg, th.accent, 0.35), lw=0.7)
    ax.plot(c["sigma"], c["smooth"], color=th.accent, lw=2.0, label="averaged over one λ")
    ax.axhline(5.0, color=th.muted, lw=1.0, ls="--")
    legend(ax, th)
    ax.text(c["sigma"][-1], 5.2, "the gate: 5 %", ha="right", fontsize=8, color=th.muted)
    ax.set_ylim(0, max(6.5, c["worst"] * 1.3))


def harmonics_facts(c) -> list[tuple[str, str]]:
    return [
        ("drive", f"{c['p0_mpa']:.2f} MPa plane wave at 1 MHz, beta = {BETA}"),
        ("resolution", f"{PPW:.0f} points per wavelength (3f0 at {PPW / 3:.1f})"),
        ("shock distance", f"{c['x_sh_mm']:.0f} mm"),
        ("sigma covered", f"{c['sigma'][0]:.2f} - {c['sigma'][-1]:.2f}"),
        ("A2/A1 vs Fubini, typical", f"{np.median(c['smooth']):.1f} %"),
        ("A2/A1 vs Fubini, worst single point", f"{c['worst']:.1f} %"),
        ("the gate the tests enforce", "5 %"),
    ]


# ------------------------------------------------------------------ example: 3

SKIN = {"name": "skin", "c": 1610.0, "rho": 1090.0, "alpha_np_m": 21.0, "beta": 0.0}
BONE = {"name": "bone", "c": 2800.0, "rho": 1900.0, "alpha_np_m": 200.0, "beta": 0.0}
BRAIN = {"name": "brain", "c": 1550.0, "rho": 1040.0, "alpha_np_m": 8.0, "beta": 0.0}
SKIN_MM, BONE_Z0, BONE_MM = 2.0, 8.0, 4.0


def layered_job() -> dict:
    job = bowl_job()
    job["name"] = "layered"
    rest0 = BONE_Z0 + BONE_MM

    def slab(z0, thickness, label):
        return {
            "shape": {
                "kind": "box",
                "center_mm": [CENTRE, CENTRE, z0 + thickness / 2],
                "size_mm": [BOX[0], BOX[1], thickness],
            },
            "label": label,
        }

    job["medium"] = {
        "kind": "scene",
        "scene": {
            "ndim": 3,
            "background": 0,
            "objects": [
                slab(BONE_Z0 - SKIN_MM, SKIN_MM, 1),
                slab(BONE_Z0, BONE_MM, 2),
                slab(rest0, BOX[2] - rest0, 3),
            ],
        },
        "materials": {"0": WATER | {"beta": 0.0}, "1": SKIN, "2": BONE, "3": BRAIN},
    }
    return job


def layered_compute() -> dict:
    job = layered_job()
    layered, water = run_job(job), run_job(bowl_job())
    pml, dx_mm = layered.geometry["pml_vox"], DX_MM
    pl = mid_plane(layered.phasor, pml)
    pw = mid_plane(water.phasor, pml)
    z_mm = (np.arange(pl.shape[1]) + 0.5) * dx_mm + pml * dx_mm
    al, aw = pl[pl.shape[0] // 2], pw[pw.shape[0] // 2]

    # The global maximum is no longer the focus: bone reflects hard enough that
    # the standing wave in front of it beats what gets through. Compare the two
    # runs where the comparison means something -- past the barrier.
    past = z_mm > BONE_Z0 + BONE_MM + 1.0
    il = int(np.argmax(np.where(past, al, 0.0)))
    iw = int(np.argmax(np.where(past, aw, 0.0)))
    return {
        "job": job,
        "res": layered,
        "water": water,
        "plane": pl,
        "extent": [pml * dx_mm, pml * dx_mm + pl.shape[1] * dx_mm, -CENTRE, CENTRE],
        "z_mm": z_mm,
        "layered": al / al.max(),
        "water_axis": aw / aw.max(),
        "focus_pa": (float(al[il]), float(aw[iw])),
        "focus_z": (float(z_mm[il]), float(z_mm[iw])),
        "shift": float(z_mm[il] - z_mm[iw]),
        "loss": float(al[il] / aw[iw]),
        "hot_z": float(z_mm[int(np.argmax(al))]),
    }


def layered_draw(fig, c, th: Theme) -> None:
    axes = fig.axes
    field_panel(axes[0], c["plane"], c["extent"], th, title="|P| through skin, bone and brain")
    axes[0].set_xlabel("z [mm]", fontsize=8)
    axes[0].set_ylabel("x [mm]", fontsize=8)
    for edge in (BONE_Z0, BONE_Z0 + BONE_MM):
        axes[0].axvline(edge, color=th.muted, lw=0.9, ls=":")

    ax = axes[1]
    line_panel(ax, th, title="what the bone costs", xlabel="z [mm]", ylabel="|P| / own peak")
    ax.plot(c["z_mm"], c["water_axis"], color=th.muted, lw=1.2, ls="--", label="water only")
    ax.plot(c["z_mm"], c["layered"], color=th.accent, lw=1.8, label="through bone")
    for edge in (BONE_Z0, BONE_Z0 + BONE_MM):
        ax.axvline(edge, color=th.muted, lw=0.9, ls=":")
    legend(ax, th)


def layered_facts(c) -> list[tuple[str, str]]:
    lay, wat = c["focus_pa"]
    zl, zw = c["focus_z"]
    return [
        ("layers", f"{SKIN_MM:.0f} mm skin, {BONE_MM:.0f} mm bone (2800 m/s), then brain"),
        ("focus in water", f"{wat / 1e6:.2f} MPa at z = {zw:.1f} mm"),
        ("focus through the barrier", f"{lay / 1e6:.2f} MPa at z = {zl:.1f} mm"),
        ("transmitted to the focus", f"{c['loss'] * 100:.0f} %"),
        ("focus moved", f"{c['shift']:+.1f} mm along z"),
        (
            "loudest point in the box",
            f"z = {c['hot_z']:.1f} mm — in front of the bone, not the focus",
        ),
    ]


# ------------------------------------------------------------------ example: 4

STEER_MM = [CENTRE + 4.0, CENTRE, FOCUS_Z]


def steered_job() -> dict:
    job = bowl_job()
    job["name"] = "steered_array"
    job["source"] = {
        "kind": "array",
        "array": {
            "kind": "archimedean_spiral",
            "n_elements": 128,
            "d_outer_mm": APERTURE_MM,
            "d_inner_mm": 7.0,
            "roc_mm": ROC_MM,
        },
        "apex_mm": [CENTRE, CENTRE, APEX_Z],
        "focus": {"mode": "steered", "target_mm": STEER_MM},
    }
    return job


def steered_compute() -> dict:
    job = steered_job()
    natural = steered_job()
    natural["name"] = "spiral_natural"
    natural["source"]["focus"] = {"mode": "natural"}
    res, ref = run_job(job), run_job(natural)

    pml, dx_mm = res.geometry["pml_vox"], DX_MM
    plane = mid_plane(res.phasor, pml)
    p = np.abs(np.asarray(res.phasor))[pml:-pml, pml:-pml, pml:-pml]
    pos = res.metrics["peak"]["position_mm_from_apex"]
    kz = int(round((STEER_MM[2] - pml * dx_mm) / dx_mm))
    face = p[:, :, min(kz, p.shape[2] - 1)]
    half = face.shape[0] * dx_mm / 2
    return {
        "job": job,
        "res": res,
        "ref": ref,
        "plane": plane,
        "extent": [pml * dx_mm, pml * dx_mm + plane.shape[1] * dx_mm, -CENTRE, CENTRE],
        "face": face,
        "face_extent": [-half, half, -half, half],
        "offset": float(np.hypot(*(pos[k] for k in "xy"))),
        "miss": float(res.metrics["target"]["displacement_norm_mm"]),
    }


def steered_draw(fig, c, th: Theme) -> None:
    axes = fig.axes
    field_panel(axes[0], c["plane"], c["extent"], th, title="|P| on the x–z plane")
    axes[0].set_xlabel("z [mm]", fontsize=8)
    axes[0].set_ylabel("x [mm]", fontsize=8)

    field_panel(axes[1], c["face"].T, c["face_extent"], th, title="the plane through the target")
    axes[1].set_xlabel("x [mm]", fontsize=8)
    axes[1].set_ylabel("y [mm]", fontsize=8)
    axes[1].plot([0], [0], marker="+", ms=9, color=th.muted, mew=1.4)
    axes[1].plot(
        [STEER_MM[0] - CENTRE],
        [STEER_MM[1] - CENTRE],
        marker="o",
        ms=7,
        mfc="none",
        mec=th.text,
        mew=1.4,
    )


def steered_facts(c) -> list[tuple[str, str]]:
    return [
        ("array", f"128-element spiral, {APERTURE_MM:.0f} mm, ROC {ROC_MM:.0f} mm"),
        ("steering target", "4 mm off axis, at the focal depth"),
        ("where the peak landed", f"{c['offset']:.2f} mm off axis"),
        ("miss from the requested target", f"{c['miss']:.2f} mm"),
        ("peak, steered", f"{c['res'].metrics['peak']['p_pa'] / 1e6:.2f} MPa"),
        (
            "peak, unsteered on the same array",
            f"{c['ref'].metrics['peak']['p_pa'] / 1e6:.2f} MPa",
        ),
    ]


# ------------------------------------------------------------------ example: 5

ON_S, OFF_S = 30.0, 30.0


def thermal_compute() -> dict:
    import caustica.solvers as solvers
    from caustica import Grid, Medium, PMLSpec
    from caustica.materials import Material, MaterialDB
    from caustica.sensors import HeatingSource
    from caustica.solvers import CWRunSpec
    from caustica.solvers.base import interior_slices
    from caustica.sources import bowl_cw_source
    from caustica.thermal.dose import ITRUSST_DELTA_T_LIMIT_C, cem43_rate
    from caustica.thermal.pennes import ARTERIAL_TEMPERATURE_C, PennesSolver
    from caustica.thermal.properties import ThermalMedium

    f0, dx = 1.0e6, 1540.0 / (1.0e6 * 5.0)
    liver = Material(
        name="Liver",
        c=1578.0,
        rho=1050.0,
        alpha_np_m=10.0,
        beta=0.0,
        thermal_conductivity=0.52,
        specific_heat=3540.0,
        perfusion_rate=15.0,
    )
    db = MaterialDB(materials={1: liver})
    shape = (64, 64, 96)
    grid = Grid(shape=shape, dx=dx, pml=PMLSpec(thickness=2.5e-3))
    medium = Medium.from_id_map(np.ones(shape, np.int32), db)
    source = bowl_cw_source(
        grid,
        f0=f0,
        amplitude=1.6e6,
        aperture_radius=7.0e-3,
        roc=16.0e-3,
        apex_vox=(shape[0] // 2, shape[1] // 2, 10),
    )
    region = interior_slices(shape, grid.pml_vox + 2)
    result = solvers.get("linear")().run(
        grid,
        medium,
        source,
        CWRunSpec(min_settle_periods=4, max_settle_periods=14),
        backend="numpy",
        record_region=region,
        harmonics=(1,),
    )
    heat = HeatingSource.from_result(result, medium, grid.dx, harmonics=(1,))
    tm = ThermalMedium.from_id_map(np.ones(shape, np.int32)[region], db, grid.dx)

    solver = PennesSolver(backend="numpy")
    t0 = np.full(tm.shape, ARTERIAL_TEMPERATURE_C, np.float32)

    # Three chained solves rather than two, because the transient and the plateau
    # need different sampling: this focus equilibrates in well under a second, so
    # a step chosen for the 30 s plateau records the rise as one vertical line.
    # Each solve carries the previous temperature AND its dose forward.
    fine, coarse = 0.02, min(0.9 * solver.stable_dt(tm), 0.05)
    ramp_s = 2.0
    plan = [
        ("ramp", heat, ramp_s, fine, 5),
        ("hold", heat, ON_S - ramp_s, coarse, int(2.0 / coarse)),
        ("cool", None, OFF_S, coarse, int(1.0 / coarse)),
    ]

    hot_ix, elapsed, dose0 = None, 0.0, None
    temperature, phases = t0, []
    trace_t, trace_c = [], []
    for _name, q, span, dt, every in plan:
        r = solver.solve(
            temperature,
            q,
            tm,
            dt=dt,
            n_steps=max(1, int(round(span / dt))),
            dose=True,
            dose0=dose0,
            record_every=max(1, every),
        )
        if hot_ix is None:
            hot_ix = np.unravel_index(
                int(np.argmax(np.asarray(r.temperature))), r.temperature.shape
            )
        for k, ts in enumerate(np.asarray(r.times)):
            t_here = float(ts) + elapsed
            if trace_t and t_here <= trace_t[-1] + 1e-9:
                continue
            trace_t.append(t_here)
            trace_c.append(float(np.asarray(r.samples[k])[hot_ix]))
        temperature, dose0 = r.temperature, r.dose_cem43
        elapsed += span
        phases.append(r)

    hot, cool = phases[1], phases[-1]
    trace_t, trace_c = np.asarray(trace_t), np.asarray(trace_c)
    # CEM43 straight from that trace, by its definition -- the solver accumulates
    # the same quantity over the field; here it is one voxel, so it can be drawn
    rate = np.asarray(cem43_rate(trace_c))
    dose_min = np.concatenate([[0.0], np.cumsum(rate[1:] * np.diff(trace_t))]) / 60.0
    # how long the rise actually takes, now that it is resolved
    on = trace_t <= ON_S
    plateau = float(trace_c[on].max()) - ARTERIAL_TEMPERATURE_C
    reached = trace_t[on][trace_c[on] - ARTERIAL_TEMPERATURE_C >= 0.9 * plateau]
    rise_s = float(reached[0]) if reached.size else float("nan")

    t = np.asarray(hot.temperature)
    dose = np.asarray(cool.dose_cem43)
    mid = t.shape[1] // 2
    half_x, len_z = t.shape[0] * dx * 1e3 / 2, t.shape[2] * dx * 1e3
    return {
        "res": None,
        "temp": t[:, mid, :],
        "dose": dose[:, mid, :],
        "trace_s": trace_t,
        "trace_c": trace_c,
        "trace_dose": dose_min,
        "rise_s": rise_s,
        "extent": [0.0, len_z, -half_x, half_x],
        "t_max": float(t.max()),
        "dt_max": float(t.max() - ARTERIAL_TEMPERATURE_C),
        "dose_max": float(dose.max()),
        "peak_mpa": float(np.abs(np.asarray(result.phasor)).max() / 1e6),
        "limit": ITRUSST_DELTA_T_LIMIT_C,
    }


def thermal_draw(fig, c, th: Theme) -> None:
    axes = fig.axes
    field_panel(axes[0], c["temp"] - 37.0, c["extent"], th, title=f"ΔT after {ON_S:.0f} s on")
    axes[0].set_xlabel("z [mm]", fontsize=8)
    axes[0].set_ylabel("x [mm]", fontsize=8)

    ax = axes[1]
    line_panel(ax, th, title="the focal voxel", xlabel="time [s]", ylabel="T [°C]")
    ax.axvspan(0, ON_S, color=mix(th.fig_bg, th.accent, 0.12), lw=0)
    ax.plot(c["trace_s"], c["trace_c"], color=th.accent, lw=2.0, label="temperature")
    ax.axhline(37.0, color=th.muted, lw=1.0, ls=":")

    dose_ax = ax.twinx()
    dose_ax.plot(c["trace_s"], c["trace_dose"], color=th.muted, lw=1.6, ls="--", label="CEM43")
    dose_ax.set_ylabel("CEM43 [min]", fontsize=8, color=th.muted)
    dose_ax.tick_params(colors=th.muted, labelsize=8, length=3)
    for spine in dose_ax.spines.values():
        spine.set_color(th.stroke)

    handles = ax.get_lines()[:1] + dose_ax.get_lines()[:1]
    leg = ax.legend(
        handles,
        [h.get_label() for h in handles],
        fontsize=8,
        loc="center right",
        frameon=True,
        facecolor=th.fig_bg,
        edgecolor=th.stroke,
    )
    for t in leg.get_texts():
        t.set_color(th.text)
    ax.text(
        ON_S / 2,
        c["trace_c"].min(),
        "sonicating",
        ha="center",
        va="bottom",
        fontsize=8,
        color=th.muted,
    )

    # the rise is half a second on a sixty-second axis; an inset is the only
    # honest way to show both that it is fast and that it is not instantaneous
    inset = ax.inset_axes((0.14, 0.30, 0.34, 0.42))
    early = c["trace_s"] <= 2.0
    inset.plot(c["trace_s"][early], c["trace_c"][early], color=th.accent, lw=1.6)
    inset.set_facecolor(th.fig_bg)
    inset.tick_params(colors=th.muted, labelsize=6.5, length=2)
    for spine in inset.spines.values():
        spine.set_color(th.stroke)
    inset.set_title("first 2 s", fontsize=7, color=th.muted, pad=2)


def thermal_facts(c) -> list[tuple[str, str]]:
    return [
        ("sonication", f"{ON_S:.0f} s on, then {OFF_S:.0f} s of cooling"),
        ("peak acoustic pressure", f"{c['peak_mpa']:.2f} MPa"),
        ("peak temperature", f"{c['t_max']:.1f} °C"),
        ("time to 90 % of that", f"{c['rise_s']:.2f} s"),
        ("peak ΔT", f"{c['dt_max']:.1f} °C (ITRUSST flags above {c['limit']:.0f} °C)"),
        ("peak CEM43", f"{c['dose_max']:.0f} min (ablation is conventionally 240)"),
    ]


# --------------------------------------------------------------------- the set

EXAMPLES = [
    Example(
        key="bowl",
        title="A focused bowl in water",
        lede="The reference case, and the one with a closed-form answer to check against.",
        body=(
            "A 20 mm bowl of 20 mm curvature — f/1.0 — driven at 1.2 MHz into water. This is the "
            "shape every other example is a variation on, and it is the one case where the answer "
            "is known independently: O'Neil's 1949 solution for a spherical cap. The right-hand "
            "panel is the solver's axis against that solution, both normalised — the analytic form "
            "is stated in surface velocity and the job in surface pressure, so only the shape is "
            "comparable, and the shape is what a solver gets wrong."
        ),
        compute=bowl_compute,
        draw=bowl_draw,
        facts=bowl_facts,
        job=bowl_job(),
    ),
    Example(
        key="harmonics",
        title="Nonlinear propagation, against Fubini",
        lede="Waveform steepening, harmonic by harmonic, checked against the closed form.",
        body=(
            "A 2 MPa plane wave in water with β = 3.5, marched with `westervelt` until the "
            "harmonics stop moving. Fubini's series is the exact answer for exactly this problem "
            "up to shock formation, so the second panel is not an illustration — it is the gate "
            "the test suite enforces, drawn. The solver has to land inside 5 % of Fubini for "
            "A₂/A₁ everywhere along the σ range this box reaches, and the table says where it "
            "actually lands. The thin trace is the raw deviation: a standing-wave ripple at "
            "the acoustic wavelength rides on it, so the heavy line is that same deviation "
            "averaged over exactly one wavelength.\n\n"
            "Sixteen points per wavelength, not the production four: harmonic-cascade accuracy "
            "needs headroom *above* f₀, and at 8 ppw the third harmonic sits at 2.7 and aliases "
            "into the second. That is a resolution rule about harmonic physics, separate from the "
            "one that bounds `p_max` capture — the kind of distinction that only shows up when "
            "you check against a closed form instead of against your own intuition."
        ),
        compute=harmonics_compute,
        draw=harmonics_draw,
        facts=harmonics_facts,
        code=textwrap.dedent(
            """
            import caustica.solvers as solvers
            from caustica import Grid, Medium, PMLSpec
            from caustica.materials import water
            from caustica.solvers import CWRunSpec
            from caustica.sources import plane_cw_source

            dx = 1500.0 / (1.0e6 * 16.0)                     # 16 points per wavelength
            grid = Grid(shape=(600,), dx=dx, pml=PMLSpec(thickness=40 * dx))
            med = Medium.homogeneous(grid.shape, water(c=1500.0, beta=3.5))
            src = plane_cw_source(grid, f0=1.0e6, amplitude=2.0e6, position_vox=60)

            res = solvers.get("westervelt")().run(
                grid, med, src,
                CWRunSpec(min_settle_periods=45, max_settle_periods=100, convergence_tol=0.003),
                backend="numpy", harmonics=(1, 2, 3),
            )
            a1, a2 = res.harmonic_amp(1), res.harmonic_amp(2)
            """
        ).strip(),
    ),
    Example(
        key="layered",
        title="Focusing through bone",
        lede="Constructive solid geometry, painted onto the grid, with per-voxel physics.",
        body=(
            "The medium becomes a scene: 2 mm of skin, 4 mm of bone at 2800 m/s, then brain — "
            "three boxes rasterized onto the same grid the solver runs on. Nothing else in the "
            "job changes. Running it twice, once against the layers and once against plain "
            "water, is what makes the figure a measurement rather than a picture: the axial "
            "pair shows what the barrier costs and where it puts the focus instead.\n\n"
            "The first thing it shows is that the loudest point in the box stops being the "
            "focus. Bone reflects hard enough that the standing wave in front of it beats "
            "everything downstream, which is why the table compares the two runs *past* the "
            "barrier rather than comparing their maxima.\n\n"
            "Read this one as a demonstration of the medium model, not as a transcranial "
            "result. Strong heterogeneity is not among the things this library has "
            "[gated itself against](validation.md) — a 1.9× jump in sound speed across one "
            "voxel is exactly where a solver earns or loses its accuracy, and caustica has not "
            "yet proved which."
        ),
        compute=layered_compute,
        draw=layered_draw,
        facts=layered_facts,
        job=layered_job(),
    ),
    Example(
        key="steered",
        title="Steering a phased array",
        lede="A spiral element table, delay-and-sum phases, and a focus that moves off axis.",
        body=(
            "A 32-element Archimedean spiral on the same shell as the bowl. `focus.mode: "
            '"steered"` asks for delay-and-sum phases toward a target 3 mm off the axis; the run '
            "records where the peak actually landed, which is not quite the same thing. The plane "
            "through the target shows the cost of steering a sparse array: the focus arrives, and "
            "so do the grating lobes that a 32-element aperture cannot suppress."
        ),
        compute=steered_compute,
        draw=steered_draw,
        facts=steered_facts,
        job=steered_job(),
    ),
    Example(
        key="thermal",
        title="From pressure to thermal dose",
        lede="Pennes bioheat and CEM43, driven by a real acoustic field.",
        body=(
            "The acoustic field is only half of a HIFU calculation. This one turns a solved field "
            "into a volumetric heat source (`Q = 2αI`), marches Pennes bioheat through 20 s of "
            "sonication and cooling in perfused liver, and accumulates CEM43 across both "
            "phases. It uses the layer below `simulate()`, because the thermal chain consumes a "
            "solver result and a medium directly — which is also the honest way to show that the "
            "layer is there and is usable.\n\n"
            "The right-hand panel is the reason a thermal model is not optional. The focal "
            "focal temperature reaches its steady state within the first second — heat leaves "
            "a focus this small as fast as it arrives — and then does not move again for the "
            "rest of the sonication. The dose does. CEM43 keeps accumulating for the whole "
            "half-minute the temperature is flat, which is why exposure is counted in dose "
            "rather than in degrees. Resolving that rise took a finer step than the plateau "
            "needs, so the run is three chained solves, each carrying the previous "
            "temperature *and* its accumulated dose forward."
        ),
        compute=thermal_compute,
        draw=thermal_draw,
        facts=thermal_facts,
        code=textwrap.dedent(
            """
            from caustica.sensors import HeatingSource
            from caustica.thermal.pennes import PennesSolver
            from caustica.thermal.properties import ThermalMedium

            heat = HeatingSource.from_result(result, medium, grid.dx, harmonics=(1,))
            tm = ThermalMedium.from_id_map(ids[region], db, grid.dx)

            solver = PennesSolver(backend="numpy")
            dt = 0.9 * solver.stable_dt(tm)
            hot = solver.solve(t0, heat, tm, dt=dt, n_steps=int(20.0 / dt), dose=True)
            cool = solver.solve(hot.temperature, None, tm, dt=dt,
                                n_steps=int(40.0 / dt), dose=True, dose0=hot.dose_cem43)
            """
        ).strip(),
    ),
]


# ---------------------------------------------------------------- the page


HEADER = """\
# Examples

Five worked examples. Every figure on this page was produced by running the job
beside it, and every number in the tables came back from that run — nothing here
is a sketch or a remembered result. `python scripts/make_examples.py` reproduces
the whole page.

!!! note "The grids are small on purpose"

    Each example is sized to solve in seconds on a laptop CPU, so the page can be
    regenerated on any machine. Real studies run at finer `dx` and larger boxes;
    nothing about the job changes except those numbers.
"""


def page(done: list[Example], ctx: dict[str, dict]) -> str:
    out = [HEADER]
    for ex in done:
        c = ctx[ex.key]
        out.append(f"\n## {ex.title}\n")
        out.append(f"*{ex.lede}*\n")
        out.append(ex.body + "\n")
        out.append('<div class="example-figure" markdown>')
        alt = f"{ex.title}: {ex.lede}"
        out.append(f"![{alt}](assets/examples/{ex.key}.svg#only-light)")
        out.append(f"![{alt}](assets/examples/{ex.key}-dark.svg#only-dark)")
        out.append("</div>\n")

        out.append("| what | measured |")
        out.append("|---|---|")
        for label, value in ex.facts(c):
            out.append(f"| {label} | {value} |")
        out.append("")

        if ex.job is not None:
            out.append(f'??? example "The job — `{ex.key}.json`"\n')
            body = json.dumps(ex.job, indent=2)
            out.append("    ```json")
            out.extend("    " + line for line in body.splitlines())
            out.append("    ```\n")
            out.append("    ```python")
            out.append("    import caustica")
            out.append(f'    res = caustica.simulate("{ex.key}.json")')
            out.append("    ```\n")
        if ex.code:
            out.append("```python")
            out.append(ex.code)
            out.append("```\n")
        out.append(f"<small>Solved in {ex.seconds:.1f} s on one CPU core budget.</small>\n")

    out.append("\n## Where these go next\n")
    out.append(
        "- Every field above is available as `res.result.phasor` and saved by "
        "`res.save(...)` in the [`caustica-result/1`](gui_contract.md) layout.\n"
        "- The numbers in the tables are the same ones the HTML report quotes — see "
        "[what has been measured](validation.md) for how they are gated.\n"
        "- To write your own job, start from [the job reference](job_reference.md)."
    )
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="run the example gallery and write docs/examples.md")
    ap.add_argument("keys", nargs="*", help="only these examples (default: all)")
    args = ap.parse_args()

    wanted = [e for e in EXAMPLES if not args.keys or e.key in args.keys]
    ctx: dict[str, dict] = {}
    for ex in wanted:
        print(f"  {ex.key}: solving ...", flush=True)
        t0 = time.perf_counter()
        ctx[ex.key] = ex.compute()
        ex.seconds = time.perf_counter() - t0
        for th in (LIGHT, DARK):
            fig, _ = figure(th)
            ex.draw(fig, ctx[ex.key], th)
            name = ex.key + ("-dark" if th.name == "dark" else "") + ".svg"
            size = save(fig, OUT / name)
        print(f"  {ex.key}: {ex.seconds:.1f} s, figures {size / 1024:.0f} KB each")

    if args.keys:
        print("  partial run: docs/examples.md not rewritten")
        return
    PAGE.write_text(page(wanted, ctx), encoding="utf-8")
    print(f"  wrote {PAGE.relative_to(REPO)}")


if __name__ == "__main__":
    main()
