#!/usr/bin/env python3
"""Render the README's "How to use caustica" flow diagram.

Writes self-contained SVGs into ``docs/assets/``::

    how-to-use.svg             light theme, schematic thumbnails
    how-to-use-dark.svg        dark  theme, schematic thumbnails
    how-to-use-real.svg        light theme, thumbnails from real caustica calls
    how-to-use-real-dark.svg   dark  theme, thumbnails from real caustica calls

Everything is vector and nothing is referenced from outside the file: the
left-hand thumbnails are matplotlib figures saved as SVG and inlined as nested
``<svg>`` elements. GitHub renders neither external references nor scripts
inside an SVG, so the diagram carries no raster payload, no ``<style>`` block
and no ``<image>``.

    python scripts/make_howto.py                 # all four
    python scripts/make_howto.py --schematic     # skip the ones that run caustica
    python scripts/make_howto.py --real          # only the caustica-backed ones
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "assets"

# text is emitted as paths so the thumbnails look identical everywhere, and the
# figure fonts never depend on what the reader's machine happens to have
matplotlib.rcParams["svg.fonttype"] = "path"
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["path.simplify_threshold"] = 1.0


# --------------------------------------------------------------------- theme


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    panel: str
    fig_bg: str
    stroke: str
    accent: str
    accent_soft: str
    text: str
    muted: str
    badge_text: str


LIGHT = Theme(
    name="light",
    bg="#FFFFFF",
    panel="#FFFFFF",
    fig_bg="#F6F8FA",
    stroke="#D5DDE5",
    accent="#1F4E79",
    accent_soft="#E8EFF6",
    text="#16202B",
    muted="#5A6B7B",
    badge_text="#FFFFFF",
)

DARK = Theme(
    name="dark",
    bg="#0D1117",
    panel="#111820",
    fig_bg="#0F161E",
    stroke="#26313D",
    accent="#7FB2E0",
    accent_soft="#16222F",
    text="#E6EDF3",
    muted="#9AA9B8",
    badge_text="#0D1117",
)


def mix(a: str, b: str, t: float) -> str:
    """Blend two ``#RRGGBB`` colours, ``t`` of the way from *a* to *b*."""
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#{:02X}{:02X}{:02X}".format(
        *tuple(round(x + (y - x) * t) for x, y in zip(ca, cb, strict=False))
    )


def cmap_of(th: Theme) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "caustica",
        [th.fig_bg, mix(th.fig_bg, th.accent, 0.5), th.accent],
    )


# ---------------------------------------------------------------- the layout

W = 940
MARGIN = 34
FIG = 144
GAP = 26
BOX_X = MARGIN + FIG + GAP
BOX_W = W - BOX_X - MARGIN
ROW_GAP = 40
HEAD_H = 122
FOOT_H = 62

SANS = (
    "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "Helvetica, Arial, sans-serif"
)
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

WRAP = 88  # characters per body line at 12.5px in the sans stack

CHIP_H = 20
CHIP_GAP = 6
CHIP_PAD = 9
CHIP_FS = 11
CHIP_CW = 6.3  # monospace advance at 11px


# ----------------------------------------------------------------- the steps

STEPS = [
    dict(
        key="install",
        title="Install it",
        lede="One pip install, and the whole library is importable.",
        chips=['pip install "caustica[gpu,report]"'],
        body=(
            "Pure Python: NumPy on the CPU, CuPy on the GPU. No compiler, no precompiled "
            "binaries, and the same code on a laptop and on a Colab A100."
        ),
    ),
    dict(
        key="space",
        title="Choose the space",
        lede="How many dimensions you need, how fine the grid is, how big the box is.",
        chips=["1-D", "2-D", "3-D", "voxel size", "box size", "absorbing border"],
        body=(
            "The absorbing border lives inside the box you asked for, not around it, so the "
            "domain you name is the domain you get. Array sources need the full 3-D grid."
        ),
    ),
    dict(
        key="medium",
        title="Import the anatomy, assign the materials",
        lede="Bring in a segmented volume, and give every label a material.",
        chips=[
            "volume_import",
            "medium_volume",
            "tissue library",
            "your own labels",
            "homogeneous",
            "scene",
        ],
        body=(
            "A labelled phantom — the UWCEM breast set, or your own segmentation — is "
            "resampled onto your grid, and each label turns into sound speed, density, "
            "absorption and B/A. Or skip the import: paint the box out of solids, or make it "
            "one material."
        ),
    ),
    dict(
        key="geometry",
        title="Build the geometry",
        lede="Draw the anatomy out of solids, the way you would in a CAD tool.",
        chips=["ball", "box", "cylinder", "ellipsoid", "half-space", "add / cut / intersect"],
        body=(
            "Shapes are implicit, so they land straight on the grid at whatever resolution you "
            "chose \u2014 no meshing step, no resampling. Or skip the drawing and import a "
            "volume that is already segmented."
        ),
    ),
    dict(
        key="source",
        title="Put a transducer in the box",
        lede="A focused bowl, a randomized spiral array, or your own element table.",
        chips=["bowl", "archimedean spiral", "element file"],
        body=(
            "You give the aperture, the radius of curvature and where the apex sits; caustica "
            "voxelizes the elements onto the same grid the medium lives on."
        ),
    ),
    dict(
        key="drive",
        title="Aim it and drive it",
        lede="Set the frequency and the drive pressure, then steer the focus electronically.",
        chips=["frequency", "amplitude", "per-element delays", "phase maps"],
        body=(
            "One convention is worth knowing before you read a number off a plot: the phasor is "
            "p(t) = Re{P\u00b7e^(\u2212i\u03c9t)}, and +z is the beam axis."
        ),
    ),
    dict(
        key="solver",
        title="Choose how much physics you want",
        lede="Linear when you only need the beam; Westervelt when nonlinearity is the point.",
        chips=["linear", "westervelt", "harmonics", "k-Wave cross-check"],
        body=(
            "Both are k-space pseudospectral solvers, marched to a continuous-wave steady "
            "state. Ask for the harmonics you actually want back, and nothing else is stored."
        ),
    ),
    dict(
        key="plan",
        title="Find out what it will cost, first",
        lede="Whether it fits in memory and how long it takes, answered before anything starts.",
        chips=["caustica validate job.json"],
        body=(
            "The planner times one step, estimates VRAM and wall-clock, and refuses a run that "
            "will not fit or that a CPU would take hours over. The refusal names the fix for "
            "the machine you are on."
        ),
    ),
    dict(
        key="run",
        title="Run it",
        lede="A command, a function call or a Colab cell \u2014 the same run behind all three.",
        chips=["caustica run job.json", "caustica.simulate(job)", "colab.run_job(job)"],
        body=(
            "Same planner, same gates, same exit codes whichever door you come in through. An "
            "interrupted run resumes where it stopped, and progress is a callback if you would "
            "rather draw it yourself."
        ),
    ),
    dict(
        key="report",
        title="Read the result",
        lede="Numbers you can quote, and a report you can hand to somebody else.",
        chips=["focal metrics", "\u22126 dB spot", "HTML report", "preview package"],
        body=(
            "Everything lands in one run folder with a stable layout. Five extension points "
            "\u2014 solver, medium kind, array kind, backend, renderer \u2014 take plugins "
            "over entry-point names that will not change."
        ),
    ),
]


# ------------------------------------------------------- real caustica calls

_CACHE: dict = {}


def _quiet() -> None:
    logging.disable(logging.WARNING)
    warnings.simplefilter("ignore")


def real_run() -> dict:
    """One small water-bowl run, shared by the last three thumbnails."""
    if "run" in _CACHE:
        return _CACHE["run"]
    _quiet()
    import caustica

    job = {
        "format": "caustica-job/1",
        "kind": "explicit",
        "name": "howto",
        "medium": {"kind": "homogeneous"},
        "grid": {
            "ndim": 3,
            "dx_mm": 0.25,
            "size_mm": [20, 20, 34],
            "pml": {"thickness_mm": 2.5},
        },
        "source": {
            "kind": "array",
            "array": {"kind": "bowl", "d_outer_mm": 14.0, "roc_mm": 16.0},
            "apex_mm": [10.0, 10.0, 5.0],
        },
        "drive": {"f0_mhz": 1.0, "amplitude_kpa": 100.0},
        "run": {"spec": {"min_settle_periods": 2, "max_settle_periods": 8}, "harmonics": [1]},
        "solver": "linear",
    }
    print("  running the real job (a few seconds on CPU) ...")
    res = caustica.simulate(job, out=None, progress=None)
    _CACHE["run"] = {
        "amp": np.abs(res.result.phasor),
        "dx_mm": 0.25,
        "pml_vox": int(round(2.5 / 0.25)),
        "plan": res.plan or {},
    }
    return _CACHE["run"]


def real_array():
    if "arr" not in _CACHE:
        _quiet()
        from caustica.arrays.transducer import archimedean_spiral

        _CACHE["arr"] = archimedean_spiral(n_elements=128, d_outer=0.10, d_inner=0.044, roc=0.10)
    return _CACHE["arr"]


# ---------------------------------------------------------------- thumbnails
#
# every one takes (ax, theme, real) and draws a square; `real` picks between a
# drawn stand-in and the same picture made by calling caustica.


def _square(ax, lo: float = 0.0, hi: float = 1.0) -> None:
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.axis("off")


# -- isometric block ---------------------------------------------------------
#
# Two steps show a 3-D box cut open at its mid-planes: the space you solve on,
# and the anatomy you import into it. Same projection, same three cut faces, so
# the two thumbnails read as the same object seen twice.

_COS30 = float(np.cos(np.pi / 6))

#: the three cut faces, as (s, t) -> (x, y, z) on the unit cube
_FACES = {
    "top": lambda s, t: (s, t, np.ones_like(s)),
    "left": lambda s, t: (np.zeros_like(s), s, 1 - t),
    "right": lambda s, t: (s, np.zeros_like(s), 1 - t),
}

_FACE_CORNERS = {
    "top": [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    "left": [(0, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 0)],
    "right": [(0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)],
}

#: only the edges a viewer at (-1, -1, +1) can actually see
_VISIBLE_EDGES = [
    ((0, 0, 1), (1, 0, 1)),
    ((0, 0, 1), (0, 1, 1)),
    ((1, 0, 1), (1, 1, 1)),
    ((0, 1, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 0, 1)),
    ((1, 0, 0), (1, 0, 1)),
    ((0, 1, 0), (0, 1, 1)),
    ((0, 0, 0), (1, 0, 0)),
    ((0, 0, 0), (0, 1, 0)),
]

_ALL_EDGES = _VISIBLE_EDGES + [
    ((1, 1, 0), (1, 1, 1)),
    ((1, 1, 0), (1, 0, 0)),
    ((1, 1, 0), (0, 1, 0)),
]


def iso(x, y, z):
    """Isometric projection of the unit cube; +z is up, the near corner is (0,0,1)."""
    return (x - y) * _COS30, z + (x + y) * 0.5


def face_uv(kind: str, n0: int, n1: int):
    s = np.linspace(0, 1, n0)[:, None] * np.ones((1, n1))
    t = np.ones((n0, 1)) * np.linspace(0, 1, n1)[None, :]
    return iso(*_FACES[kind](s, t))


def draw_block(ax, th: Theme, faces: dict[str, str] | None = None) -> None:
    """The three cut faces as flat panels, so the box reads as a solid."""
    faces = faces or {
        "top": mix(th.fig_bg, th.accent, 0.10),
        "left": mix(th.fig_bg, th.accent, 0.20),
        "right": mix(th.fig_bg, th.accent, 0.04),
    }
    for kind, shade in faces.items():
        ax.add_patch(
            Polygon([iso(*p) for p in _FACE_CORNERS[kind]], closed=True, fc=shade, ec="none")
        )


def draw_edges(ax, th, edges, *, lo=0.0, hi=1.0, color=None, lw=1.1, ls="solid") -> None:
    for a, b in edges:
        p0 = iso(*(lo + (hi - lo) * np.asarray(a, float)))
        p1 = iso(*(lo + (hi - lo) * np.asarray(b, float)))
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color or th.accent, lw=lw, ls=ls)


def _frame_block(ax) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-_COS30 * 1.10, _COS30 * 1.10)
    ax.set_ylim(-0.26, 2.10)


def fig_install(ax, th: Theme, real: bool) -> None:
    _square(ax)
    ax.add_patch(
        FancyBboxPatch(
            (0.10, 0.26),
            0.80,
            0.48,
            boxstyle="round,pad=0,rounding_size=0.05",
            fc=th.accent_soft,
            ec=th.accent,
            lw=1.1,
        )
    )
    for x in (0.17, 0.23, 0.29):
        ax.add_patch(Circle((x, 0.655), 0.017, fc=th.accent, ec="none", alpha=0.5))
    ax.plot([0.10, 0.90], [0.595, 0.595], color=th.accent, lw=0.8, alpha=0.45)
    ax.text(
        0.16,
        0.45,
        "$ pip install",
        color=th.accent,
        fontsize=7.0,
        family="monospace",
        fontweight="bold",
    )
    ax.text(0.16, 0.34, "    caustica", color=th.accent, fontsize=7.0, family="monospace")


def fig_space(ax, th: Theme, real: bool) -> None:
    """The box, cut open: voxels in the middle, the absorbing shell around them."""
    n, w = 56, 9
    if real:
        from caustica.core.pml import sponge_profile_1d

        prof = np.asarray(sponge_profile_1d(n, w))
    else:
        d = np.minimum(np.arange(n), n - 1 - np.arange(n))
        prof = np.clip(d / (w - 1.0), 0, 1) ** 2
    frame = 1.0 - np.minimum.outer(prof, prof)

    _frame_block(ax)
    draw_block(ax, th)

    # the shell, on each cut face, shaded by how hard it is absorbing
    for kind in ("top", "left", "right"):
        u, v = face_uv(kind, n, n)
        ax.contourf(u, v, frame, levels=np.linspace(0.03, 1.0, 6), cmap=cmap_of(th))

    # voxels, drawn only where the solver is actually solving
    edge = w / (n - 1.0)
    for kind in ("top", "left", "right"):
        for q in np.linspace(edge, 1 - edge, 6):
            for line in (
                [(q, r) for r in np.linspace(edge, 1 - edge, 2)],
                [(r, q) for r in np.linspace(edge, 1 - edge, 2)],
            ):
                pts = [iso(*_FACES[kind](np.array(a), np.array(b))) for a, b in line]
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    color=th.stroke,
                    lw=0.45,
                    alpha=0.9,
                )

    draw_edges(ax, th, _ALL_EDGES, lo=edge, hi=1 - edge, color=th.accent, lw=0.9, ls=(0, (3, 2)))
    draw_edges(ax, th, _VISIBLE_EDGES, lw=1.2)

    # one cell, called out: the voxel size is the other half of this choice
    cell = (1 - 2 * edge) / 6
    quad = [(edge, edge), (edge + cell, edge), (edge + cell, edge + cell), (edge, edge + cell)]
    ax.add_patch(
        Polygon(
            [iso(*_FACES["top"](np.array(a), np.array(b))) for a, b in quad],
            closed=True,
            fc=th.accent,
            ec="none",
        )
    )
    ax.text(
        *np.add(iso(*_FACES["top"](np.array(edge + 1.6 * cell), np.array(edge))), (0.0, -0.02)),
        "dx",
        color=th.accent,
        fontsize=6.0,
        family="monospace",
    )


#: three mid-plane slices of a real UWCEM breast phantom, cached next to the
#: SVGs so a rebuild does not need the (git-ignored, multi-GB) phantom dataset
PHANTOM_CACHE = OUT / "howto-phantom.npz"


def phantom_faces() -> dict | None:
    """Label slices + the label->sound-speed table, from the cache or the dataset."""
    if "phantom" in _CACHE:
        return _CACHE["phantom"]
    if not PHANTOM_CACHE.exists():
        _CACHE["phantom"] = _extract_phantom()
    else:
        d = np.load(PHANTOM_CACHE, allow_pickle=True)
        _CACHE["phantom"] = {
            "ax": d["ax"],
            "sa": d["sa"],
            "co": d["co"],
            "table": {int(k): v for k, v in json.loads(str(d["table"])).items()},
        }
    return _CACHE["phantom"]


def _extract_phantom() -> dict | None:
    """Cut three mid-planes out of a packed phantom and cache them (~50 KB)."""
    found = sorted((REPO / "data" / "phantoms").glob("uwcem-*.npz"))
    if not found:
        print(f"  no phantom dataset: {PHANTOM_CACHE.name} not rebuilt, drawing a stand-in")
        return None
    print(f"  reading {found[0].name} (labels only) ...")
    z = np.load(found[0], allow_pickle=True)
    labels = z["labels"]
    table = json.loads(str(z["materials"]))["materials"]
    c_of = {int(k): float(v["c"]) for k, v in table.items()}

    nz = np.argwhere(labels != 0)
    lo, hi = nz.min(0), nz.max(0)
    mid = (lo + hi) // 2
    k = 4
    out = {
        "ax": labels[lo[0] : hi[0] : k, lo[1] : hi[1] : k, mid[2]],
        "co": labels[lo[0] : hi[0] : k, mid[1], lo[2] : hi[2] : k],
        "sa": labels[mid[0], lo[1] : hi[1] : k, lo[2] : hi[2] : k],
        "table": c_of,
    }
    PHANTOM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PHANTOM_CACHE,
        ax=out["ax"].astype(np.int8),
        co=out["co"].astype(np.int8),
        sa=out["sa"].astype(np.int8),
        table=json.dumps(c_of),
    )
    print(f"  cached {PHANTOM_CACHE.name} ({PHANTOM_CACHE.stat().st_size / 1024:.0f} KB)")
    return out


def _block_mean(a: np.ndarray, k: int) -> np.ndarray:
    n0, n1 = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:n0, :n1].reshape(n0 // k, k, n1 // k, k).mean(axis=(1, 3))


def _synthetic_phantom() -> dict:
    """A stand-in with the same composition, for when the dataset is not there."""
    rng = np.random.default_rng(7)
    faces = {}
    for key, shape in (("ax", (104, 112)), ("co", (104, 100)), ("sa", (112, 100))):
        n0, n1 = shape
        u = np.linspace(-1, 1, n0)[:, None]
        v = np.linspace(-1, 1, n1)[None, :]
        r = np.sqrt((u / 0.82) ** 2 + (v / 0.9) ** 2)
        lab = np.where(r < 1.0, 7, 0)
        lab = np.where((r > 0.93) & (r < 1.0), 1, lab)
        for cx, cy, rad, val in rng.permutation(
            [
                [-0.3, 0.1, 0.30, 3],
                [0.25, -0.2, 0.26, 4],
                [0.05, 0.35, 0.20, 5],
                [0.4, 0.3, 0.16, 6],
            ]
        ):
            blob = ((u - cx) ** 2 + (v - cy) ** 2) < rad**2
            lab = np.where(blob & (r < 0.92), int(val), lab)
        faces[key] = lab
    faces["table"] = {
        0: 1522.5,
        1: 1627.5,
        3: 1555.0,
        4: 1537.5,
        5: 1520.0,
        6: 1500.0,
        7: 1475.0,
    }
    return faces


def fig_medium(ax, th: Theme, real: bool) -> None:
    """An imported anatomy, cut open, with a material behind every label."""
    data = phantom_faces() if real else None
    if data is None:
        data = _synthetic_phantom()
    table = data["table"]

    _frame_block(ax)
    draw_block(ax, th)

    cmap = cmap_of(th)
    for key, kind in (("ax", "top"), ("sa", "left"), ("co", "right")):
        labels = np.asarray(data[key])
        c = np.vectorize(table.get)(labels).astype(float)
        tissue = (labels != 0).astype(float)
        # average the assigned sound speed over 3x3 blocks: at 144 px the raw
        # voxel speckle is noise, and the contours would cost 10x the bytes
        occ = _block_mean(tissue, 3)
        cc = np.divide(_block_mean(c * tissue, 3), occ, out=np.zeros_like(occ), where=occ > 0)
        cc[occ < 0.35] = np.nan
        u, v = face_uv(kind, *cc.shape)
        ax.contourf(u, v, cc, levels=np.linspace(1440, 1630, 7), cmap=cmap, extend="both")

    draw_edges(ax, th, _VISIBLE_EDGES, lw=1.2)
    ax.text(
        0.0,
        -0.20,
        "labels → materials",
        color=th.accent,
        fontsize=6.2,
        ha="center",
        family="monospace",
    )


def fig_geometry(ax, th: Theme, real: bool) -> None:
    """One solid, and the primitives it was welded and cut out of."""
    r = 0.021
    n = 190
    g = np.linspace(-r, r, n)
    X, Y = np.meshgrid(g, g)
    ang = 0.32

    if real:
        from caustica.geometry.shapes import Ball, Box

        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        a = Ball(center=(-0.006, -0.001), radius=0.011)
        b = Box(center=(0.007, 0.005), size=(0.017, 0.013)).rotated(ang)
        cut = Ball(center=(0.004, -0.007), radius=0.0065)

        def mask(shape) -> np.ndarray:
            return np.asarray(shape.contains(pts)).reshape(n, n).astype(float)

        ma, mb, mc, solid = mask(a), mask(b), mask(cut), mask((a | b) - cut)
    else:
        xr = (X - 0.007) * np.cos(ang) + (Y - 0.005) * np.sin(ang)
        yr = -(X - 0.007) * np.sin(ang) + (Y - 0.005) * np.cos(ang)
        ma = ((X + 0.006) ** 2 + (Y + 0.001) ** 2 < 0.011**2).astype(float)
        mb = ((np.abs(xr) < 0.0085) & (np.abs(yr) < 0.0065)).astype(float)
        mc = ((X - 0.004) ** 2 + (Y + 0.007) ** 2 < 0.0065**2).astype(float)
        solid = np.clip(ma + mb, 0, 1) * (1 - mc)

    _square(ax, -r, r)
    ax.contourf(X, Y, solid, levels=[0.5, 1.5], colors=[mix(th.fig_bg, th.accent, 0.32)])
    for m in (ma, mb, mc):
        ax.contour(
            X, Y, m, levels=[0.5], colors=[th.muted], linewidths=0.7, linestyles=[(0, (2, 2))]
        )
    ax.contour(X, Y, solid, levels=[0.5], colors=[th.accent], linewidths=1.6)
    ax.text(
        0.0,
        -r * 0.90,
        "a | b  -  c",
        color=th.accent,
        fontsize=6.2,
        ha="center",
        family="monospace",
    )


def fig_source(ax, th: Theme, real: bool) -> None:
    if real:
        arr = real_array()
        xy = np.asarray(arr.positions)[:, :2]
        rad = float(arr.elem_radius)
    else:
        k = np.arange(128)
        t = np.sqrt(k / 128.0)
        rr = 0.022 + t * (0.048 - 0.022)
        a = k * np.pi * (3 - np.sqrt(5))
        xy = np.stack([rr * np.cos(a), rr * np.sin(a)], axis=1)
        rad = 0.0045
    _square(ax, -0.055, 0.055)
    ax.add_patch(Circle((0, 0), 0.050, fc="none", ec=th.stroke, lw=1.0, ls=(0, (3, 2))))
    ax.add_patch(Circle((0, 0), 0.022, fc="none", ec=th.stroke, lw=1.0, ls=(0, (3, 2))))
    # one <defs> path reused by every element, instead of 128 circle paths
    span = 0.11
    ms = 2 * rad / span * (1.6 * 0.88 * 72)
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        linestyle="none",
        marker="o",
        markersize=ms,
        markerfacecolor=th.accent,
        markeredgecolor="none",
        alpha=0.82,
    )


def fig_drive(ax, th: Theme, real: bool) -> None:
    """The focus, moved off the axis by delays alone -- and where it sits without them."""
    # a square window in millimetres, centred on the steered focus, so the beam
    # keeps its true 1:7 proportions instead of being squeezed into a needle
    x0, x1, z0, z1 = -20.0, 20.0, 72.0, 112.0
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(x0, x1)
    ax.set_ylim(z0, z1)

    natural = (0.0, 100.0)
    steered = (6.0, 92.0)

    if real:
        arr = real_array()
        f0 = 0.5e6
        nx = nz = 120
        xx, zz = np.meshgrid(np.linspace(x0, x1, nx) * 1e-3, np.linspace(z0, z1, nz) * 1e-3)
        pts = np.stack([xx.ravel(), np.zeros(xx.size), zz.ravel()], axis=1)

        def field(target):
            ph = arr.das_phases(np.asarray(target), f0=f0)
            p = np.abs(np.asarray(arr.rayleigh_preview(pts, f0, phases=ph))).reshape(nz, nx)
            return 20 * np.log10(np.maximum(p / p.max(), 1e-3))

        db_steered = field([steered[0] * 1e-3, 0.0, steered[1] * 1e-3])
        db_natural = field([natural[0] * 1e-3, 0.0, natural[1] * 1e-3])
        xm, zm = xx * 1e3, zz * 1e3
    else:
        xm, zm = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(z0, z1, 200))

        def lobe(fx, fz):
            amp = np.exp(-(((xm - fx) / 1.7) ** 2) - ((zm - fz) / 6.0) ** 2)
            amp += 0.25 * np.exp(-(((xm - fx) / 5.0) ** 2) - ((zm - fz) / 11.0) ** 2)
            return 20 * np.log10(np.maximum(amp / amp.max(), 1e-3))

        db_steered = lobe(*steered)
        db_natural = lobe(*natural)

    ax.contourf(xm, zm, db_steered, levels=np.linspace(-12, 0, 7), cmap=cmap_of(th))

    # where the focus sits with no delays at all, for comparison
    ax.contour(
        xm,
        zm,
        db_natural,
        levels=[-6.0],
        colors=[th.muted],
        linewidths=0.9,
        linestyles=[(0, (3, 2))],
    )
    ax.plot(*natural, marker="+", ms=6, mew=1.0, color=th.muted)

    # the beam axis, and the move away from it
    ax.plot([0, 0], [z0, z1], color=th.muted, lw=0.6, ls=(0, (4, 3)), alpha=0.8)
    ax.annotate(
        "",
        xy=steered,
        xytext=natural,
        arrowprops=dict(arrowstyle="-|>", color=th.accent, lw=1.1, shrinkA=5, shrinkB=5),
    )
    ax.plot(*steered, marker="+", ms=7, mew=1.3, color=th.bg)


def fig_solver(ax, th: Theme, real: bool) -> None:
    ax.set_aspect("auto")
    ax.axis("off")
    if real:
        from caustica.analytic.planewave import fubini_harmonic

        s = np.linspace(0.02, 1.0, 240)
        for n, a in ((1, 1.0), (2, 0.72), (3, 0.5)):
            y = np.asarray(fubini_harmonic(n, s))
            ax.plot(s, y, color=th.accent, alpha=a, lw=1.6)
            ax.text(
                1.03,
                float(y[-1]),
                str(n),
                color=th.accent,
                alpha=max(a, 0.6),
                fontsize=6.4,
                va="center",
            )
        ax.set_xlim(0, 1.16)
        ax.set_ylim(-0.03, 1.05)
    else:
        t = np.linspace(0, 2 * np.pi, 500)
        sine = np.sin(t)
        shock = sum((-1) ** (n + 1) * np.sin(n * t) / n for n in range(1, 12)) * (2 / np.pi)
        for y, off, a in ((sine, 0.70, 0.55), (shock, 0.24, 1.0)):
            ax.plot(t, off + 0.16 * y / np.abs(y).max(), color=th.accent, lw=1.7, alpha=a)
        ax.annotate(
            "",
            xy=(np.pi, 0.40),
            xytext=(np.pi, 0.54),
            arrowprops=dict(arrowstyle="-|>", color=th.muted, lw=1.0),
        )
        ax.text(0.10, 0.93, "linear", color=th.muted, fontsize=6.2)
        ax.text(0.10, 0.02, "westervelt", color=th.accent, fontsize=6.2)
        ax.set_xlim(-0.2, 2 * np.pi + 0.2)
        ax.set_ylim(0, 1)


def fig_plan(ax, th: Theme, real: bool) -> None:
    ax.set_aspect("auto")
    ax.axis("off")
    if real:
        plan = real_run()["plan"]
        vals = [float(plan.get("t_expected_s") or 0.0), float(plan.get("gpu_t_expected_s") or 0.0)]
        names = ["this CPU", str(plan.get("gpu", "A100"))]
        note = f"{float(plan.get('vram_gib', 0.0)):.2f} GiB — fits"
    else:
        vals = [8.0, 3.0]
        names = ["this CPU", "A100-40GB"]
        note = "0.07 GiB — fits"
    span = max(vals) or 1.0
    ax.barh([1, 0], vals, height=0.42, color=[mix(th.fig_bg, th.accent, 0.45), th.accent])
    for i, (v, label) in enumerate(zip(vals, names, strict=False)):
        y = 1 - i
        ax.text(0.02 * span, y + 0.30, label, color=th.text, fontsize=6.2, va="bottom")
        ax.text(v + 0.03 * span, y, f"{v:.1f} s", color=th.muted, fontsize=6.2, va="center")
    ax.text(0.0, -0.72, note, color=th.accent, fontsize=6.4, family="monospace")
    ax.set_xlim(0, span * 1.34)
    ax.set_ylim(-1.0, 1.7)


def fig_run(ax, th: Theme, real: bool) -> None:
    ax.set_aspect("auto")
    ax.axis("off")
    if real:
        d = real_run()
        amp = d["amp"]
        ix, iy, iz = np.unravel_index(np.argmax(amp), amp.shape)
        w = d["pml_vox"]
        line = amp[ix, iy, w:-w]
        z = (np.arange(line.size) + w) * d["dx_mm"]
        y = line / line.max()
        zf = iz * d["dx_mm"]
    else:
        z = np.linspace(0, 28, 400)
        y = np.exp(-(((z - 17.5) / 3.0) ** 2)) * (0.35 + 0.65 * np.tanh(z / 6))
        y = y / y.max()
        zf = 17.5
    ax.plot(z, y, color=th.accent, lw=1.7)
    ax.fill_between(z, 0, y, color=th.accent, alpha=0.13)
    ax.plot([zf, zf], [0, 1.0], color=th.muted, lw=0.8, ls=(0, (3, 2)))
    ax.text(zf, 1.05, "focus", color=th.muted, fontsize=6.2, ha="center")
    ax.text(z[-1], -0.14, "z [mm]", color=th.muted, fontsize=6.0, ha="right")
    ax.set_xlim(z[0], z[-1])
    ax.set_ylim(-0.18, 1.22)


def fig_report(ax, th: Theme, real: bool) -> None:
    half = 4.5
    if real:
        d = real_run()
        amp = d["amp"]
        ix, iy, iz = np.unravel_index(np.argmax(amp), amp.shape)
        plane = amp[:, iy, :] / amp.max()
        x = (np.arange(plane.shape[0]) - ix) * d["dx_mm"]
        z = (np.arange(plane.shape[1]) - iz) * d["dx_mm"]
        zz, xx = np.meshgrid(z, x)
        field = 20 * np.log10(np.maximum(plane, 1e-3))
    else:
        g = np.linspace(-half, half, 130)
        zz, xx = np.meshgrid(g, g)
        lobe = np.exp(-((zz / 3.4) ** 2) - (xx / 1.35) ** 2)
        field = 20 * np.log10(np.maximum(lobe, 1e-3))
    levels = np.linspace(-12, 0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.contourf(zz, xx, field, levels=levels, cmap=cmap_of(th))
    ax.contour(zz, xx, field, levels=[-6.0], colors=[th.bg], linewidths=1.1)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.text(
        0.0,
        -half * 0.86,
        "−6 dB",
        color=th.muted,
        fontsize=6.2,
        ha="center",
        family="monospace",
    )


THUMBS = {
    "install": fig_install,
    "space": fig_space,
    "medium": fig_medium,
    "geometry": fig_geometry,
    "source": fig_source,
    "drive": fig_drive,
    "solver": fig_solver,
    "plan": fig_plan,
    "run": fig_run,
    "report": fig_report,
}


# ------------------------------------------------------------- svg plumbing

_ROOT = re.compile(r'<svg[^>]*viewBox="([^"]+)"[^>]*>', re.S)


def inline_figure(fig, x: float, y: float, width: float, height: float, salt: str) -> str:
    """A matplotlib figure as a nested <svg> element, ready to drop into a page.

    The figure must already have been drawn with ``svg.hashsalt`` set to *salt*:
    matplotlib mints its ``<defs>`` ids from that salt, and two figures sharing
    a salt would collide once they are inlined into the same document.
    """
    buf = io.StringIO()
    fig.savefig(buf, format="svg", transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    s = re.sub(r"<\?xml.*?\?>", "", s, flags=re.S)
    s = re.sub(r"<!DOCTYPE.*?>", "", s, flags=re.S)
    s = re.sub(r"<metadata>.*?</metadata>", "", s, flags=re.S)
    m = _ROOT.search(s)
    if m is None:  # pragma: no cover - matplotlib changed its preamble
        raise RuntimeError("could not find the matplotlib <svg> root")
    body = _round_floats(s[m.end() : s.rindex("</svg>")])
    return (
        f'<svg x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'viewBox="{m.group(1)}" preserveAspectRatio="xMidYMid meet">{body}</svg>'
    )


def thumb_svg(key: str, th: Theme, real: bool, x: float, y: float, size: float) -> str:
    """One matplotlib thumbnail, inlined as a nested <svg> at (x, y)."""
    salt = f"{key}-{th.name}-{int(real)}"
    matplotlib.rcParams["svg.hashsalt"] = salt
    fig = plt.figure(figsize=(1.6, 1.6), dpi=100)
    ax = fig.add_axes((0.06, 0.06, 0.88, 0.88))
    THUMBS[key](ax, th, real)
    return inline_figure(fig, x, y, size, size, salt)


def _round_floats(s: str) -> str:
    """Trim matplotlib's coordinate precision; 0.01 pt is far below a pixel."""
    return re.sub(r"\d+\.\d{3,}", lambda m: f"{float(m.group()):.2f}", s)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(
    x: float,
    y: float,
    s: str,
    *,
    size: float,
    fill: str,
    family: str = SANS,
    weight: str = "normal",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>'
    )


def chip_rows(chips, width: float):
    """Pack the little option pills into rows no wider than *width*."""
    rows, row, x = [], [], 0.0
    for c in chips:
        w = len(c) * CHIP_CW + 2 * CHIP_PAD
        if row and x + w > width:
            rows.append(row)
            row, x = [], 0.0
        row.append((c, w))
        x += w + CHIP_GAP
    if row:
        rows.append(row)
    return rows


def box_height(step: dict) -> float:
    h = 82 + 17 * len(textwrap.wrap(step["body"], WRAP))
    rows = chip_rows(step.get("chips", []), BOX_W - 72)
    if rows:
        h += 4 + len(rows) * (CHIP_H + CHIP_GAP)
    return h


def build(th: Theme, real: bool) -> str:
    rows = []
    y = HEAD_H
    for step in STEPS:
        h = box_height(step)
        rows.append((step, y, max(FIG, h), h))
        y += max(FIG, h) + ROW_GAP
    total = y - ROW_GAP + FOOT_H

    o: list[str] = []
    o.append(
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{total:.0f}" viewBox="0 0 {W} {total:.0f}" '
        'role="img" aria-label="How to use caustica, in ten steps">'
    )
    o.append(f'<rect width="{W}" height="{total:.0f}" fill="{th.bg}"/>')

    # header
    o.append(text(MARGIN, 52, "How to use", size=27, fill=th.text, weight="700"))
    o.append(text(MARGIN + 162, 52, "caustica", size=27, fill=th.accent, weight="700"))
    o.append(
        text(
            MARGIN,
            80,
            "Ten steps from an empty shell to a focal metric — the same ten decisions whether "
            "you write a job file,",
            size=13,
            fill=th.muted,
        )
    )
    o.append(
        text(
            MARGIN,
            98,
            "call simulate() from Python, or run it in a Colab cell.",
            size=13,
            fill=th.muted,
        )
    )
    o.append(
        f'<line x1="{MARGIN}" y1="{HEAD_H - 14}" x2="{W - MARGIN}" y2="{HEAD_H - 14}" '
        f'stroke="{th.stroke}" stroke-width="1"/>'
    )

    spine = BOX_X + 30
    for i, (step, ry, rh, bh) in enumerate(rows):
        fy = ry + (rh - FIG) / 2
        by = ry + (rh - bh) / 2

        # thumbnail card
        o.append(
            f'<rect x="{MARGIN}" y="{fy:.1f}" width="{FIG}" height="{FIG}" rx="10" '
            f'fill="{th.fig_bg}" stroke="{th.stroke}" stroke-width="1"/>'
        )
        o.append(thumb_svg(step["key"], th, real, MARGIN + 6, fy + 6, FIG - 12))

        # step card
        o.append(
            f'<rect x="{BOX_X}" y="{by:.1f}" width="{BOX_W}" height="{bh:.1f}" rx="10" '
            f'fill="{th.panel}" stroke="{th.stroke}" stroke-width="1"/>'
        )
        o.append(f'<circle cx="{spine}" cy="{by + 30:.1f}" r="13" fill="{th.accent}"/>')
        o.append(
            text(
                spine,
                by + 34.5,
                str(i + 1),
                size=12.5,
                fill=th.badge_text,
                weight="700",
                anchor="middle",
            )
        )
        tx = BOX_X + 56
        o.append(text(tx, by + 26, step["title"], size=15.5, fill=th.text, weight="700"))
        o.append(text(tx, by + 48, step["lede"], size=13.5, fill=th.accent, weight="600"))
        lines = textwrap.wrap(step["body"], WRAP)
        for j, line in enumerate(lines):
            o.append(text(tx, by + 70 + 17 * j, line, size=12.5, fill=th.muted))
        cy = by + 70 + 17 * len(lines) - 8
        for row in chip_rows(step.get("chips", []), BOX_W - 72):
            cx = tx
            for label, cw in row:
                o.append(
                    f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cw:.1f}" height="{CHIP_H}" '
                    f'rx="{CHIP_H / 2:.1f}" fill="{th.accent_soft}" '
                    f'stroke="{th.stroke}" stroke-width="1"/>'
                )
                o.append(
                    text(cx + CHIP_PAD, cy + 14, label, size=CHIP_FS, fill=th.accent, family=MONO)
                )
                cx += cw + CHIP_GAP
            cy += CHIP_H + CHIP_GAP

        # arrow down to the next card
        if i + 1 < len(rows):
            nstep, nry, nrh, nbh = rows[i + 1]
            y0 = by + bh + 6
            y1 = nry + (nrh - nbh) / 2 - 6
            o.append(
                f'<line x1="{spine}" y1="{y0:.1f}" x2="{spine}" y2="{y1 - 7:.1f}" '
                f'stroke="{th.accent}" stroke-width="2" opacity="0.55"/>'
            )
            o.append(
                f'<path d="M {spine - 5} {y1 - 8:.1f} L {spine + 5} {y1 - 8:.1f} '
                f'L {spine} {y1:.1f} Z" fill="{th.accent}" opacity="0.55"/>'
            )

    fy = total - FOOT_H + 22
    o.append(
        f'<line x1="{MARGIN}" y1="{fy - 22:.1f}" x2="{W - MARGIN}" y2="{fy - 22:.1f}" '
        f'stroke="{th.stroke}" stroke-width="1"/>'
    )
    o.append(
        text(
            MARGIN,
            fy,
            "docs/job_reference.md  ·  docs/conventions.md  ·  docs/extending.md  "
            "·  docs/gui_contract.md",
            size=12,
            fill=th.muted,
            family=MONO,
        )
    )
    o.append(
        text(
            W - MARGIN,
            fy,
            "github.com/ebx0/caustica",
            size=12,
            fill=th.accent,
            family=MONO,
            anchor="end",
        )
    )
    o.append("</svg>")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + "\n".join(o) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="render docs/assets/how-to-use*.svg")
    ap.add_argument("--schematic", action="store_true", help="only the drawn thumbnails")
    ap.add_argument("--real", action="store_true", help="only the caustica-backed thumbnails")
    args = ap.parse_args()

    wants = []
    if not args.real:
        wants.append(False)
    if not args.schematic:
        wants.append(True)

    OUT.mkdir(parents=True, exist_ok=True)
    for real in wants:
        for th in (LIGHT, DARK):
            name = (
                "how-to-use"
                + ("-real" if real else "")
                + ("-dark" if th.name == "dark" else "")
                + ".svg"
            )
            path = OUT / name
            svg = build(th, real)
            path.write_text(svg, encoding="utf-8")
            print(f"  wrote {path.relative_to(REPO)}  ({len(svg) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
