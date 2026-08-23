#!/usr/bin/env python3
"""Render the documentation site's hero: a real 2-D section, animating.

Writes two self-contained animated SVGs into ``docs/assets/``::

    hero-field.svg        light theme
    hero-field-dark.svg   dark  theme

What is on screen is one x–z plane through a real solve — a focused bowl in
water, the same k-space solver the library ships — drawn twice over:

* the **envelope** ``|P|``, static, in the single-hue palette the how-to-use
  diagram uses;
* the **wavefronts**, ``Re{P·e^(−iωt)}`` sampled at N phases of one acoustic
  period, one SVG layer each, cycled with SMIL.

Because the frames are one full period of the *same* phasor, the loop closes
exactly: there is no seam to hide and no easing to fake.

The animation is SMIL (``<animate>`` on opacity), not CSS, so it needs no
``<style>`` element and survives being loaded as a plain ``<img>``.

    python scripts/make_hero.py             # from the cached field
    python scripts/make_hero.py --resolve   # re-run the solve, refresh the cache
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
from make_howto import DARK, LIGHT, OUT, REPO, Theme, cmap_of, inline_figure, mix  # noqa: E402

matplotlib.rcParams["svg.fonttype"] = "path"
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["path.simplify_threshold"] = 1.0

#: the solved plane, cached next to the SVGs (~40 KB) so a rebuild needs no solve
FIELD_CACHE = OUT / "hero-field.npz"

# --- the picture -------------------------------------------------------------

FRAMES = 12
PERIOD_S = 1.6  # one loop of the animation, in wall-clock seconds
#: below this fraction of the peak the phase is noise, and drawing fronts there
#: quadruples the file for a haze nobody reads
FRONT_FLOOR = 0.075

W = 1200
PAD = 10
Z0, Z1 = 4.0, 35.2  # the window, in millimetres along the beam
XHALF = 8.5
#: the canvas follows the data, not the other way round -- a canvas with its own
#: aspect ratio would letterbox the field and leave dead bands down both sides
H = round(2 * PAD + (W - 2 * PAD) * (2 * XHALF) / (Z1 - Z0))


def solve() -> dict:
    """One focused bowl in water; keep the mid-plane, throw the rest away."""
    logging.disable(logging.WARNING)
    warnings.simplefilter("ignore")
    import caustica

    job = {
        "format": "caustica-job/1",
        "kind": "explicit",
        "name": "hero",
        "medium": {"kind": "homogeneous"},
        "grid": {
            "ndim": 3,
            "dx_mm": 0.25,
            "size_mm": [22, 22, 38],
            "pml": {"thickness_mm": 2.5},
        },
        "source": {
            "kind": "array",
            "array": {"kind": "bowl", "d_outer_mm": 20.0, "roc_mm": 20.0},
            "apex_mm": [11.0, 11.0, 5.0],
        },
        "drive": {"f0_mhz": 1.0, "amplitude_kpa": 100.0},
        "run": {"spec": {"min_settle_periods": 2, "max_settle_periods": 8}, "harmonics": [1]},
        "solver": "linear",
    }
    print("  solving (about 20 s on a CPU) ...")
    res = caustica.simulate(job, out=None, progress=None)
    p = np.asarray(res.result.phasor)
    plane = p[:, p.shape[1] // 2, :]

    w = int(round(2.5 / 0.25))
    plane = plane[w:-w, w:-w]  # the PML is not part of the picture
    peak = np.abs(plane).max()
    FIELD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FIELD_CACHE,
        amp=(np.abs(plane) / peak).astype(np.float16),
        phase=np.angle(plane).astype(np.float16),
        dx_mm=0.25,
        z_start_mm=2.5,
        apex_z_mm=5.0,
        roc_mm=20.0,
        aperture_mm=20.0,
        peak_mpa=peak / 1e6,
    )
    print(f"  cached {FIELD_CACHE.name} ({FIELD_CACHE.stat().st_size / 1024:.0f} KB)")
    return field()


def field() -> dict:
    d = np.load(FIELD_CACHE)
    amp = d["amp"].astype(np.float64)
    nx, nz = amp.shape
    dx = float(d["dx_mm"])
    z = np.arange(nz) * dx + float(d["z_start_mm"])
    x = (np.arange(nx) - nx / 2 + 0.5) * dx
    zz, xx = np.meshgrid(z, x)
    return {
        "amp": amp,
        "phase": d["phase"].astype(np.float64),
        "Z": zz,
        "X": xx,
        "apex_z_mm": float(d["apex_z_mm"]),
        "roc_mm": float(d["roc_mm"]),
        "aperture_mm": float(d["aperture_mm"]),
        "peak_mpa": float(d["peak_mpa"]),
    }


def _axes(fig):
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(Z0, Z1)
    ax.set_ylim(-XHALF, XHALF)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def _figure():
    inner_w = (W - 2 * PAD) / 100
    return plt.figure(figsize=(inner_w, inner_w * (2 * XHALF) / (Z1 - Z0)), dpi=100)


def layer(draw, th: Theme, salt: str) -> str:
    """One transparent overlay, drawn on the same axes as every other one."""
    matplotlib.rcParams["svg.hashsalt"] = salt
    fig = _figure()
    draw(_axes(fig), th)
    return inline_figure(fig, PAD, PAD, W - 2 * PAD, H - 2 * PAD, salt)


def _smooth(a: np.ndarray) -> np.ndarray:
    """3x3 mean. Only ever applied to the envelope, never to the phase."""
    p = np.pad(a, 1, mode="edge")
    return sum(p[i : i + a.shape[0], j : j + a.shape[1]] for i in range(3) for j in range(3)) / 9.0


def draw_envelope(ax, th: Theme, f: dict) -> None:
    ax.contourf(
        f["Z"],
        f["X"],
        _smooth(f["amp"]),
        levels=np.linspace(0.055, 1.0, 8),
        cmap=cmap_of(th),
        extend="max",
    )


def draw_bowl(ax, th: Theme, f: dict) -> None:
    r, half, apex = f["roc_mm"], f["aperture_mm"] / 2, f["apex_z_mm"]
    t = np.linspace(-np.arcsin(half / r), np.arcsin(half / r), 240)
    ax.plot(
        apex + r - r * np.cos(t),
        r * np.sin(t),
        color=th.accent,
        lw=3.0,
        solid_capstyle="round",
    )


def draw_fronts(ax, th: Theme, f: dict, phase: float) -> None:
    front = np.cos(f["phase"] - phase)
    ax.contour(
        f["Z"],
        f["X"],
        np.where(f["amp"] > FRONT_FLOOR, front, np.nan),
        levels=[0.0],
        colors=[th.bg],
        linewidths=1.0,
    )


def build(th: Theme, f: dict) -> str:
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="A focused ultrasound beam converging in water: one plane of a real '
        f'caustica solve, with the wavefronts animated over one acoustic period.">',
        f'<rect width="{W}" height="{H}" rx="10" fill="{th.fig_bg}"/>',
        layer(lambda ax, t: draw_envelope(ax, t, f), th, f"hero-env-{th.name}"),
    ]

    # one layer per phase of the period; SMIL shows exactly one at a time
    for i in range(FRAMES):
        t0, t1 = i / FRAMES, (i + 1) / FRAMES
        if i == 0:
            values, key_times = "1;0;0", f"0;{t1:.4f};1"
        else:
            values, key_times = "0;1;0;0", f"0;{t0:.4f};{t1:.4f};1"
        o.append(f'<g opacity="{1 if i == 0 else 0}">')
        o.append(
            f'<animate attributeName="opacity" values="{values}" keyTimes="{key_times}" '
            f'calcMode="discrete" dur="{PERIOD_S}s" repeatCount="indefinite"/>'
        )
        o.append(
            layer(
                lambda ax, t, ph=2 * np.pi * i / FRAMES: draw_fronts(ax, t, f, ph),
                th,
                f"hero-f{i}-{th.name}",
            )
        )
        o.append("</g>")

    o.append(layer(lambda ax, t: draw_bowl(ax, t, f), th, f"hero-bowl-{th.name}"))
    o.append(
        f'<text x="{W - PAD - 14}" y="{H - PAD - 14}" text-anchor="end" '
        f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" '
        f'font-size="13" fill="{mix(th.fig_bg, th.muted, 0.85)}">'
        f"1 MHz bowl in water &#183; |P| and Re{{P&#183;e^(&#8722;i&#969;t)}} "
        f"&#183; peak {f['peak_mpa']:.2f} MPa</text>"
    )
    o.append("</svg>")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + "\n".join(o) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="render docs/assets/hero-field*.svg")
    ap.add_argument("--resolve", action="store_true", help="re-run the solve and refresh the cache")
    args = ap.parse_args()

    f = solve() if (args.resolve or not FIELD_CACHE.exists()) else field()
    OUT.mkdir(parents=True, exist_ok=True)
    for th in (LIGHT, DARK):
        name = "hero-field" + ("-dark" if th.name == "dark" else "") + ".svg"
        path = OUT / name
        svg = build(th, f)
        path.write_text(svg, encoding="utf-8")
        print(f"  wrote {path.relative_to(REPO)}  ({len(svg) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
