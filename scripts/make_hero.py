#!/usr/bin/env python3
"""Render the documentation site's hero: a real 2-D section, animating.

Writes two self-contained animated SVGs into ``docs/assets/``::

    hero-field.svg        light theme
    hero-field-dark.svg   dark  theme

What is on screen is one x-z plane through a real solve -- a focused bowl in
water, the same k-space solver the library ships -- cropped to a square window
centred on the focus, so the converging cone, the focal spot and the diverging
cone all read at once.

Brightness is the pressure amplitude ``|P|``, gamma-compressed so the cone is
visible next to a focus twenty times brighter, and rippled by the instantaneous
pressure ``Re{P.e^(-iwt)}`` at N phases of one acoustic period. One raster layer
per phase, cycled with SMIL.

Because the frames are one full period of the *same* phasor, the loop closes
exactly: there is no seam to hide and no easing to fake.

The animation is SMIL (``<animate>`` on opacity), not CSS, so it needs no
``<style>`` element and survives being loaded as a plain ``<img>``.

    python scripts/make_hero.py             # from the cached field
    python scripts/make_hero.py --resolve   # re-run the solve, refresh the cache
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_howto import DARK, LIGHT, OUT, REPO, Theme, cmap_of, mix  # noqa: E402

#: the solved plane, cached next to the SVGs so a rebuild needs no solve
FIELD_CACHE = OUT / "hero-field.npz"

# --- the solve ---------------------------------------------------------------

#: 6 points per wavelength at 1 MHz in water. The solver is spectral and gated at
#: 4; the cost of this picture is a 40 mm cube, so the extra points buy nothing.
DX = 0.25
PML_MM = 2.5
#: 45 mm cubed, of which the 40 mm interior *is* the frame -- there is no crop.
NX = NZ = 180
#: f/1.0, apex 2 mm inside the frame. Three things have to fit at once: the bowl,
#: which needs the focus off-centre; a focal region small enough to read as a spot,
#: which needs the frame several times longer than 4.lambda.f#^2; and an aperture
#: filling enough of the frame's height that the cone is not lost in white space.
#: Faster than about f/0.8 and the rim's diffraction haze swamps all three.
APEX_Z = 4.5
ROC = 20.0
APERTURE = 20.0
F0_MHZ = 1.0
AMPLITUDE_KPA = 100.0
#: where the focus lands across the frame; with the numbers above it is exact.
FOCUS_AT = 0.55

# --- the picture -------------------------------------------------------------

FRAMES = 12
PERIOD_S = 1.6  # one loop of the animation, in wall-clock seconds

#: brightness is |P| ** GAMMA, floored. A linear map shows a bright dot in an
#: empty box; too much compression (0.55 was tried) lifts the whole diffraction
#: haze -- edge waves radiating backwards off the rim included -- to the same
#: weight as the beam, and the beam stops reading as a beam.
GAMMA = 0.75
#: everything below this fraction of the compressed peak is background. It buys
#: a clean frame and, because flat regions cost nothing, most of the file size.
FLOOR = 0.08
#: how deep the wavefronts cut into that envelope, 0 = still, 1 = down to zero
WAVE_DEPTH = 0.85
#: pixels per side of each embedded frame. The field is ~3x coarser than this;
#: the browser's own smoothing covers the last step up to display size.
RASTER = 440
#: shades in the ramp. One hue needs nowhere near 256, and each halving is worth
#: about a third of the bytes across twelve frames.
LEVELS = 128

S = 720  # the SVG's own square, in user units
PAD = 0


def solve() -> dict:
    """One focused bowl in water; keep a square of the mid-plane, drop the rest.

    The domain is sized so that the crop lands inside it with the PML excluded --
    change ``ROC``, ``APERTURE`` or ``FOCUS_AT`` and ``NZ`` has to follow.
    """
    logging.disable(logging.WARNING)
    warnings.simplefilter("ignore")
    import caustica

    centre = (NX * DX) / 2
    job = {
        "format": "caustica-job/1",
        "kind": "explicit",
        "name": "hero",
        "medium": {"kind": "homogeneous"},
        "grid": {
            "ndim": 3,
            "dx_mm": DX,
            "size_mm": [NX * DX, NX * DX, NZ * DX],
            "pml": {"thickness_mm": PML_MM},
        },
        "source": {
            "kind": "array",
            "array": {"kind": "bowl", "d_outer_mm": APERTURE, "roc_mm": ROC},
            "apex_mm": [centre, centre, APEX_Z],
        },
        "drive": {"f0_mhz": F0_MHZ, "amplitude_kpa": AMPLITUDE_KPA},
        "run": {"spec": {"min_settle_periods": 2, "max_settle_periods": 8}, "harmonics": [1]},
        "solver": "linear",
    }
    print("  solving (a couple of minutes on a CPU) ...")
    res = caustica.simulate(job, out=None, progress=None)
    p = np.asarray(res.result.phasor)
    plane = p[:, p.shape[1] // 2, :]

    w = int(round(PML_MM / DX))
    plane = plane[w:-w, w:-w]  # the PML is not part of the picture
    nx, nz = plane.shape

    # a square around the geometric focus
    focus = int(round((APEX_Z + ROC - PML_MM) / DX))
    z0 = min(max(focus - int(FOCUS_AT * nx), 0), nz - nx)
    plane = plane[:, z0 : z0 + nx]

    peak = np.abs(plane).max()
    plane = plane / peak
    FIELD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FIELD_CACHE,
        re=plane.real.astype(np.float16),
        im=plane.imag.astype(np.float16),
        dx_mm=DX,
        x_half_mm=nx * DX / 2,
        z_start_mm=PML_MM + z0 * DX,
        apex_z_mm=APEX_Z,
        roc_mm=ROC,
        aperture_mm=APERTURE,
        f0_mhz=F0_MHZ,
        peak_mpa=peak / 1e6,
    )
    print(f"  cached {FIELD_CACHE.name} ({FIELD_CACHE.stat().st_size / 1024:.0f} KB)")
    return field()


def field() -> dict:
    """The cached square, upsampled once to raster resolution.

    Real and imaginary parts are resampled separately, which is exactly right:
    every frame is the linear combination ``re.cos(phi) + im.sin(phi)``, so
    interpolating the phasor interpolates all of them at once. Amplitude and
    phase could not be resampled this way -- phase wraps.
    """
    d = np.load(FIELD_CACHE)

    def up(a):
        im = Image.fromarray(np.asarray(a, dtype=np.float32), mode="F")
        return np.asarray(im.resize((RASTER, RASTER), Image.LANCZOS), dtype=np.float64)

    re, im = up(d["re"]), up(d["im"])
    return {
        "re": re,
        "im": im,
        "amp": np.hypot(re, im),
        "x_half_mm": float(d["x_half_mm"]),
        "z_start_mm": float(d["z_start_mm"]),
        "z_end_mm": float(d["z_start_mm"]) + 2 * float(d["x_half_mm"]),
        "apex_z_mm": float(d["apex_z_mm"]),
        "roc_mm": float(d["roc_mm"]),
        "aperture_mm": float(d["aperture_mm"]),
        "f0_mhz": float(d["f0_mhz"]),
        "peak_mpa": float(d["peak_mpa"]),
    }


# --- drawing -----------------------------------------------------------------


def palette(th: Theme) -> np.ndarray:
    """The same single-hue ramp the how-to-use thumbnails are drawn in."""
    return (cmap_of(th)(np.linspace(0, 1, LEVELS))[:, :3] * 255).round().astype(np.uint8)


def bowl_path(f: dict) -> str:
    """The transducer arc, in SVG user units -- vector, so it stays crisp."""
    r, half, apex = f["roc_mm"], f["aperture_mm"] / 2, f["apex_z_mm"]
    span = 2 * f["x_half_mm"]
    t = np.linspace(-np.arcsin(half / r), np.arcsin(half / r), 160)
    u = (apex + r - r * np.cos(t) - f["z_start_mm"]) / span * S
    v = (r * np.sin(t) + f["x_half_mm"]) / span * S
    return "M " + " L ".join(f"{a:.1f},{b:.1f}" for a, b in zip(u, v, strict=False))


def frame_png(f: dict, lut: np.ndarray, phase: float) -> str:
    """One phase of the period, as an indexed PNG in a ``data:`` URI.

    Indexed rather than truecolour because the ramp is one hue: a palette loses
    nothing here and costs a fraction of the bytes.
    """
    amp = f["amp"]
    with np.errstate(divide="ignore", invalid="ignore"):
        env = np.where(amp > 1e-9, amp**GAMMA, 0.0)
        # cos(phase of P - phase), without ever unwrapping a phase
        ripple = np.where(
            amp > 1e-9, (f["re"] * np.cos(phase) + f["im"] * np.sin(phase)) / amp, 0.0
        )
    env = np.clip((env - FLOOR) / (1.0 - FLOOR), 0.0, 1.0)
    # the modulation is multiplicative, so the cone ripples as visibly as the focus
    v = env * (1.0 - WAVE_DEPTH / 2 + (WAVE_DEPTH / 2) * ripple)

    idx = np.clip(np.rint(v * (LEVELS - 1)), 0, LEVELS - 1).astype(np.uint8)
    img = Image.fromarray(idx, mode="P")
    img.putpalette(lut.reshape(-1).tolist())
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build(th: Theme, f: dict) -> str:
    lut = palette(th)
    label = (
        f"{f['f0_mhz']:.0f} MHz f/1.0 bowl in water &#183; "
        f"|P| rippled by Re{{P&#183;e^(&#8722;i&#969;t)}} &#183; "
        f"peak {f['peak_mpa']:.2f} MPa"
    )
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{S}" height="{S}" viewBox="0 0 {S} {S}" role="img" '
        f'aria-label="A focused ultrasound beam converging on its focus in water: one plane '
        f'of a real caustica solve, with the wavefronts animated over one acoustic period.">',
        f'<defs><clipPath id="hero-{th.name}"><rect width="{S}" height="{S}" rx="14"/>'
        f"</clipPath></defs>",
        f'<rect width="{S}" height="{S}" rx="14" fill="{th.fig_bg}"/>',
        f'<g clip-path="url(#hero-{th.name})">',
    ]

    # one layer per phase of the period; SMIL shows exactly one at a time
    for i in range(FRAMES):
        t0, t1 = i / FRAMES, (i + 1) / FRAMES
        if i == 0:
            values, key_times = "1;0;0", f"0;{t1:.4f};1"
        else:
            values, key_times = "0;1;0;0", f"0;{t0:.4f};{t1:.4f};1"
        href = frame_png(f, lut, 2 * np.pi * i / FRAMES)
        o.append(f'<g opacity="{1 if i == 0 else 0}">')
        o.append(
            f'<animate attributeName="opacity" values="{values}" keyTimes="{key_times}" '
            f'calcMode="discrete" dur="{PERIOD_S}s" repeatCount="indefinite"/>'
        )
        o.append(
            f'<image x="{PAD}" y="{PAD}" width="{S - 2 * PAD}" height="{S - 2 * PAD}" '
            f'preserveAspectRatio="none" xlink:href="{href}"/>'
        )
        o.append("</g>")

    o.append(
        f'<path d="{bowl_path(f)}" fill="none" stroke="{th.accent}" stroke-width="4.5" '
        f'stroke-linecap="round" stroke-opacity="0.85"/>'
    )
    o.append("</g>")
    o.append(
        f'<text x="{S - 20}" y="{S - 18}" text-anchor="end" '
        f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" '
        f'font-size="13" fill="{mix(th.fig_bg, th.muted, 0.9)}">{label}</text>'
    )
    o.append("</svg>")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + "\n".join(o) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="render docs/assets/hero-field*.svg")
    ap.add_argument("--resolve", action="store_true", help="re-run the solve, refresh the cache")
    args = ap.parse_args()

    f = solve() if (args.resolve or not FIELD_CACHE.exists()) else field()
    OUT.mkdir(parents=True, exist_ok=True)
    for th in (LIGHT, DARK):
        path = OUT / ("hero-field" + ("-dark" if th.name == "dark" else "") + ".svg")
        svg = build(th, f)
        path.write_text(svg, encoding="utf-8")
        print(f"  wrote {path.relative_to(REPO)}  ({len(svg) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
