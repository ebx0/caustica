#!/usr/bin/env python3
"""Render the documentation site's hero: a real solve, cut open, animating.

Writes two self-contained animated SVGs into ``docs/assets/``::

    hero-field.svg        light theme
    hero-field-dark.svg   dark  theme

What is on screen is one real solve -- a focused bowl in water, the same k-space
solver the library ships -- drawn as an isometric cutaway. A quarter is taken out
down to the base, and the box is cut off exactly at the **focal plane**, so three
surfaces carry field data and all three meet at the focus:

* the two interior walls: ``|P|`` on the axial half-planes, the converging cone;
* the top face: ``|P|`` across the focal plane, the spot itself.

The rest is the outside of the box and is flat, the way the how-to-use diagram
draws a solid.

Two things made this the shape to draw. An isometric box of these proportions
projects to 1.732 x 1.759 -- square to within 2 %, so it fills a square frame
where a beam laid out flat leaves half of it empty. And the cutaway puts the
focus on the corner where all three data faces meet, which is where the eye goes
anyway.

Brightness is ``|P| ** GAMMA``, floored, multiplied by the instantaneous
``Re{P.e^(-iwt)}`` at N phases of one acoustic period -- one indexed PNG per face
per phase, cycled with SMIL. Because the frames are one full period of the same
phasor, the loop closes exactly: there is no seam to hide and no easing to fake.

    python scripts/make_hero.py             # from the cached planes
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

#: the solved planes, cached next to the SVGs so a rebuild needs no solve
FIELD_CACHE = OUT / "hero-field.npz"

# --- the solve ---------------------------------------------------------------

#: 6 points per wavelength at 1.5 MHz in water; the solver is spectral and gated
#: at 4, so this is headroom rather than necessity.
DX = 0.16
PML_MM = 2.4
#: The drawn box is not the domain: it is the beam. Its floor is the transducer's
#: apex plane and its ceiling is the focal plane, which is the only framing that
#: fills it -- a box drawn around the whole domain is mostly empty water, because
#: a real focus is a millimetre across and an aperture is fifteen.
N_XY = 146  # 23.4 mm; the 18.6 mm interior is the box's footprint
N_Z = 148
APEX_Z = 3.2
#: f/1.0 at 1.5 MHz is ka = 44 and a linear focal gain near 11. It also sets the
#: box's proportions: height = ROC, footprint = the interior, and those project
#: to 1.732 x 1.754 -- square to within 2 %, which is the frame this has to fill.
ROC = 14.0
APERTURE = 14.0
F0_MHZ = 1.5
AMPLITUDE_KPA = 100.0

# --- the picture -------------------------------------------------------------

FRAMES = 12
PERIOD_S = 1.8  # one loop of the animation, in wall-clock seconds

#: brightness is |P| ** GAMMA, floored. A linear map shows a bright dot in an
#: empty box: the cone that feeds the focus is about a tenth of its amplitude.
#: More compression than the 2-D framing wanted, because this box is cropped to
#: the beam -- there is no open water left for the rim's haze to fill.
GAMMA = 0.55
#: everything below this fraction of the compressed peak is background. It buys a
#: clean frame and, because flat regions cost nothing, most of the file size.
FLOOR = 0.08
#: how deep the wavefronts cut into that envelope, 0 = still, 1 = down to zero
WAVE_DEPTH = 0.85
#: shades in the ramp. One hue needs nowhere near 256, and with three faces to
#: pay for on every one of twelve frames, each step down is worth real bytes.
LEVELS = 96
#: pixels in the embedded rasters; the browser's own smoothing covers the last
#: step up to display size
RASTER_TOP = 260
RASTER_WALL = (150, 220)  # (across the face, down the beam)

S = 760  # the SVG's own square, in user units
PAD = 24
MID = 0.5  # where the quarter comes out: through the beam axis, by construction

_COS30 = float(np.cos(np.pi / 6))


def solve() -> dict:
    """One focused bowl in water; keep three planes, throw the volume away."""
    logging.disable(logging.WARNING)
    warnings.simplefilter("ignore")
    import caustica

    centre = N_XY * DX / 2
    job = {
        "format": "caustica-job/1",
        "kind": "explicit",
        "name": "hero",
        "medium": {"kind": "homogeneous"},
        "grid": {
            "ndim": 3,
            "dx_mm": DX,
            "size_mm": [N_XY * DX, N_XY * DX, N_Z * DX],
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
    print("  solving (about a minute on a CPU) ...")
    res = caustica.simulate(job, out=None, progress=None)
    p = np.asarray(res.result.phasor)

    w = int(round(PML_MM / DX))
    p = p[w:-w, w:-w, w:-w]  # the PML is not part of the picture
    nx = p.shape[0]
    mid = nx // 2
    # the box runs from the transducer's apex plane up to the focal plane
    k0 = int(round((APEX_Z - PML_MM) / DX))
    k1 = min(int(round((APEX_Z + ROC - PML_MM) / DX)), p.shape[2] - 1)

    planes = {"xz": p[:, mid, k0 : k1 + 1], "yz": p[mid, :, k0 : k1 + 1], "xy": p[:, :, k1]}
    peak = max(np.abs(a).max() for a in planes.values())

    FIELD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FIELD_CACHE,
        **{
            f"{name}_{part}": getattr(arr / peak, part).astype(np.float16)
            for name, arr in planes.items()
            for part in ("real", "imag")
        },
        dx_mm=DX,
        span_mm=nx * DX,
        height_mm=(k1 - k0) * DX,
        apex_z_mm=0.0,  # the box's floor IS the apex plane
        roc_mm=ROC,
        aperture_mm=APERTURE,
        f0_mhz=F0_MHZ,
        peak_mpa=peak / 1e6,
    )
    print(f"  cached {FIELD_CACHE.name} ({FIELD_CACHE.stat().st_size / 1024:.0f} KB)")
    return field()


def _up(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resample to (width, height)."""
    im = Image.fromarray(np.asarray(a, dtype=np.float32), mode="F")
    return np.asarray(im.resize(size, Image.LANCZOS), dtype=np.float64)


def field() -> dict:
    """The cached planes, laid out the way each face wants to read them.

    Real and imaginary parts are resampled separately, which is exactly right:
    every frame is the linear combination ``re.cos(phi) + im.sin(phi)``, so
    interpolating the phasor interpolates all of them at once. Amplitude and
    phase could not be resampled this way -- phase wraps.
    """
    d = np.load(FIELD_CACHE)
    half = d["xz_real"].shape[0] // 2

    def wall(name: str):
        # s runs from the outer edge in to the axis; t runs down the beam, so the
        # array is transposed and flipped before it becomes an image row order
        return tuple(
            _up(d[f"{name}_{part}"][: half + 1, :].astype(np.float64).T[::-1, :], RASTER_WALL)
            for part in ("real", "imag")
        )

    faces = {
        "top": tuple(
            _up(d[f"xy_{part}"].astype(np.float64).T, (RASTER_TOP, RASTER_TOP))
            for part in ("real", "imag")
        ),
        "wall_x": wall("yz"),
        "wall_y": wall("xz"),
    }
    return {
        "faces": faces,
        "span_mm": float(d["span_mm"]),
        "height_mm": float(d["height_mm"]),
        "apex_z_mm": float(d["apex_z_mm"]),
        "roc_mm": float(d["roc_mm"]),
        "aperture_mm": float(d["aperture_mm"]),
        "f0_mhz": float(d["f0_mhz"]),
        "peak_mpa": float(d["peak_mpa"]),
    }


# --- the projection ----------------------------------------------------------


class Iso:
    """Isometric projection of the drawn box, in SVG user units.

    The same ``(x - y)cos30, z + (x + y)/2`` the how-to-use diagram is drawn in,
    with z scaled by the box's own aspect so the projection stays true.
    """

    def __init__(self, height_mm: float, span_mm: float) -> None:
        self.h = height_mm / span_mm
        self.scale = (S - 2 * PAD) / max(2 * _COS30, self.h + 1.0)
        self.cx = S / 2
        self.cy = S / 2 + (self.h + 1.0) * self.scale / 2

    def uv(self, x, y, z):
        return (x - y) * _COS30, z * self.h + (x + y) * 0.5

    def xy(self, x, y, z) -> tuple[float, float]:
        u, v = self.uv(x, y, z)
        return self.cx + u * self.scale, self.cy - v * self.scale

    def path(self, points) -> str:
        return "M " + " L ".join(f"{a:.1f},{b:.1f}" for a, b in (self.xy(*p) for p in points))

    def matrix(self, origin, e_s, e_t) -> str:
        """The affine that lays a unit-square image onto one face of the box."""
        x0, y0 = self.xy(*origin)

        def direction(e):
            u, v = self.uv(*e)  # linear, so it applies to a direction too
            return self.scale * u, -self.scale * v

        (a, b), (c, dd) = direction(e_s), direction(e_t)
        return f"matrix({a:.4f},{b:.4f},{c:.4f},{dd:.4f},{x0:.3f},{y0:.3f})"


#: (origin, s-direction, t-direction) for each face that carries field data
FACE_FRAMES = {
    # the top face is the focal plane: s along x, t along y
    "top": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    # the wall at x = MID: s from the outer edge in to the axis, t down the beam
    "wall_x": ((MID, 0.0, 1.0), (0.0, MID, 0.0), (0.0, 0.0, -1.0)),
    # the wall at y = MID, likewise
    "wall_y": ((0.0, MID, 1.0), (MID, 0.0, 0.0), (0.0, 0.0, -1.0)),
}

#: the L the top face is clipped to, once the quarter is gone
TOP_L = [
    (MID, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
    (0.0, MID, 1.0),
    (MID, MID, 1.0),
]

#: outside of the box, and the floor the notch was cut down to: flat shades
FLAT_FACES = [
    ([(0, MID, 0), (0, MID, 1), (0, 1, 1), (0, 1, 0)], 0.22),
    ([(MID, 0, 0), (MID, 0, 1), (1, 0, 1), (1, 0, 0)], 0.10),
    ([(0, 0, 0), (MID, 0, 0), (MID, MID, 0), (0, MID, 0)], 0.15),
]

EDGES = [
    *zip(TOP_L, TOP_L[1:] + TOP_L[:1], strict=True),
    ((1, 0, 1), (1, 0, 0)),
    ((0, 1, 1), (0, 1, 0)),
    ((MID, MID, 1), (MID, MID, 0)),
    ((0, MID, 1), (0, MID, 0)),
    ((MID, 0, 1), (MID, 0, 0)),
    ((0, 1, 0), (0, 0, 0)),
    ((0, 0, 0), (1, 0, 0)),
    ((0, MID, 0), (MID, MID, 0)),
    ((MID, MID, 0), (MID, 0, 0)),
]

#: the far corner, which the box hides -- dashed, so the solid reads as a solid
HIDDEN_EDGES = [((1, 1, 1), (1, 1, 0)), ((1, 1, 0), (1, 0, 0)), ((1, 1, 0), (0, 1, 0))]


# --- drawing -----------------------------------------------------------------


def palette(th: Theme) -> np.ndarray:
    """The same single-hue ramp the how-to-use thumbnails are drawn in."""
    return (cmap_of(th)(np.linspace(0, 1, LEVELS))[:, :3] * 255).round().astype(np.uint8)


def frame_png(re: np.ndarray, im: np.ndarray, lut: np.ndarray, phase: float) -> str:
    """One phase of the period on one face, as an indexed PNG in a ``data:`` URI.

    Indexed rather than truecolour because the ramp is one hue: a palette loses
    nothing here and costs a fraction of the bytes.
    """
    amp = np.hypot(re, im)
    with np.errstate(divide="ignore", invalid="ignore"):
        env = np.where(amp > 1e-9, amp**GAMMA, 0.0)
        # cos(phase of P - phase), without ever unwrapping a phase
        ripple = np.where(amp > 1e-9, (re * np.cos(phase) + im * np.sin(phase)) / amp, 0.0)
    env = np.clip((env - FLOOR) / (1.0 - FLOOR), 0.0, 1.0)
    # the modulation is multiplicative, so the cone ripples as visibly as the focus
    v = env * (1.0 - WAVE_DEPTH / 2 + (WAVE_DEPTH / 2) * ripple)

    idx = np.clip(np.rint(v * (LEVELS - 1)), 0, LEVELS - 1).astype(np.uint8)
    img = Image.fromarray(idx, mode="P")
    img.putpalette(lut.reshape(-1).tolist())
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def bowl_path(f: dict, iso: Iso) -> str:
    """The transducer as one closed outline: the cap, with the same quarter gone.

    Drawn as a surface rather than as three separate arcs, because three arcs
    read as three arcs -- a filled dish reads as a dish. The boundary walks out
    along one cut wall, three quarters of the way round the rim, and back down
    the other wall to the apex.
    """
    span, height = f["span_mm"], f["height_mm"]
    r, half, apex = f["roc_mm"], f["aperture_mm"] / 2, f["apex_z_mm"]
    theta = float(np.arcsin(half / r))
    rr = half / span

    t = np.linspace(0.0, theta, 60)
    depth = (apex + r - r * np.cos(t)) / height
    offset = r * np.sin(t) / span

    pts = [(MID - o, MID, d) for o, d in zip(offset, depth, strict=True)]
    for a in np.linspace(np.pi, -np.pi / 2, 220):  # the kept three quarters
        pts.append((MID + rr * float(np.cos(a)), MID + rr * float(np.sin(a)), float(depth[-1])))
    pts += [(MID, MID - o, d) for o, d in zip(offset[::-1], depth[::-1], strict=True)]
    return iso.path(pts) + " Z"


def build(th: Theme, f: dict) -> str:
    lut = palette(th)
    iso = Iso(f["height_mm"], f["span_mm"])
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{S}" height="{S}" viewBox="0 0 {S} {S}" role="img" '
        f'aria-label="A focused ultrasound beam in water, drawn as an isometric cutaway of a '
        f"real caustica solve: the converging cone on two cut planes, the focal spot on the "
        f'top face, with the wavefronts animated over one acoustic period.">',
        "<defs>",
        f'<clipPath id="round-{th.name}"><rect width="{S}" height="{S}" rx="14"/></clipPath>',
        f'<clipPath id="topL-{th.name}"><path d="{iso.path(TOP_L)} Z"/></clipPath>',
        "</defs>",
        f'<rect width="{S}" height="{S}" rx="14" fill="{th.fig_bg}"/>',
        f'<g clip-path="url(#round-{th.name})">',
    ]

    # the outside of the box: flat, the way the how-to diagram draws a solid
    for corners, weight in FLAT_FACES:
        o.append(f'<path d="{iso.path(corners)} Z" fill="{mix(th.fig_bg, th.accent, weight)}"/>')

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
        for name, (origin, e_s, e_t) in FACE_FRAMES.items():
            re, im = f["faces"][name]
            href = frame_png(re, im, lut, 2 * np.pi * i / FRAMES)
            clip = f' clip-path="url(#topL-{th.name})"' if name == "top" else ""
            o.append(
                f"<g{clip}>"
                f'<image width="1" height="1" preserveAspectRatio="none" '
                f'transform="{iso.matrix(origin, e_s, e_t)}" xlink:href="{href}"/>'
                f"</g>"
            )
        o.append("</g>")

    faint = mix(th.fig_bg, th.accent, 0.5)
    for a, b in HIDDEN_EDGES:
        o.append(
            f'<path d="{iso.path([a, b])}" stroke="{faint}" stroke-width="1.2" fill="none" '
            f'stroke-dasharray="5 4"/>'
        )
    for a, b in EDGES:
        o.append(
            f'<path d="{iso.path([a, b])}" stroke="{th.accent}" stroke-width="1.6" '
            f'fill="none" stroke-opacity="0.75"/>'
        )
    # translucent, because an opaque dish hides the near field it is generating
    o.append(
        f'<path d="{bowl_path(f, iso)}" fill="{th.accent}" fill-opacity="0.22" '
        f'stroke="{th.accent}" stroke-width="1.8" stroke-linejoin="round" '
        f'stroke-opacity="0.85"/>'
    )

    o.append("</g>")
    o.append(
        f'<text x="{S - 22}" y="{S - 20}" text-anchor="end" '
        f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" '
        f'font-size="13" fill="{mix(th.fig_bg, th.muted, 0.9)}">'
        f"{f['f0_mhz']:.1f} MHz f/1.0 bowl in water &#183; apex plane to focal plane &#183; "
        f"peak {f['peak_mpa']:.2f} MPa</text>"
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
