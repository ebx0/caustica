"""Transducer elements: four surface shapes, each with its own area and quadrature.

An :class:`Element` is one radiating patch of a transducer. It knows three
things a voxel mask cannot: where its surface actually is (in metres, off the
lattice), how much surface there is (a closed form, not a voxel count), and how
it is driven (amplitude, phase, delay). Everything else in this package is
built on those three.

Frame convention
----------------
``center_m`` is a point on the element's surface and ``normal`` points the way
the element radiates (for a curved element, toward its own centre of
curvature). A flat element lies in the plane through ``center_m``
perpendicular to ``normal``; a ``spherical_segment`` is the patch of the sphere
of radius ``curvature`` whose near pole is ``center_m`` and whose centre is
``center_m + curvature * normal``.

The in-plane angles (``rotation`` for a rectangle, ``phi0``/``phi1`` for a
sector) are measured in the element's own tangent frame
:func:`tangent_frame`, which is fixed by the normal alone, so an element's
geometry is a function of its stored numbers and nothing else.

Shapes and their ``size`` tuples, all in metres and radians:

``disc``
    ``(radius,)``. Rotationally symmetric, so its azimuth reference does not
    matter and it reuses the golden-angle sampler that the array voxelizer has
    always used.
``rect``
    ``(width, height, rotation)``. ``width`` runs along ``u`` and ``height``
    along ``v`` before the in-plane ``rotation`` is applied.
``ring_sector``
    ``(r_in, r_out, phi0, phi1)``. A flat annular sector; ``r_in = 0`` and a
    full turn make it a disc.
``spherical_segment``
    ``(r_in, r_out, phi0, phi1)`` with ``curvature = roc``. The radii are the
    in-plane (projected) radii, so ``r_in = 0``, a full turn and
    ``r_out = aperture radius`` is a whole focused bowl.

Quadrature
----------
:func:`element_points` returns equal-area point sets: every point carries
``area / n``, whatever the shape, so a deposit's weights sum to the element's
own area by construction. That is the property the absolute-amplitude gate
rests on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

__all__ = [
    "SHAPES",
    "Element",
    "disc",
    "element_points",
    "rect",
    "ring_sector",
    "spherical_segment",
    "tangent_frame",
]

#: Golden angle: the azimuth increment that keeps an equal-area radial
#: stratification from lining up into spokes.
_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))

#: Jitter seed for the rectangular lattice. Fixed, so a rectangle deposits the
#: same weights on every run and in every process; stratified rather than
#: regular, so the lattice does not beat against the grid.
_RECT_JITTER_SEED = 20260906

SHAPES = ("disc", "rect", "ring_sector", "spherical_segment")

#: Number of ``size`` entries each shape takes.
_SIZE_LEN = {"disc": 1, "rect": 3, "ring_sector": 4, "spherical_segment": 4}

_TWO_PI = 2.0 * np.pi


def tangent_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed ``(u, v, n)`` with ``n`` along ``normal``.

    ``u`` is the world x axis projected into the element plane, unless the
    normal is within about 26 degrees of x, in which case y seeds it instead.
    The point of pinning it is that ``rotation`` and ``phi`` then mean the same
    thing on every element with the same normal, and that a normal of ``+z``
    gives back exactly the world axes ``(x, y, z)``, which is what makes a
    single ``spherical_segment`` reproduce the analytic bowl point for point.
    """
    n = np.asarray(normal, np.float64)
    norm = float(np.linalg.norm(n))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("element normal must be a finite non-zero vector")
    n = n / norm
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = seed - float(seed @ n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v, n


@dataclass(frozen=True)
class Element:
    """One radiating patch: a surface, a place, and a drive.

    Attributes
    ----------
    shape:
        One of :data:`SHAPES`.
    center_m:
        ``(3,)`` surface point [m]: the centre of a flat element, the near pole
        of a spherical segment.
    normal:
        ``(3,)`` radiation direction, normalized on construction.
    size:
        Shape parameters [m, rad]; see the module docstring.
    amplitude:
        Dimensionless drive weight, 1.0 for a fully driven element. This is
        the apodization knob.
    phase:
        Drive phase [rad]. The drive is ``sin(omega t - phase)``, the same
        sign convention the voxelized sources have always used.
    delay_s:
        Drive delay [s]. For a continuous wave it is folded into the phase as
        ``omega * delay_s``; for a transient it is carried separately, because
        a delay and a phase are not the same thing off the carrier.
    curvature:
        Radius of curvature [m] for ``spherical_segment``, ``None`` for the
        flat shapes.
    """

    shape: Literal["disc", "rect", "ring_sector", "spherical_segment"]
    center_m: np.ndarray
    normal: np.ndarray
    size: tuple[float, ...]
    amplitude: float = 1.0
    phase: float = 0.0
    delay_s: float = 0.0
    curvature: float | None = None
    label: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"unknown element shape {self.shape!r}; use one of {SHAPES}")
        center = np.asarray(self.center_m, np.float64).reshape(-1)
        if center.shape != (3,):
            raise ValueError(f"center_m must be (3,), got {np.shape(self.center_m)}")
        u, v, n = tangent_frame(self.normal)
        size = tuple(float(s) for s in self.size)
        want = _SIZE_LEN[self.shape]
        if len(size) != want:
            raise ValueError(
                f"a {self.shape!r} element takes {want} size value(s), got {len(size)}: "
                f"{self._size_help()}"
            )
        if not np.isfinite(center).all() or not np.isfinite(size).all():
            raise ValueError("element centre and size must be finite")
        if self.amplitude < 0.0:
            raise ValueError(f"amplitude must be >= 0, got {self.amplitude}")
        if not np.isfinite(self.phase) or not np.isfinite(self.delay_s):
            raise ValueError("element phase and delay must be finite")
        self._check_size(size)
        object.__setattr__(self, "center_m", center)
        object.__setattr__(self, "normal", n)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "amplitude", float(self.amplitude))
        object.__setattr__(self, "phase", float(self.phase))
        object.__setattr__(self, "delay_s", float(self.delay_s))
        object.__setattr__(self, "_frame", (u, v, n))

    def __eq__(self, other: object) -> bool:
        # Written out because the generated one compares numpy arrays with
        # ``==`` and then calls bool() on the result, which raises.
        if not isinstance(other, Element):
            return NotImplemented
        return (
            self.shape == other.shape
            and np.array_equal(self.center_m, other.center_m)
            and np.array_equal(self.normal, other.normal)
            and self.size == other.size
            and self.amplitude == other.amplitude
            and self.phase == other.phase
            and self.delay_s == other.delay_s
            and self.curvature == other.curvature
        )

    def __hash__(self) -> int:
        return hash((self.shape, self.size, self.amplitude, self.phase, self.delay_s))

    def _size_help(self) -> str:
        return {
            "disc": "(radius,)",
            "rect": "(width, height, rotation)",
            "ring_sector": "(r_in, r_out, phi0, phi1)",
            "spherical_segment": "(r_in, r_out, phi0, phi1) with curvature=roc",
        }[self.shape]

    def _check_size(self, size: tuple[float, ...]) -> None:
        if self.shape == "disc":
            if size[0] <= 0.0:
                raise ValueError(f"disc radius must be > 0, got {size[0]}")
        elif self.shape == "rect":
            if size[0] <= 0.0 or size[1] <= 0.0:
                raise ValueError(f"rect width and height must be > 0, got {size[:2]}")
        else:
            r_in, r_out, phi0, phi1 = size
            if r_in < 0.0:
                raise ValueError(f"r_in must be >= 0, got {r_in}")
            if r_out <= r_in:
                raise ValueError(f"need r_out > r_in, got r_in={r_in}, r_out={r_out}")
            if not 0.0 < phi1 - phi0 <= _TWO_PI + 1e-12:
                raise ValueError(f"need 0 < phi1 - phi0 <= 2*pi, got phi0={phi0}, phi1={phi1}")
        if self.shape == "spherical_segment":
            if self.curvature is None or not np.isfinite(self.curvature) or self.curvature <= 0.0:
                raise ValueError(
                    "a spherical_segment needs curvature=roc > 0 (its radius of curvature [m])"
                )
            if size[1] >= self.curvature:
                raise ValueError(
                    f"r_out {size[1]:g} m reaches or passes the curvature centre "
                    f"(roc {self.curvature:g} m); a segment is at most a hemisphere"
                )
        elif self.curvature is not None:
            raise ValueError(
                f"curvature belongs to a spherical_segment; a {self.shape!r} element is flat"
            )

    # ---------- geometry ----------

    @property
    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The element's own ``(u, v, n)`` basis; see :func:`tangent_frame`."""
        return self._frame  # type: ignore[attr-defined,no-any-return]

    @property
    def area(self) -> float:
        """Surface area [m^2], in closed form. Never a voxel count."""
        if self.shape == "disc":
            return float(np.pi * self.size[0] ** 2)
        if self.shape == "rect":
            return float(self.size[0] * self.size[1])
        r_in, r_out, phi0, phi1 = self.size
        span = phi1 - phi0
        if self.shape == "ring_sector":
            return float(0.5 * span * (r_out**2 - r_in**2))
        roc = float(self.curvature)  # type: ignore[arg-type]
        cos_in, cos_out = _cap_cosines(r_in, r_out, roc)
        return float(span * roc**2 * (cos_in - cos_out))

    @property
    def outer_radius(self) -> float:
        """Largest in-plane distance from ``center_m`` to the surface [m]."""
        if self.shape == "disc":
            return float(self.size[0])
        if self.shape == "rect":
            return float(0.5 * np.hypot(self.size[0], self.size[1]))
        return float(self.size[1])

    @property
    def centroid(self) -> np.ndarray:
        """Area centroid of the surface [m].

        Not the same point as ``center_m``: every segment of one bowl shares
        that bowl's pole as its frame anchor, so the anchor cannot phase an
        annular array. The centroid can, and is what the delay-and-sum helpers
        use.
        """
        u, v, n = self.frame
        if self.shape in ("disc", "rect"):
            return self.center_m
        r_in, r_out, phi0, phi1 = self.size
        span = phi1 - phi0
        phi_m = 0.5 * (phi0 + phi1)
        # A full turn closes the sector, and its centroid is on the axis.
        fold = 1.0 if span >= _TWO_PI - 1e-12 else np.sinc(span / (2.0 * np.pi))
        if self.shape == "ring_sector":
            if span >= _TWO_PI - 1e-12:
                return self.center_m
            r_c = (2.0 / 3.0) * (r_out**3 - r_in**3) / (r_out**2 - r_in**2) * fold
            return self.center_m + r_c * (np.cos(phi_m) * u + np.sin(phi_m) * v)
        roc = float(self.curvature)  # type: ignore[arg-type]
        cos_in, cos_out = _cap_cosines(r_in, r_out, roc)
        t_in, t_out = np.arccos(cos_in), np.arccos(cos_out)
        i1 = cos_in - cos_out
        i2 = 0.5 * ((t_out - t_in) - 0.5 * (np.sin(2 * t_out) - np.sin(2 * t_in)))
        axial = roc * (1.0 - 0.5 * (cos_in + cos_out))
        if span >= _TWO_PI - 1e-12:
            return self.center_m + axial * n
        lateral = roc * (i2 / i1) * fold
        return self.center_m + lateral * (np.cos(phi_m) * u + np.sin(phi_m) * v) + axial * n

    @property
    def min_extent(self) -> float:
        """Narrowest width of the patch [m]: what the grid has to resolve.

        A disc is as wide as its diameter; an annular sector is as narrow as
        the smaller of its radial thickness and its mid-radius arc. An element
        thinner than a voxel has no shape the grid can carry, which is the
        refusal :mod:`caustica.transducers.deposit` raises.
        """
        if self.shape == "disc":
            return float(2.0 * self.size[0])
        if self.shape == "rect":
            return float(min(self.size[0], self.size[1]))
        r_in, r_out, phi0, phi1 = self.size
        span = phi1 - phi0
        full = r_in <= 0.0 and span >= _TWO_PI - 1e-12
        if self.shape == "spherical_segment":
            roc = float(self.curvature)  # type: ignore[arg-type]
            if full:
                return float(2.0 * roc * np.arcsin(min(1.0, r_out / roc)))
            cos_in, cos_out = _cap_cosines(r_in, r_out, roc)
            radial = roc * (np.arccos(cos_out) - np.arccos(cos_in))
            azimuth = span * 0.5 * (r_in + r_out)
            return float(min(radial, azimuth))
        if full:
            return float(2.0 * r_out)
        return float(min(r_out - r_in, span * 0.5 * (r_in + r_out)))

    def moved(self, **changes: object) -> Element:
        """A copy with some fields replaced (drive or placement)."""
        data = {
            "shape": self.shape,
            "center_m": self.center_m,
            "normal": self.normal,
            "size": self.size,
            "amplitude": self.amplitude,
            "phase": self.phase,
            "delay_s": self.delay_s,
            "curvature": self.curvature,
            "label": self.label,
        }
        data.update(changes)
        return Element(**data)  # type: ignore[arg-type]


def _cap_cosines(r_in: float, r_out: float, roc: float) -> tuple[float, float]:
    """``(cos theta_in, cos theta_out)`` for in-plane radii on a sphere."""
    return (
        float(np.sqrt(max(0.0, 1.0 - (r_in / roc) ** 2))),
        float(np.sqrt(max(0.0, 1.0 - (r_out / roc) ** 2))),
    )


# ---------------------------------------------------------------- constructors


def disc(center_m, normal, radius: float, **drive: object) -> Element:
    """Flat circular piston of radius ``radius`` [m]."""
    return Element(shape="disc", center_m=center_m, normal=normal, size=(radius,), **drive)  # type: ignore[arg-type]


def rect(center_m, normal, width: float, height: float, rotation: float = 0.0, **drive) -> Element:
    """Flat rectangle ``width`` along ``u``, ``height`` along ``v``, then rotated."""
    return Element(
        shape="rect",
        center_m=center_m,
        normal=normal,
        size=(width, height, rotation),
        **drive,
    )


def ring_sector(
    center_m, normal, r_in: float, r_out: float, phi0: float = 0.0, phi1: float = _TWO_PI, **drive
) -> Element:
    """Flat annular sector between radii ``r_in`` and ``r_out`` [m]."""
    return Element(
        shape="ring_sector",
        center_m=center_m,
        normal=normal,
        size=(r_in, r_out, phi0, phi1),
        **drive,
    )


def spherical_segment(
    center_m,
    normal,
    roc: float,
    r_out: float,
    r_in: float = 0.0,
    phi0: float = 0.0,
    phi1: float = _TWO_PI,
    **drive,
) -> Element:
    """Curved patch of the sphere of radius ``roc`` whose near pole is ``center_m``.

    ``r_in`` and ``r_out`` are in-plane (projected) radii, so a full bowl of
    aperture radius ``a`` is ``spherical_segment(apex, +z, roc, a)``.
    """
    return Element(
        shape="spherical_segment",
        center_m=center_m,
        normal=normal,
        size=(r_in, r_out, phi0, phi1),
        curvature=roc,
        **drive,
    )


# ------------------------------------------------------------------ quadrature


def _weyl_azimuth(k: np.ndarray, phi0: float, span: float) -> np.ndarray:
    """Low-discrepancy azimuths in ``[phi0, phi0 + span)``.

    For a full turn this is exactly the golden-angle sequence ``k * GOLDEN``
    reduced modulo ``2 pi``, so a full ring or cap reproduces the samplers this
    library already had; for a sector the same sequence is folded into the
    sector instead of into the circle.
    """
    return phi0 + span * np.mod(k * (_GOLDEN_ANGLE / _TWO_PI), 1.0)


def element_points(element: Element, n: int) -> np.ndarray:
    """``(m, 3)`` equal-area quadrature points on the element's surface [m].

    ``n`` is the requested count; ``m`` is what the shape's lattice can give
    (equal to ``n`` for every shape but ``rect``, whose lattice rounds up to a
    rectangle of cells). Each point carries ``element.area / m``, so the caller
    divides by ``len(points)`` and never by ``n``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    u, v, nv = element.frame
    center = element.center_m

    if element.shape == "disc":
        # Verbatim reuse: the array voxelizer's golden-angle disc, so an array
        # of discs deposits exactly what it deposited before this model.
        from caustica.geometry.offgrid import disc_points  # noqa: PLC0415

        return disc_points(center, nv, element.size[0], n)

    if element.shape == "rect":
        w, h, rot = element.size
        nx = max(1, int(round(np.sqrt(n * w / h))))
        ny = max(1, int(np.ceil(n / nx)))
        gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        rng = np.random.default_rng(_RECT_JITTER_SEED)
        jitter = rng.random((nx * ny, 2)) - 0.5
        cx = ((gi.reshape(-1) + 0.5) / nx - 0.5) * w + jitter[:, 0] * (w / nx)
        cy = ((gj.reshape(-1) + 0.5) / ny - 0.5) * h + jitter[:, 1] * (h / ny)
        cr, sr = np.cos(rot), np.sin(rot)
        a = cx * cr - cy * sr
        b = cx * sr + cy * cr
        return center + a[:, None] * u + b[:, None] * v

    r_in, r_out, phi0, phi1 = element.size
    span = phi1 - phi0
    k = np.arange(n, dtype=np.float64)
    phi = _weyl_azimuth(k, phi0, span)

    if element.shape == "ring_sector":
        r = np.sqrt(r_in**2 + (k + 0.5) / n * (r_out**2 - r_in**2))
        return center + r[:, None] * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v)

    roc = float(element.curvature)  # type: ignore[arg-type]
    cos_in, cos_out = _cap_cosines(r_in, r_out, roc)
    # Uniform in cos(theta) is uniform in area on a sphere.
    cos_t = cos_in - (cos_in - cos_out) * (k + 0.5) / n
    sin_t = np.sqrt(np.clip(1.0 - cos_t**2, 0.0, None))
    lateral = roc * sin_t
    return (
        center
        + lateral[:, None] * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v)
        + (roc * (1.0 - cos_t))[:, None] * nv
    )
