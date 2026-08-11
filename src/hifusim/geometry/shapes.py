"""Geometric primitives with CSG operator algebra (2-D and 3-D).

Every shape answers one vectorized question — ``contains(points)`` for an
``(n, ndim)`` array of physical coordinates [m] — and the whole system
(boolean algebra, affine transforms, scene rasterization) composes on top
of that single contract. This keeps primitives tiny and makes user-defined
shapes trivial (subclass, implement ``contains``).

Operators (COMSOL-style booleans):
    a | b   union (OR)         a & b   intersection (AND)
    a - b   difference         ~a      complement (NOT)

Transforms return wrapped shapes; the original is never mutated:
    shape.translated((dx, dy, dz))
    shape.rotated(angle, axis=..., center=...)   # axis only in 3-D
    shape.scaled(factors, center=...)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Shape(ABC):
    """Base shape: an implicit region of 2-D or 3-D space."""

    ndim: int

    @abstractmethod
    def contains(self, points: np.ndarray) -> np.ndarray:
        """Boolean membership for ``(n, ndim)`` physical points [m]."""

    def _check(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != self.ndim:
            raise ValueError(f"expected (n, {self.ndim}) points, got {pts.shape}")
        return pts

    # ---- boolean algebra ----
    def __or__(self, other: Shape) -> Shape:
        return Union(self, other)

    def __and__(self, other: Shape) -> Shape:
        return Intersection(self, other)

    def __sub__(self, other: Shape) -> Shape:
        return Difference(self, other)

    def __invert__(self) -> Shape:
        return Complement(self)

    # ---- affine transforms ----
    def translated(self, offset) -> Shape:
        return AffineShape(self, offset=np.asarray(offset, np.float64))

    def rotated(self, angle: float, axis=None, center=None) -> Shape:
        """Rotate by ``angle`` [rad]; 3-D needs ``axis`` (rotation axis vector)."""
        if self.ndim == 2:
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s], [s, c]])
        else:
            if axis is None:
                raise ValueError("3-D rotation needs an axis vector")
            ax = np.asarray(axis, np.float64)
            ax = ax / np.linalg.norm(ax)
            c, s = np.cos(angle), np.sin(angle)
            x, y, z = ax
            rot = np.array(
                [
                    [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                    [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                    [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
                ]
            )
        return AffineShape(self, matrix=rot, center=center)

    def scaled(self, factors, center=None) -> Shape:
        f = np.broadcast_to(np.asarray(factors, np.float64), (self.ndim,)).copy()
        if (f <= 0).any():
            raise ValueError(f"scale factors must be > 0, got {factors}")
        return AffineShape(self, matrix=np.diag(f), center=center)


def _binary_ndim(a: Shape, b: Shape) -> int:
    if a.ndim != b.ndim:
        raise ValueError(f"cannot combine {a.ndim}-D and {b.ndim}-D shapes")
    return a.ndim


class Union(Shape):
    def __init__(self, a: Shape, b: Shape):
        self.ndim = _binary_ndim(a, b)
        self.a, self.b = a, b

    def contains(self, points):
        return self.a.contains(points) | self.b.contains(points)


class Intersection(Shape):
    def __init__(self, a: Shape, b: Shape):
        self.ndim = _binary_ndim(a, b)
        self.a, self.b = a, b

    def contains(self, points):
        return self.a.contains(points) & self.b.contains(points)


class Difference(Shape):
    def __init__(self, a: Shape, b: Shape):
        self.ndim = _binary_ndim(a, b)
        self.a, self.b = a, b

    def contains(self, points):
        return self.a.contains(points) & ~self.b.contains(points)


class Complement(Shape):
    def __init__(self, a: Shape):
        self.ndim = a.ndim
        self.a = a

    def contains(self, points):
        return ~self.a.contains(points)


class AffineShape(Shape):
    """A shape observed through an inverse affine map (rotate/scale/translate)."""

    def __init__(self, base: Shape, matrix=None, offset=None, center=None):
        self.ndim = base.ndim
        self.base = base
        m = np.eye(self.ndim) if matrix is None else np.asarray(matrix, np.float64)
        if m.shape != (self.ndim, self.ndim):
            raise ValueError(f"matrix must be ({self.ndim},{self.ndim}), got {m.shape}")
        off = np.zeros(self.ndim) if offset is None else np.asarray(offset, np.float64)
        ctr = np.zeros(self.ndim) if center is None else np.asarray(center, np.float64)
        # forward: y = M @ (x - ctr) + ctr + off  ->  inverse solved once here.
        self._minv = np.linalg.inv(m)
        self._pre = ctr + off  # subtracted before Minv
        self._post = ctr  # added after Minv

    def contains(self, points):
        pts = self._check(points)
        local = (pts - self._pre) @ self._minv.T + self._post
        return self.base.contains(local)


class Ball(Shape):
    """Solid sphere (3-D) or disk (2-D); ndim follows len(center)."""

    def __init__(self, center, radius: float):
        self.center = np.asarray(center, dtype=np.float64)
        if self.center.ndim != 1 or self.center.size not in (2, 3):
            raise ValueError(f"center must have 2 or 3 entries, got {center}")
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")
        self.radius = float(radius)
        self.ndim = self.center.size

    def contains(self, points):
        pts = self._check(points)
        d2 = ((pts - self.center) ** 2).sum(axis=1)
        return d2 <= self.radius**2


class Ellipsoid(Shape):
    """Axis-aligned ellipsoid/ellipse (rotate via ``.rotated``)."""

    def __init__(self, center, semiaxes):
        self.center = np.asarray(center, dtype=np.float64)
        self.semiaxes = np.asarray(semiaxes, dtype=np.float64)
        if self.center.shape != self.semiaxes.shape or self.center.size not in (2, 3):
            raise ValueError("center and semiaxes must both have 2 or 3 entries")
        if (self.semiaxes <= 0).any():
            raise ValueError(f"semiaxes must be > 0, got {semiaxes}")
        self.ndim = self.center.size

    def contains(self, points):
        pts = self._check(points)
        return (((pts - self.center) / self.semiaxes) ** 2).sum(axis=1) <= 1.0


class Box(Shape):
    """Axis-aligned solid box/rectangle: ``center`` +/- ``size``/2."""

    def __init__(self, center, size):
        self.center = np.asarray(center, dtype=np.float64)
        self.size = np.asarray(size, dtype=np.float64)
        if self.center.shape != self.size.shape or self.center.size not in (2, 3):
            raise ValueError("center and size must both have 2 or 3 entries")
        if (self.size <= 0).any():
            raise ValueError(f"size must be > 0, got {size}")
        self.ndim = self.center.size

    def contains(self, points):
        pts = self._check(points)
        return (np.abs(pts - self.center) <= self.size / 2.0).all(axis=1)


class Cylinder(Shape):
    """Finite solid cylinder along a coordinate axis (3-D only)."""

    def __init__(self, center, radius: float, height: float, axis: int = 2):
        self.center = np.asarray(center, dtype=np.float64)
        if self.center.size != 3:
            raise ValueError("Cylinder is 3-D; center needs 3 entries")
        if radius <= 0 or height <= 0:
            raise ValueError("radius and height must be > 0")
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
        self.radius, self.height, self.axis = float(radius), float(height), int(axis)
        self.ndim = 3

    def contains(self, points):
        pts = self._check(points)
        rel = pts - self.center
        along = rel[:, self.axis]
        trans = np.delete(rel, self.axis, axis=1)
        return (np.abs(along) <= self.height / 2.0) & ((trans**2).sum(axis=1) <= self.radius**2)


class HalfSpace(Shape):
    """Points x with ``normal . (x - point) <= 0`` (useful for slabs/cuts)."""

    def __init__(self, point, normal):
        self.point = np.asarray(point, dtype=np.float64)
        self.normal = np.asarray(normal, dtype=np.float64)
        if self.point.shape != self.normal.shape or self.point.size not in (2, 3):
            raise ValueError("point and normal must both have 2 or 3 entries")
        n = np.linalg.norm(self.normal)
        if n == 0:
            raise ValueError("normal must be nonzero")
        self.normal = self.normal / n
        self.ndim = self.point.size

    def contains(self, points):
        pts = self._check(points)
        return (pts - self.point) @ self.normal <= 0.0
