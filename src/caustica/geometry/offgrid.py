"""Off-grid sources: give a surface its own area back.

A binary voxel mask cannot represent the area of an oblique surface. A flat
source is exact — one voxel per ``dx**2`` of aperture — but a tilted one
crosses more voxels per unit area than a flat one, and by a factor that
depends on its orientation rather than on anything physical. Measured on this
library's own bowl (2026-08-24): 1.18 voxels per ``dx**2`` of cap area at the
shipped sampling, and the focal pressure over-shot O'Neil's closed form by the
same kind of margin, flat from 3.8 to 15 points per wavelength. Refining the
grid does not help, because a staircase factor is not a discretization error.

The repair is the one k-Wave adopted (Wise, Cox, Jaros and Treeby, JASA 146,
2019): stop describing the source as a set of voxels and describe it as a
*measure*. The element carries its physical area; that area is divided over
quadrature points covering the continuous surface; and each point is deposited
on the grid through a band-limited interpolant of a delta function. The grid
weights then sum to the physical area in grid squares by construction, whatever
the surface's orientation, and the deposited field contains nothing above the
grid's own Nyquist frequency.

Two consequences worth knowing before using it:

* The source stops being thin. A band-limited delta has infinite support, and
  even truncated it reaches a couple of voxels in every direction, so an
  off-grid bowl touches several times the grid points its voxel shell did and
  carries negative weights in the sinc side-lobes. Both are correct.
* It needs room. The halo has to fit inside the domain, and weight that falls
  outside is dropped; :func:`band_limited_weights` reports how much, so the
  caller can say so rather than quietly radiating less than it asked for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Deposit:
    """The result of depositing a point set onto a grid.

    Attributes
    ----------
    indices:
        ``(n, ndim)`` integer voxel coordinates carrying non-zero weight.
    weights:
        ``(n,)`` real weights. Negative entries are the interpolant's
        side-lobes and are as load-bearing as the positive ones.
    requested:
        The total weight the caller asked to deposit — for a surface, its
        area in grid squares.
    deposited:
        What landed inside the grid.
    dropped:
        What fell off an edge. This is the number that matters when a source
        sits near the domain boundary: the two above differ from each other
        by the kernel's truncation as well, which is a fraction of a percent
        and not a sign of anything wrong, whereas any ``dropped`` at all
        means part of the transducer is outside the domain.
    """

    indices: np.ndarray
    weights: np.ndarray
    requested: float
    deposited: float
    dropped: float

    @property
    def dropped_fraction(self) -> float:
        return float(self.dropped / self.requested) if self.requested else 0.0


def star_offsets(ndim: int, tolerance: float) -> np.ndarray:
    """Neighbour offsets where a separable sinc is worth evaluating.

    The kernel is a product of one-dimensional sincs, so its magnitude falls
    like ``1 / (pi**ndim * |i*j*k|)`` rather than with the distance: a point
    seven voxels away *along an axis* matters, one seven voxels away along
    every axis at once does not. Bounding the product instead of the radius
    is what makes this affordable: at k-Wave's 0.05 it keeps 855 offsets of
    3375 in 3-D, and the ones it drops are the ones that contribute nothing.

    ``tolerance`` is the smallest kernel magnitude worth keeping, so a larger
    value gives a tighter (cheaper, less accurate) support. The rule and the
    default match k-Wave's ``BLITolerance``.
    """
    if not 0.0 < tolerance < 1.0:
        raise ValueError(f"tolerance must be in (0, 1), got {tolerance}")
    decay = int(np.ceil(1.0 / (np.pi * tolerance)))
    lin = np.arange(-decay, decay + 1)
    mesh = np.meshgrid(*([lin] * ndim), indexing="ij")
    off = np.stack([m.reshape(-1) for m in mesh], axis=1)
    return off[np.abs(np.prod(off, axis=1)) <= decay]


def band_limited_weights(
    shape: tuple[int, ...],
    points: np.ndarray,
    scale: float | np.ndarray,
    *,
    tolerance: float = 0.2,
    normalize: bool = True,
    chunk: int = 4096,
) -> Deposit:
    """Deposit ``points`` onto a grid through band-limited delta functions.

    Parameters
    ----------
    shape:
        Grid shape in voxels.
    points:
        ``(n, ndim)`` positions in **voxel units** — position ``2.5`` sits
        halfway between voxel 2 and voxel 3. Working in voxels rather than
        metres keeps ``dx`` out of the kernel, where it would only cancel.
    scale:
        Weight carried by each point, scalar or per-point. For a surface this
        is its area in grid squares divided by the number of points, so the
        deposited total is the area however the points happen to fall.
    tolerance:
        Smallest kernel magnitude to evaluate; see :func:`star_offsets`.
    normalize:
        Divide each point's truncated kernel by its own sum, so the point
        deposits exactly the weight it was given.

        Truncating a sinc costs accuracy as a *scale* error, and a
        sub-voxel-position-dependent one: measured on an f/1.2 bowl at 15
        points per wavelength, the focal pressure came out at 1.079, 0.970,
        1.019, 0.991 and 1.004 times the closed form for tolerances 0.5
        through 0.05, tracking the deposited total to three decimals in every
        case and ringing rather than converging as the window widened.
        Dividing it out removes the whole effect and leaves a two-voxel
        window as accurate as a seven-voxel one — which matters, because that
        window is the clearance a transducer needs from the absorbing layer.

        The normalized kernel is no longer exactly band-limited. It was not
        exactly band-limited once truncated either, and being exactly right
        about how much surface there is buys more than the last part per
        thousand of spectral purity.
    chunk:
        Points per batch. Only affects peak memory.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != len(shape):
        raise ValueError(f"points must be (n, {len(shape)}), got {pts.shape}")
    ndim = len(shape)
    w = np.broadcast_to(np.asarray(scale, dtype=np.float64), (len(pts),))
    requested = float(w.sum())
    offsets = star_offsets(ndim, tolerance)
    decay = int(np.abs(offsets).max())

    # Accumulate over the source's own bounding box, not the whole domain: a
    # bowl's support is a thin curved slab, and a dense accumulator over a
    # 400^3 job would cost a third of a gigabyte to hold mostly zeros.
    lo = np.maximum(np.floor(pts.min(axis=0)).astype(np.int64) - decay, 0)
    hi = np.minimum(np.ceil(pts.max(axis=0)).astype(np.int64) + decay + 1, np.asarray(shape))
    if np.any(hi <= lo):
        return Deposit(
            np.zeros((0, ndim), np.int64), np.zeros(0, np.float64), requested, 0.0, requested
        )
    box = tuple(int(h - lo_) for h, lo_ in zip(hi, lo, strict=True))
    acc = np.zeros(int(np.prod(box)), dtype=np.float64)
    dropped = 0.0

    for start in range(0, len(pts), chunk):
        block = pts[start : start + chunk]
        weight = w[start : start + chunk]
        nearest = np.rint(block)
        frac = block - nearest  # in [-0.5, 0.5]
        # kernel at node (nearest + o) is prod_d sinc(o_d - frac_d)
        kern = np.ones((len(block), len(offsets)), dtype=np.float64)
        for d in range(ndim):
            kern *= np.sinc(offsets[None, :, d] - frac[:, d, None])
        if normalize:
            total = kern.sum(axis=1)
            kern /= np.where(np.abs(total) < 1e-12, 1.0, total)[:, None]
        kern *= weight[:, None]

        nodes = nearest.astype(np.int64)[:, None, :] + offsets[None, :, :]
        inside = np.ones(nodes.shape[:2], dtype=bool)
        local = np.empty_like(nodes)
        for d in range(ndim):
            local[..., d] = nodes[..., d] - lo[d]
            inside &= (local[..., d] >= 0) & (local[..., d] < box[d])
        flat = np.ravel_multi_index(
            tuple(np.clip(local[..., d], 0, box[d] - 1) for d in range(ndim)), box
        )
        acc += np.bincount(flat[inside], weights=kern[inside], minlength=acc.size)
        dropped += float(kern[~inside].sum())

    hits = np.flatnonzero(acc)
    indices = np.stack(np.unravel_index(hits, box), axis=1) + lo
    return Deposit(indices.astype(np.int64), acc[hits], requested, float(acc[hits].sum()), dropped)


#: Golden angle, the increment that makes a Fibonacci lattice equal-area.
_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def disc_points(center: np.ndarray, normal: np.ndarray, radius: float, n: int) -> np.ndarray:
    """``n`` equal-area points on a flat disc, in the plane normal to ``normal``.

    ``r_k = radius * sqrt((k + 1/2) / n)`` puts equal area between successive
    radii, and the golden-angle azimuth keeps the lattice from lining up into
    spokes. The point of using it for a transducer element is that the points
    land where the element actually is, in metres, rather than on the voxel
    lattice — which is what an element's own position error costs at the
    focus.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    nv = np.asarray(normal, dtype=np.float64)
    nv = nv / np.linalg.norm(nv)
    # Any vector not parallel to the normal seeds the tangent frame; the disc
    # is rotationally symmetric so which one is arbitrary.
    seed = np.array([1.0, 0.0, 0.0]) if abs(nv[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(nv, seed)
    u /= np.linalg.norm(u)
    v = np.cross(nv, u)
    k = np.arange(n, dtype=np.float64)
    r = radius * np.sqrt((k + 0.5) / n)
    theta = k * _GOLDEN_ANGLE
    return np.asarray(center, dtype=np.float64) + r[:, None] * (
        np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    )


def spherical_cap_deposit(
    shape: tuple[int, ...],
    dx: float,
    aperture_radius: float,
    roc: float,
    apex_vox: tuple[int, ...],
    *,
    upsampling: int = 10,
    tolerance: float = 0.2,
    normalize: bool = True,
) -> Deposit:
    """A focused bowl as grid weights that sum to its own area.

    The cap's area is the closed form ``2 pi R^2 (1 - cos theta_max)`` — the
    same quantity the analytic references integrate over, so the source and
    the thing it is graded against now agree about how much surface there is.

    ``upsampling`` sets how many quadrature points cover each grid square of
    that area. The sampling is equal-area by construction (see
    :func:`caustica.analytic.geometry.spherical_cap_points`), so every point
    carries the same share and ``scale`` stays a scalar.
    """
    from caustica.analytic.geometry import spherical_cap_points  # noqa: PLC0415

    if upsampling < 1:
        raise ValueError(f"upsampling must be >= 1, got {upsampling}")
    cos_tmax = np.sqrt(1.0 - (aperture_radius / roc) ** 2)
    area = 2.0 * np.pi * roc**2 * (1.0 - cos_tmax)
    m_grid = area / dx**2
    # spherical_cap_points sizes itself from a target spacing; asking for
    # dx/sqrt(upsampling) is asking for `upsampling` points per grid square.
    points, _normals, _areas = spherical_cap_points(
        aperture_radius, roc, dx / float(np.sqrt(upsampling))
    )
    pts_vox = points / dx + np.asarray(apex_vox, dtype=np.float64)
    return band_limited_weights(
        shape, pts_vox, m_grid / len(points), tolerance=tolerance, normalize=normalize
    )
