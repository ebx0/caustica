"""Scene pieces a mirrored k-Wave example needs and the library has not got.

Only one thing lives here today: a 2-D arc source. k-Wave's ``makeArc`` is
the 2-D counterpart of ``makeBowl`` and several examples in the gallery drive
one, while caustica's own curved source constructor is 3-D
(``bowl_cw_source``). Rather than staircase the arc into a binary voxel
shell, which is exactly the discretization decision D-022 removed, it is
sampled in arc length and deposited through the same band-limited interpolant
the bowl uses, so its grid weights sum to the arc's own length in grid
squares.

Nothing here is parity-specific except its home: when a second caller wants
an arc it belongs in :mod:`caustica.sources` beside the bowl.
"""

from __future__ import annotations

import warnings

import numpy as np

from caustica.core.backend import CausticaWarning
from caustica.core.grid import Grid
from caustica.geometry.offgrid import band_limited_weights
from caustica.sources import CWSource


def _check_arc(roc_vox: float, aperture_vox: float) -> None:
    """A radius and a chord that describe an arc, or a refusal that says why."""
    if roc_vox <= 0.0:
        raise ValueError(f"roc must be positive, got {roc_vox}")
    if not 0.0 < aperture_vox < 2.0 * roc_vox:
        raise ValueError(
            f"an arc of radius {roc_vox} voxels cannot have a chord of {aperture_vox} "
            f"voxels; the chord is at most the diameter."
        )


def arc_points(
    apex_vox: tuple[float, float],
    roc_vox: float,
    aperture_vox: float,
    axis: tuple[float, float],
    n: int,
) -> np.ndarray:
    """``(n, 2)`` points, in voxel units, equally spaced along a 2-D arc.

    Parameters
    ----------
    apex_vox:
        The arc's midpoint (k-Wave's ``arc_pos``), in voxel coordinates.
    roc_vox:
        Radius of curvature in voxels (k-Wave's ``radius``).
    aperture_vox:
        Chord length in voxels (k-Wave's ``diameter``).
    axis:
        Unit vector from the apex towards the centre of curvature, which is
        also the direction the arc focuses in.
    n:
        Quadrature points.
    """
    _check_arc(roc_vox, aperture_vox)
    if n < 2:
        raise ValueError(f"an arc needs at least two quadrature points, got {n}")
    # A copy, not the caller's array: normalizing in place would rewrite the
    # axis vector the caller still holds.
    u = np.array(axis, dtype=np.float64)
    u /= np.linalg.norm(u)
    perp = np.array([-u[1], u[0]])
    centre = np.asarray(apex_vox, dtype=np.float64) + roc_vox * u
    half_angle = float(np.arcsin(0.5 * aperture_vox / roc_vox))
    theta = np.linspace(-half_angle, half_angle, n)
    # Measured from the centre of curvature, the apex sits at -roc * u.
    return centre[None, :] + roc_vox * (
        np.cos(theta)[:, None] * (-u)[None, :] + np.sin(theta)[:, None] * perp[None, :]
    )


def arc_cw_source(
    grid: Grid,
    f0: float,
    amplitude: float,
    apex_vox: tuple[float, float],
    roc_vox: float,
    aperture_vox: float,
    axis: tuple[float, float],
    *,
    upsampling: int = 10,
    bli_tolerance: float = 0.2,
    label: str = "",
) -> CWSource:
    """Focused 2-D arc source, deposited band-limited (k-Wave's ``makeArc``).

    Geometry is given in VOXELS because that is how the k-Wave examples give
    it (``arc_pos``, ``radius`` and ``diameter`` are grid points on
    k-wave.org), and converting to metres and back would only lose the
    example's own numbers.

    Parameters
    ----------
    upsampling:
        The same knob :func:`~caustica.geometry.offgrid.spherical_cap_deposit`
        takes, and it sets the same thing: the quadrature spacing, which is
        ``dx / sqrt(upsampling)`` in both. On a surface that spacing is
        ``upsampling`` points per grid square; on a curve it is
        ``sqrt(upsampling)`` points per voxel of arc length, so the default 10
        puts about 3.2 points in every voxel the arc crosses. Spacing is what
        the band-limited kernel reads, which is why the word keeps its meaning
        across a surface and a curve. Measured at the default: the deposited
        weights sum to the arc's own length to 3.1e-8 relative.
    bli_tolerance:
        Where the band-limited interpolant is truncated, as a fraction of its
        peak, passed straight to
        :func:`~caustica.geometry.offgrid.band_limited_weights`. Lower reaches
        further and costs more.
    """
    if grid.ndim != 2:
        raise ValueError(f"arc_cw_source requires a 2-D grid, got {grid.ndim}-D")
    _check_arc(roc_vox, aperture_vox)
    half_angle = float(np.arcsin(0.5 * aperture_vox / roc_vox))
    length_vox = 2.0 * roc_vox * half_angle
    n = max(2, int(np.ceil(length_vox * np.sqrt(upsampling))))
    pts = arc_points(apex_vox, roc_vox, aperture_vox, axis, n)
    dep = band_limited_weights(grid.shape, pts, length_vox / n, tolerance=bli_tolerance)
    if not len(dep.indices):
        raise ValueError(
            f"the arc (apex {tuple(apex_vox)}, roc {roc_vox} vox, aperture "
            f"{aperture_vox} vox) deposited nothing inside grid {grid.shape}"
        )
    if dep.dropped_fraction > 1e-6:
        warnings.warn(
            f"{dep.dropped_fraction * 100:.2f}% of this arc's drive falls outside grid "
            f"{grid.shape}: the band-limited source reaches several voxels beyond the "
            f"arc and the domain does not hold it. Enlarge the grid or move the apex "
            f"inward.",
            CausticaWarning,
            stacklevel=2,
        )
    return CWSource(
        indices=dep.indices,
        phases=np.zeros(len(dep.indices), np.float32),
        amplitude=amplitude,
        f0=f0,
        weights=dep.weights.astype(np.float32),
        label=label or f"arc(roc={roc_vox:g}vox, aperture={aperture_vox:g}vox)",
        discretization="offgrid",
    )
