"""Band-limited deposition of a transducer onto a grid, one routine per shape.

The rule is the same for every shape and is the whole point of the model: an
element carries its own area, that area is divided over equal-area quadrature
points on the true surface, and each point is deposited through a band-limited
interpolant at its own sub-voxel position
(:func:`caustica.geometry.offgrid.band_limited_weights`). The grid weights then
sum to the physical area in grid squares whatever the element's shape or
orientation, and no element is ever rounded onto the lattice.

Two deposits, because a continuous wave and a pulse need different things from
the same geometry:

:func:`deposit_cw`
    Contributions are summed as complex phasors, ``amplitude *
    exp(-i (phase + omega * delay))`` per element, so two elements that reach
    the same voxel superpose instead of one of them being discarded. The
    result is a :class:`~caustica.sources.CWSource` with per-voxel weight and
    phase.

:func:`deposit_transient`
    The same weights, but kept per (element, voxel) pair with the element's
    delay attached, because off the carrier a delay is not a phase and two
    delays at one voxel cannot be collapsed into one number.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from caustica.core.backend import CausticaWarning
from caustica.core.grid import Grid
from caustica.sources import CWSource
from caustica.transducers.elements import Element, element_points

if TYPE_CHECKING:  # pragma: no cover - typing only
    from caustica.transducers.model import Transducer

log = logging.getLogger("caustica")

__all__ = [
    "ArraySource",
    "TransientDeposit",
    "deposit_cw",
    "deposit_transient",
    "element_deposit",
    "quadrature_count",
]

#: Fraction of an element's total drive that may fall outside the grid before
#: the caller is warned. A band-limited source has a halo; a halo hanging off
#: the domain edge is a transducer partly outside the domain.
_DROP_WARN = 1e-6


@dataclass(frozen=True)
class ArraySource:
    """A voxelized transducer: the CWSource plus per-voxel element ownership.

    Only radiating elements appear: one apodized to zero contributes no voxels
    and is not counted by :attr:`n_elements_represented`.
    """

    source: CWSource
    element_of_voxel: np.ndarray  # (n_voxels,) int32

    @property
    def n_elements_represented(self) -> int:
        return int(np.unique(self.element_of_voxel).size)


@dataclass(frozen=True)
class TransientDeposit:
    """Per (element, voxel) weights and delays for a time-domain drive.

    Attributes
    ----------
    indices:
        ``(n, 3)`` voxel coordinates. The same voxel appears once per element
        that reaches it.
    weights:
        ``(n,)`` real deposit weight times the element's ``amplitude``.
        Negative entries are the interpolant's side-lobes.
    delays_s:
        ``(n,)`` the owning element's ``delay_s`` [s].
    phases:
        ``(n,)`` the owning element's ``phase`` [rad], for a drive that
        carries one.
    element:
        ``(n,)`` index of the owning element.
    n_elements:
        How many elements the transducer had.
    """

    indices: np.ndarray
    weights: np.ndarray
    delays_s: np.ndarray
    phases: np.ndarray
    element: np.ndarray
    n_elements: int

    @property
    def n_points(self) -> int:
        return int(self.indices.shape[0])


def quadrature_count(element: Element, dx: float, upsampling: int) -> int:
    """Quadrature points for one element: ``upsampling`` per grid square of area.

    The floor of 16 is the same one the disc and spherical-cap samplers have
    always carried: below it the point set stops being a lattice at all.
    """
    if upsampling < 1:
        raise ValueError(f"upsampling must be >= 1, got {upsampling}")
    return max(int(np.ceil(element.area / dx**2 * upsampling)), 16)


def _check_resolved(element: Element, dx: float, index: int) -> None:
    extent = element.min_extent
    if extent < dx:
        raise ValueError(
            f"element {index} ({element.shape}) is {extent * 1e3:g} mm across at its "
            f"narrowest = {extent / dx:.2f} voxels at dx={dx * 1e3:g} mm. Below one voxel "
            f"an element has no shape the grid can carry and neighbouring elements merge "
            f"into one patch. Refine dx (need dx <= {extent * 1e3:g} mm) or use larger "
            f"elements."
        )


def _check_elements(elements: Sequence[Element], dx: float) -> None:
    """Refuse a transducer no deposit routine can honour.

    Every element is graded against the voxel size, silenced ones included, so
    that apodizing an element to zero cannot hide a geometry the grid could not
    have carried. A transducer whose elements are all silenced is refused here
    rather than reported as an empty deposit, which reads as a placement error.
    """
    for i, el in enumerate(elements):
        _check_resolved(el, dx, i)
    if all(el.amplitude == 0.0 for el in elements):
        raise ValueError(
            "every element is apodized to zero, so this transducer radiates nothing; "
            "give at least one element a non-zero amplitude"
        )


def element_deposit(
    element: Element,
    grid: Grid,
    origin_vox: tuple[int, int, int] | np.ndarray,
    *,
    tolerance: float = 0.2,
    upsampling: int = 10,
    index: int = 0,
):
    """One element's band-limited weights on the grid, in grid squares of area.

    Returns the :class:`~caustica.geometry.offgrid.Deposit`; its weights sum to
    ``element.area / dx**2`` up to what falls off the domain edge.
    """
    from caustica.geometry.offgrid import band_limited_weights  # noqa: PLC0415

    if grid.ndim != 3:
        raise ValueError("a transducer deposit requires a 3-D grid")
    _check_resolved(element, grid.dx, index)
    n_q = quadrature_count(element, grid.dx, upsampling)
    pts = element_points(element, n_q)
    area_grid = element.area / grid.dx**2
    origin = np.asarray(origin_vox, dtype=np.float64)
    return band_limited_weights(
        grid.shape, pts / grid.dx + origin, area_grid / len(pts), tolerance=tolerance
    )


def deposit_cw(
    transducer: Transducer,
    grid: Grid,
    origin_vox: tuple[int, int, int],
    f0: float,
    amplitude: float,
    *,
    tolerance: float = 0.2,
    upsampling: int = 10,
    label: str | None = None,
    ramp_periods: float = 3.0,
) -> ArraySource:
    """Deposit a transducer as one continuous-wave source.

    ``origin_vox`` is the voxel the transducer frame's origin sits at. Element
    surfaces are placed in metres relative to it, so nothing is rounded onto
    the lattice: rounding an element centre costs it up to half a voxel of path
    length to the focus, which is a different phase error per element and
    therefore defocuses rather than averaging out (measured on the 128-element
    production spiral at dx = 0.5 mm and 1 MHz: 0.61 rad rms, 17.6 % of the
    coherent focal sum).

    Each element contributes ``amplitude_i * exp(-i (phase_i + omega *
    delay_i))`` times its deposit, and voxels two elements both reach carry the
    phasor sum: ``sum_i w_i sin(omega t - phi_i)`` is exactly
    ``|S| sin(omega t - Phi)`` for ``S = sum_i w_i exp(-i phi_i)``.

    An element apodized to exactly zero is not deposited at all, so it leaves no
    zero-weight voxels behind and does not appear in ``element_of_voxel``: the
    source describes what radiates, and ``n_points`` and
    ``n_elements_represented`` are falsifiable numbers a job report can carry.
    """
    elements = transducer.world_elements()
    if not elements:
        raise ValueError("a transducer with no elements deposits nothing")
    _check_elements(elements, grid.dx)
    omega = 2.0 * np.pi * f0
    acc: dict[tuple[int, int, int], complex] = {}
    best: dict[tuple[int, int, int], tuple[float, int]] = {}
    dropped = 0.0
    requested = 0.0

    for i, el in enumerate(elements):
        if el.amplitude == 0.0:
            continue
        dep = element_deposit(
            el, grid, origin_vox, tolerance=tolerance, upsampling=upsampling, index=i
        )
        dropped += dep.dropped
        requested += dep.requested
        drive = el.amplitude * np.exp(-1j * (el.phase + omega * el.delay_s))
        for key, w in zip(map(tuple, dep.indices.tolist()), dep.weights, strict=True):
            acc[key] = acc.get(key, 0j) + w * drive
            mag = abs(float(w)) * el.amplitude
            if key not in best or mag > best[key][0]:
                best[key] = (mag, i)

    if not acc:
        raise ValueError(
            f"the transducer deposited nothing inside grid {grid.shape}; check origin_vox"
        )
    # A voxel whose phasors cancelled exactly carries no drive, so it is not part
    # of the source: keeping it would inflate n_points and, through best, hand a
    # silent element voxels that radiate nothing.
    keys = [k for k in sorted(acc) if acc[k] != 0j]
    if not keys:
        raise ValueError(
            "every voxel this transducer reached cancelled to exactly zero drive; "
            "check the per-element amplitudes and phases"
        )
    idx = np.asarray(keys, dtype=np.int64)
    s = np.asarray([acc[k] for k in keys], dtype=np.complex128)
    _warn_dropped(dropped, requested, grid)
    elem_of_voxel = np.asarray([best[k][1] for k in keys], dtype=np.int32)
    source = CWSource(
        indices=idx,
        phases=np.angle(np.conj(s)).astype(np.float32),
        weights=np.abs(s).astype(np.float32),
        amplitude=amplitude,
        f0=f0,
        ramp_periods=ramp_periods,
        label=label if label is not None else transducer.label(),
        discretization="offgrid",
    )
    source.check_inside(grid)
    log.debug(
        "transducer deposit: %d elements, %d points, |drive| total %.1f grid squares",
        len(elements),
        len(idx),
        float(np.abs(s).sum()),
    )
    return ArraySource(source=source, element_of_voxel=elem_of_voxel)


def deposit_transient(
    transducer: Transducer,
    grid: Grid,
    origin_vox: tuple[int, int, int],
    *,
    tolerance: float = 0.2,
    upsampling: int = 10,
) -> TransientDeposit:
    """Deposit a transducer as ``(weights, delays)``, one row per element-voxel.

    The continuous-wave path folds an element's delay into its phase; a pulse
    cannot, because a delay shifts every frequency of the pulse by the same
    time and a phase does not. So the rows stay separate and the time-domain
    driver applies ``w * s(t - delay)`` per row.

    As in :func:`deposit_cw`, an element apodized to exactly zero contributes no
    rows; ``n_elements`` still reports how many elements the transducer had.
    """
    elements = transducer.world_elements()
    if not elements:
        raise ValueError("a transducer with no elements deposits nothing")
    _check_elements(elements, grid.dx)
    idx_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    d_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
    e_parts: list[np.ndarray] = []
    dropped = 0.0
    requested = 0.0

    for i, el in enumerate(elements):
        if el.amplitude == 0.0:
            continue
        dep = element_deposit(
            el, grid, origin_vox, tolerance=tolerance, upsampling=upsampling, index=i
        )
        dropped += dep.dropped
        requested += dep.requested
        n = len(dep.indices)
        if not n:
            continue
        idx_parts.append(dep.indices)
        w_parts.append(dep.weights * el.amplitude)
        d_parts.append(np.full(n, el.delay_s, np.float64))
        p_parts.append(np.full(n, el.phase, np.float64))
        e_parts.append(np.full(n, i, np.int32))

    if not idx_parts:
        raise ValueError(
            f"the transducer deposited nothing inside grid {grid.shape}; check origin_vox"
        )
    _warn_dropped(dropped, requested, grid)
    return TransientDeposit(
        indices=np.concatenate(idx_parts).astype(np.int64),
        weights=np.concatenate(w_parts).astype(np.float64),
        delays_s=np.concatenate(d_parts),
        phases=np.concatenate(p_parts),
        element=np.concatenate(e_parts),
        n_elements=len(elements),
    )


def _stacklevel_to_caller() -> int:
    """Frames from :func:`_warn_dropped` out to the first one outside the package.

    ``deposit_cw`` is reached three different ways: directly, through
    :meth:`Transducer.deposit_cw` and through ``TransducerArray.voxelize``. A
    fixed ``stacklevel`` therefore attributes the drop warning to library code
    for two of the three, and the caller is told a line of caustica put its
    transducer off the grid. Walking out to the first non-caustica frame names
    the line the caller actually wrote.
    """
    root = str(Path(__file__).resolve().parent.parent)
    frame = inspect.currentframe()
    frame = None if frame is None else frame.f_back
    level = 1
    while frame is not None:
        if not str(Path(frame.f_code.co_filename).resolve()).startswith(root):
            return level
        level += 1
        frame = frame.f_back
    return level


def _warn_dropped(dropped: float, requested: float, grid: Grid) -> None:
    if requested > 0.0 and dropped / requested > _DROP_WARN:
        warnings.warn(
            f"{100 * dropped / requested:.2f}% of this transducer's drive falls outside "
            f"grid {grid.shape}: the band-limited elements reach a couple of voxels beyond "
            f"their own surfaces. Enlarge the grid or move the transducer inward.",
            CausticaWarning,
            stacklevel=_stacklevel_to_caller(),
        )
