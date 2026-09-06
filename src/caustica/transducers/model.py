"""The transducer itself: elements, a frame, and provenance.

A :class:`Transducer` is a tuple of :class:`~caustica.transducers.elements.Element`
in its own frame, plus a :class:`Frame` that places that frame in the world
(grid) coordinates, plus a :class:`TransducerMeta` saying where the geometry
came from and whether it is nominal. Everything a solve needs is derived from
those: the deposit (:mod:`caustica.transducers.deposit`), the Rayleigh integral
over the same surfaces, the focus and the aperture.

Frame convention, unchanged from the array model it replaces: the transducer's
own origin is its apex, the beam axis is ``+z``, and the geometric focus of a
focused device sits at ``(0, 0, focal_length)``.

:class:`TransducerArray` (element centres plus one circular-piston radius) is
kept as the v1 shape of the same idea; it now builds a :class:`Transducer` of
``disc`` elements and deposits through the shared routine, so there is one
deposition path in the library rather than two.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from caustica.analytic.rayleigh import rayleigh_pressure
from caustica.core.grid import Grid
from caustica.sources import _check_discretization
from caustica.transducers.deposit import (
    ArraySource,
    TransientDeposit,
    deposit_cw,
    deposit_transient,
)
from caustica.transducers.elements import Element, element_points

log = logging.getLogger("caustica")

__all__ = [
    "ArraySource",
    "Frame",
    "Transducer",
    "TransducerArray",
    "TransducerMeta",
    "archimedean_spiral",
]

#: Rayleigh quadrature default: points per wavelength along the surface.
_RAYLEIGH_PPW = 8.0


@dataclass(frozen=True)
class Frame:
    """A rigid placement: rotate about the transducer origin, then translate.

    ``rotation`` maps the transducer's own axes into world axes, so a point
    ``p`` of the transducer sits at ``rotation @ p + origin_m`` in the world.
    The identity frame is the apex frame every builder writes in.
    """

    origin_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self) -> None:
        o = np.asarray(self.origin_m, np.float64).reshape(-1)
        r = np.asarray(self.rotation, np.float64)
        if o.shape != (3,):
            raise ValueError(f"origin_m must be (3,), got {np.shape(self.origin_m)}")
        if r.shape != (3, 3):
            raise ValueError(f"rotation must be (3, 3), got {r.shape}")
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-9):
            raise ValueError("rotation must be orthonormal (R^T R = I)")
        if float(np.linalg.det(r)) < 0.0:
            raise ValueError("rotation must be right-handed (det R = +1), not a mirror")
        object.__setattr__(self, "origin_m", o)
        object.__setattr__(self, "rotation", r)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Frame):
            return NotImplemented
        return np.array_equal(self.origin_m, other.origin_m) and np.array_equal(
            self.rotation, other.rotation
        )

    def __hash__(self) -> int:
        return hash((self.origin_m.tobytes(), self.rotation.tobytes()))

    @property
    def is_identity(self) -> bool:
        return bool(
            np.array_equal(self.rotation, np.eye(3)) and np.array_equal(self.origin_m, np.zeros(3))
        )

    def to_world(self, points: np.ndarray) -> np.ndarray:
        """Map ``(..., 3)`` points from the transducer frame into the world."""
        return np.asarray(points, np.float64) @ self.rotation.T + self.origin_m

    def direction(self, vectors: np.ndarray) -> np.ndarray:
        """Map ``(..., 3)`` directions; translation does not apply."""
        return np.asarray(vectors, np.float64) @ self.rotation.T


@dataclass(frozen=True)
class TransducerMeta:
    """Where this geometry came from, and whether it is measured.

    ``nominal`` is the honest default: a datasheet or a paper states what a
    probe is supposed to be, and a real one differs. A geometry taken from a
    measurement of the device in hand sets it False, and a report that mixes
    the two can then say which is which.
    """

    name: str = ""
    source: str = ""
    nominal: bool = True
    notes: str = ""


@dataclass(frozen=True)
class Transducer:
    """A set of elements with a placement and a provenance."""

    elements: tuple[Element, ...]
    frame: Frame = field(default_factory=Frame)
    meta: TransducerMeta = field(default_factory=TransducerMeta)

    def __post_init__(self) -> None:
        els = tuple(self.elements)
        if not els:
            raise ValueError("a transducer needs at least one element")
        for i, el in enumerate(els):
            if not isinstance(el, Element):
                raise TypeError(f"elements[{i}] is {type(el).__name__}, not an Element")
        object.__setattr__(self, "elements", els)

    # ---------- shape ----------

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def area(self) -> float:
        """Total radiating surface [m^2]: the sum of the element areas."""
        return float(sum(el.area for el in self.elements))

    @property
    def centers(self) -> np.ndarray:
        """``(n, 3)`` element centres in the transducer's own frame [m]."""
        return np.array([el.center_m for el in self.elements], np.float64)

    @property
    def centroids(self) -> np.ndarray:
        """``(n, 3)`` element area centroids in the transducer's own frame [m].

        This, not :attr:`centers`, is where an element radiates from as far as
        phasing is concerned: every segment of one bowl shares that bowl's pole
        as its frame anchor.
        """
        return np.array([el.centroid for el in self.elements], np.float64)

    @property
    def normals(self) -> np.ndarray:
        """``(n, 3)`` unit element normals in the transducer's own frame."""
        return np.array([el.normal for el in self.elements], np.float64)

    @property
    def axis(self) -> np.ndarray:
        """Unit beam axis: the normalized mean element normal."""
        m = self.normals.mean(axis=0)
        norm = float(np.linalg.norm(m))
        if norm < 1e-9:
            # Normals that cancel (a hemisphere closed on itself) leave no mean
            # direction; +z is the frame's own axis and the honest fallback.
            return np.array([0.0, 0.0, 1.0])
        return m / norm

    @property
    def focus(self) -> np.ndarray | None:
        """Geometric focus in the transducer frame [m], or ``None`` if there is none.

        Curved elements state it directly (a segment's centre of curvature);
        flat ones only imply it, through where their axes cross, so the answer
        for an array of flat elements is the least-squares intersection of the
        element axes. A probe whose normals are parallel (a flat linear array)
        has no geometric focus at all, and gets ``None`` rather than a huge
        number from an ill-conditioned solve.
        """
        curvatures = [el.curvature for el in self.elements]
        if all(c is not None for c in curvatures):
            centers = self.centers + np.array(curvatures, np.float64)[:, None] * self.normals
            return centers.mean(axis=0)
        n = self.normals
        c = self.centroids
        proj = np.eye(3)[None, :, :] - n[:, :, None] * n[:, None, :]
        a = proj.sum(axis=0)
        b = np.einsum("kij,kj->i", proj, c)
        eig = np.linalg.eigvalsh(a)
        if eig.min() <= 1e-9 * max(eig.max(), 1.0):
            return None
        return np.linalg.solve(a, b)

    @property
    def aperture(self) -> float:
        """Overall width across the beam axis [m].

        Element surfaces are projected onto the plane perpendicular to
        :attr:`axis` and bounded by discs of radius ``element.outer_radius``;
        the aperture is the largest distance across that union. For a bowl or a
        ring this is the outer diameter, for a linear probe the array length.
        """
        ax = self.axis
        c = self.centers
        flat = c - np.outer(c @ ax, ax)
        r = np.array([el.outer_radius for el in self.elements], np.float64)
        d = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=2)
        return float((d + r[:, None] + r[None, :]).max())

    def element_table(self) -> dict[str, Any]:
        """Every per-element number, as arrays: the record a report can print."""
        return {
            "shape": [el.shape for el in self.elements],
            "label": [el.label for el in self.elements],
            "center_m": self.centers,
            "centroid_m": self.centroids,
            "normal": self.normals,
            "area_m2": np.array([el.area for el in self.elements], np.float64),
            "amplitude": np.array([el.amplitude for el in self.elements], np.float64),
            "phase_rad": np.array([el.phase for el in self.elements], np.float64),
            "delay_s": np.array([el.delay_s for el in self.elements], np.float64),
            "curvature_m": np.array(
                [np.nan if el.curvature is None else el.curvature for el in self.elements],
                np.float64,
            ),
            "size": [el.size for el in self.elements],
        }

    def label(self) -> str:
        """Short one-line description, used as the source label."""
        shapes = sorted({el.shape for el in self.elements})
        name = f"{self.meta.name}: " if self.meta.name else ""
        return f"{name}transducer(n={self.n_elements}, {'+'.join(shapes)})"

    # ---------- placement and drive ----------

    def world_elements(self) -> tuple[Element, ...]:
        """The elements with :attr:`frame` applied, ready to deposit."""
        if self.frame.is_identity:
            return self.elements
        return tuple(
            el.moved(
                center_m=self.frame.to_world(el.center_m),
                normal=self.frame.direction(el.normal),
            )
            for el in self.elements
        )

    def placed(self, frame: Frame) -> Transducer:
        """A copy at a different placement."""
        return Transducer(self.elements, frame, self.meta)

    def with_drive(
        self,
        amplitudes: np.ndarray | None = None,
        phases: np.ndarray | None = None,
        delays_s: np.ndarray | None = None,
    ) -> Transducer:
        """A copy with per-element amplitude, phase and/or delay replaced.

        This is the apodization door: ``with_drive(amplitudes=hann)`` is a
        Hann-weighted aperture and nothing else in the model changes.
        """
        n = self.n_elements

        def _v(x: np.ndarray | None, what: str) -> np.ndarray | None:
            if x is None:
                return None
            a = np.asarray(x, np.float64).reshape(-1)
            if a.shape != (n,):
                raise ValueError(f"{what} must be ({n},), got {a.shape}")
            return a

        amp, ph, dl = _v(amplitudes, "amplitudes"), _v(phases, "phases"), _v(delays_s, "delays_s")
        out = []
        for i, el in enumerate(self.elements):
            changes: dict[str, Any] = {}
            if amp is not None:
                changes["amplitude"] = float(amp[i])
            if ph is not None:
                changes["phase"] = float(ph[i])
            if dl is not None:
                changes["delay_s"] = float(dl[i])
            out.append(el.moved(**changes) if changes else el)
        return Transducer(tuple(out), self.frame, self.meta)

    def das_phases(self, target_m: np.ndarray, f0: float, c0: float = 1500.0) -> np.ndarray:
        """Delay-and-sum focusing phases for a target point (world frame, [m]).

        ``phase = -k * distance``, wrapped to ``[0, 2 pi)`` and offset so the
        smallest is zero, which is the convention the array model has used
        since the notebook port.
        """
        k0 = 2.0 * np.pi * f0 / c0
        centroids = np.array([el.centroid for el in self.world_elements()], np.float64)
        d = np.linalg.norm(centroids - np.asarray(target_m, np.float64), axis=1)
        phase = (-k0 * d) % (2.0 * np.pi)
        phase -= phase.min()
        return phase.astype(np.float64)

    def das_delays(self, target_m: np.ndarray, c0: float = 1500.0) -> np.ndarray:
        """Focusing delays [s] for a target point: the transient twin of DAS phases."""
        centroids = np.array([el.centroid for el in self.world_elements()], np.float64)
        d = np.linalg.norm(centroids - np.asarray(target_m, np.float64), axis=1)
        return (d.max() - d) / c0

    # ---------- what it radiates ----------

    def deposit_cw(
        self,
        grid: Grid,
        origin_vox: tuple[int, int, int],
        f0: float,
        amplitude: float,
        *,
        tolerance: float = 0.2,
        upsampling: int = 10,
        label: str | None = None,
    ) -> ArraySource:
        """Band-limited CW deposit; see :func:`caustica.transducers.deposit.deposit_cw`."""
        return deposit_cw(
            self,
            grid,
            origin_vox,
            f0,
            amplitude,
            tolerance=tolerance,
            upsampling=upsampling,
            label=label,
        )

    def deposit_transient(
        self,
        grid: Grid,
        origin_vox: tuple[int, int, int],
        *,
        tolerance: float = 0.2,
        upsampling: int = 10,
    ) -> TransientDeposit:
        """Per-element weights and delays; see :func:`deposit_transient`."""
        return deposit_transient(self, grid, origin_vox, tolerance=tolerance, upsampling=upsampling)

    def surface_quadrature(self, spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(points, areas, element_index)`` over the true element surfaces.

        The same equal-area sets the deposit uses, at a spacing chosen for an
        integral rather than for a grid. This is what makes the Rayleigh
        reference a reference: it integrates over the geometry the solver was
        given, not over an idealized bowl that resembles it.
        """
        if spacing <= 0:
            raise ValueError(f"spacing must be > 0, got {spacing}")
        pts: list[np.ndarray] = []
        areas: list[np.ndarray] = []
        owner: list[np.ndarray] = []
        for i, el in enumerate(self.world_elements()):
            n = max(int(np.ceil(el.area / spacing**2)), 16)
            p = element_points(el, n)
            pts.append(p)
            areas.append(np.full(len(p), el.area / len(p)))
            owner.append(np.full(len(p), i, np.int32))
        return np.concatenate(pts), np.concatenate(areas), np.concatenate(owner)

    def rayleigh_field(
        self,
        field_points: np.ndarray,
        f0: float,
        *,
        c0: float = 1500.0,
        rho: float = 1000.0,
        u0: float = 1.0,
        alpha_np_m: float = 0.0,
        points_per_wavelength: float = _RAYLEIGH_PPW,
    ) -> np.ndarray:
        """Complex CW pressure [Pa] from the Rayleigh integral over the elements.

        Every quadrature point carries its owning element's drive
        ``u0 * amplitude * exp(-i (phase + omega * delay))``, the same complex
        weight the deposit applies, so the integral and the grid source
        describe the same transducer.
        """
        lam = c0 / f0
        pts, areas, owner = self.surface_quadrature(lam / points_per_wavelength)
        omega = 2.0 * np.pi * f0
        # ``rayleigh_pressure`` propagates with exp(+i k R), so a drive
        # ``sin(omega t - phi)`` is the phasor exp(+i phi) there, while the
        # grid deposit books the same physics with the conjugate. Getting this
        # backwards would defocus a steered array and nothing else would show
        # it, so the two are written next to each other on purpose.
        drive = np.array(
            [
                el.amplitude * np.exp(1j * (el.phase + omega * el.delay_s))
                for el in self.world_elements()
            ],
            np.complex128,
        )
        k = complex(2.0 * np.pi * f0 / c0, alpha_np_m)
        return rayleigh_pressure(pts, areas, u0 * drive[owner], field_points, k=k, rho=rho, c=c0)

    # ---------- construction ----------

    @staticmethod
    def from_disc_array(
        positions: np.ndarray,
        normals: np.ndarray,
        elem_radius: float,
        *,
        amplitudes: np.ndarray | None = None,
        phases: np.ndarray | None = None,
        delays_s: np.ndarray | None = None,
        frame: Frame | None = None,
        meta: TransducerMeta | None = None,
    ) -> Transducer:
        """The v1 shape: identical circular pistons at given centres and normals."""
        pos = np.asarray(positions, np.float64)
        nrm = np.asarray(normals, np.float64)
        n = len(pos)
        amp = np.ones(n) if amplitudes is None else np.asarray(amplitudes, np.float64)
        ph = np.zeros(n) if phases is None else np.asarray(phases, np.float64)
        dl = np.zeros(n) if delays_s is None else np.asarray(delays_s, np.float64)
        els = tuple(
            Element(
                shape="disc",
                center_m=pos[i],
                normal=nrm[i],
                size=(float(elem_radius),),
                amplitude=float(amp[i]),
                phase=float(ph[i]),
                delay_s=float(dl[i]),
            )
            for i in range(n)
        )
        return Transducer(els, frame or Frame(), meta or TransducerMeta())


@dataclass(frozen=True)
class TransducerArray:
    """Element centers on a spherical cap plus one circular-piston radius.

    The v1 model, kept because the spiral builder, the element-table job kind
    and the phase-map tooling all speak it. It is a :class:`Transducer` of
    ``disc`` elements with one radius, and :meth:`to_transducer` says so.
    """

    positions: np.ndarray  # (n, 3) [m], apex frame
    normals: np.ndarray  # (n, 3) unit, toward the focus
    elem_radius: float  # [m]
    focal_length: float  # [m] (radius of curvature)

    def __post_init__(self) -> None:
        pos = np.asarray(self.positions, np.float64)
        nrm = np.asarray(self.normals, np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3 or nrm.shape != pos.shape:
            raise ValueError(f"positions/normals must be matching (n, 3), got {pos.shape}")
        if self.elem_radius <= 0 or self.focal_length <= 0:
            raise ValueError("elem_radius and focal_length must be > 0")
        # Copies, not views: a frozen geometry must not shift when the caller
        # mutates the arrays it built us from (review finding, 2026-08-11).
        object.__setattr__(self, "positions", np.array(pos, dtype=np.float64, copy=True))
        object.__setattr__(self, "normals", np.array(nrm, dtype=np.float64, copy=True))

    @property
    def n_elements(self) -> int:
        return int(self.positions.shape[0])

    @property
    def focus(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.focal_length])

    def to_transducer(
        self,
        phases: np.ndarray | None = None,
        amplitudes: np.ndarray | None = None,
        delays_s: np.ndarray | None = None,
    ) -> Transducer:
        """The same geometry in the v2 model: one ``disc`` element per centre."""
        return Transducer.from_disc_array(
            self.positions,
            self.normals,
            self.elem_radius,
            amplitudes=amplitudes,
            phases=phases,
            delays_s=delays_s,
            meta=TransducerMeta(name=f"array(n={self.n_elements})"),
        )

    # ---------- phasing ----------

    def das_phases(self, target_m: np.ndarray, f0: float, c0: float = 1500.0) -> np.ndarray:
        """Delay-and-sum focusing phases for a target point (apex frame, [m]).

        Port of the notebook's ``calc_das_phases``: phase = -k * distance,
        wrapped to [0, 2 pi) and offset so min(phase) = 0.
        """
        k0 = 2.0 * np.pi * f0 / c0
        d = np.linalg.norm(self.positions - np.asarray(target_m, np.float64), axis=1)
        phase = (-k0 * d) % (2.0 * np.pi)
        phase -= phase.min()
        return phase.astype(np.float32)

    # ---------- fast previews (no solver) ----------

    def rayleigh_preview(
        self,
        field_points: np.ndarray,
        f0: float,
        phases: np.ndarray | None = None,
        c0: float = 1500.0,
        u0: float = 1.0,
    ) -> np.ndarray:
        """Complex CW pressure at ``field_points`` from element-center pistons.

        Elements are collapsed to points carrying their full piston area,
        accurate away from the aperture (>= a few element radii), which is
        exactly the array-design use case. This is the future GUI's live
        beam preview and the KZK initial-plane projector. For the integral
        over the true element surfaces, use
        :meth:`Transducer.rayleigh_field` on :meth:`to_transducer`.
        """
        ph = np.zeros(self.n_elements) if phases is None else np.asarray(phases)
        v_n = u0 * np.exp(1j * ph.astype(np.float64))
        areas = np.full(self.n_elements, np.pi * self.elem_radius**2)
        k = 2.0 * np.pi * f0 / c0
        return rayleigh_pressure(self.positions, areas, v_n, field_points, k=k, c=c0)

    # ---------- grid voxelization ----------

    def voxelize(
        self,
        grid: Grid,
        apex_vox: tuple[int, int, int],
        f0: float,
        amplitude: float,
        phases: np.ndarray | None = None,
        amplitudes: np.ndarray | None = None,
        *,
        discretization: str = "offgrid",
        bli_tolerance: float = 0.2,
        upsampling: int = 10,
    ) -> ArraySource:
        """Project elements onto the grid as one band-limited CW source.

        Each element is a flat disc of area ``pi r^2`` sampled at its own
        position in metres and deposited through a band-limited interpolant;
        overlapping contributions are summed as complex phasors rather than one
        of them being discarded. ``amplitudes`` apodizes: one dimensionless
        weight per element, 1.0 for fully driven; an element at 0.0 is left out
        of the source entirely rather than deposited with zero weight.

        ``discretization`` survives only to refuse the removed voxel-shell model
        by name (decision D-022), as the two source builders in
        :mod:`caustica.sources` do; drop the argument.
        """
        if grid.ndim != 3:
            raise ValueError("voxelize requires a 3-D grid")
        _check_discretization(discretization)
        ph = None
        if phases is not None:
            ph = np.asarray(phases, np.float64)
            if ph.shape != (self.n_elements,):
                raise ValueError(f"phases must be ({self.n_elements},), got {ph.shape}")
        amp = None
        if amplitudes is not None:
            amp = np.asarray(amplitudes, np.float64)
            if amp.shape != (self.n_elements,):
                raise ValueError(f"amplitudes must be ({self.n_elements},), got {amp.shape}")
        td = self.to_transducer(phases=ph, amplitudes=amp)
        return deposit_cw(
            td,
            grid,
            apex_vox,
            f0,
            amplitude,
            tolerance=bli_tolerance,
            upsampling=upsampling,
            label=f"array(n={self.n_elements}, r_elem={self.elem_radius * 1e3:.2f}mm)",
        )


def archimedean_spiral(
    n_elements: int = 128,
    d_outer: float = 0.100,
    d_inner: float = 0.044,
    roc: float = 0.100,
    active_fraction: float = 0.60,
    n_arc_samples: int = 10000,
) -> TransducerArray:
    """Spherically curved Archimedean-spiral phased array (notebook port).

    Elements are placed at equal arc-length intervals along a spiral wound
    from ``r_start`` to ``r_end`` (aperture radii inset by one element
    radius); the element radius comes from dividing the active shell area
    equally. Defaults reproduce the production 128-element geometry.
    """
    if not 0 < d_inner < d_outer <= 2 * roc:
        raise ValueError(
            f"need 0 < d_inner < d_outer <= 2*roc, got d_inner={d_inner}, "
            f"d_outer={d_outer}, roc={roc}"
        )
    if not 0 < active_fraction <= 1:
        raise ValueError(f"active_fraction must be in (0, 1], got {active_fraction}")
    r_out, r_in = d_outer / 2.0, d_inner / 2.0
    cap_h = roc - np.sqrt(roc**2 - r_out**2)
    hole_h = roc - np.sqrt(roc**2 - r_in**2)
    active_area = 2.0 * np.pi * roc * (cap_h - hole_h) * active_fraction
    elem_radius = np.sqrt((active_area / n_elements) / np.pi)
    r_start, r_end = r_in + elem_radius, r_out - elem_radius
    if r_start >= r_end:
        raise ValueError(
            f"element radius {elem_radius * 1e3:.2f} mm does not fit between the "
            f"apertures; reduce n_elements or active_fraction."
        )
    turns = np.sqrt(((r_end - r_start) * n_elements) / (np.pi * (r_end + r_start)))
    b = (r_end - r_start) / (2.0 * np.pi * turns)
    theta_hr = np.linspace(0.0, 2.0 * np.pi * turns, n_arc_samples)
    arc_len = cumulative_trapezoid(
        np.sqrt((r_start + b * theta_hr) ** 2 + b**2), theta_hr, initial=0
    )
    theta = interp1d(arc_len, theta_hr, kind="linear")(np.linspace(0.0, arc_len[-1], n_elements))
    r = r_start + b * theta
    positions = np.column_stack(
        (r * np.cos(theta), r * np.sin(theta), roc - np.sqrt(roc**2 - r**2))
    )
    normals = np.array([0.0, 0.0, roc]) - positions
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return TransducerArray(
        positions=positions, normals=normals, elem_radius=float(elem_radius), focal_length=roc
    )
