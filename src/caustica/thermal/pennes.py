"""Pennes bioheat solver: explicit conservative finite differences on the backend.

The equation
------------
::

    rho C dT/dt = div(k grad T) + Q - w_b rho_b C_b (T - T_a)

with per-voxel ``k`` [W/m/K], ``rho`` [kg/m^3], ``C`` [J/kg/K], ``w_b``
[1/s] from :class:`~caustica.thermal.properties.ThermalMedium`, ``Q``
[W/m^3] from :class:`caustica.sensors.HeatingSource`, and blood constants
``rho_b``, ``C_b``, ``T_a`` that are documented, defaulted and overridable.

Why explicit finite differences and not a k-space/spectral step
---------------------------------------------------------------
The acoustic engine is spectral, so the honest question is why its thermal
sibling is not. Four reasons, in the order they decided it:

1. **The conduction term is ``div(k grad T)``, not ``k lap T``.** A spectral
   step multiplies by ``-|k|^2``, which is the CONSTANT-coefficient
   Laplacian; heterogeneous conductivity then needs a homogeneous reference
   plus correction terms (what MATLAB's ``kWaveDiffusion`` does) or an outer
   iteration. The conservative real-space stencil handles it directly, and
   with harmonic-mean face conductivities its steady state is EXACT at cell
   centres for layered media — which is the fat/muscle/bone interface case
   HIFU planning is made of, not a corner case.
2. **An FFT imposes periodic boundaries.** Heat leaving the top of the block
   would re-enter at the bottom. A thermal domain is not periodic; getting
   insulated or fixed-temperature walls out of a spectral solve costs a
   mirrored/padded domain (2^ndim memory) on top of a volume that already
   holds the acoustic property maps.
3. **The stability limit is a closed form over the assembled coefficients.**
   That is what makes the contract in this module exact rather than a rule
   of thumb: the solver computes the largest stable ``dt`` from the actual
   ``k``, ``rho C``, ``w_b`` and ``dx`` of THIS medium (a Gershgorin bound on
   the update matrix) and either refuses the requested step or sub-steps it.
   It never integrates unstably.
4. **Cost.** One step is a 7-point stencil, O(N) with no transforms, against
   two O(N log N) FFTs per spectral step; on the GPU it is bandwidth-bound
   elementwise work with no plan memory.

The honest cost of the choice: explicit stepping is conditionally stable, so
``dt_stable`` scales as ``dx^2``. On a 0.3 mm grid in soft tissue that is
~0.1 s (a 60 s sonication is ~600 steps, cheap); on a 0.1 mm grid it is
~12 ms (~5000 steps). Media that need much finer grids want an implicit or
ADI scheme, which trades a linear solve per step for unconditional
stability. That is the natural phase-2 cross-check partner and is not in
phase 1.

Discretization
--------------
Cell-centred volumes; the conduction term is assembled as fluxes through
faces, so energy is conserved to round-off::

    (div k grad T)_i = (1/dx^2) sum_faces k_face (T_neighbour - T_i)
    k_face = 2 k_i k_j / (k_i + k_j)            (harmonic mean = series resistance)

Boundaries are ``"insulated"`` (zero flux, the default: a block of tissue
with no modelled surroundings) or ``"dirichlet"`` (walls held at
``boundary_temperature_c``, default ``T_a`` — a body-temperature bath half a
cell outside the domain). Time integration is forward Euler; states are
float32 on the backend chosen through
:func:`~caustica.core.backend.get_backend`, and the host property maps are
uploaded once per run, exactly as the acoustic engine does it.

The state is the RISE above a reference, not the absolute temperature
--------------------------------------------------------------------
Internally the solver steps ``theta = T - T_ref`` (``T_ref`` defaults to the
arterial temperature) and adds the reference back on output. This is not
cosmetic. Conduction differences neighbouring cells, and in float32 the
spacing of representable numbers near 37 is 3.8e-6 K: a temperature field
carried in absolute degrees loses that much off every difference, which is
0.01% of a 0.04 K gradient — and the two face fluxes of a cell nearly
cancel, so the error lands amplified in ``dT/dt``. Measured on the two-layer
steady state of ``tests/test_thermal.py``, whose analytic profile spans
0.039 K on a 37 C base: carried in absolute degrees the solve loses 8.4% of
that profile, carried as a rise above 37 C it keeps it to 0.006%. Since the
small-rise
regime is exactly the ITRUSST safety question (is dT above or below 2 C?),
the offset is removed before the arithmetic rather than documented as a
caveat. Subtracting a nearby float is exact, so nothing is lost going in.

Guards (the same discipline as the acoustic engine)
---------------------------------------------------
* A ``dt`` above the stability bound is REFUSED with the number it must not
  exceed, or sub-stepped if the caller asked for that — never silently
  integrated (an unstable diffusion step does not diverge quietly; it
  produces a plausible-looking checkerboard first).
* A state that stops being finite raises :class:`ThermalDivergedError`
  instead of returning NaN, on the same reasoning as
  ``_guard_finite`` in the k-space engine: a NaN dose map is a wrong answer
  wearing the shape of an answer.
* CEM43 dose is accumulated DURING the integration, sub-step by sub-step. A
  dose computed from the final temperature is a different (and for any
  transient, wrong) number — see :mod:`caustica.thermal.dose`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import ceil
from typing import Any

import numpy as np

from caustica.core.backend import get_backend
from caustica.sensors import HeatingSource
from caustica.thermal.dose import cem43_rate
from caustica.thermal.properties import ThermalMedium

log = logging.getLogger("caustica")

#: Blood density [kg/m^3] (IT'IS tissue database, whole blood).
BLOOD_DENSITY = 1050.0
#: Blood specific heat [J/kg/K] (IT'IS tissue database, whole blood).
BLOOD_SPECIFIC_HEAT = 3617.0
#: Arterial (body core) temperature [C] — the perfusion sink.
ARTERIAL_TEMPERATURE_C = 37.0

#: Bumped when the numerics change; recorded in ``ThermalResult.meta``.
SCHEME = "pennes-fd-explicit/1"

_ALL = slice(None)
_LOWER = slice(None, -1)
_UPPER = slice(1, None)


class ThermalDivergedError(RuntimeError):
    """The temperature (or the dose) stopped being a number."""


class ThermalStabilityError(ValueError):
    """The requested time step is above the explicit scheme's stability bound."""


def _ax(arr: Any, axis: int, sl: slice) -> Any:
    """View of ``arr`` sliced along one axis (basic slicing: writes go through)."""
    idx = [_ALL] * arr.ndim
    idx[axis] = sl
    return arr[tuple(idx)]


@dataclass
class ThermalResult:
    """What a Pennes solve returns.

    Attributes
    ----------
    temperature:
        Final temperature field [C], float32, shape of the medium.
    temperature_max:
        Per-voxel maximum over the whole history INCLUDING the initial state
        and every internal sub-step — the map a thermal-safety check reads,
        because the peak of a sonication is not its endpoint.
    dose_cem43:
        Accumulated CEM43 dose [equivalent minutes] when ``dose=True`` was
        asked for, else ``None``.
    samples, times:
        ``T`` at the recorded steps and their times [s] (empty when
        ``record_every`` was not given). The initial state and the final
        state are always among them when recording is on.
    dt, n_steps, t_end_s:
        The OUTER step the caller asked for, how many were taken, and the
        physical time covered.
    substeps:
        Internal sub-steps per outer step (1 unless the requested ``dt``
        needed splitting for stability).
    meta:
        Scheme, backend, stability bound, blood constants, what Q was.
    """

    temperature: np.ndarray
    temperature_max: np.ndarray
    dose_cem43: np.ndarray | None
    samples: list[np.ndarray]
    times: list[float]
    dt: float
    n_steps: int
    t_end_s: float
    substeps: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def peak_temperature_c(self) -> float:
        return float(self.temperature_max.max())

    @property
    def peak_dose_cem43(self) -> float | None:
        return None if self.dose_cem43 is None else float(self.dose_cem43.max())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        dose = "-" if self.dose_cem43 is None else f"{self.peak_dose_cem43:.4g}"
        return (
            f"ThermalResult(t_end={self.t_end_s:.4g} s, steps={self.n_steps}"
            f"x{self.substeps}, T_peak={self.peak_temperature_c:.4g} C, "
            f"CEM43_peak={dose})"
        )


class PennesSolver:
    """Explicit conservative-FD Pennes bioheat solver (see module docstring).

    Parameters
    ----------
    backend:
        Backend name resolved through :func:`caustica.core.backend.get_backend`
        at :meth:`solve` time (``"auto"``/``"numpy"``/``"cupy"``/a plugin).
    blood_density, blood_specific_heat, arterial_temperature_c:
        The perfusion sink. Defaults are the documented literature values
        (:data:`BLOOD_DENSITY`, :data:`BLOOD_SPECIFIC_HEAT`,
        :data:`ARTERIAL_TEMPERATURE_C`); they are parameters because a
        phantom study, a hypothermia protocol or another tissue database
        will want different ones — and changing them must be a visible act,
        not an edit to a constant.
    boundary:
        ``"insulated"`` (zero flux) or ``"dirichlet"`` (walls held at
        ``boundary_temperature_c``, default ``arterial_temperature_c``).
    reference_temperature_c:
        The offset the float32 state is carried relative to (default:
        ``arterial_temperature_c``). See the module docstring — this is a
        precision decision, not a physical one; the physics is identical for
        any value, but a reference far from the field's own level throws
        away significant digits in every neighbour difference.
    on_unstable:
        ``"refuse"`` (default) or ``"substep"`` — what to do when the
        requested ``dt`` exceeds the stability bound. There is no third
        option: this solver does not integrate unstably.
    finite_check_every:
        Outer steps between finite-state checks (the final state is always
        checked). Each check is a device-to-host reduction, which is why it
        is not every step — the same reasoning as the acoustic engine's
        once-per-period guard.
    """

    name = "pennes"
    scheme = SCHEME

    def __init__(
        self,
        backend: str = "auto",
        *,
        blood_density: float = BLOOD_DENSITY,
        blood_specific_heat: float = BLOOD_SPECIFIC_HEAT,
        arterial_temperature_c: float = ARTERIAL_TEMPERATURE_C,
        boundary: str = "insulated",
        boundary_temperature_c: float | None = None,
        reference_temperature_c: float | None = None,
        on_unstable: str = "refuse",
        finite_check_every: int = 25,
    ):
        if boundary not in ("insulated", "dirichlet"):
            raise ValueError(
                f"boundary must be 'insulated' (zero flux) or 'dirichlet' (walls held "
                f"at boundary_temperature_c), got {boundary!r}"
            )
        if on_unstable not in ("refuse", "substep"):
            raise ValueError(
                f"on_unstable must be 'refuse' or 'substep', got {on_unstable!r}. "
                f"There is deliberately no option that integrates an unstable step."
            )
        if blood_density <= 0 or blood_specific_heat <= 0:
            raise ValueError("blood_density and blood_specific_heat must be > 0")
        if finite_check_every < 1:
            raise ValueError(f"finite_check_every must be >= 1, got {finite_check_every}")
        self.backend = backend
        self.blood_density = float(blood_density)
        self.blood_specific_heat = float(blood_specific_heat)
        self.arterial_temperature_c = float(arterial_temperature_c)
        self.boundary = boundary
        self.boundary_temperature_c = (
            float(arterial_temperature_c)
            if boundary_temperature_c is None
            else float(boundary_temperature_c)
        )
        self.reference_temperature_c = (
            float(arterial_temperature_c)
            if reference_temperature_c is None
            else float(reference_temperature_c)
        )
        if not np.isfinite(self.reference_temperature_c):
            raise ValueError("reference_temperature_c must be finite")
        self.on_unstable = on_unstable
        self.finite_check_every = int(finite_check_every)

    # ---------- stability ----------

    def stable_dt(self, medium: ThermalMedium) -> float:
        """Largest time step [s] this medium can be integrated with.

        The bound is ``1 / max_i (sum_faces k_face/dx^2 + w_b rho_b C_b)_i /
        (rho C)_i`` — Gershgorin over the ASSEMBLED coefficients, so it uses
        the real per-voxel conductivities, the real perfusion, and the real
        boundary condition (insulated faces contribute nothing; Dirichlet
        walls contribute a half-cell face). For a uniform medium with no
        perfusion it reduces to the textbook ``dx^2 / (2 ndim D)``.

        It is the POSITIVITY bound (the explicit update stays a convex
        combination), which is twice as strict as the bare von Neumann
        bound. That is deliberate: a temperature field allowed to oscillate
        while remaining bounded still produces a hot voxel that never
        existed, and CEM43 is exponential in exactly that error.
        """
        b = get_backend(self.backend)
        xp = b.xp
        k = b.asarray(medium.k, xp.float32)
        faces = self._face_conductances(k, medium.dx, xp)
        diag = self._diagonal(k, faces, medium.dx, xp)
        diag += self._perfusion_coefficient(medium, b)
        diag *= self._inv_rho_c(medium, b)
        peak = float(diag.max())
        if peak <= 0.0:  # no conduction, no perfusion: nothing limits the step
            return float("inf")
        return 1.0 / peak

    def _face_conductances(self, k: Any, dx: float, xp: Any) -> list[Any]:
        """Harmonic-mean conductivity of every interior face, divided by dx^2."""
        inv_dx2 = 1.0 / (dx * dx)
        faces = []
        for ax in range(k.ndim):
            lo = _ax(k, ax, _LOWER)
            hi = _ax(k, ax, _UPPER)
            faces.append(((2.0 * lo * hi / (lo + hi)) * inv_dx2).astype(xp.float32))
        return faces

    def _wall_conductances(self, k: Any, dx: float, xp: Any) -> list[tuple[Any, Any]]:
        """Dirichlet wall coefficients (``2 k_edge / dx^2``), one pair per axis.

        The wall sits half a cell outside the last cell centre, so the flux
        into that cell is ``2 k_edge (T_wall - T_edge) / dx`` per unit area.
        """
        inv_dx2 = 1.0 / (dx * dx)
        out = []
        for ax in range(k.ndim):
            lo = (2.0 * _ax(k, ax, slice(0, 1)) * inv_dx2).astype(xp.float32)
            hi = (2.0 * _ax(k, ax, slice(-1, None)) * inv_dx2).astype(xp.float32)
            out.append((lo, hi))
        return out

    def _diagonal(self, k: Any, faces: list[Any], dx: float, xp: Any) -> Any:
        """Sum of the face coefficients touching each cell [W/m^3/K]."""
        diag = xp.zeros(k.shape, dtype=xp.float32)
        for ax, kf in enumerate(faces):
            _ax(diag, ax, _LOWER)[...] += kf
            _ax(diag, ax, _UPPER)[...] += kf
        if self.boundary == "dirichlet":
            for ax, (lo, hi) in enumerate(self._wall_conductances(k, dx, xp)):
                _ax(diag, ax, slice(0, 1))[...] += lo
                _ax(diag, ax, slice(-1, None))[...] += hi
        return diag

    def _perfusion_coefficient(self, medium: ThermalMedium, b: Any) -> Any:
        """``w_b rho_b C_b`` [W/m^3/K]."""
        w = b.asarray(medium.perfusion, b.xp.float32)
        return (w * np.float32(self.blood_density * self.blood_specific_heat)).astype(b.xp.float32)

    def _inv_rho_c(self, medium: ThermalMedium, b: Any) -> Any:
        """``1 / (rho C)`` [m^3 K/J]."""
        return b.asarray(1.0 / medium.volumetric_heat_capacity.astype(np.float64), b.xp.float32)

    # ---------- the solve ----------

    def solve(
        self,
        temperature0: Any,
        q: Any,
        medium: ThermalMedium,
        dt: float,
        n_steps: int,
        *,
        record_every: int | None = None,
        dose: bool = False,
        dose0: Any = None,
    ) -> ThermalResult:
        """Integrate the Pennes equation for ``n_steps`` steps of ``dt``.

        Parameters
        ----------
        temperature0:
            Initial temperature field [C] (``T0``), shape of ``medium``.
        q:
            Volumetric heating [W/m^3], held constant over the solve. A
            :class:`~caustica.sensors.HeatingSource`, an array of the medium
            shape, a scalar, or ``None`` for a pure diffusion/perfusion run
            (cooling phases and the analytic gates).
        medium:
            :class:`~caustica.thermal.properties.ThermalMedium` — carries
            ``dx``.
        dt, n_steps:
            Outer step [s] and how many. ``dt`` is the SAMPLING step: if it
            exceeds the stability bound the solve is refused, or (with
            ``on_unstable="substep"``) split internally so recorded times
            stay exactly on the caller's grid.
        record_every:
            Keep ``T`` every this many outer steps (plus the first and last).
            ``None`` records nothing but the final state.
        dose:
            Accumulate CEM43 during the integration.
        dose0:
            Dose already accumulated before this solve — the way to chain a
            sonication and its cooling phase, or a duty cycle, into one dose
            map: pass the previous result's ``dose_cem43`` together with its
            ``temperature`` as the new ``temperature0``.
        """
        if not isinstance(medium, ThermalMedium):
            raise TypeError(
                f"medium must be a caustica.thermal.ThermalMedium, got "
                f"{type(medium).__name__}. Build one with ThermalMedium.homogeneous / "
                f".from_id_map / .from_medium — the acoustic Medium has no k, C or w_b."
            )
        if not (np.isfinite(dt) and dt > 0):
            raise ValueError(f"dt must be a positive finite number of seconds, got {dt}")
        n_steps = int(n_steps)
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if record_every is not None:
            if int(record_every) < 1:
                raise ValueError(f"record_every must be >= 1 or None, got {record_every}")
            record_every = int(record_every)
        if dose0 is not None and not dose:
            raise ValueError(
                "dose0 was given but dose=False, so the dose already accumulated would "
                "be dropped and this solve would report no dose at all. Pass dose=True "
                "to continue accumulating, or drop dose0 to start fresh."
            )

        b = get_backend(self.backend)
        xp = b.xp
        shape = medium.shape

        if tuple(np.shape(temperature0)) != shape:
            raise ValueError(
                f"temperature0 has shape {tuple(np.shape(temperature0))} but the "
                f"thermal medium is {shape}."
            )
        t_ref = np.float32(self.reference_temperature_c)
        # The state is the RISE above t_ref (see the module docstring): the
        # subtraction of a nearby float is exact, and every neighbour
        # difference downstream keeps its significant digits instead of
        # spending them on a 37 C offset that cancels anyway. The .copy() in
        # asarray-then-subtract is implicit (the subtraction allocates), so
        # the caller's temperature0 is never written through.
        theta = b.asarray(temperature0, xp.float32) - t_ref
        if not bool(xp.isfinite(theta).all()):
            raise ValueError(
                "temperature0 contains non-finite values; a solve cannot start from a "
                "field that is not a temperature."
            )

        q_dev, q_kind = self._prepare_q(q, medium, b)

        # ---- coefficients (uploaded once, exactly like the acoustic engine) ----
        k_dev = b.asarray(medium.k, xp.float32)
        faces = self._face_conductances(k_dev, medium.dx, xp)
        walls = (
            self._wall_conductances(k_dev, medium.dx, xp) if self.boundary == "dirichlet" else None
        )
        perf = self._perfusion_coefficient(medium, b)
        perfused = medium.is_perfused
        inv_rho_c = self._inv_rho_c(medium, b)

        diag = self._diagonal(k_dev, faces, medium.dx, xp)
        diag += perf
        diag *= inv_rho_c
        peak_diag = float(diag.max())
        dt_stable = float("inf") if peak_diag <= 0.0 else 1.0 / peak_diag
        del diag

        n_sub = self._resolve_substeps(dt, dt_stable, medium)
        dt_sub = dt / n_sub

        # ---- per-step constants (all in "rise above t_ref" coordinates) ----
        coeff = (inv_rho_c * np.float32(dt_sub)).astype(xp.float32)
        # The perfusion sink is -perf*(T - T_a) = -perf*(theta + (t_ref - T_a));
        # with the default reference (t_ref == T_a) the constant term is zero
        # and the whole sink is one multiply.
        ref_minus_ta = np.float32(self.reference_temperature_c - self.arterial_temperature_c)
        perf_offset = (
            (perf * ref_minus_ta).astype(xp.float32)
            if perfused and float(ref_minus_ta) != 0.0
            else None
        )
        theta_wall = np.float32(self.boundary_temperature_c) - t_ref

        work = xp.zeros(shape, dtype=xp.float32)
        scratch = xp.zeros(shape, dtype=xp.float32) if perfused else None
        diff_bufs = [
            xp.zeros(_ax(theta, ax, _LOWER).shape, dtype=xp.float32) for ax in range(len(shape))
        ]

        theta_max = theta.copy()
        dose_map = None
        rate = None
        absolute = None
        if dose:
            dose_map = (
                xp.zeros(shape, dtype=xp.float32)
                if dose0 is None
                else b.asarray(np.asarray(dose0, dtype=np.float32), xp.float32).copy()
            )
            if dose_map.shape != shape:
                raise ValueError(f"dose0 has shape {dose_map.shape} but the medium is {shape}.")
            # CEM43 is defined on the ABSOLUTE temperature, so the reference
            # goes back on for the dose rate (one add per sub-step, only when
            # a dose was asked for).
            absolute = theta + t_ref
            rate = cem43_rate(absolute, xp)
        half_dt_min = np.float32(0.5 * dt_sub / 60.0)

        samples: list[np.ndarray] = []
        times: list[float] = []

        def _record(step_idx: int) -> None:
            samples.append(np.array(b.to_numpy(theta), dtype=np.float32, copy=True) + t_ref)
            times.append(step_idx * dt)

        if record_every is not None:
            _record(0)

        for step in range(1, n_steps + 1):
            for _ in range(n_sub):
                # --- conduction: sum of face fluxes (energy-conserving) ---
                work.fill(0)
                for ax, kf in enumerate(faces):
                    buf = diff_bufs[ax]
                    xp.subtract(_ax(theta, ax, _UPPER), _ax(theta, ax, _LOWER), out=buf)
                    buf *= kf
                    _ax(work, ax, _LOWER)[...] += buf
                    _ax(work, ax, _UPPER)[...] -= buf
                if walls is not None:
                    for ax, (lo, hi) in enumerate(walls):
                        _ax(work, ax, slice(0, 1))[...] += lo * (
                            theta_wall - _ax(theta, ax, slice(0, 1))
                        )
                        _ax(work, ax, slice(-1, None))[...] += hi * (
                            theta_wall - _ax(theta, ax, slice(-1, None))
                        )
                # --- source and perfusion sink ---
                if q_dev is not None:
                    work += q_dev
                if perfused:
                    xp.multiply(perf, theta, out=scratch)
                    work -= scratch
                    if perf_offset is not None:
                        work -= perf_offset
                # --- forward Euler on rho C dT/dt = work ---
                work *= coeff
                theta += work

                xp.maximum(theta_max, theta, out=theta_max)
                if dose_map is not None:
                    # Trapezoid in R^(43-T) over the sub-step: exact for a
                    # constant temperature (43 C for 60 s == 1.0 CEM43) and
                    # second-order for a transient. The new rate becomes the
                    # next sub-step's old one, so it is computed once.
                    xp.add(theta, t_ref, out=absolute)
                    new_rate = cem43_rate(absolute, xp)
                    increment = new_rate + rate
                    increment *= half_dt_min
                    dose_map += increment
                    rate = new_rate

            if record_every is not None and step % record_every == 0:
                _record(step)
            if step % self.finite_check_every == 0 and step != n_steps:
                self._guard_finite(theta, dose_map, xp, step=step, dt=dt, n_sub=n_sub)

        self._guard_finite(theta, dose_map, xp, step=n_steps, dt=dt, n_sub=n_sub)
        if record_every is not None and (not times or times[-1] != n_steps * dt):
            _record(n_steps)

        return ThermalResult(
            temperature=np.array(b.to_numpy(theta), dtype=np.float32, copy=True) + t_ref,
            temperature_max=np.array(b.to_numpy(theta_max), dtype=np.float32, copy=True) + t_ref,
            dose_cem43=(
                None
                if dose_map is None
                else np.array(b.to_numpy(dose_map), dtype=np.float32, copy=True)
            ),
            samples=samples,
            times=times,
            dt=float(dt),
            n_steps=n_steps,
            t_end_s=float(dt * n_steps),
            substeps=n_sub,
            meta={
                "scheme": SCHEME,
                "backend": b.name,
                "boundary": self.boundary,
                "boundary_temperature_c": self.boundary_temperature_c,
                "dt_stable_s": dt_stable,
                "dt_sub_s": dt_sub,
                "q": q_kind,
                "perfusion_active": perfused,
                "blood_density": self.blood_density,
                "blood_specific_heat": self.blood_specific_heat,
                "arterial_temperature_c": self.arterial_temperature_c,
                "reference_temperature_c": self.reference_temperature_c,
                "dose_accumulated": dose_map is not None,
            },
        )

    # ---------- helpers ----------

    def _prepare_q(self, q: Any, medium: ThermalMedium, b: Any) -> tuple[Any, str]:
        """Validate ``Q`` and upload it; returns ``(device array or None, tag)``."""
        if q is None:
            return None, "none"
        shape = medium.shape
        if isinstance(q, HeatingSource):
            if abs(q.dx - medium.dx) > 1e-12 * max(q.dx, medium.dx):
                raise ValueError(
                    f"the HeatingSource was built on a grid with dx = {q.dx * 1e3:.6g} mm "
                    f"but this thermal medium has dx = {medium.dx * 1e3:.6g} mm. Q and T "
                    f"would be integrated on different grids, which no error downstream "
                    f"would reveal."
                )
            if q.q.shape != shape:
                raise ValueError(
                    f"the HeatingSource covers the record region {q.region} (shape "
                    f"{q.q.shape}) but the thermal medium is {shape}. Either solve on "
                    f"the region, or place Q into the full grid explicitly with "
                    f"heating_source.embed({shape})."
                )
            return b.asarray(q.q, b.xp.float32), f"HeatingSource({q.alpha_model})"
        arr = np.asarray(q, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full(shape, float(arr), dtype=np.float32)
        if arr.shape != shape:
            raise ValueError(f"Q has shape {arr.shape} but the thermal medium is {shape}.")
        if not np.isfinite(arr).all():
            raise ValueError("Q contains non-finite values.")
        return b.asarray(arr, b.xp.float32), "array"

    def _resolve_substeps(self, dt: float, dt_stable: float, medium: ThermalMedium) -> int:
        """Refuse or split a time step above the stability bound."""
        if dt <= dt_stable:
            return 1
        if self.on_unstable == "substep":
            n_sub = int(ceil(dt / dt_stable))
            log.info(
                "pennes: dt=%.6g s exceeds the stability bound %.6g s; sub-stepping %d x %.6g s",
                dt,
                dt_stable,
                n_sub,
                dt / n_sub,
            )
            return n_sub
        d = medium.diffusivity
        raise ThermalStabilityError(
            f"dt = {dt:.6g} s is above the stability bound {dt_stable:.6g} s of the "
            f"explicit scheme for this medium (dx = {medium.dx * 1e3:.6g} mm, diffusivity "
            f"up to {float(d.max()):.4g} m^2/s"
            + (
                f", perfusion up to {float(medium.perfusion.max()):.4g} 1/s"
                if medium.is_perfused
                else ""
            )
            + f"). An explicit diffusion step past this limit does not fail loudly — it "
            f"grows a checkerboard that still looks like a temperature field — so it is "
            f"refused. Fix it in one of three ways: pass dt <= {dt_stable:.6g} s (and "
            f"{int(ceil(dt / dt_stable))}x more steps for the same duration), construct "
            f"the solver with PennesSolver(on_unstable='substep') to have it split the "
            f"step internally while keeping your sampling times, or coarsen dx (the "
            f"bound scales as dx^2)."
        )

    def _guard_finite(
        self, temperature: Any, dose_map: Any, xp: Any, *, step: int, dt: float, n_sub: int
    ) -> None:
        """Stop the run the moment the state stops being a number.

        Same reasoning as the acoustic engine's ``_guard_finite``: nothing
        downstream of a NaN is a physical result, and a dose map full of NaN
        is a wrong answer wearing the shape of an answer. The check is a
        reduction, so it runs on a cadence rather than every step.
        """
        for what, fieldarr in (("temperature", temperature), ("CEM43 dose", dose_map)):
            if fieldarr is None:
                continue
            if bool(xp.isfinite(fieldarr).all()):
                continue
            raise ThermalDivergedError(
                f"the {what} field stopped being finite at outer step {step} "
                f"(t = {step * dt:.6g} s, {n_sub} sub-step(s) per step). Nothing "
                f"downstream of this point is a physical result. Usual causes, most "
                f"common first: a heat source far outside the range float32 can carry "
                f"(Q of order 1e30 W/m^3 overflows within a few steps), a temperature "
                f"driven above ~170 C where the CEM43 rate R^(43-T) itself overflows "
                f"float32, or a medium whose k / rho / C make the assembled step "
                f"unstable in a way the bound could not see (check "
                f"PennesSolver.stable_dt(medium) against your dt)."
            )
