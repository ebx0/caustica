"""Sensors: physical quantities derived from a solved acoustic field (M18).

``sensors/`` did not exist in v1 on purpose (PLAN section 3): recording was
what ``run()``'s ``record_region``/``harmonics`` arguments and
:class:`~caustica.solvers.base.SolverResult` already did. The thermal module
is the first consumer that needs a *derived* field rather than a recorded
one, so the first sensor lands here: :class:`HeatingSource`, the bridge from
a steady-state CW pressure result to the volumetric heat deposition
``Q`` [W/m^3] that :class:`caustica.thermal.PennesSolver` integrates.

The physics
-----------
For a locally progressive (plane-like) wave the time-averaged intensity of
harmonic ``n`` is ``I_n = |P_n|^2 / (2 rho c)`` and the power absorbed per
unit volume is ``Q_n = 2 alpha_n I_n``, so::

    Q = sum_n alpha_n * |P_n|^2 / (rho * c)          [W/m^3]

with ``P_n`` the complex pressure amplitude of harmonic ``n`` in the
library-wide convention ``p(t) = Re{P exp(-i omega t)}`` (docs/conventions.md
section 1), so ``|P_n|`` is a peak amplitude, not an RMS one.

Two assumptions are baked into that expression, and both are stated rather
than hidden:

1. **Local plane-wave intensity.** ``|P|^2/(2 rho c)`` is the true
   time-averaged intensity only where the field is locally progressive. In a
   standing-wave region (a strong reflector, a resonant cavity) the reactive
   part of the field carries no net power and this over-states heating.
2. **alpha is absorption, not attenuation.** Only the absorbed part of the
   loss heats the tissue. ``Material.alpha_np_m`` is documented as an
   absorption coefficient (docs/conventions.md section 2); if a table is
   filled with *total attenuation* (absorption + scattering) instead, the
   heating computed here is an upper bound.

The harmonic-alpha contract
---------------------------
The v1 absorption model knows ``alpha`` at ONE frequency: the ``f0`` the
material table was built for (that is the whole reason
``caustica.config.job`` refuses to run a baked ``medium_volume`` at another
drive frequency). A multi-harmonic result therefore cannot be turned into
heat without an explicit statement of what absorption the harmonics see:

* ``alpha_power_law_y=y`` -> ``alpha_n = alpha_f0 * n**y``. ``y = 2`` is the
  classical thermoviscous law (water); soft tissue is nearer ``y ~ 1.1``.
* ``alpha_at_harmonics={n: alpha}`` -> the numbers, verbatim.
* ``harmonics=(1,)`` -> "fundamental only", said out loud.

If a result carries harmonics above the fundamental and none of the three is
given, :meth:`HeatingSource.from_result` REFUSES. Silently using only the
fundamental would under-report the heating (the harmonics are exactly the
part of the spectrum tissue absorbs hardest), and a dose map that is quietly
too small is the one failure mode this module must not have. The real fix is
M16 (power-law absorption with Kramers-Kronig dispersion in the solver
itself); until then the exponent is the caller's declared assumption.

Like :class:`~caustica.medium.Medium`, a :class:`HeatingSource` lives in HOST
memory: it is built once from a result and uploaded by the thermal solver.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from caustica.core.backend import CausticaWarning

if TYPE_CHECKING:  # pragma: no cover - typing only (keeps ``import caustica`` light)
    from caustica.medium import Medium
    from caustica.solvers.base import SolverResult

#: Provenance tags for :attr:`HeatingSource.alpha_model`.
ALPHA_MODEL_F0 = "f0_only"
ALPHA_MODEL_POWER_LAW = "power_law"
ALPHA_MODEL_EXPLICIT = "explicit"

#: Widest exponent accepted for ``alpha_power_law_y``. Measured tissue
#: exponents live in ~[0.9, 1.5], thermoviscous fluids at 2.0; outside
#: [0, 3] the "power law" describes no medium and is refused rather than
#: quietly extrapolated.
ALPHA_POWER_LAW_Y_RANGE = (0.0, 3.0)


def _as_float_map(value: Any, shape: tuple[int, ...], what: str) -> np.ndarray:
    """Broadcast ``value`` (scalar or full map) to ``shape`` as float64."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape not in ((), shape):
        raise ValueError(
            f"{what} has shape {arr.shape}, which is neither a scalar nor the field shape {shape}."
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"{what} contains non-finite values.")
    return np.broadcast_to(arr, shape)


@dataclass(frozen=True, eq=False)
class HeatingSource:
    """Volumetric heat deposition ``Q`` [W/m^3] over a record region.

    Attributes
    ----------
    q:
        float32 ``Q`` map, shape of :attr:`region`.
    region:
        The slices of the acoustic grid ``q`` covers (a
        :class:`~caustica.solvers.base.SolverResult` records a sub-volume).
    dx:
        Isotropic voxel spacing [m] of the grid ``q`` lives on. The thermal
        solver cross-checks this against its own medium: a ``Q`` built on a
        different grid than the one it is integrated on is silently wrong.
    harmonics:
        Which harmonics contributed, in the order they were summed.
    alpha_model:
        ``"f0_only"`` / ``"power_law"`` / ``"explicit"`` — how the absorption
        above the fundamental was obtained.
    meta:
        Provenance: the exponent used, the per-harmonic alpha range [Np/m],
        each harmonic's mean contribution, and the assumptions named in the
        module docstring.
    """

    q: np.ndarray
    region: tuple[slice, ...]
    dx: float
    harmonics: tuple[int, ...]
    alpha_model: str
    meta: dict[str, Any] = field(default_factory=dict)

    # ---------- constructors ----------

    @classmethod
    def from_result(
        cls,
        result: SolverResult,
        medium: Medium,
        dx: float,
        *,
        harmonics: tuple[int, ...] | None = None,
        alpha_power_law_y: float | None = None,
        alpha_at_harmonics: dict[int, Any] | None = None,
    ) -> HeatingSource:
        """Build ``Q`` from a CW result and the medium it was solved in.

        Parameters
        ----------
        result:
            A :class:`~caustica.solvers.base.SolverResult`; its ``phasors``
            (or the fundamental ``phasor``) and ``region`` are read.
        medium:
            The :class:`~caustica.medium.Medium` of the run — ``alpha`` (at
            f0), ``rho`` and ``c`` are sliced by ``result.region``.
        dx:
            Voxel spacing [m] (``grid.dx``).
        harmonics:
            Which harmonics to sum. Default: every harmonic the result
            carries. Passing ``(1,)`` is the explicit "fundamental only"
            opt-out of the refusal described in the module docstring.
        alpha_power_law_y, alpha_at_harmonics:
            The two ways to state absorption above f0; pass at most one.
        """
        phasors = dict(result.phasors) if result.phasors else {1: result.phasor}
        region = tuple(result.region)
        alpha, rho, c = medium.alpha, medium.rho, medium.c
        if len(region) != alpha.ndim:
            raise ValueError(
                f"result.region has rank {len(region)} but the medium is "
                f"{alpha.ndim}-D — they are not the same grid."
            )
        maps = {"alpha": alpha[region], "rho": rho[region], "c": c[region]}
        shape = tuple(np.shape(phasors[next(iter(phasors))]))
        for name, vol in maps.items():
            if vol.shape != shape:
                raise ValueError(
                    f"medium.{name} sliced by result.region has shape {vol.shape} "
                    f"but the recorded phasor has shape {shape}. The result and the "
                    f"medium do not describe the same run."
                )
        return cls.from_phasors(
            phasors,
            alpha=maps["alpha"],
            rho=maps["rho"],
            c=maps["c"],
            dx=dx,
            region=region,
            harmonics=harmonics,
            alpha_power_law_y=alpha_power_law_y,
            alpha_at_harmonics=alpha_at_harmonics,
        )

    @classmethod
    def from_phasors(
        cls,
        phasors: dict[int, np.ndarray],
        *,
        alpha: Any,
        rho: Any,
        c: Any,
        dx: float,
        region: tuple[slice, ...] | None = None,
        harmonics: tuple[int, ...] | None = None,
        alpha_power_law_y: float | None = None,
        alpha_at_harmonics: dict[int, Any] | None = None,
    ) -> HeatingSource:
        """Same physics as :meth:`from_result`, from bare arrays.

        This is the entry point for anything that is not a native
        ``SolverResult`` (an adapter's recorded field, an analytic pressure,
        a test): ``alpha`` [Np/m at f0], ``rho`` [kg/m^3] and ``c`` [m/s] may
        each be a scalar or a full map of the phasor shape.
        """
        if not phasors:
            raise ValueError("phasors is empty: there is no field to turn into heat.")
        if not (np.isfinite(dx) and dx > 0):
            raise ValueError(f"dx must be a positive finite spacing in meters, got {dx}")

        available = tuple(sorted(int(n) for n in phasors))
        if harmonics is None:
            requested = available
        else:
            requested = tuple(dict.fromkeys(int(n) for n in harmonics))
            missing = [n for n in requested if n not in phasors]
            if missing:
                raise ValueError(
                    f"harmonics {missing} were asked for but are not in the result "
                    f"(available: {list(available)}). Re-run with harmonics="
                    f"{tuple(sorted(set(available) | set(missing)))} to record them."
                )
        if any(n < 1 for n in requested):
            raise ValueError(f"harmonic numbers must be >= 1, got {requested}")

        shape = tuple(np.shape(phasors[requested[0]]))
        for n in requested:
            if tuple(np.shape(phasors[n])) != shape:
                raise ValueError(
                    f"harmonic {n} has shape {np.shape(phasors[n])} but harmonic "
                    f"{requested[0]} has {shape}: the phasors are not on one grid."
                )

        alpha_f0 = _as_float_map(alpha, shape, "alpha")
        rho_map = _as_float_map(rho, shape, "rho")
        c_map = _as_float_map(c, shape, "c")
        if (alpha_f0 < 0).any():
            raise ValueError("alpha must be >= 0 Np/m (absorption cannot be negative).")
        if (rho_map <= 0).any() or (c_map <= 0).any():
            raise ValueError("rho and c must be > 0 everywhere to convert |P|^2 into intensity.")

        alpha_by_h = _resolve_harmonic_alpha(
            requested,
            alpha_f0,
            available=available,
            alpha_power_law_y=alpha_power_law_y,
            alpha_at_harmonics=alpha_at_harmonics,
        )
        model = _alpha_model_name(requested, alpha_power_law_y, alpha_at_harmonics)

        # Q = sum_n alpha_n |P_n|^2 / (rho c)  ==  sum_n 2 alpha_n I_n.
        inv_rho_c = 1.0 / (rho_map * c_map)
        q = np.zeros(shape, dtype=np.float64)
        mean_by_harmonic: dict[int, float] = {}
        for n in requested:
            p = np.asarray(phasors[n])
            if not np.isfinite(p).all():
                raise ValueError(
                    f"the phasor of harmonic {n} contains non-finite values; the run "
                    f"that produced it did not converge on a physical field."
                )
            p2 = np.abs(p).astype(np.float64) ** 2
            contribution = alpha_by_h[n] * p2 * inv_rho_c
            q += contribution
            mean_by_harmonic[n] = float(contribution.mean())

        if region is None:
            region = tuple(slice(0, n_ax) for n_ax in shape)
        meta = {
            "alpha_model": model,
            "alpha_power_law_y": alpha_power_law_y,
            "alpha_np_m_range": {
                n: (float(np.min(alpha_by_h[n])), float(np.max(alpha_by_h[n]))) for n in requested
            },
            "mean_q_by_harmonic_w_m3": mean_by_harmonic,
            "harmonics_available": list(available),
            "intensity_model": "local plane wave: I = |P|^2 / (2 rho c)",
            "alpha_meaning": "absorption (not total attenuation); see caustica.sensors docstring",
        }
        return cls(
            q=np.ascontiguousarray(q, dtype=np.float32),
            region=tuple(region),
            dx=float(dx),
            harmonics=requested,
            alpha_model=model,
            meta=meta,
        )

    # ---------- diagnostics / helpers ----------

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.q.shape)

    @property
    def q_max(self) -> float:
        """Peak volumetric heating [W/m^3]."""
        return float(self.q.max())

    @property
    def total_power_w(self) -> float:
        """``sum(Q) * dx^ndim`` — total absorbed power [W] in the region.

        In 1-D/2-D this is power per unit area / per unit length; the number
        is only watts when the map is 3-D.
        """
        return float(self.q.sum()) * self.dx**self.q.ndim

    def embed(self, shape: tuple[int, ...]) -> np.ndarray:
        """``Q`` placed into a full grid of ``shape``, zero outside the region.

        The record region is usually smaller than the acoustic grid, while a
        thermal solve normally runs on the whole grid — this is the explicit
        way to say "no heating was recorded outside the region", instead of
        having the thermal solver guess an alignment.
        """
        shape = tuple(int(s) for s in shape)
        if len(shape) != self.q.ndim:
            raise ValueError(f"shape {shape} has rank {len(shape)}, need {self.q.ndim}")
        full = np.zeros(shape, dtype=np.float32)
        try:
            full[self.region] = self.q
        except ValueError as exc:
            raise ValueError(
                f"the region {self.region} does not fit a grid of shape {shape}: {exc}"
            ) from None
        return full

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"HeatingSource(shape={self.shape}, harmonics={self.harmonics}, "
            f"alpha_model={self.alpha_model!r}, q_max={self.q_max:.3g} W/m^3)"
        )


def _alpha_model_name(
    requested: tuple[int, ...],
    alpha_power_law_y: float | None,
    alpha_at_harmonics: dict[int, Any] | None,
) -> str:
    if alpha_at_harmonics is not None:
        return ALPHA_MODEL_EXPLICIT
    if alpha_power_law_y is not None and max(requested) > 1:
        return ALPHA_MODEL_POWER_LAW
    return ALPHA_MODEL_F0


def _resolve_harmonic_alpha(
    requested: tuple[int, ...],
    alpha_f0: np.ndarray,
    *,
    available: tuple[int, ...],
    alpha_power_law_y: float | None,
    alpha_at_harmonics: dict[int, Any] | None,
) -> dict[int, np.ndarray]:
    """The honesty contract of the module docstring, enforced.

    Returns one absorption map per requested harmonic, or raises naming both
    ways to supply the information that is missing.
    """
    if alpha_power_law_y is not None and alpha_at_harmonics is not None:
        raise ValueError(
            "pass EITHER alpha_power_law_y (alpha_n = alpha_f0 * n**y) OR "
            "alpha_at_harmonics ({n: alpha_np_m}), not both — two absorption "
            "models for one field is not a resolvable ambiguity."
        )
    shape = alpha_f0.shape
    high = [n for n in requested if n >= 2]

    if high and alpha_power_law_y is None and alpha_at_harmonics is None:
        raise ValueError(
            f"this result carries harmonics {high} above the fundamental, but the v1 "
            f"absorption model only knows alpha at f0 (it is baked into the material "
            f"table at one frequency). Heating them with the f0 value would "
            f"UNDER-REPORT the heating, because tissue absorbs harmonics more strongly "
            f"— and a dose map that is quietly too small is worse than no dose map. "
            f"State the model:\n"
            f"  - alpha_power_law_y=y  -> alpha_n = alpha_f0 * n**y (y=2 is classical "
            f"thermoviscous water; soft tissue is nearer y~1.1)\n"
            f"  - alpha_at_harmonics={{n: alpha_np_m}}  -> your own numbers, verbatim\n"
            f"  - harmonics=(1,)  -> fundamental only, said out loud\n"
            f"The real fix is M16 (power-law absorption inside the solver)."
        )

    if alpha_power_law_y is not None:
        y = float(alpha_power_law_y)
        lo, hi = ALPHA_POWER_LAW_Y_RANGE
        if not (np.isfinite(y) and lo <= y <= hi):
            raise ValueError(
                f"alpha_power_law_y={alpha_power_law_y} is outside the physical range "
                f"[{lo}, {hi}] (tissue ~1.1, thermoviscous fluids 2.0). Refusing to "
                f"extrapolate absorption with an exponent no medium has."
            )
        return {n: alpha_f0 * float(n) ** y for n in requested}

    if alpha_at_harmonics is not None:
        keys = {int(n): v for n, v in alpha_at_harmonics.items()}
        missing = [n for n in requested if n >= 2 and n not in keys]
        if missing:
            raise ValueError(
                f"alpha_at_harmonics is missing harmonic(s) {missing} "
                f"(given: {sorted(keys)}; requested: {list(requested)}; recorded: "
                f"{list(available)}). Add them, or restrict the sum with harmonics=."
            )
        out: dict[int, np.ndarray] = {}
        heterogeneous = float(alpha_f0.min()) != float(alpha_f0.max())
        for n in requested:
            if n not in keys:
                out[n] = alpha_f0  # n == 1: the baked f0 value
                continue
            if heterogeneous and np.asarray(keys[n]).ndim == 0:
                warnings.warn(
                    f"alpha_at_harmonics[{n}] is a single number but the medium's alpha "
                    f"at f0 is heterogeneous ({float(alpha_f0.min()):.3g}..."
                    f"{float(alpha_f0.max()):.3g} Np/m): the scalar is applied to EVERY "
                    f"voxel, so tissue contrast disappears at that harmonic. Pass a map "
                    f"of the field shape to keep it.",
                    CausticaWarning,
                    stacklevel=4,
                )
            arr = _as_float_map(keys[n], shape, f"alpha_at_harmonics[{n}]")
            if (arr < 0).any():
                raise ValueError(f"alpha_at_harmonics[{n}] has negative values.")
            out[n] = arr
        return out

    return {n: alpha_f0 for n in requested}
