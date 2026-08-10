"""Solver contract: capability declaration, run policy, result container.

The multi-solver architecture rests on three pieces:

* :class:`SolverCaps` — a declarative statement of what a solver can do.
  ``Simulation``/tests validate a setup against it BEFORE any compute, so an
  unsupported combination fails at setup time with an explanation (the
  "COMSOL feel" contract) instead of silently producing nonsense.
* :class:`CWRunSpec` — the steady-state run policy (settling, tolerance,
  record window), a direct port of the notebook's validated scheme.
* :class:`SolverResult` — what every solver returns: the f0 phasor and the
  time-domain peak over a record region, plus honest run metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from pydantic import Field, model_validator

from hifusim.config.models import HifusimModel
from hifusim.core.grid import Grid
from hifusim.medium import Medium
from hifusim.sources import CWSource


class SolverCapabilityError(ValueError):
    """A setup asked a solver for something it declares it cannot do."""


@dataclass(frozen=True)
class SolverCaps:
    """What a solver supports; checked before compute."""

    ndim: frozenset[int]
    nonlinear: bool
    drive: frozenset[str]  # e.g. {"cw"}
    backends: frozenset[str]  # e.g. {"numpy", "cupy"} or {"external"}
    absorption: frozenset[str] = frozenset({"exponential"})


class CWRunSpec(HifusimModel):
    """Steady-state CW run policy (exact-period dt is implied and mandatory).

    Settling semantics (notebook port): the solver waits ``tof_periods``
    (computed from geometry) plus at least ``min_settle_periods``; from then
    on it stops as soon as the per-period peak changes by less than
    ``convergence_tol`` (relative), capped at ``max_settle_periods`` after
    time-of-flight. Then it records ``n_record_periods`` for the phasor.
    """

    cfl: float = Field(0.48, gt=0.0, le=0.6)
    cfl_hard_max: float = Field(0.50, gt=0.0, le=0.6)
    min_settle_periods: int = Field(8, ge=0)
    max_settle_periods: int = Field(96, ge=1)
    convergence_tol: float = Field(0.01, ge=0.0)
    n_record_periods: int = Field(2, ge=1)
    t_end_min_us: float | None = Field(None, gt=0.0)

    @model_validator(mode="after")
    def _ranges(self) -> CWRunSpec:
        if self.max_settle_periods < self.min_settle_periods:
            raise ValueError(
                f"max_settle_periods ({self.max_settle_periods}) < "
                f"min_settle_periods ({self.min_settle_periods})"
            )
        if self.cfl > self.cfl_hard_max:
            raise ValueError(f"cfl ({self.cfl}) exceeds cfl_hard_max ({self.cfl_hard_max})")
        return self


@dataclass
class SolverResult:
    """Steady-state CW result over ``region`` of the active grid."""

    phasor: np.ndarray  # complex64, shape of region
    p_max: np.ndarray  # float32, shape of region
    region: tuple[slice, ...]
    dt: float
    spp: int
    steps_total: int
    t_end_s: float
    tof_periods: int
    converged_period: int
    settle_capped: bool
    convergence_history: list[tuple[int, float, float]]  # (period, peak, rel_change)
    phasors: dict[int, np.ndarray] = field(default_factory=dict)  # harmonic n -> phasor
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def amp(self) -> np.ndarray:
        """Derived |phasor| (never stored separately — project data contract)."""
        return np.abs(self.phasor).astype(np.float32)

    @property
    def phase(self) -> np.ndarray:
        """Derived phase [rad] (constant global offset; relative phase exact)."""
        return np.angle(self.phasor).astype(np.float32)

    def harmonic_amp(self, n: int) -> np.ndarray:
        """Derived |phasor| of harmonic ``n`` (requested via ``harmonics=`` at run)."""
        if n not in self.phasors:
            raise KeyError(
                f"harmonic {n} was not recorded (available: {sorted(self.phasors)}); "
                f"pass harmonics=(1, {n}) to run()."
            )
        return np.abs(self.phasors[n]).astype(np.float32)


def interior_slices(shape: tuple[int, ...], margin: int) -> tuple[slice, ...]:
    """Slices selecting the interior of ``shape`` with ``margin`` voxels shaved."""
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    if any(n <= 2 * margin for n in shape):
        raise ValueError(f"margin {margin} leaves no interior for shape {shape}")
    return tuple(slice(margin, n - margin) for n in shape)


class SolverBase(ABC):
    """Abstract solver. Subclasses declare ``name`` + ``caps`` and implement run()."""

    name: ClassVar[str]
    caps: ClassVar[SolverCaps]

    def validate(self, grid: Grid, medium: Medium, source: CWSource) -> None:
        """Check this setup against the solver's declared capabilities.

        Raises :class:`SolverCapabilityError` with a fix-it message; never
        silently degrades physics.
        """
        if grid.ndim not in self.caps.ndim:
            raise SolverCapabilityError(
                f"solver '{self.name}' supports {sorted(self.caps.ndim)}-D grids, "
                f"got {grid.ndim}-D."
            )
        if medium.shape != grid.shape:
            raise ValueError(f"medium shape {medium.shape} != grid shape {grid.shape}")
        if not medium.is_linear and not self.caps.nonlinear:
            raise SolverCapabilityError(
                f"solver '{self.name}' is linear-only but the medium has beta != 0. "
                f"Zero the medium's beta or pick a nonlinear solver (e.g. 'westervelt')."
            )
        if "cw" not in self.caps.drive:
            raise SolverCapabilityError(
                f"solver '{self.name}' does not support CW drive (caps: {sorted(self.caps.drive)})."
            )
        source.check_inside(grid)

    @abstractmethod
    def run(
        self,
        grid: Grid,
        medium: Medium,
        source: CWSource,
        spec: CWRunSpec | None = None,
        **kwargs: Any,
    ) -> SolverResult:
        """Execute one steady-state CW solve."""
