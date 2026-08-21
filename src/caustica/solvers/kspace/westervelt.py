"""Westervelt nonlinear k-space PSTD solver (registry name: ``"westervelt"``).

The reference full-wave solver of the library — the direct descendant of the
production notebook's engine. Shares every line of numerics with `linear`
through :mod:`caustica.solvers.kspace.engine`; the only difference is the
Westervelt term ``dp_nl = -2 beta dt p (div u)`` in the pressure update,
which produces harmonic growth and shock steepening.

Accepts linear media too (beta = 0 volumes take the identical code path as
`linear`, which is exactly the M5 equivalence gate). Request harmonic
phasors with ``harmonics=(1, 2)`` to capture the 2f0 field in the same
record pass (leakage-free thanks to the exact-period dt policy).
"""

from __future__ import annotations

from typing import Any

from caustica.core.grid import Grid
from caustica.io.checkpoint import CheckpointSpec
from caustica.medium import Medium
from caustica.solvers.base import CWRunSpec, SolverBase, SolverCaps, SolverResult
from caustica.solvers.kspace.engine import run_cw_kspace_pstd
from caustica.solvers.registry import register
from caustica.sources import CWSource


@register
class WesterveltKSpacePSTD(SolverBase):
    """Nonlinear (Westervelt) full-wave k-space PSTD (1/2/3-D, CW steady state)."""

    name = "westervelt"
    caps = SolverCaps(
        ndim=frozenset({1, 2, 3}),
        nonlinear=True,
        drive=frozenset({"cw"}),
        backends=frozenset({"numpy", "cupy"}),
    )

    def run(
        self,
        grid: Grid,
        medium: Medium,
        source: CWSource,
        spec: CWRunSpec | None = None,
        backend: str = "auto",
        record_region: tuple[slice, ...] | None = None,
        reference_point: tuple[int, ...] | None = None,
        harmonics: tuple[int, ...] = (1,),
        checkpoint: CheckpointSpec | None = None,
        **kwargs: Any,
    ) -> SolverResult:
        if kwargs:
            raise TypeError(f"unknown run() options: {sorted(kwargs)}")
        spec = spec or CWRunSpec()
        self.validate(grid, medium, source)
        return run_cw_kspace_pstd(
            self.name,
            grid,
            medium,
            source,
            spec,
            backend=backend,
            record_region=record_region,
            reference_point=reference_point,
            nonlinear=True,
            harmonics=harmonics,
            checkpoint=checkpoint,
        )
