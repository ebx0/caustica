"""Linear k-space PSTD solver (registry name: ``"linear"``).

Thin wrapper over the shared engine (:mod:`caustica.solvers.kspace.engine`)
with the nonlinear term disabled and a linear-only capability declaration:
handing it a nonlinear medium fails at setup with a pointer to `westervelt`.
Use it for validation runs and fast parameter sweeps — it skips the
nonlinear update entirely. The engine docstring documents the numerics
(including the symmetric-absorption fix over the source notebook).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from caustica.core.grid import Grid
from caustica.io.checkpoint import CheckpointSpec
from caustica.medium import Medium
from caustica.solvers.base import CWRunSpec, SolverBase, SolverCaps, SolverResult
from caustica.solvers.kspace.engine import run_cw_kspace_pstd
from caustica.solvers.registry import register
from caustica.sources import CWSource


@register
class LinearKSpacePSTD(SolverBase):
    """Linear full-wave k-space PSTD (1/2/3-D, CW steady state)."""

    name = "linear"
    caps = SolverCaps(
        ndim=frozenset({1, 2, 3}),
        nonlinear=False,
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
        progress: Callable[[dict], None] | None = None,
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
            nonlinear=False,
            harmonics=harmonics,
            checkpoint=checkpoint,
            progress=progress,
        )
