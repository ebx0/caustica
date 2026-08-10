"""Multi-solver registry and built-in solvers.

Usage::

    import hifusim.solvers as solvers
    Solver = solvers.get("linear")          # or "kwave", later "westervelt", "kzk"
    result = Solver().run(grid, medium, source)

Importing this package registers the built-in solvers; third-party packages
add theirs via the ``hifusim.solvers`` entry-point group.
"""

from hifusim.solvers.base import (
    CWRunSpec,
    SolverBase,
    SolverCapabilityError,
    SolverCaps,
    SolverResult,
    interior_slices,
)

# Importing the solver modules IS the built-in registration step.
from hifusim.solvers.kspace.linear import LinearKSpacePSTD
from hifusim.solvers.kspace.westervelt import WesterveltKSpacePSTD
from hifusim.solvers.kwave_adapter import KWaveSolver
from hifusim.solvers.registry import available, get, register

__all__ = [
    "CWRunSpec",
    "KWaveSolver",
    "LinearKSpacePSTD",
    "SolverBase",
    "SolverCapabilityError",
    "SolverCaps",
    "SolverResult",
    "WesterveltKSpacePSTD",
    "available",
    "get",
    "interior_slices",
    "register",
]
