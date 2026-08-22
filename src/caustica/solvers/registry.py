"""Solver registry: name -> solver class, with third-party plugin support.

Built-in solvers self-register at package import via the :func:`register`
decorator. Third-party packages can ship solvers through the
``caustica.solvers`` entry-point group; those load lazily on first lookup so
a broken plugin can never break ``import caustica``.

The mechanics (lazy scan, collision guard, actionable lookup failure) are
:class:`caustica.registry.PluginRegistry`, shared with the other four
extensible axes (K15). What is specific here: the registry key is the
solver class's own ``name`` attribute, so an entry-point declaration cannot
disagree with the class it points at.
"""

from __future__ import annotations

import logging
from typing import Any

from caustica.registry import SOLVER_GROUP, PluginRegistry
from caustica.solvers.base import SolverBase

log = logging.getLogger("caustica")


class SolverRegistry(PluginRegistry[type[SolverBase]]):
    """name -> solver class; the name comes from the class, not the caller."""

    def register(self, cls: type[SolverBase]) -> type[SolverBase]:
        """Class decorator: add a solver to the registry (name collision = error)."""
        name = getattr(cls, "name", None)
        if not name or not isinstance(name, str):
            raise ValueError(f"solver class {cls.__name__} must define a string 'name'")
        return self.add(name, cls)

    def _accept(self, ep_name: str, obj: Any) -> None:
        if isinstance(obj, type) and issubclass(obj, SolverBase):
            self.register(obj)
        else:
            log.warning("%s plugin '%s' is not a SolverBase subclass; ignored", self.group, ep_name)

    def collision_message(self, name: str, held: Any) -> str:
        # "solver NAME '...'": the pre-M10n wording, kept verbatim.
        return f"solver name '{name}' already registered by {held.__name__}"


#: The registry the built-in solvers register into (no private path).
solver_registry = SolverRegistry("solver", SOLVER_GROUP)

register = solver_registry.register
get = solver_registry.get
available = solver_registry.available
