"""Solver registry: name -> solver class, with third-party plugin support.

Built-in solvers self-register at package import via the :func:`register`
decorator. Third-party packages can ship solvers through the
``hifusim.solvers`` entry-point group; those load lazily on first lookup so
a broken plugin can never break ``import hifusim``.
"""

from __future__ import annotations

import logging
from importlib import metadata

from hifusim.solvers.base import SolverBase

log = logging.getLogger("hifusim")

_REGISTRY: dict[str, type[SolverBase]] = {}
_ENTRY_POINTS_LOADED = False


def register(cls: type[SolverBase]) -> type[SolverBase]:
    """Class decorator: add a solver to the registry (name collision = error)."""
    name = getattr(cls, "name", None)
    if not name or not isinstance(name, str):
        raise ValueError(f"solver class {cls.__name__} must define a string 'name'")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"solver name '{name}' already registered by {_REGISTRY[name].__name__}")
    _REGISTRY[name] = cls
    return cls


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        eps = metadata.entry_points(group="hifusim.solvers")
    except Exception as exc:  # pragma: no cover - metadata backend quirks
        log.warning("could not scan solver entry points: %s", exc)
        return
    for ep in eps:
        try:
            obj = ep.load()
            if isinstance(obj, type) and issubclass(obj, SolverBase):
                register(obj)
        except Exception as exc:  # a broken plugin must not break hifusim
            log.warning(
                "solver plugin '%s' failed to load: %s: %s", ep.name, type(exc).__name__, exc
            )


def get(name: str) -> type[SolverBase]:
    """Look up a solver class by registry name."""
    _load_entry_points()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown solver '{name}'. Available: {', '.join(available()) or '(none)'}"
        ) from None


def available() -> tuple[str, ...]:
    """Sorted names of all registered solvers."""
    _load_entry_points()
    return tuple(sorted(_REGISTRY))
