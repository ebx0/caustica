"""Kind registries: the plugin seam for job ``medium`` and ``source.array`` kinds.

Same pattern as :mod:`caustica.solvers.registry`, applied to the two places
the job schema used to hard-code a closed pydantic union (K15, PLAN rule 6):
a name -> config-class registry, an ``importlib.metadata`` entry-point group
for third parties, and a discriminated union *built from the registry*
instead of written out by hand.

The core kinds (``homogeneous`` / ``scene`` / ``volume_import`` /
``medium_volume``; ``archimedean_spiral`` / ``bowl`` / ``elements``) register
through this exact door — there is no private path for them. That is the
continuous proof the seam works: if registration breaks, caustica's own job
schema breaks first.

Entry-point groups (a third-party package declares these in its
``pyproject.toml``)::

    [project.entry-points."caustica.medium_kinds"]
    my_phantom = "my_pkg.job:MyPhantomMediumConfig"

    [project.entry-points."caustica.array_kinds"]
    my_ring = "my_pkg.job:MyRingArrayConfig"

Scanning is LAZY — it happens the first time something asks for the union,
a lookup or the available names, which in practice is the first time
:mod:`caustica.config.job` is imported. ``import caustica`` never pays for
it (``caustica.config`` re-exports the job names through PEP 562).

Plugins should import their base class from **this** module, never from
:mod:`caustica.config.job`: job.py builds its unions during its own import,
so a plugin that imports job.py at module scope would be re-entering a
half-initialized module. (A plugin that fails to load is logged and skipped,
never fatal — the same contract the solver registry keeps.)
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, get_args, get_origin

import numpy as np
from pydantic import Field

from caustica.config.models import CausticaModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from caustica.core.grid import Grid
    from caustica.medium import Medium
    from caustica.sources import CWSource

log = logging.getLogger("caustica")

#: Entry-point group for third-party medium kinds.
MEDIUM_GROUP = "caustica.medium_kinds"
#: Entry-point group for third-party transducer/array kinds.
ARRAY_GROUP = "caustica.array_kinds"


# --------------------------------------------------------------- medium seam


@dataclass
class MediumPrep:
    """What a *grid-providing* medium kind hands back before the medium exists.

    ``medium_volume`` is the motivating case: the grid (shape + dx) comes
    from the file, and every cheap refusal — drive frequency vs. the baked
    alpha, focus-in-coupling-water, source-clears-PML — must run BEFORE the
    multi-GB property volumes are materialized. So the kind returns the
    geometry first and the medium behind a callable.
    """

    grid: Grid
    c_min: float
    labels: np.ndarray | None = None
    water_label: int | None = None
    make_medium: Callable[[], Medium] | None = None

    def build_medium(self) -> Medium:
        """Materialize the medium, releasing the kind's own reference to it.

        Single-shot on purpose: the closure typically holds a loaded volume,
        and on a full-size phantom that is GBs the caller must be able to
        drop (review 2026-08-19: the explicit path used to keep both alive).
        """
        if self.make_medium is None:
            raise RuntimeError("MediumPrep.build_medium() already consumed (or never supplied)")
        make, self.make_medium = self.make_medium, None
        return make()


class MediumKindConfig(CausticaModel):
    """Base class every ``medium`` kind subclasses (and registers).

    Two shapes are supported and both are first class:

    * the common one — the job's ``grid`` section defines the grid and the
      kind paints a :class:`~caustica.medium.Medium` onto it: implement
      :meth:`c_min` and :meth:`build`;
    * a *grid-providing* kind — the data file fixes shape and dx, so a job
      must NOT carry a ``grid`` section: set ``provides_grid = True`` and
      implement :meth:`prepare`.
    """

    kind: str

    #: True when the kind supplies the grid itself (then a job's ``grid``
    #: section is refused, and :meth:`prepare` replaces build/c_min).
    provides_grid: ClassVar[bool] = False

    def resolve_paths(self, base_dir: Path | None) -> MediumKindConfig:
        """Return a copy whose file references resolve against ``base_dir``.

        Relative paths in a job resolve against the JOB FILE, never the CWD
        (see :mod:`caustica.runner`) — a kind that reads files must honour
        that here. The default is a no-op for kinds that read nothing.
        """
        return self

    def c_min(self) -> float:
        """Lowest sound speed the kind can paint [m/s] (drives the ppw check)."""
        raise NotImplementedError(f"{type(self).__name__} must implement c_min()")

    def build(self, grid: Grid) -> Medium:
        """Materialize the medium on ``grid``."""
        raise NotImplementedError(f"{type(self).__name__} must implement build()")

    def prepare(self, drive: Any) -> MediumPrep:
        """Grid-providing kinds only: geometry now, medium behind a callable."""
        raise NotImplementedError(
            f"{type(self).__name__} sets provides_grid=True but does not implement prepare()"
        )


# ---------------------------------------------------------------- array seam


class ArrayKindConfig(CausticaModel):
    """Base class every ``source.array`` kind subclasses (and registers).

    The contract is deliberately small: report the geometric focal distance
    along +z (:meth:`focal_length_mm`, which is what a ``natural`` focus
    resolves to), hand back numbers a reload can falsify (:meth:`derived`),
    and voxelize yourself onto the grid (:meth:`build_source`).
    """

    kind: str

    def resolve_paths(self, base_dir: Path | None) -> ArrayKindConfig:
        """Return a copy whose file references resolve against ``base_dir``."""
        return self

    def focal_length_mm(self) -> float:
        """Geometric focal distance from the apex along +z [mm].

        A ``natural`` focus resolves to ``apex + focal_length_mm`` on the
        beam axis. Kinds that carry a radius of curvature just return it.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement focal_length_mm()")

    def derived(self) -> dict[str, float]:
        """Numbers a stored run records so a reload can falsify the geometry.

        The M6f "nothing is baked" rule: element positions are always
        re-derived; these values exist to detect a library change that
        silently builds a DIFFERENT transducer.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement derived()")

    def build_source(
        self,
        grid: Grid,
        drive: Any,
        apex_vox: tuple[int, int, int],
        focus: Any,
        phases_rad: tuple[float, ...] | None,
    ) -> tuple[CWSource, dict[str, Any]]:
        """Voxelize onto ``grid``; returns (source, extra derived entries)."""
        raise NotImplementedError(f"{type(self).__name__} must implement build_source()")


# -------------------------------------------------------------------- registry


def _kind_name(cls: type[CausticaModel], label: str) -> str:
    """The registry key: the ``kind`` field's Literal default."""
    field = cls.model_fields.get("kind") if hasattr(cls, "model_fields") else None
    if field is None:
        raise ValueError(f"{label} kind {cls.__name__} must declare a 'kind' field")
    if get_origin(field.annotation) is not Literal:
        raise ValueError(
            f"{label} kind {cls.__name__}: 'kind' must be annotated "
            f'Literal["<name>"] (got {field.annotation!r}) so the job schema '
            f"can discriminate on it"
        )
    (tag,) = get_args(field.annotation)
    if not isinstance(tag, str) or not tag:
        raise ValueError(f"{label} kind {cls.__name__}: 'kind' Literal must be a non-empty string")
    if field.default != tag:
        raise ValueError(
            f"{label} kind {cls.__name__}: 'kind' must default to {tag!r} "
            f"(got {field.default!r}) so the kind can be constructed in Python"
        )
    return tag


def _same_definition(a: Any, b: Any) -> bool:
    """True when ``b`` is ``a`` re-executed — a module reload, not a collision.

    ``importlib.reload`` (and therefore ``%autoreload 2`` in a notebook, the
    workflow this library is aimed at) re-runs the module body and produces
    NEW class objects carrying the SAME kind tags. Identity alone would call
    that a name collision and report a class as colliding with itself, halfway
    through the reload — leaving the module a mix of new and stale objects.
    Matching module + qualified name keeps the collision guard for two
    genuinely different classes while letting a redefinition replace itself.
    """
    return (a.__module__, a.__qualname__) == (b.__module__, b.__qualname__)


class KindRegistry:
    """name -> config class, plus the discriminated union built from it."""

    def __init__(self, label: str, group: str, base: type[CausticaModel]) -> None:
        self.label = label
        self.group = group
        self.base = base
        self._kinds: dict[str, type[CausticaModel]] = {}
        self._loaded = False
        self._hooks: list[Callable[[], None]] = []

    # ---- registration ----

    def register(self, cls: type[CausticaModel]) -> type[CausticaModel]:
        """Class decorator: add a kind (name collision = error)."""
        if not (isinstance(cls, type) and issubclass(cls, self.base)):
            raise TypeError(
                f"{self.label} kind {getattr(cls, '__name__', cls)!r} must subclass "
                f"{self.base.__module__}.{self.base.__name__}"
            )
        name = _kind_name(cls, self.label)
        held = self._kinds.get(name)
        if held is not None and held is not cls and not _same_definition(held, cls):
            raise ValueError(f"{self.label} kind '{name}' already registered by {held.__name__}")
        self._kinds[name] = cls
        try:
            self._notify()
        except Exception:
            # All or nothing. If wiring the kind into the job models fails, it
            # must not linger in the registry: `available()` and `caustica
            # schema` would then advertise a kind the schema refuses — and
            # inside :meth:`discover` the failure is logged as "plugin failed
            # to load", which would make the inconsistency look explained.
            self._kinds.pop(name, None)
            with contextlib.suppress(Exception):
                self._notify()
            raise
        return cls

    def on_change(self, hook: Callable[[], None]) -> None:
        """Run ``hook`` whenever the kind set changes (job.py rebuilds its models).

        A hook re-registered from the same place replaces its predecessor —
        otherwise a module reload would leave the old module's hook installed
        forever, rebuilding classes nobody references any more.
        """
        self._hooks = [h for h in self._hooks if not _same_definition(h, hook)]
        self._hooks.append(hook)

    def _notify(self) -> None:
        for hook in self._hooks:
            hook()

    def _forget(self, name: str) -> None:
        """Remove a kind. Teardown only (tests, plugin fixtures) — not an API."""
        if self._kinds.pop(name, None) is not None:
            self._notify()

    # ---- discovery ----

    def discover(self) -> None:
        """Load third-party kinds from the entry-point group, once."""
        if self._loaded:
            return
        self._loaded = True  # set first: a broken scan must not retry forever
        try:
            eps = metadata.entry_points(group=self.group)
        except Exception as exc:  # pragma: no cover - metadata backend quirks
            log.warning("could not scan %s entry points: %s", self.group, exc)
            return
        for ep in eps:
            try:
                obj = ep.load()
                if isinstance(obj, type) and issubclass(obj, self.base):
                    self.register(obj)
                else:
                    log.warning(
                        "%s plugin '%s' is not a %s subclass; ignored",
                        self.group,
                        ep.name,
                        self.base.__name__,
                    )
            except Exception as exc:  # a broken plugin must not break caustica
                log.warning(
                    "%s plugin '%s' failed to load: %s: %s",
                    self.group,
                    ep.name,
                    type(exc).__name__,
                    exc,
                )

    # ---- lookup ----

    def get(self, name: str) -> type[CausticaModel]:
        """Look up a kind's config class by name."""
        self.discover()
        try:
            return self._kinds[name]
        except KeyError:
            raise KeyError(
                f"unknown {self.label} kind '{name}'. Available: "
                f"{', '.join(self.available()) or '(none)'}. Third-party kinds are "
                f"added through the '{self.group}' entry-point group."
            ) from None

    def available(self) -> tuple[str, ...]:
        """Sorted names of all registered kinds."""
        self.discover()
        return tuple(sorted(self._kinds))

    def union(self) -> Any:
        """The discriminated union of every registered kind, in registration order.

        Registration order (not alphabetical) on purpose: pydantic prints the
        expected tags in union order, so the wording of an unknown-kind error
        stays stable as long as the core kinds keep their order.
        """
        self.discover()
        members = list(self._kinds.values())
        if not members:  # pragma: no cover - core kinds always register
            raise RuntimeError(f"no {self.label} kinds registered")
        if len(members) == 1:
            return members[0]
        union: Any = members[0]
        for cls in members[1:]:
            union = union | cls
        return Annotated[union, Field(discriminator="kind")]

    def annotation(self) -> Any:
        """The field annotation a job model uses for this axis.

        Not the union itself: a *deferred* annotation that asks the registry
        for the union every time a schema is generated. A model whose field
        is annotated this way picks up a kind registered after the model was
        defined as soon as it is rebuilt — pydantic only re-resolves a field
        annotation that failed to resolve, so a plain module global would go
        stale (found while writing the plugin test, M10m).

        The one cost: the field's *declared* annotation is ``Any``, so
        ``model_fields[...].annotation`` and ``typing.get_type_hints`` see
        ``Any`` where a closed union would show the member classes. The
        validation schema, the JSON Schema and serialization are unaffected.
        Ask the registry (:meth:`available`, :meth:`union`) — or
        ``caustica schema`` — for the kinds, never the field annotation.
        """
        return Annotated[Any, _LazyKindUnion(self)]


class _LazyKindUnion:
    """``Annotated`` metadata that expands to a registry's union on demand."""

    __slots__ = ("registry",)

    def __init__(self, registry: KindRegistry) -> None:
        self.registry = registry

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.registry.label} kinds: {', '.join(self.registry.available())}>"

    def __get_pydantic_core_schema__(self, source: Any, handler: Any) -> Any:
        return handler.generate_schema(self.registry.union())


#: The two registries the job schema builds its unions from.
medium_kinds = KindRegistry("medium", MEDIUM_GROUP, MediumKindConfig)
array_kinds = KindRegistry("array", ARRAY_GROUP, ArrayKindConfig)
