"""One registry shape for every extensible axis (K15, PLAN.md §2 rule 6).

caustica has five plugin seams — solvers, medium kinds, array kinds, backends
and report renderers — and every one of them needs the same four things:

* a name -> implementation table a third party can add to,
* an ``importlib.metadata`` entry-point group, so adding one needs no change
  to caustica's source,
* a **lazy** scan: a registry that has never been asked a question has never
  swept the installed distributions — ``import caustica`` does not pay for it,
* a lookup failure that lists what IS registered and names the group to
  register through.

This module is that shape, written once. :class:`~caustica.config.kinds.KindRegistry`
adds the pydantic discriminated union on top of it; :class:`FactoryRegistry`
covers the axes whose implementation is a plain callable (backend factories,
report renderers); :mod:`caustica.solvers.registry` keys on a class attribute.

The core implementations register through these same doors — there is no
private path. That is the seam's continuous proof: if registration breaks,
caustica's own solvers, kinds, backends and reports break first.

Entry-point group names are **frozen** (M10n): they are part of the public
contract, so a plugin written against caustica 0.1 keeps loading. A package
declares them in its ``pyproject.toml``::

    [project.entry-points."caustica.solvers"]
    my_solver = "my_pkg.solver:MySolver"

    [project.entry-points."caustica.backends"]
    my_backend = "my_pkg.backend:make_backend"

See ``docs/extending.md`` for a working skeleton of all five.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import Any, Generic, TypeVar

log = logging.getLogger("caustica")

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

#: Entry-point group for third-party solvers.
SOLVER_GROUP = "caustica.solvers"
#: Entry-point group for third-party job ``medium`` kinds.
MEDIUM_KIND_GROUP = "caustica.medium_kinds"
#: Entry-point group for third-party job ``source.array`` kinds.
ARRAY_KIND_GROUP = "caustica.array_kinds"
#: Entry-point group for third-party compute backends.
BACKEND_GROUP = "caustica.backends"
#: Entry-point group for third-party report renderers.
REPORT_RENDERER_GROUP = "caustica.report_renderers"

#: The five frozen groups, in the order M10n fixed them. A test asserts this
#: tuple verbatim — renaming a group breaks every plugin already installed.
ENTRY_POINT_GROUPS = (
    SOLVER_GROUP,
    MEDIUM_KIND_GROUP,
    ARRAY_KIND_GROUP,
    BACKEND_GROUP,
    REPORT_RENDERER_GROUP,
)


class UnknownPluginError(KeyError):
    """Lookup miss on a registry, carrying an actionable message.

    A ``KeyError`` subclass so ``except KeyError`` around a registry lookup
    keeps working, but with a plain ``__str__``: ``KeyError`` reports
    ``repr(args[0])``, which wraps a multi-sentence message in quotes and
    escapes its punctuation — and the CLI prints these verbatim to the user.
    """

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def _label_of(obj: Any) -> str:
    return str(getattr(obj, "__name__", None) or repr(obj))


def same_definition(a: Any, b: Any) -> bool:
    """True when ``b`` is ``a`` re-executed — a module reload, not a collision.

    ``importlib.reload`` (and therefore ``%autoreload 2`` in a notebook, the
    workflow this library is aimed at) re-runs the module body and produces
    NEW objects carrying the SAME registry names. Identity alone would call
    that a name collision and report an object as colliding with itself,
    halfway through the reload — leaving the module a mix of new and stale
    objects. Matching module + qualified name keeps the collision guard for
    two genuinely different implementations while letting a redefinition
    replace itself.

    ``<lambda>`` is refused: two unrelated module-level lambdas share both
    parts of that identity, so a factory registry would let one silently
    replace the other under the same name (M10n review). An anonymous
    function cannot claim to be a redefinition of anything.
    """
    missing = object()
    a_id = (getattr(a, "__module__", missing), getattr(a, "__qualname__", missing))
    b_id = (getattr(b, "__module__", missing), getattr(b, "__qualname__", missing))
    if missing in a_id or "<lambda>" in str(a_id[1]):
        return False
    return a_id == b_id


class PluginRegistry(Generic[T]):
    """name -> implementation, with a lazily scanned entry-point group behind it.

    Subclasses supply the ergonomics of their axis: how a registration key is
    derived (from a ``kind`` field, a ``name`` attribute, or the entry-point
    name itself) and what an acceptable implementation looks like.
    """

    def __init__(self, label: str, group: str, plural: str | None = None) -> None:
        #: Singular noun for one entry ("solver", "medium kind"); used in
        #: every message this registry raises.
        self.label = label
        #: The entry-point group third parties register through.
        self.group = group
        #: Plural used by the "Third-party X are added through ..." sentence.
        self.plural = plural or f"{label}s"
        self._items: dict[str, T] = {}
        self._loaded = False
        self._hooks: list[Callable[[], None]] = []

    # ---- registration ----

    def add(self, name: str, obj: T) -> T:
        """Register ``obj`` under ``name`` (name collision = error)."""
        self._validate(name, obj)
        held = self._items.get(name)
        if held is not None and held is not obj and not same_definition(held, obj):
            raise ValueError(self.collision_message(name, held))
        self._items[name] = obj
        try:
            self._notify()
        except Exception:
            # All or nothing. If wiring the entry into its consumers fails it
            # must not linger: `available()` would then advertise something
            # the consumer refuses — and inside :meth:`discover` the failure
            # is logged as "plugin failed to load", which would make the
            # inconsistency look explained.
            self._items.pop(name, None)
            with contextlib.suppress(Exception):
                self._notify()
            raise
        return obj

    def _validate(self, name: str, obj: Any) -> None:
        """Refuse an unusable implementation. Default: accept anything."""

    def collision_message(self, name: str, held: Any) -> str:
        """Text for two genuinely different implementations claiming one name."""
        return f"{self.label} '{name}' already registered by {_label_of(held)}"

    def on_change(self, hook: Callable[[], None]) -> None:
        """Run ``hook`` whenever the entry set changes.

        A hook re-registered from the same place replaces its predecessor —
        otherwise a module reload would leave the old module's hook installed
        forever, rebuilding objects nobody references any more.
        """
        self._hooks = [h for h in self._hooks if not same_definition(h, hook)]
        self._hooks.append(hook)

    def _notify(self) -> None:
        for hook in self._hooks:
            hook()

    def _forget(self, name: str) -> None:
        """Remove an entry. Teardown only (tests, plugin fixtures) — not an API."""
        if self._items.pop(name, None) is not None:
            self._notify()

    # ---- discovery ----

    def discover(self) -> None:
        """Load third-party entries from the entry-point group, once."""
        if self._loaded:
            return
        self._loaded = True  # set first: a broken scan must not retry forever
        # Imported here, not at module scope. Not for the import cost:
        # pydantic pulls `importlib.metadata` in (via pydantic.plugin._loader)
        # as soon as a model class is built, which `import caustica` does, so
        # there is nothing to save (measured, M10n review). It is here so that
        # `caustica.core.backend` — which owns a registry and IS on the
        # `import caustica` path — states this dependency where it uses it.
        from importlib import metadata  # noqa: PLC0415

        try:
            eps = metadata.entry_points(group=self.group)
        except Exception as exc:  # pragma: no cover - metadata backend quirks
            log.warning("could not scan %s entry points: %s", self.group, exc)
            return
        for ep in eps:
            try:
                self._accept(ep.name, ep.load())
            except Exception as exc:  # a broken plugin must not break caustica
                log.warning(
                    "%s plugin '%s' failed to load: %s: %s",
                    self.group,
                    ep.name,
                    type(exc).__name__,
                    exc,
                )

    def _accept(self, ep_name: str, obj: Any) -> None:
        """Turn one loaded entry point into a registration.

        The default keys on the entry-point NAME. Axes that carry their name
        inside the object (a ``kind`` field, a ``name`` attribute) override
        this so the declaration in ``pyproject.toml`` cannot disagree with
        the implementation.
        """
        self.add(ep_name, obj)

    # ---- lookup ----

    def get(self, name: str) -> T:
        """Look up an implementation by name."""
        self.discover()
        try:
            return self._items[name]
        except KeyError:
            raise UnknownPluginError(self.missing_message(name)) from None

    def missing_message(self, name: str) -> str:
        """The actionable text for an unregistered name: what exists, and how to add."""
        return (
            f"unknown {self.label} '{name}'. Available: "
            f"{', '.join(self.available()) or '(none)'}. Third-party {self.plural} are "
            f"added through the '{self.group}' entry-point group."
        )

    def available(self) -> tuple[str, ...]:
        """Sorted names of everything registered (scanning the group first)."""
        self.discover()
        return tuple(sorted(self._items))


class FactoryRegistry(PluginRegistry[Callable[..., Any]]):
    """A registry whose entries are plain callables, keyed by their own name.

    For the axes where the implementation is a function rather than a class:
    backend factories (:mod:`caustica.core.backend`) and report renderers
    (:mod:`caustica.report.renderers`). The entry-point name IS the registry
    key here — a function has no field to read one out of.
    """

    def register(self, name: str) -> Callable[[F], F]:
        """Decorator: register the decorated callable under ``name``."""

        def decorate(fn: F) -> F:
            self.add(name, fn)
            return fn

        return decorate

    def _validate(self, name: str, obj: Any) -> None:
        if not callable(obj):
            raise TypeError(f"{self.label} '{name}' must be callable, got {type(obj).__name__}")
