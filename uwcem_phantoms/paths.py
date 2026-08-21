"""Where the phantom package keeps its bytes.

Everything this package downloads, caches or ad-hoc exports lives under ONE
root so it stays self-contained (``uwcem_phantoms/_data``) — that is a
deliberate deviation from "packages hold no data": the phantom archives are
the package's reason to exist, and keeping them beside the code means a
checkout that has run ``fetch`` once works offline forever after.

Layout::

    _data/
      uwcem/<id>/{breastInfo.txt,mtype.zip,pval.zip}   raw repository files
      cache/<id>-<kind>.npz                            decoded arrays (fast reload)
      exports/<name>.npz                               ad-hoc simulation-ready outputs

The STANDARD DATASET is the one deliberate exception: :func:`dataset_dir`
resolves to ``<repo>/data/phantoms`` because it is a repo-level artifact other
code loads by a stable path (see its docstring).

Override the ``_data`` root with ``CAUSTICA_PHANTOM_DATA`` (absolute path)
when the checkout is read-only, e.g. a shared or CI cache. The env var does
NOT move :func:`dataset_dir` — pass ``--out`` / ``out_dir`` to the dataset
commands for that.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "CAUSTICA_PHANTOM_DATA"

_PACKAGE_DIR = Path(__file__).resolve().parent


def data_root() -> Path:
    """Root directory for downloads, caches and exports."""
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return _PACKAGE_DIR / "_data"


def raw_dir(phantom_id: str | None = None) -> Path:
    """Directory holding the untouched UWCEM files for ``phantom_id``."""
    base = data_root() / "uwcem"
    return base if phantom_id is None else base / phantom_id


def cache_dir() -> Path:
    """Directory holding decoded ``.npz`` caches of the raw text files."""
    return data_root() / "cache"


def export_dir() -> Path:
    """Directory holding simulation-ready phantom exports."""
    return data_root() / "exports"


def dataset_dir() -> Path:
    """``<repo>/data/phantoms`` — the standard aligned dataset for simulations.

    Deliberately OUTSIDE ``_data``: ``_data`` is this module's private plumbing
    (downloads, caches, ad-hoc exports), while the dataset is a repo-level
    artifact other code loads by a stable path. It sits next to ``src`` so a
    simulation script can say ``data/phantoms/uwcem-012304-dx0p25mm.npz``
    without knowing this package exists.
    """
    return _PACKAGE_DIR.parent / "data" / "phantoms"


def setups_dir() -> Path:
    """``<repo>/data/setups`` — stored, simulation-ready run geometries.

    Sits beside ``data/phantoms`` for the same reason: a setup names a phantom
    file by its stable repo-relative path, and both are repo artifacts rather
    than this package's private plumbing. Unlike the phantoms these files are
    small enough to live in git.
    """
    return _PACKAGE_DIR.parent / "data" / "setups"


def export_path(name: str) -> Path:
    """``exports/<name>.npz`` for a plain basename; refuses anything path-like.

    ``name`` reaches here from the GUI request body and from ``spec.name``,
    i.e. from outside. Joined naively it is not a name but a destination:
    ``"../../x"`` walks out of the export directory and ``"C:/Windows/x"``
    replaces it wholesale (``Path.__truediv__`` lets an absolute right-hand
    side win), after which :meth:`PhantomAsset.save` happily ``mkdir -p``s the
    way there and overwrites whatever it lands on (review finding, 2026-08-18).
    """
    # Reject BOTH separators on every platform: Path(name).name only knows
    # the native one, so on POSIX "x\\y" would pass here yet be unreadable
    # as an export name on Windows (first seen on the Linux CI leg,
    # 2026-08-22).
    if not name or Path(name).name != name or "\\" in name or name in (".", ".."):
        raise ValueError(
            f"export name must be a plain filename with no path separators, got {name!r}"
        )
    return export_dir() / f"{name}.npz"


def ensure_dir(path: Path) -> Path:
    """``mkdir -p`` that returns the path (so it can be inlined)."""
    path.mkdir(parents=True, exist_ok=True)
    return path
