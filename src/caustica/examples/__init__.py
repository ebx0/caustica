"""Packaged, self-contained example jobs — no external data, seconds on CPU.

The JSON files in this package ship inside the wheel so a fresh
``pip install caustica`` can run a real simulation with zero downloads.

**Never run a packaged example in place.** Relative output paths resolve
against the *job file* (see ``caustica.runner``), so running the packaged
copy directly would try to create ``site-packages/caustica/examples/runs/…``
— read-only in many environments, pollution in the rest. Copy the job into
your own directory first::

    caustica example water_bowl_mini          # CLI: copies into the CWD
    caustica run water_bowl_mini.json

or from Python: ``caustica.examples.copy("water_bowl_mini", my_dir)``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# Wheels install unpacked, so the package directory is a real filesystem
# path; the CLI hands these paths straight to validate/run.
_HERE = Path(__file__).resolve().parent

__all__ = ["available", "copy", "path"]


def available() -> tuple[str, ...]:
    """Names of the packaged example jobs (without the ``.json`` suffix)."""
    return tuple(sorted(p.stem for p in _HERE.glob("*.json")))


def path(name: str) -> Path:
    """Path of a packaged example job — for *reading* (validate, copy).

    Raises ``KeyError`` with the available names on an unknown ``name``.
    """
    p = _HERE / f"{name}.json"
    if not p.is_file():
        raise KeyError(f"unknown example {name!r}; available: {', '.join(available())}")
    return p


def copy(name: str, dest_dir: str | Path) -> Path:
    """Copy example ``name`` into ``dest_dir`` and return the copied path.

    The copy is what you run: its relative output folder then resolves into
    *your* directory, never into the install location. Refuses to overwrite
    an existing file (it may hold your edits).
    """
    src = path(name)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / src.name
    if target.exists():
        raise FileExistsError(f"{target} already exists (delete it first if you want a fresh copy)")
    shutil.copyfile(src, target)
    return target
