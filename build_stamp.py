"""Build-time provenance: freeze the commit into the package (fix A1, 2026-08-22).

The first real Colab session exposed a traceability hole. Colab installs
caustica from a *wheel*, so there is no checkout for the runner to ask, and
every ``run_meta.json`` produced there was stamped ``"git_commit": "unknown"``
— on the one machine whose results nobody can reproduce by looking over their
own shoulder.

The fix is to write ``src/caustica/_build_info.py`` while the distribution is
being built, from the source tree's own git HEAD. ``setup.py`` calls
:func:`stamp` and setuptools does the rest, so this costs **no new
dependency**: ``setuptools_scm`` would do the same and also take over version
derivation, which this project deliberately keeps hand-written in
``caustica/__init__.py``.

Two rules make the generated module safe:

* **git wins, but only for THIS tree.** ``git rev-parse`` searches upward, so
  a source tree unpacked inside somebody else's repository would happily
  report that repository's commit. The toplevel is verified to be the tree we
  are stamping before the hash is believed.
* **A build that cannot see git keeps what is already there.** A wheel built
  from an sdist has no ``.git`` — but the sdist already ships the generated
  module, and overwriting it with ``"unknown"`` would throw away the very
  fact this file exists to carry.

Not packaged: this module runs at build time only. The runtime half lives in
:func:`caustica.env.git_commit`, which prefers a live checkout and falls back
to the module written here. The two carry the same "is this really our
repository" check, and ``tests/test_packaging.py`` pins them against each
other so the copies cannot drift.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

#: Where the generated module lands, relative to the project root.
BUILD_INFO_RELPATH = Path("src") / "caustica" / "_build_info.py"

_HEADER = "# Generated at build time by build_stamp.py. Do not edit; do not commit.\n"


def head_commit(root: Path) -> str | None:
    """``root``'s git HEAD, or None when ``root`` is not a git work tree.

    None — not ``"unknown"`` — because the caller must be able to tell "no
    git here" (keep the existing stamp) from "git said unknown", which it
    never does.
    """
    root = Path(root).resolve()
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if top.returncode != 0:
            return None
        # `git rev-parse` walks UP: without this, a tarball unpacked inside an
        # unrelated repository stamps that repository's commit into our wheel.
        if Path(top.stdout.strip()).resolve() != root:
            return None
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        commit = out.stdout.strip()
        return commit if out.returncode == 0 and commit else None
    except Exception:  # git missing, or a hung index lock — never fatal
        return None


def package_version(root: Path) -> str:
    """``__version__`` read out of ``caustica/__init__.py`` WITHOUT importing it.

    Importing the package at build time would need numpy, which build
    isolation does not provide.
    """
    text = (Path(root) / "src" / "caustica" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M)
    return match.group(1) if match else "unknown"


def render(version: str, commit: str, built_at: str) -> str:
    return f'{_HEADER}VERSION = "{version}"\nCOMMIT = "{commit}"\nBUILT_AT = "{built_at}"\n'


def stamp(root: Path | str | None = None) -> Path | None:
    """Write ``src/caustica/_build_info.py`` for the tree at ``root``.

    Returns the path written, or None when git was unavailable AND a stamp
    was already present (the sdist case — see the module docstring).
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    target = root / BUILD_INFO_RELPATH
    commit = head_commit(root)
    if commit is None:
        if target.is_file():
            return None  # keep the stamp the sdist carried
        commit = "unknown"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render(
            package_version(root),
            commit,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
        encoding="utf-8",
    )
    return target


if __name__ == "__main__":  # pragma: no cover - manual use
    print(stamp())
