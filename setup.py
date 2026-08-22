"""Build hook only — all project metadata lives in ``pyproject.toml``.

This file exists for one reason: to run :func:`build_stamp.stamp` before
setuptools collects the package, so every wheel and sdist carries the commit
it was built from (fix A1 — a wheel install has no checkout, and Colab
installs from a wheel). Nothing here configures the project.
"""

from __future__ import annotations

import sys
from pathlib import Path

from setuptools import setup

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

try:
    from build_stamp import stamp
except ImportError:  # pragma: no cover - sdist without the build helper
    # The sdist already ships the generated module; a build that cannot find
    # this helper must not fail, it just does not re-stamp.
    pass
else:
    stamp(_ROOT)

setup()
