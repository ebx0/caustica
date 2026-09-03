"""The layering rule, enforced: the import direction is one-way.

Arrows point DOWN only: nothing under ``src/caustica`` may import the repo
side packages (``apps``, ``uwcem_phantoms``) or a GUI package. This test was
written BEFORE the extraction and stayed red until it landed — it is
the proof the split actually happened, and the tripwire against relapse.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "caustica"

FORBIDDEN = ("apps", "uwcem_phantoms", "caustica_gui")


def _imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def test_caustica_never_imports_upward():
    offenders: dict[str, list[str]] = {}
    for py in sorted(SRC.rglob("*.py")):
        bad = [
            name
            for name in _imports_of(py)
            if any(name == f or name.startswith(f + ".") for f in FORBIDDEN)
        ]
        if bad:
            offenders[str(py.relative_to(SRC))] = bad
    assert not offenders, f"caustica imports upward (layering violation): {offenders}"


def test_no_uwcem_reference_survives_in_source_text():
    """The gate verbatim: `grep -ri uwcem src/` is EMPTY — not just
    imports; comments and docstrings must not smuggle the coupling back."""
    hits: dict[str, int] = {}
    for py in sorted(SRC.rglob("*.py")):
        text = py.read_text(encoding="utf-8").lower()
        n = text.count("uwcem")
        if n:
            hits[str(py.relative_to(SRC))] = n
    assert not hits, f"'uwcem' still referenced under src/caustica: {hits}"
