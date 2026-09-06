"""The claims in README.md and CITATION.cff still point at something real.

Prose goes stale silently. Two failure modes have already happened in this
repository: a README that still said the GPU backend was unverified months
after an A100 session graded it, and citations of benchmark reports whose
directories were renamed by the next run. Both are cheap to pin.

This file checks only what a machine can check: that every benchmark report
the README cites exists, that every ``caustica.validation`` subcommand it
tells the reader to run is registered, and that CITATION.cff advertises the
version the package actually carries. Whether a number in the prose matches
the number in the report it cites stays a human's job.
"""

import re
from pathlib import Path

import pytest

import caustica

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
CITATION = REPO / "CITATION.cff"

#: A subcommand as the README writes it. The class carries digits so a name
#: like ``gpu-gates2`` is not silently truncated to a registered prefix, and
#: the lookahead keeps a flag such as ``--help`` from reading as one.
SUBCOMMAND = re.compile(r"python -m caustica\.validation (?!-)([a-z0-9-]+)")

pytestmark = pytest.mark.skipif(
    not README.exists(), reason="not running from a checkout: README.md is not packaged here"
)


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def test_every_benchmark_report_the_readme_cites_exists():
    """A citation of a report path is a promise the path resolves."""
    cited = sorted(set(re.findall(r"`(benchmarks/reports/[^`]+)`", _readme())))
    assert cited, "the README cites no benchmark report; the validation section lost its evidence"
    missing = [p for p in cited if not (REPO / p).exists()]
    assert not missing, f"README cites reports that do not exist: {missing}"


def test_every_validation_subcommand_the_readme_names_is_registered():
    """``python -m caustica.validation <cmd>`` in the prose has to be runnable."""
    from caustica.validation.__main__ import build_parser

    named = sorted(set(re.findall(SUBCOMMAND, _readme())))
    assert named, "the README names no validation subcommand"
    actions = [a for a in build_parser()._actions if a.choices and hasattr(a.choices, "keys")]
    registered = set()
    for action in actions:
        registered.update(action.choices.keys())
    unknown = [c for c in named if c not in registered]
    assert not unknown, f"README names unregistered subcommands: {unknown} (have {registered})"


def test_citation_advertises_the_installed_version():
    """A CITATION.cff pinned to an older version cites a release nobody has."""
    if not CITATION.exists():  # pragma: no cover - checkout-only file
        pytest.skip("CITATION.cff is not present")
    text = CITATION.read_text(encoding="utf-8")
    match = re.search(r"^version:\s*(\S+)\s*$", text, re.MULTILINE)
    assert match, "CITATION.cff carries no version field"
    assert match.group(1) == caustica.__version__, (
        f"CITATION.cff says {match.group(1)}, the package says {caustica.__version__}"
    )
