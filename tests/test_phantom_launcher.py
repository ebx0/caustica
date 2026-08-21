"""The launcher's one falsifiable claim: the CLI line it prints.

Everything else in ``apps/phantom_launcher.py`` is prompts and formatting, which
a test cannot usefully pin down. But the wizard ends by printing

    same thing from the command line:
      python -m uwcem_phantoms build 012304 --f0 1 --dx 0.32 --pval

and a user will paste that into a script and expect the same phantom. If a flag
is ever added to :class:`PhantomSpec` without being reflected there, that line
starts quietly describing a DIFFERENT build. So: generate it, parse it back
through the real argument parser, and require the specs to match exactly.
"""

from __future__ import annotations

import shlex

import pytest
from apps.phantom_launcher import CLI_CANNOT_EXPRESS, cli_equivalent, human_bytes, ppw_line
from uwcem_phantoms.cli import build_parser, spec_from_args
from uwcem_phantoms.spec import PhantomSpec

SPECS = [
    pytest.param(PhantomSpec(phantom_id="012304"), id="defaults"),
    pytest.param(
        PhantomSpec(
            phantom_id="010204",
            f0_mhz=1.5,
            simplify={"tissue_model": "simple", "drop_muscle": True, "remove_islands_vox": 40},
            resolution={"dx_mm": 0.24, "method": "nearest"},
            crop={"mode": "tissue", "margin_mm": 8.0},
            domain={"standoff_mm": 15.0, "backing_mm": 4.0, "lateral_margin_mm": 2.0},
            heterogeneity={"use_pval": False, "noise_pct": 3.0, "correlation_mm": 0.6, "seed": 7},
            name="round-trip",
        ),
        id="everything-set",
    ),
    pytest.param(
        PhantomSpec(
            phantom_id="062204",
            simplify={"drop_skin": True, "fill_holes": True, "smooth_iterations": 2},
            crop={"mode": "none"},
            domain={"fft_friendly": False, "max_voxels": 20_000_000},
        ),
        id="no-fft-pad",
    ),
]


def _reparse(line: str) -> PhantomSpec:
    tokens = shlex.split(line)
    assert tokens[:4] == ["python", "-m", "uwcem_phantoms", "build"]
    return spec_from_args(build_parser().parse_args(tokens[3:]))


@pytest.mark.parametrize("spec", SPECS)
def test_printed_cli_line_reproduces_the_spec(spec):
    line = cli_equivalent(spec, None, "auto")
    assert line is not None
    assert _reparse(line).model_dump() == spec.model_dump()


def test_unrepresentable_options_print_no_command():
    """Rather than a command that builds something else, print nothing."""
    for field in CLI_CANNOT_EXPRESS:
        value = True if field.endswith("only") else 2
        spec = PhantomSpec(phantom_id="012304", simplify={field: value})
        assert cli_equivalent(spec, None, "auto") is None


def test_store_properties_survives_the_round_trip():
    spec = PhantomSpec(phantom_id="012304")
    line = cli_equivalent(spec, None, "never")
    assert "--store-properties never" in line
    assert build_parser().parse_args(shlex.split(line)[3:]).store_properties == "never"


@pytest.mark.parametrize(
    ("dx_mm", "expect"),
    [(0.8, "UNUSABLE"), (0.5, "marginal"), (0.2, "ok")],
)
def test_ppw_line_matches_the_builders_thresholds(dx_mm, expect):
    # 1425 m/s is the slowest LOW endpoint in the default table; the builder
    # warns below 2 ppw (unusable) and below 4 (marginal) against the same one.
    assert expect in ppw_line(1425.0, 1.0, dx_mm)


def test_human_bytes_rounds_to_a_readable_unit():
    assert human_bytes(512) == "512 B"
    assert human_bytes(1536) == "1.5 kB"
    assert human_bytes(3 * 1024**3) == "3.0 GB"
