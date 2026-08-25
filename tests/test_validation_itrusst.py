"""ITRUSST PH1 water benchmarks: the definitions, and the gate over them.

The runs themselves are nine-megavoxel and belong in a report, not in the
suite. What belongs here is everything that decides whether a run MEANS
anything: that the numbers transcribed from the paper are the paper's, that
the reference is the exact one, and that the gate fails when it should.
"""

from __future__ import annotations

import numpy as np
import pytest

from caustica.validation import itrusst as it


def test_the_transcribed_definitions_are_the_papers():
    """Every number here came off a page; a typo is a silent wrong benchmark."""
    assert it.WATER_C == 1500.0 and it.WATER_RHO == 1000.0
    assert it.F0 == 500e3
    assert it.U0 == 0.04
    # "surface velocity 0.04 m/s" and "60 kPa" are the same statement
    assert it.DRIVE_PA == pytest.approx(60e3)
    # BM2: "1 dB/cm at 500 kHz"
    assert it.BM2_ALPHA_NP_M * (20.0 / np.log(10.0)) / 100.0 == pytest.approx(1.0)
    assert it.BM1.alpha_np_m == 0.0
    # SC1 "64 mm radius of curvature and a 64 mm aperture diameter" -> f/1
    assert it.SC1.roc_mm == 64.0 and it.SC1.diameter_mm == 64.0
    assert it.SC1.roc_mm / it.SC1.diameter_mm == 1.0
    # SC2 "diameter of 20 mm"
    assert it.SC2.diameter_mm == 20.0 and it.SC2.roc_mm is None
    assert (it.DOMAIN_AXIAL_MM, it.DOMAIN_LATERAL_MM, it.DX_MM) == (120.0, 70.0, 0.5)


def test_the_comparison_domain_is_the_papers_241_by_141():
    """0.5 mm over 120 x 70 mm, with the sponge and the halo outside it."""
    grid, apex, (lat_sl, ax_sl), (n_lat, n_ax) = it._geometry(0.5e-3, 8.0, 4)

    assert (n_lat, n_ax) == (141, 241)
    assert lat_sl.stop - lat_sl.start == 141
    assert ax_sl.stop - ax_sl.start == 241
    # the source sits on the first axial plane of the window, laterally centred
    assert apex[2] == ax_sl.start
    assert apex[0] == apex[1] == grid.shape[0] // 2
    # ...and the whole window clears the sponge on every face
    assert ax_sl.start >= grid.pml_vox and ax_sl.stop <= grid.shape[2] - grid.pml_vox
    assert lat_sl.start >= grid.pml_vox and lat_sl.stop <= grid.shape[0] - grid.pml_vox


def test_the_reference_integrates_absorption_rather_than_applying_it_after():
    """BM2's absorption enters the integral, and the difference is measurable.

    Every point of the transducer is a different distance from a field point
    — on the axis of a 20 mm piston, between ``z`` and ``sqrt(z^2 + a^2)`` —
    so the lossy field is attenuated a little MORE than ``exp(-alpha z)``
    applied to the lossless one, by an amount that closes as the paths
    converge. Measured here: 1.3 % at 20 mm falling to 0.4 % at 80 mm.

    That gap is the whole reason the wavenumber carries the absorption
    instead of a decay being multiplied on at the end.
    """
    z = np.array([0.02, 0.04, 0.06, 0.08])
    pts = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    lossless = np.abs(it.reference_field(it.SC2, it.BM1, pts))
    lossy = np.abs(it.reference_field(it.SC2, it.BM2, pts))
    naive = lossless * np.exp(-it.BM2_ALPHA_NP_M * z)

    assert np.all(lossy < naive), "integrated absorption cannot be the weaker one"
    gap = 1.0 - lossy / naive
    assert np.all(np.diff(gap) < 0), "the gap must close as the paths converge"
    assert gap[0] > 3 * gap[-1] > 0
    # ...and it IS this absorption, not some other number
    assert gap.max() < 0.02


def test_the_gate_reads_the_spread_the_paper_reported():
    """The limits are published facts, not a tolerance chosen here."""
    assert it.REPORTED_SPREAD["SC1"]["all_under_pct"] == 10.0
    assert it.REPORTED_SPREAD["SC2"]["all_under_pct"] == 15.0
    gates = {g.id: g for g in it.evaluate({})}
    assert set(gates) == {"M21.PH1-SC1", "M21.PH1-SC2"}
    assert "10 %" in gates["M21.PH1-SC1"].criterion
    assert "15 %" in gates["M21.PH1-SC2"].criterion


def measured(l_inf: float, peak_err_mm: float = 0.0) -> dict:
    return {
        "l_inf_pct": l_inf,
        "peak_axial_mm": {"simulated": 62.0 + peak_err_mm, "reference": 62.0},
    }


def test_a_run_inside_the_reported_spread_passes():
    """What this library actually measured: 3.4-5.0 %, peak within a voxel."""
    cases = {
        "BM1-SC1": measured(3.41),
        "BM2-SC1": measured(3.79, 0.5),
        "BM1-SC2": measured(3.82),
        "BM2-SC2": measured(5.03),
    }
    assert all(g.verdict == "PASS" for g in it.evaluate(cases))


@pytest.mark.parametrize(
    "case,field,value",
    [
        ("BM1-SC1", "l_inf", 12.0),  # over the bowl's 10 %
        ("BM2-SC2", "l_inf", 18.0),  # over the piston's 15 %
        ("BM1-SC2", "peak", 1.5),  # three voxels off
    ],
)
def test_a_run_outside_it_fails_the_right_gate(case, field, value):
    cases = {k: measured(3.5) for k in ("BM1-SC1", "BM2-SC1", "BM1-SC2", "BM2-SC2")}
    cases[case] = measured(value) if field == "l_inf" else measured(3.5, value)
    gates = {g.id: g for g in it.evaluate(cases)}
    failed = gates[f"M21.PH1-{case.split('-')[1]}"]
    other = gates[f"M21.PH1-{'SC2' if case.endswith('SC1') else 'SC1'}"]

    assert failed.verdict == "FAIL"
    assert other.verdict == "PASS", "one bad case took an unrelated gate down"


def test_a_case_that_never_ran_leaves_its_gate_open():
    """SKIP is never PASS — the rule the rest of the validation package keeps."""
    cases = {k: measured(3.5) for k in ("BM1-SC1", "BM2-SC1", "BM1-SC2")}
    cases["BM2-SC2"] = {"error": "RuntimeError: no GPU"}
    gates = {g.id: g for g in it.evaluate(cases)}

    assert gates["M21.PH1-SC2"].verdict == "INCOMPLETE"
    assert gates["M21.PH1-SC1"].verdict == "PASS"
