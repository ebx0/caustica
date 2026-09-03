"""The whole chain, for real — sonication -> T(r,t) -> CEM43 -> report.

Every earlier thermal test isolates one link: ``test_sensors.py`` checks that
``Q = 2 alpha I``, ``test_thermal.py`` checks the Pennes solve against closed
forms, ``test_thermal_cross.py`` checks it against an independent integrator,
and ``test_thermal_report.py`` checks the report against a synthetic result.
None of them would notice if the links did not FIT: a ``dx`` that disagrees
between the acoustic grid and the thermal medium, a record region that is not
the region the thermal medium was built on, a cooling phase that silently
starts its dose at zero.

So this file runs the real thing, small enough to be a unit test:

    westervelt solve (40x40x56, ~0.7 s)
        -> HeatingSource.from_result(..., harmonics=(1,))   [the explicit contract]
        -> PennesSolver: 20 s with Q on, then 40 s with Q off
        -> CEM43 accumulated ACROSS both phases (dose0 carried)
        -> ThermalResult.chain -> caustica.thermal.report

and asserts the two things a chain can get wrong without any single link
being wrong: the temperature must rise and then fall, and the dose must only
grow. Plus the report contract — stamp, thresholds, liability note.

:func:`evidence_run` is the same chain at a slightly larger scale, written to
``benchmarks/reports/thermal/<timestamp>/`` as the milestone's evidence. It is
not a test (it writes into the repo); run it deliberately::

    ./.venv/Scripts/python.exe -c "import sys; sys.path.insert(0, 'tests'); \
        import test_thermal_e2e as e; print(e.evidence_run())"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import caustica.solvers as solvers
from caustica import Grid, Medium, PMLSpec
from caustica.materials import Material, MaterialDB
from caustica.sensors import ALPHA_MODEL_F0, HeatingSource
from caustica.solvers import CWRunSpec
from caustica.solvers.base import interior_slices
from caustica.sources import bowl_cw_source
from caustica.thermal.dose import ITRUSST_CEM43_LIMITS, ITRUSST_DELTA_T_LIMIT_C
from caustica.thermal.pennes import ARTERIAL_TEMPERATURE_C, PennesSolver, ThermalResult
from caustica.thermal.properties import ThermalMedium
from caustica.thermal.report import (
    JSON_NAME,
    MD_NAME,
    MEDICAL_LIABILITY_NOTE,
    VERDICT_EXCEEDED,
    VERDICT_PASS,
    labels_from_db,
    timestamped_dir,
    write_thermal_report,
)

F0 = 1.0e6
C0 = 1540.0
DX = C0 / (F0 * 4.0)  # 4 points per wavelength -> 0.385 mm

#: A two-tissue head-like phantom: a skin layer the beam crosses, then brain.
#: Two tissues because the report's per-tissue table is only exercised when
#: more than one ITRUSST class is present.
SKIN = Material(
    name="Skin",
    alpha_np_m=15.0,
    rho=1109.0,
    c=1600.0,
    beta=4.0,
    thermal_conductivity=0.37,
    specific_heat=3390.0,
    perfusion_rate=0.002,
)
BRAIN = Material(
    name="Brain",
    alpha_np_m=8.0,
    rho=1046.0,
    c=1546.0,
    beta=4.5,
    thermal_conductivity=0.51,
    specific_heat=3630.0,
    perfusion_rate=0.008,
)
DB = MaterialDB(materials={1: SKIN, 2: BRAIN})


def sonicate(
    *,
    shape: tuple[int, int, int] = (40, 40, 56),
    amplitude_pa: float = 4.0e5,
    on_s: float = 20.0,
    off_s: float = 40.0,
    skin_vox: int = 10,
) -> dict:
    """Run the real chain once and hand back every intermediate.

    Returned as a dict rather than a tuple because the tests below assert on
    different links and a seven-tuple is unreadable at the call site.
    """
    grid = Grid(shape=shape, dx=DX, pml=PMLSpec(thickness=2.5e-3))
    ids = np.full(shape, 2, dtype=np.int32)
    ids[:, :, :skin_vox] = 1
    medium = Medium.from_id_map(ids, DB)
    source = bowl_cw_source(
        grid,
        f0=F0,
        amplitude=amplitude_pa,
        aperture_radius=4.0e-3,
        roc=9.0e-3,
        apex_vox=(shape[0] // 2, shape[1] // 2, 9),
    )
    # Record (and heat) the interior only: the PML band absorbs by design, so
    # its |P|^2 is a sponge artefact, not a field, and turning it into Q would
    # paint a ring of fictitious heating around the domain.
    region = interior_slices(shape, grid.pml_vox + 2)
    spec = CWRunSpec(min_settle_periods=6, max_settle_periods=20, n_record_periods=2)
    result = solvers.get("westervelt")().run(
        grid, medium, source, spec, backend="numpy", record_region=region, harmonics=(1, 2)
    )

    # The honesty contract, exercised for real: this result DOES carry a second
    # harmonic, so HeatingSource would refuse to guess its absorption. Asking
    # for harmonics=(1,) is the documented way to say "fundamental only" out
    # loud — and the report prints the resulting alpha_model.
    heat = HeatingSource.from_result(result, medium, grid.dx, harmonics=(1,))

    thermal_medium = ThermalMedium.from_id_map(ids[region], DB, grid.dx)
    solver = PennesSolver(backend="numpy")
    dt = 0.9 * solver.stable_dt(thermal_medium)
    t0 = np.full(thermal_medium.shape, ARTERIAL_TEMPERATURE_C, dtype=np.float32)

    hot = solver.solve(t0, heat, thermal_medium, dt=dt, n_steps=int(on_s / dt), dose=True)
    cool = solver.solve(
        hot.temperature,
        None,
        thermal_medium,
        dt=dt,
        n_steps=int(off_s / dt),
        dose=True,
        dose0=hot.dose_cem43,
    )
    return {
        "grid": grid,
        "medium": medium,
        "result": result,
        "heat": heat,
        "thermal_medium": thermal_medium,
        "hot": hot,
        "cool": cool,
        "full": ThermalResult.chain([hot, cool]),
        "dt": dt,
    }


@pytest.fixture(scope="module")
def chain():
    return sonicate()


def test_the_acoustic_field_becomes_heat_on_the_same_grid(chain):
    """Q is built from the recorded region, at the acoustic grid's own dx."""
    heat, tmed, result = chain["heat"], chain["thermal_medium"], chain["result"]
    assert heat.shape == tmed.shape, "Q and the thermal medium are on different grids"
    assert heat.dx == pytest.approx(tmed.dx)
    assert heat.harmonics == (1,)
    assert heat.alpha_model == ALPHA_MODEL_F0
    assert set(result.phasors) == {1, 2}, (
        "the second harmonic must exist for the opt-out to mean something"
    )
    assert heat.q_max > 0.0
    print(
        f"\nE2E acoustic: peak {result.amp.max() / 1e6:.4g} MPa over {result.steps_total} steps; "
        f"Q_max {heat.q_max:.4g} W/m^3, absorbed {heat.total_power_w * 1e3:.4g} mW"
    )


def test_the_temperature_rises_under_sonication_and_falls_when_it_stops(chain):
    """The shape of a sonication: up while Q is on, down when it is off."""
    hot, cool, full = chain["hot"], chain["cool"], chain["full"]
    baseline = ARTERIAL_TEMPERATURE_C
    assert hot.peak_temperature_c > baseline + 1.0, "20 s of HIFU did not warm anything"
    assert hot.temperature.max() > cool.temperature.max(), "the tissue did not cool"
    assert cool.temperature.max() > baseline, "40 s cannot take it all the way back to body T"
    assert hot.peak_temperature_c < 100.0, "a boiling focus is outside what Pennes describes"
    # The trap ThermalResult.chain closes: the cooling phase's own maximum is
    # the switch-off temperature, NOT the peak of the exposure.
    assert cool.peak_temperature_c == pytest.approx(hot.temperature.max(), rel=1e-6)
    assert full.peak_temperature_c == pytest.approx(hot.peak_temperature_c, rel=1e-6)
    print(
        f"E2E thermal: peak {full.peak_temperature_c:.4g} C (dT "
        f"{full.peak_temperature_c - baseline:.4g} K), end of cool-down "
        f"{cool.temperature.max():.4g} C"
    )


def test_the_dose_accumulates_across_both_phases_and_never_decreases(chain):
    """CEM43 is a history integral: cooling adds to it, it never gives back."""
    hot, cool, full = chain["hot"], chain["cool"], chain["full"]
    assert (cool.dose_cem43 >= hot.dose_cem43 - 1e-6).all(), "dose went backwards"
    assert cool.peak_dose_cem43 > hot.peak_dose_cem43, "the cool-down accrued no dose at all"
    assert full.peak_dose_cem43 == pytest.approx(cool.peak_dose_cem43)
    assert full.t_end_s == pytest.approx(hot.t_end_s + cool.t_end_s)
    print(
        f"E2E dose: {hot.peak_dose_cem43:.4g} CEM43 at switch-off -> "
        f"{full.peak_dose_cem43:.4g} after cool-down"
    )


def test_the_chain_writes_a_report_with_the_stamp_thresholds_and_liability_note(chain, tmp_path):
    """The deliverable: one folder a milestone can be cited against."""
    outdir = write_thermal_report(
        chain["full"],
        chain["thermal_medium"],
        tmp_path / "report",
        label="mini sonication (e2e test)",
        tissue_labels=labels_from_db(DB),
    )
    md = (outdir / MD_NAME).read_text(encoding="utf-8")
    payload = json.loads((outdir / JSON_NAME).read_text(encoding="utf-8"))

    assert payload["format"] == "caustica-thermal/1"
    for key in ("generated", "caustica", "git_commit", "host", "environment"):
        assert payload[key], f"the stamp is missing {key}"

    assert payload["medical_liability_note"] == MEDICAL_LIABILITY_NOTE
    assert MEDICAL_LIABILITY_NOTE in md

    names = {row["name"] for row in payload["thresholds"]}
    assert names == {f"CEM43 ({t})" for t in ITRUSST_CEM43_LIMITS} | {
        "temperature rise (non-thermal line)"
    }
    for row in payload["thresholds"]:
        assert row["verdict"] in (VERDICT_PASS, VERDICT_EXCEEDED)

    # Physically sane, and the numbers the report prints are the run's own.
    assert payload["temperature"]["peak_c"] == pytest.approx(chain["full"].peak_temperature_c)
    assert payload["dose"]["peak_cem43"] == pytest.approx(chain["full"].peak_dose_cem43)
    assert payload["temperature"]["peak_delta_t_c"] > ITRUSST_DELTA_T_LIMIT_C
    assert payload["verdict_delta_t"] == VERDICT_EXCEEDED, "an ablation crosses the 2 C line"
    assert {t["label"] for t in payload["tissues"]} == {"Skin", "Brain"}
    assert {t["itrusst_class"] for t in payload["tissues"]} == {"skin", "brain"}
    assert "Per tissue" in md and "Brain" in md


def test_chaining_refuses_a_cooling_phase_that_forgot_the_earlier_dose(chain):
    """``dose0`` omitted = the sonication's dose silently thrown away.

    The cooling phase still returns a perfectly plausible dose map — it is
    just the dose of the cool-down alone. Nothing downstream could tell, so
    the chain refuses it and says which argument was missed.
    """
    solver = PennesSolver(backend="numpy")
    hot = chain["hot"]
    orphan = solver.solve(
        hot.temperature,
        None,
        chain["thermal_medium"],
        dt=chain["dt"],
        n_steps=5,
        dose=True,  # ...but no dose0
    )
    with pytest.raises(ValueError, match="dose0"):
        ThermalResult.chain([hot, orphan])


# --------------------------------------------------------------------------
# Evidence (not collected as a test: it writes into the repo)
# --------------------------------------------------------------------------


def evidence_run(outdir: str | Path | None = None) -> Path:
    """The milestone's evidence: the same chain, one size up, on the CPU.

    Larger grid, longer sonication, same code path as the test above. Writes
    ``THERMAL.md`` + ``thermal.json`` under
    ``benchmarks/reports/thermal/<UTC timestamp>/`` and returns the folder.
    """
    chain = sonicate(shape=(56, 56, 80), amplitude_pa=4.0e5, on_s=30.0, off_s=60.0)
    result, heat = chain["result"], chain["heat"]
    outdir = timestamped_dir() if outdir is None else Path(outdir)
    return write_thermal_report(
        chain["full"],
        chain["thermal_medium"],
        outdir,
        label="M18 evidence — 1 MHz bowl into a skin/brain phantom, 30 s on + 60 s off",
        tissue_labels=labels_from_db(DB),
        notes=[
            f"acoustics: westervelt on {tuple(chain['grid'].shape)} at "
            f"{DX * 1e3:.4g} mm ({result.steps_total} steps, converged period "
            f"{result.converged_period}); peak |p| = {result.amp.max() / 1e6:.4g} MPa "
            f"in the recorded interior.",
            f"heating: {heat.alpha_model} at harmonics {heat.harmonics} (the second "
            f"harmonic WAS recorded and deliberately not heated — the M18 honesty "
            f"contract); Q_max = {heat.q_max:.4g} W/m^3, "
            f"{heat.total_power_w * 1e3:.4g} mW absorbed in the region.",
            "produced by tests/test_thermal_e2e.py::evidence_run — the same chain the "
            "e2e test runs, one size up.",
        ],
    )
