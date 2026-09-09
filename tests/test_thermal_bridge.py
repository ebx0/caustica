"""One table, one label volume, both media: the acoustic-to-thermal bridge.

``tests/test_thermal.py`` checks the Pennes solver against closed forms and
``tests/test_thermal_e2e.py`` runs a whole sonication. Neither asks the
question this file asks, which is where the numbers of a thermal run come
from. The answer a planning study needs is "the shipped table, with a
citation per number", not "whatever the script typed above the solve", so
what is pinned here is the seam:

* :func:`caustica.materials.tissue_db` builds one ``MaterialDB`` out of
  :data:`caustica.materials.TISSUE_LIBRARY`, and both media read it;
* :meth:`ThermalMedium.from_medium` and :meth:`ThermalMedium.from_labels`
  reach the same volumes from the same labels;
* every material this library ships answers a thermal solve, with the IT'IS
  row it came from named in its ``source``;
* a dense medium (no ``id_map``) is refused with the constructor that fixes
  it, because there is nothing to look a tissue property up by.
"""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from caustica.core.grid import Grid
from caustica.core.pml import PMLSpec
from caustica.materials import (
    ASSUMED_ABSORBED_FRACTION_SOURCE,
    DEFAULT_ABSORBED_FRACTION,
    ITIS_BLOOD_DENSITY,
    ITIS_BLOOD_SPECIFIC_HEAT,
    ITIS_DB,
    TISSUE_LIBRARY,
    TISSUE_LIBRARY_VERSION,
    AcousticTissue,
    Material,
    MaterialDB,
    breast_default,
    breast_default_power_law,
    db_cm_to_np_m,
    tissue_db,
    tissue_library_digest,
    tissue_library_stamp,
    water,
)
from caustica.medium import Medium
from caustica.sensors import HeatingSource
from caustica.solvers import CWRunSpec, get
from caustica.sources import CWSource
from caustica.thermal.pennes import (
    ARTERIAL_TEMPERATURE_C,
    BLOOD_DENSITY,
    BLOOD_SPECIFIC_HEAT,
    PennesSolver,
)
from caustica.thermal.properties import ThermalMedium, ThermalPropertyError

F0 = 5.0e5
#: The layout every test below paints: skin at the entry face, liver behind.
NAMES = {1: "skin", 2: "liver"}
SKIN_VOX = 4


def _labels(shape: tuple[int, int, int]) -> np.ndarray:
    ids = np.full(shape, 2, dtype=np.int32)
    ids[:, :, :SKIN_VOX] = 1
    return ids


# ------------------------------------------------------- one table, two media


def test_one_table_and_one_label_volume_build_both_media():
    """Acceptance 1: nothing in this chain is typed by hand.

    The label volume and ``tissue_db`` are the only inputs; the acoustic four
    and the thermal three both come out of the same library rows, and the two
    media agree voxel by voxel about where the tissues are.
    """
    labels = _labels((10, 10, 14))
    db = tissue_db(NAMES, f0_hz=F0)
    medium = Medium.from_id_map(labels, db)
    thermal = ThermalMedium.from_medium(medium, db, dx=0.5e-3)

    skin, liver = TISSUE_LIBRARY["skin"], TISSUE_LIBRARY["liver"]
    # Acoustic side: alpha is the library's own power law, evaluated at f0.
    assert float(medium.alpha[0, 0, 0]) == pytest.approx(
        db_cm_to_np_m(2.85 * (F0 / 1e6) ** 1.0), rel=1e-6
    )
    assert float(medium.alpha[0, 0, -1]) == pytest.approx(6.915 * (F0 / 1e6) ** 1.0, rel=1e-6)
    # Thermal side: the IT'IS values, in the same voxels.
    assert float(thermal.k[0, 0, 0]) == pytest.approx(skin.thermal_conductivity, rel=1e-6)
    assert float(thermal.k[0, 0, -1]) == pytest.approx(liver.thermal_conductivity, rel=1e-6)
    assert float(thermal.specific_heat[0, 0, -1]) == pytest.approx(liver.specific_heat, rel=1e-6)
    assert float(thermal.perfusion[0, 0, -1]) == pytest.approx(liver.perfusion_rate, rel=1e-6)
    # The layout cannot drift between the two: same rho, same id map.
    assert np.array_equal(thermal.rho, medium.rho)
    assert np.array_equal(thermal.id_map, labels)


def test_from_labels_reaches_the_same_volumes_without_a_medium():
    """A thermal-only study never builds an acoustic medium; it still gets the
    same numbers out of the same library keys."""
    labels = _labels((6, 6, 8))
    db = tissue_db(NAMES, f0_hz=F0)
    by_medium = ThermalMedium.from_medium(Medium.from_id_map(labels, db), db, dx=0.4e-3)
    by_labels = ThermalMedium.from_labels(labels, NAMES, dx=0.4e-3)
    for name in ("k", "rho", "specific_heat", "perfusion"):
        assert np.array_equal(getattr(by_labels, name), getattr(by_medium, name))
    assert by_labels.dx == by_medium.dx


def test_the_bridge_carries_a_real_sonication_from_labels_to_a_temperature():
    """The whole chain on the shipped table: labels -> Medium -> Q -> Pennes.

    Two hand-checks pin the physics across the join, the same two the
    hand-typed end-to-end test in ``test_thermal.py`` makes: Q at the focus is
    2 alpha I of the recorded phasor, and the first step's rise is Q dt /
    (rho C). Here alpha, rho and C are the library's, not the test's.
    """
    grid = Grid(shape=(24, 24, 24), dx=0.5e-3, pml=PMLSpec(thickness=2.0e-3))
    labels = _labels(grid.shape)
    db = tissue_db(NAMES, f0_hz=F0)
    medium = Medium.from_id_map(labels, db)
    idx = np.array([[i, j, 5] for i in range(5, 19) for j in range(5, 19)], dtype=np.int32)
    src = CWSource(
        indices=idx, phases=np.zeros(len(idx), np.float32), amplitude=5e5, f0=F0, ramp_periods=2.0
    )
    # westervelt, not linear: the library rows carry a real B/A, and zeroing it
    # to reach the cheaper solver would be exactly the hand-edit this file
    # exists to remove.
    res = get("westervelt")().run(
        grid, medium, src, CWRunSpec(min_settle_periods=4, max_settle_periods=16), backend="numpy"
    )
    heat = HeatingSource.from_result(res, medium, grid.dx)

    # Q = 2 alpha I at the hottest voxel, with that voxel's own tissue.
    hot_idx = np.unravel_index(int(np.argmax(heat.q)), heat.q.shape)
    tissue = db[int(labels[hot_idx])]
    peak_p = float(np.abs(res.phasor[hot_idx]))
    intensity = peak_p**2 / (2 * tissue.rho * tissue.c)
    assert float(heat.q[hot_idx]) == pytest.approx(2 * tissue.alpha_np_m * intensity, rel=1e-4)

    thermal = ThermalMedium.from_labels(labels, NAMES, grid.dx)
    solver = PennesSolver(backend="numpy")
    dt = 0.02
    assert dt < solver.stable_dt(thermal)
    t_body = np.full(thermal.shape, ARTERIAL_TEMPERATURE_C, np.float32)
    one_step = solver.solve(t_body, heat, thermal, dt, 1)
    rho_c = float(tissue.rho * tissue.specific_heat)
    assert float((one_step.temperature - ARTERIAL_TEMPERATURE_C)[hot_idx]) == pytest.approx(
        float(heat.q[hot_idx]) * dt / rho_c, rel=0.01
    )
    hot = solver.solve(t_body, heat, thermal, dt, int(round(5.0 / dt)), dose=True)
    assert hot.peak_temperature_c > ARTERIAL_TEMPERATURE_C + 1.0
    print(
        f"\nbridge: Q_max {heat.q_max:.4g} W/m^3 in {tissue.name!r}, "
        f"peak T {hot.peak_temperature_c:.4g} C after 5 s"
    )


# ------------------------------------------------------------------- refusals


def test_a_dense_medium_is_refused_with_the_constructor_that_fixes_it():
    """Acceptance 2: dense properties carry no tissue identity, so there is
    nothing to look k, C and w_b up by. The refusal names the four volumes the
    caller has to supply instead."""
    shape = (4, 4, 4)
    dense = Medium(
        alpha=np.zeros(shape, np.float32),
        rho=np.full(shape, 1000.0, np.float32),
        c=np.full(shape, 1500.0, np.float32),
        beta=np.zeros(shape, np.float32),
    )
    db = tissue_db(NAMES, f0_hz=F0)
    with pytest.raises(ValueError) as exc:
        ThermalMedium.from_medium(dense, db, 1e-3)
    msg = str(exc.value)
    assert "no id_map" in msg
    assert "per-voxel thermal volumes" in msg
    assert "ThermalMedium(k=" in msg and "perfusion=" in msg
    assert "from_labels" in msg and "homogeneous" in msg
    # And the fix in the message is a real one.
    fixed = ThermalMedium(
        k=np.full(shape, 0.5, np.float32),
        rho=dense.rho,
        specific_heat=np.full(shape, 3600.0, np.float32),
        perfusion=np.zeros(shape, np.float32),
        dx=1e-3,
    )
    assert fixed.shape == shape


def test_from_labels_refuses_an_unnamed_label_and_an_unknown_tissue():
    # The phrase is from_labels' OWN message: the id list alone is also in
    # from_id_map's generic refusal downstream, so matching on it would grade
    # the wrong defence and pass with this check deleted.
    labels = _labels((4, 4, 6))
    with pytest.raises(ValueError, match=r"ids \[2\].*crop the volume"):
        ThermalMedium.from_labels(labels, {1: "skin"}, 1e-3)
    with pytest.raises(KeyError, match="cortical_bone"):
        ThermalMedium.from_labels(labels, {1: "skin", 2: "cortical_bone"}, 1e-3)
    with pytest.raises(TypeError, match="integer-typed"):
        ThermalMedium.from_labels(labels.astype(np.float32), NAMES, 1e-3)


def test_the_two_media_can_disagree_when_from_labels_is_given_another_which():
    """What ``from_labels`` cannot check, stated as a test rather than as a
    docstring promise.

    ``which`` picks an end of each library row's reported spread. Build the
    acoustic medium at one end and the thermal one at the default and the two
    solves run on different tissue with nothing raised, because this
    constructor never sees the medium. ``from_medium`` is the checked route
    and refuses the same pair.
    """
    labels = _labels((4, 4, 6))
    db_lo = tissue_db(NAMES, which="lo", f0_hz=F0)
    medium = Medium.from_id_map(labels, db_lo)
    loose = ThermalMedium.from_labels(labels, NAMES, 1e-3)  # which="mid"
    assert float(medium.rho[0, 0, -1]) == pytest.approx(1050.0)
    assert float(loose.rho[0, 0, -1]) == pytest.approx(1104.0)
    assert not np.allclose(loose.rho, medium.rho)

    with pytest.raises(ValueError, match="disagrees with the acoustic"):
        ThermalMedium.from_medium(medium, tissue_db(NAMES, which="mid", f0_hz=F0), 1e-3)
    # Same which on both sides: the checked route accepts it.
    matched = ThermalMedium.from_medium(medium, db_lo, 1e-3)
    assert np.array_equal(matched.rho, medium.rho)


def test_one_hand_typed_row_beside_the_library_is_still_refused():
    """A table half from the library and half from a script is the realistic
    case, and the missing half has to be named rather than defaulted: the
    library rows would make the refusal look unnecessary until a dose came out
    of a tissue with an invented conductivity."""
    db = MaterialDB(
        materials={
            1: TISSUE_LIBRARY["skin"].to_material(F0),
            2: Material(name="Tumour", alpha_np_m=12.0, rho=1050.0, c=1560.0, beta=4.5),
        }
    )
    labels = _labels((4, 4, 6))
    with pytest.raises(ThermalPropertyError) as exc:
        ThermalMedium.from_id_map(labels, db, 1e-3)
    msg = str(exc.value)
    assert "'Tumour'" in msg and "id 2" in msg
    assert "thermal_conductivity" in msg


def test_tissue_db_names_the_keys_it_knows_when_one_is_wrong():
    with pytest.raises(KeyError) as exc:
        tissue_db({0: "water_37c", 1: "bone"})
    msg = str(exc.value)
    assert "bone" in msg and "liver" in msg  # what was asked, and what exists
    assert "Refusing to guess" in msg


# ----------------------------------------------- every shipped material, pinned

#: Every table this library ships, by the name a user would call.
SHIPPED = {
    "water()": MaterialDB(materials={0: water()}),
    "breast_default()": breast_default(),
    "breast_default_power_law()": breast_default_power_law(),
    "tissue_db(whole library)": tissue_db(dict(enumerate(TISSUE_LIBRARY))),
}


@pytest.mark.parametrize("label", sorted(SHIPPED))
def test_every_shipped_material_answers_a_thermal_solve_and_says_where_from(label):
    """Acceptance 3: thermal fields, the absorbed fraction and a citation on
    every row of every shipped table, and the whole table builds a
    ``ThermalMedium`` rather than being refused halfway through."""
    db = SHIPPED[label]
    for tissue_id in db.ids:
        m = db[tissue_id]
        who = f"{label} id {tissue_id} ({m.name!r})"
        assert m.thermal_conductivity is not None, who
        assert m.specific_heat is not None, who
        assert m.perfusion_rate is not None, who
        assert m.absorbed_fraction is not None, who
        assert ITIS_DB in m.source, who
        assert "absorbed fraction:" in m.source, who
    ids = np.array(sorted(db.ids), dtype=np.int32).reshape(-1, 1, 1) * np.ones(
        (1, 2, 2), dtype=np.int32
    )
    tm = ThermalMedium.from_id_map(ids, db, dx=1e-3)
    assert float(tm.k.min()) > 0.0
    assert float(tm.specific_heat.min()) > 0.0


def test_the_notebook_acoustic_numbers_did_not_move_when_the_thermal_half_landed():
    """The v1 contract: adding thermal fields to ``breast_default`` changes
    nothing a solver reads."""
    db = breast_default()
    assert [db[i].alpha_np_m for i in db.ids] == [0.1, 15.0, 6.0, 10.0, 0.1]
    assert [db[i].rho for i in db.ids] == [1000.0, 1109.0, 932.0, 1050.0, 1000.0]
    assert [db[i].c for i in db.ids] == [1500.0, 1600.0, 1450.0, 1580.0, 1500.0]
    assert [db[i].beta for i in db.ids] == [3.5, 4.0, 4.5, 4.5, 3.5]
    # The thermal half is borrowed from the library row each id names.
    assert db[1].thermal_conductivity == TISSUE_LIBRARY["skin"].thermal_conductivity
    assert db[2].specific_heat == TISSUE_LIBRARY["fat"].specific_heat
    assert db[0].perfusion_rate == 0.0 and db[4].perfusion_rate == 0.0
    assert "notebook TISSUE_PROPS v6-v12, verbatim" in db[3].source


# ------------------------------------------------------------ absorbed fraction


def test_the_absorbed_fraction_is_the_part_of_alpha_that_heats():
    """``alpha_absorption = absorbed_fraction * alpha``, and an undeclared
    material assumes 1.0 (all attenuation heats), which over-states Q."""
    m = Material(alpha0_db_cm_mhz_y=0.5, y=1.1, rho=911.0, c=1440.0, beta=6.0)
    assert m.absorbed_fraction is None
    assert m.heating_fraction == DEFAULT_ABSORBED_FRACTION == 1.0
    assert m.alpha_absorption_np_m_at(2e6) == m.alpha_np_m_at(2e6)

    scattering = m.model_copy(update={"absorbed_fraction": 0.8})
    assert scattering.alpha_absorption_np_m_at(2e6) == pytest.approx(
        0.8 * m.alpha_np_m_at(2e6), rel=1e-12
    )
    # It survives the bake to the legacy form and one JSON round trip.
    baked = scattering.at_frequency(2e6)
    assert baked.absorbed_fraction == 0.8
    assert baked.alpha_absorption_np_m_at(2e6) == pytest.approx(
        scattering.alpha_absorption_np_m_at(2e6), rel=1e-12
    )
    db = MaterialDB(materials={1: scattering})
    assert MaterialDB.model_validate_json(db.model_dump_json())[1].absorbed_fraction == 0.8


def test_an_impossible_absorbed_fraction_is_refused():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="absorbed_fraction"):
            Material(alpha_np_m=1.0, rho=1000.0, c=1500.0, beta=0.0, absorbed_fraction=bad)


def test_the_shipped_fractions_say_whether_they_are_measured_or_assumed():
    """The provenance point of the field: water's 1.0 is a fact about a liquid
    with no scatterers, and every soft tissue's 1.0 is an assumption that says
    so, because no per-tissue ratio is quoted by the sources these rows come
    from."""
    assert TISSUE_LIBRARY["water_37c"].absorbed_fraction == 1.0
    assert "by construction" in TISSUE_LIBRARY["water_37c"].absorption_source
    for key, t in TISSUE_LIBRARY.items():
        assert t.absorbed_fraction == 1.0, key
        if key == "water_37c":
            continue
        assert t.absorption_source == ASSUMED_ABSORBED_FRACTION_SOURCE, key
        assert "assumption, not a measurement" in t.absorption_source
        assert "upper bound" in t.to_power_law_material().source


def test_a_tissue_that_declares_no_absorbed_fraction_stays_undeclared():
    """``None`` has to survive the trip to the ``Material``.

    A row that says nothing about scattering must not arrive looking like a
    row that declared 1.0: ``heating_fraction`` answers what a heating
    calculation will use either way, and ``absorbed_fraction`` is the only
    thing that can tell an assumed 1.0 from a measured one.
    """
    silent = replace(
        TISSUE_LIBRARY["liver"], absorbed_fraction=None, absorption_source="", name="Unstated"
    )
    m = silent.to_power_law_material()
    assert m.absorbed_fraction is None
    assert m.heating_fraction == DEFAULT_ABSORBED_FRACTION
    assert "absorbed fraction:" not in m.source
    # And the shipped rows are declared, which is what makes the difference
    # readable at all.
    assert TISSUE_LIBRARY["liver"].to_power_law_material().absorbed_fraction == 1.0


def test_water_cites_the_arguments_it_was_called_with():
    """``water()`` takes five arguments and four shipped call sites pass their
    own, so a source string quoting the defaults would be a false citation on
    exactly the materials whose provenance is load bearing (the ITRUSST
    validation medium passes alpha 0.217 Np/m, the report scripts beta 3.5)."""
    default = water().source
    assert "alpha 0 Np/m" in default and "rho 1000 kg/m^3" in default
    assert "c 1500 m/s" in default and "beta 0)" in default

    lossy = water(alpha_np_m=0.217, beta=3.5, rho=1050.0, c=1480.0)
    assert lossy.alpha_np_m == 0.217 and lossy.beta == 3.5
    assert "alpha 0.217 Np/m" in lossy.source
    assert "beta 3.5" in lossy.source
    assert "rho 1050 kg/m^3" in lossy.source and "c 1480 m/s" in lossy.source
    assert "alpha 0 Np/m" not in lossy.source
    # The thermal half is the database row and says so at its OWN density.
    assert "row 'Water'" in lossy.source and "rho = 994.035 kg/m^3" in lossy.source


# ------------------------------------------------------------ blood constants


def test_the_pennes_blood_constants_are_the_shipped_blood_row():
    """The perfusion sink uses ``rho_b C_b``; the table ships a blood row. The
    two are pinned against each other here so they cannot drift apart in
    silence, which is the failure a reader of two docstrings cannot see."""
    assert BLOOD_SPECIFIC_HEAT == ITIS_BLOOD_SPECIFIC_HEAT
    assert TISSUE_LIBRARY["blood"].specific_heat == ITIS_BLOOD_SPECIFIC_HEAT
    # The solver's density is the same row rounded to the kilogram: 1050 kg/m^3
    # against the database's 1049.75, 0.024 % apart, which is two orders of
    # magnitude below the spread the database itself reports for blood
    # (1025 to 1060 kg/m^3, 3.4 %: a factor 143).
    assert BLOOD_DENSITY == pytest.approx(ITIS_BLOOD_DENSITY, rel=3e-4)
    lo, hi = TISSUE_LIBRARY["blood"].rho
    assert lo <= BLOOD_DENSITY <= hi
    assert ITIS_DB in TISSUE_LIBRARY["blood"].thermal_source


# --------------------------------------------------------------- table version


def test_the_table_carries_a_version_that_moves_with_its_numbers():
    """The shipped table is citable as one thing, and the citation is checked.

    A version string alone is a promise; what makes it true is that the digest
    beside it is computed from the rows. Both are pinned to a literal here, so
    editing any value or any citation in ``TISSUE_LIBRARY`` fails this test
    until the version is bumped in the same change. That is the whole
    mechanism: nothing else can notice the difference between "the liver
    conductivity was always 0.5191" and "someone changed it last month".

    When this test fails after a deliberate table edit, bump
    ``TISSUE_LIBRARY_VERSION`` and paste the new digest the failure prints.
    """
    assert TISSUE_LIBRARY_VERSION == "caustica-tissue-library/1"
    assert tissue_library_digest() == "5da6d84917da9a1b", (
        f"the shipped tissue table changed: digest is now {tissue_library_digest()!r}. "
        f"If that was deliberate, bump TISSUE_LIBRARY_VERSION and update both literals."
    )
    assert tissue_library_stamp() == f"{TISSUE_LIBRARY_VERSION} sha256:{tissue_library_digest()}"


def test_every_field_of_a_library_row_reaches_the_digest():
    """The digest reads the dataclass, so a property added later is hashed
    without anyone remembering to add it.

    Checked by moving each field in turn and requiring the digest to notice:
    a hand-written field list would pass this today and fail silently on the
    first new field, which is the one change the digest exists to catch.
    """
    before = tissue_library_digest()
    liver = TISSUE_LIBRARY["liver"]
    for f in fields(AcousticTissue):
        value = getattr(liver, f.name)
        if isinstance(value, str):
            moved = value + " x"
        elif isinstance(value, bool):
            moved = not value
        elif isinstance(value, tuple):
            moved = tuple(v + 1.0 for v in value)
        elif value is None:
            moved = 0.5
        else:
            moved = value + 1.0
        try:
            TISSUE_LIBRARY["liver"] = replace(liver, **{f.name: moved})
            assert tissue_library_digest() != before, f.name
        finally:
            TISSUE_LIBRARY["liver"] = liver
    assert tissue_library_digest() == before


def test_the_digest_notices_a_changed_number_and_a_changed_citation():
    """The digest is not decoration: it covers the values AND their sources,
    because a number whose citation quietly changed is a different claim."""
    before = tissue_library_digest()
    liver = TISSUE_LIBRARY["liver"]
    try:
        TISSUE_LIBRARY["liver"] = replace(liver, thermal_conductivity=0.6)
        assert tissue_library_digest() != before
        TISSUE_LIBRARY["liver"] = replace(liver, source=liver.source + " (rechecked)")
        assert tissue_library_digest() != before
    finally:
        TISSUE_LIBRARY["liver"] = liver
    assert tissue_library_digest() == before
