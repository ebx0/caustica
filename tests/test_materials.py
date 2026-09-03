"""Material DB — notebook TISSUE_PROPS parity is a hard contract."""

import pytest
from pydantic import ValidationError

from caustica.materials import Material, MaterialDB, breast_default, water

# Verbatim from the production notebook (v6-v12): id -> (alpha, rho, c, beta).
NOTEBOOK_TISSUE_PROPS = {
    0: (0.1, 1000.0, 1500.0, 3.5),  # PML / matching layer
    1: (15.0, 1109.0, 1600.0, 4.0),  # Skin
    2: (6.0, 932.0, 1450.0, 4.5),  # Fat / glandular
    3: (10.0, 1050.0, 1580.0, 4.5),  # Muscle
    4: (0.1, 1000.0, 1500.0, 3.5),  # Coupling gel / background
}


def test_breast_default_matches_notebook_exactly():
    db = breast_default()
    assert db.ids == (0, 1, 2, 3, 4)
    for tissue_id, (alpha, rho, c, beta) in NOTEBOOK_TISSUE_PROPS.items():
        m = db[tissue_id]
        assert m.alpha_np_m == alpha, f"id {tissue_id} alpha drifted"
        assert m.rho == rho, f"id {tissue_id} rho drifted"
        assert m.c == c, f"id {tissue_id} c drifted"
        assert m.beta == beta, f"id {tissue_id} beta drifted"


def test_water_default_is_linear_lossless_validation_medium():
    w = water()
    assert (w.alpha_np_m, w.rho, w.c, w.beta) == (0.0, 1000.0, 1500.0, 0.0)


def test_material_validation():
    with pytest.raises(ValidationError):
        Material(alpha_np_m=-1.0, rho=1000.0, c=1500.0, beta=0.0)
    with pytest.raises(ValidationError):
        Material(alpha_np_m=0.0, rho=0.0, c=1500.0, beta=0.0)
    with pytest.raises(ValidationError):
        Material(alpha_np_m=0.0, rho=1000.0, c=1500.0, beta=0.0, unknown_field=1)


def test_db_json_roundtrip_preserves_int_keys():
    db = breast_default()
    again = MaterialDB.model_validate_json(db.model_dump_json())
    assert again == db
    assert 2 in again and again[2].name == "Fat"


# ---------------------------------------- literature tissue library


def test_tissue_library_values_pinned_to_the_digit():
    """The values MOVED from the uwcem tissue table are
    unchanged to the digit. These literals are frozen on purpose — a drifted
    number silently redefines what every exported phantom means."""
    from caustica.materials import DB_CM_TO_NP_M, TISSUE_LIBRARY

    assert DB_CM_TO_NP_M == 100.0 / 8.685889638065035
    expected = {
        "water_37c": (
            "Coupling medium (water, 37 C)",
            (1515.0, 1530.0),
            (992.0, 995.0),
            (0.0014, 0.0017),
            2.0,
            (5.0, 5.4),
        ),
        "skin": ("Skin", (1595.0, 1660.0), (1085.0, 1125.0), (2.20, 3.50), 1.0, (7.5, 8.3)),
        "muscle": (
            "Muscle (chest wall)",
            (1545.0, 1600.0),
            (1050.0, 1095.0),
            (0.57, 1.09),
            1.0,
            (7.0, 7.8),
        ),
        "fibroglandular": (
            "Fibroglandular-1 (highest water)",
            (1530.0, 1580.0),
            (1035.0, 1060.0),
            (0.85, 1.15),
            1.1,
            (7.0, 8.0),
        ),
        "fat": (
            "Fatty-3 (lowest water, lipid)",
            (1425.0, 1460.0),
            (900.0, 930.0),
            (0.40, 0.60),
            1.1,
            (9.6, 11.3),
        ),
    }
    assert set(TISSUE_LIBRARY) == set(expected)
    for key, (name, c, rho, alpha0, b, bona) in expected.items():
        t = TISSUE_LIBRARY[key]
        assert (t.name, t.c, t.rho, t.alpha0, t.b, t.bona) == (name, c, rho, alpha0, b, bona)
        assert t.interpolated is False  # every library row is literature-anchored


def test_tissue_library_material_conversion():
    from caustica.materials import TISSUE_LIBRARY

    skin = TISSUE_LIBRARY["skin"].to_material(1e6)  # 1 MHz, midpoints
    assert skin.c == 1627.5 and skin.rho == 1105.0
    assert skin.beta == 1.0 + 0.5 * 7.9
    assert skin.alpha_np_m == pytest.approx(2.85 * 100.0 / 8.685889638065035)


def test_uwcem_table_uses_the_moved_library_rows():
    """While the side package is still in this repo: its literature rows must
    BE the library objects (single source, zero drift). After the split the
    equivalent parity test lives in the uwcem-phantom repository."""
    uw_tissue = pytest.importorskip("uwcem_phantoms.tissue")
    from caustica.materials import TISSUE_LIBRARY

    t = uw_tissue.DEFAULT_TISSUES
    assert t[0] is TISSUE_LIBRARY["water_37c"]
    assert t[1] is TISSUE_LIBRARY["skin"]
    assert t[2] is TISSUE_LIBRARY["muscle"]
    assert t[3] is TISSUE_LIBRARY["fibroglandular"]
    assert t[9] is TISSUE_LIBRARY["fat"]
