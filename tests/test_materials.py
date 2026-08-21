"""M2 gate: material DB — notebook TISSUE_PROPS parity is a hard contract."""

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
