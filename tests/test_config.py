"""M1 gate: config contract — strict fields, JSON round-trip, one-way derivation."""

import pytest
from pydantic import ValidationError

from caustica.config import GridConfig, PMLConfig


def test_json_roundtrip_is_lossless():
    cfg = GridConfig(
        ndim=3, dx_mm=0.30, size_mm=(96.0, 96.0, 120.0), pml=PMLConfig(thickness_mm=5.0, edge=2.0)
    )
    again = GridConfig.model_validate_json(cfg.model_dump_json())
    assert again == cfg


def test_unknown_field_is_an_error():
    with pytest.raises(ValidationError):
        GridConfig(ndim=2, dx_mm=0.5, size_mm=(10.0, 10.0), dt_ns=90.0)


def test_shape_is_derived_one_way():
    cfg = GridConfig(ndim=3, dx_mm=0.30, size_mm=(96.0, 96.0, 96.0))
    assert cfg.shape == (320, 320, 320)
    assert cfg.pml_vox == 17  # 5 mm / 0.30 mm
    # Voxel counts cannot be hand-written: they are not fields.
    with pytest.raises((ValidationError, AttributeError)):
        cfg.shape = (128, 128, 128)  # type: ignore[misc]
    assert "shape" not in GridConfig.model_fields


def test_dimension_mismatch_rejected():
    with pytest.raises(ValidationError, match="one physical length per axis"):
        GridConfig(ndim=3, dx_mm=0.5, size_mm=(10.0, 10.0))


def test_too_small_axis_rejected():
    with pytest.raises(ValidationError, match="below 4 voxels"):
        GridConfig(ndim=2, dx_mm=1.0, size_mm=(3.0, 50.0))


def test_to_grid_materializes_si_units():
    cfg = GridConfig(ndim=2, dx_mm=0.5, size_mm=(20.0, 30.0))
    g = cfg.to_grid()
    assert g.shape == (40, 60)
    assert g.dx == pytest.approx(0.5e-3)
    assert g.pml_vox == 10  # default 5 mm PML at 0.5 mm
