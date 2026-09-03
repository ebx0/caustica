"""Medium property volumes — layout, mapping, refusal to guess."""

import numpy as np
import pytest

from caustica import Medium
from caustica.materials import breast_default, water


def test_homogeneous_water():
    med = Medium.homogeneous((8, 8, 8), water())
    assert med.shape == (8, 8, 8)
    assert med.c_min == med.c_max == 1500.0
    assert med.is_linear
    for name in ("alpha", "rho", "c", "beta"):
        vol = getattr(med, name)
        assert vol.dtype == np.float32
        assert vol.flags.c_contiguous
        assert np.unique(vol).size == 1


def test_from_id_map_assigns_each_tissue():
    db = breast_default()
    id_map = np.zeros((6, 6, 6), dtype=np.uint8)
    id_map[1] = 1
    id_map[2] = 2
    id_map[3] = 3
    id_map[4:] = 4
    med = Medium.from_id_map(id_map, db)
    for tissue_id in (0, 1, 2, 3, 4):
        mask = id_map == tissue_id
        m = db[tissue_id]
        assert np.all(med.alpha[mask] == np.float32(m.alpha_np_m))
        assert np.all(med.rho[mask] == np.float32(m.rho))
        assert np.all(med.c[mask] == np.float32(m.c))
        assert np.all(med.beta[mask] == np.float32(m.beta))
    assert med.c_min == pytest.approx(1450.0)
    assert med.c_max == pytest.approx(1600.0)
    assert not med.is_linear
    assert med.id_map is id_map


def test_unknown_id_refused_with_id_list():
    db = breast_default()
    id_map = np.full((4, 4), 9, dtype=np.int32)
    with pytest.raises(ValueError, match=r"ids \[9\]"):
        Medium.from_id_map(id_map, db)


def test_float_id_map_rejected():
    with pytest.raises(TypeError, match="integer-typed"):
        Medium.from_id_map(np.zeros((4, 4)), breast_default())


def test_shape_mismatch_rejected():
    a = np.zeros((4, 4), np.float32)
    with pytest.raises(ValueError, match="disagree in shape"):
        Medium(alpha=a, rho=a, c=np.zeros((4, 5), np.float32), beta=a)
    with pytest.raises(ValueError, match="id_map shape"):
        Medium(alpha=a, rho=a, c=a, beta=a, id_map=np.zeros((5, 5), np.int32))
