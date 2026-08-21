"""M1 gate: backend dispatch — numpy always, cupy optional, auto falls back."""

import numpy as np
import pytest

from caustica.core.backend import cupy_available, get_backend


def test_numpy_backend_always_works():
    b = get_backend("numpy")
    assert b.name == "numpy"
    assert b.xp is np
    assert not b.is_gpu
    b.synchronize()  # must be a no-op, not an error


def test_auto_backend_resolves():
    b = get_backend("auto")
    assert b.name in ("numpy", "cupy")
    if not cupy_available():
        assert b.name == "numpy"


def test_cupy_request_fails_actionably_without_gpu():
    if cupy_available():
        pytest.skip("machine has a GPU; failure path not reachable")
    with pytest.raises(RuntimeError, match="cupy"):
        get_backend("cupy")


def test_unknown_backend_name_rejected():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("torch")  # type: ignore[arg-type]


def test_roundtrip_to_numpy():
    b = get_backend("numpy")
    a = b.asarray([1.0, 2.0], dtype=np.float32)
    back = b.to_numpy(a)
    assert isinstance(back, np.ndarray)
    assert back.dtype == np.float32
    np.testing.assert_array_equal(back, [1.0, 2.0])
