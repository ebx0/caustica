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


def test_cupy_request_fails_actionably_without_gpu(no_gpu):
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


# ---------------------------------------------------- CPU FFT workers (M10i)


def _run_mini(workers: int, nonlinear: bool = False):
    """One tiny solve with the given worker count; returns the phasor field."""
    from caustica.core.backend import set_cpu_fft_workers
    from caustica.core.grid import Grid
    from caustica.core.pml import PMLSpec
    from caustica.materials import water
    from caustica.medium import Medium
    from caustica.solvers import CWRunSpec, get
    from caustica.sources import bowl_cw_source

    set_cpu_fft_workers(workers)
    try:
        grid = Grid(shape=(28, 28, 36), dx=0.6e-3, pml=PMLSpec(thickness=3e-3))
        db = water()
        medium = Medium.homogeneous(grid.shape, db)
        src = bowl_cw_source(
            grid, f0=1.0e6, amplitude=1e5, aperture_radius=4e-3, roc=10e-3, apex_vox=(14, 14, 6)
        )
        solver = get("westervelt" if nonlinear else "linear")()
        res = solver.run(
            grid,
            medium,
            src,
            CWRunSpec(min_settle_periods=2, max_settle_periods=4),
            backend="numpy",
        )
        return res.phasor
    finally:
        set_cpu_fft_workers(None)


def test_cpu_fft_workers_default_and_overrides(monkeypatch):
    from caustica.core import backend as B

    monkeypatch.delenv("CAUSTICA_CPU_WORKERS", raising=False)
    assert B.cpu_fft_workers() == 1  # measured decision, 2026-08-22
    monkeypatch.setenv("CAUSTICA_CPU_WORKERS", "-1")
    assert B.cpu_fft_workers() == -1
    monkeypatch.setenv("CAUSTICA_CPU_WORKERS", "junk")
    assert B.cpu_fft_workers() == 1
    B.set_cpu_fft_workers(4)
    try:
        assert B.cpu_fft_workers() == 4  # setter beats env
    finally:
        B.set_cpu_fft_workers(None)


def test_workers_wrapper_injects_and_yields_to_explicit(monkeypatch):
    import scipy.fft as spfft

    from caustica.core import backend as B

    B.set_cpu_fft_workers(-1)
    try:
        fft = B.get_backend("numpy").fft
        assert isinstance(fft, B._ScipyFFTWithWorkers)
        x = np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32)
        # injected default and explicit workers= both work and agree
        np.testing.assert_array_equal(fft.rfftn(x), spfft.rfftn(x, workers=-1))
        np.testing.assert_array_equal(fft.rfftn(x, workers=2), spfft.rfftn(x, workers=2))
        # non-transform helpers pass through untouched
        assert fft.next_fast_len is spfft.next_fast_len
    finally:
        B.set_cpu_fft_workers(None)


def test_workers_one_returns_raw_scipy_module():
    import scipy.fft as spfft

    from caustica.core import backend as B

    B.set_cpu_fft_workers(1)
    try:
        assert B.get_backend("numpy").fft is spfft
    finally:
        B.set_cpu_fft_workers(None)


def test_fields_bit_identical_across_worker_counts():
    """pocketfft distributes 1-D lines over threads without reordering the
    sums, so the solve must be BIT-identical for any worker count — the D32
    gate is strict equality, not a tolerance."""
    np.testing.assert_array_equal(_run_mini(1), _run_mini(-1))
    np.testing.assert_array_equal(_run_mini(1, nonlinear=True), _run_mini(-1, nonlinear=True))
