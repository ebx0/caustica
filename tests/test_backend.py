"""Backend dispatch — numpy always, cupy optional, auto falls back."""

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


# ---------------------------------------------------- CPU FFT workers


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
    sums, so the solve must be BIT-identical for any worker count — the
    gate is strict equality, not a tolerance."""
    np.testing.assert_array_equal(_run_mini(1), _run_mini(-1))
    np.testing.assert_array_equal(_run_mini(1, nonlinear=True), _run_mini(-1, nonlinear=True))


# ------------------------------------------------- the CuPy compile probe


class _FakeKernel:
    """Stand-in for ``cupy.ElementwiseKernel`` with a scripted outcome."""

    def __init__(self, *args, raises=None, answer=None, **kw):
        self._raises = raises
        self._answer = answer

    def __call__(self, x, y):
        if self._raises is not None:
            raise self._raises
        return x + y if self._answer is None else self._answer


def _fake_cupy(*, devices=1, raises=None, answer=None):
    """A minimal module object that ``import cupy`` can find in sys.modules.

    Enough surface for the probe and nothing else: the point is to reach the
    kernel path on a machine that may have no CUDA at all.
    """
    import types

    mod = types.ModuleType("cupy")
    mod.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: devices))
    mod.float32 = np.float32
    mod.arange = np.arange
    mod.ElementwiseKernel = lambda *a, **kw: _FakeKernel(*a, raises=raises, answer=answer, **kw)
    return mod


def _probe_with(monkeypatch, module):
    """Run one cold ``cupy_available()`` against ``module``; restore after."""
    import sys

    from caustica.core import backend as B

    monkeypatch.setitem(sys.modules, "cupy", module)
    monkeypatch.setitem(B._CUPY_STATE, "checked", False)
    monkeypatch.setitem(B._CUPY_STATE, "available", False)
    monkeypatch.setitem(B._CUPY_STATE, "module", None)
    monkeypatch.setitem(B._CUPY_STATE, "reason", None)
    monkeypatch.setitem(B._CUPY_STATE, "reason_kind", None)
    return B.cupy_available()


def test_probe_refuses_a_gpu_whose_kernels_do_not_compile(monkeypatch, caplog):
    """I-023: counting a device is not proof that a kernel runs."""
    import logging

    from caustica.core import backend as B

    boom = RuntimeError("nvrtc: could not load nvrtc-builtins64_124.dll")
    with caplog.at_level(logging.DEBUG, logger="caustica"):
        available = _probe_with(monkeypatch, _fake_cupy(raises=boom))
    assert available is False
    reason = B.cupy_unavailable_reason()
    assert reason is not None
    assert "RuntimeError" in reason and "nvrtc" in reason
    # Pin the level and the logger, not just the text: promoting this line to
    # INFO or WARNING would put a CUDA traceback in front of every CPU user,
    # and a bare substring check would not notice.
    assert any(
        rec.levelno == logging.DEBUG and rec.name == "caustica" and "nvrtc" in rec.getMessage()
        for rec in caplog.records
    )
    # and the refusal that follows is still the actionable one
    with pytest.raises(RuntimeError, match="no usable CUDA GPU"):
        get_backend("cupy")


def test_probe_refuses_a_kernel_that_computes_the_wrong_answer(monkeypatch):
    """A launch that silently returns garbage is not a usable GPU either."""
    from caustica.core import backend as B

    assert _probe_with(monkeypatch, _fake_cupy(answer=np.zeros(16, dtype=np.float32))) is False
    assert "expected" in (B.cupy_unavailable_reason() or "")


def test_probe_accepts_a_stack_where_the_kernel_runs(monkeypatch):
    from caustica.core import backend as B

    assert _probe_with(monkeypatch, _fake_cupy()) is True
    assert B.cupy_unavailable_reason() is None


def test_zero_devices_is_reported_as_a_reason_not_a_silence(monkeypatch):
    from caustica.core import backend as B

    assert _probe_with(monkeypatch, _fake_cupy(devices=0)) is False
    assert "0 devices" in (B.cupy_unavailable_reason() or "")


def test_reason_is_none_before_anything_probes(monkeypatch):
    """Reading the reason must never itself start a CUDA context."""
    from caustica.core import backend as B

    monkeypatch.setitem(B._CUPY_STATE, "checked", False)
    monkeypatch.setitem(B._CUPY_STATE, "reason", "stale")
    assert B.cupy_unavailable_reason() is None


def test_a_failing_probe_reaches_env_report(monkeypatch):
    """The reason a REAL probe cached, not one a fixture wrote, reaches the stamp."""
    from caustica.env import env_report

    assert _probe_with(monkeypatch, _fake_cupy(raises=RuntimeError("nvrtc boom"))) is False
    rep = env_report()
    assert rep["resolved_backend"] == "numpy"
    assert "nvrtc boom" in rep["gpu_unavailable_reason"]


def test_require_gpu_names_the_probe_fault_instead_of_advising_pip(monkeypatch):
    """A machine that HAS cupy and a device must not be told to install cupy."""
    from caustica import env
    from caustica.core import backend as B

    monkeypatch.setattr(env, "_on_colab", lambda: False)
    assert _probe_with(monkeypatch, _fake_cupy(raises=RuntimeError("nvrtc boom"))) is False
    assert B.cupy_unavailable_kind() == "unusable"
    with pytest.raises(RuntimeError) as exc:
        env.require_gpu("a focused run")
    msg = str(exc.value)
    assert "nvrtc boom" in msg
    assert "a focused run" in msg
    assert "pip install" not in msg


def test_the_three_failure_kinds_are_told_apart_structurally(monkeypatch):
    """Classification is by exception type, never by words in the message."""
    import sys

    from caustica.core import backend as B

    assert _probe_with(monkeypatch, _fake_cupy(devices=0)) is False
    assert B.cupy_unavailable_kind() == "no_device"
    assert _probe_with(monkeypatch, _fake_cupy(raises=RuntimeError("anything"))) is False
    assert B.cupy_unavailable_kind() == "unusable"
    monkeypatch.setitem(sys.modules, "cupy", None)
    for k, v in (("checked", False), ("available", False), ("module", None), ("reason", None)):
        monkeypatch.setitem(B._CUPY_STATE, k, v)
    monkeypatch.setitem(B._CUPY_STATE, "reason_kind", None)
    assert B.cupy_available() is False
    assert B.cupy_unavailable_kind() == "import"
    # and the documented set is the whole set, so a fourth kind cannot reach
    # the probe without this test noticing.
    assert set(B.REASON_KINDS) == {"import", "no_device", "unusable"}


def test_require_gpu_keeps_the_install_advice_for_a_missing_cupy(monkeypatch):
    """The advice survives for the one failure it actually answers."""
    import sys

    from caustica import env
    from caustica.core import backend as B

    monkeypatch.setattr(env, "_on_colab", lambda: False)
    monkeypatch.setitem(sys.modules, "cupy", None)  # import cupy -> ImportError
    for k, v in (("checked", False), ("available", False), ("module", None), ("reason", None)):
        monkeypatch.setitem(B._CUPY_STATE, k, v)
    monkeypatch.setitem(B._CUPY_STATE, "reason_kind", None)
    assert B.cupy_available() is False
    assert B.cupy_unavailable_kind() == "import"
    with pytest.raises(RuntimeError, match="pip install cupy-cuda12x"):
        env.require_gpu("a focused run")


def test_env_report_carries_the_reason_when_there_is_no_gpu(no_gpu):
    from tests.conftest import NO_GPU_REASON

    from caustica.env import env_report

    rep = env_report()
    assert rep["resolved_backend"] == "numpy"
    assert rep["gpu_unavailable_reason"] == NO_GPU_REASON


@pytest.mark.gpu
@pytest.mark.skipif(not cupy_available(), reason="needs a CUDA device")
def test_real_machine_compiles_the_probe_kernel():
    from caustica.core import backend as B
    from caustica.env import env_report

    assert cupy_available() is True
    assert B.cupy_unavailable_reason() is None
    assert B.cupy_unavailable_kind() is None
    assert get_backend("cupy").name == "cupy"
    assert "gpu_unavailable_reason" not in env_report()


@pytest.mark.gpu
@pytest.mark.skipif(not cupy_available(), reason="needs a CUDA device")
def test_cold_probe_stays_under_two_seconds():
    """The first call happens inside `caustica run` before anything is built,
    so it is a latency the user feels. Measured in a FRESH interpreter: an
    in-process call would only re-read the cache."""
    import subprocess
    import sys
    import textwrap

    src = textwrap.dedent(
        """
        import time
        from caustica.core.backend import cupy_available
        t = time.perf_counter()
        ok = cupy_available()
        print(ok, time.perf_counter() - t)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", src], capture_output=True, text=True, timeout=120, check=True
    )
    ok, seconds = out.stdout.split()
    assert ok == "True"
    assert float(seconds) < 2.0, f"cold cupy probe took {seconds} s"
