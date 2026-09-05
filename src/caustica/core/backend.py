"""Backend dispatch: one gateway to numpy (CPU) or cupy (CUDA GPU).

Design notes
------------
* Solvers and other heavy numerical code never import numpy/cupy at module
  level for math; they receive a :class:`Backend` and use ``backend.xp``.
  The same solver source then runs on CPU and GPU. This is the load-bearing
  decision that keeps caustica free of precompiled binaries (k-Wave's Colab
  failure mode) and usable on Windows (j-Wave/JAX's weak spot).
* The cupy import is *lazy*: importing caustica on a GPU-less machine never
  touches cupy. ``get_backend("cupy")`` raises an actionable error when no
  usable GPU exists; ``get_backend("auto")`` silently falls back to numpy
  (logged once at INFO level).
* A backend is intentionally a thin, frozen value object. Anything stateful
  (FFT plans, memory pools, streams) belongs to the solver that owns the run.
* Backends are a **registry**: ``numpy`` and ``cupy`` are two
  registered factories, not two branches of an ``if``. A third party adds one
  through the ``caustica.backends`` entry-point group and names it in a job.
  ``"auto"`` is not a backend but a *policy* over them (cupy if usable, else
  numpy) and stays deliberately built-in.
"""

from __future__ import annotations

import functools
import logging
import os
import warnings
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal

import numpy as np

from caustica.registry import BACKEND_GROUP, FactoryRegistry, UnknownPluginError

log = logging.getLogger("caustica")

#: The names caustica itself ships. Any string a registered factory answers
#: to is valid too — the alias is kept for the common case and for typing.
BackendName = Literal["auto", "numpy", "cupy"]

#: name -> zero-argument factory returning a :class:`Backend`.
backends: FactoryRegistry = FactoryRegistry("backend", BACKEND_GROUP)

#: Cached answer of the CUDA probe. ``reason`` carries the exception text
#: when the probe said no, so callers can say WHY instead of just "no GPU".
_CUPY_STATE: dict[str, Any] = {
    "checked": False,
    "available": False,
    "module": None,
    "reason": None,
    "reason_kind": None,
}

#: The three ways a CUDA stack fails, kept apart because each has a DIFFERENT
#: fix and the caller has to name the right one. Classified structurally (by
#: exception type), never by matching words in the message.
#:
#: * ``"import"``     cupy is not installed here.
#: * ``"no_device"``  cupy imports, the runtime counts no device. Often the
#:   wrong wheel for the installed driver, which an install DOES fix.
#: * ``"unusable"``   a device answered and the kernel still did not run:
#:   driver, toolkit or device state. No install fixes this one.
REASON_KINDS = ("import", "no_device", "unusable")

#: Element count of the compile probe. Small on purpose: the cost being
#: measured is the NVRTC compile plus one launch, not the arithmetic.
_PROBE_N = 16


class _NoCudaDevice(RuntimeError):
    """cupy imported and the runtime counted zero devices (its own kind)."""


#: One warning per process for the auto->numpy fallback: visible once,
#: noise never. Tests reset this directly.
_AUTO_FALLBACK_WARNED = False


class CausticaWarning(UserWarning):
    """Base category for every warning caustica itself emits.

    Filterable without silencing the rest of the ecosystem::

        warnings.filterwarnings("ignore", category=caustica.CausticaWarning)
    """


#: Thread-count override for CPU (scipy.fft) transforms; ``None`` defers to
#: the ``CAUSTICA_CPU_WORKERS`` env var, then the default of 1. The default
#: is DELIBERATELY single-threaded: two measurement rounds on a 10-core
#: i5-13450HX (2026-08-22) found no reproducible speed-up from any
#: worker count at 1-26 Mvox engine shapes — apparent per-cell wins and
#: losses did not survive a repeat. Machines that do profit can opt in via
#: the env var or :func:`set_cpu_fft_workers`; pocketfft splits nd
#: transforms into 1-D lines per thread without changing summation order,
#: so fields stay bit-identical across worker counts (asserted by test).
_CPU_FFT_WORKERS: int | None = None


def cpu_fft_workers() -> int:
    """Active ``workers=`` value for CPU FFTs (setter > env var > 1)."""
    if _CPU_FFT_WORKERS is not None:
        return _CPU_FFT_WORKERS
    try:
        return int(os.environ.get("CAUSTICA_CPU_WORKERS", "1"))
    except ValueError:
        return 1


def set_cpu_fft_workers(n: int | None) -> None:
    """Set (or with ``None`` reset) the process-wide CPU FFT worker count.

    Takes effect for backends resolved afterwards; a running solve keeps
    the handle it grabbed at start.
    """
    global _CPU_FFT_WORKERS
    _CPU_FFT_WORKERS = None if n is None else int(n)


class _ScipyFFTWithWorkers:
    """``scipy.fft`` facade with a default ``workers=`` injected into the
    plan-level transforms.

    This exists so the *one* branch between threaded-CPU and GPU FFTs lives
    here instead of in the solvers' hot loops: ``cupyx.scipy.fft`` has no
    ``workers`` parameter (verified against CuPy docs, 2026-08-22), so the
    cupy path returns the raw module and never sees the kwarg. An explicit
    ``workers=`` at a call site still wins over the injected default.
    """

    _WORKERED = frozenset(
        {"fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "fftn", "ifftn", "rfftn", "irfftn"}
    )

    def __init__(self, module: ModuleType, workers: int):
        self._module = module
        self._workers = workers

    def __getattr__(self, name: str) -> Any:
        fn = getattr(self._module, name)
        if name in self._WORKERED:
            fn = functools.partial(fn, workers=self._workers)
        self.__dict__[name] = fn  # cache: later lookups skip __getattr__
        return fn


def _run_probe_kernel(cupy: ModuleType) -> None:
    """Compile and run a trivial ElementwiseKernel; raise if anything fails.

    Counting devices only proves the driver answers. Every kernel caustica
    launches is compiled at run time by NVRTC, so the machines that hurt are
    the ones where the runtime is present and the compile is not: no CUDA
    toolkit headers, a driver older than the compiled PTX, a read-only
    kernel cache, a device already out of memory. This adds one add of
    ``_PROBE_N`` float32 elements and reads the result back, which exercises
    exactly that chain: compile, launch, and a device-to-host copy.
    """
    kernel = cupy.ElementwiseKernel("T x, T y", "T z", "z = x + y", "caustica_probe_add")
    a = cupy.arange(_PROBE_N, dtype=cupy.float32)
    got = float(kernel(a, a)[-1])
    want = 2.0 * (_PROBE_N - 1)
    if got != want:
        raise RuntimeError(f"probe kernel returned {got!r} for the last element, expected {want!r}")


def cupy_available() -> bool:
    """Return True when cupy imports, a CUDA device responds AND a kernel runs.

    The last clause is the honest part: a machine can import cupy and count a
    device while every kernel compile fails, and answering True there sends a
    run into a crash minutes later instead of a fallback now. Any failure is
    cached as False together with its exception text (see
    :func:`cupy_unavailable_reason`) and logged at DEBUG.

    The result is cached for the process lifetime: probing the CUDA runtime
    is not free, and availability does not change mid-process in practice.
    """
    if not _CUPY_STATE["checked"]:
        _CUPY_STATE["checked"] = True
        try:
            import cupy  # noqa: PLC0415 (lazy on purpose)

            n_dev = cupy.cuda.runtime.getDeviceCount()
            if n_dev < 1:
                raise _NoCudaDevice("cupy imports but the CUDA runtime reports 0 devices")
            _run_probe_kernel(cupy)
            _CUPY_STATE["available"] = True
            _CUPY_STATE["module"] = cupy
            _CUPY_STATE["reason"] = None
            _CUPY_STATE["reason_kind"] = None
        except Exception as exc:  # ImportError, CUDA runtime, or compile error
            reason = f"{type(exc).__name__}: {exc}"
            log.debug("cupy unavailable: %s", reason)
            _CUPY_STATE["available"] = False
            _CUPY_STATE["module"] = None
            _CUPY_STATE["reason"] = reason
            _CUPY_STATE["reason_kind"] = (
                "import"
                if isinstance(exc, ImportError)
                else "no_device"
                if isinstance(exc, _NoCudaDevice)
                else "unusable"
            )
    return bool(_CUPY_STATE["available"])


def cupy_unavailable_reason() -> str | None:
    """Why the CUDA probe said no, or ``None``.

    ``None`` means either "a usable GPU is here" or "nothing has asked yet":
    this reads the cache and never probes, so calling it can neither create
    a CUDA context nor cost time. Call :func:`cupy_available` first when the
    answer has to be definitive.
    """
    return _CUPY_STATE["reason"] if _CUPY_STATE["checked"] else None


def cupy_unavailable_kind() -> str | None:
    """Which of :data:`REASON_KINDS` the probe hit, or ``None``.

    The companion of :func:`cupy_unavailable_reason`: the text is for the
    user to read, this is for the caller to branch on, so an advice line
    never has to guess a cause by matching words in an exception message.
    Reads the cache and never probes.
    """
    return _CUPY_STATE["reason_kind"] if _CUPY_STATE["checked"] else None


@dataclass(frozen=True)
class Backend:
    """A named array-module wrapper (``numpy`` or ``cupy``)."""

    name: str
    xp: ModuleType

    @property
    def is_gpu(self) -> bool:
        return self.name == "cupy"

    @property
    def fft(self) -> Any:
        """dtype-preserving FFT interface for this backend.

        numpy.fft always upcasts float32 -> complex128, which would break
        fp32 production parity between CPU and GPU; scipy.fft (pocketfft)
        and cupyx.scipy.fft both keep float32/complex64. Solvers must use
        ``backend.fft``, never ``numpy.fft``. On CPU the transforms carry a
        default ``workers=`` (see :func:`cpu_fft_workers`) so multi-core
        machines are not silently single-threaded.
        """
        if self.is_gpu:
            import cupyx.scipy.fft as cufft  # noqa: PLC0415 (lazy on purpose)

            return cufft
        import scipy.fft as spfft  # noqa: PLC0415 (lazy on purpose)

        workers = cpu_fft_workers()
        if workers == 1:
            return spfft
        return _ScipyFFTWithWorkers(spfft, workers)

    def asarray(self, a: Any, dtype: Any = None) -> Any:
        """Move/convert ``a`` onto this backend."""
        return self.xp.asarray(a, dtype=dtype)

    def to_numpy(self, a: Any) -> np.ndarray:
        """Bring an array back to host memory as numpy (no-op on CPU)."""
        if self.is_gpu:
            return self.xp.asnumpy(a)
        return np.asarray(a)

    def synchronize(self) -> None:
        """Block until pending device work finishes (no-op on CPU).

        Needed for honest timing: GPU kernels launch asynchronously.
        """
        if self.is_gpu:
            self.xp.cuda.get_current_stream().synchronize()


@backends.register("numpy")
def _numpy_backend() -> Backend:
    """The always-available CPU backend."""
    return Backend("numpy", np)


@backends.register("cupy")
def _cupy_backend() -> Backend:
    """CUDA via CuPy; raises with a fix-it message when no GPU answers.

    The probe (and therefore the cupy import) happens HERE, when the backend
    is actually asked for — registering the factory touches nothing.
    """
    if not cupy_available():
        raise RuntimeError(
            "Backend 'cupy' requested but no usable CUDA GPU was found. "
            "Install the extra (pip install caustica[gpu]) on a CUDA machine, "
            "or use backend='auto' to fall back to numpy."
        )
    return Backend("cupy", _CUPY_STATE["module"])


def _unknown_backend(name: str) -> ValueError:
    """The actionable refusal: what IS registered, and how to add one."""
    return ValueError(
        f"Unknown backend name {name!r}. Available: {', '.join(backends.available())} "
        f"(plus 'auto', which picks cupy when a usable GPU is present and numpy "
        f"otherwise). Third-party backends are added through the "
        f"'{BACKEND_GROUP}' entry-point group."
    )


def check_backend_name(name: str) -> str:
    """Refuse an unregistered backend name early; returns it unchanged.

    Used by the job schema so a typo is a *config* error at validate time,
    not a crash minutes into a run. Costs one entry-point scan per process.
    """
    if name == "auto" or name in backends.available():
        return name
    raise _unknown_backend(name)


def get_backend(name: str = "auto") -> Backend:
    """Resolve a backend by name.

    - ``"numpy"``: always works.
    - ``"cupy"``: raises ``RuntimeError`` with a fix-it message if no GPU.
    - ``"auto"``: cupy when available, else numpy (warned once per process).
    - anything else: whatever a registered factory (yours, or a plugin's)
      answers to; an unregistered name is refused with the list of names.

    ``"auto"`` deliberately chooses only between the two built-ins: a
    third-party backend is opted into by name, never guessed at.
    """
    if name == "auto":
        if cupy_available():
            return _checked("cupy", backends.get("cupy"))
        global _AUTO_FALLBACK_WARNED
        if not _AUTO_FALLBACK_WARNED:
            # ONCE per process: the old INFO log had no handler and was
            # never seen by anyone; a warning is visible in notebooks and CI,
            # and once is signal — per-call would be noise (the suite calls
            # this hundreds of times).
            _AUTO_FALLBACK_WARNED = True
            warnings.warn(
                "backend 'auto': no usable CUDA GPU found — falling back to numpy "
                "(CPU). Expect CPU speeds; see caustica.require_gpu() for the fix.",
                CausticaWarning,
                stacklevel=2,
            )
        log.info("backend auto-select: no CUDA GPU found, using numpy (CPU).")
        return _checked("numpy", backends.get("numpy"))
    try:
        factory = backends.get(name)
    except UnknownPluginError:
        raise _unknown_backend(name) from None
    return _checked(name, factory)


def _checked(name: str, factory: Any) -> Backend:
    """Call a registered factory and hold it to the contract.

    Both checks exist because the closed ``Literal`` used to make them
    unreachable. Everything downstream — the run stamp in
    ``run_meta.json``, the ``backend`` attr in ``result.h5``, the checkpoint
    fingerprint that decides whether a resume is the SAME run — reads
    ``Backend.name``, never the name that was asked for. A factory whose
    name disagrees with its registry key would therefore mislabel a run and
    let a checkpoint written under one backend resume as another, with
    nothing to notice it.
    """
    backend = factory()
    if not isinstance(backend, Backend):
        raise TypeError(
            f"backend factory for '{name}' returned {type(backend).__name__}, "
            f"not a caustica.core.backend.Backend"
        )
    if backend.name != name:
        raise ValueError(
            f"backend factory for '{name}' returned a Backend named "
            f"{backend.name!r}. The registry key and Backend.name must match: "
            f"the run stamp, the result file and the checkpoint fingerprint all "
            f"record Backend.name, so a mismatch mislabels the run."
        )
    return backend
