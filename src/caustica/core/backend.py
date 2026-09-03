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
* Backends are a **registry** (M10n/K15): ``numpy`` and ``cupy`` are two
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

_CUPY_STATE: dict[str, Any] = {"checked": False, "available": False, "module": None}

#: One warning per process for the auto->numpy fallback (D33): visible once,
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


def cupy_available() -> bool:
    """Return True when cupy imports AND at least one CUDA device responds.

    The result is cached for the process lifetime: probing the CUDA runtime
    is not free, and availability does not change mid-process in practice.
    """
    if not _CUPY_STATE["checked"]:
        _CUPY_STATE["checked"] = True
        try:
            import cupy  # noqa: PLC0415 (lazy on purpose)

            n_dev = cupy.cuda.runtime.getDeviceCount()
            _CUPY_STATE["available"] = n_dev > 0
            _CUPY_STATE["module"] = cupy if n_dev > 0 else None
        except Exception as exc:  # ImportError or any CUDA runtime error
            log.debug("cupy unavailable: %s: %s", type(exc).__name__, exc)
            _CUPY_STATE["available"] = False
    return bool(_CUPY_STATE["available"])


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
        machines are not silently single-threaded (D32).
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
            # ONCE per process (D33): the old INFO log had no handler and was
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
    unreachable (M10n review). Everything downstream — the run stamp in
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
