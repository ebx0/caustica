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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal

import numpy as np

log = logging.getLogger("caustica")

BackendName = Literal["auto", "numpy", "cupy"]

_CUPY_STATE: dict[str, Any] = {"checked": False, "available": False, "module": None}


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
    def fft(self) -> ModuleType:
        """dtype-preserving FFT module for this backend.

        numpy.fft always upcasts float32 -> complex128, which would break
        fp32 production parity between CPU and GPU; scipy.fft (pocketfft)
        and cupyx.scipy.fft both keep float32/complex64. Solvers must use
        ``backend.fft``, never ``numpy.fft``.
        """
        if self.is_gpu:
            import cupyx.scipy.fft as cufft  # noqa: PLC0415 (lazy on purpose)

            return cufft
        import scipy.fft as spfft  # noqa: PLC0415 (lazy on purpose)

        return spfft

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


def get_backend(name: BackendName = "auto") -> Backend:
    """Resolve a backend by name.

    - ``"numpy"``: always works.
    - ``"cupy"``: raises ``RuntimeError`` with a fix-it message if no GPU.
    - ``"auto"``: cupy when available, else numpy (logged once).
    """
    if name == "numpy":
        return Backend("numpy", np)
    if name == "cupy":
        if not cupy_available():
            raise RuntimeError(
                "Backend 'cupy' requested but no usable CUDA GPU was found. "
                "Install the extra (pip install caustica[gpu]) on a CUDA machine, "
                "or use backend='auto' to fall back to numpy."
            )
        return Backend("cupy", _CUPY_STATE["module"])
    if name == "auto":
        if cupy_available():
            return Backend("cupy", _CUPY_STATE["module"])
        log.info("backend auto-select: no CUDA GPU found, using numpy (CPU).")
        return Backend("numpy", np)
    raise ValueError(f"Unknown backend name {name!r}; expected 'auto', 'numpy' or 'cupy'.")
