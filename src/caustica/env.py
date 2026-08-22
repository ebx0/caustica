"""Environment policy (M10i): what machine is this, and can it run GPU work.

:func:`env_report` is the ONE composition of environment facts — the
runner's ``run_meta.json`` stamp and a notebook printout call the same
function, so they cannot disagree. :func:`require_gpu` is D6 made concrete:
caustica never installs anything for you; it names the fix for where you
actually are (a Colab CPU runtime is fixed in the Runtime menu, not by pip).
"""

from __future__ import annotations

import os
import platform
import sys

from caustica.core.backend import Backend, cupy_available, get_backend

__all__ = ["env_report", "gpu_environment", "require_gpu"]


def gpu_environment(backend_name: str) -> dict:
    """Device/driver/cupy stamp when running on CUDA; empty otherwise.

    (Runner heritage: this is the former ``runner._gpu_environment``,
    promoted here so the stamp and the notebook share one source.)
    """
    if backend_name != "cupy":
        return {}
    try:  # pragma: no cover - requires a GPU
        import cupy  # noqa: PLC0415 (lazy on purpose)

        dev = cupy.cuda.Device()
        props = cupy.cuda.runtime.getDeviceProperties(dev.id)
        free_b, total_b = dev.mem_info
        return {
            "gpu_name": props["name"].decode()
            if isinstance(props["name"], bytes)
            else str(props["name"]),
            "driver_version": cupy.cuda.runtime.driverGetVersion(),
            "cuda_runtime_version": cupy.cuda.runtime.runtimeGetVersion(),
            "cupy_version": cupy.__version__,
            "vram_total_gib": round(total_b / 2**30, 3),
            "vram_free_gib": round(free_b / 2**30, 3),
        }
    except Exception as exc:
        return {"gpu_probe_error": f"{type(exc).__name__}: {exc}"}


def _version_of(module_name: str) -> str | None:
    try:
        import importlib.metadata as md  # noqa: PLC0415

        return md.version(module_name)
    except Exception:
        return None


def env_report(backend_name: str | None = None) -> dict:
    """One dict describing this environment. **Never raises.**

    ``backend_name=None`` resolves ``"auto"`` (what a run would get);
    the runner passes the backend it actually resolved so the stamp
    describes the run, not a hypothetical. The historical stamp keys
    (``caustica``/``python``/``numpy``/``platform`` + the GPU fields) keep
    their names — M8's Colab gates measure from them; new facts are ADDED.
    """
    report: dict = {}
    try:
        import caustica  # noqa: PLC0415 (cycle-safe at call time)

        report["caustica"] = caustica.__version__
    except Exception:
        report["caustica"] = None
    try:
        report["python"] = platform.python_version()
        report["platform"] = platform.platform()
    except Exception:
        pass
    # numpy via the imported module, like the historical stamp — it can never
    # be None there, and a gate reading environment["numpy"] may assume str.
    # The ADDED keys use dist metadata (kept import-free: h5py stays lazy, T6)
    # and may honestly be None on exotic installs.
    try:
        import numpy as np  # noqa: PLC0415

        report["numpy"] = np.__version__
    except Exception:
        report["numpy"] = None
    for key, dist in (("scipy", "scipy"), ("pydantic", "pydantic"), ("h5py", "h5py")):
        report[key] = _version_of(dist)
    try:
        if backend_name is None:
            backend_name = get_backend("auto").name
        report["resolved_backend"] = backend_name
        report.update(gpu_environment(backend_name))
    except Exception as exc:  # even a broken CUDA stack must not raise here
        report["resolved_backend"] = f"probe_error: {type(exc).__name__}"
    return report


def _on_colab() -> bool:
    return "google.colab" in sys.modules or any(k.startswith("COLAB_") for k in os.environ)


def require_gpu(reason: str = "") -> Backend:
    """Return the cupy backend or raise with the fix for THIS machine (D6).

    Never touches pip. On Colab the real failure is almost always a CPU
    runtime — which no install can fix — so that message points at the
    Runtime menu; elsewhere it names the install command for the user to
    run themselves.
    """
    if cupy_available():
        return get_backend("cupy")
    why = f" ({reason})" if reason else ""
    if _on_colab():
        raise RuntimeError(
            f"A GPU is required{why}, but this Colab runtime has no CUDA device. "
            f"Fix: Runtime -> Change runtime type -> Hardware accelerator: GPU, then "
            f"reconnect. (No pip install can fix a CPU runtime; Colab GPU runtimes "
            f"already ship cupy.)"
        )
    raise RuntimeError(
        f"A GPU is required{why}, but no usable CUDA device was found on this machine. "
        f"If it has an NVIDIA GPU with CUDA 12: pip install cupy-cuda12x (the "
        f"caustica[gpu] extra). caustica never installs it for you."
    )
