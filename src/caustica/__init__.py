"""caustica — GPU-accelerated multi-solver acoustic simulation for HIFU.

Public entry points are re-exported here so user code can stay short::

    import caustica as hs
    grid = hs.Grid(shape=(128, 128, 128), dx=0.3e-3)

or, for a whole job in one call (M10j)::

    res = caustica.simulate("job.json")      # plan, gates, progress, result
    res.metrics; res.preview(); res.save("result.h5")

Heavy optional dependencies (cupy, h5py, matplotlib) are imported lazily by
the modules that need them; ``import caustica`` itself only needs numpy.
That is why ``simulate`` is resolved through ``__getattr__`` (PEP 562):
reaching it pulls in the runner, and the runner needs h5py.
"""

from caustica.core.backend import (
    Backend,
    CausticaWarning,
    cpu_fft_workers,
    cupy_available,
    get_backend,
    set_cpu_fft_workers,
)
from caustica.core.grid import Grid
from caustica.core.pml import PMLSpec
from caustica.env import env_report, require_gpu
from caustica.materials import Material, MaterialDB
from caustica.medium import Medium

__version__ = "0.1.0.dev0"

#: Names that must not cost an h5py/pydantic import at ``import caustica``.
_LAZY = {
    "SimulationError": "caustica.facade",
    "SimulationRun": "caustica.facade",
    "simulate": "caustica.facade",
}

__all__ = [
    "Backend",
    "CausticaWarning",
    "Grid",
    "Material",
    "MaterialDB",
    "Medium",
    "PMLSpec",
    "SimulationError",
    "SimulationRun",
    "__version__",
    "cpu_fft_workers",
    "cupy_available",
    "env_report",
    "get_backend",
    "require_gpu",
    "set_cpu_fft_workers",
    "simulate",
]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'caustica' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
