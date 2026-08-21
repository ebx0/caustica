"""caustica — GPU-accelerated multi-solver acoustic simulation for HIFU.

Public entry points are re-exported here so user code can stay short::

    import caustica as hs
    grid = hs.Grid(shape=(128, 128, 128), dx=0.3e-3)

Heavy optional dependencies (cupy, h5py, matplotlib) are imported lazily by
the modules that need them; ``import caustica`` itself only needs numpy.
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
from caustica.materials import Material, MaterialDB
from caustica.medium import Medium

__version__ = "0.1.0.dev0"

__all__ = [
    "Backend",
    "CausticaWarning",
    "Grid",
    "Material",
    "MaterialDB",
    "Medium",
    "PMLSpec",
    "__version__",
    "cpu_fft_workers",
    "cupy_available",
    "get_backend",
    "set_cpu_fft_workers",
]
