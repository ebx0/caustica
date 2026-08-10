"""hifusim — GPU-accelerated multi-solver acoustic simulation for HIFU.

Public entry points are re-exported here so user code can stay short::

    import hifusim as hs
    grid = hs.Grid(shape=(128, 128, 128), dx=0.3e-3)

Heavy optional dependencies (cupy, h5py, matplotlib) are imported lazily by
the modules that need them; ``import hifusim`` itself only needs numpy.
"""

from hifusim.core.backend import Backend, cupy_available, get_backend
from hifusim.core.grid import Grid
from hifusim.core.pml import PMLSpec
from hifusim.materials import Material, MaterialDB
from hifusim.medium import Medium

__version__ = "0.1.0.dev0"

__all__ = [
    "Backend",
    "Grid",
    "Material",
    "MaterialDB",
    "Medium",
    "PMLSpec",
    "__version__",
    "cupy_available",
    "get_backend",
]
