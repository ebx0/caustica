"""Core building blocks: grid geometry, PML specification, backend dispatch."""

from hifusim.core.backend import Backend, cupy_available, get_backend
from hifusim.core.grid import Grid
from hifusim.core.pml import PMLSpec, sponge_profile_1d

__all__ = [
    "Backend",
    "Grid",
    "PMLSpec",
    "cupy_available",
    "get_backend",
    "sponge_profile_1d",
]
