"""Transducer arrays (deprecated location; see :mod:`caustica.transducers`).

The geometry model, the per-shape quadrature and the band-limited deposit moved
to :mod:`caustica.transducers` with the v2 element model. This package stays for
one release as a re-export, plus the two pieces that are genuinely about arrays
rather than about elements: the element-table reader and the notebook's phase
maps.
"""

from caustica.arrays.elements import element_table_digest, elements_array, read_element_file
from caustica.arrays.phasemaps import build_phase_maps, select_phase_map_size
from caustica.transducers import (
    ArraySource,
    Element,
    Frame,
    Transducer,
    TransducerArray,
    TransducerMeta,
    archimedean_spiral,
)

__all__ = [
    "ArraySource",
    "Element",
    "Frame",
    "Transducer",
    "TransducerArray",
    "TransducerMeta",
    "archimedean_spiral",
    "build_phase_maps",
    "element_table_digest",
    "elements_array",
    "read_element_file",
    "select_phase_map_size",
]
