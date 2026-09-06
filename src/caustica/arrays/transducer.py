"""Deprecated location: the transducer model moved to :mod:`caustica.transducers`.

Kept as a re-export for one release so existing imports of
``caustica.arrays.transducer`` keep working. New code imports from
:mod:`caustica.transducers`, where the element model, the quadrature and the
band-limited deposit live.
"""

from __future__ import annotations

from caustica.transducers.deposit import ArraySource
from caustica.transducers.model import TransducerArray, archimedean_spiral

__all__ = ["ArraySource", "TransducerArray", "archimedean_spiral"]
