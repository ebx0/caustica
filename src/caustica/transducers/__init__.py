"""Transducers: the element model, its quadrature, and its deposit.

One element model with four shapes, per-element amplitude, phase and delay,
and one band-limited deposit routine that treats them alike::

    import numpy as np
    from caustica import Grid
    from caustica.transducers import Transducer, rect

    pitch = 0.3e-3
    probe = Transducer(
        tuple(
            rect((x, 0.0, 0.0), (0.0, 0.0, 1.0), pitch - 0.05e-3, 5e-3)
            for x in (np.arange(64) - 31.5) * pitch
        )
    )
    probe = probe.with_drive(phases=probe.das_phases([0.0, 0.0, 40e-3], 1.5e6))
    grid = Grid(shape=(240, 96, 288), dx=0.1e-3)
    src = probe.deposit_cw(grid, (120, 48, 8), f0=1.5e6, amplitude=1e5).source

The Rayleigh integral over the same element surfaces is
:meth:`Transducer.rayleigh_field`, so the reference a solve is graded against
integrates over the geometry the solve was given.

:class:`TransducerArray` and :func:`archimedean_spiral` are the v1 model, kept
and re-expressed as a transducer of ``disc`` elements.
"""

from caustica.transducers.deposit import (
    ArraySource,
    TransientDeposit,
    deposit_cw,
    deposit_transient,
    element_deposit,
    quadrature_count,
)
from caustica.transducers.elements import (
    SHAPES,
    Element,
    disc,
    element_points,
    rect,
    ring_sector,
    spherical_segment,
    tangent_frame,
)
from caustica.transducers.model import (
    Frame,
    Transducer,
    TransducerArray,
    TransducerMeta,
    archimedean_spiral,
)

__all__ = [
    "SHAPES",
    "ArraySource",
    "Element",
    "Frame",
    "TransientDeposit",
    "Transducer",
    "TransducerArray",
    "TransducerMeta",
    "archimedean_spiral",
    "deposit_cw",
    "deposit_transient",
    "disc",
    "element_deposit",
    "element_points",
    "quadrature_count",
    "rect",
    "ring_sector",
    "spherical_segment",
    "tangent_frame",
]
