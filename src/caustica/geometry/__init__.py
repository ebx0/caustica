"""Geometry system: CSG primitives, scenes, labeled-volume import/resample.

Three deliberately separate concerns:

* :mod:`~caustica.geometry.shapes` — primitives + boolean algebra (``| & - ~``)
* :mod:`~caustica.geometry.scene` — ordered label painting + rasterization
* :mod:`~caustica.geometry.volumes` — heterogeneous voxel imports + resampling
* :mod:`~caustica.geometry.offgrid` — continuous surfaces as weighted grid
  sources, so a curved transducer carries its own area rather than its voxel
  count

Materials stay elsewhere: a Scene produces integer LABELS; the MaterialDB
gives labels physical meaning at Medium construction.
"""

from caustica.geometry.configs import SceneConfig, VolumeImportConfig
from caustica.geometry.offgrid import Deposit, band_limited_weights, spherical_cap_deposit
from caustica.geometry.scene import Scene
from caustica.geometry.shapes import (
    AffineShape,
    Ball,
    Box,
    Complement,
    Cylinder,
    Difference,
    Ellipsoid,
    HalfSpace,
    Intersection,
    Shape,
    Union,
)
from caustica.geometry.volumes import (
    LabelVolume,
    load_labels_txt,
    resample_scalar,
)

__all__ = [
    "AffineShape",
    "Ball",
    "Box",
    "Complement",
    "Cylinder",
    "Deposit",
    "Difference",
    "Ellipsoid",
    "HalfSpace",
    "Intersection",
    "LabelVolume",
    "Scene",
    "SceneConfig",
    "Shape",
    "Union",
    "VolumeImportConfig",
    "band_limited_weights",
    "load_labels_txt",
    "resample_scalar",
    "spherical_cap_deposit",
]
