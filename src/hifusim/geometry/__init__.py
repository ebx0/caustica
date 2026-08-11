"""Geometry system: CSG primitives, scenes, labeled-volume import/resample.

Three deliberately separate concerns:

* :mod:`~hifusim.geometry.shapes` — primitives + boolean algebra (``| & - ~``)
* :mod:`~hifusim.geometry.scene` — ordered label painting + rasterization
* :mod:`~hifusim.geometry.volumes` — heterogeneous voxel imports + resampling

Materials stay elsewhere: a Scene produces integer LABELS; the MaterialDB
gives labels physical meaning at Medium construction.
"""

from hifusim.geometry.configs import SceneConfig, VolumeImportConfig
from hifusim.geometry.scene import Scene
from hifusim.geometry.shapes import (
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
from hifusim.geometry.volumes import (
    LabelVolume,
    breast_phantom_mapping,
    load_breast_phantom,
    load_labels_txt,
)

__all__ = [
    "AffineShape",
    "Ball",
    "Box",
    "Complement",
    "Cylinder",
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
    "breast_phantom_mapping",
    "load_breast_phantom",
    "load_labels_txt",
]
