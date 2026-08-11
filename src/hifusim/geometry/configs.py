"""JSON-serializable geometry configs (tagged-union CSG tree + file imports).

The scene a user builds in code and the scene stored in a JSON config are
the same thing: ``SceneConfig.build()`` reconstructs the exact Scene, and
imported volumes are stored as FILE REFERENCES (path + format + placement),
not as inline data — the config stays small and the file stays the single
source of truth.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from hifusim.config.models import HifusimModel
from hifusim.geometry import shapes as sh
from hifusim.geometry.scene import Scene
from hifusim.geometry.volumes import LabelVolume, load_breast_phantom

_MM = 1e-3


class BallConfig(HifusimModel):
    kind: Literal["ball"] = "ball"
    center_mm: tuple[float, ...]
    radius_mm: float = Field(..., gt=0.0)

    def build(self) -> sh.Shape:
        return sh.Ball([c * _MM for c in self.center_mm], self.radius_mm * _MM)


class BoxConfig(HifusimModel):
    kind: Literal["box"] = "box"
    center_mm: tuple[float, ...]
    size_mm: tuple[float, ...]

    def build(self) -> sh.Shape:
        return sh.Box([c * _MM for c in self.center_mm], [s * _MM for s in self.size_mm])


class EllipsoidConfig(HifusimModel):
    kind: Literal["ellipsoid"] = "ellipsoid"
    center_mm: tuple[float, ...]
    semiaxes_mm: tuple[float, ...]

    def build(self) -> sh.Shape:
        return sh.Ellipsoid([c * _MM for c in self.center_mm], [s * _MM for s in self.semiaxes_mm])


class CylinderConfig(HifusimModel):
    kind: Literal["cylinder"] = "cylinder"
    center_mm: tuple[float, float, float]
    radius_mm: float = Field(..., gt=0.0)
    height_mm: float = Field(..., gt=0.0)
    axis: int = Field(2, ge=0, le=2)

    def build(self) -> sh.Shape:
        return sh.Cylinder(
            [c * _MM for c in self.center_mm],
            self.radius_mm * _MM,
            self.height_mm * _MM,
            self.axis,
        )


class UnionConfig(HifusimModel):
    kind: Literal["union"] = "union"
    children: list[ShapeConfig] = Field(..., min_length=2)

    def build(self) -> sh.Shape:
        out = self.children[0].build()
        for child in self.children[1:]:
            out = out | child.build()
        return out


class IntersectionConfig(HifusimModel):
    kind: Literal["intersection"] = "intersection"
    children: list[ShapeConfig] = Field(..., min_length=2)

    def build(self) -> sh.Shape:
        out = self.children[0].build()
        for child in self.children[1:]:
            out = out & child.build()
        return out


class DifferenceConfig(HifusimModel):
    kind: Literal["difference"] = "difference"
    base: ShapeConfig
    subtract: ShapeConfig

    def build(self) -> sh.Shape:
        return self.base.build() - self.subtract.build()


class ComplementConfig(HifusimModel):
    kind: Literal["complement"] = "complement"
    child: ShapeConfig

    def build(self) -> sh.Shape:
        return ~self.child.build()


ShapeConfig = Annotated[
    BallConfig
    | BoxConfig
    | EllipsoidConfig
    | CylinderConfig
    | UnionConfig
    | IntersectionConfig
    | DifferenceConfig
    | ComplementConfig,
    Field(discriminator="kind"),
]


class VolumeImportConfig(HifusimModel):
    """File REFERENCE to an imported label volume (heterogeneous phantom)."""

    format: Literal["npz", "breast_phantom_txt"]
    path: str
    position_mm: tuple[float, ...] | None = None
    ignore_labels: tuple[int, ...] = ()
    resample_dx_mm: float | None = Field(None, gt=0.0)
    resample_method: Literal["nearest", "smooth"] = "nearest"

    def load(self) -> LabelVolume:
        if self.format == "npz":
            vol = LabelVolume.load_npz(self.path)
        else:
            vol = load_breast_phantom(self.path)
        if self.resample_dx_mm is not None:
            vol = vol.resample(self.resample_dx_mm * _MM, method=self.resample_method)
        return vol


class SceneObjectConfig(HifusimModel):
    shape: ShapeConfig
    label: int


class SceneConfig(HifusimModel):
    """Full scene: background + ordered shape assignments + volume imports."""

    ndim: int = Field(..., ge=2, le=3)
    axisymmetric: bool = False
    background: int = 0
    objects: list[SceneObjectConfig] = Field(default_factory=list)
    imports: list[VolumeImportConfig] = Field(default_factory=list)

    def build(self) -> Scene:
        scene = Scene(self.ndim, background=self.background, axisymmetric=self.axisymmetric)
        for imp in self.imports:
            pos = tuple(p * _MM for p in imp.position_mm) if imp.position_mm is not None else None
            scene.add_volume(imp.load(), position=pos, ignore=imp.ignore_labels)
        for obj in self.objects:
            scene.add(obj.shape.build(), obj.label)
        return scene
