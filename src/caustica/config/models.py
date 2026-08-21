"""Base config models.

Rules enforced here (project contract, do not weaken):

1. ``extra="forbid"`` everywhere — a typo'd config key is an error, never a
   silently ignored no-op. (COMSOL feel: fail loudly at setup time.)
2. Users write PHYSICAL quantities (mm, MHz). Voxel counts are DERIVED,
   one-way, and are properties — they cannot be hand-written, so a config
   can never carry an inconsistent mm/voxel pair. This is the lesson of the
   notebook's ``recompute_grid_derived()``.
3. Every model round-trips through JSON losslessly
   (``Model.model_validate_json(m.model_dump_json()) == m``).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from caustica.core.grid import Grid
from caustica.core.pml import PMLSpec


class CausticaModel(BaseModel):
    """Base for all caustica configs: strict fields, assignment validation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PMLConfig(CausticaModel):
    """Sponge-PML in user units (mm)."""

    thickness_mm: float = Field(5.0, ge=0.0, description="Sponge thickness [mm]")
    edge: float = Field(2.0, gt=0.0, description="Gaussian edge factor")

    def to_spec(self) -> PMLSpec:
        return PMLSpec(thickness=self.thickness_mm * 1e-3, edge=self.edge)


class GridConfig(CausticaModel):
    """Grid in user units: physical size [mm] + spacing [mm]; shape is derived.

    ``size_mm`` is the TOTAL physical extent per axis (including any PML that
    the solver will carve out of it). Voxel counts are computed by rounding
    size/dx and are intentionally read-only.
    """

    ndim: int = Field(3, ge=1, le=3)
    dx_mm: float = Field(..., gt=0.0, description="Isotropic spacing [mm]")
    size_mm: tuple[float, ...] = Field(..., description="Physical extent per axis [mm]")
    pml: PMLConfig = Field(default_factory=PMLConfig)

    @model_validator(mode="after")
    def _check_sizes(self) -> GridConfig:
        if len(self.size_mm) != self.ndim:
            raise ValueError(
                f"size_mm has {len(self.size_mm)} entries but ndim={self.ndim}; "
                f"give one physical length per axis."
            )
        if any(s <= 0 for s in self.size_mm):
            raise ValueError(f"size_mm entries must be > 0, got {self.size_mm}")
        shape = self.shape
        if any(n < 4 for n in shape):
            raise ValueError(
                f"derived shape {shape} has an axis below 4 voxels; "
                f"increase size_mm or decrease dx_mm."
            )
        return self

    # ---- derived, one-way (mm -> voxels) ----

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(round(s / self.dx_mm)) for s in self.size_mm)

    @property
    def pml_vox(self) -> int:
        return int(round(self.pml.thickness_mm / self.dx_mm))

    def to_grid(self) -> Grid:
        """Materialize the runtime :class:`~caustica.core.grid.Grid` (SI units)."""
        return Grid(shape=self.shape, dx=self.dx_mm * 1e-3, pml=self.pml.to_spec())
