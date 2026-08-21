"""The recipe: every knob that turns a repository phantom into a medium.

One pydantic tree, strict fields, JSON round-trip — the same contract the
rest of :mod:`caustica.config` follows. That is what lets the GUI be a thin
layer: the browser edits JSON, posts it back, and the server validates it
with the *same* model the Python API uses. There is no second definition of
what a phantom build means.

User units are millimetres and megahertz throughout; voxel counts are always
derived, never written by hand.

Pipeline order is fixed and load-bearing (see :mod:`uwcem_phantoms.builder`):

    read -> reorient -> simplify (labels) -> crop -> resample -> pad/standoff
         -> properties -> pval -> noise -> export

Simplify runs BEFORE resample because morphology at the native 0.5 mm is what
the operations were designed for; running it after a coarsening would let one
resampled voxel be a whole "island". Crop runs before resample so the
expensive interpolation touches as few voxels as possible. Noise runs last,
at the final dx, because its correlation length is physical.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from caustica.config.models import CausticaModel
from uwcem_phantoms.catalog import PHANTOM_IDS


class SimplifySpec(CausticaModel):
    """Label-domain simplification, applied at the native 0.5 mm spacing."""

    tissue_model: Literal["detailed", "grouped", "simple"] = Field(
        "detailed", description="How many tissue classes survive into the material table"
    )
    drop_muscle: bool = Field(False, description="Replace the chest wall with coupling medium")
    drop_skin: bool = Field(False, description="Replace skin with the tissue behind it")
    keep_largest_only: bool = Field(
        False, description="Discard tissue blobs disconnected from the breast"
    )
    fill_holes: bool = Field(False, description="Fill coupling-medium pockets inside tissue")
    remove_islands_vox: int = Field(
        0, ge=0, description="Dissolve connected components smaller than this many voxels"
    )
    close_skin_iterations: int = Field(
        0, ge=0, le=5, description="Morphological closing passes on the skin layer"
    )
    smooth_iterations: int = Field(
        0, ge=0, le=5, description="Majority-filter passes over the whole label field"
    )

    @property
    def touches_labels(self) -> bool:
        return bool(
            self.drop_muscle
            or self.drop_skin
            or self.keep_largest_only
            or self.fill_holes
            or self.remove_islands_vox
            or self.close_skin_iterations
            or self.smooth_iterations
        )


class ResolutionSpec(CausticaModel):
    """Target spacing and how labels get there."""

    dx_mm: float | None = Field(
        None, gt=0.0, le=10.0, description="Target isotropic spacing [mm]; None keeps 0.5 mm"
    )
    method: Literal["nearest", "smooth"] = Field(
        "smooth", description="Label resampling rule (see LabelVolume.resample)"
    )

    def dx(self) -> float | None:
        return None if self.dx_mm is None else self.dx_mm * 1e-3


class CropSpec(CausticaModel):
    """Which part of the phantom to keep."""

    mode: Literal["none", "tissue", "breast", "manual"] = Field(
        "breast",
        description=(
            "'tissue' = bounding box of everything (the chest-wall fat slab spans the whole "
            "slice, so this barely crops transversally); 'breast' = the protruding breast only; "
            "'manual' = explicit box; 'none' = keep the full phantom"
        ),
    )
    margin_mm: float = Field(5.0, ge=0.0, description="Coupling medium kept around the tissue")
    breast_coverage: float = Field(
        0.9,
        gt=0.0,
        le=1.0,
        description=(
            "'breast' mode: transverse slices whose tissue coverage exceeds this fraction are "
            "treated as chest wall and excluded from the transverse bounding box"
        ),
    )
    start_mm: tuple[float, float, float] | None = Field(
        None, description="manual mode: low corner in canonical (x, y, z) [mm]"
    )
    size_mm: tuple[float, float, float] | None = Field(
        None, description="manual mode: extent in canonical (x, y, z) [mm]"
    )

    @model_validator(mode="after")
    def _check_manual(self) -> CropSpec:
        if self.mode == "manual" and (self.start_mm is None or self.size_mm is None):
            raise ValueError("crop mode 'manual' needs both start_mm and size_mm")
        if self.size_mm is not None and any(s <= 0 for s in self.size_mm):
            raise ValueError(f"size_mm entries must be > 0, got {self.size_mm}")
        return self


class DomainSpec(CausticaModel):
    """Growing the box into a simulation domain."""

    standoff_mm: float = Field(
        0.0,
        ge=0.0,
        description="Coupling medium ADDED in front of the breast (low z) for the transducer",
    )
    backing_mm: float = Field(
        0.0, ge=0.0, description="Coupling medium added behind the chest wall (high z)"
    )
    lateral_margin_mm: float = Field(
        0.0, ge=0.0, description="Coupling medium added on all four transverse sides"
    )
    fft_friendly: bool = Field(
        True, description="Grow each axis to the next 2/3/5/7-smooth size (much faster FFTs)"
    )
    max_voxels: int | None = Field(
        None, gt=0, description="Refuse to build a grid larger than this (safety rail)"
    )


class HeterogeneitySpec(CausticaModel):
    """Within-tissue variation: the measured kind and the synthetic kind."""

    use_pval: bool = Field(
        True, description="Interpolate properties with the repository's per-voxel p value"
    )
    noise_pct: float = Field(
        0.0, ge=0.0, le=50.0, description="Std of the multiplicative scatterer noise [%]"
    )
    correlation_mm: float = Field(
        0.0, ge=0.0, le=20.0, description="Gaussian correlation length of that noise [mm]"
    )
    properties: tuple[Literal["alpha", "rho", "c", "beta"], ...] = Field(
        ("rho", "c"), description="Which property volumes the noise perturbs"
    )
    coupled: bool = Field(True, description="One shared noise field (physical) vs one per property")
    tissue_only: bool = Field(
        True, description="Leave the coupling medium noise-free (it is a water bath)"
    )
    seed: int = Field(0, ge=0, description="RNG seed; recorded in the export for reproducibility")

    @model_validator(mode="after")
    def _check(self) -> HeterogeneitySpec:
        if self.noise_pct > 0 and not self.properties:
            raise ValueError("noise_pct > 0 but no properties selected to perturb")
        return self

    @property
    def produces_continuous_fields(self) -> bool:
        return bool(self.use_pval or self.noise_pct > 0)


class PhantomSpec(CausticaModel):
    """The complete build recipe. This is what the GUI edits and the CLI reads."""

    phantom_id: str = Field(..., description=f"One of {PHANTOM_IDS}")
    f0_mhz: float = Field(
        1.0, gt=0.0, le=50.0, description="Frequency the power-law attenuation is evaluated at"
    )
    simplify: SimplifySpec = Field(default_factory=SimplifySpec)
    resolution: ResolutionSpec = Field(default_factory=ResolutionSpec)
    crop: CropSpec = Field(default_factory=CropSpec)
    domain: DomainSpec = Field(default_factory=DomainSpec)
    heterogeneity: HeterogeneitySpec = Field(default_factory=HeterogeneitySpec)
    name: str | None = Field(None, description="Export basename; defaults to a descriptive one")
    note: str = Field("", description="Free-text note carried into the export metadata")

    @model_validator(mode="after")
    def _check_id(self) -> PhantomSpec:
        if self.phantom_id not in PHANTOM_IDS:
            raise ValueError(f"unknown phantom_id {self.phantom_id!r}; pick from {PHANTOM_IDS}")
        return self

    @property
    def f0(self) -> float:
        """Frequency in Hz."""
        return self.f0_mhz * 1e6

    def default_name(self) -> str:
        dx = self.resolution.dx_mm if self.resolution.dx_mm is not None else 0.5
        parts = [
            f"uwcem{self.phantom_id}",
            f"{self.simplify.tissue_model}",
            f"dx{dx:g}mm",
            f"f{self.f0_mhz:g}MHz",
        ]
        if self.heterogeneity.use_pval:
            parts.append("pval")
        if self.heterogeneity.noise_pct > 0:
            parts.append(f"noise{self.heterogeneity.noise_pct:g}pct")
        return "-".join(parts)

    def export_name(self) -> str:
        return self.name or self.default_name()
