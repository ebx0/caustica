"""The job file: ONE JSON that describes a complete solve (``caustica-job/1``).

This is the contract the whole Colab flow rides on (M10b): the user (or,
later, the GUI) writes a job file, ``python -m caustica validate`` checks it
without burning GPU time, and the runner (M10c) executes it. Everything a
run needs is either IN the file or derived from it — nothing is baked.

One job kind: ``explicit`` — the full tree: medium (medium_volume file |
CSG scene | volume import | homogeneous) + grid + array source (spiral |
bowl | explicit element table, natural or steered focus) + drive + run
policy.

The medium and array kinds are NOT a closed union: both are built from the
registries in :mod:`caustica.config.kinds`, which the kinds below register
through — the same door a third-party package uses (M10m/K15). ``caustica
schema`` prints the JSON Schema of whatever is registered right now.

Contract rules (same as every caustica config): pydantic, ``extra="forbid"``
(a typo'd key is an error, never a silent no-op), user units are mm / MHz /
kPa, voxel counts are always derived, and every model round-trips through
JSON losslessly.

History (M10k, 2026-08-22): the ``stored_setup`` job kind and the
``phantom_dataset`` medium kind were REMOVED — a breaking ``caustica-job/1``
change, deliberate and recorded (the format number stays; no stability
guarantee before v1.0). The library carries no phantom-source-specific code;
volume media enter through the generic ``medium_volume`` kind, and the
external phantom tooling emits explicit jobs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, TypeAdapter, model_validator

from caustica.arrays.elements import elements_array, read_element_file
from caustica.arrays.transducer import TransducerArray, archimedean_spiral
from caustica.config.kinds import (
    ArrayKindConfig,
    MediumKindConfig,
    MediumPrep,
    array_kinds,
    medium_kinds,
)
from caustica.config.models import CausticaModel, GridConfig
from caustica.core.grid import Grid
from caustica.core.pml import PMLSpec
from caustica.geometry.configs import SceneConfig, VolumeImportConfig
from caustica.materials import Material, MaterialDB, breast_default, water
from caustica.medium import Medium
from caustica.solvers.base import CWRunSpec, check_source_clears_pml
from caustica.sources import CWSource, bowl_cw_source

JOB_FORMAT = "caustica-job/1"

_MM = 1e-3
#: Geometry tolerance for derived-value re-checks [mm].
GEOM_ATOL_MM = 1e-6
#: Steering phases assume propagation through coupling water (das_phases c0).
STEER_C0 = 1500.0


class JobError(ValueError):
    """A job that parses but cannot be built into a valid run."""


# ------------------------------------------------------------------ drive/run


class DriveConfig(CausticaModel):
    """CW drive in user units. No silent defaults for the physics knobs."""

    f0_mhz: float = Field(..., gt=0.0, description="Drive frequency [MHz]")
    amplitude_kpa: float = Field(..., gt=0.0, description="Source pressure amplitude [kPa]")
    ramp_periods: float = Field(3.0, gt=0.0, description="Cosine-taper ramp [periods]")

    @property
    def f0_hz(self) -> float:
        return self.f0_mhz * 1e6

    @property
    def amplitude_pa(self) -> float:
        return self.amplitude_kpa * 1e3


class RunConfig(CausticaModel):
    """Run policy + what to record."""

    spec: CWRunSpec = Field(default_factory=CWRunSpec)
    harmonics: tuple[int, ...] = (1,)
    record_region_vox: tuple[tuple[int, int], ...] | None = Field(
        None, description="Per-axis (start, stop) voxel bounds; None records the full grid"
    )

    @model_validator(mode="after")
    def _check(self) -> RunConfig:
        h = self.harmonics
        if not h or h[0] != 1 or any(b <= a for a, b in zip(h, h[1:], strict=False)):
            raise ValueError(f"harmonics must be strictly increasing and start at 1, got {h}")
        if self.record_region_vox is not None:
            for ax, (a, b) in enumerate(self.record_region_vox):
                if not 0 <= a < b:
                    raise ValueError(f"record_region_vox axis {ax}: need 0 <= start < stop")
        return self

    def region(self) -> tuple[slice, ...] | None:
        if self.record_region_vox is None:
            return None
        return tuple(slice(int(a), int(b)) for a, b in self.record_region_vox)


class OutputConfig(CausticaModel):
    """Where and how the result lands (consumed by the M10c runner)."""

    folder: str | None = Field(None, description="Output folder; None -> derived from job name")
    quantize: bool = True
    max_norm_err: float = Field(1e-3, gt=0.0)


# --------------------------------------------------------------- medium kinds


@medium_kinds.register
class HomogeneousMediumConfig(MediumKindConfig):
    """Uniform medium (validation runs, water tanks)."""

    kind: Literal["homogeneous"] = "homogeneous"
    material: Material = Field(default_factory=water)

    def build(self, grid: Grid) -> Medium:
        return Medium.homogeneous(grid.shape, self.material)

    def c_min(self) -> float:
        return self.material.c


@medium_kinds.register
class SceneMediumConfig(MediumKindConfig):
    """CSG scene rasterized onto the job grid; labels -> materials here."""

    kind: Literal["scene"] = "scene"
    scene: SceneConfig
    materials: dict[int, Material]
    supersample: int = Field(1, ge=1, le=7)

    @model_validator(mode="after")
    def _check(self) -> SceneMediumConfig:
        labels = {self.scene.background} | {o.label for o in self.scene.objects}
        missing = sorted(labels - set(self.materials))
        if missing:
            raise ValueError(
                f"scene paints labels {missing} that have no material entry "
                f"(materials cover {sorted(self.materials)})"
            )
        return self

    def resolve_paths(self, base_dir: Path | None) -> SceneMediumConfig:
        if base_dir is None or not self.scene.imports:
            return self
        imports = [
            imp.model_copy(update={"path": _resolve(imp.path, base_dir)})
            for imp in self.scene.imports
        ]
        return self.model_copy(update={"scene": self.scene.model_copy(update={"imports": imports})})

    def build(self, grid: Grid) -> Medium:
        db = MaterialDB(materials=self.materials)
        return self.scene.build().to_medium(grid, db, supersample=self.supersample)

    def c_min(self) -> float:
        return min(m.c for m in self.materials.values())


@medium_kinds.register
class VolumeImportMediumConfig(MediumKindConfig):
    """An imported label volume placed on the job grid (mtype-style phantoms)."""

    kind: Literal["volume_import"] = "volume_import"
    volume: VolumeImportConfig
    materials: dict[int, Material] | Literal["breast_default"] = "breast_default"
    background: int = 0

    def _db(self) -> MaterialDB:
        if self.materials == "breast_default":
            return breast_default()
        return MaterialDB(materials=self.materials)

    def resolve_paths(self, base_dir: Path | None) -> VolumeImportMediumConfig:
        if base_dir is None:
            return self
        vol = self.volume.model_copy(update={"path": _resolve(self.volume.path, base_dir)})
        return self.model_copy(update={"volume": vol})

    def build(self, grid: Grid) -> Medium:
        # A volume import IS a one-import scene; sharing that path keeps
        # placement/resampling semantics single-sourced.
        scene = SceneConfig(ndim=grid.ndim, background=self.background, imports=[self.volume])
        return scene.build().to_medium(grid, self._db())

    def c_min(self) -> float:
        return min(m.c for m in self._db().materials.values())


@medium_kinds.register
class MediumVolumeConfig(MediumKindConfig):
    """A caustica ``medium_volume`` file — the ONE door for volume media.

    The grid comes FROM the file (shape + dx are the file's); only the PML
    thickness is chosen here, so an explicit job cannot silently run a
    resampled ghost of the data. (Same rule as the dataset kind it
    generalizes — M10k/D16.)
    """

    provides_grid: ClassVar[bool] = True

    kind: Literal["medium_volume"] = "medium_volume"
    file: str = Field(..., description="Path to the medium-volume npz (absolute or job-relative)")
    pml_mm: float = Field(5.0, ge=0.0)
    linear: bool = Field(False, description="Zero the nonlinearity (volume.to_medium(linear=...))")
    materials: dict[int, Material] | None = Field(
        None,
        description="Override the file's MaterialDB (labels are revalidated against it)",
    )
    water_label: int | None = Field(
        0,
        description="Label treated as coupling water for the focus-in-water refusal; "
        "null disables the check (label 0 means water in caustica's own exports)",
    )

    def resolve_paths(self, base_dir: Path | None) -> MediumVolumeConfig:
        if base_dir is None:
            return self
        return self.model_copy(update={"file": _resolve(self.file, base_dir)})

    def load_volume(self, base_dir: Path | None = None):
        from caustica.io.medium_volume import (  # noqa: PLC0415 (keep import light)
            MediumVolume,
            load_medium_volume,
        )

        path = Path(self.file)
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path
        if not path.exists():
            raise JobError(f"medium volume file not found: {path}")
        vol = load_medium_volume(path)
        if self.materials is not None:
            vol = MediumVolume(
                labels=vol.labels,
                dx=vol.dx,
                materials=MaterialDB(materials=self.materials),
                origin=vol.origin,
                properties=vol.properties,
                meta=vol.meta,
            )
        return vol

    def prepare(self, drive: DriveConfig) -> MediumPrep:
        """Grid + labels now; the (multi-GB) property volumes behind a callable."""
        vol = self.load_volume()
        # The M6f protection generalizes: a file whose alpha was baked at a
        # frequency refuses to run at another one.
        _check_dataset_f0(
            drive.f0_hz,
            vol.meta.get("f0_mhz", vol.meta.get("dataset", {}).get("f0_mhz")),
            "explicit medium_volume job",
        )
        linear = self.linear
        return MediumPrep(
            grid=vol.grid(PMLSpec(thickness=self.pml_mm * _MM) if self.pml_mm > 0 else None),
            c_min=vol.c_min(),
            labels=vol.labels,
            water_label=self.water_label,
            make_medium=lambda: vol.to_medium(linear=linear),
        )


#: Discriminated union of every REGISTERED medium kind (see config.kinds).
MediumConfig = medium_kinds.union()


# ---------------------------------------------------------------- array kinds


class _ElementArrayConfig(ArrayKindConfig):
    """Shared behaviour of every kind that resolves to a multi-element array.

    The spiral recipe and an explicit element table differ only in where the
    positions come from; phasing, voxelization and the derived-geometry
    record are identical, so they live here once.
    """

    def build(self) -> TransducerArray:
        """The transducer this recipe describes (always re-derived, never baked)."""
        raise NotImplementedError(f"{type(self).__name__} must implement build()")

    def _shape_derived(self, arr: TransducerArray) -> dict[str, float]:
        """Aperture numbers every element array can report."""
        r = np.linalg.norm(arr.positions[:, :2], axis=1)
        return {
            "elem_radius_mm": float(arr.elem_radius) * 1e3,
            "shell_depth_mm": float(arr.positions[:, 2].max()) * 1e3,
            "r_max_mm": float(r.max()) * 1e3,
        }

    def build_source(
        self,
        grid: Grid,
        drive: DriveConfig,
        apex_vox: tuple[int, int, int],
        focus: FocusConfig,
        phases_rad: tuple[float, ...] | None,
    ) -> tuple[CWSource, dict[str, Any]]:
        arr = self.build()
        extra: dict[str, Any] = {}
        phases: np.ndarray | None
        if phases_rad is not None:
            if len(phases_rad) != arr.n_elements:
                raise JobError(
                    f"phases_rad has {len(phases_rad)} entries for {arr.n_elements} elements"
                )
            phases = np.asarray(phases_rad, np.float32)
            extra["phases"] = "explicit"
        elif focus.mode == "steered":
            target_m = np.asarray(focus.target_mm, np.float64) * _MM
            apex_m = np.asarray(apex_vox, np.float64) * grid.dx
            phases = arr.das_phases(target_m - apex_m, drive.f0_hz, c0=STEER_C0)
            extra["phases"] = f"das(c0={STEER_C0:g})"
        else:
            phases = None
            extra["phases"] = "zeros"
        asrc = arr.voxelize(
            grid, apex_vox, f0=drive.f0_hz, amplitude=drive.amplitude_pa, phases=phases
        )
        extra.update(self.derived(arr))
        extra["source_voxels"] = int(asrc.source.n_points)
        extra["elements_represented"] = asrc.n_elements_represented
        return asrc.source, extra


@array_kinds.register
class SpiralArrayConfig(_ElementArrayConfig):
    """Archimedean-spiral multi-element array recipe (the production S1 layout)."""

    kind: Literal["archimedean_spiral"] = "archimedean_spiral"
    n_elements: int = Field(64, ge=1)
    d_outer_mm: float = Field(..., gt=0.0)
    d_inner_mm: float = Field(..., ge=0.0)
    roc_mm: float = Field(..., gt=0.0)
    active_fraction: float = Field(0.6, gt=0.0, le=1.0)

    def build(self) -> TransducerArray:
        return archimedean_spiral(
            n_elements=self.n_elements,
            d_outer=self.d_outer_mm * _MM,
            d_inner=self.d_inner_mm * _MM,
            roc=self.roc_mm * _MM,
            active_fraction=self.active_fraction,
        )

    def focal_length_mm(self) -> float:
        return self.roc_mm

    def derived(self, arr: TransducerArray | None = None) -> dict[str, float]:
        """The numbers a stored job output records so a reload can falsify them.

        Generalization of the M6f "nothing is baked" pattern: element
        positions are always re-derived; these values exist to detect a
        library change silently producing a different transducer.
        """
        arr = arr if arr is not None else self.build()
        return {
            **self._shape_derived(arr),
            "f_number": self.roc_mm / self.d_outer_mm,
            "half_angle_deg": float(
                np.degrees(np.arcsin(min(1.0, (self.d_outer_mm / 2) / self.roc_mm)))
            ),
        }


@array_kinds.register
class BowlArrayConfig(ArrayKindConfig):
    """Single focused spherical-cap (bowl) source recipe."""

    kind: Literal["bowl"] = "bowl"
    d_outer_mm: float = Field(..., gt=0.0)
    roc_mm: float = Field(..., gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> BowlArrayConfig:
        if self.d_outer_mm / 2.0 > self.roc_mm:
            raise ValueError(
                f"aperture radius {self.d_outer_mm / 2:g} mm exceeds ROC {self.roc_mm:g} mm "
                f"(more than a hemisphere)"
            )
        return self

    def focal_length_mm(self) -> float:
        return self.roc_mm

    def derived(self) -> dict[str, float]:
        return {
            "aperture_radius_mm": self.d_outer_mm / 2.0,
            "roc_mm": self.roc_mm,
            "f_number": self.roc_mm / self.d_outer_mm,
            "half_angle_deg": float(
                np.degrees(np.arcsin(min(1.0, (self.d_outer_mm / 2) / self.roc_mm)))
            ),
        }

    def build_source(
        self,
        grid: Grid,
        drive: DriveConfig,
        apex_vox: tuple[int, int, int],
        focus: FocusConfig,
        phases_rad: tuple[float, ...] | None,
    ) -> tuple[CWSource, dict[str, Any]]:
        if focus.mode == "steered" or phases_rad is not None:
            raise JobError(
                "a bowl is a single focused element: it cannot be steered or phased. "
                "Use an archimedean_spiral array, or move the bowl's apex."
            )
        src = bowl_cw_source(
            grid,
            f0=drive.f0_hz,
            amplitude=drive.amplitude_pa,
            aperture_radius=self.d_outer_mm / 2.0 * _MM,
            roc=self.roc_mm * _MM,
            apex_vox=apex_vox,
        )
        extra: dict[str, Any] = dict(self.derived())
        extra["source_voxels"] = int(src.n_points)
        return src, extra


@array_kinds.register
class ElementsArrayConfig(_ElementArrayConfig):
    """Bring your own element table: explicit centers (+ optional normals).

    Positions are ``inline`` in the job or read from ``file`` (``.npz`` with
    a ``positions`` array, or a 3/6-column ``.csv``) — exactly one of the
    two. Everything is in MILLIMETRES, in the array's own apex frame: apex
    at the origin, beam axis +z, geometric focus at ``(0, 0, roc_mm)``. Omit
    the normals and every element is aimed at that focus.
    """

    kind: Literal["elements"] = "elements"
    elem_radius_mm: float = Field(..., gt=0.0, description="Circular element radius [mm]")
    roc_mm: float = Field(
        ..., gt=0.0, description="Geometric focal distance from the apex along +z [mm]"
    )
    file: str | None = Field(
        None,
        description="Element table (.npz with 'positions'/optional 'normals', or 3/6-column "
        ".csv); millimetres, apex frame. Absolute, or relative to the job file.",
    )
    positions_mm: tuple[tuple[float, float, float], ...] | None = Field(
        None, description="Inline element centers [mm], apex frame"
    )
    normals_mm: tuple[tuple[float, float, float], ...] | None = Field(
        None,
        description="Inline element normals (direction only, normalized here); "
        "omit to aim every element at (0, 0, roc_mm)",
    )

    @model_validator(mode="after")
    def _check(self) -> ElementsArrayConfig:
        if (self.file is None) == (self.positions_mm is None):
            raise ValueError(
                "an 'elements' array needs exactly one of 'file' or 'positions_mm' "
                "(give the table inline, or point at a .npz/.csv)"
            )
        if self.normals_mm is not None:
            if self.file is not None:
                raise ValueError(
                    "'normals_mm' belongs with inline 'positions_mm'; a file's normals "
                    "come from the file itself"
                )
            if len(self.normals_mm) != len(self.positions_mm or ()):
                raise ValueError(
                    f"normals_mm has {len(self.normals_mm)} entries for "
                    f"{len(self.positions_mm or ())} positions_mm"
                )
        if self.elem_radius_mm >= self.roc_mm:
            raise ValueError(
                f"elem_radius_mm {self.elem_radius_mm:g} is not smaller than roc_mm "
                f"{self.roc_mm:g}: that is a single element the size of the whole bowl"
            )
        return self

    def resolve_paths(self, base_dir: Path | None) -> ElementsArrayConfig:
        if base_dir is None or self.file is None:
            return self
        return self.model_copy(update={"file": _resolve(self.file, base_dir)})

    def _table(self) -> tuple[np.ndarray, np.ndarray | None]:
        if self.file is not None:
            try:
                return read_element_file(self.file)
            except (OSError, ValueError) as exc:
                raise JobError(f"elements array: {exc}") from None
        pos = np.asarray(self.positions_mm, np.float64)
        nrm = None if self.normals_mm is None else np.asarray(self.normals_mm, np.float64)
        return pos, nrm

    def build(self) -> TransducerArray:
        pos_mm, nrm = self._table()
        try:
            return elements_array(
                positions=np.asarray(pos_mm, np.float64) * _MM,
                normals=nrm,
                elem_radius=self.elem_radius_mm * _MM,
                focal_length=self.roc_mm * _MM,
            )
        except ValueError as exc:
            raise JobError(f"elements array: {exc}") from None

    def focal_length_mm(self) -> float:
        return self.roc_mm

    def derived(self, arr: TransducerArray | None = None) -> dict[str, float]:
        """Re-derivable aperture numbers (the table itself stays in its file)."""
        arr = arr if arr is not None else self.build()
        shape = self._shape_derived(arr)
        r_max_mm = shape["r_max_mm"]
        return {
            **shape,
            "n_elements": float(arr.n_elements),
            "f_number": self.roc_mm / (2.0 * r_max_mm) if r_max_mm > 0 else float("inf"),
            "half_angle_deg": float(np.degrees(np.arcsin(min(1.0, r_max_mm / self.roc_mm)))),
        }


#: Discriminated union of every REGISTERED array kind (see config.kinds).
ArrayConfig = array_kinds.union()


class FocusConfig(CausticaModel):
    """Where the beam should land.

    ``natural`` = the array's own geometric focus (all phases zero).
    ``steered`` = delay-and-sum phases toward ``target_mm`` (grid frame, mm).
    Steering assumes water sound speed on the path (das_phases c0 = 1500);
    aberration through tissue is a planning problem (M23), not a job knob.
    """

    mode: Literal["natural", "steered"] = "natural"
    target_mm: tuple[float, float, float] | None = None

    @model_validator(mode="after")
    def _check(self) -> FocusConfig:
        if self.mode == "steered" and self.target_mm is None:
            raise ValueError("focus mode 'steered' requires target_mm")
        if self.mode == "natural" and self.target_mm is not None:
            raise ValueError("focus mode 'natural' takes no target_mm (did you mean 'steered'?)")
        return self


class ArraySourceConfig(CausticaModel):
    """Transducer recipe + placement + focus -> a voxelized CWSource."""

    kind: Literal["array"] = "array"
    array: ArrayConfig
    apex_mm: tuple[float, float, float] = Field(
        ..., description="Apex position in the grid frame [mm]; beam axis is +z"
    )
    focus: FocusConfig = Field(default_factory=FocusConfig)
    phases_rad: tuple[float, ...] | None = Field(
        None,
        description="Explicit per-element phases (multi-element arrays only; overrides focus mode)",
    )

    def resolve_paths(self, base_dir: Path | None) -> ArraySourceConfig:
        """Resolve the array kind's file references against ``base_dir`` (T4)."""
        arr = self.array.resolve_paths(base_dir)
        return self if arr is self.array else self.model_copy(update={"array": arr})

    def _apex_vox(self, grid: Grid) -> tuple[int, int, int]:
        vox = tuple(int(round(a * _MM / grid.dx)) for a in self.apex_mm)
        for v, ax_n in zip(vox, grid.shape, strict=True):
            if not 0 <= v < ax_n:
                raise JobError(
                    f"apex_mm {self.apex_mm} -> voxel {vox} is outside the grid {grid.shape}"
                )
        return vox  # type: ignore[return-value]

    def _focus_vox(self, grid: Grid, apex_vox: tuple[int, int, int]) -> tuple[int, int, int]:
        if self.focus.mode == "steered":
            assert self.focus.target_mm is not None
            vox = tuple(int(round(t * _MM / grid.dx)) for t in self.focus.target_mm)
        else:
            roc_mm = self.array.focal_length_mm()
            vox = (apex_vox[0], apex_vox[1], apex_vox[2] + int(round(roc_mm * _MM / grid.dx)))
        for v, ax_n in zip(vox, grid.shape, strict=True):
            if not 0 <= v < ax_n:
                raise JobError(
                    f"focus voxel {vox} ({self.focus.mode}) is outside the grid {grid.shape}; "
                    f"enlarge the grid or move the apex"
                )
        return vox  # type: ignore[return-value]

    def build(
        self, grid: Grid, drive: DriveConfig
    ) -> tuple[CWSource, tuple[int, int, int], dict[str, Any]]:
        """Voxelize onto ``grid``; returns (source, focus_vox, derived)."""
        if grid.ndim != 3:
            raise JobError(f"array sources need a 3-D grid, got {grid.ndim}-D")
        apex_vox = self._apex_vox(grid)
        focus_vox = self._focus_vox(grid, apex_vox)
        derived: dict[str, Any] = {
            "apex_vox": list(apex_vox),
            "focus_vox": list(focus_vox),
            "focus_mode": self.focus.mode,
        }

        # Each array kind voxelizes itself (config.kinds seam): the extra
        # derived entries land in registration order after focus_mode.
        src, extra = self.array.build_source(grid, drive, apex_vox, self.focus, self.phases_rad)
        derived.update(extra)

        if abs(src.ramp_periods - drive.ramp_periods) > 1e-12:
            src = CWSource(
                indices=src.indices,
                phases=src.phases,
                amplitude=src.amplitude,
                f0=src.f0,
                ramp_periods=drive.ramp_periods,
                label=src.label,
            )
        return src, focus_vox, derived

    def check_derived(self, derived: dict[str, Any], base_dir: Path | None = None) -> None:
        """Falsify recorded derived geometry against a fresh re-derivation.

        The M6f rule generalized: a stored job output that records these
        values can prove the library still builds the SAME transducer. Raises
        :class:`JobError` naming the drifted quantity. Pass ``base_dir`` (the
        job file's directory) when the array kind reads a relative file.
        """
        fresh = self.array.resolve_paths(base_dir).derived()
        for key, want in derived.items():
            if key not in fresh:
                continue
            if not math.isclose(float(want), fresh[key], abs_tol=GEOM_ATOL_MM):
                raise JobError(
                    f"re-deriving the array gives {key} = {fresh[key]:.9g} but the record "
                    f"says {float(want):.9g}. The array construction changed under this "
                    f"job; do not trust the recorded run."
                )


# ------------------------------------------------------------------ job kinds


class ExplicitJobConfig(CausticaModel):
    """A job that describes the whole setup itself."""

    format: Literal["caustica-job/1"] = JOB_FORMAT
    kind: Literal["explicit"] = "explicit"
    name: str = Field(..., min_length=1)
    medium: MediumConfig
    grid: GridConfig | None = Field(
        None, description="Required unless medium is medium_volume (grid comes from the file)"
    )
    source: ArraySourceConfig
    drive: DriveConfig
    run: RunConfig = Field(default_factory=RunConfig)
    solver: str = "westervelt"
    backend: Literal["auto", "numpy", "cupy"] = "auto"
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _grid_rule(self) -> ExplicitJobConfig:
        grid_from_file = type(self.medium).provides_grid
        if grid_from_file and self.grid is not None:
            raise ValueError(
                f"medium '{self.medium.kind}' fixes the grid (shape + dx come from the "
                f"file); remove the job's grid section — only medium.pml_mm is yours "
                f"to choose."
            )
        if not grid_from_file and self.grid is None:
            raise ValueError(f"medium '{self.medium.kind}' requires a grid section")
        return self


#: One job kind since M10k (``stored_setup`` removed — see the module
#: docstring); the alias survives so consumers keep one import site.
JobConfig = ExplicitJobConfig

_JOB_ADAPTER: TypeAdapter = TypeAdapter(JobConfig)


def _rebuild_kind_unions() -> None:
    """Re-derive the medium/array unions after the registries changed.

    Entry-point plugins are discovered while this module is still importing
    (the unions ask for them), so the common case needs no rebuild at all.
    This exists for the other case: a package that calls ``register`` later,
    e.g. from a notebook cell. Both unions are module globals, so a forced
    ``model_rebuild`` re-resolves the annotations that name them.
    """
    global MediumConfig, ArrayConfig, _JOB_ADAPTER
    MediumConfig = medium_kinds.union()
    ArrayConfig = array_kinds.union()
    ArraySourceConfig.model_rebuild(force=True)
    ExplicitJobConfig.model_rebuild(force=True)
    _JOB_ADAPTER = TypeAdapter(JobConfig)


medium_kinds.on_change(_rebuild_kind_unions)
array_kinds.on_change(_rebuild_kind_unions)


def job_schema() -> dict[str, Any]:
    """The ``caustica-job/1`` JSON Schema, generated from the pydantic models.

    There is no second, hand-written definition of the job format anywhere —
    this IS the schema, and it grows a branch the moment a kind registers.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_JOB_ADAPTER.json_schema(ref_template="#/$defs/{model}"),
        "$id": f"urn:caustica:{JOB_FORMAT}",
        "title": JOB_FORMAT,
        "description": (
            "One JSON file = one complete caustica solve. Generated from the "
            "pydantic models; medium and source.array kinds reflect what is "
            "registered right now (see docs/job_reference.md)."
        ),
    }


def load_job(path: str | Path) -> tuple[ExplicitJobConfig, Path]:
    """Parse a job file; returns (config, base_dir for relative paths)."""
    path = Path(path)
    if not path.exists():
        raise JobError(f"no job file at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != JOB_FORMAT:
        raise JobError(f"{path.name}: format {data.get('format')!r} != {JOB_FORMAT!r}")
    return _JOB_ADAPTER.validate_python(data), path.parent


def dump_job(job: ExplicitJobConfig, path: str | Path) -> Path:
    """Write a job file (pretty JSON, atomic)."""
    from caustica.io.atomic import atomic_write  # noqa: PLC0415 (io stays lazy here)

    path = Path(path)
    with atomic_write(path) as tmp:
        tmp.write_text(job.model_dump_json(indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------- build


@dataclass
class BuiltJob:
    """Everything the runner hands to a solver, plus honest provenance."""

    name: str
    grid: Grid
    source: CWSource
    spec: CWRunSpec
    harmonics: tuple[int, ...]
    record_region: tuple[slice, ...] | None
    focus_vox: tuple[int, int, int]
    solver: str
    backend: str
    output: OutputConfig
    job: ExplicitJobConfig
    medium: Medium | None = None
    c_min_hint: float | None = None  # for ppw checks when medium is skipped
    derived: dict[str, Any] = field(default_factory=dict)


def _resolve(path_str: str, base_dir: Path | None) -> str:
    p = Path(path_str)
    if base_dir is not None and not p.is_absolute():
        return str(base_dir / p)
    return path_str


def _check_dataset_f0(job_f0_hz: float, baked_f0_mhz: float | None, what: str) -> None:
    """The M6f alpha guarantee, on EVERY path that can pair a file with a drive.

    A volume file whose absorption (alpha) was baked at one frequency must
    refuse to run at another — anything else silently uses wrong tissue
    losses (adversarial review, 2026-08-19: this guard was once missing on
    the explicit path).
    """
    if baked_f0_mhz is None:
        return
    if abs(job_f0_hz - float(baked_f0_mhz) * 1e6) > 1e-3:
        raise JobError(
            f"{what}: drive f0 = {job_f0_hz / 1e6:g} MHz != the dataset's baked "
            f"{float(baked_f0_mhz):g} MHz. The phantom's absorption (alpha) was baked at "
            f"{float(baked_f0_mhz):g} MHz; running another frequency would silently use "
            f"wrong tissue losses. Rebuild the dataset at the new f0 instead."
        )


def _build_explicit(job: ExplicitJobConfig, base_dir: Path | None, with_medium: bool) -> BuiltJob:
    medium_cfg = job.medium.resolve_paths(base_dir)
    source_cfg = job.source.resolve_paths(base_dir)

    # The medium build is the EXPENSIVE part (GBs for a full-size volume), so
    # every refusal that only needs geometry/labels runs first.
    labels = None
    water_label: int | None = None
    prep: MediumPrep | None = None
    if type(medium_cfg).provides_grid:
        prep = medium_cfg.prepare(job.drive)
        grid, c_min = prep.grid, prep.c_min
        labels, water_label = prep.labels, prep.water_label
    else:
        assert job.grid is not None  # enforced by the model validator
        grid = job.grid.to_grid()
        c_min = medium_cfg.c_min()

    src, focus_vox, derived = source_cfg.build(grid, job.drive)
    check_source_clears_pml(grid, src)

    if labels is not None and water_label is not None and labels[focus_vox] == water_label:
        raise JobError(
            f"the focus voxel {focus_vox} lands in the coupling water "
            f"(label {water_label}), not in tissue — the run would characterize a "
            f"water focus. Move the focus deeper, steer it into the target, or (for "
            f"medium_volume) set water_label to null if label {water_label} is not water."
        )

    if prep is not None:
        # build_medium() drops the kind's own reference to the loaded volume;
        # `labels` is a view into it, so release that too before returning.
        medium = prep.build_medium() if with_medium else None
        labels = None
        del prep
    else:
        medium = medium_cfg.build(grid) if with_medium else None

    return BuiltJob(
        name=job.name,
        grid=grid,
        source=src,
        spec=job.run.spec,
        harmonics=job.run.harmonics,
        record_region=job.run.region(),
        focus_vox=focus_vox,
        solver=job.solver,
        backend=job.backend,
        output=job.output,
        job=job,
        medium=medium,
        c_min_hint=c_min,
        derived=derived,
    )


def build_job(
    job: ExplicitJobConfig,
    base_dir: Path | None = None,
    with_medium: bool = True,
) -> BuiltJob:
    """Materialize a job into solver arguments.

    ``with_medium=False`` skips the expensive property volumes when only the
    geometry is wanted (validation, planning); everything geometric is still
    built and checked for real.
    """
    return _build_explicit(job, base_dir, with_medium)


# ------------------------------------------------------------------- validate


@dataclass
class JobReport:
    """What ``caustica validate`` found. ``errors`` block a run; ``warnings`` do not."""

    job_path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"=== caustica validate: {self.job_path} ==="]
        lines += [f"  {s}" for s in self.summary]
        for w in self.warnings:
            lines.append(f"  ! WARNING: {w}")
        for e in self.errors:
            lines.append(f"  X ERROR: {e}")
        lines.append("OK — job is runnable" if self.ok else "FAILED — fix the errors above")
        return "\n".join(lines)


#: Conservative soft-tissue minimum sound speed for approximate ppw checks
#: when the medium is not loaded (the cheap validation path).
_APPROX_C_MIN = 1450.0


def low_ppw_warnings(grid, f0: float, harmonics, c_min: float, approx_label: str = "") -> list[str]:
    """The low-resolution warnings (< 3 ppw per recorded harmonic), one text.

    Single source for validate, the runner's plan/status/run_meta and the
    report head (M10i/D31): loud in four places, a block in none — the
    production setting is a deliberate 1.88 ppw at 2f0.
    """
    out = []
    for h in harmonics:
        ppw = grid.ppw(h * f0, c_min)
        if ppw < 3.0:
            out.append(
                f"harmonic {h} resolved by only {ppw:.2f} points per wavelength{approx_label} "
                f"(need >= 3): its amplitude will be under-resolved"
            )
    return out


def validate_job(path: str | Path, fast: bool = False) -> JobReport:
    """Everything that can be checked WITHOUT solving (and without a GPU).

    ``fast=True`` skips medium construction for the explicit scene / volume /
    homogeneous kinds — with it goes the solver capability check (nonlinear
    medium vs linear solver). Geometry, files, source-PML clearance, focus
    placement and ppw are always checked.
    """
    report = JobReport(job_path=str(path))
    try:
        job, base_dir = load_job(path)
    except Exception as exc:
        report.errors.append(f"{type(exc).__name__}: {exc}")
        return report

    heavy_medium = isinstance(job.medium, MediumVolumeConfig)
    with_medium = not fast and not heavy_medium
    try:
        built = build_job(job, base_dir=base_dir, with_medium=with_medium)
    except Exception as exc:
        report.errors.append(f"{type(exc).__name__}: {exc}")
        return report

    g, src = built.grid, built.source
    f0 = src.f0
    report.summary += [
        f"job '{built.name}' ({job.kind}) -> solver {built.solver}, backend {built.backend}",
        f"grid {'x'.join(map(str, g.shape))} @ dx={g.dx * 1e3:g} mm, PML {g.pml_vox} vox",
        f"source {src.n_points:,} voxels @ {f0 / 1e6:g} MHz, "
        f"{src.amplitude / 1e3:g} kPa, ramp {src.ramp_periods:g} periods",
        f"focus voxel {built.focus_vox} ({built.derived.get('focus_mode', '?')}), "
        f"harmonics {built.harmonics}",
    ]

    # Recording cost is a CHOICE the user must see: an explicit job silently
    # defaults to the full grid — 188 Mvox on a full-size volume is ~1.5 GiB
    # of complex64 record buffer PER HARMONIC plus a multi-GB result file
    # over Drive.
    rec = built.record_region
    if rec is None:
        n_rec = int(np.prod(g.shape))
        rec_txt = "FULL GRID"
    else:
        n_rec = int(np.prod([sl.stop - sl.start for sl in rec]))
        rec_txt = "x".join(f"{sl.start}:{sl.stop}" for sl in rec)
    per_h_mib = n_rec * 8 / 2**20  # complex64 record buffer
    report.summary.append(
        f"record region: {rec_txt} — {n_rec:,} vox, ~{per_h_mib:,.0f} MiB per harmonic"
    )
    if rec is None and n_rec > 10_000_000:
        report.warnings.append(
            f"recording the FULL grid ({n_rec:,} voxels): ~{per_h_mib:,.0f} MiB of record "
            f"buffer per harmonic and a multi-GB result file. If you only need the focal "
            f"region, set run.record_region_vox."
        )

    # Solver capabilities (needs the medium for the nonlinearity check).
    if built.medium is not None:
        try:
            import caustica.solvers as solvers  # noqa: PLC0415

            solvers.get(built.solver)().validate(g, built.medium, src)
        except KeyError as exc:
            report.errors.append(str(exc))
        except Exception as exc:
            report.errors.append(f"{type(exc).__name__}: {exc}")
    else:
        try:
            import caustica.solvers as solvers  # noqa: PLC0415

            solvers.get(built.solver)
            report.warnings.append(
                "medium not loaded (fast/heavy path): solver capability check "
                "(nonlinearity, medium shape) deferred to run time"
            )
        except KeyError as exc:
            report.errors.append(str(exc))

    # Spatial resolution: every recorded harmonic needs >= 3 ppw at ITS OWN
    # frequency (the silent killer of multi-harmonic runs).
    c_min = built.c_min_hint
    approx = ""
    if c_min is None and built.medium is not None:
        c_min = float(built.medium.c_min)
    if c_min is None:
        c_min, approx = _APPROX_C_MIN, " (approx. c_min)"
    report.warnings.extend(low_ppw_warnings(g, f0, built.harmonics, c_min, approx))
    report.summary.append(f"ppw at f0: {g.ppw(f0, c_min):.2f}{approx}")
    return report
