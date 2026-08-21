"""Simulation-ready setups: one phantom + one transducer + one focus, stored.

:mod:`uwcem_phantoms.dataset` gives you nine breasts on one grid. That is the
medium and nothing else — a run also needs a transducer somewhere it fits, an
absorbing boundary that does not swallow the source, a focus that lands in
tissue, and a drive. Rebuilding those four by hand at every call site is how
two "identical" studies quietly stop being comparable.

A setup is that decision, written down::

    from uwcem_phantoms.setup import load_setup
    s = load_setup("s1-012304")
    result = registry.get("westervelt")().run(
        s.grid, s.medium, s.source, spec=s.run_spec,
        record_region=s.record_region, reference_point=s.focus_vox,
        harmonics=(1, 2), backend="cupy",
    )

The JSON files in ``data/setups/`` are the source of truth and are small enough
to read, diff and review. Nothing is baked: the element positions and the voxel
set are DERIVED at load time from the recipe, and the derived values are checked
against the ones the file recorded. A library change that moved an element by a
voxel would therefore fail loudly here instead of silently producing a different
transducer under an unchanged name.

The standard set (see :data:`S1`) is a 64-element Archimedean spiral, 60 mm
aperture, 60 mm radius of curvature — F/1.0. Its apex sits at a FIXED
``z = 5.50 mm`` in all nine files, two voxels clear of a 5 mm PML, and the focus
is the array's own geometric focus at ``z = 65.50 mm``: every drive phase is
zero, so there is no electronic steering, no grating-lobe question, and the
focal gain is whatever the geometry gives. What varies between the nine is
anatomy and only anatomy — which is the point of a common grid.

The price of a fixed apex is an on-axis water path that runs 16.00-24.75 mm and
a focus that lands 35.25-44.00 mm below the skin, both recorded per file.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from caustica.arrays.transducer import TransducerArray, archimedean_spiral
from caustica.core.grid import Grid
from caustica.core.pml import PMLSpec
from caustica.medium import Medium
from caustica.solvers.base import CWRunSpec, check_source_clears_pml
from caustica.sources import CWSource
from uwcem_phantoms.asset import PhantomAsset
from uwcem_phantoms.catalog import PHANTOM_IDS
from uwcem_phantoms.dataset import ACCEPTED_DATASET_FORMATS
from uwcem_phantoms.paths import dataset_dir, ensure_dir
from uwcem_phantoms.paths import setups_dir as _setups_dir

SETUP_FORMAT = "caustica-setup/1"

#: Millimetres of PML on every face. 5 mm is 20 voxels at the dataset's
#: 0.25 mm, and the dataset's 20 mm front gap is chosen to clear it.
PML_MM = 5.0

#: Voxels between the sponge and the array apex. Two is enough to keep every
#: source voxel out of the damped band and small enough that the water path
#: stays honest; :func:`build_setups` re-checks the real clearance anyway.
APEX_CLEARANCE_VOX = 2

#: Source pressure at the element faces [Pa]. Deliberately modest and easy to
#: change per run: in the linear regime the whole field scales with it, and in
#: the Westervelt regime it is exactly the number a study should be choosing on
#: purpose rather than inheriting.
AMPLITUDE_PA = 1.0e5

#: Half-width of the recorded box around the beam axis [mm]: the aperture plus
#: 5 mm. Recording the full 188 Mvox grid would hand back 1.5 GB per harmonic
#: for a beam that never leaves this box.
RECORD_HALF_MM = 35.0

#: Tolerance on a derived array quantity when a stored setup is re-read [mm].
#: These come out of the same closed-form spiral construction every time, so
#: anything above float noise means the library changed under the file.
GEOM_ATOL_MM = 1e-6


class SetupError(RuntimeError):
    """A setup could not be built, or does not match what it recorded."""


@dataclass(frozen=True)
class ArraySpec:
    """A transducer recipe. Everything else about the array is derived."""

    name: str
    label: str
    n_elements: int = 64
    d_outer_mm: float = 60.0
    d_inner_mm: float = 26.4
    roc_mm: float = 60.0
    active_fraction: float = 0.60

    def build(self) -> TransducerArray:
        return archimedean_spiral(
            n_elements=self.n_elements,
            d_outer=self.d_outer_mm * 1e-3,
            d_inner=self.d_inner_mm * 1e-3,
            roc=self.roc_mm * 1e-3,
            active_fraction=self.active_fraction,
        )

    @property
    def f_number(self) -> float:
        return self.roc_mm / self.d_outer_mm

    @property
    def half_angle_deg(self) -> float:
        return float(np.degrees(np.arcsin(min(1.0, (self.d_outer_mm / 2) / self.roc_mm))))

    def derived(self, arr: TransducerArray | None = None) -> dict[str, float]:
        """The numbers a stored file records so a reload can falsify them."""
        arr = arr if arr is not None else self.build()
        r = np.linalg.norm(arr.positions[:, :2], axis=1)
        return {
            "elem_radius_mm": float(arr.elem_radius) * 1e3,
            "shell_depth_mm": float(arr.positions[:, 2].max()) * 1e3,
            "r_max_mm": float(r.max()) * 1e3,
            "f_number": self.f_number,
            "half_angle_deg": self.half_angle_deg,
        }

    def to_json(self) -> dict:
        return {
            "kind": "archimedean_spiral",
            "name": self.name,
            "label": self.label,
            "n_elements": self.n_elements,
            "d_outer_mm": self.d_outer_mm,
            "d_inner_mm": self.d_inner_mm,
            "roc_mm": self.roc_mm,
            "active_fraction": self.active_fraction,
            "derived": self.derived(),
        }

    @classmethod
    def from_json(cls, d: dict) -> ArraySpec:
        if d.get("kind") != "archimedean_spiral":
            raise SetupError(f"unknown array kind {d.get('kind')!r}")
        return cls(
            name=d["name"],
            label=d["label"],
            n_elements=int(d["n_elements"]),
            d_outer_mm=float(d["d_outer_mm"]),
            d_inner_mm=float(d["d_inner_mm"]),
            roc_mm=float(d["roc_mm"]),
            active_fraction=float(d["active_fraction"]),
        )


#: The standard array: 64 elements, 60 mm aperture, 60 mm ROC, F/1.0.
S1 = ArraySpec(name="s1", label="Kompakt / sığ — 64 el, D60, ROC60, F/1.0")


@dataclass
class LoadedSetup:
    """Everything a solver call needs, plus the file it came from."""

    name: str
    grid: Grid
    source: CWSource
    run_spec: CWRunSpec
    record_region: tuple[slice, slice, slice]
    focus_vox: tuple[int, int, int]
    array: TransducerArray
    spec: dict
    medium: Medium | None = None
    _meta: dict = field(default_factory=dict, repr=False)

    @property
    def f0(self) -> float:
        return float(self.spec["drive"]["f0_hz"])

    def summary(self) -> str:
        p, pl, fo = self.spec["phantom"], self.spec["placement"], self.spec["focus"]
        g = self.grid
        return (
            f"{self.name}: {p['id']} (ACR {p['acr_class']}) on {g.shape} @ "
            f"{g.dx * 1e3:g} mm, PML {g.pml_vox} vox\n"
            f"  array {self.spec['array']['label']}\n"
            f"  apex z={pl['apex_mm']:g} mm, water path {pl['water_path_mm']:g} mm, "
            f"shell-skin clearance {pl['min_clearance_mm']:g} mm\n"
            f"  focus z={fo['z_mm']:g} mm ({fo['below_skin_mm']:g} mm below skin), "
            f"tissue class {fo['tissue_class']}, steering none\n"
            f"  drive {self.f0 / 1e6:g} MHz at {self.spec['drive']['amplitude_pa'] / 1e3:g} kPa, "
            f"{self.source.n_points:,} source voxels"
        )


# --------------------------------------------------------------------- paths


#: Re-exported so callers need only import this module.
setups_dir = _setups_dir


def setup_name(array: ArraySpec, phantom_id: str) -> str:
    return f"{array.name}-{phantom_id}"


# ------------------------------------------------------------------ building


def _require_source_fully_clear_of_pml(grid: Grid, src: CWSource) -> int:
    """Every source voxel outside the sponge; returns the tightest margin [vox].

    Stricter than :func:`caustica.solvers.base.check_source_clears_pml` on
    purpose. The solver-level guard has to stay permissive about PARTIAL
    overlap, because a plane source spanning the full transverse extent
    necessarily runs into the lateral band and that is the intended plane-wave
    setup. A curated setup has no such excuse: a focused shell with part of its
    aperture inside the sponge is simply a weaker, wrongly-shaped transducer,
    and the whole point of storing the geometry is that nobody has to notice
    that at run time.
    """
    band = grid.pml_vox
    if band <= 0:
        return 0
    idx = src.indices
    upper = np.asarray(grid.shape) - band
    margin = int(min((idx - band).min(), (upper - 1 - idx).min()))
    if margin < 0:
        inside = ((idx < band) | (idx >= upper)).any(axis=1)
        raise SetupError(
            f"{int(inside.sum())} of {idx.shape[0]} source voxels lie inside the "
            f"{band}-voxel PML band of {grid.shape}. A stored setup must keep the whole "
            f"aperture in the physical domain — move the apex deeper, thin the PML, or "
            f"use a smaller aperture."
        )
    return margin


def _skin_depth_columns(labels: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """First non-water z in each named (x, y) column; ``nz`` where there is none.

    Only the columns the shell actually occupies are gathered, so this costs a
    few tens of MB instead of a full-volume boolean pass over 188 Mvox.
    """
    cols = labels[ix, iy, :] != 0
    nz = labels.shape[2]
    return np.where(cols.any(axis=1), cols.argmax(axis=1), nz)


def build_setups(
    ids: Iterable[str] | None = None,
    array: ArraySpec = S1,
    out_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    f0_hz: float = 1.0e6,
    amplitude_pa: float = AMPLITUDE_PA,
    pml_mm: float = PML_MM,
) -> dict:
    """Measure and write one setup per phantom; returns (and writes) a manifest.

    The measurement is the point. Every file records the clearance between the
    shell and the skin, the water path on the axis, and the tissue class at the
    focus, all measured from the phantom this setup names — so a setup that
    would put the transducer inside the patient cannot be written at all.
    """
    data = Path(data_dir) if data_dir is not None else dataset_dir()
    out = Path(out_dir) if out_dir is not None else setups_dir()
    ensure_dir(out)

    manifest_path = data / "manifest.json"
    if not manifest_path.exists():
        raise SetupError(f"no phantom dataset at {data}; build it first (`dataset`)")
    dman = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dman.get("format") not in ACCEPTED_DATASET_FORMATS:
        raise SetupError(f"{manifest_path}: dataset format {dman.get('format')!r} is not supported")
    if abs(float(dman["f0_mhz"]) * 1e6 - f0_hz) > 1e-6:
        raise SetupError(
            f"the dataset bakes alpha at f0 = {dman['f0_mhz']} MHz but this setup drives at "
            f"{f0_hz / 1e6:g} MHz; absorption would be wrong everywhere. Rebuild the dataset "
            f"with --f0 {f0_hz / 1e6:g}, or drive at {dman['f0_mhz']} MHz."
        )

    by_id = {e["id"]: e for e in dman["phantoms"]}
    wanted = list(ids) if ids is not None else [p for p in PHANTOM_IDS if p in by_id]
    missing = [p for p in wanted if p not in by_id]
    if missing:
        raise SetupError(f"the dataset does not hold {missing}")

    shape = tuple(dman["common_shape"])
    dx_mm = float(dman["dx_mm"])
    dx = dx_mm * 1e-3
    grid = Grid(shape=shape, dx=dx, pml=PMLSpec(thickness=pml_mm * 1e-3))
    cx, cy = shape[0] // 2, shape[1] // 2
    apex_z = grid.pml_vox + APEX_CLEARANCE_VOX

    arr = array.build()
    derived = array.derived(arr)
    asrc = arr.voxelize(grid, (cx, cy, apex_z), f0=f0_hz, amplitude=amplitude_pa)
    src = asrc.source
    check_source_clears_pml(grid, src)
    pml_margin = _require_source_fully_clear_of_pml(grid, src)

    focus_z = apex_z + int(round(array.roc_mm * 1e-3 / dx))
    if not 0 <= focus_z < shape[2]:
        raise SetupError(
            f"the geometric focus lands at z={focus_z} outside a {shape[2]}-voxel axis; "
            f"this array's ROC does not fit the domain"
        )
    half = int(round(RECORD_HALF_MM / dx_mm))
    record = [
        [max(0, cx - half), min(shape[0], cx + half)],
        [max(0, cy - half), min(shape[1], cy + half)],
        [grid.pml_vox, shape[2] - grid.pml_vox],
    ]

    ix, iy = src.indices[:, 0], src.indices[:, 1]
    entries: list[dict] = []
    for pid in wanted:
        entry = by_id[pid]
        path = data / entry["file"]
        asset = PhantomAsset.load(path)
        if asset.shape != shape:
            raise SetupError(f"{path.name}: shape {asset.shape} != dataset grid {shape}")
        labels = asset.labels

        skin = _skin_depth_columns(labels, ix, iy)
        clearance_vox = int((skin - src.indices[:, 2]).min())
        if clearance_vox < 0:
            raise SetupError(
                f"{pid}: the array intersects tissue by {-clearance_vox} voxel(s) "
                f"({-clearance_vox * dx_mm:g} mm); refusing to store this setup"
            )
        axis_col = labels[cx, cy, :]
        nzc = np.flatnonzero(axis_col)
        skin_axis_vox = int(nzc[0]) if nzc.size else shape[2]
        focus_class = int(axis_col[focus_z])
        if focus_class == 0:
            raise SetupError(
                f"{pid}: the geometric focus at z={focus_z * dx_mm:g} mm sits in coupling "
                f"water, not tissue; refusing to store this setup"
            )

        spec = {
            "format": SETUP_FORMAT,
            "name": setup_name(array, pid),
            "phantom": {
                "id": pid,
                "acr_class": entry["acr_class"],
                "file": entry["file"],
                # The tag the dataset on disk ACTUALLY carries (may be a
                # legacy pre-rename tag): load_setup compares it against the
                # npz meta, so recording the constant would break against an
                # older-but-compatible build.
                "dataset_format": dman["format"],
                "shape": list(shape),
                "dx_mm": dx_mm,
            },
            "array": array.to_json(),
            "placement": {
                "apex_vox": [cx, cy, apex_z],
                "apex_mm": round(apex_z * dx_mm, 4),
                "skin_axis_vox": skin_axis_vox,
                "skin_axis_mm": round(skin_axis_vox * dx_mm, 4),
                "water_path_mm": round((skin_axis_vox - apex_z) * dx_mm, 4),
                "min_clearance_vox": clearance_vox,
                "min_clearance_mm": round(clearance_vox * dx_mm, 4),
                "source_voxels": int(src.n_points),
                "elements_represented": int(asrc.n_elements_represented),
                "pml_margin_vox": pml_margin,
            },
            "focus": {
                "mode": "natural",
                "vox": [cx, cy, focus_z],
                "z_mm": round(focus_z * dx_mm, 4),
                "below_skin_mm": round((focus_z - skin_axis_vox) * dx_mm, 4),
                "tissue_class": focus_class,
                "steer_frac_roc": 0.0,
                "phases": "zeros",
            },
            "drive": {
                "f0_hz": f0_hz,
                "amplitude_pa": amplitude_pa,
                "ramp_periods": 3.0,
            },
            "domain": {"pml_mm": pml_mm, "pml_vox": grid.pml_vox},
            "run": {
                "cfl": 0.48,
                "min_settle_periods": 8,
                "max_settle_periods": 96,
                "convergence_tol": 0.01,
                "n_record_periods": 2,
                "harmonics": [1, 2],
            },
            "record_region": record,
        }
        (out / f"{spec['name']}.json").write_text(
            json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        entries.append(
            {
                "name": spec["name"],
                "phantom": pid,
                "acr_class": entry["acr_class"],
                "water_path_mm": spec["placement"]["water_path_mm"],
                "min_clearance_mm": spec["placement"]["min_clearance_mm"],
                "focus_below_skin_mm": spec["focus"]["below_skin_mm"],
                "focus_class": focus_class,
            }
        )
        del asset, labels

    manifest = {
        "format": SETUP_FORMAT,
        "array": array.to_json(),
        "grid": {"shape": list(shape), "dx_mm": dx_mm, "pml_vox": grid.pml_vox},
        "apex_vox": [cx, cy, apex_z],
        "focus_vox": [cx, cy, focus_z],
        "focus_z_mm": round(focus_z * dx_mm, 4),
        "drive": {"f0_hz": f0_hz, "amplitude_pa": amplitude_pa},
        "dataset_manifest": {
            "generated": dman.get("generated"),
            "common_shape": dman["common_shape"],
            "depth_limit_mm": dman.get("depth_limit_mm"),
            "front_gap_mm": dman.get("front_gap_mm"),
        },
        "derived_check": derived,
        "setups": entries,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


# ------------------------------------------------------------------- loading


def setup_names(out_dir: str | Path | None = None) -> list[str]:
    """Names of the stored setups, in manifest order."""
    out = Path(out_dir) if out_dir is not None else setups_dir()
    mp = out / "manifest.json"
    if mp.exists():
        return [e["name"] for e in json.loads(mp.read_text(encoding="utf-8"))["setups"]]
    return sorted(p.stem for p in out.glob("*.json") if p.name != "manifest.json")


def load_setup(
    name: str,
    out_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    with_medium: bool = True,
) -> LoadedSetup:
    """Rebuild a stored setup into solver arguments.

    Nothing is trusted blindly: the array is reconstructed from the recipe and
    its derived geometry is compared against the recorded values, the phantom's
    dataset tag and shape are checked, the voxel count must match, and the
    source must still clear the PML. ``with_medium=False`` skips the ~3 GB of
    property volumes when only the geometry is wanted.
    """
    out = Path(out_dir) if out_dir is not None else setups_dir()
    path = Path(name) if str(name).endswith(".json") else out / f"{name}.json"
    if not path.exists():
        have = setup_names(out)
        raise SetupError(f"no setup {name!r} at {path}; stored: {have}")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("format") != SETUP_FORMAT:
        raise SetupError(f"{path.name}: format {spec.get('format')!r} != {SETUP_FORMAT!r}")

    array = ArraySpec.from_json(spec["array"])
    arr = array.build()
    have = array.derived(arr)
    want = spec["array"]["derived"]
    for k, v in want.items():
        if abs(have[k] - float(v)) > GEOM_ATOL_MM:
            raise SetupError(
                f"{path.name}: rebuilding the array gives {k} = {have[k]:.9g} but the file "
                f"records {float(v):.9g}. The array construction changed under this setup; "
                f"rebuild the setups rather than running a different transducer."
            )

    ph = spec["phantom"]
    shape = tuple(ph["shape"])
    dx = float(ph["dx_mm"]) * 1e-3
    grid = Grid(shape=shape, dx=dx, pml=PMLSpec(thickness=float(spec["domain"]["pml_mm"]) * 1e-3))
    if grid.pml_vox != int(spec["domain"]["pml_vox"]):
        raise SetupError(
            f"{path.name}: PML resolves to {grid.pml_vox} voxels, file says "
            f"{spec['domain']['pml_vox']}"
        )

    apex = tuple(spec["placement"]["apex_vox"])
    drive = spec["drive"]
    asrc = arr.voxelize(
        grid, apex, f0=float(drive["f0_hz"]), amplitude=float(drive["amplitude_pa"])
    )
    src = asrc.source
    if src.n_points != int(spec["placement"]["source_voxels"]):
        raise SetupError(
            f"{path.name}: voxelizing gives {src.n_points} source voxels, file records "
            f"{spec['placement']['source_voxels']}"
        )
    if abs(src.ramp_periods - float(drive["ramp_periods"])) > 1e-9:
        src = CWSource(
            indices=src.indices,
            phases=src.phases,
            amplitude=src.amplitude,
            f0=src.f0,
            ramp_periods=float(drive["ramp_periods"]),
            label=src.label,
        )
    check_source_clears_pml(grid, src)
    margin = _require_source_fully_clear_of_pml(grid, src)
    want_margin = spec["placement"].get("pml_margin_vox")
    if want_margin is not None and margin != int(want_margin):
        raise SetupError(
            f"{path.name}: the source now clears the PML by {margin} voxel(s), file records "
            f"{want_margin}"
        )

    medium = None
    if with_medium:
        data = Path(data_dir) if data_dir is not None else dataset_dir()
        apath = data / ph["file"]
        if not apath.exists():
            raise SetupError(f"{path.name} needs {apath}, which is not on disk")
        asset = PhantomAsset.load(apath)
        dfmt = asset.meta.get("dataset", {}).get("format")
        if dfmt != ph["dataset_format"]:
            raise SetupError(
                f"{apath.name}: dataset format {dfmt!r} != {ph['dataset_format']!r} — this "
                f"setup was measured against a different dataset build"
            )
        if asset.shape != shape:
            raise SetupError(f"{apath.name}: shape {asset.shape} != {shape}")
        medium = asset.to_medium()
        del asset

    run = spec["run"]
    run_spec = CWRunSpec(
        cfl=float(run["cfl"]),
        min_settle_periods=int(run["min_settle_periods"]),
        max_settle_periods=int(run["max_settle_periods"]),
        convergence_tol=float(run["convergence_tol"]),
        n_record_periods=int(run["n_record_periods"]),
    )
    rr = tuple(slice(int(a), int(b)) for a, b in spec["record_region"])
    return LoadedSetup(
        name=spec["name"],
        grid=grid,
        source=src,
        run_spec=run_spec,
        record_region=rr,  # type: ignore[arg-type]
        focus_vox=tuple(int(v) for v in spec["focus"]["vox"]),  # type: ignore[arg-type]
        array=arr,
        spec=spec,
        medium=medium,
        _meta={"path": str(path), "element_of_voxel": asrc.element_of_voxel},
    )


def verify_setups(
    out_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    progress=None,
) -> dict:
    """Re-check every stored setup against the phantoms on disk.

    Loads each one (geometry only, so it stays cheap), then re-measures the
    shell-skin clearance and the tissue class at the focus from the phantom
    itself and compares them to what the file claims.
    """
    say = progress or (lambda _m: None)
    out = Path(out_dir) if out_dir is not None else setups_dir()
    names = setup_names(out)
    if not names:
        raise SetupError(f"no setups at {out}; build them first")

    data = Path(data_dir) if data_dir is not None else dataset_dir()
    report: dict = {"setups": {}, "count": len(names)}
    for name in names:
        say(f"verifying {name} ...")
        s = load_setup(name, out_dir=out, data_dir=data, with_medium=False)
        ph = s.spec["phantom"]
        asset = PhantomAsset.load(data / ph["file"])
        labels = asset.labels
        idx = s.source.indices
        skin = _skin_depth_columns(labels, idx[:, 0], idx[:, 1])
        clear = int((skin - idx[:, 2]).min())
        cx, cy, fz = s.focus_vox
        cls = int(labels[cx, cy, fz])
        want_clear = int(s.spec["placement"]["min_clearance_vox"])
        want_cls = int(s.spec["focus"]["tissue_class"])
        if clear != want_clear:
            raise SetupError(
                f"{name}: measured clearance {clear} voxel(s), file records {want_clear}"
            )
        if cls != want_cls:
            raise SetupError(f"{name}: focus sits in class {cls}, file records {want_cls}")
        if clear < 0:
            raise SetupError(f"{name}: the array intersects tissue")
        report["setups"][name] = {
            "phantom": ph["id"],
            "clearance_vox": clear,
            "clearance_mm": clear * float(ph["dx_mm"]),
            "focus_class": cls,
            "source_voxels": int(s.source.n_points),
            "ok": True,
        }
        del asset, labels
    say(f"{len(names)} setup(s) verified")
    return report


__all__ = [
    "S1",
    "SETUP_FORMAT",
    "ArraySpec",
    "LoadedSetup",
    "SetupError",
    "build_setups",
    "load_setup",
    "setup_names",
    "setups_dir",
    "verify_setups",
]
