# `caustica-job/1` reference

One JSON file describes one complete solve. `caustica validate` checks it
without a GPU, `caustica run` executes it, `caustica report` renders it.

Read [conventions.md](conventions.md) first if you have not: units, the phasor
convention and what `amplitude` means are assumptions this file does not repeat.

The machine-readable version of everything below is:

```bash
caustica schema            # JSON Schema, generated from the models
caustica schema --kinds    # just the registered medium/array kind names
```

Nothing here is hand-maintained twice: the schema comes from the same pydantic
models the runner uses, and a test compares the kind headings in *this* document
against `caustica schema` output so it cannot go stale.

---

## The smallest complete job

A bowl in water, ~15 lines, seconds on a CPU. This is the packaged example
(`caustica example water_bowl_mini`) verbatim:

```json
{
  "format": "caustica-job/1",
  "kind": "explicit",
  "name": "water_bowl_mini",
  "medium": {"kind": "homogeneous"},
  "grid": {
    "ndim": 3,
    "dx_mm": 0.5,
    "size_mm": [18, 18, 24],
    "pml": {"thickness_mm": 3.0}
  },
  "source": {
    "kind": "array",
    "array": {"kind": "bowl", "d_outer_mm": 10.0, "roc_mm": 12.0},
    "apex_mm": [9.0, 9.0, 6.0]
  },
  "drive": {"f0_mhz": 0.8, "amplitude_kpa": 100.0},
  "run": {
    "spec": {"min_settle_periods": 2, "max_settle_periods": 6},
    "harmonics": [1]
  },
  "solver": "linear"
}
```

## Top level

| Field | Default | Meaning |
|---|---|---|
| `format` | `"caustica-job/1"` | Refused if it differs — a wrong format tag is not a warning. |
| `kind` | `"explicit"` | The only job kind: the file describes the whole setup. |
| `name` | **required** | Used for the default output folder (`runs/<name>`). |
| `medium` | **required** | One of the [medium kinds](#medium-kinds). |
| `grid` | `null` | Required **unless** the medium supplies the grid (`medium_volume`). |
| `source` | **required** | See [Source](#source). |
| `drive` | **required** | See [Drive](#drive). |
| `run` | defaults | See [Run](#run). |
| `solver` | `"westervelt"` | `linear`, `westervelt`, `kwave`, or any registered solver. |
| `backend` | `"auto"` | `auto` \| `numpy` \| `cupy`, or any registered backend ([docs/extending.md](extending.md)). `auto` = CuPy if a GPU is present. |
| `output` | defaults | See [Output](#output). |

Unknown keys are **errors**, everywhere, at every level. A typo is never a
silent no-op.

Relative paths (element tables, volumes, the output folder) resolve against the
**job file's directory**, never the current directory — so `caustica run
../jobs/x.json --resume` finds the same checkpoint from anywhere.

---

## Medium kinds

Exactly one `kind` per job. `caustica schema --kinds` lists what is registered
in *your* install, including kinds added by third-party packages.

### `homogeneous`

Uniform medium. The default material is water; give any material explicitly to
change it.

```json
{
  "kind": "homogeneous",
  "material": {
    "name": "water",
    "c": 1500.0,
    "rho": 1000.0,
    "alpha_np_m": 0.025,
    "beta": 3.5
  }
}
```

Omit `material` entirely and you get `water()`: c = 1500 m/s, ρ = 1000 kg/m³,
**α = 0 and β = 0** — a lossless, *linear* reference medium, because the fields
this default exists to check (O'Neil, Rayleigh) are themselves lossless. That
is not real water: the snippet above is the override you want for a nonlinear
water tank (β = 3.5, and α ≈ 0.025 Np/m at 1 MHz). A `westervelt` run against
the bare default is a linear run with extra steps.

### `scene`

COMSOL-style constructive solid geometry rasterized onto the job grid: build
shapes, combine them with booleans, paint each with an integer label, and map
labels to materials here.

```json
{
  "kind": "scene",
  "scene": {
    "ndim": 3,
    "background": 0,
    "objects": [
      {"shape": {"kind": "ball", "center_mm": [9, 9, 14], "radius_mm": 4}, "label": 2}
    ]
  },
  "materials": {
    "0": {"name": "water", "c": 1500.0, "rho": 1000.0, "alpha_np_m": 0.0, "beta": 0.0},
    "2": {"name": "fat", "c": 1450.0, "rho": 932.0, "alpha_np_m": 6.0, "beta": 4.5}
  },
  "supersample": 1
}
```

Shapes: `ball`, `box`, `ellipsoid`, `cylinder`, `halfspace`, plus the operators
`union`, `intersection`, `difference`, `complement` and `transform` (scale /
rotate / translate). `scene.imports` places external label volumes into the same
scene. Every label a scene paints — background included — must have a material
entry, or the job is refused naming the missing labels.

`supersample` (default `1`, max `7`) is the per-axis subdivision used when
rasterizing: `3` evaluates 27 sub-points per voxel and assigns the majority
label. Default 1 because supersampling costs `n³` shape evaluations and matters
only where a curved interface is a fraction of a voxel.

### `volume_import`

One imported label volume placed on the job grid — the shortcut for
"my phantom is a `.npz` of integer labels".

```json
{
  "kind": "volume_import",
  "volume": {
    "format": "npz",
    "path": "phantom.npz",
    "position_mm": [5, 5, 8],
    "ignore_labels": [],
    "resample_dx_mm": null,
    "resample_method": "nearest"
  },
  "materials": "breast_default",
  "background": 0
}
```

The `.npz` is a `LabelVolume`, and it needs **three** arrays — `labels`, `dx`
and `origin`. A file with only `labels` and `dx` fails with
`KeyError: 'origin is not a file in the archive'`, so write it with the
library rather than by hand:

```python
from caustica.geometry import LabelVolume

LabelVolume(labels=labels, dx=0.5e-3, origin=(0.0, 0.0, 0.0)).save_npz("phantom.npz")
```

`materials` is either the string `"breast_default"` (the built-in generic
soft-tissue table) or an explicit `{label: material}` map. `resample_dx_mm`
resamples the volume to another spacing before placement — `nearest` preserves
labels exactly, `smooth` reduces staircasing on curved interfaces. Unlike
`medium_volume`, the **job's** grid wins here: the volume is placed into it.

### `medium_volume`

The one door for a fully specified volume medium — a `.npz` carrying labels + a
material table (or dense per-voxel `c` / `rho` / `alpha` / `beta` arrays).

```json
{
  "kind": "medium_volume",
  "file": "my_medium.npz",
  "pml_mm": 5.0,
  "linear": false,
  "water_label": 0
}
```

**This kind supplies the grid.** Shape and `dx` come from the file, so a job
using it must have **no `grid` section** — only `pml_mm` is yours to choose.
That rule exists so a job can never silently run a resampled ghost of the data
it claims to have run.

Write one from Python:

```python
from caustica.io import write_medium_volume
write_medium_volume("my_medium.npz", dx=0.5e-3, labels=labels, materials=db)
```

Two guards worth knowing:

- If the file records the frequency its absorption was baked at, a job driving
  another `f0` is **refused** (α is frequency-independent in v1 — see
  conventions §2).
- `water_label` (default `0`) is the label treated as coupling water: if the
  focus voxel lands in it, the run is refused, because it would characterize a
  water focus instead of the target. Set it to `null` to disable the check when
  label 0 is not water in your file.

`linear: true` zeroes the nonlinearity, which lets the `linear` solver run a
medium whose table has `beta > 0`.

---

## Source

One shape today: an `array` — a transducer recipe, placed and focused.

```json
{
  "kind": "array",
  "array": {"kind": "bowl", "d_outer_mm": 10.0, "roc_mm": 12.0},
  "apex_mm": [9.0, 9.0, 6.0],
  "focus": {"mode": "natural"},
  "phases_rad": null
}
```

| Field | Default | Meaning |
|---|---|---|
| `array` | **required** | One of the [array kinds](#array-kinds). |
| `apex_mm` | **required** | Apex position in the **grid** frame [mm]; the beam runs +z. |
| `focus.mode` | `"natural"` | `natural` = the array's own geometric focus, all phases zero. `steered` = delay-and-sum phases toward `focus.target_mm`. |
| `focus.target_mm` | `null` | Required for `steered`, refused for `natural`. |
| `phases_rad` | `null` | Explicit per-element phases; overrides `focus.mode`. Multi-element kinds only. |

Steering assumes water on the path (`c₀ = 1500 m/s`). Aberration correction
through tissue is a planning problem, not a job knob.

A run records `derived` geometry (aperture radius, f-number, half angle, how
many elements survived voxelization) into `run_meta.json`. Those numbers are not
inputs — they are re-derived on reload and compared, so a library change that
silently builds a different transducer is caught instead of trusted.

## Array kinds

### `archimedean_spiral`

Elements placed at equal arc length along a spiral wound between an inner and
outer aperture radius, on a spherical shell. The production multi-element
layout.

```json
{
  "kind": "archimedean_spiral",
  "n_elements": 64,
  "d_outer_mm": 100.0,
  "d_inner_mm": 44.0,
  "roc_mm": 100.0,
  "active_fraction": 0.6
}
```

`active_fraction` (default `0.6`) is the fraction of the shell area that is
active surface; it sets the element radius (`area/n` per element). 0.6 is the
production geometry's fill factor — real arrays leave kerf between elements.
`n_elements` defaults to 64; the shipped default geometry is the 128-element
one.

**That snippet is a 100 mm production array** — dropped into the small example
job at the top of this page its focus lands outside the grid (`validate` says
so). A spiral sized for that 18 × 18 × 24 mm grid:

```json
{
  "kind": "archimedean_spiral",
  "n_elements": 16,
  "d_outer_mm": 10.0,
  "d_inner_mm": 4.0,
  "roc_mm": 12.0,
  "active_fraction": 0.6
}
```

Can be steered and phased.

### `bowl`

A single focused spherical cap. One element, so there is nothing to phase: a
`steered` focus or a `phases_rad` list on a bowl is an error telling you to use
a multi-element kind or move the apex.

```json
{"kind": "bowl", "d_outer_mm": 10.0, "roc_mm": 12.0}
```

`d_outer_mm / 2 > roc_mm` (more than a hemisphere) is refused.

### `elements`

**Bring your own transducer.** Explicit element centers, from a file or inline.
Everything is millimetres in the array's apex frame: apex at the origin, beam
axis +z, geometric focus at `(0, 0, roc_mm)`.

From a file — `.npz` with a `positions` array (and optionally `normals`), or a
3/6-column `.csv`:

```json
{
  "kind": "elements",
  "file": "my_array.npz",
  "elem_radius_mm": 1.2,
  "roc_mm": 12.0
}
```

Inline, for a handful of elements:

```json
{
  "kind": "elements",
  "positions_mm": [[4.0, 0.0, 0.69], [0.0, 4.0, 0.69], [-4.0, 0.0, 0.69], [0.0, -4.0, 0.69]],
  "elem_radius_mm": 1.2,
  "roc_mm": 12.0
}
```

Give exactly one of `file` or `positions_mm`. **Normals are optional**: omit
them and every element is aimed at `(0, 0, roc_mm)`, which is what you want for
a conventional focused array. Supply them (`normals_mm` inline, or a `normals`
array / 6-column csv in the file) for a non-focusing layout; they are normalized
for you, so direction vectors of any length are fine.

Writing the file from Python:

```python
import numpy as np
np.savez("my_array.npz", positions=positions_mm)         # (n, 3), millimetres
```

or as csv (a header line is optional, `#` comments are skipped):

```
x,y,z
4.0,0.0,0.69
0.0,4.0,0.69
```

**What a run records about your table.** `run_meta.json`'s `derived` block
carries the aperture numbers (`n_elements`, `r_max_mm`, `shell_depth_mm`,
`f_number`, `half_angle_deg`) *and* `table_sha256`, a digest of the positions
and normals actually used. The digest is the part that matters: aperture
numbers are order statistics, and they survive mirroring the array, rotating
it, re-scattering all but the outermost element, or changing every normal —
each of which moves the field by tens of per cent. A reload compares both, so
"the table under this job changed" is an error rather than a surprise. The
digest is of the *geometry*, not the file, so the same array given inline, as
`.npz` or as `.csv` digests identically.

(The other array kinds need no digest: their geometry is generated from the
handful of numbers already in the job, so pinning the aperture pins the
transducer.)

**Normals set the element plane, not a direction.** Voxelization tilts each
element's disc into the plane its normal defines; the sign is not used, so
`[0,0,1]` and `[0,0,-1]` build the identical source. Do not expect a flipped
normal to mean anything.

Can be steered and phased, exactly like a spiral array. Refusals you may meet:

- `elem_radius_mm >= roc_mm` — that is one element the size of the whole bowl.
- elements more than a metre off-axis — a metres/millimetres mix-up, refused
  rather than run.
- an element that loses all its voxels to deduplication — `dx` is too coarse for
  your element radius or pitch; the message names how many elements were lost.
- the voxelized source overlapping the PML — see conventions §5.

The same table from Python, without a job file:

```python
from caustica.arrays import elements_array, read_element_file

positions_mm, normals = read_element_file("my_array.npz")
array = elements_array(                       # NOTE: the Python API is SI
    positions=positions_mm * 1e-3,
    normals=normals,
    elem_radius=1.2e-3,
    focal_length=12e-3,
)
```

Going the other way — export an array the library built, to edit or reuse as a
table — reads the same four attributes off any `TransducerArray`
(`positions` and `normals`, both `(n, 3)` in metres, plus `elem_radius` and
`focal_length` in metres):

```python
import numpy as np
from caustica.arrays import archimedean_spiral

arr = archimedean_spiral(n_elements=24, d_outer=0.012, d_inner=0.004, roc=0.016)
np.savez("spiral_table.npz", positions=arr.positions * 1e3, normals=arr.normals)
print(arr.elem_radius * 1e3)                  # -> elem_radius_mm for the job
```

The `elements` job kind reading that file produces a **bit-identical** source to
the `archimedean_spiral` kind it came from — the two kinds share one
voxelization path.

---

## Grid

Required unless the medium supplies it.

```json
{
  "ndim": 3,
  "dx_mm": 0.5,
  "size_mm": [18, 18, 24],
  "pml": {"thickness_mm": 3.0, "edge": 2.0}
}
```

| Field | Default | Meaning |
|---|---|---|
| `ndim` | `3` | 1, 2 or 3. Array sources need 3. |
| `dx_mm` | **required** | Isotropic spacing. |
| `size_mm` | **required** | **Total** extent per axis, PML included. |
| `pml.thickness_mm` | `5.0` | Sponge thickness. `0` disables it. |
| `pml.edge` | `2.0` | Gaussian edge factor; larger damps harder at the outer cell. |

Voxel counts are **derived** (`round(size/dx)`) and cannot be written by hand,
so a job can never carry an inconsistent mm/voxel pair. An axis under 4 voxels
is refused.

Choosing `dx`: you need at least 3 points per wavelength **for every harmonic
you record**, at the *lowest* sound speed in the medium. `caustica validate`
prints the ppw at `f0` and warns per harmonic below 3. It warns rather than
blocks because the production setting is a deliberate 1.88 ppw at `2f₀` — under-
resolved second-harmonic amplitude, accepted knowingly.

## Drive

```json
{"f0_mhz": 0.8, "amplitude_kpa": 100.0, "ramp_periods": 3.0}
```

| Field | Default | Meaning |
|---|---|---|
| `f0_mhz` | **required** | Drive frequency [MHz]. |
| `amplitude_kpa` | **required** | Realized source pressure amplitude [kPa] — see conventions §4. |
| `ramp_periods` | `3.0` | Cosine taper at switch-on [periods]. |

`ramp_periods = 3` because a step-on source rings: the taper suppresses the
transient that would otherwise contaminate the settling estimate. Shorter ramps
settle sooner and ring more.

## Run

```json
{
  "spec": {
    "cfl": 0.48,
    "cfl_hard_max": 0.5,
    "min_settle_periods": 8,
    "max_settle_periods": 96,
    "convergence_tol": 0.01,
    "n_record_periods": 2,
    "t_end_min_us": null
  },
  "harmonics": [1, 2],
  "record_region_vox": null
}
```

| Field | Default | Meaning |
|---|---|---|
| `spec.cfl` | `0.48` | Courant number; `dt` is derived from it and `c_max`. |
| `spec.cfl_hard_max` | `0.5` | Refuse anything above this — the k-space PSTD stability edge. |
| `spec.min_settle_periods` | `8` | Never declare convergence before this many periods. |
| `spec.max_settle_periods` | `96` | Give up waiting after this many (recorded as `settle_capped`). |
| `spec.convergence_tol` | `0.01` | Relative change between periods that counts as settled. |
| `spec.n_record_periods` | `2` | Periods averaged into the recorded phasor. |
| `spec.t_end_min_us` | `null` | Force a minimum simulated time, e.g. to let a reflection arrive. |
| `harmonics` | `[1]` | Which harmonics to record. Strictly increasing, must start at 1. |
| `record_region_vox` | `null` | Per-axis `[start, stop]` voxel bounds. `null` records the **full grid**. |

`record_region_vox` is the field that decides whether your result is 40 MB or
4 GB: the record buffer is 8 bytes per voxel **per harmonic**, and the full grid
of a large phantom is a multi-GB `result.h5`. `validate` always prints the
region and its cost; it *warns* only when you left it `null` and the full grid
exceeds 10 million voxels — an explicit region is treated as a decision you
already made, however large. Bounds are in full-grid voxels, PML included.

## Output

```json
{"folder": null, "quantize": true, "max_norm_err": 0.001}
```

| Field | Default | Meaning |
|---|---|---|
| `folder` | `null` | Output folder; `null` → `runs/<name>` next to the job file. |
| `quantize` | `true` | Store the field as float16 + per-slice scale. |
| `max_norm_err` | `0.001` | Refuse quantization if it costs more relative error than this. |

`quantize` defaults on because it halves a multi-GB result at a normalized error
around 1e-4 — well inside the discretization error — and the check makes the
trade-off falsifiable rather than assumed.

The output folder is a fixed layout, and `--resume` depends on it:

```
job.json          normalized copy of the job that ran
plan.json/.txt    the pre-run VRAM + wall-time estimate
status.json       heartbeat (progress, ETA) while running
checkpoint.npz    resume point, deleted on success
result.h5         the field (caustica-result/1)
preview.npz       <=10 MB preview package
metrics.json      focal metrics
run_meta.json     environment, planner-vs-actual, derived geometry
```

---

## Command line

```bash
caustica validate job.json          # schema, files, geometry, PML, focus, ppw (exit 0/2)
caustica validate job.json --fast   # skip medium construction (big volumes)
caustica run job.json --out runs/x  # plan first, refuse OOM, solve, stamp
caustica run job.json --resume      # continue an interrupted run, bit-exact
caustica report runs/x              # REPORT.md + index.html + figures
caustica schema                     # this document, machine-readable
caustica example                    # list the packaged zero-data jobs
```

`run` exit codes are disjoint on purpose, so a queue can react: `0` ok, `2`
config, `3` out of memory, `4` solver failure, `5` interrupted but resumable.

## Extending the schema

Medium kinds and array kinds are a **registry**, not a closed list. A package
can add its own without touching caustica:

```python
# my_pkg/job.py
from typing import Literal
from caustica.config.kinds import MediumKindConfig

class MyPhantomMediumConfig(MediumKindConfig):
    kind: Literal["my_phantom"] = "my_phantom"
    subject: str

    def c_min(self) -> float: ...
    def build(self, grid): ...
```

```toml
# my_pkg/pyproject.toml
[project.entry-points."caustica.medium_kinds"]
my_phantom = "my_pkg.job:MyPhantomMediumConfig"
```

Array kinds use the group `caustica.array_kinds` and subclass `ArrayKindConfig`.
Once installed, `caustica schema` prints your kind, `caustica validate` accepts
it, and jobs using it run — the core kinds register through this same door, so
if it ever breaks, caustica's own schema breaks first.

Import your base class from `caustica.config.kinds`, never from
`caustica.config.job`: the job module builds its unions while it is importing,
and entry points are scanned at that moment.
