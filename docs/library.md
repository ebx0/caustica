# Using caustica from Python

The CLI and the Python API are the same machinery behind two doors: the same
`build_job`, the same planner, the same gates, the same exit codes. This page is
the library side of it.

## One call

The same job, without leaving a notebook:

```python
import caustica

res = caustica.simulate(
    "water_bowl_mini.json",   # a job path, a job dict, an ExplicitJobConfig, or a BuiltJob
    solver="westervelt",
    harmonics=(1, 2),
    out=None,                 # None = in memory, nothing written; a path = the full run folder
    progress="auto",          # per-period line + a coarse focal preview every 8 periods
)

res.metrics       # focal metrics (caustica.report.metrics — the definitions the report quotes)
res.result.phasor # the complex field, as the solver produced it
res.preview()     # the <=10 MB caustica-preview/1 package, in memory
res.save("result.h5")
```

`out=None` writes nothing at all, but it does **not** skip the planner or the two
pre-run gates: a run that will not fit in VRAM, or that a CPU would take hours
over, is refused here exactly as `caustica run` refuses it — with the same
message and the same exit code, carried on `SimulationError.exit_code`. Give
`out=<path>` and the call delegates to the runner, producing the ordinary output
folder (job copy, plan, status, result, preview, stamp).

Progress goes to stderr and turns off with `progress=None`; a callable gets the
raw payload (`period`, `stage`, `peak`, `eta_s`, …) if you would rather draw it
yourself.

## Solvers: one API, a registry of engines

| name | physics | dims | backend | status |
|---|---|---|---|---|
| `linear` | linear full-wave k-space PSTD | 1/2/3-D | numpy (cupy: M7) | validated |
| `westervelt` | nonlinear (Westervelt) k-space PSTD, multi-harmonic capture | 1/2/3-D | numpy (cupy: M7) | validated |
| `kwave` | [k-Wave](http://www.k-wave.org) `kspaceFirstOrder` via `k-wave-python` (CPU/OMP binary) | 2/3-D | external | wrapped + cross-validated |
| `kzk` | parabolic KZK (z-marching) | planned | — | M9 |

Reaching past the job file, straight at the engine:

```python
import caustica as hs
import caustica.solvers as solvers
from caustica.arrays import archimedean_spiral
from caustica.materials import water
from caustica.solvers import CWRunSpec

grid   = hs.Grid(shape=(96, 96, 96), dx=0.5e-3, pml=hs.PMLSpec(thickness=5e-3))
medium = hs.Medium.homogeneous(grid.shape, water())
array  = archimedean_spiral(n_elements=32, d_outer=0.030, d_inner=0.010, roc=0.030)
src    = array.voxelize(grid, apex_vox=(48, 48, 12), f0=1.0e6, amplitude=1e5).source

solver = solvers.get("westervelt")()          # or "linear", "kwave"
res    = solver.run(grid, medium, src, CWRunSpec(), harmonics=(1, 2))
res.amp, res.phase, res.p_max, res.harmonic_amp(2)
```

## Geometry: COMSOL-style CSG

Build media from primitives with boolean operators, import heterogeneous label
volumes, and resample everything to *your* `dx` with a selectable method:

```python
from caustica.geometry import Ball, Box, LabelVolume, Scene
from caustica.materials import breast_default

scene = Scene(ndim=3, background=4)                     # coupling gel
scene.add((Ball((0, 0, 0.05), 0.04) | Box((0, 0, 0.09), (0.08, 0.08, 0.02)))
          - Ball((0, 0, 0.05), 0.01), label=2)          # CSG: (A | B) - C
phantom = LabelVolume.load_npz("phantom.npz")           # any label volume
scene.add_volume(phantom.resample(0.3e-3, method="smooth"), ignore=(4,))
medium = scene.to_medium(grid, breast_default(), supersample=3)
```

2-D, 3-D and 2-D-axisymmetric (r–z half-plane) scenes share one code path; scenes
serialize to JSON (`SceneConfig`) with imported files kept as references (CSG
trees, affine transforms and half-spaces included).

## Volume media

Volume media enter caustica through **one** door: a `medium_volume` `.npz`
carrying a label map plus a `MaterialDB` (or dense per-voxel
`c`/`rho`/`alpha`/`beta` volumes), with the grid shape and `dx` fixed by the
file. The library both reads and writes the format:

```python
from caustica.io import write_medium_volume, load_medium_volume

write_medium_volume("my_medium.npz", dx=0.5e-3, labels=labels, materials=db)
vol = load_medium_volume("my_medium.npz")
medium = vol.to_medium()                 # straight into any registry solver
```

A job references it without a `grid` section, because the file fixes the grid:

```json
{"medium": {"kind": "medium_volume", "file": "my_medium.npz", "pml_mm": 5.0}}
```

Anatomical phantoms live in their own repository — see
[anatomical phantoms](uwcem.md).

## Your own transducer

An explicit element table (`.npz`, `.csv` or inline), millimetres in the apex
frame. Normals are optional: omit them and every element aims at the geometric
focus.

```json
"source": {
  "kind": "array",
  "array": {"kind": "elements", "file": "my_array.npz",
            "elem_radius_mm": 1.2, "roc_mm": 12.0},
  "apex_mm": [9.0, 9.0, 6.0]
}
```

```python
import numpy as np
np.savez("my_array.npz", positions=positions_mm)   # (n, 3); optional: normals=...
```

Neither the medium axis nor the array axis is a closed list: both are registries
with entry-point groups, so a package can add its own kind without touching
caustica. See [extension points](extending.md).

## Planner: will it fit? how long will it take?

Ask **before** committing a Colab GPU:

```python
from caustica import planner

print(planner.estimate(grid, medium, src, solver="westervelt", gpu="A100").summary())
print(planner.compare(grid, medium, src))   # every known GPU, one sorted table
planner.calibrate()                         # ~20 real steps on THIS device
```

VRAM comes from a byte-level inventory of the engine's actual buffers (plus a
15 % allocator margin); wall time from `t_step = a·N·log2 N + b·N` with three
sources, always labelled on the result: `db` (datasheet, coarse), `calibrated`
(fitted on-device, persisted to `~/.caustica/calibration.json`), `measured`
(timed right now). Out-of-memory verdicts carry actionable advice: the exact `dx`
factor that would fit, a smaller record region, the `linear` solver, or a larger
device.

## Where the code lives

```
src/caustica/
  core/       # Grid, PML, backend dispatch (numpy|cupy)
  config/     # Pydantic models: strict fields, mm-in / voxels-derived, JSON round-trip
              # + job.py: the caustica-job/1 schema — one JSON = one full run
              # + kinds.py: medium/array kind registries (entry-point plugin seam)
  materials.py, medium.py, sources.py, spectral.py
  analytic/   # Rayleigh, O'Neil, Fubini, cap sampling — the ground-truth layer
  arrays/     # transducer geometry (spiral, explicit element tables), DAS phasing,
              # voxelization
  geometry/   # CSG shapes, scenes, label-volume import + dx-resampling
  planner/    # pre-run VRAM + wall-time estimates (db | calibrated | measured)
  solvers/    # registry + capability declarations; kspace engine; kwave adapter
  io/         # caustica-result/1 HDF5 contract, atomic writes, float16 quantization,
              # in-run checkpoints, Drive-proof ResultStore, medium_volume format
  report/     # focal metrics (single source of truth), <=10 MB preview package,
              # figures + HTML report rendering
  runner.py   # plan-first job execution: disjoint exit codes, heartbeat, resume
  facade.py   # caustica.simulate(...): one call over the SAME build_job/plan/gates
  colab.py    # caustica.colab: the Colab bridge — environment verdict BEFORE anything
              # is prepared, output under /content, no Drive anywhere
  progress.py # progress payload presentation (tqdm or plain lines, focal preview)
  __main__.py # the CLI: python -m caustica {validate | run | report | schema | example}
apps/         # focus study (library consumer; not in the wheel)
tests/        # pytest; CPU-only by default; kwave/gpu tests auto-skip
scripts/      # validation-report and diagram generators
```

## Warnings

Low points-per-wavelength and CPU fallback are `CausticaWarning`s. Filter them
without touching the rest of the ecosystem:

```python
import warnings
import caustica

warnings.filterwarnings("ignore", category=caustica.CausticaWarning)
```
