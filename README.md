# caustica

[![CI](https://github.com/ebx0/caustica/actions/workflows/ci.yml/badge.svg)](https://github.com/ebx0/caustica/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**GPU-accelerated, multi-solver acoustic simulation library for HIFU / therapeutic ultrasound.**
Pure-Python core (NumPy on CPU, CuPy/CUDA on GPU — no precompiled binaries required), designed
to run identically on a local workstation and on Google Colab (T4/L4/A100/H100).

> Status: pre-alpha, under active development. Formerly developed under the working name
> `hifusim`; renamed to `caustica` before the first public release (2026-08-21).
> See [MILESTONES.md](MILESTONES.md) for the roadmap
> with per-milestone acceptance criteria and current progress, [PLAN.md](PLAN.md) for the
> architecture plan, [docs/job_reference.md](docs/job_reference.md) +
> [docs/conventions.md](docs/conventions.md) for the job format and its conventions, and
> [docs/devlog.md](docs/devlog.md) for the engineering log.

## Quickstart (no checkout, no external data)

```bash
pip install "caustica[report] @ git+https://github.com/ebx0/caustica"

caustica example water_bowl_mini      # copies a packaged, self-contained job here
caustica validate water_bowl_mini.json
caustica run water_bowl_mini.json     # seconds on CPU; writes runs/water_bowl_mini/
caustica report runs/water_bowl_mini  # local HTML + figures (needs the [report] extra)
```

The `example` command *copies* the job out of the install before running —
outputs resolve next to the job file, so running the packaged copy in place
would write into `site-packages`. The `[report]` extra pulls matplotlib for
`caustica report`; everything up to and including `run` needs only the base
install. GPU (CuPy) support is packaged (`pip install "caustica[gpu]"`) but
**not yet verified on real hardware** — every solver result above is
CPU-validated (see [MILESTONES.md](MILESTONES.md), M7).

Runs identically in a Colab cell (prefix each line with `!`); the same four
commands are the whole workflow.

## Run on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebx0/caustica/blob/master/notebooks/colab_run.ipynb)

[`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) is five cells, and **you edit exactly
one line**: `CONFIG`, the job to run. Every bit of behaviour lives in `caustica.colab`, so
improvements arrive with `pip install -U` and the notebook's own diff stays zero — pinned by
`tests/test_colab.py::test_notebook_cells_match_the_frozen_template`.

```python
from caustica.colab import run_job, show

outdir = run_job("my_job.json")   # a path, or an http(s) URL to a caustica-job/1 file
show(outdir)                      # the run's metrics + its report figures, inline
```

`run_job` prints `env_report()` and then **requires a GPU before it prepares anything** — no
download, no folder, no medium build on a runtime that cannot run the job. The refusal names the
fix for the machine you are on, and keeps the two causes apart, because they have two different
fixes: a missing `cupy` is not a CPU runtime, and caustica pip-installs neither for you. After
that it is the ordinary runner — plan first, the VRAM and CPU-time gates, the same output folder,
the same exit codes (carried on `SimulationError.exit_code`).

Output defaults to `/content/runs/<job>`, Colab's session disk. **caustica never mounts Google
Drive**, knows no Drive path and carries no Drive-specific retry logic: if you want a run to
outlive the session, mount your own storage in a cell and pass that folder as `out=`. `/content`
survives a runtime restart — so `resume=True` finishes an interrupted run — but not a VM
teardown.

## One call, from Python

The same job, the same planner, the same gates — without leaving a notebook:

```python
import caustica

res = caustica.simulate(
    "water_bowl_mini.json",   # a job path, a job dict, an ExplicitJobConfig, or a BuiltJob
    solver="westervelt",
    harmonics=(1, 2),
    out=None,                 # None = in memory, nothing written; a path = the full run folder
    progress="auto",          # per-period line + a coarse focal preview every 8 periods
)

res.metrics       # focal metrics (caustica.report.metrics — the definitions REPORT.md quotes)
res.result.phasor # the complex field, as the solver produced it
res.preview()     # the <=10 MB caustica-preview/1 package, in memory
res.save("result.h5")
```

`out=None` writes nothing at all, but it does **not** skip the planner or the
two pre-run gates: a run that will not fit in VRAM, or that a CPU would take
hours over, is refused here exactly as `caustica run` refuses it — with the
same message and the same exit code, carried on `SimulationError.exit_code`.
Give `out=<path>` and the call delegates to the runner, producing the ordinary
output folder (job copy, plan, status, result, preview, stamp).

Progress goes to stderr and turns off with `progress=None`; a callable gets the
raw payload (`period`, `stage`, `peak`, `eta_s`, …) if you would rather draw it
yourself.

## Bring your own setup

These documents are the contract, and each is kept honest by a test:

- **[docs/job_reference.md](docs/job_reference.md)** — every field of the job
  file: each medium kind, each array kind, drive / run / output, with a working
  snippet per kind. `caustica schema` prints the same thing as JSON Schema,
  generated from the models.
- **[docs/conventions.md](docs/conventions.md)** — the five things that make a
  result *silently* wrong if you assume otherwise: the phasor convention
  `p(t) = Re{P·e^(-iωt)}`, Np/m vs dB/cm, what `amplitude` actually means, the
  `+z` beam-axis frame, and that the PML is inside `grid.size_mm`.
- **[docs/extending.md](docs/extending.md)** — the five extension points
  (solver, medium kind, array kind, backend, report renderer), their frozen
  entry-point group names, and a copy-paste plugin package that uses all five.
- **[docs/gui_contract.md](docs/gui_contract.md)** — the surface a GUI (or any
  other program driving caustica) may rely on: the run folder, the exit codes,
  `status.json`, `error.json`, the `cancel` stop signal, and the progress
  payload. Nothing outside that page is a contract.

Edit the packaged example, or start from the two entry points below.

**Your transducer** — an explicit element table (`.npz`, `.csv` or inline),
millimetres in the apex frame; normals optional (omit them and every element
aims at the geometric focus):

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

**Your medium** — a `medium_volume` `.npz` carrying labels + a material table
(or dense per-voxel `c`/`rho`/`alpha`/`beta`). The file fixes shape and `dx`, so
the job carries no `grid` section:

```json
"medium": {"kind": "medium_volume", "file": "my_medium.npz", "pml_mm": 5.0}
```

```python
from caustica.io import write_medium_volume
write_medium_volume("my_medium.npz", dx=0.5e-3, labels=labels, materials=db)
```

`caustica validate my_job.json` checks all of it — schema, files, geometry, PML
clearance, focus placement, points-per-wavelength — before a GPU is booked.

Neither axis is a closed list: medium kinds and array kinds are registries with
entry-point groups (`caustica.medium_kinds`, `caustica.array_kinds`), so a
package can add its own without touching caustica — see the end of the job
reference.

## Solvers (one API, a registry of engines)

| name | physics | dims | backend | status |
|---|---|---|---|---|
| `linear` | linear full-wave k-space PSTD | 1/2/3-D | numpy (cupy: M7) | ✅ validated |
| `westervelt` | nonlinear (Westervelt) k-space PSTD, multi-harmonic capture | 1/2/3-D | numpy (cupy: M7) | ✅ validated |
| `kwave` | [k-Wave](http://www.k-wave.org) kspaceFirstOrder via `k-wave-python` (CPU/OMP binary) | 2/3-D | external | ✅ wrapped + cross-validated |
| `kzk` | parabolic KZK (z-marching) | planned | — | M9 |

```python
import numpy as np
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

## Geometry (COMSOL-style CSG)

Build media from primitives with boolean operators, import heterogeneous
label volumes, and resample everything to YOUR `dx` with a selectable
method:

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

2-D, 3-D and 2-D-axisymmetric (r-z half-plane) scenes share one code path;
scenes serialize to JSON (`SceneConfig`) with imported files kept as references
(CSG trees, affine transforms and half-spaces included).

## Volume media and anatomical phantoms

Volume media enter caustica through ONE door: a `medium_volume` `.npz`
carrying a label map + a `MaterialDB` (or dense per-voxel `c`/`rho`/`alpha`/
`beta` volumes), with the grid shape and `dx` fixed by the file. The library
both reads and writes the format:

```python
from caustica.io import write_medium_volume, load_medium_volume

write_medium_volume("my_medium.npz", dx=0.5e-3, labels=labels, materials=db)
vol = load_medium_volume("my_medium.npz")
medium = vol.to_medium()                 # straight into any registry solver
```

and a job references it without a `grid` section (the file fixes the grid):

```json
{"medium": {"kind": "medium_volume", "file": "my_medium.npz", "pml_mm": 5.0}}
```

**Anatomical phantoms** (the UWCEM breast repository — nine MRI-derived
phantoms, the aligned dataset, nine stored run setups and the Phantom Studio
GUI) live in their own repository, **uwcem-phantom**, which *consumes*
caustica and emits `medium_volume` files plus explicit `caustica-job/1`
JSON. caustica itself carries no phantom-source-specific code — enforced by
`tests/test_import_direction.py`.

## Planner (will it fit? how long will it take?)

Ask BEFORE committing a Colab GPU:

```python
from caustica import planner
print(planner.estimate(grid, medium, src, solver="westervelt", gpu="A100").summary())
print(planner.compare(grid, medium, src))   # every known GPU, one sorted table
planner.calibrate()                         # ~20 real steps on THIS device
```

VRAM comes from a byte-level inventory of the engine's actual buffers (+15%
allocator margin); wall time from `t_step = a·N·log2 N + b·N` with three
sources, always labeled on the result: `db` (datasheet, coarse), `calibrated`
(fitted on-device, persisted to `~/.caustica/calibration.json`), `measured`
(timed right now). Out-of-memory verdicts carry actionable advice: the exact
dx factor that would fit, a smaller record region, the `linear` solver, or a
larger device.

## Validation

Every solver milestone is gated by tests against **analytic references** (O'Neil 1949 focused
bowl, Rayleigh integral, Fubini nonlinear harmonic growth, exponential absorption, plane-wave
dispersion) **and cross-validated against k-Wave** running as a registry solver on identical
grids/media/sources. Current evidence (all automated, `pytest`):

- plane-wave phase-speed error < 0.1% at 4 ppw; measured absorption within 1% of configured α
- 3-D focused bowl vs O'Neil: focus within 1 voxel, axial correlation r > 0.99, −6 dB widths < 5%
- Westervelt vs Fubini: A2/A1 within 5% (measured 0.9–3.2%) across σ = 0.06–0.61
- `linear` vs `kwave` (real OMP binary), 2-D water: normalized-field correlation r > 0.99
- calibrated source amplitude: realized plane amplitude ≈ `source.amplitude` on both the
  native and k-Wave paths, invariant to grid/CFL/remote medium content; one phasor
  convention library-wide (`p(t) = Re{P e^{−iωt}}`, shared with the analytic references)

Figure-based comparison reports live under `benchmarks/reports/`.

## Layout

```
src/caustica/
  core/       # Grid, PML, backend dispatch (numpy|cupy)
  config/     # Pydantic models: strict fields, mm-in / voxels-derived, JSON round-trip
              # + job.py: the caustica-job/1 schema one JSON = one full run
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
              # is prepared, output under /content, no Drive anywhere (M10f)
  progress.py # progress payload presentation (tqdm or plain lines, focal preview)
  __main__.py # the CLI: python -m caustica {validate | run | report | schema | example}
apps/            # focus study (library consumer; not in the wheel)
tests/        # pytest; CPU-only by default; kwave/gpu tests auto-skip
scripts/      # validation-report generator
```

## Command line

```bash
python -m caustica validate job.json          # every check that needs no solve: schema,
                                              # files, geometry, PML, focus, ppw  (exit 0/2)
python -m caustica run job.json --out out/    # plan first, refuse OOM, solve, stamp
                                              # exit: 0 ok - 2 config - 3 OOM - 4 solver
                                              #       5 interrupted-but-resumable
python -m caustica run job.json --resume      # continue an interrupted run bit-exact
python -m caustica run job.json --allow-slow-cpu   # accept a CPU run the 5-min estimate
                                                   # gate would refuse (CAUSTICA_CPU_LIMIT_MIN)
python -m caustica run job.json --preview-only     # skip result.h5: preview + metrics only
python -m caustica run job.json --no-progress      # silence the per-period lines + preview
python -m caustica report out/                # local HTML + figures from result.h5
python -m caustica report out/ --preview      # quick look from the <=10 MB preview only
python -m caustica schema                     # the caustica-job/1 JSON Schema, generated
python -m caustica schema --kinds             # which medium/array kinds are registered
python -m caustica example                    # list the packaged zero-data example jobs
```

On CPU, a native run first prints the plan (wall-time estimate, memory, the
expected `result.h5` size) and **refuses jobs whose estimate exceeds 5
minutes** — `--allow-slow-cpu` accepts the wait, a GPU backend avoids it.
Warnings (low points-per-wavelength, CPU fallback) are `CausticaWarning`s:
filter them with `warnings.filterwarnings(..., category=caustica.CausticaWarning)`
without touching the rest of the ecosystem.

## Development

```bash
git clone https://github.com/ebx0/caustica && cd caustica
python -m venv .venv
.venv/Scripts/python -m pip install -e .[dev,kwave]   # kwave extra optional
.venv/Scripts/python -m pytest
.venv/Scripts/python -m ruff check src tests apps
```

Extras: `[gpu]` (CuPy/CUDA 12 backend), `[kwave]` (the k-Wave reference solver),
`[report]` (matplotlib, for `caustica report` figures), `[dev]` (tests + lint +
matplotlib).

License: [MIT](LICENSE).
