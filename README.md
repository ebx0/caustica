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
> architecture plan, and [docs/devlog.md](docs/devlog.md) for the engineering log.

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

Build media from primitives with boolean operators, import heterogeneous label
volumes (e.g. the breast phantom's `mtype`-style text), and resample everything
to YOUR `dx` with a selectable method:

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

## Anatomical phantoms (UWCEM breast repository)

The `uwcem_phantoms` package (repo root, next to `apps/` — a side package, not
part of the `caustica` wheel) turns the [University of Wisconsin numerical breast
phantom repository](https://uwcem.ece.wisc.edu/phantomRepository.html) — nine
MRI-derived phantoms, 0.5 mm isotropic, ACR density classes 1–4 — into **one
file the solvers import directly**:

```python
from uwcem_phantoms import PhantomSpec, build, load_phantom

spec = PhantomSpec(
    phantom_id="012304",                                   # ACR 4, very dense
    f0_mhz=1.0,                                            # alpha power law evaluated here
    simplify={"tissue_model": "detailed"},                 # 10 media numbers, or grouped/simple
    resolution={"dx_mm": 0.3, "method": "smooth"},
    crop={"mode": "breast", "margin_mm": 6},
    domain={"standoff_mm": 15, "fft_friendly": True},      # transducer room + 2/3/5/7-smooth axes
    heterogeneity={"use_pval": True, "noise_pct": 2.0,     # measured + synthetic texture
                   "correlation_mm": 0.6, "seed": 7},
)
path = build(spec).save()                  # -> uwcem_phantoms/_data/exports/<name>.npz

ph     = load_phantom(path)
grid   = ph.grid(pml=hs.PMLSpec(thickness=5e-3))
medium = ph.to_medium()                    # straight into any registry solver
# medium = ph.to_medium(linear=True)       # beta zeroed, for the `linear` solver
```

The export is *also* a plain caustica label volume, so nothing that predates this
module needs changing: `LabelVolume.load_npz(path)` reads the same file, and it
drops into a `Scene` via `add_volume`.

What the knobs actually do:

| group | what it controls |
|---|---|
| tissue model | `detailed` (all 10 media numbers) · `grouped` (5) · `simple` (ids identical to `breast_default()`) |
| resolution | any `dx`, resampled with the label-safe `nearest`/`smooth` rules |
| crop | `breast` (the protruding cone), `tissue` (full box), `manual`, or none |
| domain | coupling standoff / backing / lateral margin, and FFT-friendly padding |
| simplify | drop skin or chest wall, dissolve islands, fill pockets, majority-smooth |
| heterogeneity | the repository's per-voxel `pval`, plus seeded scatterer noise with a physical correlation length |

Attenuation is stored as a power law and evaluated at `f0`, so a phantom is never
silently pinned to the frequency it was built at. Builds that change the physics —
a removed chest wall, a `dx` too coarse for the 1.5 mm skin layer, a `dx` that
cannot carry a wave at `f0` at all — say so in `asset.meta["warnings"]`, which
travels inside the file. `plan(spec)` sizes a build before paying for it, exactly
when no label simplification is requested and as an upper bound when there is.

Verified end to end: a 0.6 mm export of the ACR-4 phantom solved with `westervelt`
puts the focus at z = 54.0 mm where the geometric focus in water would be 51.8 mm —
the 2.2 mm deepening you expect from a beam crossing fat (c = 1475 m/s).

```bash
phantoms.bat                                  # menu launcher: no flags to remember
python -m uwcem_phantoms list                 # the catalog and what is downloaded
python -m uwcem_phantoms fetch --all          # ~179 MB, once
python -m uwcem_phantoms tissues --f0 1.5     # the acoustic table, with sources
python -m uwcem_phantoms build 012304 --dx 0.4 --standoff 15 --pval --noise 2
python -m uwcem_phantoms dataset              # the standard dataset (below)
python -m uwcem_phantoms setup                # nine run-ready setups -> data/setups/
python -m apps.phantom_studio                 # the GUI (see apps/README.md)
```

`phantoms.bat` (`./phantoms.sh` elsewhere) is the shortest way in: it picks the
project virtualenv, offers everything this module does, suggests a `dx` from
the `f0` you chose, and shows the grid, voxel count and peak RAM from
`plan(spec)` before it builds anything. Every wizard ends by printing the
equivalent `python -m uwcem_phantoms build ...` line, so it teaches the flags
rather than hiding them.

### The standard dataset: nine breasts, one grid

`python -m uwcem_phantoms dataset` builds `data/phantoms/` — every phantom at
**0.25 mm** on **one common grid** (560×700×480 = 140×175×120 mm), so a
transducer position, a focus, a slicing script written against one phantom is
valid for all nine:

* the skin front face sits at the same z-plane in every file, with **20 mm of
  coupling water in front** (`--front-gap`) — one standoff fits all. That gap
  is a *transducer budget*, not padding: the PML sponge sits inside the grid,
  and a focused bowl needs room for its own shell in front of the apex. At
  0.25 mm a typical PML is 20 voxels (5 mm) and the production 128-element
  spiral is 11.6 mm deep apex-to-rim, so 20 mm leaves it ~3 mm of clearance;
* the protruding breast's bounding-box centre is on the box centre in x/y
  (±1 voxel — the chest-wall fat slab is excluded on purpose, else "centring"
  would centre the crop window instead of the breast);
* the propagation axis stops at **120 mm** (`--depth`, `0` disables), leaving
  100 mm of tissue behind the 20 mm gap. The uncapped union is 171 mm deep and
  the back of it is coupling water plus the flat chest-wall slab. This is a
  destructive crop, so it is measured, not assumed: every phantom's
  `alignment.truncated_tissue_vox` and `truncated_by_class` record exactly what
  left the domain, `--dry-run` names the cost before the build, and the deeper
  phantoms therefore END inside the body — a back-face PML absorbs into tissue
  rather than water;
* pval heterogeneity is ON (properties blended per media number between its
  literature `lo`/`hi` at the repository's per-voxel `p`), noise OFF, tissue
  model `detailed`, `f0` = 1 MHz;
* each `.npz` is the normal export format: `load_phantom(...).to_medium()` or
  `LabelVolume.load_npz(...)` both work, and `manifest.json` records the grid,
  the per-phantom alignment actually achieved and the citation.

`--verify` re-checks all of it from disk (shapes, alignment planes, the depth
ceiling and its truncation record, water padding values, per-class property
bounds — the "are the pvals right" test).

### Stored setups: nine runs, ready to load

The dataset is the medium and nothing else. A run also needs a transducer that
fits, a boundary that does not swallow it, and a focus that lands in tissue —
`python -m uwcem_phantoms setup` writes that decision down, one small JSON per
phantom in `data/setups/` (in git; ~1.8 KB each):

```python
from uwcem_phantoms.setup import load_setup
from caustica.solvers import registry

s = load_setup("s1-012304")
res = registry.get("westervelt")().run(
    s.grid, s.medium, s.source, spec=s.run_spec,
    record_region=s.record_region, reference_point=s.focus_vox,
    harmonics=(1, 2), backend="cupy",
)
```

The standard set is a **64-element Archimedean spiral, 60 mm aperture, 60 mm
ROC (F/1.0)**, apex fixed at `z = 5.50 mm` — two voxels clear of a 5 mm PML —
focusing at its own geometric focus, `z = 65.50 mm`, with **every drive phase
zero**: no steering, no grating-lobe question, and anatomy is the only thing
that differs between the nine. The cost of a fixed apex is recorded per file:
the on-axis water path runs 16.00–24.75 mm and the focus lands 35.25–44.00 mm
below the skin.

Nothing is baked. The element positions and the 23,283 source voxels are
*derived* at load time and checked against the values the file recorded, so a
change in the array construction fails loudly instead of quietly running a
different transducer. `--verify` goes further and re-measures the shell-to-skin
clearance (9.25–14.50 mm) and the tissue class at the focus from the phantom
itself. Building a setup that would put the transducer inside the patient, or
focus it in water, or drive at a frequency the dataset did not bake `alpha`
for, is refused rather than written.

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
  materials.py, medium.py, sources.py, spectral.py
  analytic/   # Rayleigh, O'Neil, Fubini, cap sampling — the ground-truth layer
  arrays/     # transducer geometry (Archimedean spiral), DAS phasing, voxelization
  geometry/   # CSG shapes, scenes, label-volume import + dx-resampling
  planner/    # pre-run VRAM + wall-time estimates (db | calibrated | measured)
  solvers/    # registry + capability declarations; kspace engine; kwave adapter
  io/         # caustica-result/1 HDF5 contract, atomic writes, float16 quantization,
              # in-run checkpoints, Drive-proof ResultStore
  report/     # focal metrics (single source of truth), <=10 MB preview package,
              # figures + HTML report rendering
  runner.py   # plan-first job execution: disjoint exit codes, heartbeat, resume
  __main__.py # the CLI: python -m caustica {validate | run | report}
uwcem_phantoms/  # UWCEM breast phantom import -> simulation-ready files (side package, not in the wheel)
apps/            # phantom launcher, phantom studio GUI, focus study
data/phantoms/   # the standard aligned dataset (generated; manifest.json in git)
data/setups/     # nine stored, run-ready transducer placements (small JSON, in git)
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
python -m caustica report out/                # local HTML + figures from result.h5
python -m caustica report out/ --preview      # quick look from the <=10 MB preview only
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
.venv/Scripts/python -m ruff check src tests apps uwcem_phantoms
```

Extras: `[gpu]` (CuPy/CUDA 12 backend), `[kwave]` (the k-Wave reference solver),
`[report]` (matplotlib, for `caustica report` figures), `[dev]` (tests + lint +
matplotlib).

License: [MIT](LICENSE).
