# hifusim

[![CI](https://github.com/ebx0/hifusim/actions/workflows/ci.yml/badge.svg)](https://github.com/ebx0/hifusim/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**GPU-accelerated, multi-solver acoustic simulation library for HIFU / therapeutic ultrasound.**
Pure-Python core (NumPy on CPU, CuPy/CUDA on GPU — no precompiled binaries required), designed
to run identically on a local workstation and on Google Colab (T4/L4/A100/H100).

> Status: pre-alpha, under active development. The name `hifusim` is a working name and may
> change before the first public release. See [MILESTONES.md](MILESTONES.md) for the roadmap
> with per-milestone acceptance criteria and current progress, [PLAN.md](PLAN.md) for the
> architecture plan, and [docs/devlog.md](docs/devlog.md) for the engineering log.

## Solvers (one API, a registry of engines)

| name | physics | dims | backend | status |
|---|---|---|---|---|
| `linear` | linear full-wave k-space PSTD | 1/2/3-D | numpy (cupy: M7) | ✅ validated |
| `westervelt` | nonlinear (Westervelt) k-space PSTD, multi-harmonic capture | 1/2/3-D | numpy (cupy: M7) | ✅ validated |
| `kwave` | [k-Wave](http://www.k-wave.org) kspaceFirstOrder via `k-wave-python` (CPU/OMP binary) | 2/3-D | external | ✅ wrapped + cross-validated |
| `kzk` | parabolic KZK (z-marching) | planned | — | M9 |

```python
import numpy as np
import hifusim as hs
import hifusim.solvers as solvers
from hifusim.arrays import archimedean_spiral
from hifusim.materials import water
from hifusim.solvers import CWRunSpec

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
from hifusim.geometry import Ball, Box, Scene, load_breast_phantom
from hifusim.materials import breast_default

scene = Scene(ndim=3, background=4)                     # coupling gel
scene.add((Ball((0, 0, 0.05), 0.04) | Box((0, 0, 0.09), (0.08, 0.08, 0.02)))
          - Ball((0, 0, 0.05), 0.01), label=2)          # CSG: (A | B) - C
phantom = load_breast_phantom("mtype.txt")              # cached as .labels.npz
scene.add_volume(phantom.resample(0.3e-3, method="smooth"), ignore=(4,))
medium = scene.to_medium(grid, breast_default(), supersample=3)
```

2-D, 3-D and 2-D-axisymmetric (r-z half-plane) scenes share one code path;
scenes serialize to JSON (`SceneConfig`) with imported files kept as references.

## Validation

Every solver milestone is gated by tests against **analytic references** (O'Neil 1949 focused
bowl, Rayleigh integral, Fubini nonlinear harmonic growth, exponential absorption, plane-wave
dispersion) **and cross-validated against k-Wave** running as a registry solver on identical
grids/media/sources. Current evidence (all automated, `pytest`):

- plane-wave phase-speed error < 0.1% at 4 ppw; measured absorption within 1% of configured α
- 3-D focused bowl vs O'Neil: focus within 1 voxel, axial correlation r > 0.99, −6 dB widths < 5%
- Westervelt vs Fubini: A2/A1 within 5% (measured 0.9–3.2%) across σ = 0.06–0.61
- `linear` vs `kwave` (real OMP binary), 2-D water: normalized-field correlation r > 0.99

Figure-based comparison reports live under `benchmarks/reports/`.

## Layout

```
src/hifusim/
  core/       # Grid, PML, backend dispatch (numpy|cupy)
  config/     # Pydantic models: strict fields, mm-in / voxels-derived, JSON round-trip
  materials.py, medium.py, sources.py, spectral.py
  analytic/   # Rayleigh, O'Neil, Fubini, cap sampling — the ground-truth layer
  arrays/     # transducer geometry (Archimedean spiral), DAS phasing, voxelization
  solvers/    # registry + capability declarations; kspace engine; kwave adapter
tests/        # pytest; CPU-only by default; kwave/gpu tests auto-skip
scripts/      # validation-report generator
```

## Development

```bash
git clone https://github.com/ebx0/hifusim && cd hifusim
python -m venv .venv
.venv/Scripts/python -m pip install -e .[dev,kwave]   # kwave extra optional
.venv/Scripts/python -m pytest
.venv/Scripts/python -m ruff check src tests
```

License: [MIT](LICENSE).
