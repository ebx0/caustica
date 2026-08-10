# hifusim

**GPU-accelerated, multi-solver acoustic simulation library for HIFU / therapeutic ultrasound.**
Pure-Python core (NumPy on CPU, CuPy/CUDA on GPU — no precompiled binaries), designed to run
identically on a local workstation and on Google Colab (T4/L4/A100/H100).

> Status: pre-alpha, under active development. The name `hifusim` is a working name and may
> change before the first public release. See [MILESTONES.md](MILESTONES.md) for the roadmap
> with per-milestone acceptance criteria, and [PLAN.md](PLAN.md) for the architecture plan.

## Design pillars

- **Multi-solver, one API** — full-wave nonlinear Westervelt k-space PSTD, an optimized linear
  path, and a parabolic KZK solver (planned), selected through a registry with explicit
  capability declarations. Invalid parameter/solver combinations fail loudly at setup time.
- **Dimension-agnostic core** — 1D/2D/3D Cartesian grids share one code path; 2D doubles as the
  cheap CI/testing environment. Axisymmetric grids are planned as a separate transform layer.
- **Analytics-first validation** — O'Neil (1949), Rayleigh integral and Fubini references ship
  inside the library and gate every solver milestone; k-Wave cross-comparisons and ITRUSST
  benchmark corridors follow (see MILESTONES.md, Faz C/E).
- **Config everywhere** — every simulation serializes to a Pydantic/JSON config; physical inputs
  are mm/MHz, voxel counts are always *derived*, never hand-written.
- **Cost before compute** — a planner module (milestone M8) estimates runtime and VRAM per GPU
  *before* a run is launched.

## Layout

```
src/hifusim/
  core/       # Grid, PML, backend dispatch (numpy|cupy)
  config/     # Pydantic models, JSON round-trip, derived quantities
  materials.py, medium.py
  analytic/   # Rayleigh integral, O'Neil bowl, plane-wave & Fubini references
  solvers/    # (M4+) registry, linear & Westervelt PSTD, KZK
  ...
tests/        # pytest; CPU-only by default, GPU tests auto-skip
```

## Development

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .[dev]
.venv/Scripts/python -m pytest
.venv/Scripts/python -m ruff check src tests
```

License: MIT (to be confirmed before first public release).
