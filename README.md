# caustica

[![CI](https://github.com/ebx0/caustica/actions/workflows/ci.yml/badge.svg)](https://github.com/ebx0/caustica/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-ebx0.github.io%2Fcaustica-1F4E79.svg)](https://ebx0.github.io/caustica/)

**GPU-accelerated, multi-solver acoustic simulation library for HIFU / therapeutic ultrasound.**
Pure-Python core (NumPy on CPU, CuPy/CUDA on GPU — no precompiled binaries required), designed
to run identically on a local workstation and on Google Colab (T4/L4/A100/H100).

📖 **Full documentation: [ebx0.github.io/caustica](https://ebx0.github.io/caustica/)**

> Status: pre-alpha, under active development. Formerly developed under the working name
> `hifusim`; renamed to `caustica` before the first public release (2026-08-21). The CuPy
> backend is packaged and has run on A100 hardware, but its parity and full-size gates are
> not closed yet — every result below is CPU-validated.
> [MILESTONES.md](MILESTONES.md) is the ledger; [PLAN.md](PLAN.md) is the architecture.

## Quickstart

No checkout, no external data:

```bash
pip install "caustica[report] @ git+https://github.com/ebx0/caustica"

caustica example water_bowl_mini      # copies a packaged, self-contained job here
caustica validate water_bowl_mini.json
caustica run water_bowl_mini.json     # seconds on CPU; writes runs/water_bowl_mini/
caustica report runs/water_bowl_mini  # local HTML + figures (needs the [report] extra)
```

Prefix each line with `!` and the same four commands are a Colab session —
or open the notebook, where you edit exactly one line:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebx0/caustica/blob/master/notebooks/colab_run.ipynb)

## How to use it

Ten steps from an empty shell to a focal metric — the same ten decisions whether you write a
job file, call `caustica.simulate()` from Python, or run it in a Colab cell.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/how-to-use-real-dark.svg">
  <img src="docs/assets/how-to-use-real.svg" width="100%"
       alt="How to use caustica, in ten steps: install it; choose the space (1-D, 2-D or 3-D, voxel size, box size, absorbing border); describe what the sound travels through (homogeneous, scene, volume_import, medium_volume); build the geometry from solids; put a transducer in the box (bowl, archimedean spiral, element file); aim it and drive it; choose how much physics you want (linear or westervelt); find out what it will cost before it starts; run it from the command line, Python or Colab; read the result (focal metrics, the -6 dB spot, an HTML report).">
</picture>

<sub>Generated, not drawn: <code>python scripts/make_howto.py</code> rebuilds it from
<a href="scripts/make_howto.py">scripts/make_howto.py</a>, and every thumbnail is made by
calling caustica.</sub>

## One call, from Python

```python
import caustica

res = caustica.simulate("water_bowl_mini.json", solver="westervelt", harmonics=(1, 2))
res.metrics        # focal metrics — the definitions the HTML report quotes
res.result.phasor  # the complex field, as the solver produced it
```

The planner and both pre-run gates apply here exactly as they do to `caustica run`: a job
that will not fit in VRAM, or that a CPU would take hours over, is refused with the same
message and the same exit code. See [Using caustica from
Python](https://ebx0.github.io/caustica/library/).

## Solvers

| name | physics | dims | backend | status |
|---|---|---|---|---|
| `linear` | linear full-wave k-space PSTD | 1/2/3-D | numpy (cupy: M7) | ✅ validated |
| `westervelt` | nonlinear (Westervelt) k-space PSTD, multi-harmonic capture | 1/2/3-D | numpy (cupy: M7) | ✅ validated |
| `kwave` | [k-Wave](http://www.k-wave.org) kspaceFirstOrder via `k-wave-python` | 2/3-D | external | ✅ wrapped + cross-validated |
| `kzk` | parabolic KZK (z-marching) | planned | — | M9 |

Each is gated by tests against analytic references (O'Neil, Rayleigh, Fubini, exponential
absorption, plane-wave dispersion) and cross-validated against k-Wave on identical grids —
[what has been measured](https://ebx0.github.io/caustica/validation/).

## Documentation

Each of these is kept honest by a test:

| page | what it settles |
|---|---|
| [Job reference](https://ebx0.github.io/caustica/job_reference/) | every field of the job file, a working snippet per kind |
| [Conventions that bite](https://ebx0.github.io/caustica/conventions/) | the five assumptions that make a result *silently* wrong |
| [Using caustica from Python](https://ebx0.github.io/caustica/library/) | the library API: solvers, CSG geometry, volume media, planner |
| [What has been measured](https://ebx0.github.io/caustica/validation/) | the analytic and k-Wave evidence, and what is not validated yet |
| [Extension points](https://ebx0.github.io/caustica/extending/) | the five plugin seams and their frozen entry-point names |
| [GUI contract](https://ebx0.github.io/caustica/gui_contract/) | the surface a program driving caustica may rely on |

The same files live under [`docs/`](docs/) if you would rather read them in the repository.

## Development

```bash
git clone https://github.com/ebx0/caustica && cd caustica
python -m venv .venv
.venv/Scripts/python -m pip install -e .[dev,kwave]   # kwave extra optional
.venv/Scripts/python -m pytest
.venv/Scripts/python -m ruff check src tests apps
```

Serve the documentation locally with `pip install -e .[docs]` and
`python -m mkdocs serve`.

## License

MIT — see [LICENSE](LICENSE). Anatomical phantoms are **not** distributed with this
repository; see [anatomical phantoms](docs/uwcem.md) for the UWCEM citation terms.
