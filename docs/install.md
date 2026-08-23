# Installation

Pure Python. No compiler, no precompiled binaries, no conda channel — `pip` is the whole
story, and the same wheel runs on a laptop, a workstation and a Colab GPU.

**Python 3.10 or newer**, on Linux, macOS or Windows.

## The usual case

```bash
pip install "caustica[report] @ git+https://github.com/ebx0/caustica"
```

caustica is pre-alpha and not on PyPI yet, so the install goes through git. The `[report]`
extra adds matplotlib, which the HTML report and the figures need; without it the solver
still runs and `caustica run` still writes results, but `caustica report` will tell you what
is missing.

Check what you got:

```bash
caustica --version
python -c "import caustica; print(caustica.env_report())"
```

`env_report()` is the honest answer to "what machine is this": interpreter, platform, whether
a usable CuPy is present, which CUDA it found, how much VRAM. Paste it into a bug report.

## Extras

| Extra | Adds | When you want it |
|---|---|---|
| `report` | matplotlib | figures and the HTML report — install it |
| `gpu` | `cupy-cuda12x` | an NVIDIA GPU with CUDA 12 |
| `kwave` | `k-wave-python` | running [k-Wave](http://www.k-wave.org) as a registry solver |
| `dev` | pytest, ruff, coverage | working on caustica itself |

They compose: `pip install "caustica[report,gpu]"`.

## On a GPU

```bash
pip install "caustica[report,gpu] @ git+https://github.com/ebx0/caustica"
```

`cupy-cuda12x` is the CUDA 12 build. If your driver is CUDA 11, install `cupy-cuda11x`
yourself instead of using the extra — caustica does not pin a CUDA version for you.

!!! warning "What the GPU backend is, and is not, today"

    The CuPy backend is packaged, runs, and has been exercised on A100 hardware. Its
    **parity and full-size gates are not closed yet** (milestone M7 in
    [MILESTONES.md](https://github.com/ebx0/caustica/blob/master/MILESTONES.md)). Everything
    quoted on this site as validated was validated on the CPU. Treat GPU results as
    provisional until those gates close.

Nothing silently falls back. `backend="auto"` means "cupy if this machine has a usable one,
numpy otherwise" — and it says which one it chose. `caustica.require_gpu()` raises instead,
for scripts that must not quietly spend an hour on a CPU.

## On Colab

Prefix the same commands with `!`:

```python
!pip install -q "caustica[report,gpu] @ git+https://github.com/ebx0/caustica"
!caustica example water_bowl_mini
!caustica run water_bowl_mini.json
```

There is a packaged notebook that does this properly — environment verdict first, output on
the session disk, nothing mounted:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebx0/caustica/blob/master/notebooks/colab_run.ipynb)

The notebook's own content does not change between releases: it installs the library and
calls it, so improvements arrive with `pip install -U` and the notebook's diff stays zero.

**caustica never mounts Google Drive.** Output goes to `/content/runs/<job>`, the session
disk. If you want it kept, copy it out yourself — that is a decision about your data, not
one a simulation library should be making.

## Anatomical phantoms

Segmented breast phantoms live in a separate package, deliberately: they carry their own
licence terms and a multi-gigabyte dataset, and neither belongs inside a solver wheel.

```bash
pip install uwcem-phantoms      # not required by caustica
```

Without it, `medium_volume` and `volume_import` still work with any label volume you supply
yourself. See [anatomical phantoms](uwcem.md).

## From a checkout

```bash
git clone https://github.com/ebx0/caustica
cd caustica
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,report]"
pytest
```

The suite is CPU-only by default; GPU and k-Wave tests skip themselves when the hardware or
the binary is absent, so a clean checkout on a laptop should be green. If it is not, that is
a bug — please [say so](contributing.md).

## Uninstalling

```bash
pip uninstall caustica
```

caustica writes nothing outside the directory you point it at, with one exception: a device
calibration recorded by `planner.calibrate()`, which lands in a platform cache directory.
`caustica.planner.default_calibration_path()` prints where.
