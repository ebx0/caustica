# Extending caustica

caustica has **five** extension points. Each one is a name → implementation
registry with an `importlib.metadata` entry-point group behind it, so a package
you install can add to it without a single change to caustica's source.

| Axis | Entry-point group | What you write | Registry |
|---|---|---|---|
| Solver | `caustica.solvers` | a `SolverBase` subclass | `caustica.solvers.registry.solver_registry` |
| Medium kind | `caustica.medium_kinds` | a `MediumKindConfig` subclass | `caustica.config.kinds.medium_kinds` |
| Array kind | `caustica.array_kinds` | an `ArrayKindConfig` subclass | `caustica.config.kinds.array_kinds` |
| Backend | `caustica.backends` | a function returning a `Backend` | `caustica.core.backend.backends` |
| Report renderer | `caustica.report_renderers` | a function that renders an output folder | `caustica.report.renderers.report_renderers` |

**These group names are frozen.** They are part of the public contract; a plugin
written against caustica 0.1 keeps loading. A test pins the tuple
(`tests/test_plugins.py::test_entry_point_group_names_are_frozen`).

**caustica's own implementations register through these same doors.** `numpy`
and `cupy` are two registered backend factories; `matplotlib` is a registered
renderer; `linear`, `westervelt` and `kwave` are registered solvers;
`homogeneous` / `scene` / `volume_import` / `medium_volume` and
`archimedean_spiral` / `bowl` / `elements` are registered kinds. There is no
private path, which is what keeps the seam honest: if registration breaks,
caustica breaks before your plugin does (PLAN.md §2 rule 6, K15).

## How discovery works

* **Lazy.** A registry scans its entry-point group the first time somebody asks
  it a question (`get`, `available`, or — for kinds — the job schema). `import
  caustica` never pays for a metadata sweep.
* **Once.** The scan happens at most once per process, even if it fails.
* **Never fatal.** A plugin that raises on import is logged at WARNING
  (`logging.getLogger("caustica")`) and skipped. Your broken package cannot
  break somebody else's run.
* **Two of the five keys come from the implementation, not from you.** A kind's
  key is its `kind` field's `Literal`; a solver's key is the class's `name`
  attribute. So the left-hand side of the entry-point line is free-form there.
  For backends and renderers the entry-point name **is** the key — it is the
  string a user writes in a job file or passes to `--renderer`.
* **In-process registration works too**, for a notebook or a test:
  `medium_kinds.register(MyKind)`, `backends.register("mine")(factory)`. It is
  the same door; entry points just save you the import.

An unregistered name is always an error that lists what *is* registered and
names the group to register through:

```
unknown solver 'kzk'. Available: kwave, linear, westervelt. Third-party solvers
are added through the 'caustica.solvers' entry-point group.
```

## A skeleton package

Two files. `pip install -e .` and caustica picks all five up.

**`pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "caustica-myext"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["caustica"]

[project.entry-points."caustica.solvers"]
# left-hand name is free: the registry key is the class's `name` attribute
my_solver = "caustica_myext:MySolver"

[project.entry-points."caustica.medium_kinds"]
# ...also free: the key is the `kind` Literal ("my_gel")
my_gel = "caustica_myext:GelMediumConfig"

[project.entry-points."caustica.array_kinds"]
my_ring = "caustica_myext:RingArrayConfig"

[project.entry-points."caustica.backends"]
# HERE the name matters: this is what `backend: "my_backend"` in a job means
my_backend = "caustica_myext:make_backend"

[project.entry-points."caustica.report_renderers"]
# ...and this is what `caustica report --renderer my_report` means
my_report = "caustica_myext:render_report"

[tool.setuptools]
py-modules = ["caustica_myext"]
```

**`caustica_myext.py`**

```python
"""One package, all five axes."""

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from caustica.arrays import elements_array
from caustica.config.kinds import ArrayKindConfig, MediumKindConfig
from caustica.core.backend import Backend
from caustica.materials import Material
from caustica.medium import Medium
from caustica.solvers.kspace.linear import LinearKSpacePSTD


# ---------------------------------------------------------------- medium kind
class GelMediumConfig(MediumKindConfig):
    """A uniform coupling gel. `medium: {"kind": "my_gel", "c": 1520}`."""

    kind: Literal["my_gel"] = "my_gel"
    c: float = Field(1520.0, gt=0.0)

    def c_min(self) -> float:
        """Lowest sound speed you can paint — drives the ppw warning."""
        return self.c

    def build(self, grid):
        mat = Material(name="my_gel", c=self.c, rho=1020.0, alpha_np_m=1.0, beta=0.0)
        return Medium.homogeneous(grid.shape, mat)


# ----------------------------------------------------------------- array kind
class RingArrayConfig(ArrayKindConfig):
    """A flat ring of elements. `source.array: {"kind": "my_ring", ...}`."""

    kind: Literal["my_ring"] = "my_ring"
    n_elements: int = Field(6, ge=1)
    ring_radius_mm: float = Field(4.0, gt=0.0)
    elem_radius_mm: float = Field(1.2, gt=0.0)
    roc_mm: float = Field(12.0, gt=0.0)

    def focal_length_mm(self) -> float:
        """Where a `natural` focus lands, measured from the apex along +z."""
        return self.roc_mm

    def derived(self) -> dict:
        """Numbers a reload re-derives and compares — the M6f "nothing is baked" rule."""
        return {"n_elements": float(self.n_elements), "ring_radius_mm": self.ring_radius_mm}

    def build_source(self, grid, drive, apex_vox, focus, phases_rad):
        th = np.linspace(0.0, 2.0 * np.pi, self.n_elements, endpoint=False)
        r = self.ring_radius_mm * 1e-3
        positions = np.column_stack((r * np.cos(th), r * np.sin(th), np.zeros_like(th)))
        array = elements_array(
            positions=positions,
            elem_radius=self.elem_radius_mm * 1e-3,
            focal_length=self.roc_mm * 1e-3,
        )
        placed = array.voxelize(
            grid, apex_vox, f0=drive.f0_hz, amplitude=drive.amplitude_pa, phases=phases_rad
        )
        return placed.source, dict(self.derived())


# --------------------------------------------------------------------- solver
class MySolver(LinearKSpacePSTD):
    """`solver: "my_linear"` in a job file. Subclass or write from scratch."""

    name = "my_linear"


# -------------------------------------------------------------------- backend
def make_backend() -> Backend:
    """`backend: "my_backend"`. A factory, not a class — called on demand."""
    return Backend("my_backend", np)


# ------------------------------------------------------------ report renderer
def render_report(outdir, *, preview_only: bool = False) -> Path:
    """`caustica report <dir> --renderer my_report`. Returns the path to open."""
    outdir = Path(outdir)
    out = outdir / "REPORT.txt"
    out.write_text(f"my report for {outdir.name}\n", encoding="utf-8")
    return out
```

A live, tested version of exactly this package is `PLUGIN_SRC` in
`tests/test_plugins.py`; the gate
`test_entry_point_plugin_extends_all_five_axes` installs it and runs two jobs
and a render through it.

## Axis by axis

### Solver — `caustica.solvers`

Subclass `caustica.solvers.SolverBase`, set `name` (the registry key) and
`caps` (a `SolverCaps`: `ndim`, `nonlinear`, `drive`, `backends`,
`absorption`). `validate(grid, medium, source)` is checked **before** any
compute, so an unsupported setup fails at setup time with an explanation
instead of quietly producing nonsense. `run(grid, medium, source, spec,
**options)` returns a `SolverResult`.

Reject options you do not understand — the native solvers end `run()` with
`raise TypeError(f"unknown run() options: ...")` — so a caller who passes
`checkpoint=` to a solver that cannot checkpoint hears about it.

**Known limitation.** `caustica run` forwards `backend=` and `checkpoint=`
only to the solvers in `caustica.runner._NATIVE_SOLVERS` (`linear`,
`westervelt`), because the external k-Wave adapter rejects unknown kwargs by
contract. A third-party solver is treated like the external one: it is called
without `backend=`, so it resolves its own backend, and it gets no checkpoint.
Declare what you accept in `caps.backends` and resolve the backend yourself
with `get_backend(...)` until this whitelist becomes a capability question.

### Medium kind — `caustica.medium_kinds`

Subclass `MediumKindConfig` **from `caustica.config.kinds`, never from
`caustica.config.job`** — job.py builds its unions during its own import, so a
plugin that imports it at module scope would re-enter a half-initialised
module.

Two shapes, both first class:

* the common one — the job's `grid` section defines the grid and you paint a
  `Medium` onto it: implement `c_min()` and `build(grid)`;
* a **grid-providing** kind — your file fixes shape and dx, so a job carrying a
  `grid` section is refused: set `provides_grid = True` and implement
  `prepare(drive) -> MediumPrep`, returning the geometry immediately and the
  medium behind a callable so the cheap refusals run before gigabytes are
  materialised.

If your kind reads files, override `resolve_paths(base_dir)`: relative paths in
a job resolve against the **job file**, never the CWD.

### Array kind — `caustica.array_kinds`

Subclass `ArrayKindConfig`. Three methods: `focal_length_mm()` (what a
`natural` focus resolves to), `derived()` (numbers a reload re-derives, so a
library change that silently builds a *different* transducer is caught), and
`build_source(grid, drive, apex_vox, focus, phases_rad)`.

`caustica.arrays.elements_array` already voxelises an arbitrary set of element
positions and normals, so most geometries are a coordinate computation plus one
call — see the skeleton.

Put shared *data* in the subclass, not in a base: pydantic emits base-class
fields first, so hoisting a field would reorder the keys of the runner's
normalised `job.json` copy, which is an audit artifact.

### Backend — `caustica.backends`

Register a **factory**: a zero-argument callable returning a
`caustica.core.backend.Backend`, which is a frozen `(name, xp)` pair where `xp`
is a numpy-compatible array module. Anything expensive — a device probe, a CUDA
import — belongs *inside* the factory, exactly like caustica's own cupy factory,
so merely listing the backends stays free.

`Backend.fft` must be dtype-preserving (`scipy.fft` / `cupyx.scipy.fft`; plain
`numpy.fft` upcasts float32 to complex128 and destroys CPU/GPU parity).

**The `Backend.name` you return must equal the name you registered under.**
`get_backend` refuses a mismatch: the run stamp in `run_meta.json`, the
`backend` attribute in `result.h5` and the checkpoint fingerprint that decides
whether a resume is the same run all record `Backend.name`, never the name
that was asked for.

Two things `"auto"` will not do for you:

* **`"auto"` only chooses between `numpy` and `cupy`.** It is a policy over the
  two built-ins, not a poll of the registry: a third-party backend is opted
  into by name, never guessed at.
* **The runner's CPU gate and GPU reporting key on the names `numpy` and
  `cupy`.** A backend called anything else skips the slow-CPU refusal (D20) and
  reports no GPU environment in `run_meta.json`.

Name your backend in a job with `"backend": "my_backend"`, on the CLI with
`caustica run job.json --backend my_backend`, or in code with
`get_backend("my_backend")`. An unregistered name is refused at *validate*
time, not minutes into a run.

### Report renderer — `caustica.report_renderers`

Register a callable:

```python
def render(outdir: Path, *, preview_only: bool = False) -> Path: ...
```

`outdir` is a runner output folder — `result.h5` and/or `preview.npz`, plus
`metrics.json` and `run_meta.json` when the runner wrote them (the module
docstring of `caustica.runner` lists what it writes). `preview_only` asks for the
quick look even when the full field is present. Return the path a reader should
open; raise a plain exception with a readable message when the folder holds
nothing you can render — `caustica report` prints it and exits 2.

Reuse `caustica.report.metrics` (numpy only, the single source of truth for
focal numbers) and `caustica.report.preview` rather than recomputing: the
runner computed `metrics.json` **with the medium in memory**, so it carries
numbers a result file alone cannot provide.

Select yours with `caustica report <dir> --renderer my_report`, or
`caustica.report.render_report(outdir, renderer="my_report")`.

## Rules and gotchas

* **Import the seam, not the schema.** `from caustica.config.kinds import
  MediumKindConfig`, never `from caustica.config.job import ...`.
* **A name collision is an error**, unless the two definitions are the same
  class re-executed (a module reload, so `%autoreload 2` in a notebook keeps
  working).
* **Registration is all or nothing.** If wiring your kind into the job models
  fails, it is removed again — `available()` never advertises something the
  schema refuses.
* **caustica installs no logging handler.** Warnings from a failing plugin go
  to the `caustica` logger; configure logging (or use the CLI, which does) to
  see them.
* **Keep imports cheap at module scope.** Your module is imported during the
  scan, which happens on the first registry question — often inside somebody's
  job validation.

## Checking your plugin

```bash
python -c "import caustica.solvers as s; print(s.available())"
python -c "from caustica.core.backend import backends; print(backends.available())"
python -c "from caustica.report import report_renderers; print(report_renderers.available())"
python -m caustica schema --kinds          # medium and array kinds
python -m caustica validate my_job.json    # the whole thing, before running
```

If your name is missing, the plugin failed to load: run the same command with
logging on and read the WARNING.

```bash
python -c "import logging; logging.basicConfig(level='WARNING'); \
import caustica.solvers as s; print(s.available())"
```
