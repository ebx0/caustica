---
hide:
  - navigation
---

<div class="hero-field" markdown>
![A focused ultrasound beam converging in water: one plane of a real caustica solve, with the wavefronts animated over one acoustic period](assets/hero-field.svg#only-light)
![A focused ultrasound beam converging in water: one plane of a real caustica solve, with the wavefronts animated over one acoustic period](assets/hero-field-dark.svg#only-dark)
</div>

# caustica

<p class="hero-tagline" markdown>
**GPU-accelerated, multi-solver acoustic simulation for HIFU and therapeutic ultrasound.**
Pure Python — NumPy on the CPU, CuPy on the GPU. No compiler, no precompiled binaries,
and the same four commands on a laptop and in a Colab cell.
</p>

[Get started](#quickstart){ .md-button .md-button--primary }
[Write a job](job_reference.md){ .md-button }
[GitHub](https://github.com/ebx0/caustica){ .md-button }

<small markdown>
Above: one x–z plane of a real solve — a 1 MHz focused bowl in water, run by the k-space
solver this library ships. The envelope is |P|; the moving fronts are Re{P·e^(−iωt)} over one
acoustic period, so the loop closes on itself exactly. `python scripts/make_hero.py --resolve`
re-solves it.
</small>

!!! warning "Pre-alpha, and honest about it"

    The API still moves between milestones. Every solver result quoted on this page is
    **CPU-validated** against analytic references and cross-checked against k-Wave. The
    CuPy backend is packaged and has run on A100 hardware, but its parity and full-size
    gates are not closed yet — see [MILESTONES.md](https://github.com/ebx0/caustica/blob/master/MILESTONES.md)
    for what is done and what is not.

## Quickstart

No checkout, no external data:

```bash
pip install "caustica[report] @ git+https://github.com/ebx0/caustica"

caustica example water_bowl_mini      # copies a packaged, self-contained job here
caustica validate water_bowl_mini.json
caustica run water_bowl_mini.json     # seconds on CPU; writes runs/water_bowl_mini/
caustica report runs/water_bowl_mini  # local HTML + figures
```

Prefix each line with `!` and the same four commands are a Colab session.

## How it fits together

Ten steps from an empty shell to a focal metric — the same ten decisions whether you write
a job file, call `caustica.simulate()` from Python, or run it in a Colab cell.

<div class="howto" markdown>
![How to use caustica, in ten steps](assets/how-to-use-real.svg#only-light)
![How to use caustica, in ten steps](assets/how-to-use-real-dark.svg#only-dark)
</div>

<small markdown>
The thumbnails are not decoration: each one is drawn by calling caustica. The absorbing
profile, the constructive geometry, the spiral element table, the steered Rayleigh preview,
the Fubini harmonics, the planner's own estimate, and the axial line and focal plane of a
real solve. Regenerate the whole diagram with `python scripts/make_howto.py`.
</small>

## What you get

<div class="grid cards" markdown>

-   :material-cube-outline:{ .lg .middle } **One job file, one full run**

    ---

    A `caustica-job/1` JSON names the grid, the medium, the source, the drive and the
    solver. `caustica schema` prints its JSON Schema, generated from the models — the
    documentation cannot drift from the code.

    [:octicons-arrow-right-24: Job reference](job_reference.md)

-   :material-shield-check-outline:{ .lg .middle } **It refuses before it wastes your GPU**

    ---

    The planner times a step, inventories the engine's actual buffers and estimates
    wall-clock, then refuses a run that will not fit or that a CPU would take hours over.
    The refusal names the fix for the machine you are on.

-   :material-layers-triple-outline:{ .lg .middle } **Anatomy in, per-voxel physics out**

    ---

    Constructive solid geometry, or a segmented volume resampled onto your grid, with
    every label carrying sound speed, density, absorption and B/A.

    [:octicons-arrow-right-24: Phantoms](uwcem.md)

-   :material-check-decagram-outline:{ .lg .middle } **Gated by analytic references**

    ---

    O'Neil, Rayleigh, Fubini, exponential absorption, plane-wave dispersion — plus k-Wave
    running as a registry solver on identical grids. All automated under `pytest`.

    [:octicons-arrow-right-24: What has been measured](validation.md)

-   :material-power-plug-outline:{ .lg .middle } **Five extension points, frozen names**

    ---

    Solver, medium kind, array kind, backend, report renderer. A third-party package plugs
    into any of them over entry points, without touching caustica.

    [:octicons-arrow-right-24: Extending](extending.md)

-   :material-monitor-dashboard:{ .lg .middle } **A contract a GUI can rely on**

    ---

    The run folder, the exit codes, `status.json`, `error.json`, the cancel signal and the
    progress payload. Nothing outside that page is a contract.

    [:octicons-arrow-right-24: GUI contract](gui_contract.md)

</div>

## Solvers

One API, a registry of engines:

| name | physics | dims | backend | status |
|---|---|---|---|---|
| `linear` | linear full-wave k-space PSTD | 1/2/3-D | numpy (cupy: M7) | validated |
| `westervelt` | nonlinear Westervelt k-space PSTD, multi-harmonic capture | 1/2/3-D | numpy (cupy: M7) | validated |
| `kwave` | [k-Wave](http://www.k-wave.org) `kspaceFirstOrder` via `k-wave-python` | 2/3-D | external binary | wrapped + cross-validated |
| `kzk` | parabolic KZK (z-marching) | planned | — | M9 |

## From Python

The same job, the same planner, the same gates — without leaving a notebook:

```python
import caustica

res = caustica.simulate(
    "water_bowl_mini.json",   # a job path, a job dict, an ExplicitJobConfig, or a BuiltJob
    solver="westervelt",
    harmonics=(1, 2),
    out=None,                 # None = in memory; a path = the full run folder
    progress="auto",
)

res.metrics        # focal metrics — the definitions the HTML report quotes
res.result.phasor  # the complex field, as the solver produced it
res.save("result.h5")
```

[:octicons-arrow-right-24: The rest of the Python API](library.md)

`out=None` writes nothing at all, but it does **not** skip the planner or the two pre-run
gates: a run that will not fit in VRAM, or that a CPU would take hours over, is refused
here exactly as `caustica run` refuses it — same message, same exit code, carried on
`SimulationError.exit_code`.

## What has been measured

Every solver milestone is gated by tests against analytic references **and** cross-validated
against k-Wave on identical grids, media and sources:

- plane-wave phase-speed error **< 0.1 %** at 4 points per wavelength; measured absorption
  within **1 %** of the configured α
- 3-D focused bowl vs O'Neil (1949): focus within **one voxel**, axial correlation
  **r > 0.99**, −6 dB widths within **5 %**
- Westervelt vs Fubini: second-harmonic ratio within **5 %** (measured 0.9–3.2 %) across
  σ = 0.06–0.61
- `linear` vs `kwave` (the real OMP binary), 2-D water: normalized-field correlation
  **r > 0.99**
- one phasor convention library-wide, `p(t) = Re{P·e^(−iωt)}`, shared with the analytic
  references — see [the conventions that bite](conventions.md)

## Where to go next

<div class="grid cards" markdown>

-   **Write your own job** — every field, every medium and array kind, with a working
    snippet per kind.

    [:octicons-arrow-right-24: Job reference](job_reference.md)

-   **Read this before you trust a number** — the five things that make a result *silently*
    wrong if you assume otherwise.

    [:octicons-arrow-right-24: Conventions](conventions.md)

-   **Plug your own physics in** — the five extension points and a copy-paste plugin package
    that uses all five.

    [:octicons-arrow-right-24: Extending](extending.md)

-   **Follow the build** — the engineering log, milestone by milestone.

    [:octicons-arrow-right-24: Devlog](devlog.md)

</div>
