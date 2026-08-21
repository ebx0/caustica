# apps/ — applications built on the caustica library

Programs that USE the library, as an outside consumer would. They import only
the public API (`caustica`, `caustica.solvers`, `caustica.geometry`, …) and never
reach into internals, so they are the honest end-to-end check that what the
milestones claim is actually usable.

Apps are not part of the library: they are not packaged by `pip install -e .`
and they are not covered by `pytest`. They are covered by `ruff` (`ruff check
apps src tests`). Run outputs land in `apps/outputs/`, which is git-ignored.

---

## focus_study — HIFU focus characterization

Runs one steady-state CW scenario end to end on the CPU and writes a complete
result folder: figures, metrics, raw fields and a report.

```bash
python -m apps.focus_study --list                    # what is available
python -m apps.focus_study water_bowl --dry-run      # plan only, no solve
python -m apps.focus_study water_bowl                # run + report
python -m apps.focus_study spiral_array --dx 0.3 --amplitude 0.2
```

Run it from the repository root (the `-m` form needs the repo root on
`sys.path`), with the project's virtualenv:
`.venv/Scripts/python.exe -m apps.focus_study ...` on Windows.

### Scenarios

| name | what it exercises | why it exists |
|---|---|---|
| `water_bowl` | bowl source, homogeneous medium, O'Neil analytic reference | the self-check: correct physics or not |
| `spiral_array` | `caustica.arrays`, DAS phasing, off-axis steering | array design question: where does the focus actually land |
| `layered_tissue` | `caustica.geometry` CSG scene + `breast_default` materials | heterogeneity: aberration, attenuation, focal shift |

### What it writes

```
apps/outputs/<scenario>-<timestamp>/
  index.html        report with the figures inline (open this)
  REPORT.md         same content as markdown
  metrics.json      every number, machine-readable
  fields.npz        amp/phase at f0, p_max, harmonic amplitudes, axes, labels
  plan.txt          the planner verdict, saved before the solve started
  fig_*.png         field maps, profiles, harmonics, convergence, medium
```

### How it works

1. Build the scenario (grid, medium, source) from the CLI knobs.
2. **Plan before spending**: `caustica.planner` predicts memory and wall time —
   optionally by timing ~20 real steps on this machine (`--no-measure` skips
   it) — and also reports what the same run would cost on a GPU. `--dry-run`
   stops here.
3. Solve with `linear` or `westervelt` on the numpy backend.
4. Analyze: peak pressure and its position, focus displacement from the
   requested target, -6 dB focal dimensions and volume, I_sppa, second-harmonic
   content, and — for a plain bowl in a homogeneous medium — a correlation
   against the O'Neil closed form.
5. Write figures + report.

### Deliberate limits (they mirror the library's own)

- **CPU only.** The CuPy path is M7; `backend="numpy"` is passed explicitly.
- **`.npz`, not HDF5.** The IO contract with resume is M10.
- **One run per invocation.** Parameter sweeps are the Study harness, M11.
- **Absorption is exponential at f0.** Harmonics are absorbed with the
  fundamental's alpha, so second-harmonic levels are optimistic (M16).
- **Resolution is checked, not enforced.** Recording harmonic `n` on a grid
  sized for f0 leaves it under-resolved; the app prints exactly how far off it
  is and what `--dx` would fix it, then runs anyway.

---

## phantom_launcher — the menu in front of all of it

```bash
phantoms.bat                     # Windows: double-click it, or run it
./phantoms.sh                    # POSIX
python -m apps.phantom_launcher  # same thing, explicitly
python -m apps.phantom_launcher build   # jump straight to one action
```

A stdlib menu over the whole package: launch the studio, build a phantom, build
or verify the standard aligned dataset (`data/phantoms/`, all nine phantoms on
one grid), print the catalog or the tissue table, inspect an export, download
archives. The
build wizard is the reason it exists — it suggests a `dx` from the chosen `f0`
(the 4.4 points-per-wavelength design rule, and the 2 ppw floor below which no
focus forms at all), then runs `plan(spec)` and shows the grid, voxel count and
peak RAM *before* the build. It finishes by printing the equivalent
`python -m uwcem_phantoms build ...` line; `tests/test_phantom_launcher.py`
parses that line back through the real argument parser and requires it to
reproduce the same spec, because a printed command that builds something else is
worse than no command.

---

## phantom_studio — build simulation-ready phantoms, and look at them

A local web GUI over `uwcem_phantoms`. Left panel: every knob of a
`PhantomSpec`. Right panel: an interactive WebGL2 volume rendering and three
slider-driven cross-sections of the *same* build.

```bash
python -m apps.phantom_studio                     # opens http://127.0.0.1:8765/
python -m apps.phantom_studio --port 9000 --no-browser
python -m apps.phantom_studio --preview-mvox 20   # bigger interactive builds
```

Run it from the repository root with the project's virtualenv
(`.venv/Scripts/python.exe -m apps.phantom_studio` on Windows). First run
downloads whatever phantom you select (~20 MB each, or `python -m
uwcem_phantoms fetch --all` for all nine up front).

### What it shows

- **3-D** — a ray-cast volume with per-tissue visibility, opacity and shading,
  and a corner cutaway whose three exposed faces ARE the three slice planes
  below it. Drag to orbit, wheel to zoom.
- **Cross-sections** — axial / coronal / sagittal, each with its own slider,
  hover readout (tissue name, or the interpolated property value), and the
  slice position in mm.
- **Any field** — the label map, or `c`, `rho`, `alpha`, `beta`, `rho·c`, on a
  colour scale fixed over the whole volume so dragging a slider never rescales
  the colours out from under you.
- **The build itself** — voxel counts and per-tissue fractions, the acoustic
  table actually in use, every warning the build produced, the full build log,
  and the exact Python needed to import the result.

### Deliberate design choices

- **No dependencies.** Stdlib `http.server` plus hand-written WebGL2 — no web
  framework to install, no CDN to reach, so it works on an offline machine.
- **Preview vs full build.** Interactive edits build at a coarsened `dx` that
  fits a voxel budget, and the UI says so *and* refuses to export a preview.
  "Build full" runs the resolution you actually asked for.
- **The server owns physics, the browser owns pixels.** Slices and volumes go
  over the wire as raw `uint8` plus a value range; colour mapping happens in
  JS, so switching field or colour ramp is instant and never rebuilds.
- **One palette everywhere.** Tissue colours come from
  `uwcem_phantoms.tissue.DEFAULT_COLORS` — five family hues validated for
  colour-vision separation against this exact dark surface, with lightness
  ramps inside the two graded families.
