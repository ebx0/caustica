# apps/ — applications built on the caustica library

Programs that USE the library, as an outside consumer would. They import only
the public API (`caustica`, `caustica.solvers`, `caustica.geometry`, …) and never
reach into internals, so they are the honest end-to-end check that what the
milestones claim is actually usable.

Apps are not part of the library: they are not packaged by `pip install -e .`.
They ARE reached by `pytest` — tests/test_report.py imports
`apps.focus_study.{scenarios,analysis}` for the metric-single-source parity
tests — and covered by `ruff` (`ruff check apps src tests`). Run outputs land
in `apps/outputs/`, which is git-ignored.

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

- **CPU only.** The CuPy path is still provisional; `backend="numpy"` is passed explicitly.
- **`.npz`, not HDF5.** The HDF5 contract is the runner's, with resume.
- **One run per invocation.** Parameter sweeps are the Study harness.
- **Absorption is exponential at f0.** Harmonics are absorbed with the
  fundamental's alpha, so second-harmonic levels are optimistic.
- **Resolution is checked, not enforced.** Recording harmonic `n` on a grid
  sized for f0 leaves it under-resolved; the app prints exactly how far off it
  is and what `--dx` would fix it, then runs anyway.

---

> The phantom launcher and Phantom Studio moved to the `uwcem-phantom`
> repository, together with the `uwcem_phantoms` package.
