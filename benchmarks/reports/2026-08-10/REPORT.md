# hifusim validation report — 2026-08-10

Environment: Windows 11, Python 3.12.10, numpy 2.5.2, scipy 1.18.0, k-wave-python 0.6.2
(kspaceFirstOrder **CPU/OMP** binary), hifusim @ `4b81f8c` (+ report tooling).
All comparisons on **normalized** fields (pressure-injected PSTD sources have no closed-form
absolute amplitude; the nonlinear scenario amplitude-matches realized f0 first).
Raw numbers: [`metrics.json`](metrics.json). Regenerate with
`python scripts/gen_validation_report.py --outdir <dir> --scenario all`.

## 1. hifusim vs k-Wave (same grid, medium, voxel source)

| scenario | solver pair | relL2 | Pearson r | peak offset [vox] | figure |
|---|---|---|---|---|---|
| 2-D water, linear | `linear` vs `kwave` | **1.14%** | **0.99981** | (0, 0) | `fig_kwave2d_linear.png` |
| 2-D water+fat slab (α=6 Np/m) | `linear` vs `kwave` | **1.29%** | **0.99977** | (0, 0) | `fig_kwave2d_hetero.png` |
| 2-D nonlinear water β=3.5, f0 | `westervelt` vs `kwave` | **1.14%** | **0.99981** | (0, 0) | `fig_kwave2d_nl_f0.png` |
| 2-D nonlinear water β=3.5, 2f0 | `westervelt` vs `kwave` | 17.3% | 0.97417 | (0, 0) | `fig_kwave2d_nl_2f0.png` |
| 3-D focused bowl, water | `linear` vs `kwave` | **1.57%** | **0.99981** | (0, 0, 1) | `fig_kwave3d_bowl.png` |

Notes:
- Nonlinear 2f0: the scenario sits at very weak nonlinearity (A2/A1 at peak: hifusim 0.0120
  vs k-Wave 0.0133 → 9.8% level difference, inside the 10% ITRUSST-style corridor), so the
  2f0 *field* relL2 is dominated by the tiny signal's discretization floor. A higher-σ
  scenario is planned for the M12 harness.
- 3-D peak amplitude ratio 1.081 reflects the different source-injection scalings of the two
  codes (comparison is shape-based by design).

## 2. hifusim vs analytic references

| gate | reference | result | criterion | figure |
|---|---|---|---|---|
| Plane-wave dispersion (4 ppw, lossless) | k = ω/c | **0.004% error** | < 0.1% | — |
| Absorption α=30 Np/m | exp(−αx) | **0.33% error** | < 1% | `fig_absorption.png` |
| 3-D bowl axial profile | O'Neil (1949) | **r = 0.9916** | r > 0.99 | `fig_oneill3d.png` |
| 3-D bowl lateral profile | Rayleigh integral | **r = 0.9989** | r > 0.98 | `fig_oneill3d.png` |
| Westervelt A2/A1, σ ∈ [0.05, 0.60] | Fubini | **max 3.8%, mean 1.9%** | < 5% | `fig_fubini.png` |
| 128-element spiral DAS steering | Rayleigh preview | peak (8.0, 84.5) mm vs target (8.0, 85.0) mm | < λ/2 lateral | `fig_spiral.png` |

## 3. Suite status

- `pytest`: **90 passed** (unit + physics gates + live k-Wave cross-checks), ruff clean.
- Milestones complete: M0–M6 (+M4b k-Wave adapter). Next: M7 CuPy/GPU (needs Colab), M8 planner.
- Known caveats carried in MILESTONES.md/devlog: source notebook's dataset realized **half**
  the labeled absorption (p-only damping — fixed in hifusim); dataset phase maps are **64×64**
  (32×32 fails element placement); k-Wave GPU binary remains quarantined behind a T0 sanity
  gate (M12).
