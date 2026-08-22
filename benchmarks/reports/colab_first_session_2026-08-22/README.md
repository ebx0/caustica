# First real GPU session — 2026-08-22, NVIDIA A100-SXM4-40GB (Colab)

The stamped artifacts of caustica's first run on a real GPU, kept verbatim as the
evidence behind M10f's Colab gate. The operator ran `notebooks/colab_run.ipynb`
against the packaged `water_bowl_mini` example, downloaded the output folder and
rendered `REPORT.md` / `index.html` locally with `caustica report`.

What it showed:

- **Physics parity is exact at the metric level.** The same job re-run on the CPU
  gives a peak pressure differing by **1.8e-7** relative; focal geometry, the −6 dB
  spot and the spot volume are identical, and the convergence trajectory matches
  period for period (converged at period 11, 104 steps).
- **Field parity is below the file's own resolution.** Comparing the stored fields
  against a local CPU run gives relative L2 3.6e-5 and relative L-infinity 4.883e-4
  — and 4.883e-4 is exactly 2^-11, one float16 ULP. 99.17% of `p_max` is
  bit-identical, 517 voxels differ by one ULP, none by more. This is why
  `caustica.validation`'s parity gate is measured on in-memory fp32 fields and
  never on a `result.h5` round trip.

Three defects it exposed, all fixed afterwards (see docs/devlog.md, 2026-08-23):
`git_commit: "unknown"` on a wheel install (fix A1), `t_step_measured_s` hiding a
2.66 s one-time warmup inside a per-step average (fix A2), and CI not testing
Colab's Python 3.13 (fix A3).

`result.h5` and `preview.npz` are deliberately absent: the repository does not carry
binary field data (`.gitignore`). The operator keeps them locally.
