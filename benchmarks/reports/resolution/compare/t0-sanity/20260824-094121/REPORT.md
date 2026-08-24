# caustica multi-engine comparison

**Overall verdict: PASS** (`caustica-compare/1`, generated 2026-08-24T09:41:21+00:00)

The SAME built job -- one grid, one medium, one source -- run on each registered solver below, in memory, on one backend. The first solver is the reference; every other is compared against it. Comparison is NORMALIZED (each field divided by its own interior peak): absolute amplitude is a per-engine source convention, field SHAPE is the cross-check. Gated tolerances name the test they are inherited from; ungated columns are reported because no test in this repository establishes a cross-engine limit for them.

| | |
|---|---|
| job | t0-sanity (`dict`) |
| grid | [40, 40, 48] @ 0.5 mm, PML 6 vox |
| reference solver | `linear` |
| backend | cupy (requested `cupy`) |
| host | LOQ |
| caustica | 0.1.0.dev0 @ 55e4ee8a8666ade2e5d46aca6451a811525e1584 |
| python / numpy | 3.12.10 / 2.2.6 |
| platform | Windows-11-10.0.26200-SP0 |
| suite runtime | 3.506 s |

## solvers

`environment` is a machine problem, never a physics verdict: the row stays, the error is copied verbatim, and its checks are SKIP. `unsupported` means the solver's own capability contract refused this job.

| solver | status | T0 s | run s | steps | interior peak (MPa) | detail |
|---|---|---|---|---|---|---|
| `linear` | ok | 0.21 | 0.1793 | 120 | 0.6972 | -- |
| `westervelt` | ok | 0.1947 | 0.229 | 120 | 0.6972 | -- |
| `kwave` | ok | 1.515 | 1.119 | 220 | 0.6607 | -- |

## pairs (normalized |phasor|)

`r` and the peak shift are GATED at r >= 0.99 and <= 1 voxel (`tests/test_kwave_adapter.py::test_kwave_vs_linear_2d_water`). `rel L2` and the peak ratio are reported, not gated -- no test in this repository establishes a cross-engine limit for either, and one invented here would be one chosen to pass. The peak ratio is the UN-normalized amplitude comparison, kept visible so an engine's source convention cannot hide inside the normalization.

| pair | rel L2 (norm) | Pearson r | peak shift (vox) | peak ratio cmp/ref | voxels |
|---|---|---|---|---|---|
| linear vs westervelt | 0 | 1 | 0 | 1 | 18432 |
| linear vs kwave | 0.03856 | 0.9988 | 0 | 0.9476 | 18432 |

## focal metrics

`caustica.report.metrics.focus_metrics` -- the same function `caustica report` and `apps/focus_study` use, so these numbers are comparable to any run folder's `metrics.json`. Reported per engine, not gated.

| solver | peak (MPa) | peak voxel | z from apex (mm) | axial -6 dB (mm) | lat-x -6 dB (mm) | lat-y -6 dB (mm) | vol >-6 dB (mm3) | ISPPA (W/cm2) |
|---|---|---|---|---|---|---|---|---|
| `linear` | 0.6972 | [20, 20, 22] | 7 | 9.968 | 2.567 | 2.583 | 39.62 | 16.2 |
| `westervelt` | 0.6972 | [20, 20, 22] | 7 | 9.968 | 2.567 | 2.583 | 39.62 | 16.2 |
| `kwave` | 0.6607 | [20, 20, 22] | 7 | 9.946 | 2.568 | 2.585 | 41.12 | 14.5 |

## gates

### M11.t0 -- PASS

every solver with a working environment produces a finite field on the trivial T0 job, with an interior peak within a factor 5 of the analytic focal gain, BEFORE its field is allowed into any comparison (needs 6 passing check(s), has 6).

| check | verdict | measured vs limit | source of the tolerance |
|---|---|---|---|
| t0.linear.finite | PASS | phasor and p_max are finite everywhere | tests/test_divergence_guard.py::test_a_diverging_run_raises_instead_of_returning_nan |
| t0.linear.peak | PASS | 1.45244 x analytic (band 0.2..5 x analytic) | tests/test_linear_oneill_3d.py::test_axial_profile_matches_oneill (via caustica.analytic.oneill.focal_gain) |
| t0.westervelt.finite | PASS | phasor and p_max are finite everywhere | tests/test_divergence_guard.py::test_a_diverging_run_raises_instead_of_returning_nan |
| t0.westervelt.peak | PASS | 1.45244 x analytic (band 0.2..5 x analytic) | tests/test_linear_oneill_3d.py::test_axial_profile_matches_oneill (via caustica.analytic.oneill.focal_gain) |
| t0.kwave.finite | PASS | phasor and p_max are finite everywhere | tests/test_divergence_guard.py::test_a_diverging_run_raises_instead_of_returning_nan |
| t0.kwave.peak | PASS | 1.37638 x analytic (band 0.2..5 x analytic) | tests/test_linear_oneill_3d.py::test_axial_profile_matches_oneill (via caustica.analytic.oneill.focal_gain) |

### M11.cross -- PASS

normalized |phasor| correlation r >= 0.99 between the reference solver 'linear' and every other compared solver (needs 2 passing check(s), has 2).

| check | verdict | measured vs limit | source of the tolerance |
|---|---|---|---|
| linear-vs-westervelt.corr | PASS | 1 (floor 0.99) | tests/test_kwave_adapter.py::test_kwave_vs_linear_2d_water |
| linear-vs-kwave.corr | PASS | 0.998792 (floor 0.99) | tests/test_kwave_adapter.py::test_kwave_vs_linear_2d_water |

### M11.focus -- PASS

the interior peak of the normalized field agrees within 1 voxel per axis across engines (needs 2 passing check(s), has 2).

| check | verdict | measured vs limit | source of the tolerance |
|---|---|---|---|
| linear-vs-westervelt.peak_shift | PASS | 0.000e+00 vox (limit 1.000e+00 vox) | tests/test_kwave_adapter.py::test_kwave_vs_linear_2d_water |
| linear-vs-kwave.peak_shift | PASS | 0.000e+00 vox (limit 1.000e+00 vox) | tests/test_kwave_adapter.py::test_kwave_vs_linear_2d_water |

Reproduce: `python -m caustica.validation compare --solvers linear,westervelt,kwave --backend cupy`. The job that was actually built is stored beside this file as `job_input.json`.
