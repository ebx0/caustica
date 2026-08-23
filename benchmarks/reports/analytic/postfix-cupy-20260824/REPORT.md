# caustica analytic validation suite

**Overall verdict: PASS** (`caustica-analytic/1`, generated 2026-08-23T23:26:53+00:00)

Solver output against the closed forms in `caustica.analytic`, measured on this machine. Every tolerance below is inherited from the physics test that established it — the `source` column names it.

| | |
|---|---|
| backend | cupy |
| host | LOQ |
| size | full |
| caustica | 0.1.0.dev0 @ b41850d794884dc74c3e8f1b5f56f684fbb2cc5d |
| python / numpy / scipy | 3.12.10 / 2.2.6 / 1.15.3 |
| platform | Windows-11-10.0.26200-SP0 |
| suite runtime | 4.8 s |

## Gates

| gate | verdict | passes / required | criterion |
|---|---|---|---|
| `M4.planewave` | **PASS** | 3 / 3 | 1-D plane wave: the realized drive amplitude honours the mass-source contract, the numerical phase speed matches k = omega/c, and the decay exponent matches the configured alpha of analytic.attenuate |
| `M4.oneill` | **PASS** | 3 / 3 | 3-D focused bowl vs analytic.axial_pressure (O'Neil): normalized axial profile correlation, focal peak position, and -6 dB axial width |
| `M5.linear_limit` | **PASS** | 3 / 3 | westervelt with beta = 0 is bit-identical to the linear solver, and says so in its metadata |
| `M5.fubini` | **PASS** | 5 / 4 | second-harmonic ratio A2/A1 within 5% of analytic.fubini_harmonic at >= 4 pre-shock sigma stations |

### Every check: measured vs limit

| check | verdict | measured vs limit | source of the tolerance |
|---|---|---|---|
| planewave: realized amplitude / p0 | PASS | 1.08245 (band 0.9..1.12) | `tests/test_review_hardening.py::test_mass_source_normalization_and_phase_convention` |
| planewave: phase speed (k) | PASS | predicted 4189 rad/m vs actual 4189 rad/m (-0.0%, tolerance +/-0.1%) | `tests/test_linear_planewave.py::test_plane_wave_dispersion_below_0p1_percent` |
| planewave: absorption exponent | PASS | predicted 30.1 Np/m vs actual 30 Np/m (+0.3%, tolerance +/-1%) | `tests/test_linear_planewave.py::test_absorption_matches_configured_alpha_within_1_percent` |
| oneill: axial profile correlation r | PASS | 0.99923 (floor 0.99) | `tests/test_linear_oneill_3d.py::test_axial_profile_matches_oneill` |
| oneill: focal peak position error | PASS | 1.000e+00 vox (limit 1.000e+00 vox) | `tests/test_linear_oneill_3d.py::test_focus_lands_within_one_voxel` |
| oneill: -6 dB axial width | PASS | predicted 16.08 mm vs actual 15.97 mm (+0.6%, tolerance +/-5%) | `tests/test_linear_oneill_3d.py::test_axial_profile_matches_oneill` |
| linear limit: max \|phasor difference\| | PASS | 0.000e+00 Pa (limit 0.000e+00 Pa) | `tests/test_westervelt.py::test_beta_zero_is_bitwise_identical_to_linear` |
| linear limit: max \|p_max difference\| | PASS | 0.000e+00 Pa (limit 0.000e+00 Pa) | `tests/test_westervelt.py::test_beta_zero_is_bitwise_identical_to_linear` |
| linear limit: westervelt reports nonlinear_active | PASS | nonlinear_active is False in linear water (expected False) | `tests/test_westervelt.py::test_beta_zero_is_bitwise_identical_to_linear` |
| fubini: A2/A1 at sigma=0.110 | PASS | 3.171e-02 (limit 5.000e-02) | `tests/test_westervelt.py::test_fubini_second_harmonic_within_5_percent` |
| fubini: A2/A1 at sigma=0.233 | PASS | 2.239e-02 (limit 5.000e-02) | `tests/test_westervelt.py::test_fubini_second_harmonic_within_5_percent` |
| fubini: A2/A1 at sigma=0.355 | PASS | 1.016e-02 (limit 5.000e-02) | `tests/test_westervelt.py::test_fubini_second_harmonic_within_5_percent` |
| fubini: A2/A1 at sigma=0.478 | PASS | 1.311e-02 (limit 5.000e-02) | `tests/test_westervelt.py::test_fubini_second_harmonic_within_5_percent` |
| fubini: A2/A1 at sigma=0.588 | PASS | 9.026e-03 (limit 5.000e-02) | `tests/test_westervelt.py::test_fubini_second_harmonic_within_5_percent` |

## Plane wave (1-D, linear solver)

| quantity | analytic | measured |
|---|---|---|
| drive amplitude [Pa] | 1e+05 | 1.082e+05 |
| wavenumber k [rad/m] | 4189 | 4189 |
| alpha [Np/m] | 30 | 30.1 |

Envelope vs `analytic.attenuate` over the window, worst point: 3.904% (**informational, not gated** — its floor is the residual standing ripple off the sponge, which no tolerance in this repository pins).

## O'Neil bowl (3-D, linear solver)

| | |
|---|---|
| grid | 80x80x120 @ 0.375 mm |
| bowl | aperture radius 7.5 mm, roc 18 mm (f/1.2) |
| axial samples compared | 48 |
| peak voxel | [40, 40, 57] (O'Neil peak at z = 56 vox) |
| lateral peak offset | 0 vox |
| -6 dB axial width | solver 16.08 mm vs O'Neil 15.97 mm |

## Fubini second harmonic (1-D, westervelt solver)

Realized source amplitude 2.006 MPa, shock distance 76.51 mm, 16 points per wavelength.

| sigma | A2/A1 measured | A2/A1 Fubini | relative error |
|---|---|---|---|
| 0.1103 | 0.05675 | 0.055 | 3.17% |
| 0.2328 | 0.1177 | 0.1151 | 2.24% |
| 0.3554 | 0.1748 | 0.173 | 1.02% |
| 0.4779 | 0.2307 | 0.2277 | 1.31% |
| 0.5882 | 0.2757 | 0.2733 | 0.90% |

## Planner: estimate vs actual

`caustica.planner.estimate` (`measure=False`) for each scenario's setup, taken before the solve it describes — target `unknown:NVIDIA GeForce RTX 5050 Laptop GPU`, estimate source `db`. A scenario that solves more than once (plane wave, linear limit) adds its solves up; the memory column is the largest of them, since they run one after another. Deviation is (predicted - measured) / |measured|.

| scenario | predicted t [s] | measured t [s] | deviation | predicted steps | actual steps | vram predicted [GiB] |
|---|---|---|---|---|---|---|
| planewave | 6 | 1.324 | +353.1% | 1640 | 1640 | 2.103e-05 |
| oneill | 3.056 | 0.4622 | +561.2% | 184 | 184 | 0.06953 |
| linear_limit | 6 | 0.5225 | +1048.3% | 768 | 768 | 9.374e-06 |
| fubini | 3 | 1.625 | +84.6% | 2673 | 2673 | 6.174e-05 |

**Informational, not gated — no gate reads these rows.** The measured column is wall clock for the 6 solves these 4 scenarios run, on a machine that is also doing other things, and the model behind the predicted column is coarse by construction: off the GPU it is a ~20-step calibration fitted on 3-D probe grids — or, with none on this machine, nothing at all, which is why that cell can be empty rather than invented. A few-thousand-voxel 1-D run is then predicted by a per-element fit that never saw a grid that small, and comes out short; the 3-D bowl is the row where the model is being asked something it was fitted for. The step counts are the part worth reading regardless: they come from the same settling policy the engine runs, so they are exact when convergence lands where the planner assumed. Gating any of this would grade the machine's afternoon; the gates above grade the library.

Of the predicted totals above, 18 s is the planner's one-time `warmup_s` — CUDA context, cuFFT plans and kernel compilation, paid once per solve. A `cpu` target reports zero for it and means it: a numpy run creates none of those.

## Scenario runtimes

| scenario | seconds | note |
|---|---|---|
| planewave | 1.358 |  |
| oneill | 0.4715 |  |
| linear_limit | 0.5251 |  |
| fubini | 1.626 |  |

---

Reproduce: `python -m caustica.validation run-analytic`. Every scenario is generated by the library (homogeneous water, library-built sources) and graded against `caustica.analytic`; no external dataset is involved.
