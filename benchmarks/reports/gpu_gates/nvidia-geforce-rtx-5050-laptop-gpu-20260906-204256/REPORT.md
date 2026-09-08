# caustica GPU gate suite

**Overall verdict: FAIL** (`caustica-gpu-gates/1`, generated 2026-09-06T20:42:56+00:00)

| | |
|---|---|
| device | NVIDIA GeForce RTX 5050 Laptop GPU |
| VRAM total / free at start | 7.96 / 6.9 GiB |
| caustica | 0.1.0.dev0 @ dfd19414eea5069f5520002ca06390deaddb6fba |
| python / cupy | 3.12.10 / 14.2.0 |
| datasheet key | unknown:NVIDIA GeForce RTX 5050 Laptop GPU |

## Milestone gates

| gate | verdict | criterion |
|---|---|---|
| `parity` | **PASS** | numpy vs cupy on a mini 3-D scenario: field relative error < 1e-05 |
| `fullsize` | **INCOMPLETE** | a full-size run (dx=0.30 mm, 512^3 FFT class) completes without OOM |
| `vram` | **PASS** | VRAM prediction within +/-10% of the mempool peak measured in the rung's OWN process, on at least 2 grid sizes |
| `time` | **FAIL** | post-calibration wall-time prediction within +/-25% of actual, on at least 2 scenarios (same device); a plan whose source is not 'calibrated' does not count |
| `oom` | **PASS** | a run larger than the device is REFUSED before solving (exit 3) with advice |

### Every check

| check | verdict | detail |
|---|---|---|
| parity phasor: relative L2 | PASS | 2.262e-06 (limit 1.000e-05) |
| parity phasor: relative Linf | PASS | 1.327e-06 (limit 1.000e-05) |
| parity p_max: relative L2 | PASS | 1.691e-06 (limit 1.000e-05) |
| parity p_max: relative Linf | PASS | 1.338e-06 (limit 1.000e-05) |
| vram-1.587gib-250: VRAM | PASS | predicted 1.544 GiB vs actual 1.431 GiB (+7.9%, tolerance +/-10%) |
| vram-3.809gib-324: VRAM | PASS | predicted 3.359 GiB vs actual 3.113 GiB (+7.9%, tolerance +/-10%) |
| vram-1.587gib-250: wall time | PASS | predicted 8.6 s vs actual 8.04 s (+7.0%, tolerance +/-25%) |
| vram-3.809gib-324: wall time | FAIL | predicted 34.1 s vs actual 23.37 s (+45.9%, tolerance +/-25%) |
| oom-432: refused with exit 3 | PASS | a 8.0 GiB run on this device exited 3 |
| oom-432: refusal carries advice | PASS | increase dx by >= x1.05 (same physical extent at grid ~(410, 410, 410); voxel count scales 1/m^3); shrink the record region (AOI): record buffers are 0.90 GiB — pass record_region=... to run(); the 'linear' solver drops the beta map and nonlinear temporaries (~0.90 GiB) — valid only if harmonics are not needed; or pick a larger device: A100-40GB, A100-80GB, H100-PCIe, H100-SXM, L4, T4, V100 |

## Ladder: plan vs actual

| rung | shape | plan VRAM | actual VRAM | dev | plan time | actual time | dev | exit |
|---|---|---|---|---|---|---|---|---|
| vram-1.587gib-250 | 250x250x250 | 1.544 GiB | 1.431 GiB | +7.9% | 8.6 s | 8.04 s | +7.0% | 0 |
| vram-3.809gib-324 | 324x324x324 | 3.359 GiB | 3.113 GiB | +7.9% | 34.1 s | 23.37 s | +45.9% | 0 |
| oom-432 | 432x432x432 | 7.957 GiB | -- GiB | -- | 152.3 s | -- s | -- | 3 |

## Step-time baseline

| shape | voxels | steady s/step | measured s/step (incl. warmup) | warmup s |
|---|---|---|---|---|
| 250x250x250 | 15,625,000 | 0.02655 | 0.03094 | 1.142 |
| 324x324x324 | 34,012,224 | 0.0664 | 0.07303 | 2.121 |

## numpy vs cupy parity (whole field)

Measured on: in-memory fp32 SolverResult fields, same process, no result.h5 round trip.

| field | rel L2 | rel Linf |
|---|---|---|
| phasor | 2.262e-06 | 1.327e-06 |
| p_max | 1.691e-06 | 1.338e-06 |

Storage floor (**informational, not gated**) — the same two fields after the store's float16 quantization. One float16 ULP is 2^-11 = 4.883e-4, so a relative L-infinity of about that size here means the solvers agree below the resolution of the file:

| field (stored) | rel L2 | rel Linf |
|---|---|---|
| phasor | 2.832e-05 | 0.0002346 |
| p_max | 3.406e-05 | 0.0002439 |

## Notes

- recorded 2 warmup sample(s) into the calibration for NVIDIA GeForce RTX 5050 Laptop GPU; it applies to the NEXT plan, not to the numbers graded above

---

Reproduce: `python -m caustica.validation gpu-gates`. Every scenario in this report is generated from the library (homogeneous water + a bowl scaled to the grid); no external dataset is involved.
