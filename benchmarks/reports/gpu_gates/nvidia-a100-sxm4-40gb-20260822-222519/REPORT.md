# caustica GPU gate suite

**Overall verdict: FAIL** (`caustica-gpu-gates/1`, generated 2026-08-22T22:25:19+00:00)

| | |
|---|---|
| device | NVIDIA A100-SXM4-40GB |
| VRAM total / free at start | 39.494 / 39.076 GiB |
| caustica | 0.1.0.dev0 @ cc3046d31357031ee03fb20e5da4e021c039db9b |
| python / cupy | 3.13.15 / 14.0.1 |
| datasheet key | A100-40GB |

## Milestone gates

| gate | verdict | criterion |
|---|---|---|
| `M7.parity` | **PASS** | numpy vs cupy on a mini 3-D scenario: field relative error < 1e-05 |
| `M7.fullsize` | **FAIL** | a full-size run (dx=0.30 mm, 512^3 FFT class) completes without OOM |
| `M8.vram` | **FAIL** | VRAM prediction within +/-10% of the measured mempool peak, on at least 2 grid sizes |
| `M8.time` | **FAIL** | post-calibration wall-time prediction within +/-25% of actual, on at least 2 scenarios (same device); a plan whose source is not 'calibrated' does not count |
| `M8.oom` | **PASS** | a run larger than the device is REFUSED before solving (exit 3) with advice |

### Every check

| check | verdict | detail |
|---|---|---|
| parity phasor: relative L2 | PASS | 2.503e-06 (limit 1.000e-05) |
| parity phasor: relative Linf | PASS | 1.471e-06 (limit 1.000e-05) |
| parity p_max: relative L2 | PASS | 1.762e-06 (limit 1.000e-05) |
| parity p_max: relative Linf | PASS | 1.196e-06 (limit 1.000e-05) |
| vram-14gib-512: completed | PASS | shape (512, 512, 512) at dx=0.3 mm exited 0 (0 = solved, 3 = refused for memory) |
| vram-28gib-640: completed | FAIL | shape (640, 640, 640) at dx=0.3 mm exited 3 (0 = solved, 3 = refused for memory) |
| vram-2gib-256: VRAM | PASS | predicted 1.729 GiB vs actual 1.764 GiB (-2.0%, tolerance +/-10%) |
| vram-8gib-400: VRAM | FAIL | predicted 6.591 GiB vs actual 8.096 GiB (-18.6%, tolerance +/-10%) |
| vram-14gib-512: VRAM | FAIL | predicted 13.82 GiB vs actual 21.37 GiB (-35.3%, tolerance +/-10%) |
| vram-28gib-640: VRAM | SKIP | not measured |
| vram-2gib-256: wall time | FAIL | predicted 16.6 s vs actual 6.54 s (+153.8%, tolerance +/-25%) |
| vram-8gib-400: wall time | FAIL | predicted 77.1 s vs actual 22.68 s (+239.9%, tolerance +/-25%) |
| vram-14gib-512: wall time | FAIL | predicted 185.1 s vs actual 45.4 s (+307.7%, tolerance +/-25%) |
| vram-28gib-640: wall time | SKIP | not measured |
| oom-750: refused with exit 3 | PASS | a 43.4 GiB run on this device exited 3 |
| oom-750: refusal carries advice | PASS | increase dx by >= x1.04 (same physical extent at grid ~(722, 722, 722); voxel count scales 1/m^3); shrink the record region (AOI): record buffers are 4.71 GiB — pass record_region=... to run(); the 'linear' solver drops the beta map and nonlinear temporaries (~4.71 GiB) — valid only if harmonics are not needed; or pick a larger device: A100-80GB, H100-PCIe, H100-SXM |

## Ladder: plan vs actual

| rung | shape | plan VRAM | actual VRAM | dev | plan time | actual time | dev | exit |
|---|---|---|---|---|---|---|---|---|
| vram-2gib-256 | 256x256x256 | 1.729 GiB | 1.764 GiB | -2.0% | 16.6 s | 6.54 s | +153.8% | 0 |
| vram-8gib-400 | 400x400x400 | 6.591 GiB | 8.096 GiB | -18.6% | 77.1 s | 22.68 s | +239.9% | 0 |
| vram-14gib-512 | 512x512x512 | 13.82 GiB | 21.37 GiB | -35.3% | 185.1 s | 45.4 s | +307.7% | 0 |
| vram-28gib-640 | 640x640x640 | 26.98 GiB | -- GiB | -- | 427 s | -- s | -- | 3 |
| oom-750 | 750x750x750 | 43.41 GiB | -- GiB | -- | 767.6 s | -- s | -- | 3 |

## Step-time baseline (M19 reads this)

| shape | voxels | steady s/step | measured s/step (incl. warmup) | warmup s |
|---|---|---|---|---|
| 256x256x256 | 16,777,216 | 0.0077 | 0.02517 | 4.542 |
| 400x400x400 | 64,000,000 | 0.0341 | 0.063 | 10.4 |
| 512x512x512 | 134,217,728 | 0.0583 | 0.1081 | 20.91 |

## numpy vs cupy parity (whole field)

Measured on: in-memory fp32 SolverResult fields, same process, no result.h5 round trip.

| field | rel L2 | rel Linf |
|---|---|---|
| phasor | 2.503e-06 | 1.471e-06 |
| p_max | 1.762e-06 | 1.196e-06 |

Storage floor (**informational, not gated**) — the same two fields after the store's float16 quantization. One float16 ULP is 2^-11 = 4.883e-4, so a relative L-infinity of about that size here means the solvers agree below the resolution of the file:

| field (stored) | rel L2 | rel Linf |
|---|---|---|
| phasor | 3.28e-05 | 0.0002396 |
| p_max | 2.699e-05 | 0.0002442 |

## Notes

- recorded warmup 20.91 s into the calibration for NVIDIA A100-SXM4-40GB; it applies to the NEXT plan, not to the numbers graded above

---

Reproduce: `python -m caustica.validation gpu-gates`. Every scenario in this report is generated from the library (homogeneous water + a bowl scaled to the grid); no external dataset is involved.
