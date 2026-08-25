# ITRUSST PH1 water benchmarks

Reference: Rayleigh surface integral over the transducer's own geometry, which is exact for a baffled source in a homogeneous medium; the intercomparison used FOCUS, a Rayleigh-integral code, for the same reason.

Paper: Aubry et al., J. Acoust. Soc. Am. 152(2), 1003 (2022); arXiv:2202.04552 — reported L-infinity spread across eleven models against FOCUS

| case | medium | source | L-inf % | L2 % | peak kPa (sim/ref) | ratio | peak z mm |
|---|---|---|---|---|---|---|---|
| BM1-SC1 | lossless water | bowl | 3.41 | 0.42 | 1131.4 / 1094.4 | 1.0338 | 62.0 / 62.0 |
| BM1-SC2 | lossless water | piston | 3.82 | 1.00 | 124.5 / 120.0 | 1.0378 | 32.5 / 32.5 |
| BM2-SC1 | water with 1 dB/cm at 500 kHz | bowl | 3.79 | 0.48 | 559.5 / 539.3 | 1.0375 | 61.0 / 60.5 |
| BM2-SC2 | water with 1 dB/cm at 500 kHz | piston | 5.03 | 0.70 | 114.4 / 111.0 | 1.0314 | 3.0 / 3.0 |

## Gates

**PASS  M21.PH1-SC1** — ITRUSST PH1 benchmarks 1 and 2 with source condition SC1: the field over the paper's comparison domain agrees with the Rayleigh integral to within the spread the intercomparison itself reported (10 % L-infinity across all eleven models), and the peak lands within two voxels

- `PASS` BM1-SC1: L-infinity vs the Rayleigh integral: 3.409e+00 % (limit 1.000e+01 %)
- `PASS` BM1-SC1: peak position error: 0.000e+00 mm (limit 1.000e+00 mm)
- `PASS` BM2-SC1: L-infinity vs the Rayleigh integral: 3.787e+00 % (limit 1.000e+01 %)
- `PASS` BM2-SC1: peak position error: 5.000e-01 mm (limit 1.000e+00 mm)

**PASS  M21.PH1-SC2** — ITRUSST PH1 benchmarks 1 and 2 with source condition SC2: the field over the paper's comparison domain agrees with the Rayleigh integral to within the spread the intercomparison itself reported (15 % L-infinity across all eleven models), and the peak lands within two voxels

- `PASS` BM1-SC2: L-infinity vs the Rayleigh integral: 3.824e+00 % (limit 1.500e+01 %)
- `PASS` BM1-SC2: peak position error: 0.000e+00 mm (limit 1.000e+00 mm)
- `PASS` BM2-SC2: L-infinity vs the Rayleigh integral: 5.031e+00 % (limit 1.500e+01 %)
- `PASS` BM2-SC2: peak position error: 0.000e+00 mm (limit 1.000e+00 mm)

## Run

```json
{
  "caustica": "0.1.0.dev0",
  "git_commit": "883d8776f30219f7e867ac3eb827f9fe1fbbfcc5",
  "host": "LOQ",
  "solver": "linear",
  "dx_mm": 0.5
}
```
