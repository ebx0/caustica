# Local GPU run

Written by `scripts/run_gpu_local.sh` on 2026-09-06T20:45:25Z (UTC).

| | |
|---|---|
| device | NVIDIA GeForce RTX 5050 Laptop GPU |
| mode | full |
| interpreter | /c/Users/bulbu/Desktop/hifusim/.venv/Scripts/python.exe |

| leg | command | exit | summary |
|---|---|---|---|
| gpu tests | `python -m pytest -m "gpu and not kwave and not network"` | 0 | 8 passed, 966 deselected, 1 warning in 2.06s |
| gate suite | `python -m caustica.validation gpu-gates` | 4 | overall: INCOMPLETE |

## Gates

- PASS        parity: numpy vs cupy on a mini 3-D scenario: field relative error < 1e-05
- INCOMPLETE  fullsize: a full-size run (dx=0.30 mm, 512^3 FFT class) completes without OOM
- PASS        vram: VRAM prediction within +/-10% of the mempool peak measured in the rung's OWN process, on at least 2 grid sizes
- PASS        time: post-calibration wall-time prediction within +/-25% of actual, on at least 2 scenarios (same device); a plan whose source is not 'calibrated' does not count
- PASS        oom: a run larger than the device is REFUSED before solving (exit 3) with advice

`fullsize` needs a 512-cubed run at dx = 0.30 mm, 13.2 GiB predicted, so
the ladder only reaches it on a device with about 15 GiB free. On a
smaller card that gate stays INCOMPLETE and the suite exits 4 however
green the rest is; the criterion closes on a hosted A100 or H100.

`time` is the least reproducible of the five gates: it grades a plan
built from a calibration probe taken seconds earlier, and on a laptop
card the probe moves with the clocks. One `time` FAIL is a reason to
run again, not a defect on its own.

Full output: `pytest-gpu.txt`, `gpu-gates.log`. The suite's own
report is `REPORT.md` beside them, with `gpu_gates.json` as its
machine half.
