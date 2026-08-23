# GPU gate session #1 — NVIDIA A100-SXM4-40GB (2026-08-22, Colab)

First run of `python -m caustica.validation gpu-gates` on a real device, at
commit `cc3046d`. **Overall verdict: FAIL — and every FAIL taught us something.**
This session (plus the same ladder on an RTX PRO 6000 Blackwell the same night)
exposed the four defects fixed in `aed5088` + `e2a1a23`:

- **F1** — the 256³ rung went NaN at period 2 and still exited 0 with a full
  result.h5; the suite then counted it as M8.vram evidence.
- **F2** — rungs shared one process, so cupy's monotone memory pool folded each
  rung's retained blocks into the next rung's "peak" (-2.0% → -18.6% → -35.3%
  apparent planner drift; the pool's 21.4 GiB also short-changed the free-VRAM
  gate and refused a 26.98 GiB rung the card can hold).
- **F3** — calibration probed 48³/72³, both launch-latency-bound on an A100
  (~1.0 ms/step each); the fit extrapolated 6.8× too slow, hence M8.time
  missing by +154%…+308%. Warmup also proved size-dependent (4.5/10.4/20.9 s),
  not the flat 3.0 s.
- **F4** — (Blackwell leg) an unrecognised device was silently planned as an
  A100: 95 GiB judged against 38.88 GiB usable, fresh calibration unreachable.

What PASSED, cleanly: **M7.parity** (numpy vs cupy, in-memory fp32: rel L∞
1.5e-6 phasor / 1.2e-6 p_max) and **M8.oom** (43.4 GiB refused, exit 3, with
advice). `parity_fields.npz` holds the four fields; `stored_float16_reference`
in the JSON shows the same fields after float16 quantization (~4.9e-4 = 1 ULP),
which is why the parity gate measures memory, not files.

The 256³ NaN root cause was still open when this was archived — identical on
both GPUs, absent on numpy/CPU with the same job. See MILESTONES M7/M8 notes.
