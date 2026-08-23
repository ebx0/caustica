# dev_validate — Colab session #3 (RTX PRO 6000 Blackwell, 2026-08-23)

`scripts/dev_validate.py --profile colab` @ `114b3f1`, all six U-stages. The JSON
is the operator-supplied evidence file; the gpu-gates report folder it references
stayed on the Colab VM (its parsed rung table is embedded here under U2.data).

What this session PROVED (and what each M7/M8 box cites):

- **U1 — the 256³ NaN is discriminated: `shape_itself`.** Fresh-process probes
  refuted process-state (cal256/warm256 still diverge), PML geometry (apex at
  0.15×extent still diverges, period-1 peak 189.3 MPa) and reproduction doubt.
  256³ on cupy corrupts within the first 20 steps (period-1 ~190.657 MPa vs a
  200 kPa drive; the same job on numpy reads 45 kPa and stays healthy); 400³ in
  the same process is clean. Bit-identical across A100 (Ampere) and this card.
- **U2 — the gate ladder after fix F2 (per-rung subprocesses):**
  M7.parity PASS, M7.fullsize PASS (512³ completed), **M8.vram PASS with worst
  deviation −0.32 %** (−0.29/−0.31/−0.32 % at 400³/512³/640³ — the −18…−35 %
  "drift" of session #1 was pool contamination, as diagnosed), M8.oom PASS
  (972³ refused, exit 3, advice). M8.time FAIL: −33.4/−36.8/−39.5 %.
- **The M8.time decomposition (the finding that shapes the fix):** subtracting
  each rung's measured warmup from its wall time gives steady step costs of
  25.75/52.7/103 ms vs predictions 23.65/49.61/96.89 ms — the t_step model was
  only **−6…−9 %** off. The gate failed on the WARMUP half: plans used the
  probe-fitted warmup (0.2/0.6/1.2/2.3 s) while real runs paid 4.39/12.84/32.38 s.
  Those real warmups are superlinear in P (per-elem 6.9e-8 → 9.6e-8 → 1.24e-7),
  so the linear two-term model cannot represent them either — U5 measured it
  over-predicting 2× at 192³ (ratio 0.50). Same medicine as t_step:
  interpolation over stored samples, not a fixed functional form.
- **U3** — calibration self-check: global-fit residual 0.705 (the a·P·log₂P+b·P
  form cannot fit this card even with saturating 216³/320³/432³ probes), yet the
  throughput anchor hit a 320³ spot-check at −9.7 %.
- **U4 — fix F4 end-to-end on a genuinely unknown card:** key
  `unknown:NVIDIA RTX PRO 6000…`, VRAM 94.97 GiB read from the device, and the
  negative control: token-only lookup does NOT find the calibration —
  device_name is load-bearing.
- **U6 — fix F1 on cupy:** a diverging job exits 4 with SolverDivergedError in
  error.json and leaves no result.h5.
