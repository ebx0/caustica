# caustica — Competitive Landscape and Gap Analysis, August 2026

**Question this report answers:** what would it actually mean for caustica to be "the best nonlinear
acoustic wave solver on the market for focused ultrasound", and what is the shortest credible path there?

**Research date:** all repository metrics, release dates and download figures were fetched
**2026-08-22** unless stated otherwise. Repository figures come from the GitHub REST API
(`api.github.com`), download figures from `pypistats.org` / `pepy.tech`, licences from the repo's own
`LICENSE` file or the API's SPDX field. Anything not confirmed against a primary source is marked
**`[unverified]`** and no date or version number in this document was inferred or reconstructed.

**Relationship to the earlier reports.** `research/gemini1.md` and `research/gemini2.md` (both
2026-08-10) laid out an architecture thesis. This report does **not** repeat them; it audits their
market claims against primary sources and updates them for 2025–2026. Several of their central
premises are now **false**. Section 0 lists those corrections first, because they change the strategy.

---

## Bottom line

1. **caustica's stated differentiator no longer exists.** `k-wave-python` v0.6.0 (2026-03-26) shipped a
   pure NumPy/CuPy k-space solver with power-law absorption and B/A nonlinearity, and v0.6.2/0.6.3rc1
   rebuilt the CUDA binaries through Blackwell. On physics, caustica is now a strict *subset* of it.
2. **What caustica has that nobody else has is the software contract**, not the physics: job JSON in →
   stamped, resumable, planner-gated run folder out, with frozen plugin and GUI seams. No solver in this
   field ships anything comparable. That is the position to defend.
3. **The one genuinely empty niche in the field is differentiable *nonlinear* acoustics.** j-Wave owns
   differentiability, is **linear-only**, and received **3 commits in 12 months**. High cost, high reward.
4. **"Most accurate solver" is not a winnable claim.** The 2026 literature puts end-to-end transcranial
   error at 20–77% and attributes it to skull property mapping, *not* to the wave model.
5. **The shortest path to standing is unusually cheap and is not being taken:** the ITRUSST Phase-1
   benchmark data, skull maps and comparison harness are public and self-serve, and the benchmarks are
   **500 kHz CW — exactly caustica's regime**. Run them, publish the position.
6. **Order of work:** GPU (G1) → PyPI + run-contract framing (G5) → power-law absorption (G3) → ITRUSST
   (G2) → a measured CW speed claim (G4) → thermal/CEM43 (G7) → UQ (G8).

---

## 0. What changed since gemini1/gemini2 — corrections that matter

These are not nuances. Each one invalidates a differentiator the earlier reports treated as caustica's
core advantage.

| gemini1/gemini2 claim | 2026-08-22 reality | Consequence |
|---|---|---|
| "A pure-Python NumPy/CuPy k-space solver with no precompiled binaries is the gap in the market." | **`k-wave-python` v0.6.0 (2026-03-26) added exactly that.** Release note, verbatim: *"k-wave-python can now be run completely in Python and no longer depends on the k-wave binary executables."* Its `kwave/solvers/kspace_solver.py` (≈41 kB) is a NumPy/CuPy k-space PSTD engine for 1/2/3-D with **fractional-Laplacian power-law absorption, Kramers–Kronig dispersion and B/A nonlinearity** — a strict superset of caustica's physics. | **caustica's headline differentiator is gone.** It survived roughly five months after being written down. |
| "k-Wave's CUDA binaries are broken on A100/H100/Colab; JIT-compiled Python is essential." | Half true, half stale. *Upstream* k-Wave still ships **C++/CUDA binaries v1.3 (2020-02-28)** advertised for **SM 3.0–7.5**, i.e. not Ampere/Hopper. But `k-wave-python` **rebuilt them**: v0.6.2 (2026-05-18) ships k-Wave 1.4.1 Linux CUDA binaries including Blackwell `sm_120`, and v0.6.3rc1 (2026-06-21) is described as the *"first unified binary release supporting Turing through Blackwell GPUs."* | The "k-Wave doesn't run on modern GPUs" wedge has been closed **by the competitor**, not by caustica. |
| "k-Wave is GPL; commercial labs legally cannot integrate it; MIT is a decisive market advantage." | **k-Wave is LGPL**, and k-wave.org states explicitly: *"The copyleft restrictions only apply directly to the toolbox, but not to other (non-derivative) software that simply links to or uses the toolbox."* `k-wave-python`, `j-Wave` and `jaxdf` are all **LGPL-3.0**. | MIT is still a genuine but **much smaller** edge than claimed. It matters for vendoring/static embedding, not for "can a company use it at all." |
| "j-Wave is the differentiable competitor to beat." | j-Wave is **linear-only** — a GitHub code search for `nonlinear` across `ucl-bug/jwave` returns **0 hits** (control search for `attenuation` returns 13), and `time_varying.py` contains no B/A term. Its absorption is fixed at **y = 2** (`α₀ω²`), not a tissue power law. And it is **release-stalled**: last PyPI release **0.2.1, 2024-09-17**; **3 commits in the 12 months to 2026-08-22**; open issue #247 (2026-04-03) asks for a release supporting modern JAX and is unanswered. | The differentiability niche is **abandoned, and it was never nonlinear.** This is the single largest genuinely open gap in the field (§3, G6). |
| Ecosystem framed as "k-Wave vs j-Wave vs BabelBrain vs FullWAVE(closed)". | **Fullwave went open source.** `pinton-lab/fullwave25` was created **2025-10-28**, is **LGPL-3.0**, `pip install fullwave25` (PyPI 1.2.3, 2026-02-24), CUDA/C core with **multi-GPU 2-D and 3-D**, spatially varying power-law *exponent* γ, and a per-voxel nonlinearity parameter `beta = 1 + B/A/2` (verified in `fullwave/medium.py`). Its README explicitly targets *"a user experience similar to k-Wave and k-wave-python."* | A new, fast-moving, nonlinear, multi-GPU, Python-API competitor appeared 10 months ago. |
| Implicit: "the solver is the bottleneck in transcranial accuracy." | Two 2026 papers say otherwise, independently. See §2.4. | Being the best *solver* buys less end-to-end accuracy than assumed. Positioning must account for this. |

---

## 1. The 2026 landscape

### 1.1 Master comparison

Sorted roughly by threat level to caustica. "Health" = commits on the default branch in the 12 months
to 2026-08-22 (capped at 100 by the API query).

| Tool | Latest release (verified) | Stars | Licence | Nonlinear | Absorption model | Heterog. | Elastic / skull | Thermal / CEM43 | GPU story | Health |
|---|---|---|---|---|---|---|---|---|---|---|
| **k-wave-python** | v0.6.2 **2026-05-18** (rc: v0.6.3rc1 **2026-06-21**) | **220** | LGPL-3.0 | **yes** (B/A) | **power law, fractional Laplacian + K-K dispersion** | voxel | via k-Wave `pstdElastic` (CPU) | via k-Wave `kWaveDiffusion` | **both**: CC≥7.5 binaries incl. Blackwell **and** pure NumPy/CuPy backend | **100+ commits/yr** |
| **k-Wave (MATLAB)** | v1.4.1 **2025-12-17**; C++/CUDA binaries **v1.3, 2020-02-28** (SM 3.0–7.5) | n/a | LGPL | **yes** | same | voxel | yes (CPU only) | yes | binaries stale upstream | maintenance only; **k-Wave-I declared end-of-line 2026-06-08** |
| **Fullwave 2.5** | PyPI 1.2.3 **2026-02-24**; push **2026-08-03** | 49 | LGPL-3.0 | **yes** (per-voxel β) | **multi-relaxation power law with spatially varying α₀ *and* γ** — richest in the field | voxel | not documented | no | **CUDA/C multi-GPU, NVIDIA+Linux only, no CPU fallback** | new + active (created 2025-10-28) |
| **BabelBrain** (+ BabelViscoFDTD) | BabelBrain v0.8.1 **2026-05-10**, prerelease v0.8.8 **2026-08-19**; BabelViscoFDTD PyPI 1.2.7 | 48 / 62 | BSD-3-Clause / `NOASSERTION` | **no** | relaxation-based Q | voxel | **yes — viscoelastic (shear in skull)** | **yes — Pennes BHTE + CEM43 + MI** | **CUDA + Metal + OpenCL** (only serious Apple-Silicon tool) | **100+ commits/yr** |
| **Stride** | **no software release ever** (only a 2021 dataset tag); push 2026-07-24 | 134 | **AGPL-3.0** | no | y ∈ {0, 2} only | voxel | `iso_elastic` present `[unverified maturity]` | no | Devito → OpenACC, cluster-scale | 66 commits/yr, but **infra work, not physics** |
| **j-Wave** | 0.2.1 **2024-09-17** | 214 | LGPL-3.0 | **no** | **y = 2 fixed** | voxel | no (issue #240 open since 2024) | no | JAX: CPU/GPU/TPU, **differentiable** | **3 commits/yr — release-stalled** |
| **UltraWave** | v0.2.0 `[unverified date]`; IEEE TUFFC 2026 | ~10 | MIT | no | power law | voxel | **yes (acoustic + elastic)** | no | **OpenACC + MPI multi-GPU**; claims 8 h (k-Wave, 1 GPU) → 20 min (4 GPU) | brand new |
| **OptimUS** | v0.2.1 **2025-03-17** | 34 | **MIT** | no | complex-k (frequency domain) | **piecewise-homogeneous only** | no | BioHeat added 2025 | **none** | 1 commit/yr |
| **mSOUND** | "beta 0.2", site-dated **2022-06-27**; push 2025-06-05 | 19 | GPL-3.0 | **yes** | spatially varying power law | voxel | no | not documented | **none documented** | **0 commits/yr** |
| **FOCUS** | 0.981, **2025-07-28** | n/a | **freeware, closed-source, binaries only** | `[unverified]` | — | no (layered) | no | `[unverified]` | `[unverified]` | released, but closed |
| **FDA HITU Simulator** | push **2019-11-26** | 36 | GPL-3.0 | **yes** (wide-angle KZK) | layered | axisym. layered | no | **yes** | none | **abandoned ~6.7 yr** |
| **Kranion** | **no release ever**; push **2022-04-26** | 49 | MIT | — (visualiser, ray tracing; not a solver) | — | — | — | — | render only | dormant; current builds behind a gated user group |
| **k-Plan** (Brainbox, commercial) | version not published | n/a | commercial, **"Research Use Only… non-clinical"** | via k-Wave | via k-Wave | voxel | via k-Wave | **yes (heat + perfusion)** | cloud | commercial |
| **Sim4Life** (ZMT, commercial) | web V9.2 `[unverified date]` | n/a | commercial | **yes** (P-ACOUSTICS linear + nonlinear) | dispersive | voxel, CT-based skull | yes | **yes** | cloud | commercial |
| **caustica** | not released; public since **2026-08-22** | **0** | **MIT** | **yes** (Westervelt) | **exponential only** | voxel | no | no | CuPy packaged, **never executed on a GPU** | 69 commits since 2026-08-10 |

### 1.2 Adoption, measured

PyPI downloads, last 30 days, fetched 2026-08-22 — the only comparable adoption metric across the field:

| Package | Downloads / 30 d |
|---|---|
| `k-wave-python` | **4,767** |
| `jwave` | 364 |
| `babelviscofdtd` | 317 |
| `stride` | 280 |
| `fullwave25` | **117** |
| `caustica` | not on PyPI (404) |

Read this carefully: **`k-wave-python` alone out-downloads every other Python solver in this table
combined by roughly 4.4×** (4,767 vs 1,078), and it is the one that just absorbed caustica's
differentiator. Meanwhile Fullwave 2.5,
the most physics-complete new entrant, has only 117 downloads/month — **the field is not
winner-take-all on merit; distribution and brand dominate.** "k-Wave" is the brand.

### 1.3 The structural fact nobody states plainly

**SimNIBS `charm` (segmentation) + k-Wave (solver) is the de-facto substrate under almost every TUS
planning tool that is not BabelBrain.** k-Plan, PRESTUS (`Donders-Institute/PRESTUS`, GPL-3.0, v0.6.1
2026-05-22, active), TUSX (BSD-3, dormant since 2022), and BRIC_TUS_Simulation_Tools all call k-Wave.
PlanTUS (MIT, 31 stars, *Brain Stimulation* 2025) is a geometric planner that **exports to** Brainsight,
Localite, k-Plan, k-Wave and BabelBrain — and BabelBrain's 2026-08-19 prerelease now **bundles a frozen
copy of PlanTUS**. SimNIBS itself has **no acoustic module** (TMS/TES only).

Two implications:
1. A new solver does not enter the TUS market by being better in isolation — it enters by becoming
   **substitutable underneath an existing planner**. The cheapest real adoption path for caustica is to
   be a drop-in solver backend that PRESTUS/PlanTUS-style pipelines can call.
2. The ecosystem is **consolidating** (BabelBrain absorbing PlanTUS), which shortens the window.

### 1.4 Per-tool notes worth keeping

- **k-Wave itself is winding down.** k-wave.org announced **2026-06-08**: *"k-Wave Version 1.4.X will be
  the last full version of k-Wave-I. It will eventually be superseded by k-Wave-II, a community-driven
  major re-write."* `ucl-bug/k-wave-ii` exists — **9 stars, 56 open issues, MATLAB (R2023b+),
  object-oriented, pre-release, no licence file, releases planned every six months.** Read that
  correctly: **UCL is rewriting k-Wave in MATLAB again, from a near-zero base.** The Python-native
  centre of gravity has moved to `k-wave-python`, which is a *different maintainer* (Walter Simson).
  This is a genuine opening — but `k-wave-python` is already standing in it.
- **k-wave-python's pure-Python backend is a portability path, not a speed path.** Its own published
  3-D benchmarks (T4): Python/GPU **6 / 71 / 382 s** at 64³ / 128³ / 256³ versus C++/CUDA **2 / 7 / 51 s**
  — roughly **7.5× slower at 256³**. It also hard-pins dependencies (`scipy==1.15.3`, `numpy<2.3.0`,
  `h5py==3.16.0`), which is hostile in shared environments. Open issue **#760 (2026-06-21)**: the C++
  backend silently returns transposed sensor data when `n_sensor < Nt`.
- **BabelBrain is linear.** BabelViscoFDTD is a viscoelastic FDTD (Virieux staggered grid, relaxation
  absorption) with **no nonlinear term**. It is the right tool for low-intensity TUS and the wrong tool
  for HIFU ablation harmonics, shock formation or cavitation-relevant peak pressures. It is also the
  only tool here with a complete image→field→temperature→CEM43 workflow and a GUI.
- **The dead and dying:** mSOUND (0 commits/yr, still "beta 0.2" four years on, no GPU), OptimUS (1
  commit/yr, no GPU, piecewise-homogeneous only — cannot represent voxel CT), HITU Simulator
  (abandoned 2019 but still listed by the FUS Foundation), TUSX (dormant 2022), Kranion (dormant 2022).
  §5 draws the lesson from *why* these died.

### 1.5 The ML / learned-surrogate flank

Not competitors for the same job, but they reshape what a solver is *for*.

- **TUSNet** — Naftchi-Ardebili, Singh, Popelka, Butts Pauly (Stanford), *"A deep-learning model for
  one-shot transcranial ultrasound simulation and phase aberration correction"*, arXiv:2410.19995
  (submitted 2024-10-25), published in *Medical Physics* 2026. Computes transcranial pressure fields
  **and** phase corrections in **21 ms, >1200× faster than k-Wave**, with **98.3%** peak-pressure
  accuracy and **0.18 mm** mean focal-position error versus k-Wave ground truth. **Crucially it is
  2-D only**, and its ground truth *is* k-Wave — i.e. it inherits, and cannot exceed, the solver's
  accuracy. Code is public: `kbp-lab/TUSNet`, MIT, 9 stars, last push 2025-02-10.
- **TFUScapes + DeepTFUS** — Srivastav et al. (CAMMA, Strasbourg), *"A Skull-Adaptive Framework for
  AI-Based 3D Transcranial Focused Ultrasound Simulation"*, arXiv:2505.12998; repo `CAMMA-public/TFUScapes`
  (created 2025-05-16, 14 stars). **The first large-scale public dataset of 3-D transcranial FUS
  simulations**, hosted on Hugging Face (`vinkle-srivastav/TFUScapes`), built with an MRI→skull pipeline
  and a scalable **k-Wave** simulation engine. Note two details that matter a great deal here: *"each
  simulation returns a **steady-state pressure field**"*, and the generator was **k-Wave**.
- Other work in the same direction: *"Finite difference-embedded UNet for solving transcranial
  ultrasound frequency-domain wavefield"* (JASA 155(3):2257, 2024); *"Iterative Born Solver for the
  Acoustic Helmholtz Equation…"* (arXiv:2507.16087, 2025).

**Three consequences for caustica.**
1. **Do not race the surrogates on inference speed.** A 21 ms network cannot be beaten by any PSTD
   solver, ever.
2. **Every surrogate needs a solver as its ground truth**, and every one of them currently uses k-Wave
   for that. Being *the* data factory — reproducible jobs, planner-gated sweeps, stamped HDF5, an
   `≤10 MB` preview format — is a role caustica is already built for and nobody else is competing for.
3. **The surrogates validate the CW steady-state framing.** TUSNet predicts a steady-state field;
   TFUScapes explicitly stores steady-state pressure fields. **The largest public ML dataset in this
   field is exactly the quantity caustica's solver is built to emit.** The market for CW-phasor output is
   real, and the ad-hoc "scalable simulation engine pipeline" TFUScapes had to build around k-Wave is
   precisely the job caustica's job/planner/HDF5/resume machinery already does properly.

---

## 2. What "best" means — the criteria users actually apply

### 2.1 The decision criteria, ordered by observed weight

Derived from what the tools that won actually optimised for, plus what the 2026 accuracy literature
says the field is worried about.

1. **Provenance and citability.** In practice the first filter is "has this been used in a published
   study I can cite, and did it appear in the ITRUSST intercomparison?" k-Wave wins here structurally.
   A solver with zero papers is not evaluated on merit; it is not evaluated at all.
2. **Benchmark position.** Concretely: agreement with the ITRUSST PH1 corridors and with k-Wave. Not
   "is it right" but "is it inside the band the community already accepted."
3. **Installation cost.** `pip install X` and a working Colab cell versus MATLAB licences, NVIDIA HPC
   SDK, or a compiler. This is why `k-wave-python` beat everything, and why FOCUS/mSOUND lost despite
   being older and well-validated. **Fullwave 2.5 requires an NVIDIA GPU with no CPU fallback and is
   Linux-only** — that is a real, exploitable weakness.
4. **Skull support end-to-end** (CT or ZTE/pseudo-CT → acoustic properties → field), for the TUS half of
   the market. This is table stakes for neuromodulation and **caustica has none of it.**
5. **Thermal / CEM43**, for the HIFU-ablation half. BabelBrain, k-Plan, HITU Simulator and Sim4Life all
   have it; k-Wave has `kWaveDiffusion`. **A HIFU treatment-planning tool without a thermal model is not
   perceived as a treatment-planning tool.**
6. **Speed and memory at realistic problem sizes**, in wall-clock on hardware people actually have (one
   Colab GPU; a single A100). Multi-GPU is a differentiator only for the handful of groups with clusters.
7. **Documentation and worked examples.** k-wave-python ports 29 examples; BabelBrain ships a GUI.
8. **Differentiability**, for the ML-adjacent minority — small but growing, and currently **unserved**
   for nonlinear problems.
9. **Licence.** Matters at the margin for commercial embedding; LGPL is already permissive enough that
   MIT is a tiebreaker, not a wedge.

### 2.2 ITRUSST: current status, and the practical way in

**Phase 1** — Aubry et al., *"Benchmark problems for transcranial ultrasound simulation:
Intercomparison of compressional wave models"*, **JASA 152(2):1003, 2022** (arXiv:2202.04552). Nine
benchmarks (`PH1-BM1`…`BM9`), two 500 kHz source conditions (focused bowl, plane piston) → 18
permutations, spanning homogeneous water, single-layer bone, multi-layer, and full skull. **Eleven
modelling tools** participated, spanning FDTD, angular-spectrum, pseudospectral, boundary-element and
spectral-element methods. **The corridors actually reported in the abstract are: median focal-pressure
difference < 10%, median focal-position difference < 1 mm.** A −6 dB focal-volume corridor is *not*
stated in the abstract `[unverified — the per-benchmark tolerances quoted in MILESTONES.md M21 as
"0.2–0.6 mm" should be re-checked against the paper body before being used as an acceptance criterion]`.

**The eleven codes in the club**, read directly from `intercomparison/getModelNames.m`:
`BABELVISCOFDTD`, `FULLWAVE`, `GMFDTD`, `HAS`, `JWAVE`, `KWAVE`, `MSOUND`, `OPTIMUS`, `SALVUS`,
`SIM4LIFE`, `STRIDE`. This list *is* the field's membership roster — note that it includes the dormant
(mSOUND, OptimUS) and the commercial (Salvus, Sim4Life) alike. Membership is what confers standing, not
activity. **caustica is not on it, and that is the single most consequential absence in this report.**

**The entry path is open and self-serve — this is the most actionable finding in this report.**
The intercomparison library is public at
[`ucl-bug/transcranial-ultrasound-benchmarks`](https://github.com/ucl-bug/transcranial-ultrasound-benchmarks)
(LGPL-3.0, 23 stars, 5 forks) with the data on **Zenodo, doi:10.5281/zenodo.6020543** — including the
skull `.stl` models for BM7–BM9, the affine transducer transforms, **pre-rasterized `SKULL-MAPS`**, and
**every participating model's result files**. Its README documents the procedure verbatim:

> **Adding new model results**
> 1. Decide on a name for the model, e.g., `NEWMODEL` and add this to the list `intercomparison/getModelNames`.
> 2. Create a `NEWMODEL` folder in the downloaded results. Add the model results following the naming convention, e.g., `PH1-BM1-SC1-NEWMODEL.mat`.
> 3. To compare results for one benchmark, call `compareTwo` as outlined above.
> 4. To re-generate the intercomparison results with the new model, run `processAll`.

**No invitation, no consortium membership, no gatekeeper.** caustica can produce
`PH1-BM1-SC1-CAUSTICA.mat`, add `'CAUSTICA'` to `getModelNames.m`, run `processAll`, and obtain its
position against all eleven published codes — locally, today, for the cost of a MATLAB session (or a
reimplementation of `compareTwo`'s metrics, which is modest). The repo has five public forks, one of
them by `ProteusMRIgHIFU` (the BabelBrain author) — evidence that this path is actually used.

**Two important caveats.**
- The benchmark repo has **not been touched since 2022-02-10**. There is **no public Phase 2** in it. A
  formal round-2 (elastic, thermal, or heterogeneous-skull) is `[unverified]` — I found no primary
  evidence of one. Treat "ITRUSST benchmarks" as *Phase 1, frozen*.
- The benchmarks are **500 kHz continuous-wave, steady-state**. **caustica's CW-only restriction is
  therefore not a blocker for ITRUSST** — a genuinely favourable alignment, and it should be exploited
  early rather than deferred to M21.

**Related ITRUSST consensus output** (these are standards/reporting papers, not benchmarks): *ITRUSST
consensus on standardised reporting for transcranial ultrasound stimulation* (PMID 38670224, 2024) and
*ITRUSST consensus on biophysical safety for transcranial ultrasound stimulation* (arXiv:2311.05359;
published in *Brain Stimulation*, ScienceDirect S1935861X25003535 / PMID 41072763). Conforming to the
**reporting** consensus in caustica's report output is a cheap credibility signal.

### 2.3 Other verification anchors

Beyond ITRUSST, the accepted ladder is analytic first: **O'Neil (1949)** focused bowl, **Rayleigh
integral**, **Fubini/Blackstock** nonlinear harmonic growth, plane-wave dispersion and absorption.
**caustica already passes all of these** (README "Validation" section: focus within 1 voxel, axial
r > 0.99, −6 dB widths < 5%; A2/A1 within 0.9–3.2% across σ = 0.06–0.61; phase-speed error < 0.1% at
4 ppw), **plus** a cross-validation against a real k-Wave OMP binary in 2-D water (r > 0.99). That is a
stronger starting evidence base than most new entrants have — it is simply invisible, because nobody
has seen it.

### 2.4 The uncomfortable finding: the solver is not the bottleneck

Two independent 2026 studies converge on this, and any "best solver" positioning must survive it.

- **Brandts et al., medRxiv 2026-01-22** (doi 10.64898/2026.01.20.26344460, PMID 41646764; subsequently
  in *Brain Stimulation*), *"Empirical limitations of current low-intensity focused ultrasound
  simulation platforms"* — the first head-to-head of **k-Plan vs BabelBrain** on identical subjects and
  trajectories. Finds substantial discrepancies between platforms and between CT and pseudo-CT, with
  safety-relevant divergence (k-Plan crossing a 2 °C threshold where BabelBrain did not, in 3 of the
  subjects). Root cause stated by the authors: *"the larger discrepancies stemmed primarily from
  differences in how each platform derives skull acoustic properties and models attenuation, rather than
  from the compressional-wave models."*
- **Li et al., arXiv:2606.09497 (2026-06-08)**, *"Experimental Validation of Skull Acoustic Modelling
  Strategies…"* — five k-Wave skull models against hydrophone measurement: **peak-pressure errors
  20–31%, intensity errors 41–77%, focal-volume errors 11–67%**, focal position off by several
  millimetres, and a systematic **underestimation of skull attenuation** (i.e. error in the *unsafe*
  direction). *"No model showed a consistent advantage across all metrics."*
- For context, earlier work reports pseudo-CT-vs-CT mean differences of ~9.9% focal pressure / 1.5 mm
  for T1-based, ~5.7% / 0.6 mm for ZTE-based pseudo-CT.

**Strategic reading.** Numerical-solver differences among the credible codes are at the few-percent
level; end-to-end error against reality is 20–77%. Therefore:
- "Most accurate solver" is a **weak, unfalsifiable and slightly dishonest** market position.
- The defensible positions are the ones that address *this* problem: **uncertainty quantification over
  property maps**, **reproducibility of a given run**, and **throughput** (enough speed to sweep the
  uncertain parameters instead of guessing them once). All three are software problems, and caustica is
  unusually well-equipped for all three.

---

## 3. Gap analysis — where caustica can actually win

### 3.1 Honest statement of caustica's position, 2026-08-22

**Has:** linear + Westervelt k-space PSTD in 1/2/3-D (NumPy); a k-Wave adapter as a registry solver,
cross-validated; analytic reference layer (O'Neil, Rayleigh, Fubini, plane wave); exact-period `dt`;
adaptive settling with in-solver steady-state phasor and multi-harmonic capture; CSG geometry + label
volumes + `medium_volume` format; a pre-run planner (byte-level VRAM inventory + wall-time model with
`db`/`calibrated`/`measured` provenance); a full `caustica-job/1` JSON job contract with `validate`,
disjoint exit codes, heartbeat, bit-exact resume, checkpoints; HDF5 result contract with atomic writes;
report + ≤10 MB preview; a one-call façade; a Colab bridge; a frozen GUI contract; a five-axis
entry-point plugin architecture; MIT licence; ~402 tests.

**Lacks:** any GPU execution ever (CuPy is packaged, never run); power-law absorption; broadband /
transient drive (`drive = {"cw"}`); thermal; elastic/skull; axisymmetric; KZK; differentiability;
multi-GPU; any CT/MRI → acoustic-property pipeline; a PyPI release; a paper; a user.

**The decisive framing:** on *physics*, caustica is a strict subset of `k-wave-python`. On *software
contract* — reproducible jobs, planning before booking a GPU, plugin seams, a frozen GUI/automation
surface — **caustica has no competitor at all.** Nobody else in this field ships a run contract. That is
where the differentiation actually is, and it is already built.

### 3.2 Prioritised gaps

Effort in engineer-weeks assuming the current pace; impact is on *credibility + adoption*, not on
personal interest. **Risk** is the chance the work fails to pay off, not the chance it is hard.

| # | Gap / move | Effort | Impact | Risk | Verdict |
|---|---|---|---|---|---|
| **G1** | **Run on a GPU and publish the numbers** (M7 + M8 Colab half). Everything below is blocked on this; the entire planner model is currently unfalsified, and "GPU-accelerated" in the README is a claim with no evidence. | 1–2 wk | **critical** | low | **Do first. Nothing else counts until this is done.** |
| **G2** | **ITRUSST PH1 self-serve entry** (§2.2). The data, skull maps and comparison harness are already public; the benchmarks are 500 kHz **CW**, which caustica's steady-state path natively suits. Start with BM1–BM6 (water / single-layer / multi-layer), defer BM7–BM9 (full skull) until G3. | 3–5 wk (BM1–6) | **critical** | low–med | **Do second.** This is the credibility gate, and it is unusually cheap because the entry path is open. |
| **G3** | **Power-law absorption via fractional Laplacian** (M16). Not optional: skull is α ∝ f^~1.1–1.2, ITRUSST bone layers are specified this way, and any multi-frequency comparison with k-Wave is meaningless without it. Every serious competitor has it; **exponential-only is the single most visible physics deficiency.** | 3–4 wk | **critical** | low | **Do third.** Table stakes. In k-space it is a `\|k\|^y` multiply — cheap relative to its credibility value. |
| **G4** | **Make the CW steady-state fast path a measured, published claim.** Everyone else runs a full transient and post-processes; caustica converges to a phasor with exact-period `dt`, adaptive settling and in-GPU harmonic extraction. For CW therapy — the dominant use case, and the ITRUSST regime — this should be a real wall-clock and VRAM win. **It is already built; what is missing is a measurement.** Target: a table of caustica vs `k-wave-python` (both backends) for identical CW problems. | 2–3 wk (after G1) | **high** | med (the win must be real; if it is <2× it is not a story) | **Highest ratio of impact to remaining effort in this table.** |
| **G5** | **Ship the run contract as the product story.** PyPI release; frame caustica as *"reproducible simulation runs"* — job JSON in, stamped run folder out, plan before you book a GPU, resume bit-exact, plugin seams, frozen automation surface. No competitor has any of this: `k-wave-python` is a library, BabelBrain is a monolith, Fullwave 2.5 is an engine. Also starts the **JOSS six-month public-history clock** (§4). | 1–2 wk | **high** | low | **Do alongside G1–G3.** This is the differentiator that survived 2026. |
| **G6** | **Differentiable *nonlinear* solver.** The genuinely unoccupied niche: j-Wave owns differentiability but is **linear-only and dormant**; nothing in the field offers autodiff through a nonlinear acoustic solve. Applications: phase optimisation, aberration correction, property inversion. But CuPy has no autodiff — this needs either a hand-written adjoint through the PSTD loop or a second (JAX/PyTorch) backend, and memory for reverse-mode over 10³–10⁵ steps is brutal (checkpointing mandatory). | **12–20 wk** | **very high if it lands** | **high** | **Scope a feasibility spike (2 wk) in the 6–12 month window; do not commit blind.** The plugin backend seam makes a JAX backend architecturally cheap; the adjoint is the hard part. |
| **G7** | **Thermal: Pennes + CEM43** (M18). Expected by anyone who says "HIFU treatment planning" — BabelBrain, k-Plan, HITU Simulator and Sim4Life all have it. Caustica already has `HeatingSource` as a planned sensor hook and the material DB already carries thermal fields. Straightforward diffusion+perfusion on the existing GPU backend. | 4–6 wk | **high** (unlocks the HIFU-ablation half of the market) | low | **Do in months 4–8.** Low risk, well-understood physics, large perceived-completeness gain. |
| **G8** | **Uncertainty quantification over property maps.** Attacks the field's *actual* bottleneck (§2.4): ship parametric sweeps over uncertain skull/tissue properties with error bars on focal pressure and position. Precedent exists (Martin & Treeby, linear uncertainty propagation, JASA-EL 2023, arXiv:2212.04405) but **no library ships it turnkey.** caustica's job/sweep/planner machinery makes this nearly free — it is a `Study` over existing parts. | 3–5 wk (after G7) | **high, and uniquely defensible** | med (needs a convincing demo) | **Strong candidate for the differentiating paper.** Reframes caustica from "another solver" to "the tool that tells you how much to trust the answer." |
| **G9** | **Broadband / transient drive** (M17). Needed for time-reversal aberration correction, pulsed protocols, imaging and TOF work. Moderate effort (generalise the source and the spectral extraction; windowed/decimated recording). **Not needed for ITRUSST or for CW therapy** — so it is important but not urgent. | 4–6 wk | med–high | low | **Months 6–12.** Sequence after G7. |
| **G10** | **CT / ZTE → acoustic property pipeline + NIfTI/DICOM I/O.** Table stakes for TUS, but a crowded, thankless and scientifically contested area (§2.4 shows nobody has a validated mapping). Better consumed than reimplemented. | 4–8 wk | med | **high** (contested science, easy to get wrong) | **Do not build. Integrate.** Become a solver backend PRESTUS/PlanTUS-style pipelines can call (§1.3). Keep it in a companion repo, as with `uwcem-phantom`. |
| **G11** | **Elastic / viscoelastic skull** (M22). Occupied on three sides: BabelViscoFDTD, UltraWave, k-Wave `pstdElastic`. Very large effort, and shear is a second-order correction at TUS frequencies. | **> 20 wk** | med | high | **Skip in the 6–12 month window.** |
| **G12** | **Multi-GPU** (M20). Occupied by Fullwave 2.5 and UltraWave. Global 3-D FFTs make PSTD domain decomposition communication-bound, and Colab gives one GPU. | 8–12 wk | low–med | high | **Skip.** A written feasibility/negative-result note is a valid and much cheaper deliverable. |
| **G13** | **ML surrogate / dataset generation.** caustica's origin, and the natural home for its job/sweep/HDF5/preview machinery. But note the competition: **TUSNet** (arXiv:2410.19995) reports transcranial fields + phase correction in **21 ms, >1200× faster than k-Wave, 98.3% peak-pressure accuracy, 0.18 mm focal error** — but **in 2-D, trained on k-Wave output.** Surrogates win on inference speed and are structurally dependent on a solver for ground truth; the durable role is **data factory and verifier**, not competitor. The precedent is **TFUScapes** (arXiv:2505.12998) — which had to hand-build a "scalable simulation engine pipeline" around k-Wave to emit **steady-state** fields, i.e. it reinvented, badly, what caustica already ships. | 4–6 wk (pipeline) | med–high | med | **Position as "the trainer's solver", not "the surrogate".** Publishing a large open dataset — or being the engine under someone else's — is a cheaper reputational win than publishing another model. |

### 3.3 Answers to the specific questions asked

- **Broadband vs CW-only — how much does it cost?** Less than feared. The ITRUSST benchmarks are 500 kHz
  CW; CW steady state is the dominant regime for both TUS neuromodulation and HIFU sonication. CW-only
  **excludes** imaging/tomography (Stride/k-Wave territory, not caustica's market) and **blocks**
  time-reversal aberration correction. Verdict: **not a market blocker, but a ceiling on aberration
  work** — sequence it at months 6–12 (G9), not now.
- **Is power-law absorption mandatory for ITRUSST?** Effectively **yes**. Bone attenuation in the
  benchmarks is frequency-dependent and every comparison code models it that way; a frequency-independent
  exponential fit will diverge from the published corridors in exactly the layers that matter.
  Additionally, an exponential-only solver reads as unfinished to any reviewer. **G3 is not deferrable.**
- **How much of the market is skull/elastic?** The *skull* is most of the TUS market — but skull work is
  overwhelmingly done with **compressional-only** models (all eleven ITRUSST Phase-1 codes; k-Wave;
  k-Plan; PRESTUS). *Elastic/shear* is a minority refinement served by BabelViscoFDTD and UltraWave.
  So: **skull geometry and skull absorption are essential (G3, G10-by-integration); elastic is not (G11).**
- **Thermal/CEM43 in HIFU planning — expected?** **Yes, unambiguously.** It is the one capability shared
  by every tool that positions as treatment planning. Its absence is what makes caustica read as a
  physics engine rather than a planning tool.
- **Is differentiability j-Wave's monopoly?** **No — it is a vacancy.** j-Wave is linear-only (0 code
  hits for `nonlinear`), stuck on a 2024 PyPI build, and received 3 commits in the last year.
  Differentiable *nonlinear* acoustics does not exist anywhere. Highest upside, highest cost (G6).
- **ML surrogate / dataset generation.** The surrogates are already faster than any solver will ever be
  (TUSNet: 1200×). Compete as the **data factory and ground-truth verifier**, not the predictor (G13).
- **Multi-GPU.** Skip (G12). **Clinical workflow integration.** Integrate, don't build (G10).

### 3.4 The recommended 12-month sequence

| Window | Work | Outcome |
|---|---|---|
| Months 0–2 | **G1** (GPU, real numbers) → **G5** (PyPI + run-contract framing) | caustica is installable, runs on a GPU, and the JOSS clock starts. |
| Months 2–5 | **G3** (power-law) → **G2** (ITRUSST BM1–BM6) → **G4** (measured CW speed claim) | caustica has a defensible physics scope, a published benchmark position against 11 codes, and a quantified reason to choose it. |
| Months 5–9 | **G7** (thermal + CEM43) → **G2 continued** (BM7–BM9, full skull) | caustica reads as a treatment-planning-capable tool, with the hardest benchmarks answered. |
| Months 9–12 | **G8** (UQ) + **G9** (broadband); **G6 feasibility spike** (2 wk, timeboxed) | A differentiating scientific contribution, and a go/no-go on the one genuinely open niche. |

---

## 4. The credibility chain

A new solver is not believed because it is good. It is believed because of an evidence chain, in this
order — and each link is worthless without the ones before it.

1. **Analytic.** O'Neil, Rayleigh, Fubini/Blackstock, plane-wave dispersion and absorption.
   **caustica already has this, automated in `pytest`.** Its only defect is that it is invisible.
   *Action: put the numbers on the README and in the docs as a validation page with figures — done, but
   under-advertised.*
2. **Cross-code.** Against k-Wave on identical grids/media/sources. **caustica already has this in 2-D
   water via a real OMP binary.** *Action: extend to 3-D and to heterogeneous media after G1/G3.*
3. **Community benchmark.** ITRUSST PH1. **This is the link that converts a private project into a
   recognised code**, and §2.2 shows the door is unlocked. *Action: G2. Publish results under
   `benchmarks/reports/itrusst/` with the same JSON+Markdown stamping the repo already uses.*
4. **Experimental.** Hydrophone comparison. Expensive, requires a collaborator with a tank, and §2.4
   shows the honest result will be 20–30% error like everyone else's. *Action: do not attempt alone;
   this is what a collaboration is for. Do not let its absence block 1–3.*
5. **Publication.** See below.

### 4.1 Which paper, and when

**Both, staged — but in this order and not before the prerequisites.**

**(a) A software paper first — JOSS.** JOSS explicitly requires *"at least six months of public history
prior to submission, with evidence of releases, public issues/pull requests."* caustica went public
**2026-08-22**, so the **earliest eligible submission is roughly late February 2027**, and only if there
are real releases and public issue traffic in between. That constraint alone argues for G5 (PyPI, tagged
releases, public issues) **now** rather than later. JOSS also requires the software be "feature-complete
(no half-baked solutions)" — a solver whose GPU backend has never been executed and whose absorption is
frequency-independent would likely be desk-rejected or bounced. **So: G1 + G3 + G5 are JOSS
prerequisites, not merely nice to have.**

Note the alternative precedent: **j-Wave went to *SoftwareX* (22:101338, 2023)** and Stride to
*Computer Methods and Programs in Biomedicine* (221:106855, 2022). SoftwareX is a reasonable fallback
with a less rigid history requirement, but JOSS's review is more visible in this community.

**(b) A methods paper second — and it should not be "we wrote another PSTD solver."** That paper has
been written, by better-resourced groups, several times. The publishable contribution has to be the
thing nobody else did. In descending order of defensibility:
- **UQ over property maps** (G8) — attacks the documented bottleneck (§2.4), builds on but goes beyond
  Martin & Treeby 2023, and no library ships it. *Best candidate.*
- **A measured CW steady-state fast path** (G4) — a concrete speed/memory result against `k-wave-python`
  on identical CW problems, with the ITRUSST benchmarks as the accuracy control. Modest but honest, and
  directly useful.
- **Differentiable nonlinear acoustics** (G6) — the highest-impact paper available in this field right
  now, and the least likely to be finished within 12 months.

**Timing.** ITRUSST results (G2) should be *public in the repo* well before either paper, because both
papers will be judged on whether the solver has a benchmark position. Do not wait for a paper to publish
benchmark results.

### 4.2 A note on how to make the claim

Given §2.4, **do not claim "most accurate."** It cannot be substantiated and it invites exactly the
comparison that embarrasses everyone. The claims caustica can actually defend, in order:
"reproducible", "plans before it runs", "installs anywhere", "fastest for CW steady state" (once
measured), "tells you how uncertain the answer is". Those are true, checkable, and unoccupied.

---

## 5. Risks

**R1 — The differentiator already evaporated once, and can again.** "Pure Python + CuPy" was caustica's
thesis in gemini1/gemini2 and was matched by `k-wave-python` v0.6.0 on 2026-03-26. That competitor ships
**100+ commits/year**, carries the **k-Wave brand**, and has **~4,800 PyPI downloads/month**. Any
differentiator that is a few weeks of work for them is not a differentiator. *Mitigation: build on the
run-contract/planner/plugin layer (G5), which is months of design work, not weeks of coding, and which
they have shown no interest in.*

**R2 — Solo-maintainer mortality. This is why the graveyard exists.** The pattern is unmistakable and
directly applicable: **j-Wave** (bus factor 1; 3 commits in 12 months; PyPI frozen at 2024 while its own
dependency `jaxdf` moved on), **OptimUS** (1 commit/yr), **mSOUND** (0 commits/yr, "beta 0.2" since
2022, NIH-grant-shaped lifecycle), **HITU Simulator** (FDA staff project, abandoned 2019), **TUSX**
(student project, dormant 2022), **Kranion** (public repo dormant since 2022, real builds moved behind a
gated user group). None of these died of being wrong. They died because one person's attention moved.
caustica is currently a bus factor of one with zero external users. *Mitigation: the plugin architecture
and the frozen contracts are the right structural answer; a second contributor, or an institutional
home, is the real one.*

**R3 — Overclaiming.** Publishing "best/most accurate" against a literature that measures 20–77% error
against experiment (§2.4) is a fast way to lose credibility permanently in a small field. *Mitigation:
claim only what is measured; keep the `[unverified]` discipline of this report in the README.*

**R4 — Physics-subset trap.** Marketing a solver as a k-Wave alternative while lacking power-law
absorption, thermal, broadband and skull support invites a one-line dismissal. *Mitigation: G3 and G7
before any broad announcement; be explicit about scope in the meantime — "CW steady-state therapeutic
fields" is a defensible narrow claim; "general acoustic solver" is not.*

**R5 — Unvalidated GPU claims.** The README leads with "GPU-accelerated" and the planner emits VRAM and
wall-time predictions that **have never been checked against hardware**. To its credit the README
discloses this ("packaged but **not yet verified on real hardware**"), which is the right discipline —
but the disclosure does not remove the risk. If the first real GPU run
disagrees with the planner, every number the project has published becomes suspect at once.
*Mitigation: G1 immediately, and publish the planner-vs-measured comparison honestly, including the
misses.*

**R6 — Window closing through consolidation.** BabelBrain absorbed PlanTUS (2026-08-19). Fullwave went
open source (2025-10-28) and is actively developed. `k-wave-python` is absorbing the pure-Python niche.
The gaps identified here are gaps *as of 2026-08-22* and several are being actively filled.

**R7 — Entering a market where distribution beats merit.** Fullwave 2.5 is arguably the most physically
complete open solver in the field and gets **117 downloads/month**; `k-wave-python` gets 4,767. Merit
alone does not move users. *Mitigation: the ITRUSST position (G2) and being callable from existing
pipelines (§1.3) are distribution moves, not physics moves — treat them as the priority they are.*

**R8 — Dependence on Colab as the development platform.** No local GPU means every GPU iteration depends
on a free-tier cloud runtime with session limits. This is why G1 has been deferred for two weeks already
and is the single largest schedule risk in §3.4.

**R9 — MATLAB gravity.** k-Wave-II is a **MATLAB** rewrite; PRESTUS, mSOUND, FOCUS, HITU and the ITRUSST
comparison harness are all MATLAB. Being Python-native is right for the future and is a friction point
today — notably, running the ITRUSST comparison harness (G2) requires MATLAB or reimplementing
`compareTwo`'s metrics.

**R10 — Regulatory ceiling.** No open tool in this space is cleared for clinical use; even commercial
k-Plan is explicitly *"Research Use Only… non-clinical."* Treatment-planning language must stay
research-scoped, and the medical-disclaimer discipline already planned for M18 should apply from the
first thermal commit.

---

## Appendix A — verification status

**Verified from primary sources** (GitHub API, repo `LICENSE`/README, PyPI JSON, pypistats, publisher
pages), fetched 2026-08-22:
- All star counts, fork counts, `pushed_at` dates, 12-month commit counts, licences and release
  tags/dates in §1.1 except where marked.
- k-wave-python v0.6.0 release note text and date; the presence and content of
  `kwave/solvers/kspace_solver.py` (fractional Laplacian, `alpha_power`, `BonA`, cupy device selection).
- j-Wave: LGPL-3.0, 0 code-search hits for `nonlinear` (control: 13 for `attenuation`), release 0.2.1
  (2024-09-17), 3 commits since 2025-08-22, open issue #247.
- Fullwave 2.5: LGPL-3.0, created 2025-10-28, `beta = 1 + B/A/2` per-voxel in `fullwave/medium.py`,
  README claims (multi-GPU, NVIDIA-only, 8th/4th-order FDTD, spatially varying γ).
- k-wave.org: v1.4.1 dated 17 December 2025; the 2026-06-08 k-Wave-II announcement text; LGPL licence
  wording; C++/CUDA binaries v1.3 (2020-02-28), SM 3.0–7.5.
- `ucl-bug/transcranial-ultrasound-benchmarks`: existence, LGPL-3.0, last push 2022-02-10, Zenodo DOI
  10.5281/zenodo.6020543, the verbatim "Adding new model results" procedure, and the eleven model names
  read from `intercomparison/getModelNames.m`.
- PyPI 30-day downloads for all packages listed in §1.2.
- JOSS's six-month public-history and feature-completeness requirements.
- Li et al. arXiv:2606.09497 error figures; Brandts et al. medRxiv identifiers and stated root cause.
- Aubry et al. 2022 abstract: nine benchmarks, two transducers → 18 permutations, **eleven** modelling
  tools, median focal-pressure difference < 10%, median focal-position difference < 1 mm.
- TUSNet: authors, arXiv:2410.19995 submitted 2024-10-25, the 21 ms / >1200× / 98.3% / 0.18 mm figures,
  and that it is **2-D**; repo `kbp-lab/TUSNet` (MIT, 9 stars, push 2025-02-10).
- TFUScapes: repo `CAMMA-public/TFUScapes` created 2025-05-16 (14 stars), arXiv:2505.12998, Hugging Face
  hosting, and the README's statements that it uses k-Wave and returns **steady-state** pressure fields.
  Its dataset size in GB/samples and its licence (`NOASSERTION` on GitHub) are `[unverified]`.

**Explicitly unverified** (do not repeat as fact):
- Any **ITRUSST Phase 2 / round 2**. No primary evidence found; the public benchmark repo is frozen at
  2022-02-10.
- Fullwave 2.5's treatment of **elastic waves and thermal** (README silent; the *nonlinearity* question
  is resolved — it is present as a medium parameter).
- **FOCUS**: nonlinear support, heterogeneous support, thermal, GPU, and per-version dates (its
  release-notes page carries no dates at all).
- **Stride** `iso_elastic` maturity; **UltraWave** created/pushed dates and star count (API 403).
- **k-Plan** version, pricing, and any CE/FDA status; **Sim4Life.web V9.2** release date.
- Whether any **FDA-cleared TUS planning software** exists as of 2026-08 (the FDA 510(k)/PMA databases
  were not queried directly; k-Plan is explicitly Research Use Only).
- **Citation counts** for j-Wave, Stride, OptimUS — none are quoted anywhere in this report, by design.
- **k-Wave CUDA source availability/licence** (k-wave.org's licence page is silent on the C++/CUDA
  components; `downloadcpp.php` 404s).
- **BabelViscoFDTD's exact SPDX licence** (GitHub reports `NOASSERTION`; PyPI classifier says "BSD").
- k-wave-python's **maintainer count / bus factor** (contributors endpoint 403).

## Appendix B — sources

**Repositories and packages**
[k-wave-python](https://github.com/waltsims/k-wave-python) ·
[k-wave-python releases](https://github.com/waltsims/k-wave-python/releases) ·
[k-wave-python docs](https://k-wave-python.readthedocs.io/) ·
[j-Wave](https://github.com/ucl-bug/jwave) · [jaxdf](https://github.com/ucl-bug/jaxdf) ·
[k-Wave-II](https://github.com/ucl-bug/k-wave-ii) ·
[Stride](https://github.com/trustimaging/stride) ·
[BabelBrain](https://github.com/ProteusMRIgHIFU/BabelBrain) ·
[BabelViscoFDTD](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD) ·
[Fullwave 2.5](https://github.com/pinton-lab/fullwave25) ·
[UltraWave](https://github.com/zixuant5/UltraWave) ·
[OptimUS](https://github.com/optimuslib/optimus) · [mSOUND](https://github.com/m-SOUND/mSOUND) ·
[HITU Simulator](https://github.com/jsoneson/HITU_Simulator) · [Kranion](https://github.com/jws2f/Kranion) ·
[PRESTUS](https://github.com/Donders-Institute/PRESTUS) · [TUSX](https://github.com/ianheimbuch/tusx) ·
[PlanTUS](https://github.com/mlueckel/PlanTUS) ·
[ITRUSST intercomparison library](https://github.com/ucl-bug/transcranial-ultrasound-benchmarks)

**Official pages**
[k-wave.org](http://www.k-wave.org/) · [k-Wave licence](http://www.k-wave.org/license.php) ·
[k-Wave download](http://www.k-wave.org/download.php) ·
[FOCUS](https://www.egr.msu.edu/~fultras-web/download.php) · [mSOUND](https://m-sound.github.io/mSOUND/home) ·
[k-Plan (Brainbox)](https://brainbox-neuro.com/products/k-plan) ·
[Sim4Life focused ultrasound](https://sim4life.swiss/focused-ultrasound) ·
[FUS Foundation open-access tools](https://www.fusfoundation.org/for-researchers-and-clinicians/open-access-technical-tools/) ·
[ITRUSST](https://www.itrusst.com/) · [JOSS submission requirements](https://joss.readthedocs.io/en/latest/submitting.html)

**Papers**
[Aubry et al. 2022, ITRUSST benchmarks, JASA 152(2):1003](https://pubs.aip.org/asa/jasa/article/152/2/1003/2838380/) ·
[arXiv:2202.04552](https://arxiv.org/abs/2202.04552) ·
[Benchmark data, Zenodo doi:10.5281/zenodo.6020543](https://doi.org/10.5281/zenodo.6020543) ·
[ITRUSST standardised reporting (PMID 38670224)](https://pubmed.ncbi.nlm.nih.gov/38670224/) ·
[ITRUSST biophysical safety, arXiv:2311.05359](https://arxiv.org/abs/2311.05359) ·
[Brandts et al. 2026, medRxiv (PMID 41646764)](https://pubmed.ncbi.nlm.nih.gov/41646764/) ·
[Li et al. 2026, arXiv:2606.09497](https://arxiv.org/abs/2606.09497) ·
[Martin & Treeby, UQ, arXiv:2212.04405](https://arxiv.org/abs/2212.04405) ·
[Naftchi-Ardebili et al., TUSNet, arXiv:2410.19995](https://arxiv.org/abs/2410.19995) ·
[TUSNet code](https://github.com/kbp-lab/TUSNet) ·
[Srivastav et al., TFUScapes, arXiv:2505.12998](https://arxiv.org/abs/2505.12998) ·
[TFUScapes repo](https://github.com/CAMMA-public/TFUScapes) ·
[TFUScapes dataset (Hugging Face)](https://huggingface.co/datasets/vinkle-srivastav/TFUScapes) ·
[Drainville et al. 2025, CIVA benchmark, JASA 157(4):3148](https://pubs.aip.org/asa/jasa/article/157/4/3148/3345137/) ·
[Sode & Pinton, Fullwave 2.5 attenuation, arXiv:2606.11103](https://arxiv.org/abs/2606.11103) ·
[Stanziola et al., j-Wave, arXiv:2207.01499](https://arxiv.org/abs/2207.01499) ·
[Cueto et al., Stride, CMPB 221:106855](https://doi.org/10.1016/j.cmpb.2022.106855)
