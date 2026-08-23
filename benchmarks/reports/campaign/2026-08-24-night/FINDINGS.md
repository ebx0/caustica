# What the assumptions were worth — campaign of 2026-08-24

A validation campaign run overnight on a laptop with an RTX 5050 (8 GiB,
Blackwell CC 12.0) and a local k-wave-python install. The question was not "do
the tests pass" — they do — but **which of the things this project treats as
true have never actually been measured**, with the newest and least-examined
change first: the Nyquist fix of 2026-08-24 (`f462ce4`), which altered the
spectral derivative operator and therefore every number the library produces.

Harness: `scripts/dev_campaign.py` (committed, reproducible). Raw numbers:
`campaign.json` beside this file. Each experiment runs the current operator and,
where the comparison is the point, the pre-fix operator restored in-process by
`legacy_operator()`, so both are measured under identical conditions in one
run — same machine, same float32 arithmetic, same scenario object.

---

## 1. The headline: the fix is safe, and it reaches further than anyone thought

### 1.1 It does not move the physics

Comparing the committed pre-fix analytic evidence (`analytic/numpy-20260823-073910`,
commit `cbec29f`) against a fresh post-fix run (`analytic/postfix-cupy-20260824`,
commit `b41850d`):

| gate | movement |
|---|---|
| plane wave: realized amplitude / p0 | **0.000 %** |
| plane wave: phase speed (k) | **0.000 %** |
| plane wave: absorption exponent | **0.000 %** |
| O'Neil: axial profile correlation | **0.000 %** |
| O'Neil: focal peak position | **0.000 %** |
| O'Neil: −6 dB axial width | **0.000 %** |
| Fubini A2/A1 at five σ | 0.001 – 0.021 % |

Five or six significant figures unchanged. Independently, a 1-D plane wave
(E3) gives phase-speed error 0.0508 % and absorption error 0.058 % for **both**
operators — identical to four significant figures. A resolved wave cannot tell
them apart, which is exactly what a correct Nyquist treatment should look like:
it edits only the part of the spectrum the grid was never representing.

### 1.2 But it reaches nearly every run, not just 256³

The engine pads each axis up to the next 2/3/5-smooth size. Of the 109 such
sizes below 2048 only 22 are odd, and **88.5 % of user grid sizes between 32
and 1024 pad to an even axis** — so in three dimensions roughly 99.85 % of runs
had at least one even padded axis and were therefore affected. The pre-fix
operator was not a 256³ curiosity; it touched almost everything this library
has ever computed. 256³ is simply where cuFFT's freedom turned into NaN.

Measured field movement (E2, focused bowl, phasor, max relative):

| padded shape | parity | field moved |
|---|---|---|
| (64, 64, 64) | e e e | 2.726 % |
| (96, 96, 96) | e e e | 1.281 % |
| (100, 100, 100) | e e e | 1.244 % |
| (64, 80, 96) | e e e | 2.547 % |
| (64, 64, 81) | e e **o** | 2.756 % |
| (81, 81, 64) | **o o** e | **0.000 %** |
| (81, 81, 81) | o o o | **0.000 %** |
| (135, 135, 135) | o o o | **0.000 %** |

### 1.3 The sharpest result of the night: two different axes

Read that table by axis rather than by shape and something falls out that no
one had predicted:

* **Odd transverse axes ⇒ bit-identical**, even when the beam axis is even
  (`81, 81, 64` moved by exactly zero).
* **Even transverse axes ⇒ the field moves**, even when the beam axis is odd
  (`64, 64, 81` moved the most, 2.756 %).

E11 confirms it independently: the ppw-5 rung landed on `(75, 75, 108)` —
odd transverse, even beam — and its gap was 4.6 × 10⁻⁷, i.e. nothing.

So the **field change is carried by the transverse (complex-to-complex) axes**,
because that is where a voxelized bowl puts its near-Nyquist energy. Meanwhile
the U1b GPU evidence of 2026-08-23 showed the **divergence** was carried by the
**beam axis** — the real-to-complex axis — because that is the axis whose
self-paired Nyquist plane violates the C2R Hermitian contract in the way cuFFT
resolves differently from pocketfft (400×400×256 diverged; 256×256×400 did
not).

Two different axes, two different phenomena, one root cause. This is why the
256³ failure looked beam-axis-specific while the correction is transverse-heavy,
and it is the piece of the story that was missing until tonight.

### 1.4 The gap shrinks with refinement

E11, focused bowl (f-number 1) against O'Neil, over the rungs whose transverse
axes are even:

| ppw | grid | fixed vs pre-fix (rel L2) |
|---|---|---|
| 7.5 | (96, 96, 135) | 1.100 % |
| 10 | (120, 120, 180) | 1.053 % |
| 15 | (160, 160, 240) | 0.755 % |

Monotone downward: the two operators are two discretizations of one continuum
operator and they converge together, so the difference is a discretization
artefact rather than a change of physics.

### 1.5 A prediction of mine that the measurement refuted

I predicted the change would **grow** with harmonic order, on the grounds that a
3f₀ field sits three times closer to the grid Nyquist. E12 says the opposite:

| harmonic | ppw at that harmonic | peak | field change (rel L2) |
|---|---|---|---|
| 1 | 7.50 | 7008.1 kPa (−0.019 %) | 1.103 % |
| 2 | 3.75 | 914.0 kPa (+0.002 %) | 0.055 % |
| 3 | 2.50 | 197.7 kPa (−0.001 %) | 0.022 % |

The change **falls** with harmonic order. The explanation has to be corrected:
the change tracks *where the near-Nyquist energy sits in space* — the voxelized
source surface, which dominates the fundamental — not the harmonic order of the
field. The practical consequence is reassuring and worth stating on its own:
**the harmonic amplitudes a HIFU user actually reports move by 0.02–0.06 %, and
the peak pressures by under 0.02 %.**

### 1.6 What the analytic axial profile cannot do

E1 was designed to let O'Neil referee the two operators. It cannot: correlation
0.963409 (fixed) vs 0.963422 (pre-fix), and −6 dB width error −8.11 % for both,
while the fields differ by 2.0 %. The metric is insensitive to the difference by
three orders of magnitude.

Note also that r ≈ 0.965 here is **not** a solver quality figure — the library's
own analytic gate reaches 0.99923 on its scenario. This campaign deliberately
used a much more strongly focused f/1 bowl driven as a pressure source on the
cap, where O'Neil's uniform-normal-velocity assumption is a poorer fit. The
number is only ever used here as a *paired* comparison between two operators on
the identical scene.

---

## 2. Defects found

### 2.1 Fixed tonight: the checkpoint scheme tag was not bumped

`engine.py` builds a checkpoint fingerprint whose first field is a scheme tag,
carrying the comment *"bump when the numerics change"*. The Nyquist fix changed
the numerics and the tag was left at `cw-kspace-pstd/1`, so a checkpoint written
before the fix would have been silently resumed into a post-fix run — splicing
two different trajectories into one field that is neither. Bumped to
`cw-kspace-pstd/2`; `load_checkpoint` now refuses the mismatch, and the
checkpoint tests still pass.

### 2.2 Open, and reported rather than fixed: the runner's skip-guard has no job identity

`runner.py:823` skips a run when the output folder already holds a structurally
valid `result.h5`:

```python
if result_path.exists() and validate_result_file(result_path):
    ...
    print(f"already complete: {result_path} (skip-guard; delete it to regenerate)")
    return EXIT_OK
```

The guard proves only that the file is not a torn write. It does not check that
the stored result belongs to *this* job, and — the reason it matters tonight —
it does not check which numerics produced it. **Every `runs/` folder created
before 2026-08-24 will now be handed back unchanged, pre-fix, exit 0, with no
warning.** `result.h5` records `caustica_version`, which is `0.1.0.dev0` on both
sides of the fix, so nothing in the file distinguishes them.

Proposed fix (not applied — it needs a stored field and this repository has a
second session working in it tonight): record the engine scheme tag and a digest
of the built job in `result.h5`, and make the skip-guard require both to match,
refusing to skip and saying why when they do not.

### 2.3 Open: every committed evidence report predates the fix

The newest evidence in `benchmarks/reports/` is stamped 2026-08-23 13:18; the
fix landed 2026-08-24 00:45. Everything in the repository describing what the
library measures — the analytic suite, the k-Wave cross-check, the GPU parity
figures, the O'Neil numbers — was produced by the pre-fix operator. §1.1 shows
the numbers barely moved, but *"barely moved"* is now a measured claim rather
than an assumption, and the reports themselves should be regenerated. One has
been: `analytic/postfix-cupy-20260824`.

---



## 3. A claim of mine that the campaign refuted

I wrote, in `CHANGELOG.md`, `CONTRIBUTING.md`, `MILESTONES.md`, the operator
docstring and the test module docstring:

> numpy's pocketfft silently projects that away and returns something sane

That is true only on the **last** axis. `irfftn` is a complex-to-complex inverse
over axes 0…d−2 followed by a complex-to-real transform on the last axis, and
only the second stage discards a non-Hermitian component. Measured on this
machine, the pre-fix operator differs from the canonical Hermitian projection
`Re(ifftn(i·k·fftn(f)))` by:

| axis | 8³ | 16³ | 60×60×80 |
|---|---|---|---|
| 0 (transverse) | 0.462 | 0.371 | 0.180 |
| 1 (transverse) | 0.438 | 0.315 | 0.145 |
| 2 (beam, last) | **0.000** | **0.000** | **0.000** |

So the pre-fix CPU numbers were correct on the beam axis and **wrong on the
transverse axes**. This is not a GPU-only repair; it is a correctness repair
whose most visible symptom happened to be a GPU divergence. The documentation
has been corrected in `operators.py` and `tests/test_kspace_operators.py`;
`CHANGELOG.md` and `MILESTONES.md` still carry the old phrasing where they
describe the fix historically.

The same review established the strongest positive statement available, and it
is algebraic rather than empirical: the **fixed** operator is bit-for-bit the
canonical Hermitian projection on every axis and every parity (residual
2.2 × 10⁻¹⁶), while the pre-fix one is 0.56 and 0.50 away from it on the two
transverse axes. Zeroing the Nyquist bin is not one convention among several —
within a collocated scheme it is forced, because a real-preserving,
translation-equivariant, odd symbol has to vanish at the self-paired bin.

## 4. Two attempts to find an independent referee, both inconclusive

The question "which operator is *right*" was put to two outside authorities.
Neither could answer, and both failed for the same reason: **the referee's own
uncertainty is larger than the question.**

**O'Neil's axial solution (E1).** Correlation 0.963409 (fixed) vs 0.963422
(pre-fix); −6 dB axial width error −8.11 % for both; fields 2.0 % apart. The
metric is three orders of magnitude less sensitive than the difference it was
asked to resolve. (The r ≈ 0.965 here is a property of the scenario, not of the
solver — this campaign used a much more strongly focused f/1 bowl than the
library's own analytic gate, which reaches 0.99923 on its own geometry.)

**k-Wave (E4).** A staggered-grid implementation, which never had this defect
because its half-sample shift rotates `i·k_Nyq` onto the real axis.

| scenario | operator | rel L2 vs k-Wave | Pearson r | peak ratio | focal shift |
|---|---|---|---|---|---|
| water | fixed | 0.202003 | 0.976842 | 1.02108 | 0.000 mm |
| water | pre-fix | 0.201940 | 0.976876 | 1.02127 | 0.000 mm |
| layered | fixed | 0.204730 | 0.974266 | 1.02636 | 0.000 mm |
| layered | pre-fix | 0.204676 | 0.974284 | 1.02666 | 0.000 mm |

The fixed operator is 0.03 % *further* from k-Wave — while caustica and k-Wave
disagree by 20 % on this scenario to begin with. The referee's noise is 650×
the signal.

Two caveats, stated so the numbers are not misread:

* This comparison is over the **full volume**, including the voxelized source
  shell where the two engines' source models differ most. The repository's
  recorded cross-check (rel L2 0.0286, r 0.9992) is over a record region on a
  gentler scenario. **The two are not comparable** and nothing here says the
  cross-engine agreement got worse.
* The layered scenario's focal peak came out only 0.02 % below the water one,
  where the impedance step and the 6 Np/m slab should cost 2–4 %. That is
  unexplained and the layered scenario should not be used as evidence for
  anything until it is chased.

## 5. Where the fix should have bitten hardest, and did not

E12 tested a prediction of mine: that the change would **grow** with harmonic
order, since a 3f₀ field sits three times closer to the grid Nyquist. It falls
instead — 1.103 % → 0.055 % → 0.022 % for h1, h2, h3. The change tracks *where
the near-Nyquist energy sits in space* (the voxelized source surface, which
dominates the fundamental), not the harmonic order of the field.

The practical statement is worth separating from the mechanism: **the harmonic
amplitudes a HIFU user reports moved by 0.02–0.06 %, and the peak pressures by
under 0.02 %.**

## 6. Knobs the library calls safe defaults

E10 varies CFL, PML thickness and resolution while holding the interior domain
fixed and sizing every grid from the transducer outwards. An earlier pass of
this experiment reported a 26 % "PML sensitivity" that was really a clipped
source — a 16 mm bowl inside a 13 mm interior — which is why `grid_for_bowl()`
now exists and why every scenario records whether its aperture was clipped.

## 7. Method

`scripts/dev_campaign.py --list` enumerates the experiments; `--all` runs them.
Each writes into `campaign.json` as it finishes, so an interrupted run keeps
what it measured. The pre-fix operator is restored in-process by
`legacy_operator()`, a context manager that swaps
`ops.spectral_derivative_factors` for the pre-2026-08-24 body — which is what
lets one process measure both operators on the same scenario object, in the
same float32 arithmetic, on the same machine.

Two experiments were re-designed mid-campaign after their first pass measured
the harness rather than the library:

* **E5** originally ran a numpy leg at every size as a reference. A single
  288³ step costs seconds on this laptop's CPU and milliseconds on its GPU, so
  the sweep would have cost hours to re-derive what the divergence already
  says. It now runs cupy-only, with numpy anchors at two small sizes.
* **E1/E10** originally used grids that did not fit their own transducer.

---

## 8. The GPU half

Measured on the local RTX 5050 (8 GiB, Blackwell CC 12.0) with cupy 14.2.

### 8.1 M8.time: the warm-process hypothesis is confirmed (E7)

The standing explanation for the open M8.time gate is that calibration probes
run inside a process that has already paid the CUDA context, module-load and
plan costs, so the warmup they measure is not the warmup a real run pays.
Measured directly, same shapes, same card:

| shape | warm process | fresh process | ratio |
|---|---|---|---|
| 128³ | 0.035 s | 0.411 s | **11.6×** |
| 192³ | 0.004 s | 0.415 s | **96.8×** |
| 256³ | 0.007 s | 0.411 s | **58.9×** |

The absolute numbers are small on a laptop card at these shapes — but the
~0.4 s floor is essentially shape-independent, and that is the informative
part: it is a *process* constant, not a *size* term. A warm process measures
almost none of it, and the ratio grows with shape only because the
warm-process figure shrinks. That is the shape of a systematic
under-prediction, and it is invisible to the interpolated warmup model because
every sample that model takes is warm.

The repair follows from the measurement rather than from a guess: either the
calibration takes one probe in a fresh subprocess, or the planner carries a
per-device process constant separate from its per-element term.

### 8.2 The planner's VRAM inventory on a card it has never seen (E8)

`spec_for_device` reports `unknown:NVIDIA GeForce RTX 5050 Laptop GPU`, so
nothing about this card is in `gpu_db.json`.

| shape | planned | measured pool peak | deviation |
|---|---|---|---|
| 96³ | 0.0916 GiB | 0.0848 GiB | +8.01 % |
| 128³ | 0.2167 GiB | 0.2006 GiB | +8.07 % |
| 160³ | 0.4229 GiB | 0.3912 GiB | +8.10 % |
| 192³ | 0.7303 GiB | 0.6754 GiB | +8.12 % |

Inside the ±10 % M8.vram band, and the deviation is a near-constant +8.1 %
rather than drift — consistent with the documented +15 % allocator margin being
partly unused at these sizes, not with an inventory error.

### 8.3 Two backend claims that had only ever been checked on one backend (E13)

| claim | result |
|---|---|
| a resumed run is bit-identical — **on cupy** | max rel `0.00e+00` |
| the thermal solver never touches numpy for state maths — **numpy vs cupy** | max rel `0.00e+00` |

Both hold exactly. float32 reductions are not associative, so neither was free.

### 8.4 Not completed in the window

* **E5** — which grid sizes the pre-fix operator actually broke on cuFFT.
  Redesigned twice and still over budget: a fresh scene per size per operator
  allocates hundreds of megabytes of host arrays and the sweep ran out of
  clock. It is the experiment that would settle whether 256 is unique or a
  class, and it is also the one that would **attribute** the GPU repair —
  commit `f462ce4` bundled the Nyquist zeroing with adding an explicit `axes=`
  to both `irfftn` calls, and nothing yet discriminates them.
* **E6** (the full numpy/cupy parity matrix) and **E9** (a wheel-install
  reproduction of the Colab collection errors) — queued behind E5.

---

## 9. What the adversarial review found

Four reviewers were each told to refute the fix. Both completed verdicts came
back **"correct but incomplete"**, and three of their objections are measured
rather than rhetorical.

### 9.1 Two of my cleanest numbers were tautologies

The 1-D plane-wave agreement and the all-odd-grid bit-identity were offered as
evidence that the fix costs no accuracy. **In 1-D the only axis is the rfft
axis**, which is exactly where the fix is a bit-exact no-op on the CPU. A
reviewer ran it: `max |ΔP| = 0.000000e+00`, exact equality, not "identical to
four figures". Both checks were structurally incapable of failing, and should
be read as consistency checks on the harness rather than accuracy evidence.

What survives as accuracy evidence: the analytic-suite diff of §1.1 (which does
exercise transverse axes), the refinement trend of §1.4, and the algebraic
identity below.

### 9.2 The strongest defence is algebraic, and stronger than what was claimed

The **fixed** operator is bit-for-bit the canonical Hermitian projection
`Re(ifftn(i·k·fftn(f)))` on every axis and every parity — residual
2.2 × 10⁻¹⁶ — while the pre-fix operator sits 0.56 and 0.50 away from it on the
two transverse axes. Within a collocated scheme the zeroing is not one
convention among several; it is forced, because a real-preserving,
translation-equivariant, odd symbol has to vanish at the self-paired bin.

And the honest indictment of the old operator is sharper than "the backends
disagree": its *realized* multiplier on the axis-0 Nyquist plane was `−i·π/dx`
on the interior columns and exactly `0.0` on the two C2R-constrained columns.
**A symbol whose value inside a single plane was decided by which columns
pocketfft happens to discard.** That has no continuum meaning at all.

Two defences that do **not** work and should stop being offered: both operators
are skew-adjoint to roundoff (`|⟨Da,b⟩+⟨a,Db⟩| ≈ 1e-17`) and both are
shift-equivariant to roundoff (3.8 × 10⁻¹⁶). Conservation and homogeneity
arguments do not distinguish them.

### 9.3 A real gap the fix leaves open: kappa is now cross-axis inconsistent

`kappa = sinc(c_max·dt·|k|/2)` still uses the true Nyquist inside `|k|`. On the
*self* axis that is harmless — `deriv` is zero there, so kappa multiplies
nothing. The live case is **cross-axis**: at a bin with `k_x = π/dx`,
`deriv_y = i·k_y·sinc(c·dt·√(k_x²+k_y²+k_z²)/2)` is alive, and kappa is being
evaluated at a wavevector magnitude the operator has just declared
unrepresentable. Measured at the shipped `cfl = 0.48`: up to ~10 % amplitude
error on the transverse derivative over the Nyquist hyperplanes (kappa 0.8067
vs 0.9003 at CFL 0.5 for the 2-D corner; 3.7 % at CFL 0.3).

Those planes carry ~0.1 % of the field's energy, so the effect is small — but
it is a self-inconsistency introduced *by* the fix, and the operator docstring
currently implies it was reasoned away when it was not. Making it consistent
costs three kappa arrays instead of one, each with that axis's k zeroed inside
the magnitude: a VRAM decision, not an unknown.

### 9.4 The upstream cause nobody had named: the source is a raw delta

Why do the Nyquist planes carry any energy at all? Because the engine injects
the source as unsmoothed voxel deltas, and a delta's spectrum is flat.
Measured: a single-voxel source puts 1.0417 % of its energy on each Nyquist
line of a 96-grid — the flat-spectrum value to four figures — and the converged
field still carries 0.12 % per Nyquist line.

**That is the only reason the operator choice was worth 2.9 %.** k-Wave smooths
its source masks (Blackman) for exactly this reason. The Nyquist fix is correct
and necessary, but it treats a symptom of an unsmoothed source. Source
smoothing is the upstream repair, and this library does not have it.

---

## 10. What I would do next, in order

1. **Attribute the GPU repair.** Run the pre-fix operator at 256³ on cupy under
   the *current* engine, which already carries the explicit `axes=`. If it
   diverges, the Nyquist zeroing is proven to be the cause; if it does not, the
   `axes=` change was, and the story needs rewriting. One run.
2. **Source mask smoothing** (§9.4) — the upstream fix, and the one that would
   shrink the disputed 2.9 % rather than redistribute it.
3. **Job identity in the skip-guard** (§2.2) — a live hazard for anyone holding
   a pre-2026-08-24 `runs/` folder.
4. **Cross-axis kappa** (§9.3) — decide and document, even if the decision is
   "leave it, it is 0.1 % of the energy".
5. **Regenerate the k-Wave and thermal evidence** (§2.3).
6. **Finish E5, E6, E9.**
