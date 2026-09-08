# Changelog

Notable changes to **caustica**. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims at
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Nothing has been released yet: `0.1.0` is the first planned release and the
sections below describe what is on `master` today. The rule behind every entry
is that nothing is claimed without a measurement to point at; what has been
measured, and how, is [documented here](https://ebx0.github.io/caustica/validation/).

## [Unreleased]

### Changed

- **Transducer model v2.** `caustica.transducers` gives an element four
  shapes (disc, rectangle, ring sector, spherical segment), per-element
  amplitude, phase and delay, an equal-area quadrature per shape and one
  band-limited deposit, plus the Rayleigh integral over the same surfaces.
  Deposited weights carry the element's closed-form area to 1.2e-16
  relative at any orientation. `caustica.arrays` re-exports it for one
  release. An element apodized to zero is left out of the source entirely
  rather than deposited with zero weight, so `n_points` and the
  represented-element count describe what actually radiates. Measured
  against the pre-v2 path in one process, the deposit is unchanged: a
  128-element spiral gives an identical index set and identical drives, and
  a focused bowl agrees to 1.9e-8 of its peak voxel. V-11's focal peak ratio
  now holds for disc, rectangle and spherical segment (1.0100, 1.0118,
  1.0144 against 1.00 plus or minus 0.02) and is a known failure for a flat
  annular array at 1.0253; the cause is the engine's absolute amplitude for
  a flat source lying in one grid plane, not the element model, since a
  plain disc source containing no element model reads 1.0340 on the same
  grid and the annular array sits below it at both resolutions.
- **Removed the `binary` source discretization** (decision D-022):
  `source.array.discretization` is gone from the job schema and a job that
  still carries it is refused by name; `bowl_cw_source`, `disc_cw_source`
  and `TransducerArray.voxelize` keep a `discretization` argument for one
  release only, to refuse the removed value and say why. `bowl_cw_source`
  also lost its unused `spacing` argument. Result files keep their
  `source_discretization` stamp and stay readable.

- **README and `pyproject` claim only what a report or a test backs.** The
  Quickstart said GPU support was "not yet verified on real hardware" while an
  A100 session had already graded the analytic and ITRUSST suites from cupy. It
  now cites that session's relative L-infinity of 1.5e-6 and names all three
  gates it failed, the last of them a 640^3 rung at dx 0.3 mm refused for
  memory against a 26.98 GiB plan on a 39.494 GiB device. The Fubini range is
  the five stations actually graded, sigma = 0.11 to 0.59, rather than a test
  docstring's wider figure, and "every solver claim is gated by tests" is
  narrowed to the gate, test or named report behind each one. Two tests keep
  the prose honest from here: `test_doc_claims.py` pins the citations, the
  documented commands and the version, and a gpu-marked `test_backend_parity`
  runs the parity job on numpy and on cupy in one process, so the parity number
  stands behind today's engine instead of an August one.

- **The planner times the engine's own step, not a copy of it.** The
  calibration probe kept a hand-written replica of the k-space step, and when
  the engine fused its two damping passes into one the replica went on paying
  for both. Both now build the same closure from one factory. Measured on an
  RTX 5050 against the closure a real solve ran, timed in alternating rounds
  in a single process: the replica read 9.4 % high at 128^3 and 10.5 % high at
  192^3, the shared closure reads within 1.8 % and 0.5 %. The probe also
  stops allocating a damping volume the engine does not have. Solver output is
  bit-identical, linear and Westervelt, so the numerics scheme stays
  `cw-kspace-pstd/4`. A `~/.caustica/calibration.json` written before this
  change reads 7 to 10 % high and should be re-measured. The probe still
  times the grid-sized part of a step only; against a whole solve it reads
  about 15 % low at 128^3 and 10 % low at 192^3, which the planner's next
  revision has to carry.
- **`cupy_available()` compiles a kernel before it answers yes.** Counting a
  CUDA device only proves the driver answers; every kernel caustica launches is
  compiled at run time, so a machine with a broken NVRTC used to be told a GPU
  was ready and crashed minutes into a solve. The probe now compiles, launches
  and reads back a 16-element add, caches the failure text, and classifies the
  fault as a missing cupy, a zero-device count or a device that will not run a
  kernel. `env_report()` carries the reason as `gpu_unavailable_reason`, the
  Colab environment line prints it, and `require_gpu()` withholds the
  `pip install cupy-cuda12x` advice for the one fault no install can fix.
  Measured on an RTX 5050 with a genuine compile error: the old logic answered
  True, the new probe answers False. First call costs 0.30 s with a warm kernel
  cache and 0.57 s with an empty one; later calls in the same process are
  cached.
- **An axisymmetric scene is refused by every Cartesian solver.** A
  `Medium` now carries the geometry it was sampled on (`cartesian` or
  `axisymmetric`), `SolverCaps.geometry` declares what a solver integrates,
  and a job whose scene sets `axisymmetric: true` is refused at build time
  with a message naming the planned `kspace-as` solver, instead of being
  solved on the (r, z) half-plane as a two-dimensional line source.
- **One damping volume in the k-space engine.** Absorption and the sponge
  were two multiplicative passes per field per step; they are now one
  float32 product formed at setup. Measured on an RTX 5050 at 192^3: the
  linear step is 8.6 % faster and one padded float32 volume of device
  memory is freed; the O'Neil focal peak is unchanged in lossless water and
  moves by 1.1e-7 relative in absorbing water, so the numerics scheme stays
  `cw-kspace-pstd/4`. The planner inventory names the `damping volume` line
  that replaces `sponge`.
- **The two k-Wave adapter tests above 20 s carry the `slow` marker**, so
  `pytest -m "not slow"` runs the rest of the suite in about 70 s on a
  laptop; the full CPU pass is about 190 s.
- **The validation gates lost their milestone prefixes.** `M4.planewave` is now
  `planewave`, `M8.vram` is `vram`, `M21.PH1-SC1` is `PH1-SC1`, and so on for
  every gate the analytic, compare, gpu-gates and ITRUSST suites write. The
  second half of the name always said what the gate measured; the first half
  indexed a ledger that no longer ships with the repository. Reports written
  before this change carry the old ids, so a diff across that boundary has to
  map them by hand.

### Added

- **The k-Wave parity harness.** `caustica.parity` runs a mirrored k-Wave
  example through every native solver that can express it plus the k-Wave
  reference, at matched CFL, and writes one `metrics.json` per example
  against a fixed thirty-metric registry. Each example declares the subset
  that applies to it: four mirrors from three of k-Wave's own categories
  emit 11, 15, 10 and 14 values out of 30, and no two subsets are the same.
  Every cell is a value, a stated "not applicable" or a stated "not
  computed", each absence carrying its own reason; the data classes refuse
  a blank, a dash, a placeholder string, a boolean and a non-finite number,
  so a metric cannot silently become a zero. Measured on the four pages
  against k-Wave and against closed forms: whole-field correlation
  0.99999918 (heat diffusion against the exact Pennes solution), 0.99999934
  and 0.99892 (against k-Wave), and 0.99154 (against the compiled
  `kspaceFirstOrder` binary in 3-D), with focal peak ratios of +0.195 %,
  +1.42 %, +0.045 % and +2.22 %. The realized CFL is recorded per engine
  rather than assumed, because the two codes round the period to a whole
  number of steps in opposite directions. Timing and memory ratios report
  "not computed" until a run carries a load regime, since a measurement
  taken while other work shares the machine is not one.

- **One command for a machine's GPU evidence.** `scripts/run_gpu_local.ps1`
  and its `.sh` twin run the `gpu`-marked tests and
  `python -m caustica.validation gpu-gates` together and leave both logs, a
  `SUMMARY.md` and the suite's own `REPORT.md` in one stamped folder under
  `benchmarks/reports/gpu_gates/<device>-<date>/`. On an RTX 5050 Laptop the
  run takes 54 s and closes the parity, VRAM and refusal gates every time.
  The `fullsize` gate stays incomplete on that card by design, because its
  criterion needs a 512-cubed run and the device has 8 GiB, and the `time`
  gate passed 11 of 12 idle runs. A dormant self-hosted CI job is included
  for anyone who wants the same evidence on their own runner.

- **Power-law absorption in the material schema.** `Material` carries
  `alpha0_db_cm_mhz_y` and `y` beside the legacy single-frequency
  `alpha_np_m`, plus thermal conductivity, specific heat, perfusion and a
  source string. `MaterialDB.y()` returns the table's common exponent or
  refuses a mixed table naming every offender; jobs gained an `absorption`
  section (`single_frequency` or `power_law`, `y: null` reads the exponent
  off the table) resolved at build time, so `caustica validate` refuses a
  mixed table before anything is built. The law is evaluated at the drive
  frequency where that frequency is known, in the job builder and in
  `MediumVolume.to_medium(f0_hz=...)`. `TISSUE_LIBRARY` carries IT'IS V4.2
  thermal values for water, skin, fat, muscle, fibroglandular, liver, brain
  and blood, the last three new rows. The solvers still apply one `alpha`
  at every frequency, and say so in a warning on every power-law job, until
  the engine's absorption model lands. No trajectory moves.

- **Solvers.** A shared k-space PSTD CW engine behind two registered solvers,
  `linear` and `westervelt`, differing only in whether the nonlinear term
  enters the pressure update. Exact-period time step under a CFL limit,
  leakage-free single-bin harmonic recording, and the library-wide phasor
  convention `p(t) = Re{P e^{-iωt}}`.
- **Backends.** One dispatch layer (`caustica.core.backend`) over numpy and
  cupy; the same operators run on both. cupy is never installed implicitly.
- **k-Wave adapter.** `kwave` solver wrapping `k-wave-python` for cross-checks.
- **Planner.** VRAM inventory that matches the engine allocation for allocation,
  a per-step time model calibrated on the live device, out-of-memory refusal
  with actionable advice, and a shipped GPU datasheet (`gpu_db.json`). Unknown
  cards are reported as `unknown:<name>` and calibrated rather than guessed.
- **Geometry.** Scene/shape/volume construction, bowl and array transducer
  geometry, element tables, and phase maps for steering and focusing.
- **Analytic references.** O'Neil, Rayleigh integral, and plane-wave solutions
  used as validation targets rather than as fixtures.
- **Thermal.** Pennes bioheat solver with a conservative finite-difference
  scheme, `Q = 2αI` heating from an acoustic result, CEM43 thermal dose
  accumulated during the solve, and a `caustica-thermal/1` dose report.
- **Validation suites.** `caustica.validation` runs the analytic suite, a
  multi-engine comparison, and the GPU gate ladder, each emitting a stamped
  JSON report.
- **Runs.** Job schema and CLI (`caustica run`), atomic HDF5 result store,
  resumable checkpoints, progress contract, run reports and figures, and
  `caustica.Study` for sweeps that reuse the single run path.
- **Reproducibility.** Every wheel and sdist carries the commit it was built
  from, and every report carries the environment that produced it.
- **The first two ITRUSST benchmarks run as gates.** `caustica.validation
  itrusst` runs PH1 BM1 and BM2 (water, lossless and 1 dB/cm at 500 kHz)
  against both source conditions (focused bowl, plane piston) and grades the
  field over the paper's own 241 x 141 comparison domain against the Rayleigh
  integral — exact for a baffled source in a homogeneous medium, and the same
  kind of reference the intercomparison used. Measured: L-infinity 3.41, 3.82,
  3.79 and 5.03 %, peaks 1.031–1.038 times the reference on its own voxel. The
  limits are the spread the paper itself reported across eleven models (under
  10 % for the bowl, under 15 % for the piston), not a tolerance invented here.
- **The Rayleigh integral takes a complex wavenumber**, so absorption is
  integrated along every source-to-field path instead of being applied to a
  lossless answer afterwards. The difference has no fixed sign: a flat
  piston's paths are never shorter than the axial distance, a concave cap's
  can be.
- **`disc_cw_source`**: the flat circular piston, the one shape the library
  had no constructor for.
- **`result.h5` carries its numerics generation** — `numerics_scheme` and
  `source_discretization` at the root, so a pressure in pascals says which
  side of the 2026-08-24 source changes it came from.
- **An absolute-amplitude gate** (`absolute` in the analytic suite). Every
  other check in that suite grades a normalized quantity, which is how two
  independent 13–18 % absolute errors survived months of green gates. Its
  tolerance came from measurement: the error follows `3.7 * ppw^-2.5`, so a
  single number at one spacing cannot separate a coarse grid from a wrong
  source. The gate is three layers — the source's own measure against the
  cap's area, the level at a fine rung, and how much a 2x refinement moved the
  error. Restoring the old binary shell fails all three.

### Fixed

- **The 256³ GPU divergence.** The collocated spectral first derivative kept a
  live Nyquist wavenumber on even-length axes, so `deriv * rfftn(p)` was not a
  legal half-spectrum: it violated the C2R Hermitian contract by ~150% of its
  own magnitude. cuFFT, which documents its input as Hermitian, resolved that
  differently at 256³ and reached NaN by the second period while the identical
  CPU run stayed at 45 kPa. The Nyquist bin is now dropped — which is also
  simply the right derivative on a collocated grid, and is bit-for-bit the
  canonical Hermitian projection on every axis and parity — and the gates are
  written on the transform's *input*, so they fail on any machine rather than
  only on a GPU.

  Two corrections to what was first said about this, both measured
  (2026-08-24, `benchmarks/reports/campaign/` and `benchmarks/reports/resolution/`):

  - It was **not** a GPU-only defect. `irfftn` inverts the last axis with a
    C2R transform and the rest with C2C, and only the C2R stage discards
    illegal content — so on the *transverse* axes the violation entered the
    answer on the CPU too, moving the shipped bowl example's field by ~1.4% of
    peak. The GPU's visible failure and the CPU's silent error were the same
    defect on different axes.
  - The repair is the Nyquist zeroing, not the explicit `axes=` that landed in
    the same commit. Restoring only the old derivative factory inside today's
    engine still diverges at 128³ and 256³ on cupy while the shipped operator
    diverges at none of the eight sizes tested.
- **A diverged run no longer exits 0.** A NaN field used to run to the settle
  cap and write a result file full of NaN; it now raises `SolverDivergedError`
  at the first non-finite period.
- **The skip-guard checks job identity.** A folder holding a complete
  `result.h5` short-circuits to exit 0, which is what makes an interrupted
  sweep resumable. It did so without asking whether the result answered *this*
  job, so editing a job and rerunning the same command line handed back the old
  field with no warning. The folder's own `job.json` is now compared against the
  job about to run, and a mismatch is refused with the differing sections named.
  Only `output.folder` is exempt, because `--out` routinely sends a job
  somewhere other than the folder its config names.
- **VRAM measurement** is taken in the rung's own fresh process, so an earlier
  rung's pool cannot be counted as this one's peak.
- **Calibration** sizes its probes as a fraction of free VRAM and interpolates
  its measurements instead of forcing one power law per card.
- **The k-Wave adapter records the region it was asked for.** Its binary
  cannot accumulate a DFT, so it dumps every recorded step to HDF5; the sensor
  mask was the whole grid regardless of `record_region`. On the ITRUSST
  geometry that was 731 MB in and 1.6 GB out per run.

### Changed

- **A focused bowl is now a band-limited source carrying the cap's area, not a
  voxel shell.** This changes every absolute pressure the library produces from
  a bowl, by roughly −13%.

  The engine drives every source voxel with the same normalized amplitude,
  which is exact for a flat source — one voxel per `dx²` of aperture — and is
  not for a curved one. A digitized cap crosses 1.18 voxels per `dx²` of its own
  area, so a binary bowl radiated in proportion to its voxel count. Measured on
  an f/1.2 bowl: 1.15–1.17× O'Neil's closed form, *flat* from 3.8 to 15 points
  per wavelength, because a staircase factor is not a discretization error and
  refinement does not remove it. The same sampling also left 10–12% of the
  shell undriven, and closing those holes alone made it worse, not better.

  `bowl_cw_source` now takes the cap's closed-form area, divides it over
  equal-area quadrature points, and deposits each through a band-limited
  interpolant (`caustica.geometry.offgrid`), so the grid weights sum to the
  area in grid squares whatever the orientation. This is the method k-Wave uses
  for the same reason (Wise, Cox, Jaros and Treeby, JASA 146, 2019); our
  deposit reproduces `kWaveArray`'s weights to the last printed digit.
  Measured after: 1.083 → 1.032 → **1.004** over the same three rungs, which is
  what an ordinary discretization error looks like. The truncated kernel is
  renormalized per point, which removes its ringing and lets a two-voxel window
  do the work of a seven-voxel one.

  `CWSource` gained a `weights` field (`None` still means a uniform drive, and
  a plane source is unchanged bit for bit). Jobs take
  `source.array.discretization`, `"offgrid"` by default and `"binary"` for
  reproducing an older result. The checkpoint fingerprint scheme is
  `cw-kspace-pstd/3`; a checkpoint from before this refuses to resume into it.
- **The PML clearance check grades the drive, not voxel presence.** A
  band-limited source has a halo whose outermost voxels carry thousandths of
  the drive, so "is any source voxel in the sponge" stopped meaning anything.
  A setup is now refused when more than 90% of `|drive|` lands in the band —
  measured separations: a bowl deliberately buried damps 94.5%, a legitimate
  full-width plane source damps 44%, a bowl with proper standoff damps none.

- **An element array's elements now sit where they are, not on the lattice.**
  `TransducerArray.voxelize` rounded each element centre to a voxel. That is up
  to half a voxel of path length to the focus, a *different* error for every
  element, so it does not average out — it defocuses. Measured on the
  production 128-element spiral at dx = 0.5 mm and 1 MHz: 0.61 rad rms, costing
  17.6% of the coherent focal sum, which `exp(-σ²/2)` predicts to three
  decimals.

  Two smaller errors rode along and pulled the other way: the notebook's
  in-plane disc test sheared onto the element plane inflates a tilted element
  by `1/cos(tilt)` (8% on average at production tilts, 14% at the rim), while
  voxel quantization of a small disc takes some back — the net landed at +4.4%
  on the production array and −2.5% on a smaller one, wrong by a few percent
  with a sign that depends on the grid. And where two elements claimed a voxel,
  the first one's phase won and the other's drive was dropped (1.7% of the
  pairs on the production array, up to 5.8% of one element).

  Elements are now deposited at their own positions through the same
  band-limited interpolant, each carrying `π r²` exactly, and overlapping
  contributions are summed as complex phasors — `Σ wᵢ sin(ωt − φᵢ)` is exactly
  `|S| sin(ωt − Φ)` for `S = Σ wᵢ e^(−iφᵢ)`, so `CWSource` can hold the result.
  Graded against the Rayleigh integral over the true element discs on a
  refinement ladder: the binary path goes 0.861 → 0.931 → **0.951** and is still
  5% short at fifteen points per wavelength; the off-grid path goes 1.142 →
  1.018 → **1.001**. `voxelize(discretization="binary")` restores the old one.

  An element smaller than half a voxel is now refused as a resolution error
  rather than as "lost all voxels to deduplication" — there is no deduplication
  any more, overlapping elements superpose, which is what two real elements
  would do.

### Known issues

- **`kappa` is cross-axis inconsistent** since the Nyquist fix. `sinc(c dt |k|/2)`
  still evaluates `|k|` with the true Nyquist component included, so on a
  Nyquist hyperplane the surviving transverse derivative is scaled by a
  dispersion factor evaluated at a wavevector the operator has just declared
  unrepresentable: up to ~10% amplitude error there at the shipped CFL, over
  planes carrying ~0.1% of the field's energy. Both conventions are defensible
  and neither has been measured against the analytic references yet.

[Unreleased]: https://github.com/ebx0/caustica/commits/master
