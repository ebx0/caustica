# Changelog

Notable changes to **caustica**. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims at
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Nothing has been released yet: `0.1.0` is the first planned release and the
sections below describe what is on `master` today. The milestone ledger that
drives this work — including the rule that no box is ticked without measured
evidence — lives in [`MILESTONES.md`](https://github.com/ebx0/caustica/blob/master/MILESTONES.md).

## [Unreleased]

### Added

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

### Known issues

- **Element arrays still carry the staircase.** The repair above applies to
  `bowl` sources. `archimedean_spiral` and explicit element tables go through
  `TransducerArray.voxelize`, which projects each element as a disc sheared onto
  its own plane — an area of `π r² / cos(tilt)`, about 14% over at the
  production rim's 28°, kept for parity with the notebook the datasets came
  from. `caustica.geometry.offgrid` is the machinery to fix it the same way;
  nothing has measured what it would move yet.
- **`kappa` is cross-axis inconsistent** since the Nyquist fix. `sinc(c dt |k|/2)`
  still evaluates `|k|` with the true Nyquist component included, so on a
  Nyquist hyperplane the surviving transverse derivative is scaled by a
  dispersion factor evaluated at a wavevector the operator has just declared
  unrepresentable: up to ~10% amplitude error there at the shipped CFL, over
  planes carrying ~0.1% of the field's energy. Both conventions are defensible
  and neither has been measured against the analytic references yet.

[Unreleased]: https://github.com/ebx0/caustica/commits/master
