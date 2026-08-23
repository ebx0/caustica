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
  own magnitude. numpy's pocketfft projects that away; cuFFT, which documents
  the input as Hermitian, resolved it differently at 256³ and reached NaN by
  the second period while the identical CPU run stayed at 45 kPa. The Nyquist
  bin is now dropped — which is also simply the right derivative on a
  collocated grid — and the gates are written on the transform's *input*, so
  they fail on any machine rather than only on a GPU.
- **A diverged run no longer exits 0.** A NaN field used to run to the settle
  cap and write a result file full of NaN; it now raises `SolverDivergedError`
  at the first non-finite period.
- **VRAM measurement** is taken in the rung's own fresh process, so an earlier
  rung's pool cannot be counted as this one's peak.
- **Calibration** sizes its probes as a fraction of free VRAM and interpolates
  its measurements instead of forcing one power law per card.

[Unreleased]: https://github.com/ebx0/caustica/commits/master
