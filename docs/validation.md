# What has been measured

Every solver milestone is gated by tests against **analytic references** — O'Neil
(1949) focused bowl, the Rayleigh integral, Fubini nonlinear harmonic growth,
exponential absorption, plane-wave dispersion — **and** cross-validated against
[k-Wave](http://www.k-wave.org) running as a registry solver on identical grids,
media and sources.

None of this is a claim made in prose and checked by hand: all of it is
automated under `pytest`, and a milestone does not close until its gate is green.

## Current evidence

- **Plane-wave dispersion.** Phase-speed error < 0.1 % at 4 points per
  wavelength; measured absorption within 1 % of the configured α.
- **3-D focused bowl vs O'Neil.** Focus within one voxel, axial correlation
  r > 0.99, −6 dB widths within 5 %.
- **Westervelt vs Fubini.** Second-to-first harmonic ratio within 5 % (measured
  0.9–3.2 %) across σ = 0.06–0.61.
- **`linear` vs `kwave`.** Against the real OMP binary, 2-D water:
  normalized-field correlation r > 0.99.
- **Calibrated source amplitude.** The realized plane amplitude matches
  `source.amplitude` on both the native and the k-Wave path, invariant to grid,
  CFL and remote medium content. For a *curved* source the same calibration
  needs the source to carry its own area rather than its voxel count — see
  below.
- **Grid refinement, 1.9 to 15 points per wavelength.** An f/1.2 bowl in a few
  cubic millimetres of water at dx = 0.4, 0.2, 0.1 and 0.05 mm. Axial
  correlation with O'Neil reaches 0.998 by 7.5 points per wavelength and
  plateaus; the −6 dB width lands within 0.1 mm.
- **Two propagators on one digitized source.** Over the same ladder, the native
  solver and k-Wave converge onto each other: focal peaks 8.7 % apart at 3.8
  points per wavelength and 0.2 % apart at 15, where both land on O'Neil's
  absolute prediction to within a percent.
- **One phasor convention library-wide.** `p(t) = Re{P·e^(−iωt)}`, shared with
  the analytic references — see [the conventions that bite](conventions.md).

Figure-based comparison reports live under `benchmarks/reports/` in the
repository.

## How a curved source is represented, and why it matters

A flat source is easy: one voxel per `dx²` of aperture, exactly. A curved one
is not, and getting it wrong costs an absolute pressure rather than a shape.

Until 2026-08-24 a bowl was a one-voxel-thick shell with every voxel driven
alike. A digitized spherical cap crosses **1.18 voxels per `dx²` of its own
area**, so the bowl radiated in proportion to its voxel count instead of its
area, and the on-axis focal pressure sat 1.13–1.17× O'Neil's closed form. That
figure did *not* shrink with resolution — flat from 3.8 to 15 points per
wavelength — because a staircase factor is a property of digitizing a tilted
surface, not a discretization error. Nothing in the gates above would have
caught it: they compare normalized shape, peak position and −6 dB width, all of
which agreed well and improved with dx.

A bowl is now an **off-grid source**: the cap's closed-form area is divided
over equal-area quadrature points and each is deposited through a band-limited
interpolant, so the grid weights sum to that area whatever the surface's
orientation. This is the method k-Wave adopted for the same problem (Wise, Cox,
Jaros and Treeby, JASA 146, 2019). Measured over the same three rungs, the
absolute level now goes 1.136 → 1.023 → **0.999**, while the binary shell's
stayed at 1.162 → 1.146 → 1.165 — the difference between an ordinary
discretization error and a bias.

Two practical consequences:

- **A band-limited source is not thin.** It reaches a couple of voxels beyond
  the cap in every direction and carries negative weights in the interpolant's
  side-lobes; both are correct, and both mean a bowl needs slightly more
  clearance from the absorbing layer than its shell did. The constructor warns
  when part of the drive falls outside the domain.
- **`discretization: "binary"`** in a job's `source.array` block restores the
  old shell, for reproducing a result computed before this change.
- **This applies to `bowl` sources only.** Element arrays
  (`archimedean_spiral`, explicit element tables) still project each element as
  a sheared disc, whose area runs about 14 % over at a 28° rim tilt. Same
  defect, same available cure, not yet measured.

Measured in `benchmarks/reports/geometry/` and `benchmarks/reports/resolution/`;
pinned by `tests/test_geometry_fidelity.py` and `tests/test_sources.py`.

## What is *not* validated yet

The CuPy backend is packaged and has run on A100 hardware, but its parity and
full-size gates (milestone M7) are not closed. Every number on this page was
measured on the **CPU** path.

[MILESTONES.md](https://github.com/ebx0/caustica/blob/master/MILESTONES.md) is
the honest ledger: each milestone carries its acceptance criteria and whether it
is met.

Two of these gates are drawn, at the scale they are actually run, on the [examples page](examples.md): the focused bowl against O'Neil, and nonlinear steepening against Fubini.
