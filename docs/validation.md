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
  CFL and remote medium content.
- **One phasor convention library-wide.** `p(t) = Re{P·e^(−iωt)}`, shared with
  the analytic references — see [the conventions that bite](conventions.md).

Figure-based comparison reports live under `benchmarks/reports/` in the
repository.

## What is *not* validated yet

The CuPy backend is packaged and has run on A100 hardware, but its parity and
full-size gates (milestone M7) are not closed. Every number on this page was
measured on the **CPU** path.

[MILESTONES.md](https://github.com/ebx0/caustica/blob/master/MILESTONES.md) is
the honest ledger: each milestone carries its acceptance criteria and whether it
is met.
