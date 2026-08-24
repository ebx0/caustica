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
- **Calibrated source amplitude — for a *plane* source.** The realized plane
  amplitude matches `source.amplitude` on both the native and the k-Wave path,
  invariant to grid, CFL and remote medium content. It does **not** carry over
  to a curved source: see the focused-bowl amplitude limit below.
- **Grid refinement, 1.9 to 15 points per wavelength.** An f/1.2 bowl in a few
  cubic millimetres of water at dx = 0.4, 0.2, 0.1 and 0.05 mm. Axial
  correlation with O'Neil reaches 0.998 by 7.5 points per wavelength and
  plateaus; the −6 dB width lands within 0.1 mm.
- **Two propagators on one digitized source.** Over the same ladder, the native
  solver and k-Wave converge onto each other: focal peaks 9.1 % apart at 3.8
  points per wavelength and 0.1 % apart at 15.
- **One phasor convention library-wide.** `p(t) = Re{P·e^(−iωt)}`, shared with
  the analytic references — see [the conventions that bite](conventions.md).

Figure-based comparison reports live under `benchmarks/reports/` in the
repository.

## A limit worth knowing before you quote a pressure

**A focused bowl radiates about 15 % more than its aperture implies.** The
engine drives every source voxel with the same normalized amplitude, which is
exact for a flat source — one voxel per `dx²` of aperture — and is not for a
curved one. A digitized spherical cap crosses 1.18 voxels per `dx²` of its own
area at the sampling the library ships, so the bowl radiates in proportion to
its voxel count rather than its area, and the on-axis focal pressure sits
1.13–1.17× O'Neil's closed form.

That figure does **not** shrink with resolution: it is flat from 3.8 to 15
points per wavelength, because a staircase factor is a property of digitizing a
tilted surface and not a discretization error. k-Wave, driven from the same
voxel set through a completely different propagator, lands on the same excess.

None of the gates above would catch it, and that is the point of naming it
here: the analytic suite compares normalized shape, peak position and −6 dB
width, all of which agree well and improve with resolution. **Beam shapes,
focal positions and relative comparisons are unaffected. An absolute pressure
in pascals is high by roughly this factor.** Measured in
`benchmarks/reports/geometry/` and `benchmarks/reports/resolution/`, and pinned
by `tests/test_geometry_fidelity.py`.

## What is *not* validated yet

The CuPy backend is packaged and has run on A100 hardware, but its parity and
full-size gates (milestone M7) are not closed. Every number on this page was
measured on the **CPU** path.

[MILESTONES.md](https://github.com/ebx0/caustica/blob/master/MILESTONES.md) is
the honest ledger: each milestone carries its acceptance criteria and whether it
is met.

Two of these gates are drawn, at the scale they are actually run, on the [examples page](examples.md): the focused bowl against O'Neil, and nonlinear steepening against Fubini.
