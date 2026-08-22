# Conventions

Five things that make a caustica result **silently wrong** if you assume the
other convention. Nothing here changes a number in your job — it changes what
the numbers you get back *mean*.

If you read only one section, read [Coordinate frame](#3-coordinate-frame-z-is-the-beam-axis)
and [What `amplitude` means](#4-what-amplitude-means).

---

## 1. Phasor convention

caustica is a continuous-wave (CW) code: a run converges to a steady state and
records a complex phasor `P` per voxel, not a time trace. The convention,
library-wide and shared with the analytic references:

```
p(t) = Re{ P · e^(-i ω t) }          outgoing wave = e^(+i k x)
```

The `result.h5` file states it in its own metadata — the `phase_convention`
attribute, on the file root and on the `output` group — so a downstream tool
never has to guess:

```python
import h5py
with h5py.File("result.h5") as f:
    print(f.attrs["phase_convention"])
# p(t) = Re{P exp(-i omega t)}; outgoing wave = exp(+ikx) (matches the analytic
# O'Neil/Rayleigh references). The absolute phase zero is not source-referenced;
# relative phase (voxel-to-voxel) is exact.
```

**Read that last sentence before you export a phase map.** Voxel-to-voxel
phase *differences* are exact; the absolute zero is not tied to the source, so
a phase map lifted straight from `result.phase` carries an arbitrary global
offset. For hardware drive, reference every element to one of them.

**What this costs you if you assume `e^(+iωt)`:** every phase flips sign. An
amplitude-only analysis is unaffected; a delay-and-sum reconstruction, a phase
map exported to drive hardware, or a comparison against another simulator is
inverted. If you cross-check against a code that uses `e^(+iωt)` (much of the
optics and some of the ultrasound literature), conjugate one side.

`amp` is `|P|`, i.e. the **peak** pressure amplitude of the CW field, not RMS.
For a harmonic `h`, `harmonic_amp(h)` is the peak amplitude of that harmonic
component.

## 2. Absorption: Np/m, and only Np/m

`Material.alpha_np_m` is an absorption coefficient in **nepers per metre**, for
the pressure amplitude:

```
|p(z)| = |p(0)| · e^(-α z)
```

Most of the ultrasound literature quotes dB/cm or dB/cm/MHz instead. The
conversion lives in the library so you never do it by hand:

```python
from caustica.materials import db_cm_to_np_m, np_m_to_db_cm

db_cm_to_np_m(0.6)     # 0.6 dB/cm -> 6.9 Np/m
np_m_to_db_cm(6.0)     # 6.0 Np/m  -> 0.52 dB/cm
```

`1 dB/cm = 100 / (20·log₁₀ e) Np/m ≈ 11.5129 Np/m`.

**The v1 absorption model is frequency-independent.** `alpha_np_m` is the value
*at your drive frequency* — it is not scaled to the harmonics, and it is not a
power law. Two consequences:

- A material table is only valid for the `f0` it was built for. If you drive at
  another frequency, rebuild the table. A `medium_volume` file that records the
  frequency its α was baked at **refuses** to run at a different `drive.f0_mhz`
  rather than quietly using the wrong losses.
- Harmonics are absorbed at the fundamental's α. Real tissue absorbs `2f₀`
  roughly twice as strongly, so multi-harmonic amplitudes are optimistic. Treat
  harmonic ratios as an upper bound until the power-law model lands.

`beta` is the nonlinearity coefficient `β = 1 + B/2A` (water ≈ 3.5), not `B/A`.

## 3. Coordinate frame: +z is the beam axis

Two frames, one rule: **the beam always points along +z**.

- **Apex frame** — the transducer's own frame, used by every array builder and
  by an `elements` table. The array apex is at the origin, the beam axis is +z,
  and the geometric focus is at `(0, 0, roc)`. Element centers sit on the
  spherical shell; normals point at the focus.
- **Grid frame** — voxel indices `(i, j, k)` with axis 2 = z. A job places the
  transducer by giving `source.apex_mm`, the apex position **in the grid frame**;
  a `natural` focus then lands at `apex + (0, 0, roc)` and a `steered`
  `focus.target_mm` is a grid-frame point.

Everything in a job file is in **millimetres**; everything in the Python API is
in **metres** (SI). This is deliberate — job files are written by hand, the
library is not — and it is the single most common mistake when bringing your own
element table. An element table whose positions land more than a metre off-axis
is refused with a message that says "unit mistake" rather than running.

There is no origin offset: voxel `(0,0,0)` is at `0 mm` on every axis, so
`voxel = round(mm / dx_mm)`.

## 4. What `amplitude` means

`drive.amplitude_kpa` is **the pressure amplitude the source actually
realizes**, not a raw injection coefficient.

Naively adding `A·sin(ωt)` into a pressure field at the source voxels makes the
realized amplitude depend on `dx`, on the CFL number and on the medium — change
your grid and the "same" job drives a different transducer. caustica applies the
mass-source normalization `2·c·dt/dx` at the source voxels (k-Wave applies its
own equivalent internally), so:

> a plane source with `amplitude_kpa = 100` produces a plane wave of ≈100 kPa,
> rather than a number that moves when you change the discretization.

That is a measured property, and it is worth knowing exactly how far the
measurement goes, because the sentence above is easy to over-read:

- The native path is **asserted** by the suite
  (`test_mass_source_normalization_and_phase_convention`), with the realized
  plateau bounded to **−10 % / +12 %** of the requested amplitude. "Few
  per cent" is what is typically observed; ±10 % is what is guaranteed.
- **Invariance is asserted along one axis**, the one that used to break it:
  raising `c_max` (hence `dt` and steps-per-period) shifts the realized drive
  by **< 2 %**. Before the normalization the same change moved it ~20 %.
  Nothing in the suite sweeps `dx` or `cfl` for this property.
- The **k-Wave** cross-validation compares *normalized* fields (correlation
  and normalized L2), so it does not assert absolute amplitude at all. The
  adapter applies k-Wave's own equivalent scaling and measurement agrees
  (100 kPa requested → ~100 kPa realized), but that is an observation, not
  a gate.

Two caveats that are physics, not bookkeeping:

- **A focused source is not a plane source.** `amplitude_kpa` is the pressure at
  the *aperture*; the focus is higher by the focusing gain (tens of times, for a
  therapy bowl). Peak focal pressure is a result, not an input.
- **Element voxelization is discrete.** Each element becomes a one-voxel-thick
  disc, so a coarse `dx` changes the *radiating area* slightly even though the
  amplitude is normalized. An element that would lose **all** its voxels to
  deduplication is a hard refusal at build time, so the
  `derived.elements_represented` field in a run's `run_meta.json` always
  equals your element count — it is a receipt that the refusal did not fire,
  not a partial-loss counter. Partial area loss on a coarse grid is real and
  is not reported; halve `dx` and compare if it matters.

## 5. The PML is part of the grid

`grid.size_mm` is the **total** physical extent, sponge included. A
`size_mm: [18, 18, 24]` grid with `pml.thickness_mm: 3` has ~12 × 12 × 18 mm of
usable interior, because 3 mm is absorbed at each end of every axis.

The absorbing boundary is a multiplicative Gaussian sponge applied per axis:

```
s(r) = exp(-edge · ((width - r)/width)²),   r = 0 … width-1
```

so the outermost cell damps hardest and the profile blends to 1.0 at the inner
edge. `pml.edge` (default 2.0) sets how hard; `pml.thickness_mm` (default 5.0)
sets how wide. `thickness_mm: 0` disables it — useful only for periodic
plane-wave tests.

**Three ways this bites:**

1. **A source inside the sponge is damped as fast as it is driven.** The run
   still converges, on a field that is quietly wrong. `caustica validate`
   refuses it — but only because the check exists; nothing in the physics
   complains. Keep `source.apex_mm[2]` at least one PML thickness from the face.
2. **A focus near the boundary is not a focus.** Leave the focal region and its
   post-focal lobe inside the interior.
3. **Voxel counts include the sponge.** A record region given in voxels is in
   *full-grid* coordinates, PML included.

---

## Cheat sheet

| Quantity | Job file | Python API |
|---|---|---|
| Length | mm (`dx_mm`, `apex_mm`, `roc_mm`) | m (`dx`, `roc`) |
| Frequency | MHz (`f0_mhz`) | Hz (`f0`) |
| Pressure | kPa (`amplitude_kpa`) | Pa (`amplitude`) |
| Absorption | Np/m (`alpha_np_m`) | Np/m |
| Phase | rad (`phases_rad`) | rad |
| Angles | degrees (`angle_deg`) | radians in the analytic layer |
| Voxel counts | never written, always derived | derived |

See [job_reference.md](job_reference.md) for every field of the job file.
