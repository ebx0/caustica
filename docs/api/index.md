# API reference

Generated from the docstrings in the source, so it cannot drift from the code.

caustica has two levels, and both are supported:

**The facade.** [`simulate()`](simulate.md#caustica.facade.simulate) takes a job — a path, a
dict, a validated model — and hands back the field, the metrics and the plan. It runs the
same `build_job → plan → gates → solve` path the command line runs, so a Python call and
`caustica run` cannot disagree about what a job means.

**The pieces underneath.** `Grid`, `Medium`, a source, a solver from the registry. This is
what the facade is made of, and what you drop to when you want something the job schema does
not express — a custom time loop, a field consumed by the thermal chain, a solver of your own.

| Page | What is in it |
|---|---|
| [Running a simulation](simulate.md) | `simulate`, the run object, the solver registry, parameter studies |
| [Grid, media, materials](model.md) | the discretization, the absorbing border, per-voxel physics, backends |
| [Geometry](geometry.md) | constructive solid geometry and segmented-volume import |
| [Transducers](arrays.md) | element tables, spiral arrays, phasing, voxelization |
| [Analytic references](analytic.md) | O'Neil, Rayleigh, Fubini — the ground truth the tests gate on |
| [Thermal](thermal.md) | `Q = 2αI`, Pennes bioheat, CEM43 |
| [Planner and results](planner.md) | will-it-fit / how-long, and the result and volume formats |

Names that are *not* on these pages are not API. The five [extension points](../extending.md)
and the [GUI contract](../gui_contract.md) are the other two frozen surfaces; everything else
is free to move between milestones.

## Conventions that apply everywhere

- **SI in, SI out.** The Python API is metres, pascals, hertz, Np/m. Only the JSON job schema
  speaks millimetres and MHz, and it converts at the boundary.
- **`p(t) = Re{P·e^(−iωt)}`.** One phasor convention library-wide, shared with the analytic
  references. Getting this backwards conjugates every phase you compute — see
  [conventions](../conventions.md#1-phasor-convention).
- **`+z` is the beam axis**, and the PML is *inside* the box you asked for.
- **Warnings are `CausticaWarning`s**, so you can filter ours without silencing anyone else's.
