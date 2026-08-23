# Grid, media, materials

## The grid

::: caustica.core.grid.Grid

::: caustica.core.pml.PMLSpec

## Materials and media

A `Material` is the physics of one substance; a `MaterialDB` maps integer labels to
materials; a `Medium` is those materials painted onto a grid, one value per voxel.

::: caustica.materials.Material

::: caustica.materials.MaterialDB

::: caustica.medium.Medium

## Backends

One array API, two implementations. `"auto"` means "cupy if this machine has a usable one,
numpy otherwise" — and says so out loud rather than falling back in silence.

::: caustica.core.backend.get_backend

::: caustica.core.backend.cupy_available

::: caustica.core.backend.CausticaWarning

::: caustica.env.env_report
