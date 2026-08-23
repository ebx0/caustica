# Transducers

A transducer is an **element table** — positions, normals, radii — plus a way of phasing it.
Everything else (bowls, spirals, files) is a way of producing that table, and the table is
what gets voxelized onto the grid.

A run re-derives the aperture radius, f-number, half angle and surviving element count and
compares them against what was recorded, so a library change that silently builds a different
transducer is caught rather than trusted.

## Building an element table

::: caustica.arrays.transducer.TransducerArray

::: caustica.arrays.transducer.archimedean_spiral

::: caustica.arrays.elements.elements_array

::: caustica.arrays.elements.read_element_file

::: caustica.arrays.elements.element_table_digest

## Phasing

Delay-and-sum steering assumes water on the path. Aberration correction through tissue is a
planning problem, not a job knob.

::: caustica.arrays.phasemaps.build_phase_maps

::: caustica.arrays.phasemaps.select_phase_map_size

## Onto the grid

::: caustica.arrays.transducer.ArraySource

::: caustica.sources.bowl_cw_source
