# Geometry

Shapes are **implicit**: a shape is a predicate on a point, so it lands straight on whatever
grid you ask for at whatever resolution you chose. There is no meshing step and no resampling
of a mesh — rasterization evaluates the predicate per voxel, optionally supersampled.

Booleans are operators on shapes, so `(a | b) - c` is a shape like any other and can be
rotated, translated and rasterized as one.

## Scenes

::: caustica.geometry.scene.Scene

::: caustica.geometry.configs.SceneConfig

## Shapes

::: caustica.geometry.Shape

::: caustica.geometry.Ball

::: caustica.geometry.Box

::: caustica.geometry.Ellipsoid

::: caustica.geometry.Cylinder

::: caustica.geometry.HalfSpace

## Combining them

::: caustica.geometry.Union

::: caustica.geometry.Intersection

::: caustica.geometry.Difference

::: caustica.geometry.Complement

::: caustica.geometry.AffineShape

## Segmented volumes

For anatomy that was never a solid: a labelled volume from a segmentation, resampled onto
your grid. See [anatomical phantoms](../uwcem.md) for the packaged datasets.

::: caustica.geometry.volumes.LabelVolume

::: caustica.geometry.configs.VolumeImportConfig

::: caustica.geometry.volumes.load_labels_txt

::: caustica.geometry.volumes.resample_scalar

## Off-grid sources

A binary voxel mask cannot represent the area of an oblique surface: a digitized spherical
cap crosses about 1.18 voxels per `dx²` of its own area, and since the engine drives every
source voxel alike, a shell-shaped bowl radiates in proportion to its voxel count instead.
These functions describe a source as a *measure* rather than a set of voxels — the surface's
closed-form area, spread over grid weights by a band-limited interpolant — which is what
makes a curved source's realized amplitude match the one that was asked for.

::: caustica.geometry.offgrid.spherical_cap_deposit

::: caustica.geometry.offgrid.band_limited_weights

::: caustica.geometry.offgrid.Deposit

::: caustica.geometry.offgrid.star_offsets
