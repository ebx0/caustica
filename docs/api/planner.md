# Planner and results

## Will it fit? How long will it take?

The planner answers both **before** anything is allocated. It times one real step on the
machine you are on, inventories the engine's actual buffers rather than guessing at them, and
either estimates wall-clock or refuses with the fix named for that machine.

Three levels of confidence, and the planner says which one it used:

| source | how | typical error |
|---|---|---|
| `measured` | one real step, on this device | the honest one |
| `calibrated` | a fit recorded by `calibrate()` on this device | ±25 % |
| `db` | the packaged GPU datasheet | datasheet-coarse, ~2× |

::: caustica.planner.estimate

::: caustica.planner.Estimate

::: caustica.planner.calibration.calibrate

::: caustica.planner.calibration.measure_step_time

::: caustica.planner.calibration.predict_step_time

## Devices

::: caustica.planner.GPUSpec

::: caustica.planner.list_gpus

::: caustica.planner.spec_for_device

::: caustica.planner.load_gpu_db

## Results on disk

`caustica-result/1` is the HDF5 contract: written atomically, float16-quantized with a
recorded normalization error bound, resumable from an in-run checkpoint. The full layout —
and everything a GUI is allowed to depend on — is on the [GUI contract](../gui_contract.md)
page.

::: caustica.io.store.ResultStore

::: caustica.io.atomic.atomic_write

::: caustica.io.quantize.Quantized

## Volume media

::: caustica.io.medium_volume.MediumVolume

::: caustica.io.medium_volume.load_medium_volume

::: caustica.io.medium_volume.write_medium_volume
