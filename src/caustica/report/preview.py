"""The preview package (M10d): a <=10 MB answer to "did the run work?".

A full-field ``result.h5`` from a Colab session is 0.5-0.8 GB on Drive;
waiting for it to sync just to learn the focus missed is absurd. The runner
therefore writes, NEXT to the result:

* ``preview.npz``  — peak slices along the 3 axes for every recorded
  harmonic and ``p_max``, a block-mean coarsened fundamental amplitude
  volume, mm axes, and the convergence history. Slices/volume are stored
  with the M10 dynamic-float16 contract (measured error <= 1e-3 * peak,
  float32 fallback), each array next to its ``*_scale``.
* ``metrics.json`` — every number :func:`caustica.report.metrics.focus_metrics`
  produces, machine-readable (a future GUI's result tab reads exactly this).

``caustica report --preview`` renders a quick look from the package alone.
The coarsening step is chosen from the FULL field's size so the criterion
(<= 10 MB, measured on a full grid) holds for any grid; the written size is
verified before publishing and the step is enlarged if compression ever
undershoots the estimate.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np

from caustica.io.atomic import atomic_write
from caustica.io.quantize import restore, try_float16
from caustica.report.metrics import (
    FieldFrame,
    argmax_interior,
    interior_slices,
    mm_axes,
    to_grid,
)
from caustica.solvers.base import SolverResult

PREVIEW_FORMAT = "caustica-preview/1"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
#: Fraction of the budget given to the coarse volume; slices + axes +
#: overhead take the rest (they are bounded by 2-D planes, far smaller).
_COARSE_BUDGET_FRACTION = 0.6


def block_mean(arr: np.ndarray, step: int) -> np.ndarray:
    """Block-mean coarsening by ``step`` per axis (edge remainder cropped).

    A preview volume, not provenance: the up-to-``step - 1`` trailing voxels
    dropped per axis sit against the PML where the field is sponge-damped
    anyway.
    """
    if step <= 1:
        return np.asarray(arr, dtype=np.float32)
    ns = [(n // step) * step for n in arr.shape]
    a = arr[tuple(slice(0, n) for n in ns)].astype(np.float32)
    for ax, n in enumerate(ns):
        a = a.reshape(*a.shape[:ax], n // step, step, *a.shape[ax + 1 :]).mean(axis=ax + 1)
    return a


def _coarse_step(shape: tuple[int, ...], budget_bytes: int) -> int:
    n = math.prod(shape)
    return max(1, math.ceil((2.0 * n / max(budget_bytes, 1)) ** (1.0 / 3.0)))


def _put_quantized(arrays: dict, name: str, arr: np.ndarray) -> None:
    q = try_float16(arr)
    arrays[name] = q.stored
    arrays[f"{name}_scale"] = np.asarray(q.scale, dtype=np.float64)


def build_preview(
    result: SolverResult,
    frame: FieldFrame,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> bytes:
    """The package as npz BYTES — the same package, not yet written anywhere.

    Split out of :func:`write_preview` so ``simulate(out=None)`` can hand a
    notebook the identical <=10 MB package without touching the disk. The
    budget is still MEASURED here, on the compressed bytes, not estimated.
    """
    amp = result.amp
    interior = interior_slices(result, frame)
    pk = argmax_interior(amp, interior)
    x_mm, y_mm, z_mm = mm_axes(result, frame)

    step = _coarse_step(amp.shape, int(max_bytes * _COARSE_BUDGET_FRACTION))
    for _attempt in range(4):
        arrays: dict = {}
        _put_quantized(arrays, "coarse_amp", block_mean(amp, step))
        for h in sorted(result.phasors):
            ah = result.harmonic_amp(h)
            _put_quantized(arrays, f"amp_h{h}_slice_x", ah[pk[0], :, :])
            _put_quantized(arrays, f"amp_h{h}_slice_y", ah[:, pk[1], :])
            _put_quantized(arrays, f"amp_h{h}_slice_z", ah[:, :, pk[2]])
        _put_quantized(arrays, "p_max_slice_y", result.p_max[:, pk[1], :])
        arrays["x_mm"] = np.asarray(x_mm, np.float32)
        arrays["y_mm"] = np.asarray(y_mm, np.float32)
        arrays["z_mm"] = np.asarray(z_mm, np.float32)
        arrays["convergence_history"] = np.asarray(
            result.convergence_history, dtype=np.float64
        ).reshape(-1, 3)
        arrays["meta_json"] = np.str_(
            json.dumps(
                {
                    "format": PREVIEW_FORMAT,
                    "dx_m": float(frame.dx),
                    "grid_shape": list(frame.grid_shape),
                    "pml_vox": int(frame.pml_vox),
                    "region_start": [int(s.start) for s in result.region],
                    "region_stop": [int(s.stop) for s in result.region],
                    "apex_vox": list(frame.apex_vox),
                    "focus_vox": None if frame.focus_vox is None else list(frame.focus_vox),
                    "peak_voxel_grid": list(to_grid(pk, result)),
                    "coarse_step": int(step),
                    "coarse_note": (
                        "block mean over step^3 blocks of the fundamental amplitude; "
                        "block i covers region voxels [i*step, (i+1)*step)"
                    ),
                    "harmonics": sorted(result.phasors),
                    "slices": "through the realized peak voxel (not the geometric center)",
                }
            )
        )

        # Size the package in memory BEFORE publishing: the budget math is an
        # estimate, the criterion is a measurement.
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        payload = buf.getvalue()
        if len(payload) <= max_bytes:
            break
        step += 1
    else:  # pragma: no cover - 4 growth attempts always suffice
        raise RuntimeError(
            f"preview would be {len(payload)} bytes even at coarse step {step}; "
            f"budget is {max_bytes}"
        )
    return payload


def write_preview(
    outdir: str | Path,
    result: SolverResult,
    frame: FieldFrame,
    *,
    metrics: dict,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Path:
    """Write ``preview.npz`` + ``metrics.json`` into ``outdir`` (atomic)."""
    outdir = Path(outdir)
    payload = build_preview(result, frame, max_bytes=max_bytes)
    path = outdir / "preview.npz"
    with atomic_write(path) as tmp:
        tmp.write_bytes(payload)
    with atomic_write(outdir / "metrics.json") as tmp:
        tmp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def _decode(npz) -> dict:
    """Arrays with their scales applied, plus the parsed ``meta``."""
    out: dict = {}
    names = set(npz.files)
    for name in names:
        if name.endswith("_scale") or name == "meta_json":
            continue
        arr = npz[name]
        if f"{name}_scale" in names:
            arr = restore(arr, npz[f"{name}_scale"])
        out[name] = arr
    out["meta"] = json.loads(str(npz["meta_json"]))
    return out


def load_preview(path: str | Path) -> dict:
    """Read a preview package back: scales applied, ``meta`` parsed."""
    with np.load(Path(path), allow_pickle=False) as npz:
        out = _decode(npz)
    if out["meta"].get("format") != PREVIEW_FORMAT:
        raise ValueError(f"{path}: not a {PREVIEW_FORMAT} package")
    return out


def decode_preview(payload: bytes) -> dict:
    """Same as :func:`load_preview`, for a package that is only in memory."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as npz:
        out = _decode(npz)
    if out["meta"].get("format") != PREVIEW_FORMAT:
        raise ValueError(f"not a {PREVIEW_FORMAT} package")
    return out
