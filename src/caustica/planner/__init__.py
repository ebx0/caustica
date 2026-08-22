"""Pre-run resource planner (M8): "will it fit, how long will it take?"

COMSOL-feel contract: before committing a Colab GPU (or hours of CPU), ask::

    from caustica import planner
    est = planner.estimate(grid, medium, src, spec, solver="westervelt", gpu="A100")
    est.fits, est.t_expected_s, est.source     # source: "db"|"calibrated"|"measured"
    print(planner.compare(grid, medium, src, spec))   # all known GPUs, one table

Three estimate sources, always labeled on the result (M8 gate):

* ``"db"`` — datasheet numbers from ``gpu_db.json``; coarse (~2x), for
  device comparison before touching hardware.
* ``"calibrated"`` — ``t_step = a*P*log2(P) + b*P`` coefficients fitted by
  :func:`caustica.planner.calibrate` (~20 real steps on the device, persisted
  in ``~/.caustica/calibration.json``). The ±25% accuracy gate applies here.
* ``"measured"`` — ``measure=True`` times the step composition on the
  CURRENT machine right now (~20 steps).

Wall time is ``warmup + steps * t_step`` (fix A2). The constant term is the
one-time cost a GPU solve pays for cuFFT plans, kernel compilation and the
first allocations — dropping it made the first real Colab session look like a
25.9x miss when the per-step accounting was in fact correct (see
``model.GPU_WARMUP_S``). ``t_expected_s`` therefore INCLUDES the warmup, and
``Estimate.warmup_s`` says how much of it that is.

VRAM comes from a byte-level inventory of the engine's buffers
(:mod:`caustica.planner.model`); when it does not fit, ``advice`` carries
actionable fixes (coarser dx / smaller record AOI / linear solver / larger
device).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from math import ceil
from pathlib import Path

from caustica.core.grid import Grid
from caustica.medium import Medium
from caustica.planner import model
from caustica.planner.calibration import (
    calibrate,
    default_calibration_path,
    find_calibration_for,
    measure_step_time,
    record_warmup,
)
from caustica.solvers.base import CWRunSpec
from caustica.solvers.kspace.engine import (
    cw_discretization,
    cw_tof_periods,
    normalize_record_region,
)
from caustica.sources import CWSource

__all__ = [
    "Comparison",
    "Estimate",
    "GPUSpec",
    "calibrate",
    "compare",
    "default_calibration_path",
    "estimate",
    "list_gpus",
    "load_gpu_db",
    "measure_step_time",
    "record_warmup",
]

_GIB = 2**30


@dataclass(frozen=True)
class GPUSpec:
    """Datasheet-nominal device description from ``gpu_db.json``."""

    key: str
    vram_gib: float
    mem_bw_gbs: float
    fp32_tflops: float
    notes: str = ""

    @property
    def vram_bytes(self) -> int:
        return int(self.vram_gib * _GIB)

    @property
    def usable_bytes(self) -> int:
        """VRAM available to caustica: capacity minus CUDA context/runtime."""
        return max(0, self.vram_bytes - model.CONTEXT_OVERHEAD_BYTES)


def load_gpu_db() -> tuple[dict[str, GPUSpec], dict[str, str]]:
    """Device specs and alias map shipped with the package."""
    raw = json.loads(resources.files("caustica.planner").joinpath("gpu_db.json").read_text())
    devices = {
        key: GPUSpec(key=key, **{k: v for k, v in spec.items()})
        for key, spec in raw["devices"].items()
    }
    return devices, dict(raw.get("aliases", {}))


def list_gpus() -> list[str]:
    devices, aliases = load_gpu_db()
    return sorted(devices) + sorted(aliases)


def _resolve_gpu(name: str) -> GPUSpec:
    devices, aliases = load_gpu_db()
    key = aliases.get(name, name)
    if key not in devices:
        raise ValueError(f"unknown gpu {name!r}; known: {', '.join(list_gpus())}")
    return devices[key]


@dataclass(frozen=True)
class Estimate:
    """One planner verdict for (setup, device). All times are wall-clock."""

    solver: str
    gpu: str
    source: str  # "db" | "calibrated" | "measured"
    vram_bytes: int
    vram_breakdown: dict[str, int]
    vram_usable_bytes: int
    fits: bool
    dt: float
    spp: int
    t_step_s: float
    warmup_s: float
    steps_expected: int
    steps_worst: int
    t_expected_s: float
    t_worst_s: float
    warnings: tuple[str, ...]
    advice: tuple[str, ...]

    @property
    def vram_gib(self) -> float:
        return self.vram_bytes / _GIB

    def summary(self) -> str:
        lines = [
            f"planner [{self.source}] {self.solver} on {self.gpu}: "
            f"{'FITS' if self.fits else 'DOES NOT FIT'} "
            f"({self.vram_gib:.2f} GiB needed / {self.vram_usable_bytes / _GIB:.2f} GiB usable)",
            f"  t_step={self.t_step_s * 1e3:.2f} ms, spp={self.spp}, "
            f"warmup={self.warmup_s:.1f} s (one-time), "
            f"expected={_fmt_t(self.t_expected_s)} ({self.steps_expected} steps), "
            f"worst-case={_fmt_t(self.t_worst_s)}",
        ]
        lines += [f"  ! {w}" for w in self.warnings]
        lines += [f"  -> {a}" for a in self.advice]
        return "\n".join(lines)


def _fmt_t(seconds: float) -> str:
    if seconds < 120.0:
        return f"{seconds:.1f} s"
    if seconds < 7200.0:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds / 3600.0:.2f} h"


def estimate(
    grid: Grid,
    medium: Medium,
    source: CWSource,
    spec: CWRunSpec | None = None,
    *,
    solver: str = "westervelt",
    gpu: str = "A100",
    harmonics: tuple[int, ...] = (1,),
    record_region: tuple[slice, ...] | None = None,
    reference_point: tuple[int, ...] | None = None,
    measure: bool = False,
    measure_backend: str = "auto",
    calibration_path: str | Path | None = None,
) -> Estimate:
    """Predict VRAM and wall time for one CW solve WITHOUT running it.

    Mirrors the engine's own discretization (same dt/spp — single source of
    truth) and settling policy: ``expected`` assumes convergence at
    time-of-flight + ``min_settle_periods``; ``worst`` runs to
    ``max_settle_periods``. ``measure=True`` times ~20 real steps on the
    current machine and overrides the model (source ``"measured"``) —
    ``measure_backend`` must be the backend the run will actually use: on a
    GPU machine forced to numpy, an ``"auto"`` probe would time cuFFT and
    label an hours-wrong number "measured" (review finding, 2026-08-22).
    """
    if solver not in ("linear", "westervelt"):
        raise ValueError(
            f"planner models the native k-space engine ('linear'/'westervelt'); "
            f"'{solver}' runs out-of-process and is not modeled."
        )
    spec = spec if spec is not None else CWRunSpec()
    nonlinear = solver == "westervelt" and not medium.is_linear
    gpu_spec = _resolve_gpu(gpu)

    spp, dt = cw_discretization(grid, medium, spec, source.f0, tuple(harmonics))
    tof = cw_tof_periods(grid, medium, source, reference_point)

    rec_slices = (
        normalize_record_region(record_region, grid.shape)
        if record_region is not None
        else tuple(slice(0, n) for n in grid.shape)
    )
    rec_elems = 1
    for sl in rec_slices:
        rec_elems *= sl.stop - sl.start

    mem = model.kspace_memory(grid.shape, nonlinear, len(harmonics), rec_elems)
    p_elems = 1
    for n in mem.padded_shape:
        p_elems *= n

    warnings: list[str] = []
    if measure:
        run = measure_step_time(
            grid.shape, nonlinear=nonlinear, backend=measure_backend, n_steps=20
        )
        t_step = run["t_step_s"]
        warmup_s = float(run.get("warmup_s", 0.0))
        src_label = "measured"
        if run["backend"] == "numpy":
            warnings.append(
                f"measured on the CPU backend — wall time reflects THIS machine, not {gpu_spec.key}"
            )
    else:
        entry = find_calibration_for(gpu_spec.key, calibration_path)
        if entry is not None:
            t_step = model.step_time(entry["a"], entry["b"], p_elems)
            # Entries written before fix A2 carry no warmup at all; a GPU
            # entry without one gets the constant rather than a silent zero,
            # which is the term the whole fix exists to stop dropping.
            warmup_s = float(entry.get("warmup_s", model.GPU_WARMUP_S))
            src_label = "calibrated"
        else:
            a, b = model.db_time_coeffs(
                gpu_spec.fp32_tflops, gpu_spec.mem_bw_gbs, grid.ndim, nonlinear
            )
            t_step = model.step_time(a, b, p_elems)
            warmup_s = model.GPU_WARMUP_S
            src_label = "db"
            warnings.append(
                "db estimate is datasheet-coarse (~2x); run planner.calibrate() on the "
                "target device for the ±25% path"
            )

    # ---- settling policy → step counts (engine semantics, incl. the
    # ramp guard: convergence cannot arm before the source ramp ends) ----
    period = 1.0 / source.f0
    settle_floor = max(spec.min_settle_periods, ceil(source.ramp_periods) + 1)
    pre_expected = tof + settle_floor
    pre_worst = max(tof + spec.max_settle_periods, pre_expected)
    if spec.t_end_min_us is not None:
        need = ceil(spec.t_end_min_us * 1e-6 / period) - spec.n_record_periods
        pre_expected = max(pre_expected, need)
        pre_worst = max(pre_worst, need)
    steps_expected = (pre_expected + spec.n_record_periods) * spp
    steps_worst = (pre_worst + spec.n_record_periods) * spp

    fits = mem.total_bytes <= gpu_spec.usable_bytes
    advice: list[str] = []
    if not fits:
        factor = mem.total_bytes / max(gpu_spec.usable_bytes, 1)
        m = factor ** (1.0 / grid.ndim)
        new_shape = tuple(max(8, int(n / m)) for n in grid.shape)
        advice.append(
            f"increase dx by >= x{m:.2f} (same physical extent at grid ~{new_shape}; "
            f"voxel count scales 1/m^{grid.ndim})"
        )
        rec_bytes = mem.breakdown["record buffers"]
        if rec_bytes > 0.10 * mem.total_bytes:
            advice.append(
                f"shrink the record region (AOI): record buffers are "
                f"{rec_bytes / _GIB:.2f} GiB — pass record_region=... to run()"
            )
        if len(harmonics) > 1:
            advice.append(
                f"each extra harmonic costs {rec_elems * 8 / _GIB:.2f} GiB of record buffer — "
                f"drop harmonics you do not need"
            )
        if nonlinear:
            nl_bytes = 3 * 4 * p_elems  # beta map + 2 extra step temporaries
            advice.append(
                f"the 'linear' solver drops the beta map and nonlinear temporaries "
                f"(~{nl_bytes / _GIB:.2f} GiB) — valid only if harmonics are not needed"
            )
        bigger = [
            k
            for k, s in load_gpu_db()[0].items()
            if s.usable_bytes >= mem.total_bytes and s.key != gpu_spec.key
        ]
        if bigger:
            advice.append(f"or pick a larger device: {', '.join(sorted(bigger))}")

    return Estimate(
        solver=solver,
        gpu=gpu_spec.key,
        source=src_label,
        vram_bytes=mem.total_bytes,
        vram_breakdown=dict(mem.breakdown),
        vram_usable_bytes=gpu_spec.usable_bytes,
        fits=fits,
        dt=dt,
        spp=spp,
        t_step_s=t_step,
        warmup_s=warmup_s,
        steps_expected=steps_expected,
        steps_worst=steps_worst,
        # A GPU run pays its plan/JIT/allocation cost ONCE, not per step; a
        # model without the constant term under-predicts every short run and
        # is what made the first Colab session look like a 25.9x miss (A2).
        t_expected_s=warmup_s + t_step * steps_expected,
        t_worst_s=warmup_s + t_step * steps_worst,
        warnings=tuple(warnings),
        advice=tuple(advice),
    )


@dataclass(frozen=True)
class Comparison:
    """Planner verdicts for one setup across devices; ``print()`` it."""

    estimates: tuple[Estimate, ...]

    def table(self) -> str:
        header = (
            f"{'GPU':<10} {'fits':<5} {'VRAM GiB':>9} {'usable':>7} "
            f"{'t_step':>9} {'expected':>10} {'worst':>10} {'src':<10}"
        )
        rows = [header, "-" * len(header)]
        for e in self.estimates:
            rows.append(
                f"{e.gpu:<10} {'yes' if e.fits else 'NO':<5} {e.vram_gib:>9.2f} "
                f"{e.vram_usable_bytes / _GIB:>7.1f} {e.t_step_s * 1e3:>7.2f}ms "
                f"{_fmt_t(e.t_expected_s):>10} {_fmt_t(e.t_worst_s):>10} {e.source:<10}"
            )
        return "\n".join(rows)

    def __str__(self) -> str:
        return self.table()


def compare(
    grid: Grid,
    medium: Medium,
    source: CWSource,
    spec: CWRunSpec | None = None,
    *,
    gpus: tuple[str, ...] | None = None,
    **kwargs,
) -> Comparison:
    """Run :func:`estimate` across devices; sorted fits-first, fastest-first."""
    devices, _ = load_gpu_db()
    keys = tuple(gpus) if gpus is not None else tuple(sorted(devices))
    ests = [estimate(grid, medium, source, spec, gpu=k, **kwargs) for k in keys]
    ests.sort(key=lambda e: (not e.fits, e.t_expected_s))
    return Comparison(estimates=tuple(ests))
