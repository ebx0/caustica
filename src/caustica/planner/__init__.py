"""Pre-run resource planner (M8): "will it fit, how long will it take?"

COMSOL-feel contract: before committing a Colab GPU (or hours of CPU), ask::

    from caustica import planner
    est = planner.estimate(grid, medium, src, spec, solver="westervelt", gpu="A100")
    est.fits, est.t_expected_s, est.source     # source: "db"|"calibrated"|"measured"
    print(planner.compare(grid, medium, src, spec))   # all known GPUs, one table

Three estimate sources, always labeled on the result (M8 gate):

* ``"db"`` — datasheet numbers from ``gpu_db.json``; coarse (~2x), for
  device comparison before touching hardware.
* ``"calibrated"`` — interpolated through the ``(P, t_step)`` samples
  :func:`caustica.planner.calibrate` measured on the device (~20 real steps
  per size, persisted in ``~/.caustica/calibration.json``); stores too old to
  carry samples fall back to their ``t_step = a*P*log2(P) + b*P`` fit. The
  ±25% accuracy gate applies here — and it is why prediction interpolates:
  one global law through a real device's per-element cost curve missed its
  own probes by 70% on a Blackwell card.
* ``"measured"`` — ``measure=True`` times the step composition on the
  CURRENT machine right now (~20 steps).

Wall time is ``warmup + steps * t_step`` (fix A2). The constant term is the
one-time cost a GPU solve pays for cuFFT plans, kernel compilation and the
first allocations — dropping it made the first real Colab session look like a
25.9x miss when the per-step accounting was in fact correct (see
``model.GPU_WARMUP_S``). ``t_expected_s`` therefore INCLUDES the warmup, and
``Estimate.warmup_s`` says how much of it that is -- zero by default on a
cpu target, which pays no CUDA context and no cuFFT plan. It is NOT a
constant and not linear in the grid either: like the per-step cost it is
interpolated from measured runs when the calibration carries them (see
:func:`caustica.planner.calibrated_warmup`); a real card's warmup is
superlinear in the element count.

VRAM comes from a byte-level inventory of the engine's buffers
(:mod:`caustica.planner.model`); when it does not fit, ``advice`` carries
actionable fixes (coarser dx / smaller record AOI / linear solver / larger
device).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from importlib import resources
from math import ceil
from pathlib import Path

from caustica.core.grid import Grid
from caustica.medium import Medium
from caustica.planner import model
from caustica.planner.calibration import (
    EXTRAPOLATION_WARN_FACTOR,
    FIT_RESIDUAL_LIMIT,
    calibrate,
    calibrated_warmup,
    default_calibration_path,
    default_probe_shapes,
    find_calibration_for,
    measure_step_time,
    predict_step_time,
    record_warmup,
    record_warmup_model,
    time_model_quality,
    warmup_model_quality,
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
    "calibrated_warmup",
    "compare",
    "default_calibration_path",
    "estimate",
    "gpu_key_for_device",
    "is_cpu_target",
    "list_gpus",
    "load_gpu_db",
    "default_probe_shapes",
    "measure_step_time",
    "predict_step_time",
    "spec_for_device",
    "record_warmup",
    "record_warmup_model",
    "time_model_quality",
    "warmup_model_quality",
]

_GIB = 2**30


@dataclass(frozen=True)
class GPUSpec:
    """Device description: a ``gpu_db.json`` row, or a live device.

    ``device_name`` is the CUDA product string when this spec describes a
    real device rather than a datasheet row; it is what the calibration
    store is keyed by, so carrying it is what lets a card the database has
    never heard of still be planned from its own measurements.
    """

    key: str
    vram_gib: float
    mem_bw_gbs: float
    fp32_tflops: float
    notes: str = ""
    device_name: str | None = None
    source: str = "db"  # "db" (a datasheet row) | "device" (read from the card)

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


def _norm_device(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def gpu_key_for_device(device_name: str, total_gib: float | None = None) -> str | None:
    """Best ``gpu_db.json`` key for a CUDA product string, or None.

    None -- not a guess -- is the point. Falling back to a fixed key made an
    RTX PRO 6000 Blackwell plan itself as an A100: the 95 GiB card was judged
    against 38.88 GiB of usable VRAM, and the calibration it had just
    measured became unreachable because the lookup searched for "a100" in a
    name that does not contain it (measured, 2026-08-22). On a card SMALLER
    than the guessed one the same fallback accepts a run that cannot fit.
    """
    devices, _ = load_gpu_db()
    norm = _norm_device(device_name)
    best: tuple[int, float, str] | None = None
    for key, spec in devices.items():
        # .lower() BEFORE stripping punctuation, or "A100" becomes "100" and
        # "SXM" becomes "" -- and an empty token is a substring of anything.
        tokens = [tok for tok in (_norm_device(part) for part in key.split("-")) if tok]
        if not tokens or tokens[0] not in norm:
            continue
        score = sum(1 for t in tokens if t in norm)
        gap = -abs(spec.vram_gib - total_gib) if total_gib is not None else 0.0
        cand = (score, gap, key)
        if best is None or cand > best:
            best = cand
    return best[2] if best is not None else None


def spec_for_device(device_name: str, total_bytes: int | None = None) -> GPUSpec:
    """A :class:`GPUSpec` for a real device, database row or not.

    A card the database knows comes back as its datasheet row with the live
    name attached, so the calibration lookup can be exact. A card it does
    not know is described by what the device itself reports -- real VRAM,
    key ``unknown:<name>`` -- with throughput taken from the median known
    device, which is a guess the ``db`` label already warns about and which
    one calibration run replaces outright.
    """
    total_gib = (total_bytes / _GIB) if total_bytes else None
    devices, _ = load_gpu_db()
    key = gpu_key_for_device(device_name, total_gib)
    if key is not None:
        return replace(devices[key], device_name=device_name, source="db")
    known = sorted(devices.values(), key=lambda s: s.mem_bw_gbs)
    mid = known[len(known) // 2]
    return GPUSpec(
        key=f"unknown:{device_name}",
        vram_gib=total_gib if total_gib else mid.vram_gib,
        mem_bw_gbs=mid.mem_bw_gbs,
        fp32_tflops=mid.fp32_tflops,
        notes=f"not in gpu_db.json; VRAM read from the device ({device_name})",
        device_name=device_name,
        source="device",
    )


def is_cpu_target(spec: GPUSpec) -> bool:
    """Is this "device" the host CPU rather than a card?

    The same rule :func:`caustica.planner.calibration.find_calibration_for`
    matches the ``"cpu"`` calibration entry by: the key's primary token, or
    the live device name, is literally ``cpu``. Nothing in ``gpu_db.json``
    can collide with it -- every row there is a vendor product name.

    It exists because one number in this module is CUDA-specific and must
    not follow a plan onto a CPU: see ``model.GPU_WARMUP_S`` in
    :func:`estimate`.
    """
    key_token = _norm_device(spec.key.split("-")[0])
    name = _norm_device(spec.device_name) if spec.device_name else ""
    return key_token == "cpu" or name == "cpu"


def _resolve_gpu(name: str | GPUSpec) -> GPUSpec:
    if isinstance(name, GPUSpec):
        return name
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
    gpu: str | GPUSpec = "A100",
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

    # The one-time cost a run pays before its per-step clock means anything.
    # GPU_WARMUP_S is a MEASURED CUDA constant -- context creation, cuFFT
    # plans, kernel compilation -- and a numpy run pays none of it, so on a
    # cpu target the honest default is zero rather than a conservative
    # guess. (Inherited, it made the analytic suite's planner table predict
    # 6 s for a pair of 1-D solves that cost 0.03 s: 3 s of pure fiction per
    # solve, 2026-08-23.) A measured or a warmup-carrying calibrated entry
    # overrides this either way; it is only ever the fallback.
    warmup_default = 0.0 if is_cpu_target(gpu_spec) else model.GPU_WARMUP_S

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
        entry = find_calibration_for(
            gpu_spec.key, calibration_path, device_name=gpu_spec.device_name
        )
        if entry is not None:
            # Interpolation through the measured sizes when the entry carries
            # them; the global (a, b) law only for stores too old to.
            t_step = predict_step_time(entry, p_elems)
            # Entries written before fix A2 carry no warmup at all; a GPU
            # entry without one gets the constant rather than a silent zero,
            # which is the term the whole fix exists to stop dropping. A cpu
            # entry gets zero, which is what a cpu actually pays.
            warmup_s = calibrated_warmup(entry, p_elems, default=warmup_default)
            src_label = "calibrated"
            # A "calibrated" label is a promise about accuracy; these two
            # warnings are where the label stops being able to keep it.
            p_probe = entry.get("p_max_probe")
            if p_probe and p_elems > EXTRAPOLATION_WARN_FACTOR * float(p_probe):
                warnings.append(
                    f"this plan is {p_elems / float(p_probe):.0f}x larger than the biggest "
                    f"grid the calibration measured ({int(p_probe):,} elements) — the step "
                    f"cost is extrapolated, not measured; recalibrate nearer this size"
                )
            # Grade whatever actually predicted: an interpolator's residual
            # against its own knots is 0 by construction, so the number that
            # can lapse is its leave-one-out error.
            mode, worst = time_model_quality(entry)
            if worst is not None and float(worst) > FIT_RESIDUAL_LIMIT:
                if mode == "interpolated":
                    warnings.append(
                        f"the calibration's time curve misses a HELD-OUT probe by up to "
                        f"{100 * float(worst):.0f}% (leave-one-out) — the probed sizes are "
                        f"too far apart to interpolate between; recalibrate with more grid "
                        f"sizes, nearer this one"
                    )
                else:
                    warnings.append(
                        f"the calibration's time model misses its own samples by up to "
                        f"{100 * float(worst):.0f}% — recalibrate (the probe grids were "
                        f"probably too small to saturate this device)"
                    )
        else:
            a, b = model.db_time_coeffs(
                gpu_spec.fp32_tflops, gpu_spec.mem_bw_gbs, grid.ndim, nonlinear
            )
            t_step = model.step_time(a, b, p_elems)
            warmup_s = warmup_default
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
