"""On-device calibration for the M8 planner (~20 timed engine steps).

:func:`measure_step_time` replays ``run_cw_kspace_pstd``'s step composition
op-for-op with synthetic data (timing/memory only, no physics claim), so the
measured per-step cost pays exactly the FFT mix and elementwise passes of a
real solve. :func:`calibrate` fits ``t_step = a*P*log2(P) + b*P`` over >= 2
grid sizes and persists the coefficients to ``~/.caustica/calibration.json``
keyed by device name; estimates targeting a matching device are then labeled
``"calibrated"`` (the M8 ±25% gate applies to this path, on-device).

Works on the numpy backend too (device key ``"cpu"``) — that is how the
mechanics are tested locally before the Colab session.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np

from caustica.core.backend import get_backend
from caustica.planner.model import fft_sizes
from caustica.solvers.kspace import operators as ops


def default_calibration_path() -> Path:
    return Path.home() / ".caustica" / "calibration.json"


def device_name(backend: str = "auto") -> str:
    """Stable key for the executing device: GPU product name, or ``"cpu"``."""
    b = get_backend(backend)
    if b.name == "cupy":
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        return props["name"].decode()
    return "cpu"


def measure_step_time(
    active_shape: tuple[int, ...],
    nonlinear: bool = False,
    backend: str = "auto",
    n_steps: int = 20,
    warmup: int = 3,
) -> dict:
    """Median per-step wall time (and mempool footprint) on THIS device.

    Returns ``{"device", "backend", "shape", "p_elems", "t_step_s",
    "warmup_s", "vram_peak_bytes"}`` — ``vram_peak_bytes`` is the cupy
    mempool high-water mark (``total_bytes``, i.e. what nvidia-smi attributes
    to the process) and ``None`` on the numpy backend.

    ``warmup_s`` (fix A2) is the ONE-TIME cost this measurement pays before
    the per-step clock is meaningful: device allocation, CUDA context and
    module load on a cold process, cuFFT plan creation and kernel compilation
    for this op mix. It is measured as *everything up to the end of the
    warmup iterations, minus what those iterations should have cost at the
    steady rate* — so a second call in the same process honestly reports
    almost nothing, and the first one carries the cold start. Without this
    term, ``elapsed / steps`` on a short GPU run reads ~26x the steady cost
    and the planner looks broken when it is merely incomplete (measured,
    first Colab session 2026-08-22).
    """
    t_cold0 = time.perf_counter()
    b = get_backend(backend)
    xp, fft = b.xp, b.fft
    padded, p_elems, _ = fft_sizes(tuple(active_shape))
    nd = len(padded)

    rng = np.random.default_rng(0)
    p = xp.asarray(rng.standard_normal(padded, dtype=np.float32))
    u = [xp.zeros(padded, dtype=xp.float32) for _ in range(nd)]
    # Sub-unity factors keep the synthetic state bounded over the timed loop
    # (all-ones would amplify p to float32 overflow within ~10 steps).
    dt_over_rho = xp.full(padded, 1e-3, dtype=xp.float32)
    rhoc2_dt = xp.full(padded, 1e-3, dtype=xp.float32)
    absorb = xp.full(padded, 0.999, dtype=xp.float32)
    sponge = xp.full(padded, 0.999, dtype=xp.float32)
    beta2_dt = xp.full(padded, 1e-3, dtype=xp.float32) if nonlinear else None
    ks = ops.k_vectors(padded, 1e-3, xp)
    kappa = ops.kappa_sinc(ks, c_ref=1500.0, dt=1e-7, xp=xp)
    deriv = ops.spectral_derivative_factors(ks, kappa, xp)
    del ks, kappa

    pool = None
    if b.name == "cupy":
        import cupy as cp

        pool = cp.get_default_memory_pool()

    def sync() -> None:
        if b.name == "cupy":
            import cupy as cp

            cp.cuda.get_current_stream().synchronize()

    def one_step() -> None:
        # Mirror of engine.step() — keep in lockstep with engine.py.
        pk = fft.rfftn(p)
        for i in range(nd):
            grad_i = fft.irfftn(deriv[i] * pk, s=padded)
            u[i] -= dt_over_rho * grad_i
            u[i] *= absorb
            u[i] *= sponge
        acc = None
        for i in range(nd):
            term = deriv[i] * fft.rfftn(u[i])
            acc = term if acc is None else acc + term
        divu = fft.irfftn(acc, s=padded)
        p_local = p
        if beta2_dt is None:
            p_local -= rhoc2_dt * divu
        else:
            p_local -= (rhoc2_dt + beta2_dt * p_local) * divu
        p_local *= absorb
        p_local *= sponge

    for _ in range(warmup):
        one_step()
    sync()
    t_cold = time.perf_counter() - t_cold0
    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        one_step()
        sync()
        times.append(time.perf_counter() - t0)

    t_step = float(median(times))
    return {
        "device": device_name(backend),
        "backend": b.name,
        "shape": tuple(active_shape),
        "p_elems": p_elems,
        "t_step_s": t_step,
        "warmup_s": max(0.0, t_cold - warmup * t_step),
        "vram_peak_bytes": int(pool.total_bytes()) if pool is not None else None,
    }


def fit_time_model(samples: list[tuple[int, float]]) -> tuple[float, float]:
    """Nonnegative least-squares ``(a, b)`` for ``t = a*P*log2(P) + b*P``."""
    p_arr = np.array([s[0] for s in samples], dtype=np.float64)
    t_arr = np.array([s[1] for s in samples], dtype=np.float64)
    design = np.stack([p_arr * np.log2(p_arr), p_arr], axis=1)
    coef, *_ = np.linalg.lstsq(design, t_arr, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    if a < 0.0:
        a = 0.0
        b = float(np.linalg.lstsq(design[:, 1:], t_arr, rcond=None)[0][0])
    elif b < 0.0:
        b = 0.0
        a = float(np.linalg.lstsq(design[:, :1], t_arr, rcond=None)[0][0])
    return max(a, 0.0), max(b, 0.0)


def calibrate(
    shapes: tuple[tuple[int, ...], ...] = ((48, 48, 48), (72, 72, 72)),
    *,
    nonlinear: bool = False,
    backend: str = "auto",
    n_steps: int = 20,
    path: str | Path | None = None,
) -> dict:
    """Measure ``shapes`` on this device, fit ``(a, b)``, persist, return entry.

    The entry lands in ``calibration.json`` under the device name;
    :func:`caustica.planner.estimate` picks it up automatically when the
    ``gpu=`` argument matches the device.
    """
    if len(shapes) < 2:
        raise ValueError("calibration needs >= 2 grid sizes to fit a*N*log2(N) + b*N")
    runs = [
        measure_step_time(s, nonlinear=nonlinear, backend=backend, n_steps=n_steps) for s in shapes
    ]
    samples = [(r["p_elems"], r["t_step_s"]) for r in runs]
    a, b = fit_time_model(samples)
    entry = {
        "a": a,
        "b": b,
        # The COLD start, i.e. the largest warmup any of the shapes paid: the
        # first measurement carries the context/module load, the ones after it
        # in the same process do not, and it is the first that a fresh solve
        # looks like (fix A2). record_warmup() replaces this with what a real
        # run actually paid once the validation suite has measured one.
        "warmup_s": max(float(r["warmup_s"]) for r in runs),
        "warmup_source": "probe",
        "backend": runs[0]["backend"],
        "n_steps": n_steps,
        "nonlinear": nonlinear,
        "samples": [[int(p), float(t)] for p, t in samples],
        "shapes": ["x".join(map(str, r["shape"])) for r in runs],
        "vram_peak_bytes_by_shape": {
            "x".join(map(str, r["shape"])): r["vram_peak_bytes"] for r in runs
        },
        "calibrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    target = Path(path) if path is not None else default_calibration_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(target.read_text()) if target.exists() else {"version": 1, "devices": {}}
    data["devices"][runs[0]["device"]] = entry
    target.write_text(json.dumps(data, indent=2))
    return entry


def record_warmup(
    device: str,
    warmup_s: float,
    *,
    source: str = "measured",
    path: str | Path | None = None,
) -> dict | None:
    """Teach the calibration what a REAL run's warmup cost on this device.

    The probe in :func:`measure_step_time` replays the step composition, not
    a whole solve: it never builds the medium's property maps, never touches
    the source scatter, and so it under-counts the one-time cost of a real
    run. ``caustica.validation``'s GPU-gate suite measures that number from
    an actual stamped run and calls this to write it back — which is what
    "the warmup is measured on the device and stored in the calibration"
    means in practice (fix A2).

    Returns the updated entry, or None when the device has no calibration
    yet (there is nothing to attach to, and inventing one would let the
    planner label a datasheet guess "calibrated").
    """
    target = Path(path) if path is not None else default_calibration_path()
    data = load_calibration(target)
    entry = data.get("devices", {}).get(device)
    if entry is None:
        return None
    entry["warmup_s"] = max(0.0, float(warmup_s))
    entry["warmup_source"] = source
    entry["warmup_recorded_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2))
    return entry


def load_calibration(path: str | Path | None = None) -> dict:
    target = Path(path) if path is not None else default_calibration_path()
    if not target.exists():
        return {"version": 1, "devices": {}}
    return json.loads(target.read_text())


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_calibration_for(gpu_key: str, path: str | Path | None = None) -> dict | None:
    """Best calibration entry whose device name matches a gpu_db key.

    Matching is token-based: the key's primary token (e.g. ``a100`` of
    ``A100-40GB``) must appear in the normalized device name
    (``NVIDIA A100-SXM4-40GB``); further tokens break ties. ``cpu`` only
    matches the cpu entry.
    """
    devices = load_calibration(path).get("devices", {})
    tokens = [_norm(t) for t in gpu_key.split("-") if t]
    if not tokens:
        return None
    best: tuple[int, dict] | None = None
    for name, entry in devices.items():
        name_n = _norm(name)
        if tokens[0] == "cpu":
            if name_n != "cpu":
                continue
            hits = 1
        else:
            if name_n == "cpu" or tokens[0] not in name_n:
                continue
            hits = sum(1 for t in tokens if t in name_n)
        if best is None or hits > best[0]:
            best = (hits, entry)
    return best[1] if best is not None else None
