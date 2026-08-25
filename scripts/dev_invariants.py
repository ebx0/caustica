"""Ten things a dataset rests on that no gate was watching.

Every gate this library ships grades a magnitude: how close |P| is to a
closed form, to another code, to itself under refinement. That is the right
first question and it is not the only one. A dataset is a pile of complex
numbers written to disk by a run that may have been interrupted, on a grid
whose edges absorb imperfectly, from a transducer placed at some particular
spot, at some particular time step, and the number that comes back out of
the file is not the number the solver computed unless somebody checked.

Nothing here needs a closed form. Each one is a property the code must have
because of what it *is*, so each is falsifiable without an oracle:

    V1  the PHASE, not just the magnitude, and the phase speed it implies
    V2  superposition: two halves of an array, driven apart and together
    V3  the drive scales the field exactly in a linear medium, and must not
        in a nonlinear one
    V4  what comes out of result.h5 is what went in, and what the default
        quantization costs in megapascals and in radians
    V5  a resumed run is the uninterrupted run, bit for bit
    V6  the same job twice is the same numbers
    V7  where the transducer sits in the grid does not change the field
    V8  what the sponge reflects, against its thickness
    V9  the answer at fixed dx does not depend on dt
    V10 delay-and-sum lands where it is aimed, over a volume of targets

Run it::

    python scripts/dev_invariants.py --out benchmarks/reports/invariants
    python scripts/dev_invariants.py --only V1,V4

V5 and V6 want a GPU to be interesting: determinism and resume are trivially
true on one CPU thread and are exactly what a GPU reduction can break.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

MM = 1e-3
F0 = 1.0e6
C_W, RHO_W = 1500.0, 1000.0
DRIVE = 1.0e5
APERTURE, ROC = 10.0 * MM, 25.0 * MM
PML_MM = 5.0

CHECKS: list[tuple[str, str, Callable]] = []


def check(cid: str, title: str):
    def wrap(fn):
        CHECKS.append((cid, title, fn))
        return fn

    return wrap


# --------------------------------------------------------------------------
# shared scene
# --------------------------------------------------------------------------


def scene(
    ppw: float,
    *,
    beta: float = 0.0,
    apex_shift=(0, 0, 0),
    aperture: float = APERTURE,
    roc: float = ROC,
):
    """A focused source in water, on an FFT-friendly grid sized to hold it.

    ``aperture`` and ``roc`` are parameters and not constants because the
    array checks use a transducer twice the width and twice the focal length
    of the bowl the others use. Sizing every scene from the bowl was the
    first version of this file and it put the array's focus in the sponge
    and 2.6 % of its drive outside the domain, which showed up as a 2.5 %
    superposition error that had nothing to do with superposition.

    ``apex_shift`` moves the transducer inside the domain without changing
    the domain, which is what V7 needs and what nothing else may notice.
    """
    from caustica.core.grid import Grid
    from caustica.core.pml import PMLSpec
    from caustica.medium import Medium
    from caustica.solvers.kspace.operators import optimal_fft_size

    dx = C_W / F0 / ppw
    pml_vox = int(round(PML_MM * MM / dx))
    margin = 6
    n_xy = 2 * (int(np.ceil(aperture / dx)) + pml_vox + margin) + 1
    apex_z = pml_vox + margin
    n_z = apex_z + int(round(1.4 * roc / dx)) + pml_vox + margin
    shape = tuple(optimal_fft_size(int(v)) for v in (n_xy, n_xy, n_z))
    grid = Grid(shape=shape, dx=dx, pml=PMLSpec(thickness=PML_MM * MM))
    apex = (
        shape[0] // 2 + apex_shift[0],
        shape[1] // 2 + apex_shift[1],
        apex_z + apex_shift[2],
    )
    ones = np.ones(shape, np.float32)
    medium = Medium(
        alpha=np.zeros(shape, np.float32),
        rho=ones * RHO_W,
        c=ones * C_W,
        beta=ones * float(beta),
    )
    focus = (apex[0], apex[1], apex[2] + int(round(roc / dx)))
    return grid, medium, apex, focus, dx


#: The array the two steering checks drive, and the scene that holds it.
ARRAY_N, ARRAY_D, ARRAY_HOLE, ARRAY_ROC = 32, 40.0 * MM, 16.0 * MM, 40.0 * MM


def array_scene(ppw: float, **kw):
    from caustica.arrays.transducer import archimedean_spiral

    array = archimedean_spiral(
        n_elements=ARRAY_N, d_outer=ARRAY_D, d_inner=ARRAY_HOLE, roc=ARRAY_ROC
    )
    return array, scene(ppw, aperture=ARRAY_D / 2.0, roc=ARRAY_ROC, **kw)


def bowl(grid, apex, amplitude=DRIVE):
    from caustica.sources import bowl_cw_source

    return bowl_cw_source(grid, F0, amplitude, APERTURE, ROC, apex)


def solve(grid, medium, source, focus, *, ctx, solver="linear", spec=None, **kw):
    from caustica.solvers import CWRunSpec, get

    spec = spec or CWRunSpec(min_settle_periods=12, max_settle_periods=60, n_record_periods=2)
    return get(solver)().run(
        grid, medium, source, spec, backend=ctx["backend"], reference_point=focus, **kw
    )


def release_gpu_pool() -> None:
    try:
        import cupy
    except Exception:
        return
    try:
        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def axis_of(field, apex, grid, dx):
    """On-axis samples clear of the cap and of the sponge."""
    z = (np.arange(grid.shape[2]) - apex[2]) * dx
    pml_vox = int(round(PML_MM * MM / dx))
    sel = (z > 0.35 * ROC) & (z < (grid.shape[2] - pml_vox - 3 - apex[2]) * dx)
    return z[sel], field[apex[0], apex[1], :][sel]


def plane_wave_speed(ppw: float) -> float:
    """The speed the scheme actually propagates a plane wave at, from its phase.

    Deliberately NOT measured on the focused beam next to it. A converging
    beam's on-axis phase carries the geometry of the convergence and the
    Gouy shift through the focus, so its axial phase gradient is not the
    wavenumber, and reading one off it reports a dispersion error the scheme
    does not have: the first version of this check called a focused bowl
    4.6 % dispersive. A plane wave in a uniform medium has phase ``-k z`` and
    nothing else, which is what makes the slope mean what it says.
    """
    if ppw in plane_wave_speed.cache:
        return plane_wave_speed.cache[ppw]
    from caustica.core.grid import Grid
    from caustica.core.pml import PMLSpec
    from caustica.medium import Medium
    from caustica.solvers import CWRunSpec, get
    from caustica.sources import plane_cw_source

    dx = C_W / F0 / ppw
    pml_vox = int(round(PML_MM * MM / dx))
    n = pml_vox + int(round(40.0 * MM / dx)) + pml_vox
    grid = Grid(shape=(n,), dx=dx, pml=PMLSpec(thickness=PML_MM * MM))
    ones = np.ones(n, np.float32)
    medium = Medium(alpha=np.zeros(n, np.float32), rho=ones * RHO_W, c=ones * C_W, beta=ones * 0.0)
    src = plane_cw_source(grid, f0=F0, amplitude=DRIVE, axis=0, position_vox=pml_vox + 4)
    res = get("linear")().run(
        grid,
        medium,
        src,
        CWRunSpec(min_settle_periods=40, max_settle_periods=200),
        backend="numpy",
    )
    p = np.asarray(res.phasor)
    lam = C_W / F0
    lo = pml_vox + int(round(4 * lam / dx))
    hi = n - pml_vox - int(round(4 * lam / dx))
    x = np.arange(lo, hi) * dx
    slope = np.polyfit(x, np.unwrap(np.angle(p[lo:hi])), 1)[0]
    c_num = 2.0 * np.pi * F0 / abs(slope)
    plane_wave_speed.cache[ppw] = c_num
    return c_num


plane_wave_speed.cache = {}

# --------------------------------------------------------------------------
# V1 — the phase
# --------------------------------------------------------------------------


@check("V1", "the phase, not just the magnitude, and the phase speed it implies")
def _v1(ctx):
    """Every shipped gate grades ``|P|``. The dataset records ``P``.

    A phasor whose magnitude is right and whose argument is wrong reproduces
    every number in every report and is useless for anything that
    superposes, steers, or reconstructs a waveform. Two claims here, and the
    second is the sharper one.

    The first: the argument matches the Rayleigh integral's, once the
    arbitrary global offset is removed. Both codes pick their own time
    origin, so only the relative phase is meaningful, and that is exactly
    what is being asked for.

    The second: the slope of the unwrapped phase along the axis IS the
    numerical wavenumber, so it measures the phase speed the scheme actually
    propagates at. A pseudospectral method should be within a fraction of a
    percent of c; a scheme with a dispersion error shows it here at a
    resolution where the magnitude still looks perfect.
    """
    from caustica.analytic import rayleigh_pressure
    from caustica.analytic.geometry import spherical_cap_points

    rows = []
    plane_wave_speed.cache.clear()
    for ppw in ctx["ladder"]:
        grid, medium, apex, focus, dx = scene(ppw)
        src = bowl(grid, apex)
        res = solve(grid, medium, src, focus, ctx=ctx)
        z, p_sim = axis_of(np.asarray(res.phasor), apex, grid, dx)
        pts, _normals, areas = spherical_cap_points(APERTURE, ROC, dx / 2.0)
        field = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
        p_ref = rayleigh_pressure(
            pts,
            areas,
            np.full(len(pts), DRIVE / (RHO_W * C_W), dtype=complex),
            field,
            k=2.0 * np.pi * F0 / C_W,
            c=C_W,
        )
        # The two codes start their clocks where they like, so a constant
        # offset is not an error. Weight by amplitude: the phase of a null is
        # noise and would otherwise dominate an unweighted mean.
        w = np.abs(p_sim)
        d = np.angle(p_sim * np.conj(p_ref))
        offset = np.angle(np.sum(w * np.exp(1j * d)))
        err = np.angle(np.exp(1j * (d - offset)))
        rms = float(np.sqrt(np.sum(w * err**2) / np.sum(w)))
        rows.append(
            {
                "geometry": "focused bowl",
                "ppw": ppw,
                "dx_mm": dx / MM,
                "phase_rms_rad": rms,
                "phase_rms_wavelengths": rms / (2 * np.pi),
                "phase_max_rad": float(np.abs(err[w > 0.2 * w.max()]).max()),
                "numerical_c_ms": plane_wave_speed(ppw),
                "c_rel_error": abs(plane_wave_speed(ppw) - C_W) / C_W,
            }
        )
        release_gpu_pool()
    best = min(rows, key=lambda r: r["ppw"])
    fine = max(rows, key=lambda r: r["ppw"])
    return {
        "rows": rows,
        "verdict": (
            f"against the Rayleigh integral the on-axis phase is within "
            f"{fine['phase_rms_rad']:.4f} rad rms at {fine['ppw']:.0f} points per wavelength "
            f"({fine['phase_rms_wavelengths'] * 100:.3f} % of a wavelength) and "
            f"{best['phase_rms_rad']:.4f} rad at {best['ppw']:.0f}; the phase speed the scheme "
            f"propagates a PLANE wave at is {fine['numerical_c_ms']:.3f} m/s against "
            f"{C_W:.0f}, {fine['c_rel_error'] * 100:.5f} % out"
        ),
    }


# --------------------------------------------------------------------------
# V2 — superposition
# --------------------------------------------------------------------------


@check("V2", "superposition: two halves of an array, driven apart and together")
def _v2(ctx):
    """A linear solver has to be linear, and the array had a bug that was not.

    Before 2026-08-24 a voxel two elements both reached kept the first one's
    phase and dropped the other's drive. That is a superposition failure, it
    was worth 1.7 % of the element pairs, and no gate would have caught it:
    the field it produces is smooth, focused, and plausible.

    So: split the array down the middle, run each half alone, run both
    together, and compare complex fields. In a linear medium the equality is
    exact up to float32, with no tolerance to negotiate.
    """
    array, (grid, medium, apex, focus, dx) = array_scene(ctx["ppw"])
    phases = array.das_phases(array.focus, F0, C_W).astype(np.float64)

    def half(mask):
        sub = replace(array, positions=array.positions[mask], normals=array.normals[mask])
        asrc = sub.voxelize(
            grid, apex, f0=F0, amplitude=DRIVE, phases=phases[mask].astype(np.float32)
        )
        res = solve(grid, medium, asrc.source, focus, ctx=ctx)
        release_gpu_pool()
        return np.asarray(res.phasor, dtype=np.complex128)

    left = array.positions[:, 0] < 0
    a, b = half(left), half(~left)
    whole_src = array.voxelize(grid, apex, f0=F0, amplitude=DRIVE, phases=phases.astype(np.float32))
    both = np.asarray(solve(grid, medium, whole_src.source, focus, ctx=ctx).phasor, np.complex128)
    release_gpu_pool()

    total = a + b
    scale = float(np.abs(both).max())
    err = np.abs(both - total)
    return {
        "elements": int(array.n_elements),
        "grid": "x".join(str(int(v)) for v in grid.shape),
        "rows": [
            {"driven": "left half alone", "peak_mpa": float(np.abs(a).max()) / 1e6},
            {"driven": "right half alone", "peak_mpa": float(np.abs(b).max()) / 1e6},
            {"driven": "sum of the two", "peak_mpa": float(np.abs(total).max()) / 1e6},
            {"driven": "both together", "peak_mpa": scale / 1e6},
        ],
        "max_abs_error_pa": float(err.max()),
        "max_rel_error": float(err.max() / scale),
        "rms_rel_error": float(np.sqrt(np.mean(err**2)) / scale),
        "verdict": (
            f"driving the halves apart and adding the complex fields reproduces the field of "
            f"driving them together to {err.max() / scale:.2e} of the peak "
            f"({np.sqrt(np.mean(err**2)) / scale:.2e} rms) over {int(both.size)} voxels"
        ),
    }


# --------------------------------------------------------------------------
# V3 — the drive scales the field
# --------------------------------------------------------------------------


@check("V3", "the drive scales the field exactly, and stops doing so when beta is on")
def _v3(ctx):
    """Two runs, one number, and it has to be exactly two.

    A linear medium cannot know the amplitude it is driven at: doubling the
    drive doubles every phasor and moves nothing else. Any normalization
    that quietly depends on the drive, any saturation, any clipping shows up
    here as a ratio that is not 2.

    The same measurement in a nonlinear medium must FAIL to be 2, and by a
    known amount, which is what makes the first half a test rather than a
    tautology about float multiplication.
    """
    rows = []
    for beta in (0.0, 3.5):
        grid, medium, apex, focus, dx = scene(ctx["ppw"], beta=beta)
        solver = "linear" if beta == 0.0 else "westervelt"
        fields = {}
        for factor in (1.0, 2.0):
            res = solve(
                grid,
                medium,
                bowl(grid, apex, DRIVE * factor),
                focus,
                ctx=ctx,
                solver=solver,
                **({"harmonics": (1, 2)} if beta else {}),
            )
            fields[factor] = np.asarray(res.phasor, np.complex128)
            release_gpu_pool()
        ratio = np.abs(fields[2.0]) / np.maximum(np.abs(fields[1.0]), 1e-12)
        strong = np.abs(fields[1.0]) > 0.05 * np.abs(fields[1.0]).max()
        rows.append(
            {
                "beta": beta,
                "solver": solver,
                "peak_1x_mpa": float(np.abs(fields[1.0]).max()) / 1e6,
                "peak_2x_mpa": float(np.abs(fields[2.0]).max()) / 1e6,
                "ratio_at_peak": float(np.abs(fields[2.0]).max() / np.abs(fields[1.0]).max()),
                "ratio_median": float(np.median(ratio[strong])),
                "worst_departure_from_2": float(np.abs(ratio[strong] - 2.0).max()),
            }
        )
    lin, non = rows[0], rows[1]
    return {
        "rows": rows,
        "verdict": (
            f"in water the field scales by {lin['ratio_at_peak']:.9f} when the drive doubles, "
            f"worst departure {lin['worst_departure_from_2']:.2e} over the illuminated volume; "
            f"with beta = 3.5 the same doubling gives {non['ratio_at_peak']:.5f}, which is the "
            f"nonlinearity and is why the first number is a measurement"
        ),
    }


# --------------------------------------------------------------------------
# V4 — the file is the field
# --------------------------------------------------------------------------


@check("V4", "what comes out of result.h5, and what the default quantization costs")
def _v4(ctx):
    """The dataset is the file, not the array the solver returned.

    ``save_result`` quantizes by default, to a normalized error budget of
    1e-3. That is a deliberate space-for-accuracy trade and it is invisible
    from inside a run: the numbers in the report are the solver's, the
    numbers in the dataset are the file's. So both are measured here, in
    megapascals and in radians, and the unquantized path is measured beside
    them to show the round-trip itself loses nothing.
    """
    from caustica.io.store import load_result, save_result

    grid, medium, apex, focus, dx = scene(ctx["ppw"], beta=3.5)
    src = bowl(grid, apex)
    res = solve(grid, medium, src, focus, ctx=ctx, solver="westervelt", harmonics=(1, 2))
    truth = {h: np.asarray(v, np.complex128) for h, v in res.phasors.items()}
    peak = float(np.abs(truth[1]).max())

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for quantize in (True, False):
            path = Path(tmp) / f"q{int(quantize)}.h5"
            save_result(
                path,
                res,
                src,
                dx=grid.dx,
                grid_shape=grid.shape,
                pml_vox=grid.pml_vox,
                quantize=quantize,
            )
            back = load_result(path)
            row: dict[str, Any] = {
                "quantize": quantize,
                "file_mb": path.stat().st_size / 1e6,
                "harmonics_present": sorted(back.phasors),
            }
            for h in sorted(truth):
                got = np.asarray(back.phasors[h], np.complex128)
                amp = np.abs(truth[h])
                strong = amp > 0.05 * amp.max()
                row[f"h{h}_max_abs_err_kpa"] = float(np.abs(got - truth[h]).max()) / 1e3
                row[f"h{h}_max_rel_err"] = float(np.abs(got - truth[h]).max() / amp.max())
                row[f"h{h}_phase_rms_rad"] = float(
                    np.sqrt(np.mean(np.angle(got[strong] * np.conj(truth[h][strong])) ** 2))
                )
            row["stamps"] = {
                k: back.meta.get(k)
                for k in ("numerics_scheme", "source_discretization", "solver")
                if k in back.meta
            }
            rows.append(row)
            release_gpu_pool()

    q, raw = rows[0], rows[1]
    return {
        "peak_mpa": peak / 1e6,
        "rows": rows,
        "verdict": (
            f"the unquantized round-trip returns the field unchanged "
            f"(worst {raw['h1_max_rel_err']:.1e} of the peak); the DEFAULT quantized write "
            f"costs {q['h1_max_abs_err_kpa']:.2f} kPa on f0 and "
            f"{q['h2_max_abs_err_kpa']:.2f} kPa on 2f0, with {q['h1_phase_rms_rad']:.2e} rad of "
            f"phase, for {raw['file_mb'] / q['file_mb']:.1f}x less disk"
        ),
    }


# --------------------------------------------------------------------------
# V5 — resume
# --------------------------------------------------------------------------


@check("V5", "a resumed run is the uninterrupted run, bit for bit")
def _v5(ctx):
    """A 24-hour Colab run that cannot be resumed is a 24-hour gamble.

    Resume is not "close enough": the checkpoint carries the exact state, so
    the field that comes out of an interrupted-and-continued run has to be
    the identical array, not a similar one. Anything less means a dataset
    whose entries differ by whether the session dropped.
    """
    from caustica.io.checkpoint import CheckpointSpec, RunInterrupted
    from caustica.solvers import CWRunSpec, get

    grid, medium, apex, focus, dx = scene(ctx["ppw"], beta=3.5)
    src = bowl(grid, apex)
    spec = CWRunSpec(min_settle_periods=12, max_settle_periods=40, n_record_periods=2)

    def run(checkpoint=None):
        return get("westervelt")().run(
            grid,
            medium,
            src,
            spec,
            backend=ctx["backend"],
            reference_point=focus,
            harmonics=(1, 2),
            checkpoint=checkpoint,
        )

    straight = run()
    release_gpu_pool()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "resume.ckpt.npz"
        seen = {"n": 0}

        def stop_after_three() -> bool:
            seen["n"] += 1
            return seen["n"] >= 3

        interrupted = False
        try:
            run(CheckpointSpec(path=path, every_periods=2, stop_when=stop_after_three))
        except RunInterrupted:
            interrupted = True
        if not interrupted:
            return {"verdict": "the run finished before the stop fired; nothing was resumed"}
        resumed = run(CheckpointSpec(path=path))
    release_gpu_pool()

    same = {
        h: bool(np.array_equal(straight.phasors[h], resumed.phasors[h])) for h in straight.phasors
    }
    worst = max(
        float(np.abs(np.asarray(resumed.phasors[h]) - np.asarray(straight.phasors[h])).max())
        for h in straight.phasors
    )
    return {
        "rows": [
            {
                "run": "uninterrupted",
                "converged_period": straight.converged_period,
                "steps": straight.steps_total,
                "peak_mpa": float(np.abs(straight.phasor).max()) / 1e6,
            },
            {
                "run": "killed and resumed",
                "converged_period": resumed.converged_period,
                "steps": resumed.steps_total,
                "peak_mpa": float(np.abs(resumed.phasor).max()) / 1e6,
            },
        ],
        "identical_per_harmonic": same,
        "worst_abs_difference_pa": worst,
        "verdict": (
            "a resumed run reproduces the uninterrupted one bit for bit on every harmonic"
            if all(same.values())
            else f"resume differs from the straight run by up to {worst:.3e} Pa: {same}"
        ),
    }


# --------------------------------------------------------------------------
# V6 — determinism
# --------------------------------------------------------------------------


@check("V6", "the same job twice is the same numbers")
def _v6(ctx):
    """Reproducibility is a claim about the machine, not about the code.

    Atomics and reduction orders on a GPU are free to vary between launches,
    and a library that stamps a commit into every result is promising that
    the commit determines the answer. Two identical runs, compared exactly.
    """
    grid, medium, apex, focus, dx = scene(ctx["ppw"], beta=3.5)
    src = bowl(grid, apex)
    out = []
    for _ in range(2):
        res = solve(grid, medium, src, focus, ctx=ctx, solver="westervelt", harmonics=(1, 2))
        out.append({h: np.asarray(v).copy() for h, v in res.phasors.items()})
        release_gpu_pool()
    same = {h: bool(np.array_equal(out[0][h], out[1][h])) for h in out[0]}
    worst = max(float(np.abs(out[0][h] - out[1][h]).max()) for h in out[0])
    peak = float(np.abs(out[0][1]).max())
    return {
        "backend": ctx["backend"],
        "grid": "x".join(str(int(v)) for v in grid.shape),
        "identical_per_harmonic": same,
        "worst_abs_difference_pa": worst,
        "worst_rel_difference": worst / peak,
        "verdict": (
            f"two identical runs on {ctx['backend']} return identical arrays on every harmonic"
            if all(same.values())
            else f"two identical runs differ by up to {worst:.3e} Pa "
            f"({worst / peak:.2e} of the peak): {same}"
        ),
    }


# --------------------------------------------------------------------------
# V7 — placement invariance
# --------------------------------------------------------------------------


@check("V7", "where the transducer sits in the grid does not change the field")
def _v7(ctx):
    """Physics does not know about array indices.

    Move the transducer a few voxels and the field must move with it,
    unchanged. What this catches is anything anchored to the grid rather
    than to the geometry: an off-by-one in the source deposition, a sponge
    profile that is not symmetric, an FFT pad that leaks. The shift is
    deliberately not a multiple of anything.
    """
    shifts = [(0, 0, 0), (3, 0, 0), (0, 5, 0), (2, 3, 4)]
    grid0, medium, apex0, focus0, dx = scene(ctx["ppw"])
    base = None
    rows = []
    for sh in shifts:
        grid, medium_s, apex, focus, _ = scene(ctx["ppw"], apex_shift=sh)
        res = solve(grid, medium_s, bowl(grid, apex), focus, ctx=ctx)
        amp = np.abs(np.asarray(res.phasor))
        # Compare on the window that both runs cover, aligned on the apex.
        # Only the interior: the sponge sits at fixed grid positions, so a
        # source that moved is a different distance from it and the field
        # inside the absorbing band legitimately differs. Comparing there
        # would be grading the boundary, not the invariance.
        guard = int(round(PML_MM * MM / dx)) + 4
        lo = [max(guard, guard + s) for s in sh]
        hi = [n - guard + min(0, s) for n, s in zip(grid.shape, sh, strict=True)]
        win = tuple(slice(lo[d], hi[d]) for d in range(3))
        ref_win = tuple(slice(lo[d] - sh[d], hi[d] - sh[d]) for d in range(3))
        row = {
            "shift_vox": str(sh),
            "peak_mpa": float(amp.max()) / 1e6,
            "peak_index": str(
                tuple(int(v) for v in np.unravel_index(int(amp.argmax()), amp.shape))
            ),
        }
        if base is None:
            base = amp
        else:
            a, b = amp[win], base[ref_win]
            row["max_rel_difference"] = float(np.abs(a - b).max() / base.max())
            row["peak_rel_difference"] = float(abs(a.max() - b.max()) / b.max())
        rows.append(row)
        release_gpu_pool()
    moved = [r for r in rows if "max_rel_difference" in r]
    return {
        "rows": rows,
        "verdict": (
            f"moving the transducer by up to {max(sum(abs(v) for v in s) for s in shifts[1:])} "
            f"voxels leaves the focal peak within "
            f"{max(r['peak_rel_difference'] for r in moved):.1e} of itself; across the interior "
            f"the field differs by up to {max(r['max_rel_difference'] for r in moved):.1e} of "
            f"the peak, which is the domain edges sitting at a different distance and not the "
            f"source being placed differently"
        ),
    }


# --------------------------------------------------------------------------
# V8 — what the sponge reflects
# --------------------------------------------------------------------------


@check("V8", "what the sponge reflects, against its thickness")
def _v8(ctx):
    """The absorbing boundary is the one part of the domain with no physics.

    Whatever it sends back adds to every field in the library, and nothing
    has ever measured it. A plane wave into a sponge with nothing behind it
    makes a standing wave whose ratio IS the reflection coefficient, so the
    measurement needs no reference at all.

    Reported in dB as well as in percent, because that is the unit a
    boundary condition is usually quoted in and it makes the thickness trade
    legible: each extra wavelength of sponge should buy a fixed number of dB
    until it stops.
    """
    from caustica.core.grid import Grid
    from caustica.core.pml import PMLSpec
    from caustica.medium import Medium
    from caustica.solvers import CWRunSpec, get
    from caustica.sources import plane_cw_source

    lam = C_W / F0
    dx = lam / 20.0
    rows = []
    for pml_mm in ctx["pml_ladder"]:
        pml = pml_mm * MM
        pml_vox = int(round(pml / dx))
        n = pml_vox + int(round(30.0 * MM / dx)) + pml_vox
        grid = Grid(shape=(n,), dx=dx, pml=PMLSpec(thickness=pml))
        ones = np.ones(n, np.float32)
        medium = Medium(
            alpha=np.zeros(n, np.float32), rho=ones * RHO_W, c=ones * C_W, beta=ones * 0.0
        )
        src = plane_cw_source(grid, f0=F0, amplitude=DRIVE, axis=0, position_vox=pml_vox + 4)
        res = get("linear")().run(
            grid,
            medium,
            src,
            CWRunSpec(min_settle_periods=40, max_settle_periods=200),
            backend="numpy",
        )
        amp = np.abs(np.asarray(res.phasor))
        # Between the source and the far sponge, clear of both.
        win = slice(pml_vox + int(round(3 * lam / dx)), n - pml_vox - int(round(2 * lam / dx)))
        a = amp[win]
        swr = float(a.max() / a.min())
        r = (swr - 1.0) / (swr + 1.0)
        rows.append(
            {
                "pml_mm": pml_mm,
                "pml_vox": pml_vox,
                "pml_wavelengths": pml / lam,
                "swr": swr,
                "reflection": r,
                "reflection_db": 20.0 * np.log10(max(r, 1e-12)),
            }
        )
    best = min(rows, key=lambda r: r["reflection"])
    worst = max(rows, key=lambda r: r["reflection"])
    return {
        "rows": rows,
        "verdict": (
            f"the sponge reflects {worst['reflection'] * 100:.2f} % "
            f"({worst['reflection_db']:.1f} dB) at {worst['pml_wavelengths']:.1f} wavelengths "
            f"thick and {best['reflection'] * 100:.2f} % ({best['reflection_db']:.1f} dB) at "
            f"{best['pml_wavelengths']:.1f}"
        ),
    }


# --------------------------------------------------------------------------
# V9 — the time step
# --------------------------------------------------------------------------


@check("V9", "the answer at fixed dx does not depend on dt")
def _v9(ctx):
    """Space and time are refined together everywhere else, which hides this.

    Every convergence study in this repo moves dx, and dt follows it through
    the CFL. So a time-discretization error would ride along looking like a
    space one. Holding dx fixed and shortening dt alone separates them: what
    is left over is the scheme's temporal error, and for a dataset recorded
    at f0 and 2f0 it is worth knowing which of the two is the binding
    constraint.
    """
    from caustica.solvers import CWRunSpec

    grid, medium, apex, focus, dx = scene(ctx["ppw"], beta=3.5)
    src = bowl(grid, apex)
    rows, fields = [], {}
    for cfl in ctx["cfl_ladder"]:
        spec = CWRunSpec(
            cfl=cfl,
            cfl_hard_max=max(cfl, 0.5),
            min_settle_periods=12,
            max_settle_periods=60,
            n_record_periods=2,
        )
        res = solve(
            grid, medium, src, focus, ctx=ctx, solver="westervelt", spec=spec, harmonics=(1, 2)
        )
        fields[cfl] = {h: np.abs(np.asarray(v)).astype(np.float64) for h, v in res.phasors.items()}
        rows.append(
            {
                "cfl": cfl,
                "samples_per_period": int(res.spp),
                "dt_ns": res.dt * 1e9,
                "steps": int(res.steps_total),
                "f0_peak_mpa": float(fields[cfl][1].max()) / 1e6,
                "h2_peak_mpa": float(fields[cfl][2].max()) / 1e6,
            }
        )
        release_gpu_pool()
    ref = ctx["cfl_ladder"][-1]
    for row in rows:
        for h in (1, 2):
            row[f"h{h}_rel_to_finest_dt"] = float(
                abs(fields[row["cfl"]][h].max() - fields[ref][h].max()) / fields[ref][h].max()
            )
    coarse = rows[0]
    return {
        "grid": "x".join(str(int(v)) for v in grid.shape),
        "rows": rows,
        "verdict": (
            f"holding dx fixed and taking dt from {rows[0]['samples_per_period']} to "
            f"{rows[-1]['samples_per_period']} samples per period moves the fundamental by "
            f"{coarse['h1_rel_to_finest_dt'] * 100:.3f} % and the second harmonic by "
            f"{coarse['h2_rel_to_finest_dt'] * 100:.3f} %"
        ),
    }


# --------------------------------------------------------------------------
# V10 — steering lands where it is aimed
# --------------------------------------------------------------------------


@check("V10", "delay-and-sum lands where it is aimed, over a volume of targets")
def _v10(ctx):
    """One target proves nothing; the error is a function of where you aim.

    Steering off-axis costs a phased array both amplitude and accuracy, and
    both grow with the angle. A dataset that steers needs the shape of that
    curve, not a single number from the one target somebody happened to
    test. So: a spread of targets, and the displacement and the amplitude
    reported for each.
    """
    array, (grid, medium, apex, focus, dx) = array_scene(ctx["ppw"])
    lam = C_W / F0
    origin = np.array(apex, np.float64) * dx
    targets = [(0, 0, 0), (2, 0, 0), (4, 0, 0), (0, 3, 0), (0, 0, -4), (0, 0, 4), (3, 3, 3)]
    rows = []
    for off in targets:
        target = array.focus + np.array(off, np.float64) * MM
        phases = array.das_phases(target, F0, C_W).astype(np.float32)
        asrc = array.voxelize(grid, apex, f0=F0, amplitude=DRIVE, phases=phases)
        want_vox = np.round(origin_to_vox(target, origin, dx)).astype(int)
        ref = tuple(int(np.clip(v, 2, n - 3)) for v, n in zip(want_vox, grid.shape, strict=True))
        res = solve(grid, medium, asrc.source, ref, ctx=ctx)
        amp = np.abs(np.asarray(res.phasor))
        half = max(6, int(round(0.4 * ROC / dx)))
        box = tuple(
            slice(max(0, c - half), min(n, c + half + 1))
            for c, n in zip(ref, grid.shape, strict=True)
        )
        sub = amp[box]
        loc = np.unravel_index(int(sub.argmax()), sub.shape)
        got = np.array([b.start + o for b, o in zip(box, loc, strict=True)], np.float64)
        d = (got - want_vox) * dx
        rows.append(
            {
                "target_offset_mm": str(off),
                "steer_mm": float(np.linalg.norm(np.array(off, float))),
                "lateral_error_lambda": float(np.hypot(d[0], d[1]) / lam),
                "axial_error_lambda": float(abs(d[2]) / lam),
                "peak_mpa": float(sub.max()) / 1e6,
            }
        )
        release_gpu_pool()
    # The unsteered case already sits off its geometric focus: an annular
    # aperture pulls the peak, and that offset is the transducer's, not the
    # steering's. Subtracting it leaves the part steering is answerable for.
    base_axial = rows[0]["axial_error_lambda"]
    on_axis = rows[0]["peak_mpa"]
    for r in rows:
        r["gain_vs_unsteered"] = r["peak_mpa"] / on_axis
        r["axial_error_vs_unsteered"] = r["axial_error_lambda"] - base_axial
    return {
        "elements": int(array.n_elements),
        "rows": rows,
        "verdict": (
            f"over {len(targets)} targets spanning "
            f"{max(r['steer_mm'] for r in rows):.1f} mm of steering, the focus lands within "
            f"{max(r['lateral_error_lambda'] for r in rows):.3f} wavelengths laterally and "
            f"{max(abs(r['axial_error_vs_unsteered']) for r in rows):.3f} axially once the "
            f"array's own {base_axial:.3f}-wavelength focal offset is taken out, and the peak "
            f"falls to {min(r['gain_vs_unsteered'] for r in rows):.3f} of the unsteered one"
        ),
    }


def origin_to_vox(point_m, origin_m, dx):
    return (np.asarray(point_m, np.float64) + origin_m) / dx


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def environment() -> dict:
    import numpy

    import caustica
    from caustica.core.backend import cupy_available

    env = {
        "caustica": caustica.__version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if cupy_available():
        import cupy

        env["gpu"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    return env


def cell(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.5g}" if v == 0 or 1e-4 <= abs(v) < 1e5 else f"{v:.3e}"
    return str(v).replace("|", "/")


def flatten(value: Any, prefix: str = "") -> dict:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = value
    return out


def render_markdown(payload: dict) -> str:
    lines = [
        "# Ten things a dataset rests on",
        "",
        "Generated by `scripts/dev_invariants.py`. Nothing here is graded against a",
        "closed form: each check is a property the code must have because of what it",
        "is, so each is falsifiable without an oracle.",
        "",
        "| check | question | verdict |",
        "|---|---|---|",
    ]
    for e in payload["checks"]:
        v = (e.get("data") or {}).get("verdict", e.get("error", ""))
        lines.append(f"| [{e['id']}](#{e['id'].lower()}) | {e['title']} | {cell(v)} |")
    for e in payload["checks"]:
        data = e.get("data") or {}
        lines += ["", f"## {e['id']}", "", f"**{e['title']}**", ""]
        if e["status"] != "OK":
            lines += ["```", str(e.get("error", "")), "```"]
            continue
        lines += [str(data.get("verdict", "")), ""]
        rows = [flatten(r) for r in data.get("rows", [])]
        if rows:
            cols: list[str] = []
            for r in rows:
                for k in r:
                    if k not in cols:
                        cols.append(k)
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("|" + "---|" * len(cols))
            for r in rows:
                lines.append("| " + " | ".join(cell(r.get(c, "")) for c in cols) + " |")
    env = json.dumps(payload["environment"], indent=2)
    lines += ["", "## Environment", "", "```json", env, "```"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="benchmarks/reports/invariants")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--ppw", type=float, default=10.0, help="points per wavelength")
    ap.add_argument("--ladder", default="6,10,16", help="ppw ladder for V1")
    ap.add_argument("--pml-ladder", default="2,4,6,8,12", help="sponge thickness in mm for V8")
    ap.add_argument("--cfl-ladder", default="0.48,0.36,0.24,0.12", help="CFL ladder for V9")
    args = ap.parse_args(argv)

    from caustica.core.backend import cupy_available

    ctx = {
        "backend": "cupy" if cupy_available() else "numpy",
        "ppw": args.ppw,
        "ladder": [float(s) for s in args.ladder.split(",") if s.strip()],
        "pml_ladder": [float(s) for s in args.pml_ladder.split(",") if s.strip()],
        "cfl_ladder": [float(s) for s in args.cfl_ladder.split(",") if s.strip()],
    }
    only = {s.strip().upper() for s in args.only.split(",") if s.strip()}
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for cid, title, fn in CHECKS:
        if (only and cid not in only) or cid in skip:
            continue
        print(f"[{cid}] {title} ...", flush=True)
        t0 = time.perf_counter()
        entry: dict[str, Any] = {"id": cid, "title": title}
        try:
            entry["data"] = fn(ctx)
            entry["status"] = "OK"
        except Exception as exc:
            entry["status"] = "ERROR"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc(limit=8)
        entry["elapsed_s"] = round(time.perf_counter() - t0, 2)
        results.append(entry)
        mark = "OK " if entry["status"] == "OK" else "ERR"
        detail = (entry.get("data") or {}).get("verdict", entry.get("error", ""))
        print(f"  {mark} {entry['elapsed_s']:>8.2f}s  {detail}", flush=True)

    path = outdir / "invariants.json"
    previous = []
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8")).get("checks", [])
        except (OSError, ValueError):
            previous = []
    merged = {e["id"]: e for e in previous}
    merged.update({e["id"]: e for e in results})
    order = [cid for cid, *_ in CHECKS]
    results = [merged[c] for c in order if c in merged]

    payload = {
        "format": "caustica-invariants/1",
        "environment": environment(),
        "checks": results,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (outdir / "REPORT.md").write_text(render_markdown(payload), encoding="utf-8")
    bad = [e["id"] for e in results if e["status"] != "OK"]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks ran -> {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
