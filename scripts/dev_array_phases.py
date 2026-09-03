"""A 64-element spiral driven element by element, with phases you supply.

Steering in this library has two faces. Delay-and-sum computes the phases from
a target point, and everything standing in the suite grades that one. The
other face is the one a dataset actually uses: a phase vector handed in from
outside, one number per element, which no closed form predicts and no gate
covers. This asks whether the library does what that vector says.

    A1  do the phases reach the grid at all? Every element present, and the
        complex drive summing to the elements' own areas rotated by their own
        phases -- one equality covering area, phase and superposition
    A2  ABSOLUTE focal pressure against the Rayleigh integral driven with the
        SAME vector, on a refinement ladder. The bowl half of this gate closed
        on 2026-08-25; the array half is still open, and an arbitrary phase
        vector is the case it was left open on
    A3  where the beam actually goes. With phases from outside there is no
        geometric focus to check against, so the reference is the Rayleigh
        field over a volume: predicted peak against simulated peak, in
        position and in megapascals
    A4  the same vector through k-Wave, on absolute amplitude
    A5  what rounding element centres to voxels costs THIS vector. Rounding
        turns per-element phase error into a coherent-sum loss of
        exp(-sigma^2/2); the repair was measured on delay-and-sum phases, and
        a vector with more phase spread has more to lose

Supply the vector with ``--phases``: a path to a ``.npy`` or whitespace/comma
text file of ``n`` values in radians, or ``das`` for delay-and-sum to the
array's own focus, or ``das:x,y,z`` to steer it to a point in millimetres, or
``zeros``. The default is ``das``, and the report says so, because a run that
silently invented its own steering would answer a question nobody asked.

Run it::

    python scripts/dev_array_phases.py --phases my_phases.npy
    python scripts/dev_array_phases.py --phases das:3,0,62 --only A2,A3

Sized for a large card: the S1 array is 60 mm across and its domain at 12
points per wavelength is about 300 Mvoxel. Every run is planned against the
device before it starts and skipped by name if it does not fit.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

MM = 1e-3
C_W, RHO_W = 1500.0, 1000.0

#: The project's standard array ("S1"): 64 elements on a
#: 60 mm spherical cap of 60 mm radius, inner hole 26.4 mm, f/1.0. Nine stored
#: setups use it, so a phase vector graded here is graded on the transducer
#: the dataset will be produced with.
N_ELEMENTS = 64
D_OUTER, D_INNER, ROC = 60.0 * MM, 26.4 * MM, 60.0 * MM
APEX_Z_MM = 5.5
PML_MM = 5.0

#: Leave this much of the card for whatever else is on it. The k-Wave CUDA
#: binary is sized separately because the planner models the native engine,
#: not an external binary: 320 bytes per voxel, measured (24.1 Mvoxel filled
#: 7.6 GB).
GPU_HEADROOM_BYTES = 3.0e9
KWAVE_GPU_BYTES_PER_VOXEL = 320

CHECKS: list[tuple[str, str, Callable]] = []


def check(cid: str, title: str):
    def wrap(fn):
        CHECKS.append((cid, title, fn))
        return fn

    return wrap


# --------------------------------------------------------------------------
# the array, the phases, and the grid they live on
# --------------------------------------------------------------------------


def build_array():
    from caustica.arrays.transducer import archimedean_spiral

    return archimedean_spiral(n_elements=N_ELEMENTS, d_outer=D_OUTER, d_inner=D_INNER, roc=ROC)


def resolve_phases(spec: str, array, f0: float) -> tuple[np.ndarray, str]:
    """Turn the ``--phases`` argument into ``n`` radians, and say what it was.

    The label travels with the numbers into the report. A phase vector is the
    experiment here, so a report that did not name which one it graded would
    be a table of numbers about nothing in particular.
    """
    n = array.n_elements
    if spec == "zeros":
        return np.zeros(n, np.float64), "all zeros (no steering)"
    if spec == "das":
        target = array.focus
        return (
            array.das_phases(target, f0, C_W).astype(np.float64),
            f"delay-and-sum to the array's own focus, z = {target[2] / MM:.1f} mm",
        )
    if spec.startswith("das:"):
        target = np.array([float(v) for v in spec[4:].split(",")], np.float64) * MM
        if target.shape != (3,):
            raise ValueError(f"das: wants three millimetre coordinates, got {spec!r}")
        return (
            array.das_phases(target, f0, C_W).astype(np.float64),
            f"delay-and-sum steered to ({', '.join(f'{v / MM:.1f}' for v in target)}) mm",
        )
    path = Path(spec)
    if not path.is_file():
        raise ValueError(f"--phases {spec!r} is not 'zeros', 'das', 'das:x,y,z' or a file")
    raw = (
        np.load(path)
        if path.suffix == ".npy"
        else np.fromstring(path.read_text(encoding="utf-8").replace(",", " "), sep=" ")
    )
    ph = np.asarray(raw, np.float64).ravel()
    if ph.shape != (n,):
        raise ValueError(f"{path} holds {ph.size} values; this array has {n} elements")
    return ph, f"{path.name} ({n} values supplied)"


def array_scene(ppw: float, f0: float, *, layers: bool = False):
    """A grid the S1 array fits in, sized from the array outwards.

    The domain carries the whole aperture plus a sponge, a full focal length
    of propagation, and half of one again past the focus so the far side of
    the lobe is not truncated by the absorbing band.
    """
    from caustica.core.grid import Grid
    from caustica.core.pml import PMLSpec
    from caustica.materials import breast_default
    from caustica.medium import Medium
    from caustica.solvers.kspace.operators import optimal_fft_size

    dx = C_W / f0 / ppw
    pml_vox = int(round(PML_MM * MM / dx))
    margin = 4
    n_xy = 2 * (int(np.ceil(D_OUTER / 2 / dx)) + pml_vox + margin) + 1
    apex_z = int(round(APEX_Z_MM * MM / dx))
    n_z = apex_z + int(round(1.5 * ROC / dx)) + pml_vox + margin
    # k-Wave takes the grid as given and slows to a crawl on primes like 83;
    # the native engine pads to this internally either way.
    shape = tuple(optimal_fft_size(int(v)) for v in (n_xy, n_xy, n_z))
    grid = Grid(shape=shape, dx=dx, pml=PMLSpec(thickness=PML_MM * MM))
    apex = (shape[0] // 2, shape[1] // 2, apex_z)

    c = np.full(shape, C_W, np.float32)
    rho = np.full(shape, RHO_W, np.float32)
    alpha = np.zeros(shape, np.float32)
    beta = np.zeros(shape, np.float32)
    if layers:
        db = breast_default().materials
        z_skin = apex_z + int(round(10.0 * MM / dx))
        z_fat = z_skin + int(round(2.0 * MM / dx))
        for lo, hi, mat in ((z_skin, z_fat, db[1]), (z_fat, shape[2], db[2])):
            c[:, :, lo:hi], rho[:, :, lo:hi] = mat.c, mat.rho
            alpha[:, :, lo:hi] = mat.alpha_np_m
    medium = Medium(alpha=alpha, rho=rho, c=c, beta=beta)
    focus = (apex[0], apex[1], apex[2] + int(round(ROC / dx)))
    return grid, medium, apex, focus, dx


def release_gpu_pool() -> None:
    """Hand CuPy's cached blocks back to the driver between rungs.

    CuPy keeps freed memory in a pool, so ``memGetInfo`` counts the previous
    rung's arrays as used and the next rung's budget reads far smaller than
    the card actually has.
    """
    try:
        import cupy
    except Exception:
        return
    try:
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def gpu_free_bytes() -> float | None:
    try:
        import cupy
    except Exception:
        return None
    release_gpu_pool()
    try:
        free, _total = cupy.cuda.runtime.memGetInfo()
    except Exception:
        return None
    return float(free)


def plan_native(grid, medium, source, spec, *, solver: str, harmonics=(1,)):
    """Ask the planner whether this fits, and how long it thinks it will take.

    The planner is the library's own answer to "will it fit"; using it here
    rather than a hand-rolled bytes-per-voxel means the pre-flight and the
    thing it is a pre-flight for agree by construction.
    """
    from caustica import planner

    try:
        est = planner.estimate(grid, medium, source, spec, solver=solver, harmonics=harmonics)
    except Exception as exc:  # a planner failure must not decide the physics
        return {"planner_error": f"{type(exc).__name__}: {exc}"[:120]}
    return {
        "vram_gib": est.vram_bytes / 2**30,
        "vram_usable_gib": est.vram_usable_bytes / 2**30,
        "fits": bool(est.fits),
        "t_expected_s": round(float(est.t_expected_s), 1),
        "estimate_source": est.source,
    }


def run_array(
    grid,
    medium,
    apex,
    focus,
    array,
    phases,
    f0,
    amplitude,
    *,
    solver,
    ctx,
    discretization="offgrid",
    harmonics=(1,),
):
    """Drive the array on this grid and hand back the field and the source."""
    from caustica.solvers import CWRunSpec, get

    asrc = array.voxelize(
        grid,
        apex,
        f0=f0,
        amplitude=amplitude,
        phases=phases.astype(np.float32),
        discretization=discretization,
    )
    spec = CWRunSpec(min_settle_periods=30, max_settle_periods=90, n_record_periods=2)
    if solver == "kwave":
        free = gpu_free_bytes()
        want = grid.n_voxels * KWAVE_GPU_BYTES_PER_VOXEL
        on_gpu = free is not None and want + GPU_HEADROOM_BYTES <= free
        kw, where = {"use_gpu_binary": on_gpu}, ("cuda" if on_gpu else "omp")
    else:
        kw, where = {"backend": ctx["backend"]}, ctx["backend"]
    t0 = time.perf_counter()
    res = get(solver)().run(
        grid, medium, asrc.source, spec, reference_point=focus, harmonics=harmonics, **kw
    )
    elapsed = time.perf_counter() - t0
    amp = np.abs(np.asarray(res.phasor)).astype(np.float64)
    release_gpu_pool()
    return amp, asrc, elapsed, where, res


def rayleigh_at(array, points, phases, f0, amplitude):
    """The Rayleigh integral over the element discs, driven the same way.

    ``u0 = p0 / (rho c)`` is the surface velocity a drive of ``p0`` pascals
    corresponds to -- the same convention the bowl gate uses against O'Neil,
    so the two absolute comparisons mean the same thing.
    """
    return np.abs(
        array.rayleigh_preview(points, f0=f0, phases=phases, c0=C_W, u0=amplitude / (RHO_W * C_W))
    )


def rel(a: float, b: float) -> float:
    return float(abs(a - b) / abs(b)) if b else float("nan")


# --------------------------------------------------------------------------
# A1 — did the vector reach the grid?
# --------------------------------------------------------------------------


@check("A1", "do the supplied phases reach the grid, element for element?")
def _a1(ctx):
    """One equality covers area, phase and superposition at once.

    Each element deposits its own disc area rotated by its own drive phase,
    and elements that overlap on the grid superpose as complex phasors rather
    than one of them winning. So the complex drive summed over every source
    voxel has to equal the sum of the element areas times their phasors. A
    phase vector that was dropped, truncated, re-sorted or silently replaced
    by zeros fails it; a merely plausible field would not.

    Cheap enough to run at three spacings, because the identity is exact and
    should not care which one.
    """
    array = build_array()
    phases = ctx["phases"]
    rows = []
    for ppw in (6.0, 8.0, 12.0):
        grid, _medium, apex, _focus, dx = array_scene(ppw, ctx["f0"])
        asrc = array.voxelize(
            grid, apex, f0=ctx["f0"], amplitude=ctx["amplitude"], phases=phases.astype(np.float32)
        )
        src = asrc.source
        w = src.drive_weights.astype(np.float64)
        deposited = complex(np.sum(w * np.exp(1j * src.phases.astype(np.float64))) * dx**2)
        wanted = complex(np.sum(np.pi * array.elem_radius**2 * np.exp(1j * phases)))
        rows.append(
            {
                "ppw": ppw,
                "dx_mm": dx / MM,
                "grid": "x".join(str(int(v)) for v in grid.shape),
                "elements_represented": int(asrc.n_elements_represented),
                "source_voxels": int(src.n_points),
                "deposited_mm2": abs(deposited) / MM**2,
                "expected_mm2": abs(wanted) / MM**2,
                "magnitude_rel_error": rel(abs(deposited), abs(wanted)),
                "phase_error_rad": float(abs(np.angle(deposited / wanted))),
            }
        )
    worst_mag = max(r["magnitude_rel_error"] for r in rows)
    worst_ph = max(r["phase_error_rad"] for r in rows)
    missing = [r["ppw"] for r in rows if r["elements_represented"] != N_ELEMENTS]
    return {
        "phase_source": ctx["phase_label"],
        "phase_spread_rad": float(np.ptp(phases)),
        "phase_std_rad": float(np.std(phases)),
        "element_radius_mm": array.elem_radius / MM,
        "rows": rows,
        "verdict": (
            f"all {N_ELEMENTS} elements reach the grid at every spacing"
            if not missing
            else f"elements missing at ppw {missing}"
        )
        + (
            f"; the complex drive matches the elements' own areas rotated by their own "
            f"phases to {worst_mag * 100:.3f} % in magnitude and {worst_ph:.2e} rad in phase"
        ),
    }


# --------------------------------------------------------------------------
# A2 — absolute focal pressure, against the Rayleigh integral
# --------------------------------------------------------------------------


@check("A2", "absolute pressure against the Rayleigh integral, same phase vector")
def _a2(ctx):
    """The gate M30 left open, asked with the phases a dataset will use.

    The bowl half of the absolute-amplitude gate closed against O'Neil. The
    array half did not, and this is the case it was left open on: with an
    arbitrary phase vector there is no O'Neil to appeal to, only the Rayleigh
    integral over the element discs driven by the same numbers.

    Graded the way the bowl gate is: a level in a band, and the error
    SHRINKING as dx falls. Either alone is weak. A source that carries the
    wrong area is wrong by a fixed factor at every spacing, so it can sit
    inside a band and never move; an honest discretization error sits outside
    the band at four points per wavelength and walks in.
    """
    array = build_array()
    phases = ctx["phases"]
    rows = []
    for ppw in ctx["ladder"]:
        grid, medium, apex, focus, dx = array_scene(ppw, ctx["f0"])
        origin = np.array(apex, np.float64) * dx
        try:
            from caustica.solvers import CWRunSpec

            probe = array.voxelize(
                grid,
                apex,
                f0=ctx["f0"],
                amplitude=ctx["amplitude"],
                phases=phases.astype(np.float32),
            )
            plan = plan_native(grid, medium, probe.source, CWRunSpec(), solver="linear")
            if plan.get("fits") is False:
                rows.append(
                    {
                        "ppw": ppw,
                        "megavoxels": grid.n_voxels / 1e6,
                        "error": f"needs {plan['vram_gib']:.1f} GiB, "
                        f"{plan['vram_usable_gib']:.1f} GiB usable",
                    }
                )
                continue
            amp, asrc, elapsed, where, _res = run_array(
                grid,
                medium,
                apex,
                focus,
                array,
                phases,
                ctx["f0"],
                ctx["amplitude"],
                solver="linear",
                ctx=ctx,
            )
        except Exception as exc:
            rows.append({"ppw": ppw, "error": f"{type(exc).__name__}: {exc}"[:140]})
            continue
        # Read both at the SIMULATED peak, found in a box around the array's
        # own focus: with steered phases the peak is not at the geometric
        # focus, and sampling the analytic reference at a point the field
        # does not peak at would grade the steering, not the amplitude.
        half = max(4, int(round(0.25 * ROC / dx)))
        box = tuple(
            slice(max(0, f - half), min(n, f + half + 1))
            for f, n in zip(focus, grid.shape, strict=True)
        )
        sub = amp[box]
        loc = np.unravel_index(int(sub.argmax()), sub.shape)
        peak_idx = tuple(int(b.start + o) for b, o in zip(box, loc, strict=True))
        point = (np.array(peak_idx, np.float64) * dx - origin)[None, :]
        exact = float(rayleigh_at(array, point, phases, ctx["f0"], ctx["amplitude"])[0])
        rows.append(
            {
                "ppw": ppw,
                "dx_mm": dx / MM,
                "megavoxels": grid.n_voxels / 1e6,
                "ran_on": where,
                "peak_mpa": float(sub.max()) / 1e6,
                "rayleigh_mpa": exact / 1e6,
                "ratio": float(sub.max()) / exact,
                "abs_error": abs(float(sub.max()) / exact - 1.0),
                "peak_z_mm": (peak_idx[2] - apex[2]) * dx / MM,
                "elapsed_s": round(elapsed, 1),
                "planner_t_expected_s": plan.get("t_expected_s"),
            }
        )

    good = [r for r in rows if "error" not in r]
    shrink = (
        good[-1]["abs_error"] / good[0]["abs_error"]
        if len(good) >= 2 and good[0]["abs_error"] > 0
        else None
    )
    return {
        "phase_source": ctx["phase_label"],
        "rows": rows,
        "error_shrink_factor": shrink,
        "verdict": (
            "no rung completed"
            if not good
            else (
                f"at {good[-1]['ppw']:.0f} points per wavelength the focal pressure is "
                f"{good[-1]['ratio']:.4f} of the Rayleigh integral driven by the same "
                f"{N_ELEMENTS} phases ({good[-1]['peak_mpa']:.4f} against "
                f"{good[-1]['rayleigh_mpa']:.4f} MPa)"
                + (
                    f", and the error shrank by x{shrink:.3f} from "
                    f"{good[0]['ppw']:.0f} points per wavelength"
                    if shrink is not None
                    else ""
                )
            )
        ),
    }


# --------------------------------------------------------------------------
# A3 — where does the beam actually go?
# --------------------------------------------------------------------------


@check("A3", "does the beam go where the phase vector says it should?")
def _a3(ctx):
    """No closed form predicts the focus of an arbitrary phase vector.

    Delay-and-sum has a target to check against; a vector handed in from
    outside has none. So the reference is the Rayleigh field itself, over the
    same volume: where it peaks is where those phases say the beam should go,
    and the simulated field either agrees or does not.

    A displacement in voxels is the honest unit for the position, because the
    grid is what limits it -- and reporting the lateral and axial parts
    separately matters, since an f/1 array's focus is several wavelengths
    long axially and under one across.
    """
    array = build_array()
    phases = ctx["phases"]
    grid, medium, apex, focus, dx = array_scene(ctx["ppw"], ctx["f0"])
    origin = np.array(apex, np.float64) * dx
    amp, asrc, elapsed, where, res = run_array(
        grid,
        medium,
        apex,
        focus,
        array,
        phases,
        ctx["f0"],
        ctx["amplitude"],
        solver="linear",
        ctx=ctx,
    )
    # Search the same box in both, centred on the array's own focus.
    half = max(6, int(round(0.35 * ROC / dx)))
    box = tuple(
        slice(max(0, f - half), min(n, f + half + 1))
        for f, n in zip(focus, grid.shape, strict=True)
    )
    sub = amp[box]
    grids = np.meshgrid(*(np.arange(b.start, b.stop, dtype=np.float64) for b in box), indexing="ij")
    pts = np.stack([g.ravel() for g in grids], axis=1) * dx - origin
    predicted = rayleigh_at(array, pts, phases, ctx["f0"], ctx["amplitude"]).reshape(sub.shape)

    def peak_of(vol):
        loc = np.unravel_index(int(vol.argmax()), vol.shape)
        idx = np.array([b.start + o for b, o in zip(box, loc, strict=True)], np.float64)
        return idx, float(vol.max())

    i_sim, p_sim = peak_of(sub)
    i_ref, p_ref = peak_of(predicted)
    d_vox = i_sim - i_ref
    lam_vox = (C_W / ctx["f0"]) / dx
    lateral = float(np.hypot(d_vox[0], d_vox[1]))
    return {
        "phase_source": ctx["phase_label"],
        "points_per_wavelength": ctx["ppw"],
        "grid": "x".join(str(int(v)) for v in grid.shape),
        "megavoxels": grid.n_voxels / 1e6,
        "ran_on": where,
        "converged_period": res.converged_period,
        "rows": [
            {
                "field": "simulated",
                "peak_mpa": p_sim / 1e6,
                "x_mm": (i_sim[0] - apex[0]) * dx / MM,
                "y_mm": (i_sim[1] - apex[1]) * dx / MM,
                "z_mm": (i_sim[2] - apex[2]) * dx / MM,
            },
            {
                "field": "Rayleigh (same phases)",
                "peak_mpa": p_ref / 1e6,
                "x_mm": (i_ref[0] - apex[0]) * dx / MM,
                "y_mm": (i_ref[1] - apex[1]) * dx / MM,
                "z_mm": (i_ref[2] - apex[2]) * dx / MM,
            },
        ],
        "displacement": {
            "lateral_vox": lateral,
            "lateral_wavelengths": lateral / lam_vox,
            "axial_vox": float(abs(d_vox[2])),
            "axial_wavelengths": float(abs(d_vox[2])) / lam_vox,
        },
        "amplitude_ratio": p_sim / p_ref,
        "elapsed_s": round(elapsed, 1),
        "verdict": (
            f"the simulated peak sits {lateral / lam_vox:.3f} wavelengths across and "
            f"{abs(d_vox[2]) / lam_vox:.3f} along the axis from where these {N_ELEMENTS} "
            f"phases predict it, carrying {p_sim / p_ref:.4f} of the predicted amplitude "
            f"({p_sim / 1e6:.4f} against {p_ref / 1e6:.4f} MPa)"
        ),
    }


# --------------------------------------------------------------------------
# A4 — the same vector through k-Wave
# --------------------------------------------------------------------------


@check("A4", "the same phase vector through k-Wave, on absolute amplitude")
def _a4(ctx):
    """An independent implementation of the same per-element drive.

    The two codes share the source voxels and nothing else: different
    staggering, different source smoothing, a different absorption
    implementation. What they should not differ on is what 64 numbers mean.
    """
    array = build_array()
    phases = ctx["phases"]
    grid, medium, apex, focus, dx = array_scene(ctx["ppw_cross"], ctx["f0"])
    rows, fields = [], {}
    for solver in ("linear", "kwave"):
        try:
            amp, asrc, elapsed, where, res = run_array(
                grid,
                medium,
                apex,
                focus,
                array,
                phases,
                ctx["f0"],
                ctx["amplitude"],
                solver=solver,
                ctx=ctx,
            )
        except Exception as exc:
            rows.append({"solver": solver, "error": f"{type(exc).__name__}: {exc}"[:140]})
            continue
        axis = amp[apex[0], apex[1], :]
        lo = apex[2] + int(round(0.3 * ROC / dx))
        hi = grid.shape[2] - grid.pml_vox - 2
        fields[solver] = axis[lo:hi]
        rows.append(
            {
                "solver": solver,
                "ran_on": where,
                "peak_mpa": float(amp.max()) / 1e6,
                "on_axis_peak_mpa": float(axis[lo:hi].max()) / 1e6,
                "on_axis_peak_z_mm": (lo + int(axis[lo:hi].argmax()) - apex[2]) * dx / MM,
                "elapsed_s": round(elapsed, 1),
            }
        )
    agree = {}
    if len(fields) == 2:
        a, b = fields["linear"], fields["kwave"]
        agree = {
            "peak_rel_difference": rel(float(a.max()), float(b.max())),
            "profile_rms_rel": float(np.sqrt(np.mean((a - b) ** 2)) / b.max()),
            "profile_correlation": float(np.corrcoef(a / a.max(), b / b.max())[0, 1]),
        }
    return {
        "phase_source": ctx["phase_label"],
        "points_per_wavelength": ctx["ppw_cross"],
        "grid": "x".join(str(int(v)) for v in grid.shape),
        "megavoxels": grid.n_voxels / 1e6,
        "rows": rows,
        "agreement": agree,
        "verdict": (
            "no pair completed"
            if not agree
            else (
                f"driven by the same {N_ELEMENTS} phases the two codes differ by "
                f"{agree['peak_rel_difference'] * 100:.2f} % on absolute on-axis peak "
                f"pressure; the axial profiles correlate "
                f"{agree['profile_correlation']:.5f}"
            )
        ),
    }


# --------------------------------------------------------------------------
# A5 — what rounding the element centres costs THIS vector
# --------------------------------------------------------------------------


@check("A5", "what rounding element centres to voxels costs this vector")
def _a5(ctx):
    """Rounding a centre is a phase error, and phase errors do not average out.

    An element centre rounded to the nearest voxel moves by up to half a
    voxel, which at 6 points per wavelength is a twelfth of a wavelength of
    phase. Across 64 elements those errors are independent, and a coherent
    sum with phase spread ``sigma`` keeps ``exp(-sigma^2/2)`` of itself: the
    loss measured when this was repaired was 0.61 rad rms and 17.6 %.

    That measurement was made on delay-and-sum phases. A vector with more
    spread of its own has more to lose, so the number is re-measured for the
    vector actually being used, at every spacing on the ladder.
    """
    array = build_array()
    phases = ctx["phases"]
    rows = []
    for ppw in ctx["ladder"]:
        grid, medium, apex, focus, dx = array_scene(ppw, ctx["f0"])
        origin = np.array(apex, np.float64) * dx
        got = {}
        for mode in ("offgrid", "binary"):
            try:
                amp, asrc, elapsed, where, _res = run_array(
                    grid,
                    medium,
                    apex,
                    focus,
                    array,
                    phases,
                    ctx["f0"],
                    ctx["amplitude"],
                    solver="linear",
                    ctx=ctx,
                    discretization=mode,
                )
            except Exception as exc:
                rows.append({"ppw": ppw, "mode": mode, "error": f"{exc}"[:120]})
                continue
            half = max(4, int(round(0.25 * ROC / dx)))
            box = tuple(
                slice(max(0, f - half), min(n, f + half + 1))
                for f, n in zip(focus, grid.shape, strict=True)
            )
            sub = amp[box]
            loc = np.unravel_index(int(sub.argmax()), sub.shape)
            peak_idx = np.array([b.start + o for b, o in zip(box, loc, strict=True)], np.float64)
            point = (peak_idx * dx - origin)[None, :]
            exact = float(rayleigh_at(array, point, phases, ctx["f0"], ctx["amplitude"])[0])
            got[mode] = float(sub.max())
            rows.append(
                {
                    "ppw": ppw,
                    "mode": mode,
                    "dx_mm": dx / MM,
                    "peak_mpa": got[mode] / 1e6,
                    "ratio_to_rayleigh": got[mode] / exact,
                    "elapsed_s": round(elapsed, 1),
                }
            )
        if len(got) == 2:
            rows[-1]["binary_over_offgrid"] = got["binary"] / got["offgrid"]

    pairs = [r for r in rows if "binary_over_offgrid" in r]
    return {
        "phase_source": ctx["phase_label"],
        "phase_std_rad": float(np.std(phases)),
        "rows": rows,
        "verdict": (
            "no spacing produced both discretizations"
            if not pairs
            else (
                f"rounding element centres to voxels keeps "
                f"{min(p['binary_over_offgrid'] for p in pairs) * 100:.1f} to "
                f"{max(p['binary_over_offgrid'] for p in pairs) * 100:.1f} % of the "
                f"coherent sum across the ladder, against a band-limited deposition "
                f"that carries every element's own area and phase"
            )
        ),
    }


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
    try:
        from caustica import _build_info

        env["commit"] = getattr(_build_info, "commit", "") or "(editable checkout)"
    except Exception:
        env["commit"] = "(editable checkout)"
    if cupy_available():
        import cupy

        props = cupy.cuda.runtime.getDeviceProperties(0)
        free, total = cupy.cuda.runtime.memGetInfo()
        env["gpu"] = props["name"].decode()
        env["vram_total_gib"] = round(total / 2**30, 1)
        env["vram_free_gib"] = round(free / 2**30, 1)
    return env


def cell(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.5g}" if v == 0 or 1e-3 <= abs(v) < 1e5 else f"{v:.3e}"
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
        "# A 64-element spiral, driven element by element",
        "",
        f"Generated by `scripts/dev_array_phases.py`. Phase vector: "
        f"**{payload.get('phase_source', '?')}**.",
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
    ap.add_argument("--out", default="benchmarks/reports/array_phases")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument(
        "--phases",
        default="das",
        help="'zeros', 'das', 'das:x,y,z' in mm, or a .npy/.txt file of n radians",
    )
    ap.add_argument("--f0", type=float, default=1.0e6, help="drive frequency [Hz]")
    ap.add_argument("--amplitude", type=float, default=1.0e5, help="element drive [Pa]")
    ap.add_argument("--ppw", type=float, default=12.0, help="points per wavelength for A3")
    ap.add_argument("--ppw-cross", type=float, default=10.0, help="points per wavelength for A4")
    ap.add_argument("--ladder", default="6,8,10,12", help="ppw ladder for A2 and A5")
    args = ap.parse_args(argv)

    from caustica.core.backend import cupy_available

    array = build_array()
    phases, label = resolve_phases(args.phases, array, args.f0)
    ctx = {
        "backend": "cupy" if cupy_available() else "numpy",
        "phases": phases,
        "phase_label": label,
        "f0": args.f0,
        "amplitude": args.amplitude,
        "ppw": args.ppw,
        "ppw_cross": args.ppw_cross,
        "ladder": [float(s) for s in args.ladder.split(",") if s.strip()],
    }
    only = {s.strip().upper() for s in args.only.split(",") if s.strip()}
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"array: {array.n_elements} elements, r = {array.elem_radius * 1e3:.4f} mm, "
        f"ROC = {array.focal_length * 1e3:.1f} mm",
        flush=True,
    )
    print(
        f"phases: {label}; spread {np.ptp(phases):.4f} rad, std {np.std(phases):.4f} rad",
        flush=True,
    )

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

    path = outdir / "array_phases.json"
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
        "format": "caustica-array-phases/1",
        "phase_source": label,
        "phases_rad": [float(v) for v in phases],
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
