"""Harmonics in a focused field: is 2f0 a measurement or a number?

The library records ``harmonics=(1, 2)`` and hands back a complex phasor for
each. Everything standing in the suite that grades the nonlinear term does so
in one dimension against Fubini's plane-wave series -- exact, sharp, and not
the geometry anything is used in. A focused beam has no closed form for its
harmonics, so the questions have to be asked another way:

    N1  does 2f0 grow as the SQUARE of f0 at low drive, and depart when it
        stops being quasi-linear? A law that holds in any geometry, so it
        probes the nonlinear term where Fubini cannot follow
    N2  a focused bowl in water, against k-Wave, on ABSOLUTE MPa at both
        harmonics rather than a normalized profile
    N3  does 2f0 converge as dx shrinks, and at what points per wavelength
        does it stop moving? 2f0 sees half the sampling f0 does, so a grid
        that resolves the fundamental can still be inventing the harmonic
    N4  the same bowl through skin and fat, against k-Wave
    N5  what the frequency-independent absorber costs at 2f0 in tissue

N5 is the one that is not a cross-check. The adapter deliberately hands
k-Wave ``alpha_power = 0``, because that is the absorption law this library
implements and a cross-check has to compare like with like -- so no amount of
agreement between the two codes says anything about whether that law is right
for tissue. It is not: soft tissue absorbs roughly as ``f^1.1``, so 2f0 is
under-absorbed by both codes together. N5 reaches past the adapter to measure
how much, in MPa, in the regime the dataset will actually be generated in.

Run it::

    python scripts/dev_nonlinear.py --out benchmarks/reports/nonlinear
    python scripts/dev_nonlinear.py --only N1,N3

N1 and N3 are caustica alone and want a GPU; N2, N4 and N5 each run k-Wave.
Every run is sized against the card before it starts: the k-Wave binary drops
to the CPU when its footprint would not leave the display anything, and a
native rung that does not fit is refused by name rather than attempted. Both
thresholds come from measured bytes per voxel, and the report records which
device each row ran on.
"""

from __future__ import annotations

import argparse
import contextlib
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
F0 = 1.0e6
C_W, RHO_W = 1500.0, 1000.0
BETA_W = 3.5  # water, the standard value the shock-distance helper defaults to

#: The bowl every check drives, shared so the water and tissue answers are
#: about the medium and not about the transducer. f/1.25 at 1 MHz: focal gain
#: near 20, so a fraction of a megapascal at the face is a few at the focus,
#: which is the regime a therapy dataset lives in.
APERTURE, ROC = 10.0 * MM, 25.0 * MM
PML_MM = 5.0

#: Bytes per voxel each code actually took on this geometry, measured rather
#: than derived: the native engine filled 2.3 GB at 24.1 Mvoxel, the k-Wave
#: CUDA binary filled 7.6 GB at the same size -- it carries far more matrices,
#: and the sensor buffer on top of them.
NATIVE_BYTES_PER_VOXEL = 100
KWAVE_GPU_BYTES_PER_VOXEL = 320

#: What to leave on the card for whatever else is drawing on it. On a laptop
#: GPU that is the display, and taking the last of it is not a slow run: this
#: machine went down on 2026-08-25 with the k-Wave binary holding 7.6 of 8.1
#: GB. A run that cannot fit inside the budget falls back to the CPU binary
#: (k-Wave) or is skipped and said so (native), which is slower and finishes.
GPU_HEADROOM_BYTES = 2.5e9


def release_gpu_pool() -> None:
    """Hand CuPy's cached blocks back to the driver.

    CuPy does not return freed memory to the card; it keeps it in a pool. So
    ``memGetInfo`` reports the pool as USED, and a budget read after one rung
    sees whatever the previous rung is still holding. The first version of
    this guard refused a 2.5 GB rung against a 2.1 GB budget for exactly that
    reason -- the card was not full, the pool was.
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


def gpu_budget_bytes() -> float | None:
    """Free VRAM minus the headroom, or ``None`` when there is no GPU."""
    try:
        import cupy
    except Exception:
        return None
    release_gpu_pool()
    try:
        free, _total = cupy.cuda.runtime.memGetInfo()
    except Exception:
        return None
    return max(0.0, float(free) - GPU_HEADROOM_BYTES)


def fits_on_gpu(n_voxels: int, bytes_per_voxel: int) -> bool:
    budget = gpu_budget_bytes()
    return budget is not None and n_voxels * bytes_per_voxel <= budget


CHECKS: list[tuple[str, str, Callable]] = []


def check(cid: str, title: str):
    def wrap(fn):
        CHECKS.append((cid, title, fn))
        return fn

    return wrap


# --------------------------------------------------------------------------
# one scene, built the same way every time
# --------------------------------------------------------------------------


def bowl_scene(ppw: float, *, layers: bool = False, beta: bool = True):
    """A focused bowl, in water or behind skin and fat, sized from the bowl out.

    ``ppw`` is points per wavelength at the FUNDAMENTAL. The second harmonic
    sees half of it, which is the whole subject of N3 and the reason this
    returns the number rather than the spacing.
    """
    from caustica.core.grid import Grid
    from caustica.core.pml import PMLSpec
    from caustica.materials import breast_default
    from caustica.medium import Medium
    from caustica.solvers.kspace.operators import optimal_fft_size

    dx = C_W / F0 / ppw
    pml_vox = int(round(PML_MM * MM / dx))
    margin = 4
    n_xy = 2 * (int(np.ceil(APERTURE / dx)) + pml_vox + margin) + 1
    apex_z = pml_vox + margin
    n_z = apex_z + int(round(1.5 * ROC / dx)) + pml_vox + margin
    # Grown to the next size whose prime factors are all in {2, 3, 5}. The
    # native engine pads to this internally whatever it is handed, but k-Wave
    # takes the grid as given -- and it says so, loudly, when the sizes carry
    # primes like 83 and 97, which the geometry produces at almost every ppw.
    # The extra voxels are water between the bowl and the sponge, so the two
    # codes are still compared on the same domain and neither is disadvantaged.
    shape = tuple(optimal_fft_size(n) for n in (n_xy, n_xy, n_z))
    n_xy, _, n_z = shape
    grid = Grid(shape=shape, dx=dx, pml=PMLSpec(thickness=PML_MM * MM))
    apex = (n_xy // 2, n_xy // 2, apex_z)

    c = np.full(shape, C_W, np.float32)
    rho = np.full(shape, RHO_W, np.float32)
    alpha = np.zeros(shape, np.float32)
    b = np.full(shape, BETA_W if beta else 0.0, np.float32)
    if layers:
        # Water standoff, then skin, then fat all the way through the focus:
        # the order a beam meets a breast, and the same stack H4 uses so the
        # linear and nonlinear answers are about the same geometry.
        db = breast_default().materials
        z_skin = apex_z + int(round(6.0 * MM / dx))
        z_fat = z_skin + int(round(2.0 * MM / dx))
        for lo, hi, mat in ((z_skin, z_fat, db[1]), (z_fat, n_z, db[2])):
            c[:, :, lo:hi], rho[:, :, lo:hi] = mat.c, mat.rho
            alpha[:, :, lo:hi] = mat.alpha_np_m
            if beta:
                b[:, :, lo:hi] = mat.beta
    medium = Medium(alpha=alpha, rho=rho, c=c, beta=b)
    focus = (apex[0], apex[1], apex[2] + int(round(ROC / dx)))
    return grid, medium, apex, focus, dx


def axis_window(shape, apex, dx):
    """The on-axis samples that are neither in the bowl nor in the sponge."""
    n_z = shape[2]
    z = (np.arange(n_z) - apex[2]) * dx
    pml_vox = int(round(PML_MM * MM / dx))
    sel = (z > 0.3 * ROC) & (z < (n_z - pml_vox - 2 - apex[2]) * dx)
    return z, sel


def run_bowl(grid, medium, apex, focus, drive, *, solver, ctx, harmonics=(1, 2)):
    """One run, returning the on-axis harmonic profiles in Pa."""
    from caustica.solvers import CWRunSpec, get
    from caustica.sources import bowl_cw_source

    src = bowl_cw_source(grid, F0, drive, APERTURE, ROC, apex)
    # Thirty periods of settle, not the default eight, and the same number to
    # both codes. The native engine now decides for itself when a harmonic has
    # stopped moving; the k-Wave adapter runs a FIXED schedule and cannot, so
    # it has to be told. Measured on this geometry with beta = 0, where the
    # true 2f0 is zero: a ten-period floor put 2.1 % of the fundamental into
    # the harmonic channel. Handing the two codes the same schedule and
    # letting only one of them be converged would have shown up as a 2f0
    # disagreement and read like a difference between the codes.
    spec = CWRunSpec(min_settle_periods=30, max_settle_periods=90, n_record_periods=2)
    if solver == "kwave":
        on_gpu = fits_on_gpu(grid.n_voxels, KWAVE_GPU_BYTES_PER_VOXEL)
        kw = {"use_gpu_binary": on_gpu}
        where = "cuda" if on_gpu else "omp"
    else:
        kw = {"backend": ctx["backend"]}
        where = ctx["backend"]
        if ctx["backend"] == "cupy" and not fits_on_gpu(grid.n_voxels, NATIVE_BYTES_PER_VOXEL):
            raise MemoryError(
                f"{grid.n_voxels / 1e6:.1f} Mvoxel needs about "
                f"{grid.n_voxels * NATIVE_BYTES_PER_VOXEL / 1e9:.1f} GB and the budget is "
                f"{(gpu_budget_bytes() or 0) / 1e9:.1f} GB; run this rung on a larger card"
            )
    t0 = time.perf_counter()
    res = get(solver)().run(
        grid, medium, src, spec, reference_point=focus, harmonics=harmonics, **kw
    )
    elapsed = time.perf_counter() - t0
    z, sel = axis_window(grid.shape, apex, grid.dx)
    out = {}
    for h in harmonics:
        amp = np.abs(np.asarray(res.phasors[h])).astype(np.float64)
        out[h] = amp[apex[0], apex[1], :][sel]
    # The profiles are on the host now, so nothing is lost by giving the card
    # back what this rung took -- and the next rung's budget then means what
    # it says.
    release_gpu_pool()
    return z[sel], out, elapsed, res, where


@contextlib.contextmanager
def kwave_alpha_power(y: float):
    """Give k-Wave a power-law exponent the adapter would never pass it.

    The adapter pins ``alpha_power = 0`` deliberately: that is the law this
    library implements, so a cross-check that changed it would be comparing
    two different physics and calling the difference a numerics error. N5
    wants exactly that difference, which is why reaching past the adapter
    lives in a script and not in the library.

    ``alpha_coeff`` is unchanged, so both runs absorb identically at 1 MHz
    and differ only in what they do to 2 MHz -- a factor ``2^y``.

    It wraps ``__init__`` rather than substituting the class, and that detail
    is load-bearing: ``kspaceFirstOrder2D`` and its 3-D sibling are wrapped in
    beartype, which resolves the ``medium`` hint to whatever
    ``kwave.kmedium.kWaveMedium`` was on the FIRST call and caches it. Swap
    the class and the second run hands beartype an object of a class that is
    no longer the one it remembers, and the run dies on a type violation
    rather than on anything to do with acoustics.
    """
    import kwave.kmedium as km

    original = km.kWaveMedium.__init__

    def patched(self, *args: Any, **kwargs: Any) -> None:
        kwargs["alpha_power"] = y
        original(self, *args, **kwargs)

    km.kWaveMedium.__init__ = patched
    try:
        yield
    finally:
        km.kWaveMedium.__init__ = original


def peak(profile) -> float:
    return float(np.max(profile))


def mpa(value: float) -> float:
    return value / 1e6


# --------------------------------------------------------------------------
# N1 — the law the nonlinear term has to obey in any geometry
# --------------------------------------------------------------------------


@check("N1", "does 2f0 grow as the square of f0, and stop when it should?")
def _n1(ctx):
    """Quasi-linear theory says ``p2 ~ p1^2``; a mis-scaled term says otherwise.

    Fubini grades the nonlinear term in one dimension, where it is exact. In
    a focused beam there is no series to compare against -- but the SCALING
    survives the geometry: while the second harmonic is a small perturbation
    fed by the fundamental, doubling the drive quadruples it, whatever the
    beam is doing. That is a two-decade prediction with no free parameters,
    and it fails loudly if the Westervelt term carries the wrong power of
    pressure or the wrong factor of beta.

    Fitting a slope to ``log p2`` against ``log p1`` also says where the
    quasi-linear regime ENDS, which is the number that decides how hard a
    dataset can be driven before its harmonics stop being a perturbation.
    """
    from caustica.analytic import shock_distance

    grid, medium, apex, focus, dx = bowl_scene(ctx["ppw"])
    rows = []
    for drive in ctx["drives"]:
        _z, prof, elapsed, _res, _where = run_bowl(
            grid, medium, apex, focus, drive, solver="westervelt", ctx=ctx
        )
        p1, p2 = peak(prof[1]), peak(prof[2])
        rows.append(
            {
                "drive_kpa": drive / 1e3,
                "p1_mpa": mpa(p1),
                "p2_mpa": mpa(p2),
                "p2_over_p1": p2 / p1,
                "focal_gain": p1 / drive,
                "shock_distance_mm": shock_distance(p1, F0, C_W, RHO_W, BETA_W) / MM,
                "elapsed_s": round(elapsed, 1),
            }
        )

    p1 = np.array([r["p1_mpa"] for r in rows])
    p2 = np.array([r["p2_mpa"] for r in rows])
    slope_all = float(np.polyfit(np.log(p1), np.log(p2), 1)[0])
    # The quasi-linear slope is read from the WEAKEST half of the ladder,
    # where the perturbation assumption is the one being tested; including
    # the saturating end would fold the departure into the number that is
    # supposed to detect it.
    half = max(2, len(rows) // 2)
    slope_low = float(np.polyfit(np.log(p1[:half]), np.log(p2[:half]), 1)[0])
    # Whether the ladder actually left the quasi-linear regime is something to
    # read off the two slopes, not to assert: a ladder that never departs is a
    # statement about the drives chosen, and saying otherwise would put a
    # conclusion in the report that the numbers do not carry.
    departs = abs(slope_all - slope_low) > 0.05
    return {
        "points_per_wavelength": ctx["ppw"],
        "grid": list(map(int, grid.shape)),
        "dx_mm": dx / MM,
        "rows": rows,
        "quasilinear_slope": slope_low,
        "slope_over_full_ladder": slope_all,
        "leaves_the_quasilinear_regime": departs,
        "verdict": (
            f"over the weakest {half} rungs the second harmonic grows as the "
            f"{slope_low:.3f} power of the fundamental, against the quasi-linear "
            f"prediction of 2; across the whole ladder the slope "
            + (f"falls to {slope_all:.3f}" if departs else f"is unchanged at {slope_all:.3f}")
            + f", with 2f0 reaching {rows[-1]['p2_over_p1'] * 100:.1f} % of f0 at "
            f"{rows[-1]['p1_mpa']:.2f} MPa"
        ),
    }


# --------------------------------------------------------------------------
# N2 — the focused harmonics against an independent code
# --------------------------------------------------------------------------


@check("N2", "focused bowl in water: absolute f0 and 2f0 against k-Wave")
def _n2(ctx):
    """The comparison item 3 is actually about, graded in megapascals.

    Both codes solve Westervelt on a k-space pseudospectral grid, so this is
    not an independent derivation of the physics -- it is an independent
    implementation of it, with different staggering, different source
    smoothing and a different absorption implementation. That is enough for
    the failure modes that matter here: a harmonic extracted over the wrong
    window, a nonlinear term scaled by the wrong constant, a drive that is
    not the amplitude it claims.

    Both harmonics are graded on ABSOLUTE amplitude. The shipped cross-check
    grades a normalized correlation, which is how an 18 % amplitude error
    lived behind green gates for months.
    """
    grid, medium, apex, focus, dx = bowl_scene(ctx["ppw"])
    rows, fields = [], {}
    for solver in ("westervelt", "kwave"):
        try:
            z, prof, elapsed, res, where = run_bowl(
                grid, medium, apex, focus, ctx["drive"], solver=solver, ctx=ctx
            )
        except Exception as exc:
            rows.append({"solver": solver, "error": f"{exc}"[:140]})
            continue
        fields[solver] = (z, prof)
        rows.append(
            {
                "solver": solver,
                "ran_on": where,
                "converged_period": res.converged_period,
                "settle_capped": bool(res.settle_capped),
                "p1_mpa": mpa(peak(prof[1])),
                "p2_mpa": mpa(peak(prof[2])),
                "p2_over_p1_pct": 100.0 * peak(prof[2]) / peak(prof[1]),
                "p1_peak_z_mm": float(z[int(prof[1].argmax())] / MM),
                "p2_peak_z_mm": float(z[int(prof[2].argmax())] / MM),
                "samples_per_period": int(res.spp),
                "elapsed_s": round(elapsed, 1),
            }
        )

    agree = {}
    if "westervelt" in fields and "kwave" in fields:
        _z, a = fields["westervelt"]
        _z, b = fields["kwave"]
        for h in (1, 2):
            agree[f"h{h}"] = {
                "peak_rel_difference": abs(peak(a[h]) - peak(b[h])) / peak(b[h]),
                "profile_rms_rel": float(np.sqrt(np.mean((a[h] - b[h]) ** 2)) / peak(b[h])),
                "profile_correlation": float(
                    np.corrcoef(a[h] / peak(a[h]), b[h] / peak(b[h]))[0, 1]
                ),
            }
    return {
        "points_per_wavelength": ctx["ppw"],
        "points_per_wavelength_at_2f0": ctx["ppw"] / 2,
        "grid": list(map(int, grid.shape)),
        "megavoxels": grid.n_voxels / 1e6,
        "dx_mm": dx / MM,
        "drive_kpa": ctx["drive"] / 1e3,
        "rows": rows,
        "agreement": agree,
        "verdict": (
            "no pair completed"
            if not agree
            else (
                f"the two codes differ by {agree['h1']['peak_rel_difference'] * 100:.2f} % on "
                f"absolute focal f0 and {agree['h2']['peak_rel_difference'] * 100:.2f} % on 2f0, "
                f"at {ctx['ppw'] / 2:.0f} points per wavelength at the harmonic; axial profiles "
                f"correlate {agree['h1']['profile_correlation']:.5f} and "
                f"{agree['h2']['profile_correlation']:.5f}"
            )
        ),
    }


# --------------------------------------------------------------------------
# N3 — is the harmonic converged, or is it the grid talking?
# --------------------------------------------------------------------------


@check("N3", "does 2f0 converge as dx shrinks, and where does it stop moving?")
def _n3(ctx):
    """The question that decides what dx a dataset can be generated at.

    A grid chosen to resolve the fundamental gives the second harmonic half
    as many points per wavelength, and a spectral method that is comfortable
    at 12 is not necessarily comfortable at 6. Nothing about a converged f0
    implies a converged 2f0, so the two are refined together and reported
    apart. The rung-to-rung change is what matters, not the value: the
    finest rung is the reference and the question is when the others stopped
    moving toward it.
    """
    ladder = ctx["ladder"]
    rows = []
    for ppw in ladder:
        grid, medium, apex, focus, dx = bowl_scene(ppw)
        try:
            z, prof, elapsed, res, where = run_bowl(
                grid, medium, apex, focus, ctx["drive"], solver="westervelt", ctx=ctx
            )
        except Exception as exc:
            rows.append({"ppw_at_f0": ppw, "error": f"{exc}"[:140]})
            continue
        rows.append(
            {
                "ppw_at_f0": ppw,
                "ppw_at_2f0": ppw / 2,
                "ran_on": where,
                "converged_period": res.converged_period,
                "dx_mm": dx / MM,
                "megavoxels": grid.n_voxels / 1e6,
                "samples_per_period": int(res.spp),
                "p1_mpa": mpa(peak(prof[1])),
                "p2_mpa": mpa(peak(prof[2])),
                "p2_over_p1_pct": 100.0 * peak(prof[2]) / peak(prof[1]),
                "elapsed_s": round(elapsed, 1),
            }
        )

    good = [r for r in rows if "error" not in r]
    if len(good) < 2:
        return {"rows": rows, "verdict": "fewer than two rungs completed"}
    ref = good[-1]
    for r in good:
        r["p1_rel_to_finest"] = abs(r["p1_mpa"] - ref["p1_mpa"]) / ref["p1_mpa"]
        r["p2_rel_to_finest"] = abs(r["p2_mpa"] - ref["p2_mpa"]) / ref["p2_mpa"]
    # The coarsest rung whose harmonic is already within 5 % of the finest --
    # a working answer to "what dx", not a claim that 5 % is the right bar.
    settled = [r for r in good[:-1] if r["p2_rel_to_finest"] < 0.05]
    at = min((r["ppw_at_f0"] for r in settled), default=None)
    return {
        "drive_kpa": ctx["drive"] / 1e3,
        "reference_ppw": ref["ppw_at_f0"],
        "rows": rows,
        "settles_within_5pct_at_ppw": at,
        "verdict": (
            f"against the finest rung ({ref['ppw_at_f0']:.0f} points per wavelength at f0, "
            f"{ref['ppw_at_2f0']:.0f} at 2f0), the fundamental is within "
            f"{100 * good[0]['p1_rel_to_finest']:.1f} % from the coarsest rung on, while the "
            f"second harmonic is {100 * good[0]['p2_rel_to_finest']:.1f} % out there and "
            + (
                f"first comes within 5 % at {at:.0f} points per wavelength at f0"
                if at is not None
                else "never comes within 5 % on this ladder"
            )
        ),
    }


# --------------------------------------------------------------------------
# N4 — harmonics through tissue, against k-Wave
# --------------------------------------------------------------------------


@check("N4", "bowl through skin and fat: harmonics against k-Wave in tissue")
def _n4(ctx):
    """Nonlinearity and heterogeneity together, which is the dataset's regime.

    Two things change at once relative to N2, and that is deliberate: skin
    and fat carry both a higher beta than water and real absorption, so the
    harmonic is being generated faster and removed faster. If the two codes
    still agree in absolute terms, the disagreement in N2 was not hiding a
    compensation.
    """
    grid, medium, apex, focus, dx = bowl_scene(ctx["ppw"], layers=True)
    rows, fields = [], {}
    for solver in ("westervelt", "kwave"):
        try:
            z, prof, elapsed, res, where = run_bowl(
                grid, medium, apex, focus, ctx["drive"], solver=solver, ctx=ctx
            )
        except Exception as exc:
            rows.append({"solver": solver, "error": f"{exc}"[:140]})
            continue
        fields[solver] = prof
        rows.append(
            {
                "solver": solver,
                "ran_on": where,
                "converged_period": res.converged_period,
                "settle_capped": bool(res.settle_capped),
                "p1_mpa": mpa(peak(prof[1])),
                "p2_mpa": mpa(peak(prof[2])),
                "p2_over_p1_pct": 100.0 * peak(prof[2]) / peak(prof[1]),
                "p1_peak_z_mm": float(z[int(prof[1].argmax())] / MM),
                "elapsed_s": round(elapsed, 1),
            }
        )

    agree = {}
    if len(fields) == 2:
        a, b = fields["westervelt"], fields["kwave"]
        for h in (1, 2):
            agree[f"h{h}"] = {
                "peak_rel_difference": abs(peak(a[h]) - peak(b[h])) / peak(b[h]),
                "profile_correlation": float(
                    np.corrcoef(a[h] / peak(a[h]), b[h] / peak(b[h]))[0, 1]
                ),
            }
    return {
        "points_per_wavelength": ctx["ppw"],
        "grid": list(map(int, grid.shape)),
        "layers": "6 mm water, 2 mm skin, then fat through the focus",
        "rows": rows,
        "agreement": agree,
        "verdict": (
            "no pair completed"
            if not agree
            else (
                f"through skin and fat the two codes differ by "
                f"{agree['h1']['peak_rel_difference'] * 100:.2f} % on absolute f0 and "
                f"{agree['h2']['peak_rel_difference'] * 100:.2f} % on 2f0; profiles correlate "
                f"{agree['h1']['profile_correlation']:.5f} and "
                f"{agree['h2']['profile_correlation']:.5f}"
            )
        ),
    }


# --------------------------------------------------------------------------
# N5 — the model gap the cross-checks cannot see
# --------------------------------------------------------------------------


@check("N5", "what frequency-independent absorption costs at 2f0 in tissue")
def _n5(ctx):
    """Two codes agreeing on the wrong law still agree.

    N2 and N4 grade numerics: same equations, different implementations. The
    absorption LAW is shared, because the adapter hands k-Wave the exponent
    this library implements -- zero. Soft tissue is not zero; it is close to
    ``f^1.1``, so 2 MHz should be absorbed about ``2^1.1 = 2.14`` times
    harder than 1 MHz and in both codes it is absorbed exactly as hard.

    Same scene, same ``alpha_coeff``, one number changed. The fundamental is
    absorbed identically either way -- ``1 MHz^1.1`` is 1 MHz -- but it does
    NOT come back unchanged, and that is the second thing this measures. A
    power law carries Kramers-Kronig dispersion with it; a
    frequency-independent absorber carries none, so it also gets the sound
    speed slightly wrong. A 2-D rehearsal of this check put the fundamental
    2.6 % apart and the harmonic 29 % apart, which is why both are reported
    rather than the harmonic alone.

    What comes back is the size of the library's known absorption gap, in
    megapascals, in the regime a breast dataset will be generated in -- not
    an argument that the gap is acceptable.
    """
    grid, medium, apex, focus, dx = bowl_scene(ctx["ppw"], layers=True)
    rows, fields = [], {}
    for label, y in (("y = 0 (what the adapter sends)", 0.0), ("y = 1.1 (soft tissue)", 1.1)):
        try:
            with kwave_alpha_power(y):
                z, prof, elapsed, _res, where = run_bowl(
                    grid, medium, apex, focus, ctx["drive"], solver="kwave", ctx=ctx
                )
        except Exception as exc:
            rows.append({"absorption_law": label, "error": f"{exc}"[:140]})
            continue
        fields[y] = prof
        rows.append(
            {
                "absorption_law": label,
                "ran_on": where,
                "alpha_power": y,
                "p1_mpa": mpa(peak(prof[1])),
                "p2_mpa": mpa(peak(prof[2])),
                "p2_over_p1_pct": 100.0 * peak(prof[2]) / peak(prof[1]),
                "elapsed_s": round(elapsed, 1),
            }
        )

    gap = {}
    if len(fields) == 2:
        flat, law = fields[0.0], fields[1.1]
        for h in (1, 2):
            gap[f"h{h}"] = {
                "flat_mpa": mpa(peak(flat[h])),
                "power_law_mpa": mpa(peak(law[h])),
                "overprediction": (peak(flat[h]) - peak(law[h])) / peak(law[h]),
            }
    return {
        "points_per_wavelength": ctx["ppw"],
        "layers": "6 mm water, 2 mm skin, then fat through the focus",
        "note": (
            "both runs are k-Wave with identical alpha_coeff; only the exponent differs. "
            "Absorption at f0 is the same either way, but the power law also carries "
            "Kramers-Kronig dispersion, so the fundamental moves too -- the harmonic "
            "carries most of the difference, not all of it"
        ),
        "rows": rows,
        "gap": gap,
        "verdict": (
            "no pair completed"
            if not gap
            else (
                f"holding absorption frequency-independent moves the fundamental "
                f"{gap['h1']['overprediction'] * 100:+.2f} % (dispersion, not absorption: "
                f"the coefficient at f0 is identical) and 2f0 by "
                f"{gap['h2']['overprediction'] * 100:+.2f} % "
                f"({gap['h2']['flat_mpa']:.3f} against {gap['h2']['power_law_mpa']:.3f} MPa), "
                f"which is the size of the library's absorption gap in this regime"
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
    if cupy_available():
        import cupy

        env["gpu"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    return env


def cell(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}" if v == 0 or 1e-3 <= abs(v) < 1e5 else f"{v:.3e}"
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
        "# Harmonics in a focused field",
        "",
        "Generated by `scripts/dev_nonlinear.py`. N1 and N3 are caustica against a",
        "scaling law and against itself under refinement; N2 and N4 are against",
        "k-Wave on absolute amplitude; N5 measures the shared absorption model.",
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
    ap.add_argument("--out", default="benchmarks/reports/nonlinear")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--ppw", type=float, default=12.0, help="points per wavelength at f0")
    ap.add_argument("--drive", type=float, default=1.5e5, help="source amplitude [Pa]")
    ap.add_argument(
        "--drives", default="25e3,50e3,100e3,200e3,400e3,800e3", help="N1 drive ladder [Pa]"
    )
    ap.add_argument("--ladder", default="6,8,10,12,14", help="ppw ladder for N3")
    args = ap.parse_args(argv)

    from caustica.core.backend import cupy_available

    ctx = {
        "backend": "cupy" if cupy_available() else "numpy",
        "ppw": args.ppw,
        "drive": args.drive,
        "drives": [float(s) for s in args.drives.split(",") if s.strip()],
        "ladder": [float(s) for s in args.ladder.split(",") if s.strip()],
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
        print(f"  {mark} {entry['elapsed_s']:>7.2f}s  {detail}", flush=True)

    path = outdir / "nonlinear.json"
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
        "format": "caustica-nonlinear/1",
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
