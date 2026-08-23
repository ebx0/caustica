"""M11's second box: the library's ANALYTIC references, graded on this machine.

``caustica.analytic`` holds the closed forms every solver in this repository
was built against — exponential absorption, O'Neil's on-axis bowl solution,
the Rayleigh integral, Fubini's pre-shock harmonic series. The physics test
suite already checks solver-vs-closed-form agreement, but a test run is a
green dot in somebody's terminal: it produces no artifact, records no
environment, and cannot be handed to a reader who asks *"does this library
still reproduce the physics on the machine you are about to run it on?"*.

That is what this suite is, and the operator's whole job is::

    python -m caustica.validation run-analytic

It runs four scenarios on the resolved backend (cupy if this machine has a
GPU, else numpy — the analytic references themselves are always CPU/numpy),
compares each against its closed form, and writes a stamped PASS/FAIL report
under ``benchmarks/reports/analytic/<backend>-<timestamp>/``:

1. **plane wave** — one 1-D lossless run and one absorbing run. Three
   contracts: the realized drive amplitude against ``p0`` (the mass-source
   normalization), the numerical phase speed against ``k = omega/c``, and the
   fitted decay exponent against the configured ``alpha`` of
   :func:`caustica.analytic.attenuate`.
2. **O'Neil bowl** — a 3-D focused bowl in water against
   :func:`caustica.analytic.axial_pressure`: axial profile correlation, focal
   peak position, and the -6 dB axial width.
3. **linear limit** — the westervelt solver with ``beta = 0`` against the
   linear solver. Not "close": the same arithmetic, so bit-identical.
4. **Fubini** — second-harmonic growth ``A2/A1`` across a range of ``sigma``
   against :func:`caustica.analytic.fubini_harmonic`.

**Every tolerance here is inherited, never invented.** Each gated number
names, in the report and in the JSON, the test that already established its
limit; a suite free to choose its own tolerances is a suite that can be tuned
until it passes. The verdict algebra is the shared one in
:mod:`caustica.validation._verdict`: a scenario that raised contributes SKIPs,
SKIP never counts toward a gate, a gate short of its required count is
INCOMPLETE, and the exit code is 0 only when every gate is PASS.

Nothing external is involved — every medium is homogeneous water from
:mod:`caustica.materials` and every source is built by the library. The whole
suite is seconds of CPU: ``--size quick`` shrinks the one 3-D scenario to
about a second so the path can be exercised in the test suite itself.
"""

from __future__ import annotations

import json
import platform
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from caustica.validation._verdict import (
    EXIT_FAILED,
    EXIT_OK,
    Check,
    Gate,
    fmt_num,
    overall_verdict,
)

#: Report format tag; the JSON companion of the Markdown page.
FORMAT = "caustica-analytic/1"

# --------------------------------------------------------------- the physics
#
# One frequency and one medium across every scenario, so a reader comparing
# two rows of the report is comparing the solver and not the setup.

F0 = 1.0e6  # [Hz]
C0 = 1500.0  # [m/s]
BETA = 3.5  # water's nonlinearity parameter
DRIVE_PA = 1.0e5  # linear-scenario drive amplitude [Pa]
ALPHA_NP_M = 30.0  # absorbing-scenario alpha [Np/m] -> e^{-1.35} over the window
PPW_1D = 4.0  # points per wavelength for the 1-D runs
DX_1D = C0 / (F0 * PPW_1D)  # 0.375 mm
K0 = 2.0 * np.pi * F0 / C0

#: Measurement window for the 1-D runs [voxels]: 30 wavelengths, with the
#: source (voxel 40) and both sponges outside it. Same window the physics
#: tests measure in, so the numbers here are comparable to theirs.
WINDOW_1D = slice(80, 200)

#: Sigma stations for the Fubini scenario [voxel index]. Fubini is exact only
#: pre-shock, so a station whose sigma falls outside 0.05..0.95 is dropped
#: rather than graded against a formula that does not apply there.
FUBINI_STATIONS = (150, 250, 350, 450, 540)
FUBINI_SIGMA_BAND = (0.05, 0.95)

# ------------------------------------------------------------- tolerances
#
# INHERITED, NOT INVENTED. Every limit below is the one an existing test
# already validated this library against, and every gated check carries the
# test's name into the report (see Check.citing). If one of these should be
# tightened, tighten the physics test first: that is where the claim lives.

AMPLITUDE_BAND = (0.90, 1.12)
AMPLITUDE_SOURCE = (
    "tests/test_review_hardening.py::test_mass_source_normalization_and_phase_convention"
)

PHASE_SPEED_TOL_PCT = 0.1
PHASE_SPEED_SOURCE = "tests/test_linear_planewave.py::test_plane_wave_dispersion_below_0p1_percent"

ALPHA_TOL_PCT = 1.0
ALPHA_SOURCE = (
    "tests/test_linear_planewave.py::test_absorption_matches_configured_alpha_within_1_percent"
)

ONEILL_CORR_MIN = 0.99
ONEILL_WIDTH_TOL_PCT = 5.0
ONEILL_AXIAL_SOURCE = "tests/test_linear_oneill_3d.py::test_axial_profile_matches_oneill"

FOCUS_POS_TOL_VOX = 1.0
FOCUS_POS_SOURCE = "tests/test_linear_oneill_3d.py::test_focus_lands_within_one_voxel"

LINEAR_LIMIT_SOURCE = "tests/test_westervelt.py::test_beta_zero_is_bitwise_identical_to_linear"

FUBINI_TOL = 0.05
FUBINI_STATIONS_REQUIRED = 4  # the source test's own `assert checked >= 4`
FUBINI_SOURCE = "tests/test_westervelt.py::test_fubini_second_harmonic_within_5_percent"


# ------------------------------------------------------------------- sizes


@dataclass(frozen=True)
class Size:
    """The one thing that changes between a ``full`` and a ``quick`` run.

    Only the 3-D bowl is worth shrinking: the 1-D scenarios are already
    hundredths of a second, so making them smaller would change physics that
    was validated at these numbers and buy nothing back.

    The quick bowl is a DIFFERENT bowl, not a coarser one — same 0.375 mm
    voxel, lower f-number, so its -6 dB lobe still fits inside a smaller box
    with the sponge cleared. It is graded against exactly the same tolerances
    as the full one, which it therefore has to earn honestly; a "quick" mode
    with relaxed limits would be a way to make the suite pass by asking it
    less.
    """

    name: str
    shape: tuple[int, int, int]
    aperture_vox: int
    roc_vox: int
    apex_vox: tuple[int, int, int]
    pml_mm: float
    dx_mm: float = 0.375

    @property
    def aperture_m(self) -> float:
        return self.aperture_vox * self.dx_mm * 1e-3

    @property
    def roc_m(self) -> float:
        return self.roc_vox * self.dx_mm * 1e-3

    @property
    def focus_vox(self) -> tuple[int, int, int]:
        return (self.apex_vox[0], self.apex_vox[1], self.apex_vox[2] + self.roc_vox)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "bowl_shape": list(self.shape),
            "dx_mm": self.dx_mm,
            "aperture_mm": self.aperture_vox * self.dx_mm,
            "roc_mm": self.roc_vox * self.dx_mm,
            "apex_vox": list(self.apex_vox),
            "pml_mm": self.pml_mm,
        }


#: ``full`` is the geometry ``tests/test_linear_oneill_3d.py`` validated.
SIZES: dict[str, Size] = {
    "full": Size("full", (80, 80, 120), 20, 48, (40, 40, 16), 5.0),
    "quick": Size("quick", (60, 60, 80), 18, 32, (30, 30, 12), 3.75),
}


def size_preset(name: str) -> Size:
    try:
        return SIZES[name]
    except KeyError:
        raise ValueError(f"unknown size {name!r}; known: {sorted(SIZES)}") from None


# ------------------------------------------------------------ measurements


def minus6db_width(coord: np.ndarray, profile: np.ndarray) -> float:
    """Width of the -6 dB (half-amplitude) main lobe around the peak.

    Raises when a crossing falls outside the profile: a lobe that runs off
    the end of the window has no width, and returning the window's length
    instead would be a made-up number that could pass a 5% gate. (Same
    implementation as the physics test's, so the two measure the same thing.)
    """
    prof = np.asarray(profile, dtype=np.float64)
    i_pk = int(np.argmax(prof))
    half = prof[i_pk] / 2.0
    lo = i_pk
    while lo > 0 and prof[lo] > half:
        lo -= 1
    hi = i_pk
    while hi < prof.size - 1 and prof[hi] > half:
        hi += 1
    if lo == 0 or hi == prof.size - 1:
        raise ValueError("-6 dB lobe is not fully contained in the measurement window")

    def cross(i0: int, i1: int) -> float:
        f = (half - prof[i0]) / (prof[i1] - prof[i0])
        return float(coord[i0] + f * (coord[i1] - coord[i0]))

    return cross(hi - 1, hi) - cross(lo + 1, lo)


def _run_1d(
    *,
    alpha: float,
    n: int,
    pml_vox: int,
    src_vox: int,
    min_settle: int,
    max_settle: int,
    solver: str = "linear",
    beta: float = 0.0,
    amplitude: float = DRIVE_PA,
    backend: str = "auto",
    harmonics: tuple[int, ...] | None = None,
    dx: float = DX_1D,
    convergence_tol: float = 0.005,
    n_record_periods: int = 2,
):
    """One 1-D CW solve. Imports stay local so `--help` builds no engine."""
    import caustica.solvers as solvers  # noqa: PLC0415
    from caustica import Grid, Medium, PMLSpec  # noqa: PLC0415
    from caustica.materials import water  # noqa: PLC0415
    from caustica.solvers import CWRunSpec  # noqa: PLC0415
    from caustica.sources import plane_cw_source  # noqa: PLC0415

    grid = Grid(shape=(n,), dx=dx, pml=PMLSpec(thickness=pml_vox * dx))
    medium = Medium.homogeneous(grid.shape, water(alpha_np_m=alpha, c=C0, beta=beta))
    source = plane_cw_source(grid, f0=F0, amplitude=amplitude, position_vox=src_vox)
    spec = CWRunSpec(
        min_settle_periods=min_settle,
        max_settle_periods=max_settle,
        convergence_tol=convergence_tol,
        n_record_periods=n_record_periods,
    )
    kwargs: dict[str, Any] = {"backend": backend}
    if harmonics is not None:
        kwargs["harmonics"] = harmonics
    return solvers.get(solver)().run(grid, medium, source, spec, **kwargs)


def measure_planewave(size: Size, *, backend: str = "auto") -> dict:
    """1-D plane wave vs the drive contract, ``k = omega/c``, and ``e^{-alpha x}``.

    Two runs, because the lossless one is the only place the phase speed can
    be read without absorption bending it, and the absorbing one is the only
    place the decay exponent exists. Both are hundredths of a second.
    """
    from caustica.analytic import attenuate  # noqa: PLC0415

    t0 = time.perf_counter()
    lossless = _run_1d(
        alpha=0.0, n=256, pml_vox=24, src_vox=40, min_settle=40, max_settle=90, backend=backend
    )

    amp = np.asarray(lossless.amp[WINDOW_1D], dtype=np.float64)
    realized = float(np.median(amp))

    # Convention p(t) = Re{P e^{-i w t}}: right of the source the outgoing
    # wave is P(x) ~ exp(+i k x), so the per-voxel phase step gives k.
    ph = np.asarray(lossless.phasor[WINDOW_1D])
    k_num = float(np.angle(ph[1:] * np.conj(ph[:-1])).mean() / DX_1D)

    absorbing = _run_1d(
        alpha=ALPHA_NP_M,
        n=288,
        pml_vox=32,
        src_vox=40,
        min_settle=45,
        max_settle=95,
        backend=backend,
    )
    x = np.arange(*WINDOW_1D.indices(288)) * DX_1D
    envelope = np.asarray(absorbing.amp[WINDOW_1D], dtype=np.float64)
    alpha_measured = float(-np.polyfit(x, np.log(envelope), 1)[0])

    # Informational, not gated: the measured envelope against the analytic
    # one anchored at the window's first sample. Its floor is the residual
    # standing ripple off the sponge, which no tolerance in this repository
    # pins, so it is reported rather than judged.
    reference = np.asarray(attenuate(envelope[0], ALPHA_NP_M, x - x[0]), dtype=np.float64)
    envelope_dev = float(np.max(np.abs(envelope - reference) / reference))

    return {
        "backend": lossless.meta.get("backend"),
        "spp": int(lossless.spp),
        "steps_total": int(lossless.steps_total) + int(absorbing.steps_total),
        "p0_pa": DRIVE_PA,
        "realized_pa": realized,
        "amplitude_ratio": realized / DRIVE_PA,
        "k_analytic_rad_m": float(K0),
        "k_measured_rad_m": k_num,
        "alpha_target_np_m": ALPHA_NP_M,
        "alpha_measured_np_m": alpha_measured,
        "envelope_max_rel_dev": envelope_dev,
        "elapsed_s": time.perf_counter() - t0,
    }


def measure_oneill(size: Size, *, backend: str = "auto") -> dict:
    """3-D focused bowl vs O'Neil's closed-form on-axis solution.

    Pressure-injected sources have no closed-form ABSOLUTE amplitude, so the
    comparison is the validated one from the physics gate: normalized axial
    shape (correlation), where the peak lands, and how wide the -6 dB lobe is.
    """
    import caustica.solvers as solvers  # noqa: PLC0415
    from caustica import Grid, Medium, PMLSpec  # noqa: PLC0415
    from caustica.analytic import axial_pressure  # noqa: PLC0415
    from caustica.materials import water  # noqa: PLC0415
    from caustica.solvers import CWRunSpec  # noqa: PLC0415
    from caustica.sources import bowl_cw_source  # noqa: PLC0415

    dx = size.dx_mm * 1e-3
    a, roc = size.aperture_m, size.roc_m
    apex = size.apex_vox

    t0 = time.perf_counter()
    grid = Grid(shape=size.shape, dx=dx, pml=PMLSpec(thickness=size.pml_mm * 1e-3))
    medium = Medium.homogeneous(grid.shape, water(c=C0))
    source = bowl_cw_source(
        grid, f0=F0, amplitude=DRIVE_PA, aperture_radius=a, roc=roc, apex_vox=apex
    )
    spec = CWRunSpec(min_settle_periods=8, max_settle_periods=30, n_record_periods=2)
    res = solvers.get("linear")().run(
        grid, medium, source, spec, backend=backend, reference_point=size.focus_vox
    )
    elapsed = time.perf_counter() - t0

    field = np.asarray(res.amp, dtype=np.float64)
    axial = field[apex[0], apex[1], :]
    z = (np.arange(size.shape[2]) - apex[2]) * dx

    # Never measure inside the sponge: stay 2 voxels clear of where it starts.
    z_pml_limit = (size.shape[2] - grid.pml_vox - 2 - apex[2]) * dx
    sel = (z > 0.5 * roc) & (z < min(1.5 * roc, z_pml_limit))
    p_oneill = np.abs(axial_pressure(z[sel], a, roc, F0, C0))
    corr = float(np.corrcoef(axial[sel] / axial[sel].max(), p_oneill / p_oneill.max())[0, 1])

    # The width needs the full distal crossing: widest window still clear of
    # the sponge. Both profiles are measured the same way, so the comparison
    # is of lobes and not of two different conventions.
    sel_w = (z > 0.5 * roc) & (z < z_pml_limit)
    solver_prof = axial[sel_w] / axial[sel_w].max()
    oneill_prof = np.abs(axial_pressure(z[sel_w], a, roc, F0, C0))
    oneill_prof = oneill_prof / oneill_prof.max()
    width_solver = minus6db_width(z[sel_w], solver_prof)
    width_oneill = minus6db_width(z[sel_w], oneill_prof)

    # O'Neil's true maximum sits slightly proximal of the GEOMETRIC focus, so
    # the peak is graded against the closed form's own peak, not against roc.
    z_true_vox = apex[2] + z[sel_w][int(np.argmax(oneill_prof))] / dx
    peak = np.unravel_index(int(np.argmax(field)), field.shape)

    return {
        "backend": res.meta.get("backend"),
        "shape": list(size.shape),
        "dx_mm": size.dx_mm,
        "aperture_mm": size.aperture_vox * size.dx_mm,
        "roc_mm": size.roc_vox * size.dx_mm,
        "f_number": roc / (2.0 * a),
        "n_axial_samples": int(sel.sum()),
        "axial_corr": corr,
        "peak_vox": [int(v) for v in peak],
        "oneill_peak_z_vox": float(z_true_vox),
        "focus_pos_err_vox": abs(float(peak[2]) - float(z_true_vox)),
        "lateral_pos_err_vox": float(max(abs(int(peak[0]) - apex[0]), abs(int(peak[1]) - apex[1]))),
        "width_solver_mm": width_solver * 1e3,
        "width_oneill_mm": width_oneill * 1e3,
        "steps_total": int(res.steps_total),
        "converged_period": int(res.converged_period),
        "elapsed_s": elapsed,
    }


def measure_linear_limit(size: Size, *, backend: str = "auto") -> dict:
    """westervelt with ``beta = 0`` against the linear solver.

    The only check in this suite whose limit is exactly zero, and it should
    be: in linear water the nonlinear term is switched off entirely, so the
    two solvers run the SAME arithmetic. "Within 1e-6" would quietly tolerate
    a nonlinear buffer leaking a rounding-sized contribution into a medium
    that has no nonlinearity.
    """
    t0 = time.perf_counter()
    common = dict(
        alpha=0.0, n=128, pml_vox=16, src_vox=24, min_settle=20, max_settle=40, backend=backend
    )
    linear = _run_1d(solver="linear", **common)
    westervelt = _run_1d(solver="westervelt", **common)
    return {
        "backend": westervelt.meta.get("backend"),
        "nonlinear_active": bool(westervelt.meta.get("nonlinear_active")),
        "phasor_max_abs_diff_pa": float(
            np.max(np.abs(np.asarray(linear.phasor) - np.asarray(westervelt.phasor)))
        ),
        "pmax_max_abs_diff_pa": float(
            np.max(np.abs(np.asarray(linear.p_max) - np.asarray(westervelt.p_max)))
        ),
        "pmax_scale_pa": float(np.max(np.asarray(linear.p_max))),
        "elapsed_s": time.perf_counter() - t0,
    }


def measure_fubini(size: Size, *, backend: str = "auto") -> dict:
    """Second-harmonic growth vs Fubini, across a range of ``sigma``.

    16 points per wavelength, not the production 8: at 8 ppw the third
    harmonic sits at 2.67 ppw and aliases, inflating A2/A1 by ~10%. The
    resolution note on ``tests/test_westervelt.py`` is the calibration.

    ``sigma = x / x_shock`` needs a source amplitude the grid actually
    realized, not the one that was asked for, so ``p0`` is back-solved once
    from the fundamental near the source through Fubini's own ``B1``.
    """
    from caustica.analytic import fubini_harmonic, shock_distance  # noqa: PLC0415

    ppw = 16.0
    dx = C0 / (F0 * ppw)
    src_vox = 60
    t0 = time.perf_counter()
    res = _run_1d(
        alpha=0.0,
        n=600,
        pml_vox=40,
        src_vox=src_vox,
        min_settle=45,
        max_settle=100,
        solver="westervelt",
        beta=BETA,
        amplitude=2.0e6,
        backend=backend,
        harmonics=(1, 2, 3),
        dx=dx,
        convergence_tol=0.003,
        n_record_periods=2,
    )
    elapsed = time.perf_counter() - t0

    a1 = np.asarray(res.harmonic_amp(1), dtype=np.float64)
    a2 = np.asarray(res.harmonic_amp(2), dtype=np.float64)
    i0 = 90  # sigma ~0.11 there, so the B1 correction below is ~0.2%
    sigma_guess = ((i0 - src_vox) * dx) / shock_distance(a1[i0], F0, C0, beta=BETA)
    p0_eff = float(a1[i0] / fubini_harmonic(1, sigma_guess))
    x_shock = float(shock_distance(p0_eff, F0, C0, beta=BETA))

    stations = []
    for iv in FUBINI_STATIONS:
        sigma = ((iv - src_vox) * dx) / x_shock
        if not FUBINI_SIGMA_BAND[0] <= sigma <= FUBINI_SIGMA_BAND[1]:
            continue
        measured = float(a2[iv] / a1[iv])
        analytic = float(fubini_harmonic(2, sigma) / fubini_harmonic(1, sigma))
        stations.append(
            {
                "index": int(iv),
                "sigma": float(sigma),
                "measured_a2_a1": measured,
                "analytic_a2_a1": analytic,
                "rel_err": abs(measured - analytic) / analytic,
            }
        )

    return {
        "backend": res.meta.get("backend"),
        "nonlinear_active": bool(res.meta.get("nonlinear_active")),
        "ppw": ppw,
        "p0_effective_pa": p0_eff,
        "shock_distance_mm": x_shock * 1e3,
        "stations": stations,
        "steps_total": int(res.steps_total),
        "elapsed_s": elapsed,
    }


# -------------------------------------------------------------- the harness


@dataclass
class Harness:
    """Every way this suite touches the world, in one substitutable object.

    Mirrors :class:`caustica.validation.gpu_gates.Harness` for the same
    reason, turned inside out: there the seam existed because the hardware
    was unavailable, here it exists because a real solve costs seconds and
    the report schema, the exit codes, the SKIP handling and the gate algebra
    should not have to pay for one to be checked. The end-to-end path is
    still exercised for real, once, on the quick size.
    """

    env_report: Callable[[], dict]
    backend: Callable[[], str]
    scenarios: dict[str, Callable[..., dict]]


#: The real scenarios, in report order.
SCENARIOS: dict[str, Callable[..., dict]] = {
    "planewave": measure_planewave,
    "oneill": measure_oneill,
    "linear_limit": measure_linear_limit,
    "fubini": measure_fubini,
}


def default_harness() -> Harness:
    """The real thing: the solvers, on whatever backend ``auto`` resolves to."""
    from caustica import env as env_mod  # noqa: PLC0415
    from caustica.core.backend import get_backend  # noqa: PLC0415

    def _backend() -> str:
        try:
            return str(get_backend("auto").name)
        except Exception:  # noqa: BLE001 - a broken CUDA stack must not lose the run
            return "numpy"

    return Harness(env_report=env_mod.env_report, backend=_backend, scenarios=dict(SCENARIOS))


# ------------------------------------------------------------------ gates


def _num(scenarios: dict[str, dict], scenario: str, key: str) -> float | None:
    """One measured number, or None when the scenario did not produce it.

    None is the whole point: it becomes SKIP downstream, and SKIP never
    counts toward a gate. A scenario that raised has ``error`` set and every
    number it would have produced is treated as absent rather than as zero.
    """
    data = scenarios.get(scenario) or {}
    if data.get("error"):
        return None
    value = data.get(key)
    return None if value is None else float(value)


def evaluate(scenarios: dict[str, dict]) -> list[Gate]:
    """Turn measurements into gate verdicts. Pure — no I/O, no solving."""
    planewave = Gate(
        id="M4.planewave",
        criterion=(
            "1-D plane wave: the realized drive amplitude honours the mass-source "
            "contract, the numerical phase speed matches k = omega/c, and the decay "
            "exponent matches the configured alpha of analytic.attenuate"
        ),
        required=3,
        checks=[
            Check.in_range(
                "planewave: realized amplitude / p0",
                _num(scenarios, "planewave", "amplitude_ratio"),
                *AMPLITUDE_BAND,
            ).citing(AMPLITUDE_SOURCE),
            Check.relative(
                "planewave: phase speed (k)",
                _num(scenarios, "planewave", "k_measured_rad_m"),
                _num(scenarios, "planewave", "k_analytic_rad_m"),
                PHASE_SPEED_TOL_PCT,
                " rad/m",
            ).citing(PHASE_SPEED_SOURCE),
            Check.relative(
                "planewave: absorption exponent",
                _num(scenarios, "planewave", "alpha_measured_np_m"),
                _num(scenarios, "planewave", "alpha_target_np_m"),
                ALPHA_TOL_PCT,
                " Np/m",
            ).citing(ALPHA_SOURCE),
        ],
    )

    oneill = Gate(
        id="M4.oneill",
        criterion=(
            "3-D focused bowl vs analytic.axial_pressure (O'Neil): normalized axial "
            "profile correlation, focal peak position, and -6 dB axial width"
        ),
        required=3,
        checks=[
            Check.at_least(
                "oneill: axial profile correlation r",
                _num(scenarios, "oneill", "axial_corr"),
                ONEILL_CORR_MIN,
            ).citing(ONEILL_AXIAL_SOURCE),
            Check.at_most(
                "oneill: focal peak position error",
                _num(scenarios, "oneill", "focus_pos_err_vox"),
                FOCUS_POS_TOL_VOX,
                " vox",
            ).citing(FOCUS_POS_SOURCE),
            Check.relative(
                "oneill: -6 dB axial width",
                _num(scenarios, "oneill", "width_solver_mm"),
                _num(scenarios, "oneill", "width_oneill_mm"),
                ONEILL_WIDTH_TOL_PCT,
                " mm",
            ).citing(ONEILL_AXIAL_SOURCE),
        ],
    )

    limit_data = scenarios.get("linear_limit") or {}
    nonlinear_active = None if limit_data.get("error") else limit_data.get("nonlinear_active")
    linear_limit = Gate(
        id="M5.linear_limit",
        criterion=(
            "westervelt with beta = 0 is bit-identical to the linear solver, and says "
            "so in its metadata"
        ),
        required=3,
        checks=[
            Check.at_most(
                "linear limit: max |phasor difference|",
                _num(scenarios, "linear_limit", "phasor_max_abs_diff_pa"),
                0.0,
                " Pa",
            ).citing(LINEAR_LIMIT_SOURCE),
            Check.at_most(
                "linear limit: max |p_max difference|",
                _num(scenarios, "linear_limit", "pmax_max_abs_diff_pa"),
                0.0,
                " Pa",
            ).citing(LINEAR_LIMIT_SOURCE),
            Check.happened(
                "linear limit: westervelt reports nonlinear_active",
                nonlinear_active,
                False,
                f"nonlinear_active is {nonlinear_active!r} in linear water (expected False)",
            ).citing(LINEAR_LIMIT_SOURCE),
        ],
    )

    fubini_data = scenarios.get("fubini") or {}
    stations = [] if fubini_data.get("error") else (fubini_data.get("stations") or [])
    fubini = Gate(
        id="M5.fubini",
        criterion=(
            f"second-harmonic ratio A2/A1 within {FUBINI_TOL:.0%} of analytic.fubini_harmonic "
            f"at >= {FUBINI_STATIONS_REQUIRED} pre-shock sigma stations"
        ),
        required=FUBINI_STATIONS_REQUIRED,
        checks=[
            Check.at_most(
                f"fubini: A2/A1 at sigma={row['sigma']:.3f}",
                row.get("rel_err"),
                FUBINI_TOL,
            ).citing(FUBINI_SOURCE)
            for row in stations
        ],
    )
    if not stations:
        # An empty gate is INCOMPLETE by construction, but a reader deserves
        # to see WHY rather than an empty table.
        fubini.checks.append(
            Check(
                "fubini: pre-shock stations",
                "SKIP",
                "the scenario produced no sigma station inside the pre-shock band",
                {"source": FUBINI_SOURCE},
            )
        )

    return [planewave, oneill, linear_limit, fubini]


# ----------------------------------------------------------------- report


def _cell(text: Any) -> str:
    """One Markdown table cell. Escapes the pipes that would split the row.

    Not hypothetical: ``max |phasor difference|`` is the natural name for the
    linear-limit check, and unescaped it turned that row into five columns.
    """
    return str(text).replace("|", "\\|")


def render_markdown(payload: dict) -> str:
    """The human half of the report. The JSON is the machine half."""
    env = payload.get("environment", {})
    scenarios = payload.get("scenarios", {})
    lines = [
        "# caustica analytic validation suite",
        "",
        f"**Overall verdict: {payload['verdict']}** (`{FORMAT}`, generated {payload['generated']})",
        "",
        "Solver output against the closed forms in `caustica.analytic`, measured on "
        "this machine. Every tolerance below is inherited from the physics test that "
        "established it — the `source` column names it.",
        "",
        "| | |",
        "|---|---|",
        f"| backend | {payload.get('backend')} |",
        f"| host | {payload.get('host')} |",
        f"| size | {payload.get('size')} |",
        f"| caustica | {payload.get('caustica')} @ {payload.get('git_commit')} |",
        f"| python / numpy / scipy | {env.get('python')} / {env.get('numpy')} / "
        f"{env.get('scipy')} |",
        f"| platform | {env.get('platform')} |",
        f"| suite runtime | {fmt_num(payload.get('elapsed_s'))} s |",
        "",
        "## Gates",
        "",
        "| gate | verdict | passes / required | criterion |",
        "|---|---|---|---|",
    ]
    for g in payload["gates"]:
        lines.append(
            f"| `{g['id']}` | **{g['verdict']}** | {g['n_pass']} / {g['required']} "
            f"| {g['criterion']} |"
        )

    lines += [
        "",
        "### Every check: measured vs limit",
        "",
        "| check | verdict | measured vs limit | source of the tolerance |",
        "|---|---|---|---|",
    ]
    for g in payload["gates"]:
        for c in g["checks"]:
            lines.append(
                f"| {_cell(c['name'])} | {c['verdict']} | {_cell(c['detail'])} "
                f"| `{c.get('source', '--')}` |"
            )

    pw = scenarios.get("planewave") or {}
    if pw and not pw.get("error"):
        lines += [
            "",
            "## Plane wave (1-D, linear solver)",
            "",
            "| quantity | analytic | measured |",
            "|---|---|---|",
            f"| drive amplitude [Pa] | {fmt_num(pw.get('p0_pa'))} "
            f"| {fmt_num(pw.get('realized_pa'))} |",
            f"| wavenumber k [rad/m] | {fmt_num(pw.get('k_analytic_rad_m'))} "
            f"| {fmt_num(pw.get('k_measured_rad_m'))} |",
            f"| alpha [Np/m] | {fmt_num(pw.get('alpha_target_np_m'))} "
            f"| {fmt_num(pw.get('alpha_measured_np_m'))} |",
            "",
            f"Envelope vs `analytic.attenuate` over the window, worst point: "
            f"{fmt_num(100.0 * (pw.get('envelope_max_rel_dev') or 0.0))}% "
            f"(**informational, not gated** — its floor is the residual standing "
            f"ripple off the sponge, which no tolerance in this repository pins).",
        ]

    on = scenarios.get("oneill") or {}
    if on and not on.get("error"):
        lines += [
            "",
            "## O'Neil bowl (3-D, linear solver)",
            "",
            "| | |",
            "|---|---|",
            f"| grid | {'x'.join(str(s) for s in on.get('shape', []))} @ "
            f"{fmt_num(on.get('dx_mm'))} mm |",
            f"| bowl | aperture radius {fmt_num(on.get('aperture_mm'))} mm, roc "
            f"{fmt_num(on.get('roc_mm'))} mm (f/{fmt_num(on.get('f_number'))}) |",
            f"| axial samples compared | {on.get('n_axial_samples')} |",
            f"| peak voxel | {on.get('peak_vox')} (O'Neil peak at z = "
            f"{fmt_num(on.get('oneill_peak_z_vox'))} vox) |",
            f"| lateral peak offset | {fmt_num(on.get('lateral_pos_err_vox'))} vox |",
            f"| -6 dB axial width | solver {fmt_num(on.get('width_solver_mm'))} mm vs "
            f"O'Neil {fmt_num(on.get('width_oneill_mm'))} mm |",
        ]

    fu = scenarios.get("fubini") or {}
    if fu.get("stations"):
        lines += [
            "",
            "## Fubini second harmonic (1-D, westervelt solver)",
            "",
            f"Realized source amplitude {fmt_num((fu.get('p0_effective_pa') or 0.0) / 1e6)} MPa, "
            f"shock distance {fmt_num(fu.get('shock_distance_mm'))} mm, "
            f"{fmt_num(fu.get('ppw'))} points per wavelength.",
            "",
            "| sigma | A2/A1 measured | A2/A1 Fubini | relative error |",
            "|---|---|---|---|",
        ]
        for row in fu["stations"]:
            lines.append(
                f"| {fmt_num(row['sigma'])} | {fmt_num(row['measured_a2_a1'])} "
                f"| {fmt_num(row['analytic_a2_a1'])} | {100.0 * row['rel_err']:.2f}% |"
            )

    lines += ["", "## Scenario runtimes", "", "| scenario | seconds | note |", "|---|---|---|"]
    for name, data in scenarios.items():
        lines.append(
            f"| {name} | {fmt_num(data.get('elapsed_s'))} | {_cell(data.get('error', ''))} |"
        )

    notes = payload.get("notes") or []
    if notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    lines += [
        "",
        "---",
        "",
        "Reproduce: `python -m caustica.validation run-analytic`. Every scenario is "
        "generated by the library (homogeneous water, library-built sources) and "
        "graded against `caustica.analytic`; no external dataset is involved.",
        "",
    ]
    return "\n".join(lines)


def write_report(outdir: Path, payload: dict) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "analytic.json"
    md_path = outdir / "REPORT.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, md_path


# ------------------------------------------------------------------ suite


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower() or "cpu"


def analytic_suite(
    *,
    out: str | Path | None = None,
    size: str = "full",
    harness: Harness | None = None,
    log: Callable[[str], None] = print,
) -> tuple[int, dict]:
    """Run the whole suite. Returns ``(exit_code, report_payload)``."""
    from caustica import __version__ as version  # noqa: PLC0415
    from caustica.env import git_commit  # noqa: PLC0415

    preset = size_preset(size)
    harness = harness or default_harness()
    notes: list[str] = []
    started = time.perf_counter()

    env = harness.env_report()
    log("--- environment ----------------------------------------------")
    for key, value in env.items():
        log(f"  {key}: {value}")

    backend = harness.backend()
    host = platform.node()
    stamp = datetime.now(timezone.utc)
    root = Path(out) if out is not None else Path("benchmarks/reports/analytic")
    outdir = root / f"{_slug(backend or host)}-{stamp.strftime('%Y%m%d-%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"\nbackend: {backend}   size: {preset.name}   report folder: {outdir}")

    results: dict[str, dict] = {}
    for name, measure in harness.scenarios.items():
        log(f"\n=== {name} ===")
        t0 = time.perf_counter()
        try:
            results[name] = measure(preset, backend=backend)
        except Exception as exc:  # noqa: BLE001 - one bad scenario must not lose the rest
            # Recorded, never swallowed: every check it feeds becomes SKIP,
            # its gate comes out INCOMPLETE, and the suite exits non-zero.
            message = f"{type(exc).__name__}: {exc}"
            log(f"  raised {message}")
            notes.append(f"scenario {name!r} raised {message}; its gate cannot close")
            results[name] = {"error": message, "elapsed_s": time.perf_counter() - t0}
            continue
        for key, value in sorted(results[name].items()):
            if key != "stations":
                log(f"  {key}: {value}")

    gates = evaluate(results)
    payload = {
        "format": FORMAT,
        "generated": stamp.isoformat(timespec="seconds"),
        "caustica": version,
        "git_commit": git_commit(),
        "backend": backend,
        "host": host,
        "size": preset.name,
        "size_preset": preset.as_dict(),
        "environment": env,
        "scenarios": results,
        "gates": [g.as_dict() for g in gates],
        "verdict": overall_verdict(gates),
        "elapsed_s": time.perf_counter() - started,
        "notes": notes,
    }

    json_path, md_path = write_report(outdir, payload)
    log("\n--- verdict --------------------------------------------------")
    for g in gates:
        log(f"  {g.verdict:<11} {g.id}: {g.criterion}")
    log(f"\noverall: {payload['verdict']}  ({payload['elapsed_s']:.1f} s)")
    log(f"report: {md_path}")
    log(f"json:   {json_path}")
    return (EXIT_OK if payload["verdict"] == "PASS" else EXIT_FAILED), payload
