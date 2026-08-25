"""M18 gate: Pennes bioheat + CEM43 dose, against closed forms.

Every physics test here carries its analytic solution in the test body, and
every one of them is measured against the milestone's 2% bar:

* Gaussian spreading      sigma^2(t) = sigma0^2 + 2 D t
* instantaneous point     T = E/(rho C) (4 pi D t)^-3/2 exp(-r^2/(4 D t))
* perfusion relaxation    T = T_a + (T0 - T_a) exp(-w rho_b C_b t / (rho C))
* perfused steady state   T_ss = T_a + Q/(w rho_b C_b)
* two-layer conduction    dT across a face = flux dx / k_harmonic

The remaining tests are about the refusals and the traps: an unstable dt, a
material with no thermal properties, a NaN state, and the dose-from-final-T
mistake that makes any transient look cold.

Everything runs on small grids on the CPU backend in a couple of seconds;
the GPU path is the same source (``backend.xp``), exercised here through
``get_backend("numpy")``.
"""

from __future__ import annotations

import numpy as np
import pytest

from caustica.core.backend import cupy_available
from caustica.materials import Material, MaterialDB, water
from caustica.medium import Medium
from caustica.sensors import HeatingSource
from caustica.thermal.dose import (
    CEM43_R_ABOVE,
    CEM43_R_BELOW,
    ITRUSST_CEM43_LIMITS,
    ITRUSST_DELTA_T_LIMIT_C,
    ITRUSST_SOURCE,
    cem43_minutes,
    cem43_rate,
)
from caustica.thermal.pennes import (
    ARTERIAL_TEMPERATURE_C,
    BLOOD_DENSITY,
    BLOOD_SPECIFIC_HEAT,
    PennesSolver,
    ThermalDivergedError,
    ThermalResult,
    ThermalStabilityError,
)
from caustica.thermal.properties import ThermalMedium, ThermalPropertyError

#: Generic soft tissue with the thermal fields filled in (k, C from the IT'IS
#: ranges; perfusion is switched on per test because "no perfusion" must be
#: stated, not defaulted).
TISSUE = Material(
    name="generic soft tissue",
    alpha_np_m=6.0,
    rho=1050.0,
    c=1540.0,
    beta=4.5,
    thermal_conductivity=0.5,
    specific_heat=3600.0,
    perfusion_rate=0.0,
)
D_TISSUE = TISSUE.thermal_conductivity / (TISSUE.rho * TISSUE.specific_heat)  # 1.3228e-7 m^2/s
RHO_C = TISSUE.rho * TISSUE.specific_heat
LIMIT = 0.02  # the milestone's 2% bar


def _gate(label: str, measured: float, limit: float = LIMIT) -> None:
    """Assert a relative error against its limit, printing the measurement.

    Printed rather than only asserted so ``pytest -s`` is a report of how
    accurate the scheme actually is, not just a row of dots.
    """
    print(f"GATE {label}: rel err {measured * 100:.4f}%  (limit {limit * 100:.2f}%)")
    assert measured < limit, f"{label}: {measured * 100:.4f}% exceeds {limit * 100:.2f}%"


def _radius_squared(shape, dx, centre):
    """|r|^2 [m^2] from ``centre`` (in voxel units) on a grid of ``shape``."""
    axes = [((np.arange(n) - c) * dx) ** 2 for n, c in zip(shape, centre, strict=True)]
    out = np.zeros(shape, dtype=np.float64)
    for ax, a in enumerate(axes):
        out = out + a.reshape([-1 if i == ax else 1 for i in range(len(shape))])
    return out


def _perfusion_rate_per_second(w_b: float) -> float:
    """The exponential rate w_b rho_b C_b / (rho C) [1/s] of the Pennes sink."""
    return w_b * BLOOD_DENSITY * BLOOD_SPECIFIC_HEAT / RHO_C


# --------------------------------------------------------------------------
# The analytic diffusion gates
# --------------------------------------------------------------------------


def test_a_gaussian_hot_spot_spreads_with_sigma_squared_equal_to_two_d_t():
    """Exact solution of the heat equation for a Gaussian initial condition.

    In d dimensions a Gaussian stays Gaussian: its variance grows as
    sigma^2(t) = sigma0^2 + 2 D t and its peak falls as (sigma0/sigma)^d,
    which conserves the deposited energy. Body temperature is a steady state
    of the source-free equation, so the whole profile rides on 37 C.
    """
    n, dx, sigma0, rise = 64, 1e-3, 5e-3, 10.0
    med = ThermalMedium.homogeneous((n,) * 3, TISSUE, dx)
    r2 = _radius_squared((n,) * 3, dx, ((n - 1) / 2.0,) * 3)
    t0 = (ARTERIAL_TEMPERATURE_C + rise * np.exp(-r2 / (2 * sigma0**2))).astype(np.float32)

    res = PennesSolver(backend="numpy").solve(t0, None, med, dt=1.0, n_steps=90, record_every=30)

    assert res.times == [0.0, 30.0, 60.0, 90.0]
    for t_s, sample in zip(res.times[1:], res.samples[1:], strict=True):
        s2 = sigma0**2 + 2 * D_TISSUE * t_s
        exact = ARTERIAL_TEMPERATURE_C + rise * (sigma0**2 / s2) ** 1.5 * np.exp(-r2 / (2 * s2))
        err = np.abs(sample - exact).max() / (exact.max() - ARTERIAL_TEMPERATURE_C)
        _gate(f"gaussian diffusion t={t_s:.0f}s (sigma {np.sqrt(s2) * 1e3:.2f} mm)", err)


def test_an_instantaneous_point_source_spreads_as_the_greens_function():
    """T(r,t) = E/(rho C) (4 pi D t)^(-3/2) exp(-r^2/(4 D t)).

    Away from the singular first steps only: one voxel of energy is a BOX,
    not a delta, and a box carries an extra variance dx^2/12 per axis. At
    t = 30 s that is 1.0% of 2 D t and the peak (which scales as
    variance^-3/2) is 1.5% low — visible against a 2% bar. The test
    therefore grades the delta form from t = 60 s on, and separately checks
    that correcting the analytic variance for the box removes most of the
    residual, which is what proves the gap is the initial condition's shape
    and not the scheme.
    """
    n, dx = 64, 1e-3
    med = ThermalMedium.homogeneous((n,) * 3, TISSUE, dx)
    ctr = n // 2
    energy_j = 1.0
    t0 = np.full((n,) * 3, ARTERIAL_TEMPERATURE_C, np.float32)
    t0[ctr, ctr, ctr] += energy_j / (RHO_C * dx**3)
    r2 = _radius_squared((n,) * 3, dx, (ctr,) * 3)

    res = PennesSolver(backend="numpy").solve(t0, None, med, dt=1.0, n_steps=120, record_every=30)

    for t_s, sample in zip(res.times, res.samples, strict=True):
        if t_s == 0.0:
            continue
        rise = sample.astype(np.float64) - ARTERIAL_TEMPERATURE_C
        for label, variance, graded in (
            ("delta", 2 * D_TISSUE * t_s, t_s >= 60.0),
            ("box-corrected", 2 * D_TISSUE * t_s + dx**2 / 12.0, True),
        ):
            green = energy_j / RHO_C / (2 * np.pi * variance) ** 1.5 * np.exp(-r2 / (2 * variance))
            near = r2 <= (3 * np.sqrt(variance)) ** 2  # inside 3 sigma
            err = np.abs(rise - green)[near].max() / green.max()
            if graded:
                _gate(
                    f"point source {label} t={t_s:.0f}s", err, LIMIT if label == "delta" else 0.01
                )


def test_perfusion_alone_relaxes_exponentially_towards_arterial_temperature():
    """T(t) = T_a + (T0 - T_a) exp(-w_b rho_b C_b t / (rho C)).

    A uniform field has no conduction at all (every face flux is zero), so
    this isolates the perfusion term and its blood constants.
    """
    w_b = 0.002
    tissue = TISSUE.model_copy(update={"perfusion_rate": w_b})
    med = ThermalMedium.homogeneous((8, 8, 8), tissue, 1e-3)
    rate = _perfusion_rate_per_second(w_b)
    t_start = 45.0
    n_steps = int(round(1.0 / rate))  # one time constant

    res = PennesSolver(backend="numpy").solve(
        np.full((8, 8, 8), t_start, np.float32),
        None,
        med,
        dt=1.0,
        n_steps=n_steps,
        record_every=n_steps // 4,
    )

    for t_s, sample in zip(res.times[1:], res.samples[1:], strict=True):
        exact = ARTERIAL_TEMPERATURE_C + (t_start - ARTERIAL_TEMPERATURE_C) * np.exp(-rate * t_s)
        err = abs(float(sample.mean()) - exact) / (exact - ARTERIAL_TEMPERATURE_C)
        _gate(f"perfusion relaxation t={t_s:.0f}s", err)
    assert res.temperature.min() > ARTERIAL_TEMPERATURE_C  # cools towards, never past


def test_constant_heating_against_perfusion_settles_at_the_analytic_steady_state():
    """T_ss = T_a + Q / (w_b rho_b C_b) — the classic perfusion-limited rise.

    Run for eight time constants, which leaves exp(-8) = 0.03% of the
    approach outstanding, so what the 2% bar actually grades is the scheme.
    """
    w_b = 0.002
    q_w_m3 = 20000.0
    tissue = TISSUE.model_copy(update={"perfusion_rate": w_b})
    med = ThermalMedium.homogeneous((8, 8, 8), tissue, 1e-3)
    rate = _perfusion_rate_per_second(w_b)
    t_ss = ARTERIAL_TEMPERATURE_C + q_w_m3 / (w_b * BLOOD_DENSITY * BLOOD_SPECIFIC_HEAT)

    res = PennesSolver(backend="numpy").solve(
        np.full((8, 8, 8), ARTERIAL_TEMPERATURE_C, np.float32),
        q_w_m3,
        med,
        dt=1.0,
        n_steps=int(round(8.0 / rate)),
    )

    got = float(res.temperature.mean())
    err = abs(got - t_ss) / (t_ss - ARTERIAL_TEMPERATURE_C)
    _gate(f"perfused steady state (T_ss = {t_ss:.3f} C, got {got:.3f} C)", err)


def test_a_conductivity_jump_obeys_the_series_resistance_of_the_two_layers():
    """Steady conduction through k=0.5 then k=2.0: dT_face = flux dx / k_face.

    With harmonic-mean face conductivities the discrete steady state equals
    the continuum solution AT CELL CENTRES exactly, so this grades the
    interface treatment at a much tighter bar than 2%. It is also the test
    that pins the float32 reference offset: carrying this 0.039 C profile in
    absolute degrees on a 37 C base loses 8.4% of it to cancellation, while
    the rise-above-reference state keeps it to 0.01%.
    """
    n, dx, power = 32, 1e-3, 1000.0
    k = np.where(np.arange(n) < n // 2, 0.5, 2.0).astype(np.float32)
    med = ThermalMedium(
        k=k,
        rho=np.full(n, TISSUE.rho, np.float32),
        specific_heat=np.full(n, TISSUE.specific_heat, np.float32),
        perfusion=np.zeros(n, np.float32),
        dx=dx,
    )
    q = np.zeros(n, np.float32)
    q[0], q[-1] = power, -power  # heat in at one end, out at the other
    t0 = np.full(n, ARTERIAL_TEMPERATURE_C, np.float32)

    res = PennesSolver(backend="numpy").solve(t0, q, med, dt=0.9, n_steps=10000)

    got = res.temperature.astype(np.float64)
    k_face = 2 * k[:-1].astype(np.float64) * k[1:] / (k[:-1] + k[1:])
    flux = power * dx  # W/m^2 crossing every interior face
    exact = np.concatenate([[0.0], -np.cumsum(flux * dx / k_face)])
    exact = exact - exact.mean() + got.mean()  # insulated ends fix only the profile
    span = exact.max() - exact.min()
    _gate("two-layer conduction (harmonic-mean faces)", np.abs(got - exact).max() / span, 0.001)


def test_the_reference_offset_is_what_keeps_that_profile_accurate():
    """The same solve with a deliberately distant reference loses precision.

    Not a physics test — a pin on the float32 design decision. If someone
    removes the rise-above-reference state, the gate above fails; if someone
    removes the PARAMETER, this one does.
    """
    n, dx, power = 32, 1e-3, 1000.0
    k = np.where(np.arange(n) < n // 2, 0.5, 2.0).astype(np.float32)
    med = ThermalMedium(
        k=k,
        rho=np.full(n, TISSUE.rho, np.float32),
        specific_heat=np.full(n, TISSUE.specific_heat, np.float32),
        perfusion=np.zeros(n, np.float32),
        dx=dx,
    )
    q = np.zeros(n, np.float32)
    q[0], q[-1] = power, -power
    t0 = np.full(n, ARTERIAL_TEMPERATURE_C, np.float32)
    args = (t0, q, med, 0.9, 10000)

    good = PennesSolver(backend="numpy").solve(*args).temperature
    bad = PennesSolver(backend="numpy", reference_temperature_c=0.0).solve(*args).temperature
    good_span = float(good.max() - good.min())
    bad_span = float(bad.max() - bad.min())
    assert good_span > bad_span, "the far reference should LOSE amplitude, not gain it"
    assert (good_span - bad_span) / good_span > 0.01


def test_insulated_walls_conserve_the_deposited_energy():
    """No source, no perfusion, zero-flux walls: sum(rho C T) cannot change.

    The conduction term is assembled as face fluxes precisely so that what
    leaves one cell enters its neighbour — this is that guarantee, measured.
    """
    n = 24
    med = ThermalMedium.homogeneous((n,) * 3, TISSUE, 1e-3)
    rng = np.random.default_rng(0)
    t0 = (ARTERIAL_TEMPERATURE_C + 10.0 * rng.random((n,) * 3)).astype(np.float32)

    res = PennesSolver(backend="numpy").solve(t0, None, med, dt=1.0, n_steps=200)

    before = float(t0.astype(np.float64).sum())
    after = float(res.temperature.astype(np.float64).sum())
    assert abs(after - before) / before < 1e-6
    assert res.temperature.std() < 0.1 * t0.std()  # and it did diffuse


# --------------------------------------------------------------------------
# Time stepping: stability, sub-stepping, recording
# --------------------------------------------------------------------------


def test_the_stable_step_matches_the_textbook_diffusion_number():
    """For uniform k and no perfusion the bound is dx^2 / (2 ndim D)."""
    for ndim in (1, 2, 3):
        med = ThermalMedium.homogeneous((16,) * ndim, TISSUE, 1e-3)
        expected = 1e-6 / (2 * ndim * D_TISSUE)
        assert PennesSolver(backend="numpy").stable_dt(med) == pytest.approx(expected, rel=1e-4)


def test_an_unstable_time_step_is_refused_with_the_number_it_must_not_exceed():
    """An explicit diffusion step past the limit grows a plausible-looking
    checkerboard rather than failing loudly, so it is refused up front."""
    med = ThermalMedium.homogeneous((8, 8, 8), TISSUE, 1e-3)
    with pytest.raises(ThermalStabilityError) as exc:
        PennesSolver(backend="numpy").solve(
            np.full((8, 8, 8), 37.0, np.float32), None, med, dt=10.0, n_steps=5
        )
    msg = str(exc.value)
    assert "stability bound" in msg
    assert "1.26" in msg  # the actual number, not just a complaint
    assert "on_unstable='substep'" in msg  # and every way out
    assert "coarsen dx" in msg


def test_sub_stepping_reproduces_the_small_step_solution():
    """``on_unstable='substep'`` splits the step internally and keeps the
    caller's sampling times — the answer must be the small-step answer.

    The initial condition is a resolved Gaussian, not a single hot voxel: a
    one-voxel spike excites the modes at the grid's Nyquist, where two
    different forward-Euler steps legitimately disagree by O(1) on an
    amplitude that is already decaying to nothing. Comparing time steps
    there measures the stiffest mode, not the solver.
    """
    n, sigma = 16, 3e-3
    med = ThermalMedium.homogeneous((n,) * 3, TISSUE, 1e-3)
    r2 = _radius_squared((n,) * 3, 1e-3, ((n - 1) / 2.0,) * 3)
    t0 = (ARTERIAL_TEMPERATURE_C + 10.0 * np.exp(-r2 / (2 * sigma**2))).astype(np.float32)

    fine = PennesSolver(backend="numpy").solve(t0, None, med, dt=1.0, n_steps=100)
    coarse = PennesSolver(backend="numpy", on_unstable="substep").solve(
        t0, None, med, dt=5.0, n_steps=20
    )

    assert coarse.substeps == 4
    assert coarse.t_end_s == fine.t_end_s
    rise = float((fine.temperature - ARTERIAL_TEMPERATURE_C).max())
    _gate(
        "sub-stepped vs small-step solution",
        float(np.abs(coarse.temperature - fine.temperature).max()) / rise,
        0.01,
    )
    assert np.array_equal(
        t0, (ARTERIAL_TEMPERATURE_C + 10.0 * np.exp(-r2 / (2 * sigma**2))).astype(np.float32)
    ), "the solver mutated the caller's T0"


def test_recording_keeps_the_first_and_last_state_and_their_times():
    med = ThermalMedium.homogeneous((8, 8), TISSUE, 1e-3)
    t0 = np.full((8, 8), 40.0, np.float32)
    res = PennesSolver(backend="numpy").solve(t0, None, med, dt=1.0, n_steps=7, record_every=3)
    assert res.times == [0.0, 3.0, 6.0, 7.0]
    assert len(res.samples) == 4
    assert np.array_equal(res.samples[0], t0)
    assert np.array_equal(res.samples[-1], res.temperature)
    assert PennesSolver(backend="numpy").solve(t0, None, med, 1.0, 7).samples == []


def test_the_max_temperature_map_remembers_the_peak_not_the_endpoint():
    """A sonication's peak is mid-run; a safety map read off the final state
    would miss it entirely."""
    med = ThermalMedium.homogeneous((8, 8), TISSUE, 1e-3)
    t0 = np.full((8, 8), 37.0, np.float32)
    t0[4, 4] = 80.0
    res = PennesSolver(backend="numpy").solve(t0, None, med, dt=1.0, n_steps=50)
    assert res.peak_temperature_c == pytest.approx(80.0)
    assert float(res.temperature.max()) < 45.0


# --------------------------------------------------------------------------
# CEM43
# --------------------------------------------------------------------------


def test_the_cem43_unit_definitions_are_exact():
    """43 C for a minute IS one CEM43 minute; each degree is a factor of two
    above the break and a factor of four below it."""
    assert float(cem43_minutes(43.0, 60.0)) == 1.0
    assert float(cem43_minutes(44.0, 60.0)) == 2.0
    assert float(cem43_minutes(45.0, 60.0)) == 4.0
    assert float(cem43_minutes(42.0, 60.0)) == 0.25
    assert float(cem43_minutes(41.0, 60.0)) == 0.0625
    assert float(cem43_minutes(43.0, 120.0)) == 2.0
    assert float(cem43_rate(43.0)) == 1.0
    assert (CEM43_R_ABOVE, CEM43_R_BELOW) == (0.5, 0.25)


def test_the_cem43_rate_is_elementwise_and_dtype_preserving():
    temps = np.array([37.0, 43.0, 47.0], dtype=np.float32)
    rates = cem43_rate(temps)
    assert rates.dtype == np.float32
    assert rates[1] == 1.0
    assert rates[2] == pytest.approx(16.0)


def test_dose_accumulated_during_a_transient_matches_the_analytic_integral():
    """Cooling from 47 C with perfusion has a closed-form T(t), so its dose
    has a closed-form integral — the solver's running sum must match it."""
    w_b = 0.002
    tissue = TISSUE.model_copy(update={"perfusion_rate": w_b})
    med = ThermalMedium.homogeneous((4, 4, 4), tissue, 1e-3)
    rate = _perfusion_rate_per_second(w_b)
    dt, n_steps = 0.5, 600

    res = PennesSolver(backend="numpy").solve(
        np.full((4, 4, 4), 47.0, np.float32), None, med, dt, n_steps, dose=True
    )

    t_grid = np.linspace(0.0, dt * n_steps, 200001)
    analytic = (
        np.trapezoid(cem43_rate(ARTERIAL_TEMPERATURE_C + 10.0 * np.exp(-rate * t_grid)), t_grid)
        / 60.0
    )
    err = abs(res.peak_dose_cem43 - analytic) / analytic
    _gate(f"CEM43 during a transient ({res.peak_dose_cem43:.3f} vs {analytic:.3f} min)", err)


def test_dose_from_the_final_temperature_is_a_different_and_wrong_number():
    """The trap this API exists to close.

    A voxel heated from 37 C to 43.3 C over two minutes accrued a fraction
    of a CEM43 minute; judging it by its final temperature held for the
    whole run claims seven times more. Neither is a rounding error of the
    other — dose is an integral over the history.
    """
    med = ThermalMedium.homogeneous((4, 4, 4), TISSUE, 1e-3)
    dt, n_steps = 0.5, 240

    res = PennesSolver(backend="numpy").solve(
        np.full((4, 4, 4), ARTERIAL_TEMPERATURE_C, np.float32),
        2.0e5,
        med,
        dt,
        n_steps,
        dose=True,
    )

    final_t = float(res.temperature.max())
    naive = float(cem43_minutes(final_t, dt * n_steps))
    assert final_t == pytest.approx(43.35, abs=0.05)
    assert res.peak_dose_cem43 == pytest.approx(0.35, abs=0.02)
    assert naive / res.peak_dose_cem43 > 5.0


def test_dose_carries_across_a_sonication_and_its_cooling_phase():
    """A duty cycle is several solves; the dose is one number across them."""
    med = ThermalMedium.homogeneous((4, 4, 4), TISSUE, 1e-3)
    t0 = np.full((4, 4, 4), ARTERIAL_TEMPERATURE_C, np.float32)
    hot = PennesSolver(backend="numpy").solve(t0, 2.0e5, med, 0.5, 240, dose=True)
    cool = PennesSolver(backend="numpy").solve(
        hot.temperature, None, med, 0.5, 240, dose=True, dose0=hot.dose_cem43
    )
    assert cool.peak_dose_cem43 > hot.peak_dose_cem43
    fresh = PennesSolver(backend="numpy").solve(hot.temperature, None, med, 0.5, 240, dose=True)
    assert cool.peak_dose_cem43 == pytest.approx(
        hot.peak_dose_cem43 + fresh.peak_dose_cem43, rel=1e-5
    )


def test_a_dose_is_only_computed_when_it_is_asked_for():
    med = ThermalMedium.homogeneous((4, 4), TISSUE, 1e-3)
    res = PennesSolver(backend="numpy").solve(np.full((4, 4), 50.0, np.float32), None, med, 1.0, 5)
    assert res.dose_cem43 is None
    assert res.peak_dose_cem43 is None
    assert res.meta["dose_accumulated"] is False


def test_carrying_a_dose_in_without_asking_for_one_out_is_refused():
    """``dose0`` with ``dose=False`` would silently drop the dose so far."""
    med = ThermalMedium.homogeneous((4, 4), TISSUE, 1e-3)
    with pytest.raises(ValueError, match="dose0 was given but dose=False"):
        PennesSolver(backend="numpy").solve(
            np.full((4, 4), 50.0, np.float32),
            None,
            med,
            1.0,
            5,
            dose0=np.zeros((4, 4), np.float32),
        )


def test_the_itrusst_thresholds_are_recorded_with_their_source():
    """Phase 1 ships the numbers as data; the report table is phase 2."""
    assert ITRUSST_CEM43_LIMITS == {"brain": 2.0, "bone": 16.0, "skin": 21.0}
    assert ITRUSST_DELTA_T_LIMIT_C == 2.0
    assert "Brain Stimulation (2025)" in ITRUSST_SOURCE


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_a_material_without_thermal_fields_is_refused_by_name():
    """``Material``'s thermal fields are Optional because acoustics does not
    read them. A thermal solve does, and guessing them decides the answer."""
    with pytest.raises(ThermalPropertyError) as exc:
        ThermalMedium.homogeneous((4, 4), water(), 1e-3)
    msg = str(exc.value)
    assert "'water'" in msg
    assert "thermal_conductivity" in msg and "specific_heat" in msg
    assert "perfusion_rate=0.0" in msg  # the fix for a non-perfused medium


def test_an_id_map_with_one_incomplete_tissue_names_every_offender():
    db = MaterialDB(
        materials={
            1: TISSUE,
            2: Material(name="Bone", alpha_np_m=50.0, rho=1900.0, c=3000.0, beta=4.0),
        }
    )
    id_map = np.array([[1, 1], [2, 2]], dtype=np.int32)
    with pytest.raises(ThermalPropertyError) as exc:
        ThermalMedium.from_id_map(id_map, db, 1e-3)
    assert "'Bone'" in str(exc.value)
    assert "id 2" in str(exc.value)


def test_a_thermal_medium_mirrors_the_acoustic_tissue_layout():
    db = MaterialDB(
        materials={
            1: TISSUE,
            2: TISSUE.model_copy(
                update={"name": "Fat", "rho": 950.0, "thermal_conductivity": 0.21}
            ),
        }
    )
    id_map = np.array([[1, 2], [2, 1]], dtype=np.int32)
    acoustic = Medium.from_id_map(id_map, db)
    thermal = ThermalMedium.from_medium(acoustic, db, 1e-3)
    assert thermal.k[0, 1] == np.float32(0.21)
    assert thermal.rho[0, 1] == np.float32(950.0)
    assert np.array_equal(thermal.rho, acoustic.rho)
    with pytest.raises(ValueError, match="no id_map"):
        ThermalMedium.from_medium(Medium.homogeneous((4, 4), TISSUE), db, 1e-3)


def test_impossible_thermal_volumes_are_refused():
    ones = np.ones((4, 4), np.float32)
    with pytest.raises(ValueError, match="must be > 0"):
        ThermalMedium(k=ones * 0.0, rho=ones, specific_heat=ones, perfusion=ones, dx=1e-3)
    with pytest.raises(ValueError, match="heat SOURCE"):
        ThermalMedium(k=ones, rho=ones, specific_heat=ones, perfusion=-ones, dx=1e-3)
    with pytest.raises(ValueError, match="disagree in shape"):
        ThermalMedium(
            k=ones, rho=np.ones((4, 5), np.float32), specific_heat=ones, perfusion=ones, dx=1e-3
        )
    with pytest.raises(ValueError, match="positive finite spacing"):
        ThermalMedium(k=ones, rho=ones, specific_heat=ones, perfusion=ones, dx=0.0)


def test_a_non_finite_initial_temperature_is_refused_before_stepping():
    med = ThermalMedium.homogeneous((8, 8), TISSUE, 1e-3)
    t0 = np.full((8, 8), 37.0, np.float32)
    t0[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        PennesSolver(backend="numpy").solve(t0, None, med, 1.0, 5)


def test_a_state_that_stops_being_finite_raises_instead_of_returning_nan():
    """A real divergence through the real path (cf. test_divergence_guard).

    A near-insulating medium has a huge stable step, and a heat source
    float32 cannot carry then overflows the state within a few hundred of
    them. The run must fail, not hand back a NaN dose map.
    """
    insulator = TISSUE.model_copy(update={"name": "near-insulator", "thermal_conductivity": 1e-6})
    med = ThermalMedium.homogeneous((8, 8, 8), insulator, 1e-3)
    solver = PennesSolver(backend="numpy")
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ThermalDivergedError) as exc:
            solver.solve(
                np.full((8, 8, 8), 37.0, np.float32), 1e37, med, solver.stable_dt(med), 300
            )
    msg = str(exc.value)
    assert "stopped being finite" in msg
    assert "outer step" in msg  # says WHERE
    assert "stable_dt" in msg  # and what to check


def test_an_overflowing_dose_is_caught_by_the_same_guard():
    """R^(43-T) overflows float32 above ~170 C; an inf dose must not be
    returned as a dose."""
    med = ThermalMedium.homogeneous((4, 4, 4), TISSUE, 1e-3)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ThermalDivergedError, match="CEM43 dose"):
            PennesSolver(backend="numpy").solve(
                np.full((4, 4, 4), 37.0, np.float32), 2.0e6, med, 1.0, 400, dose=True
            )


def test_a_heating_source_from_another_grid_is_refused():
    """Q on a 0.5 mm grid integrated on a 1 mm grid is silently wrong."""
    med = ThermalMedium.homogeneous((4, 4), TISSUE, 1e-3)
    heat = HeatingSource.from_phasors(
        {1: np.full((4, 4), 1e6, np.complex64)}, alpha=6.0, rho=1050.0, c=1540.0, dx=0.5e-3
    )
    with pytest.raises(ValueError, match="different grids"):
        PennesSolver(backend="numpy").solve(np.full((4, 4), 37.0, np.float32), heat, med, 1.0, 2)


def test_a_heating_source_covering_only_the_record_region_says_how_to_place_it():
    med = ThermalMedium.homogeneous((6, 4), TISSUE, 1e-3)
    heat = HeatingSource.from_phasors(
        {1: np.full((2, 4), 1e6, np.complex64)},
        alpha=6.0,
        rho=1050.0,
        c=1540.0,
        dx=1e-3,
        region=(slice(2, 4), slice(0, 4)),
    )
    t0 = np.full((6, 4), 37.0, np.float32)
    with pytest.raises(ValueError, match=r"embed\(\(6, 4\)\)"):
        PennesSolver(backend="numpy").solve(t0, heat, med, 1.0, 2)
    res = PennesSolver(backend="numpy").solve(t0, heat.embed((6, 4)), med, 1.0, 2)
    assert res.temperature[3, 0] > res.temperature[0, 0]


def test_the_acoustic_medium_is_not_a_thermal_medium():
    with pytest.raises(TypeError, match="ThermalMedium"):
        PennesSolver(backend="numpy").solve(
            np.full((4, 4), 37.0, np.float32),
            None,
            Medium.homogeneous((4, 4), TISSUE),
            1.0,
            2,
        )


def test_nonsense_run_parameters_are_refused():
    med = ThermalMedium.homogeneous((4, 4), TISSUE, 1e-3)
    t0 = np.full((4, 4), 37.0, np.float32)
    solver = PennesSolver(backend="numpy")
    with pytest.raises(ValueError, match="dt must be"):
        solver.solve(t0, None, med, 0.0, 5)
    with pytest.raises(ValueError, match="n_steps"):
        solver.solve(t0, None, med, 1.0, 0)
    with pytest.raises(ValueError, match="record_every"):
        solver.solve(t0, None, med, 1.0, 5, record_every=0)
    with pytest.raises(ValueError, match="non-finite"):
        solver.solve(t0, np.full((4, 4), np.inf, np.float32), med, 1.0, 5)
    with pytest.raises(ValueError, match="but the thermal medium is"):
        solver.solve(t0, np.zeros((3, 3), np.float32), med, 1.0, 5)
    with pytest.raises(ValueError, match="temperature0 has shape"):
        solver.solve(np.zeros((3, 3), np.float32), None, med, 1.0, 5)


def test_a_solver_that_integrates_unstably_cannot_be_configured():
    with pytest.raises(ValueError, match="no option that integrates an unstable step"):
        PennesSolver(on_unstable="ignore")
    with pytest.raises(ValueError, match="boundary must be"):
        PennesSolver(boundary="periodic")


# --------------------------------------------------------------------------
# Backend contract
# --------------------------------------------------------------------------


def _dose_setup():
    med = ThermalMedium.homogeneous(
        (8, 8, 8), TISSUE.model_copy(update={"perfusion_rate": 0.001}), 1e-3
    )
    t0 = np.full((8, 8, 8), 37.0, np.float32)
    t0[4, 4, 4] = 60.0
    return med, t0


def test_the_named_cpu_backend_and_auto_agree_bit_for_bit(no_gpu):
    """The solver never touches numpy for state maths — it uses backend.xp.

    With no CUDA device ``auto`` resolves to numpy, so the two runs are the
    same arithmetic on the same library and must agree to the last bit. The
    fixture pins that environment rather than assuming it: on a CUDA box
    ``auto`` picks cupy and this stops being a statement about dispatch at
    all (that comparison is the next test, and it is not bit-for-bit).
    """
    med, t0 = _dose_setup()
    named = PennesSolver(backend="numpy").solve(t0, 5e4, med, 1.0, 30, dose=True)
    auto = PennesSolver(backend="auto").solve(t0, 5e4, med, 1.0, 30, dose=True)
    assert np.array_equal(named.temperature, auto.temperature)
    assert np.array_equal(named.dose_cem43, auto.dose_cem43)
    assert named.meta["backend"] == "numpy"


@pytest.mark.gpu
@pytest.mark.skipif(not cupy_available(), reason="needs a CUDA device to compare against")
def test_cupy_matches_numpy_exactly_on_state_and_to_the_last_bit_of_pow_on_dose():
    """Cross-backend agreement, with the one place it is not exact named.

    The temperature field is bit-identical: the update is add/multiply on
    float32, and IEEE 754 pins those on both libraries. CEM43 is not, and
    cannot be — it accumulates ``R ** (43 - T)``, and ``pow`` is a
    transcendental whose last bit is a libm-versus-CUDA implementation
    choice, not a contract. Measured here (RTX 5050, cupy 14.2): the peak
    dose agrees to 1092.270142 CEM43 minutes on both, max relative
    difference 2.7e-14. The bound is set two orders above that so it fails
    on an arithmetic mistake and not on a library update.
    """
    med, t0 = _dose_setup()
    cpu = PennesSolver(backend="numpy").solve(t0, 5e4, med, 1.0, 30, dose=True)
    gpu = PennesSolver(backend="cupy").solve(t0, 5e4, med, 1.0, 30, dose=True)
    g_temp, g_dose = np.asarray(gpu.temperature), np.asarray(gpu.dose_cem43)
    assert np.array_equal(cpu.temperature, g_temp), "the state update is not backend-pure"
    scale = float(np.max(np.abs(cpu.dose_cem43))) or 1.0
    assert float(np.max(np.abs(cpu.dose_cem43 - g_dose))) / scale < 1e-12


def test_asking_for_the_gpu_without_one_fails_with_the_standard_message(no_gpu):
    med = ThermalMedium.homogeneous((4, 4), TISSUE, 1e-3)
    with pytest.raises(RuntimeError, match="no usable CUDA GPU"):
        PennesSolver(backend="cupy").solve(np.full((4, 4), 37.0, np.float32), None, med, 1.0, 2)


def test_a_real_solve_feeds_a_real_sonication_end_to_end():
    """The chain, on real objects: run -> HeatingSource -> Pennes -> CEM43.

    Every other test here builds its Q by hand. This one is the only place
    where ``SolverResult.region``, the acoustic ``Medium``, the material
    table and ``ThermalMedium`` have to agree about shapes, spacing and
    tissue layout — an interface mistake between the two halves of M18
    would show up nowhere else. Two hand-checks pin the physics across the
    join: Q at the focus is 2 alpha I of the recorded phasor, and the
    initial heating rate is Q/(rho C).
    """
    from caustica.core.grid import Grid
    from caustica.core.pml import PMLSpec
    from caustica.solvers import CWRunSpec, get
    from caustica.sources import CWSource

    tissue = TISSUE.model_copy(update={"name": "absorbing tissue", "alpha_np_m": 15.0, "beta": 0.0})
    grid = Grid(shape=(24, 24, 24), dx=0.5e-3, pml=PMLSpec(thickness=2.0e-3))
    db = MaterialDB(materials={1: tissue})
    id_map = np.ones(grid.shape, dtype=np.int32)
    medium = Medium.from_id_map(id_map, db)
    idx = np.array([[i, j, 5] for i in range(5, 19) for j in range(5, 19)], dtype=np.int32)
    src = CWSource(
        indices=idx, phases=np.zeros(len(idx), np.float32), amplitude=5e5, f0=5e5, ramp_periods=2.0
    )

    res = get("linear")().run(
        grid, medium, src, CWRunSpec(min_settle_periods=4, max_settle_periods=16), backend="numpy"
    )
    heat = HeatingSource.from_result(res, medium, grid.dx)
    assert heat.shape == grid.shape
    peak_p = float(np.abs(res.phasor).max())
    intensity = peak_p**2 / (2 * tissue.rho * tissue.c)
    assert heat.q_max == pytest.approx(2 * tissue.alpha_np_m * intensity, rel=1e-4)

    tmed = ThermalMedium.from_medium(medium, db, grid.dx)
    solver = PennesSolver(backend="numpy")
    dt = 0.02
    assert dt < solver.stable_dt(tmed)
    t_body = np.full(tmed.shape, ARTERIAL_TEMPERATURE_C, np.float32)

    # One step: before diffusion or perfusion have moved anything, the rise
    # is exactly Q dt / (rho C).
    one_step = solver.solve(t_body, heat, tmed, dt, 1)
    assert float((one_step.temperature - ARTERIAL_TEMPERATURE_C).max()) == pytest.approx(
        heat.q_max * dt / (tissue.rho * tissue.specific_heat), rel=0.01
    )

    hot = solver.solve(t_body, heat, tmed, dt, int(round(10.0 / dt)), dose=True, record_every=100)
    assert hot.times[-1] == pytest.approx(10.0)
    assert hot.peak_temperature_c > 45.0  # a real sonication, not a warm bath
    # A real transient obeys the same bound as the unit tests: some dose was
    # accrued, and less than if the peak had been held for the whole run.
    held_at_peak = float(cem43_minutes(hot.peak_temperature_c, hot.t_end_s))
    assert 0.0 < hot.peak_dose_cem43 < held_at_peak
    assert hot.meta["q"] == "HeatingSource(f0_only)"


def test_the_result_carries_the_provenance_of_the_run():
    med = ThermalMedium.homogeneous((4, 4), TISSUE, 1e-3)
    res = PennesSolver(backend="numpy").solve(np.full((4, 4), 37.0, np.float32), None, med, 1.0, 3)
    assert isinstance(res, ThermalResult)
    assert res.meta["scheme"] == "pennes-fd-explicit/1"
    assert res.meta["blood_density"] == BLOOD_DENSITY
    assert res.meta["blood_specific_heat"] == BLOOD_SPECIFIC_HEAT
    assert res.meta["arterial_temperature_c"] == ARTERIAL_TEMPERATURE_C
    assert res.meta["boundary"] == "insulated"
    assert res.meta["q"] == "none"
    assert res.meta["dt_stable_s"] == pytest.approx(1e-6 / (4 * D_TISSUE), rel=1e-4)
    assert res.temperature.dtype == np.float32
