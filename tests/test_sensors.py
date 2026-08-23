"""M18 gate: HeatingSource — Q = 2 alpha I, and the harmonic-alpha contract.

The physics here is one line long, so the tests that matter are the ones
about what the library REFUSES: the v1 absorption model knows alpha at f0
only, and turning a multi-harmonic field into heat without saying what the
harmonics absorb would under-report the dose. That refusal, and the two ways
out of it, are pinned below.
"""

from __future__ import annotations

import numpy as np
import pytest

from caustica.core.backend import CausticaWarning
from caustica.materials import water
from caustica.medium import Medium
from caustica.sensors import (
    ALPHA_MODEL_EXPLICIT,
    ALPHA_MODEL_F0,
    ALPHA_MODEL_POWER_LAW,
    HeatingSource,
)
from caustica.solvers.base import SolverResult

ALPHA = 10.0  # Np/m at f0
RHO = 1000.0
C = 1500.0
P0 = 1.0e6  # Pa peak


def _phasor(shape, amplitude):
    return np.full(shape, amplitude, dtype=np.complex64)


def _result(phasors, region):
    """A SolverResult carrying just what HeatingSource reads."""
    return SolverResult(
        phasor=phasors[1],
        p_max=np.abs(phasors[1]).astype(np.float32),
        region=region,
        dt=1e-8,
        spp=25,
        steps_total=1000,
        t_end_s=1e-5,
        tof_periods=3,
        converged_period=9,
        settle_capped=False,
        convergence_history=[],
        phasors=phasors,
    )


def test_plane_wave_heating_equals_two_alpha_times_the_intensity():
    """Q = alpha |P|^2/(rho c) IS 2 alpha I with I = |P|^2/(2 rho c).

    The closed form written both ways in the test, because the factor of two
    between "peak amplitude" and "time-averaged intensity" is the classic
    place a heating model goes quietly wrong by 2x.
    """
    heat = HeatingSource.from_phasors({1: _phasor((4, 4), P0)}, alpha=ALPHA, rho=RHO, c=C, dx=1e-3)
    intensity = P0**2 / (2.0 * RHO * C)
    expected = 2.0 * ALPHA * intensity
    assert expected == pytest.approx(ALPHA * P0**2 / (RHO * C))
    assert heat.q_max == pytest.approx(expected, rel=1e-6)
    assert heat.q.dtype == np.float32
    assert heat.alpha_model == ALPHA_MODEL_F0
    assert heat.harmonics == (1,)


def test_heating_grows_with_the_square_of_the_pressure():
    """Doubling the drive quadruples the heating — |P|^2, not |P|."""
    single = HeatingSource.from_phasors({1: _phasor((3,), P0)}, alpha=ALPHA, rho=RHO, c=C, dx=1e-3)
    double = HeatingSource.from_phasors(
        {1: _phasor((3,), 2 * P0)}, alpha=ALPHA, rho=RHO, c=C, dx=1e-3
    )
    assert double.q_max / single.q_max == pytest.approx(4.0, rel=1e-5)


def test_harmonics_without_a_declared_alpha_model_are_refused():
    """The honesty contract: the v1 alpha is a single-frequency number.

    Summing harmonics at the fundamental's alpha would UNDER-report heating
    (tissue absorbs 2f0 harder), so the refusal must say so and name both
    ways out. A silently small dose map is the failure mode this module
    exists to prevent.
    """
    phasors = {1: _phasor((4,), P0), 2: _phasor((4,), 0.5 * P0)}
    with pytest.raises(ValueError) as exc:
        HeatingSource.from_phasors(phasors, alpha=ALPHA, rho=RHO, c=C, dx=1e-3)
    msg = str(exc.value)
    assert "UNDER-REPORT" in msg
    assert "alpha_power_law_y" in msg  # way out 1
    assert "alpha_at_harmonics" in msg  # way out 2
    assert "harmonics=(1,)" in msg  # way out 3 (explicit fundamental-only)
    assert "M16" in msg  # and the real fix


def test_power_law_y_two_absorbs_the_second_harmonic_four_times_harder():
    """alpha_n = alpha_f0 * n**y, so y=2 gives alpha_2 = 4 alpha_f0.

    With equal harmonic amplitudes the second harmonic then deposits exactly
    four times the fundamental's heat — the documented "doubles, squared".
    """
    phasors = {1: _phasor((4,), P0), 2: _phasor((4,), P0)}
    heat = HeatingSource.from_phasors(
        phasors, alpha=ALPHA, rho=RHO, c=C, dx=1e-3, alpha_power_law_y=2.0
    )
    fundamental_only = ALPHA * P0**2 / (RHO * C)
    assert heat.q_max == pytest.approx(5.0 * fundamental_only, rel=1e-5)
    assert heat.alpha_model == ALPHA_MODEL_POWER_LAW
    assert heat.meta["alpha_np_m_range"][2] == (4 * ALPHA, 4 * ALPHA)
    per_h = heat.meta["mean_q_by_harmonic_w_m3"]
    assert per_h[2] / per_h[1] == pytest.approx(4.0, rel=1e-5)


def test_a_tissue_exponent_scales_more_gently_than_the_water_one():
    """y ~ 1.1 (soft tissue) must not be the same number as y = 2 (water)."""
    phasors = {1: _phasor((2,), P0), 2: _phasor((2,), P0)}
    tissue = HeatingSource.from_phasors(
        phasors, alpha=ALPHA, rho=RHO, c=C, dx=1e-3, alpha_power_law_y=1.1
    )
    assert tissue.meta["alpha_np_m_range"][2][0] == pytest.approx(ALPHA * 2**1.1)
    assert (
        tissue.q_max
        < HeatingSource.from_phasors(
            phasors, alpha=ALPHA, rho=RHO, c=C, dx=1e-3, alpha_power_law_y=2.0
        ).q_max
    )


def test_explicit_alpha_at_harmonics_is_used_verbatim():
    """Given numbers are used as given — no power law is fitted through them."""
    phasors = {1: _phasor((4,), P0), 2: _phasor((4,), P0), 3: _phasor((4,), P0)}
    heat = HeatingSource.from_phasors(
        phasors,
        alpha=ALPHA,
        rho=RHO,
        c=C,
        dx=1e-3,
        alpha_at_harmonics={2: 17.0, 3: 21.0},
    )
    unit = P0**2 / (RHO * C)
    assert heat.q_max == pytest.approx((ALPHA + 17.0 + 21.0) * unit, rel=1e-5)
    assert heat.alpha_model == ALPHA_MODEL_EXPLICIT
    assert heat.meta["alpha_np_m_range"][3] == (21.0, 21.0)


def test_asking_for_the_fundamental_alone_is_allowed_and_recorded():
    """ "Fundamental only" is a legitimate choice — it just has to be said.

    The provenance then records f0_only, so a report can show WHY the dose
    map ignored the harmonics that were in the result.
    """
    phasors = {1: _phasor((4,), P0), 2: _phasor((4,), P0)}
    heat = HeatingSource.from_phasors(phasors, alpha=ALPHA, rho=RHO, c=C, dx=1e-3, harmonics=(1,))
    assert heat.alpha_model == ALPHA_MODEL_F0
    assert heat.harmonics == (1,)
    assert heat.meta["harmonics_available"] == [1, 2]
    assert heat.q_max == pytest.approx(ALPHA * P0**2 / (RHO * C), rel=1e-5)


def test_two_alpha_models_at_once_are_refused():
    """A power law AND a table is two answers to one question."""
    phasors = {1: _phasor((2,), P0), 2: _phasor((2,), P0)}
    with pytest.raises(ValueError, match="EITHER"):
        HeatingSource.from_phasors(
            phasors,
            alpha=ALPHA,
            rho=RHO,
            c=C,
            dx=1e-3,
            alpha_power_law_y=2.0,
            alpha_at_harmonics={2: 40.0},
        )


def test_a_missing_harmonic_in_the_explicit_table_is_named():
    phasors = {1: _phasor((2,), P0), 2: _phasor((2,), P0), 3: _phasor((2,), P0)}
    with pytest.raises(ValueError, match=r"missing harmonic\(s\) \[3\]"):
        HeatingSource.from_phasors(
            phasors, alpha=ALPHA, rho=RHO, c=C, dx=1e-3, alpha_at_harmonics={2: 40.0}
        )


def test_an_unphysical_power_law_exponent_is_refused():
    """No medium absorbs as f^7; refusing beats extrapolating."""
    phasors = {1: _phasor((2,), P0), 2: _phasor((2,), P0)}
    with pytest.raises(ValueError, match="physical range"):
        HeatingSource.from_phasors(
            phasors, alpha=ALPHA, rho=RHO, c=C, dx=1e-3, alpha_power_law_y=7.0
        )


def test_a_scalar_harmonic_alpha_over_a_heterogeneous_medium_warns():
    """One number for a two-tissue map erases the contrast at that harmonic.

    Not an error (the caller may have exactly that number and mean it), but
    it must not pass in silence.
    """
    alpha_map = np.array([5.0, 5.0, 20.0, 20.0])
    phasors = {1: _phasor((4,), P0), 2: _phasor((4,), P0)}
    with pytest.warns(CausticaWarning, match="tissue contrast"):
        HeatingSource.from_phasors(
            phasors, alpha=alpha_map, rho=RHO, c=C, dx=1e-3, alpha_at_harmonics={2: 30.0}
        )


def test_from_result_slices_the_medium_by_the_record_region():
    """The recorded region is a sub-volume; Q must use ITS material values.

    Reading the medium at the wrong offset would put the absorption of one
    tissue on another tissue's pressure — a mistake nothing downstream can
    see.
    """
    shape = (8, 6)
    alpha = np.zeros(shape, np.float32)
    alpha[4:, :] = 20.0  # only the lower half absorbs
    medium = Medium(
        alpha=alpha,
        rho=np.full(shape, RHO, np.float32),
        c=np.full(shape, C, np.float32),
        beta=np.zeros(shape, np.float32),
    )
    region = (slice(4, 8), slice(0, 6))
    res = _result({1: _phasor((4, 6), P0)}, region)
    heat = HeatingSource.from_result(res, medium, dx=0.5e-3)
    assert heat.region == region
    assert np.allclose(heat.q, 20.0 * P0**2 / (RHO * C), rtol=1e-5)

    upper = _result({1: _phasor((4, 6), P0)}, (slice(0, 4), slice(0, 6)))
    assert HeatingSource.from_result(upper, medium, dx=0.5e-3).q_max == 0.0


def test_a_result_whose_region_does_not_match_the_medium_is_refused():
    medium = Medium.homogeneous((8, 6), water(alpha_np_m=ALPHA))
    res = _result({1: _phasor((5, 6), P0)}, (slice(4, 8), slice(0, 6)))
    with pytest.raises(ValueError, match="do not describe the same run"):
        HeatingSource.from_result(res, medium, dx=0.5e-3)


def test_embed_places_the_region_into_the_full_grid():
    """The thermal solve runs on the whole grid; Q was recorded on part of it."""
    res = _result({1: _phasor((2, 3), P0)}, (slice(1, 3), slice(0, 3)))
    medium = Medium.homogeneous((4, 3), water(alpha_np_m=ALPHA))
    heat = HeatingSource.from_result(res, medium, dx=1e-3)
    full = heat.embed((4, 3))
    assert full.shape == (4, 3)
    assert np.all(full[0] == 0.0)
    assert np.allclose(full[1:3], heat.q)
    with pytest.raises(ValueError, match="rank"):
        heat.embed((4, 3, 3))


def test_total_power_is_the_volume_integral_of_q():
    heat = HeatingSource.from_phasors(
        {1: _phasor((4, 4, 4), P0)}, alpha=ALPHA, rho=RHO, c=C, dx=2e-3
    )
    assert heat.total_power_w == pytest.approx(heat.q_max * 64 * (2e-3) ** 3, rel=1e-5)


def test_a_non_finite_phasor_is_refused():
    """A diverged run must not be laundered into a heat map."""
    bad = np.full((3,), np.nan, dtype=np.complex64)
    with pytest.raises(ValueError, match="non-finite"):
        HeatingSource.from_phasors({1: bad}, alpha=ALPHA, rho=RHO, c=C, dx=1e-3)


def test_zero_density_or_speed_is_refused():
    with pytest.raises(ValueError, match="must be > 0"):
        HeatingSource.from_phasors({1: _phasor((3,), P0)}, alpha=ALPHA, rho=0.0, c=C, dx=1e-3)


def test_a_harmonic_that_was_never_recorded_is_refused_with_the_fix():
    with pytest.raises(ValueError, match="not in the result"):
        HeatingSource.from_phasors(
            {1: _phasor((3,), P0)}, alpha=ALPHA, rho=RHO, c=C, dx=1e-3, harmonics=(1, 2)
        )
