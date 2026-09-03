"""The independent-implementation cross-check, < 5%.

The milestone asks for a second implementation to disagree with, and the
adapted (user-ratified, 2026-08-23) partner is ``tests/_thermal_reference.py``
— a backward-Euler solver on an assembled ``scipy.sparse`` operator, living
only in the test tree. Read that module's docstring for what is and is not
shared between the two paths; the short version is that the library steps an
explicit 7-point stencil in float32 and the reference factorises a matrix and
solves it in float64, so the only thing they have in common is the physics
(and the harmonic-mean face convention, deliberately, so this test grades the
time integrator rather than re-arguing the conductivity jump).

Two scenarios, as the criterion words it:

* **transient** — a two-layer slab (k, rho, C all jump), perfusion on, a
  Gaussian ``Q``, compared at several times along the way, not just at the
  end; an integrator error that cancels at steady state still shows here.
* **steady state** — source + perfusion, against the reference's DIRECT
  solve of ``-L T = b``. No time stepping is involved on that side at all,
  so an explicit scheme that had drifted would have nothing to hide behind.

Both are graded on the temperature RISE above body temperature, which is the
quantity the whole module exists to get right; grading absolute degrees would
divide every error by 37 and flatter the scheme by an order of magnitude.
"""

from __future__ import annotations

import numpy as np
import pytest

from _thermal_reference import backward_euler, steady_state
from caustica.materials import Material, MaterialDB
from caustica.thermal.pennes import ARTERIAL_TEMPERATURE_C, PennesSolver
from caustica.thermal.properties import ThermalMedium

#: The milestone's adapted bar for this cross-check.
CROSS_LIMIT = 0.05

DX = 1.0e-3
SHAPE = (16, 16, 24)

#: A two-layer slab: every thermal property jumps at the interface, so the
#: harmonic-mean face terms and the per-voxel rho*C both have to be right.
NEAR = Material(
    name="near layer",
    alpha_np_m=6.0,
    rho=1050.0,
    c=1540.0,
    beta=4.5,
    thermal_conductivity=0.52,
    specific_heat=3600.0,
    perfusion_rate=0.01,
)
FAR = Material(
    name="far layer",
    alpha_np_m=6.0,
    rho=920.0,
    c=1450.0,
    beta=4.5,
    thermal_conductivity=0.21,
    specific_heat=2350.0,
    perfusion_rate=0.005,
)
INTERFACE_VOX = 13


def _slab_medium() -> ThermalMedium:
    db = MaterialDB(materials={1: NEAR, 2: FAR})
    ids = np.ones(SHAPE, dtype=np.int32)
    ids[:, :, INTERFACE_VOX:] = 2
    return ThermalMedium.from_id_map(ids, db, DX)


def _gaussian_q(amplitude_w_m3: float = 1.5e5, sigma_m: float = 2.5e-3) -> np.ndarray:
    """A smooth focal-looking deposition, centred one layer short of the jump."""
    centre = (7.5, 7.5, 9.0)
    axes = [((np.arange(n) - c) * DX) ** 2 for n, c in zip(SHAPE, centre, strict=True)]
    r2 = np.zeros(SHAPE, dtype=np.float64)
    for ax, a in enumerate(axes):
        r2 = r2 + a.reshape([-1 if i == ax else 1 for i in range(len(SHAPE))])
    return (amplitude_w_m3 * np.exp(-r2 / (2 * sigma_m**2))).astype(np.float32)


def _agreement(library: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """``(peak-normalised max error, relative L2)`` on the rise above body T.

    Peak-normalised because the field is a localised hot spot on a large cold
    background: a plain pointwise relative error would divide by ~0 in the
    corners and report nonsense about voxels nobody is asking about.
    """
    a = np.asarray(library, dtype=np.float64) - ARTERIAL_TEMPERATURE_C
    b = np.asarray(reference, dtype=np.float64) - ARTERIAL_TEMPERATURE_C
    peak = float(np.abs(b).max())
    max_err = float(np.abs(a - b).max()) / peak
    rel_l2 = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    return max_err, rel_l2


def _cross_gate(label: str, max_err: float, rel_l2: float) -> None:
    """Assert both agreement numbers against the 5% bar, printing them.

    Printed rather than only asserted so ``pytest -s`` reports HOW closely the
    two implementations agree — the milestone criterion asks for the measured
    number, not for a green dot.
    """
    print(
        f"CROSS {label}: max err {max_err * 100:.4f}% of peak, relL2 "
        f"{rel_l2 * 100:.4f}%  (limit {CROSS_LIMIT * 100:.1f}%)"
    )
    assert max_err < CROSS_LIMIT, f"{label}: max error {max_err * 100:.4f}% exceeds 5%"
    assert rel_l2 < CROSS_LIMIT, f"{label}: relative L2 {rel_l2 * 100:.4f}% exceeds 5%"


@pytest.fixture(scope="module")
def transient_cross():
    """One explicit run and one implicit run of the same problem, same times."""
    med = _slab_medium()
    q = _gaussian_q()
    t0 = np.full(SHAPE, ARTERIAL_TEMPERATURE_C, dtype=np.float32)
    solver = PennesSolver(backend="numpy")
    # At 90% of the stability bound the explicit scheme is at its own limit,
    # which is where forward and backward Euler differ MOST. Comparing them
    # at a tiny dt would be a test of nothing.
    dt = 0.9 * solver.stable_dt(med)
    n_steps, record_every = 300, 60

    library = solver.solve(t0, q, med, dt=dt, n_steps=n_steps, record_every=record_every)
    _, samples, times = backward_euler(t0, q, med, dt, n_steps, record_every=record_every)
    return med, library, samples, times, dt


def test_the_transient_two_layer_slab_agrees_with_the_implicit_solver(transient_cross):
    """Explicit stencil vs assembled backward Euler, at five times along the way."""
    _, library, samples, times, dt = transient_cross
    assert library.times == pytest.approx(times, rel=1e-12)
    assert library.substeps == 1, "the comparison must be step-for-step, not sub-stepped"
    print(f"\ndt = {dt:.4g} s (90% of the explicit bound), {library.n_steps} steps")
    measured = []
    for t_s, lib, ref in zip(times[1:], library.samples[1:], samples[1:], strict=True):
        max_err, rel_l2 = _agreement(lib, ref)
        measured.append((t_s, max_err, rel_l2))
        _cross_gate(f"two-layer transient t={t_s:.1f}s", max_err, rel_l2)
    assert len(measured) == 5, "the criterion asks for several times, not one"


def test_the_transient_cross_check_is_actually_sensitive(transient_cross):
    """A trap: the same comparison against a WRONG reference must fail it.

    Two solvers can agree to 0.1% because they are both right or because the
    comparison cannot see anything. Perturbing the reference's perfusion by
    20% — a physically small, entirely plausible transcription error — must
    push the measured disagreement past the 5% bar, or this gate is decoration.
    """
    med, library, _, _, dt = transient_cross
    wrong = ThermalMedium(
        k=med.k,
        rho=med.rho,
        specific_heat=med.specific_heat,
        perfusion=med.perfusion * 1.2,
        dx=med.dx,
    )
    t0 = np.full(SHAPE, ARTERIAL_TEMPERATURE_C, dtype=np.float32)
    final, _, _ = backward_euler(t0, _gaussian_q(), wrong, dt, library.n_steps)
    max_err, rel_l2 = _agreement(library.temperature, final)
    print(
        f"CROSS trap (reference perfusion +20%): max err {max_err * 100:.3f}%, "
        f"relL2 {rel_l2 * 100:.3f}% — must exceed {CROSS_LIMIT * 100:.0f}%"
    )
    assert max_err > CROSS_LIMIT, "a 20% perfusion error is invisible: the metric is blind"


def test_the_perfused_steady_state_agrees_with_the_directly_solved_one():
    """Source + perfusion, run to steady state, against a direct sparse solve.

    The reference never takes a time step here: it factorises ``-L`` and
    solves ``-L T = Q + w rho_b C_b T_a`` once. Anything the explicit scheme
    accumulated over its 900 steps — a slow drift, a boundary flux leaking
    energy, a perfusion sink off by a constant — has nowhere to hide.
    """
    med = _slab_medium()
    q = _gaussian_q()
    solver = PennesSolver(backend="numpy")
    dt = 0.9 * solver.stable_dt(med)
    # ~9 perfusion time constants of the SLOWER layer, so the remaining
    # approach-to-steady-state error is far below the bar being measured.
    tau = float((med.volumetric_heat_capacity / (med.perfusion * 1050.0 * 3617.0)).max())
    n_steps = int(np.ceil(9.0 * tau / dt))

    t0 = np.full(SHAPE, ARTERIAL_TEMPERATURE_C, dtype=np.float32)
    library = solver.solve(t0, q, med, dt=dt, n_steps=n_steps)
    reference = steady_state(q, med)

    max_err, rel_l2 = _agreement(library.temperature, reference)
    print(
        f"\nsteady state: {n_steps} explicit steps of {dt:.4g} s "
        f"(9 tau = {9 * tau:.0f} s), peak rise "
        f"{float(reference.max()) - ARTERIAL_TEMPERATURE_C:.4g} K"
    )
    _cross_gate("perfused steady state (source + perfusion)", max_err, rel_l2)


def test_the_reference_refuses_a_steady_state_that_does_not_exist():
    """No perfusion + insulated walls = no steady state, and it says so.

    The reference is a test fixture, but a fixture that quietly returned the
    output of a singular solve would poison every gate above it.
    """
    db = MaterialDB(materials={1: NEAR.model_copy(update={"perfusion_rate": 0.0})})
    med = ThermalMedium.from_id_map(np.ones((4, 4, 4), dtype=np.int32), db, DX)
    with pytest.raises(ValueError, match="singular"):
        steady_state(1.0e4, med)
