"""The k-space step has ONE definition, and both callers use it.

The solver and the planner's calibration probe both need the per-step
composition: the first to run it, the second to time it so a run can be
predicted before it starts. They used to hold two copies. When the engine
fused absorption and the sponge into a single damping volume, the probe's
copy went on applying both factors separately and ran about 10 % long at
192^3 against the closure a real solve uses, which is the defect these tests
exist to keep closed.

Two things are pinned here. That the damping volume is applied exactly once
per field per step, which is the arithmetic the drift was made of; and that
neither caller has quietly grown its own copy again.
"""

from __future__ import annotations

import numpy as np
import pytest

from caustica.core.backend import get_backend
from caustica.solvers.kspace import operators as ops
from caustica.solvers.kspace.engine import make_propagation_step


def _quiet_step(shape, *, damp_value, nonlinear=False, backend="numpy"):
    """A step whose only active term is the damping multiply.

    ``dt_over_rho`` and ``rhoc2_dt`` are zero, so the gradient and divergence
    contribute nothing and the fields change by the damping alone. That
    isolates the pass count from every other cost in the step.
    """
    b = get_backend(backend)
    xp, fft = b.xp, b.fft
    nd = len(shape)
    rng = np.random.default_rng(0)
    p = xp.asarray(rng.standard_normal(shape, dtype=np.float32))
    u = [xp.asarray(rng.standard_normal(shape, dtype=np.float32)) for _ in range(nd)]
    zero = xp.zeros(shape, dtype=xp.float32)
    ks = ops.k_vectors(shape, 1e-3, xp)
    kappa = ops.kappa_sinc(ks, c_ref=1500.0, dt=1e-7, xp=xp)
    deriv = ops.spectral_derivative_factors(ks, kappa, shape, xp)
    step = make_propagation_step(
        fft=fft,
        padded=shape,
        p=p,
        u=u,
        deriv=deriv,
        damp=xp.full(shape, np.float32(damp_value), dtype=xp.float32),
        dt_over_rho=zero,
        rhoc2_dt=zero,
        beta2_dt=zero if nonlinear else None,
    )
    return step, p, u


def test_the_damping_volume_is_applied_once_per_field_per_step():
    """Twice would be 0.25, and 0.25 is exactly what the old probe paid."""
    step, p, u = _quiet_step((16, 16, 16), damp_value=0.5)
    before_p, before_u = p.copy(), [c.copy() for c in u]
    step()
    np.testing.assert_array_equal(p, (before_p * np.float32(0.5)).astype(np.float32))
    for got, was in zip(u, before_u, strict=True):
        np.testing.assert_array_equal(got, (was * np.float32(0.5)).astype(np.float32))


def test_the_damping_is_applied_once_on_the_nonlinear_path_too():
    """The Westervelt branch is a separate line and could drift on its own."""
    step, p, _ = _quiet_step((16, 16, 16), damp_value=0.5, nonlinear=True)
    before = p.copy()
    step()
    np.testing.assert_array_equal(p, (before * np.float32(0.5)).astype(np.float32))


def test_a_unit_damping_volume_leaves_a_quiet_step_untouched():
    """The control: with nothing to damp and nothing to couple, nothing moves."""
    step, p, u = _quiet_step((16, 16, 16), damp_value=1.0)
    before_p, before_u = p.copy(), [c.copy() for c in u]
    step()
    np.testing.assert_array_equal(p, before_p)
    for got, was in zip(u, before_u, strict=True):
        np.testing.assert_array_equal(got, was)


def test_the_closure_mutates_the_caller_s_arrays_rather_than_copies():
    """The engine keeps its own references to p and u and reads them after
    every step, so the closure has to write through to THOSE objects.

    Identity is not the test: ``id(p)`` is this frame's binding and no
    implementation of the closure can move it. What has to hold is that the
    very array objects handed to the factory carry the new field, so the
    assertions below read through references taken before the step and never
    through the names the closure was built from.
    """
    step, p, u = _quiet_step((8, 8, 8), damp_value=0.5)
    p_obj, u_objs = p, list(u)
    before_p, before_u = p.copy(), [c.copy() for c in u]
    step()
    np.testing.assert_array_equal(p_obj, (before_p * np.float32(0.5)).astype(np.float32))
    for got, was in zip(u_objs, before_u, strict=True):
        np.testing.assert_array_equal(got, (was * np.float32(0.5)).astype(np.float32))
    # And the list the factory closed over still holds those same objects: a
    # closure that rebound u[i] would leave the engine reading stale slots.
    assert [id(c) for c in u] == [id(c) for c in u_objs]


def test_2d_is_supported_by_the_same_factory():
    step, p, u = _quiet_step((16, 16), damp_value=0.5)
    assert len(u) == 2
    before = p.copy()
    step()
    np.testing.assert_array_equal(p, (before * np.float32(0.5)).astype(np.float32))


# ------------------------------------------- one definition, two callers


def test_the_solver_builds_its_step_with_the_factory(monkeypatch):
    """Not a style check: a re-inlined copy is exactly how this drifted once."""
    from caustica.core.grid import Grid
    from caustica.materials import water
    from caustica.medium import Medium
    from caustica.solvers import CWRunSpec, get
    from caustica.solvers.kspace import engine as E
    from caustica.sources import CWSource

    calls = []
    real = E.make_propagation_step

    def counted(**kw):
        calls.append(kw)
        return real(**kw)

    monkeypatch.setattr(E, "make_propagation_step", counted)
    grid = Grid(shape=(24, 24, 24), dx=0.5e-3, pml=None)
    src = CWSource(
        indices=np.array([[12, 12, 6]], dtype=np.int32),
        phases=np.zeros(1, dtype=np.float32),
        amplitude=1e4,
        f0=1e6,
    )
    get("linear")().run(
        grid,
        Medium.homogeneous(grid.shape, water()),
        src,
        CWRunSpec(min_settle_periods=1, max_settle_periods=3),
    )
    assert len(calls) == 1, "the solver did not build its step through the factory"
    assert calls[0]["beta2_dt"] is None


def test_the_calibration_probe_builds_its_step_with_the_factory(monkeypatch):
    from caustica.planner import calibration as C

    calls = []
    real = C.make_propagation_step

    def counted(**kw):
        calls.append(kw)
        return real(**kw)

    monkeypatch.setattr(C, "make_propagation_step", counted)
    C.measure_step_time((24, 24, 24), backend="numpy", n_steps=2, warmup=1)
    assert len(calls) == 1, "the probe did not build its step through the factory"
    kw = calls[0]
    assert kw["beta2_dt"] is None
    # And it hands the factory ONE damping volume, the way the engine does.
    assert kw["damp"].shape == kw["p"].shape


def test_the_probe_carries_one_damping_volume_not_two(monkeypatch):
    """The volume count is what the planner's VRAM inventory is fitted to."""
    from caustica.planner import calibration as C

    seen = []
    real_full = np.full

    def counting_full(shape, value, *a, **kw):
        seen.append(float(np.asarray(value).reshape(-1)[0]))
        return real_full(shape, value, *a, **kw)

    monkeypatch.setattr(np, "full", counting_full)
    C.measure_step_time((24, 24, 24), backend="numpy", n_steps=1, warmup=0)
    damping = [v for v in seen if 0.9 < v < 1.0]
    assert len(damping) == 1, f"expected one damping volume, got {len(damping)}: {damping}"
    assert damping[0] == pytest.approx(0.999 * 0.999)
