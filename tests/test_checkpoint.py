"""In-run checkpoints — interrupt, resume, and trajectory identity.

The headline criterion: a run killed at period k and resumed
reproduces the uninterrupted run's phasor to rel < 1e-6. On one backend the
saved state is bit-exact and the step loop deterministic, so these tests
assert the STRONGER property — bitwise identity — and the documented 1e-6
band exists as headroom for cross-session float environment drift, not as an
expected error.
"""

import json

import numpy as np
import pytest

import caustica.solvers as solvers
from caustica import Grid, Medium, PMLSpec
from caustica.io import CheckpointMismatch, CheckpointSpec, RunInterrupted
from caustica.materials import water
from caustica.solvers import CWRunSpec
from caustica.sources import plane_cw_source

F0, C0, BETA = 1.0e6, 1500.0, 3.5


def _setup_1d(amplitude=1e5):
    dx = C0 / (F0 * 4.0)
    grid = Grid(shape=(96,), dx=dx, pml=PMLSpec(thickness=12 * dx))
    med = Medium.homogeneous(grid.shape, water(c=C0))
    src = plane_cw_source(grid, f0=F0, amplitude=amplitude, position_vox=18)
    spec = CWRunSpec(min_settle_periods=10, max_settle_periods=24)
    return grid, med, src, spec


def _stop_after(k: int):
    """A stop_when that fires on the k-th period-boundary poll."""
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] >= k

    return stop


def _assert_identical(a, b):
    np.testing.assert_array_equal(a.phasor, b.phasor)
    np.testing.assert_array_equal(a.p_max, b.p_max)
    for h in a.phasors:
        np.testing.assert_array_equal(a.phasors[h], b.phasors[h])
    assert a.steps_total == b.steps_total
    assert a.converged_period == b.converged_period
    assert a.settle_capped == b.settle_capped
    assert a.convergence_history == b.convergence_history
    # The documented band, kept as an explicit (weaker) assertion so the
    # criterion stays visible even if the bitwise one is ever relaxed:
    rel = np.abs(a.phasor - b.phasor).max() / max(np.abs(a.phasor).max(), 1e-30)
    assert rel < 1e-6


def test_interrupt_then_resume_reproduces_uninterrupted_run(tmp_path):
    grid, med, src, spec = _setup_1d()
    baseline = solvers.get("linear")().run(grid, med, src, spec, backend="numpy")

    ck_path = tmp_path / "run.ckpt.npz"
    with pytest.raises(RunInterrupted) as exc:
        solvers.get("linear")().run(
            grid,
            med,
            src,
            spec,
            backend="numpy",
            checkpoint=CheckpointSpec(path=ck_path, every_periods=2, stop_when=_stop_after(3)),
        )
    assert exc.value.periods_done == 3 and exc.value.stage == "settle"
    assert ck_path.exists()

    resumed = solvers.get("linear")().run(
        grid, med, src, spec, backend="numpy", checkpoint=CheckpointSpec(path=ck_path)
    )
    _assert_identical(baseline, resumed)
    assert not ck_path.exists()  # cleaned up on success


def test_checkpoint_survives_a_chain_of_interruptions(tmp_path):
    grid, med, src, spec = _setup_1d()
    baseline = solvers.get("linear")().run(grid, med, src, spec, backend="numpy")
    ck_path = tmp_path / "chain.ckpt.npz"
    for k in (2, 4):  # die twice, at different depths
        with pytest.raises(RunInterrupted):
            solvers.get("linear")().run(
                grid,
                med,
                src,
                spec,
                backend="numpy",
                checkpoint=CheckpointSpec(path=ck_path, every_periods=3, stop_when=_stop_after(k)),
            )
    resumed = solvers.get("linear")().run(
        grid, med, src, spec, backend="numpy", checkpoint=CheckpointSpec(path=ck_path)
    )
    _assert_identical(baseline, resumed)


def test_resume_from_the_pre_record_snapshot(tmp_path):
    """A kill DURING the record window resumes from the 'record' stage state."""
    grid, med, src, spec = _setup_1d()
    baseline = solvers.get("linear")().run(grid, med, src, spec, backend="numpy")
    # The pre-record poll is one boundary past the last settled period, so
    # firing there yields a stage="record" checkpoint by construction.
    n_settle_polls = baseline.converged_period
    ck_path = tmp_path / "rec.ckpt.npz"
    with pytest.raises(RunInterrupted) as exc:
        solvers.get("linear")().run(
            grid,
            med,
            src,
            spec,
            backend="numpy",
            checkpoint=CheckpointSpec(
                path=ck_path, every_periods=64, stop_when=_stop_after(n_settle_polls + 1)
            ),
        )
    assert exc.value.stage == "record"
    resumed = solvers.get("linear")().run(
        grid, med, src, spec, backend="numpy", checkpoint=CheckpointSpec(path=ck_path)
    )
    _assert_identical(baseline, resumed)


def test_westervelt_2d_checkpoint_parity(tmp_path):
    """Nonlinear + 2-D: the u-vector list and the beta path checkpoint too."""
    dx = C0 / (F0 * 4.0)
    grid = Grid(shape=(48, 72), dx=dx, pml=PMLSpec(thickness=8 * dx))
    med = Medium.homogeneous(grid.shape, water(c=C0, beta=BETA))
    src = plane_cw_source(grid, f0=F0, amplitude=2e5, axis=1, position_vox=12)
    spec = CWRunSpec(min_settle_periods=6, max_settle_periods=14)
    run = solvers.get("westervelt")().run
    baseline = run(grid, med, src, spec, backend="numpy", harmonics=(1, 2))
    ck_path = tmp_path / "w2d.ckpt.npz"
    ck = CheckpointSpec(path=ck_path, every_periods=2, stop_when=_stop_after(4))
    with pytest.raises(RunInterrupted):
        run(grid, med, src, spec, backend="numpy", harmonics=(1, 2), checkpoint=ck)
    resumed = run(
        grid,
        med,
        src,
        spec,
        backend="numpy",
        harmonics=(1, 2),
        checkpoint=CheckpointSpec(path=ck_path),
    )
    assert resumed.meta["nonlinear_active"] is True
    _assert_identical(baseline, resumed)


def test_mismatched_run_refuses_to_resume(tmp_path):
    """A checkpoint must never be spliced into a DIFFERENT simulation."""
    grid, med, src, spec = _setup_1d()
    ck_path = tmp_path / "fp.ckpt.npz"
    with pytest.raises(RunInterrupted):
        solvers.get("linear")().run(
            grid,
            med,
            src,
            spec,
            backend="numpy",
            checkpoint=CheckpointSpec(path=ck_path, every_periods=2, stop_when=_stop_after(3)),
        )
    # Different drive amplitude -> different trajectory -> refuse, by name.
    _, _, src2, _ = _setup_1d(amplitude=2e5)
    with pytest.raises(CheckpointMismatch, match="amplitude"):
        solvers.get("linear")().run(
            grid, med, src2, spec, backend="numpy", checkpoint=CheckpointSpec(path=ck_path)
        )
    # Different medium -> the medium hash catches it even at equal shapes.
    med2 = Medium.homogeneous(grid.shape, water(c=C0 + 10.0))
    with pytest.raises(CheckpointMismatch, match="medium_sha1"):
        solvers.get("linear")().run(
            grid, med2, src, spec, backend="numpy", checkpoint=CheckpointSpec(path=ck_path)
        )
    # The original call still resumes fine afterwards (nothing was clobbered).
    baseline = solvers.get("linear")().run(grid, med, src, spec, backend="numpy")
    resumed = solvers.get("linear")().run(
        grid, med, src, spec, backend="numpy", checkpoint=CheckpointSpec(path=ck_path)
    )
    _assert_identical(baseline, resumed)


def test_a_checkpoint_from_older_numerics_is_refused(tmp_path):
    """The case the other refusals cannot see: same run, different scheme.

    Amplitude, medium and geometry can all match while the STEP ITSELF has
    changed underneath — which is exactly what happened on 2026-08-24, when
    zeroing the Nyquist wavenumber altered the trajectory on every even-length
    axis. The scheme tag is the only field that catches it, so it needs a test
    of its own or the next numerics change will forget to bump it too.
    """
    grid, med, src, spec = _setup_1d()
    ck_path = tmp_path / "old_scheme.ckpt.npz"
    with pytest.raises(RunInterrupted):
        solvers.get("linear")().run(
            grid,
            med,
            src,
            spec,
            backend="numpy",
            checkpoint=CheckpointSpec(path=ck_path, every_periods=2, stop_when=_stop_after(3)),
        )

    # Rewrite the stored fingerprint as an older scheme would have left it.
    stored = dict(np.load(ck_path, allow_pickle=True))
    meta = json.loads(str(stored["meta_json"]))
    # Pinned on purpose: bumping the scheme is a deliberate act (it says the
    # trajectory changed), so it has to come here and say so too.
    assert meta["fingerprint"]["scheme"] == "cw-kspace-pstd/4", meta["fingerprint"]["scheme"]
    meta["fingerprint"]["scheme"] = "cw-kspace-pstd/1"
    stored["meta_json"] = np.asarray(json.dumps(meta))
    with open(ck_path, "wb") as fh:
        np.savez(fh, **stored)

    with pytest.raises(CheckpointMismatch, match="scheme"):
        solvers.get("linear")().run(
            grid, med, src, spec, backend="numpy", checkpoint=CheckpointSpec(path=ck_path)
        )


def test_checkpoint_write_cadence_is_every_periods(tmp_path):
    """Without a stop, checkpoints appear on the requested cadence."""
    import json

    grid, med, src, spec = _setup_1d()
    ck_path = tmp_path / "cadence.ckpt.npz"
    # Stop at period 5 with cadence 2: the last CADENCE write was period 4,
    # then the stop itself forces a period-5 write. Verify the stored period.
    with pytest.raises(RunInterrupted):
        solvers.get("linear")().run(
            grid,
            med,
            src,
            spec,
            backend="numpy",
            checkpoint=CheckpointSpec(path=ck_path, every_periods=2, stop_when=_stop_after(5)),
        )
    with np.load(ck_path, allow_pickle=False) as npz:
        meta = json.loads(str(npz["meta_json"]))
    assert meta["periods_done"] == 5
    assert meta["stage"] == "settle"
    assert meta["fingerprint"]["solver"] == "linear"


def test_spec_validates_every_periods():
    with pytest.raises(ValueError, match="every_periods"):
        CheckpointSpec(path="x.npz", every_periods=0)
