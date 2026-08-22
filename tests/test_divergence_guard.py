"""A diverged run must FAIL, not return NaN with exit 0.

The first GPU gate session (2026-08-22) produced a 256^3 rung whose peak was
``nan`` from the second period onward. The run reported "converged at period
12 (SETTLE CAP HIT)", wrote a full ``result.h5``, exited 0, and the gate
suite then counted it as VRAM evidence -- the only check that passed in that
gate. These tests pin the hole shut at both levels: the engine raises, and
the runner turns that into exit 4 with an ``error.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from caustica.core.grid import Grid
from caustica.materials import water
from caustica.medium import Medium
from caustica.solvers import CWRunSpec, get
from caustica.solvers.base import SolverDivergedError
from caustica.sources import CWSource


def _tiny_setup(amplitude: float):
    grid = Grid(shape=(24, 24, 24), dx=0.5e-3, pml=None)
    medium = Medium.homogeneous(grid.shape, water())
    src = CWSource(
        indices=np.array([[12, 12, 6]], dtype=np.int32),
        phases=np.zeros(1, dtype=np.float32),
        amplitude=amplitude,
        f0=1e6,
    )
    return grid, medium, src


def test_a_diverging_run_raises_instead_of_returning_nan():
    # float32 cannot carry this drive: p overflows to inf within the first period
    # and inf - inf makes the NaN. A real divergence through the real path,
    # not an injected value.
    grid, medium, src = _tiny_setup(1e38)
    with pytest.raises(SolverDivergedError) as exc:
        get("linear")().run(
            grid, medium, src, CWRunSpec(min_settle_periods=1, max_settle_periods=3)
        )
    msg = str(exc.value)
    assert "diverged" in msg
    assert "settling period" in msg  # says WHERE, so the log locates it
    assert "CFL" in msg  # and says what to look at


def test_a_healthy_run_of_the_same_shape_still_returns():
    grid, medium, src = _tiny_setup(1e4)
    result = get("linear")().run(
        grid, medium, src, CWRunSpec(min_settle_periods=1, max_settle_periods=3)
    )
    assert np.isfinite(result.phasor).all()
    assert np.isfinite(result.p_max).all()


def test_the_guard_covers_the_record_window_too():
    """The settle guard cannot see a divergence that begins after settling."""
    from caustica.solvers.kspace import engine

    with pytest.raises(SolverDivergedError) as exc:
        engine._guard_finite(
            float("nan"), where="the record window (step 7)", solver_name="westervelt"
        )
    assert "record window" in str(exc.value)
    engine._guard_finite(0.0, where="unused", solver_name="linear")  # finite: no raise


def test_the_runner_maps_a_divergence_to_exit_4_and_an_error_json(tmp_path):
    from caustica.runner import EXIT_SOLVER, RunnerOptions, run_job_file

    job = {
        "format": "caustica-job/1",
        "kind": "explicit",
        "name": "diverge",
        "medium": {"kind": "homogeneous"},
        "grid": {
            "ndim": 3,
            "dx_mm": 0.5,
            "size_mm": [12.0, 12.0, 12.0],
            "pml": {"thickness_mm": 1.0},
        },
        "source": {
            "kind": "array",
            "array": {"kind": "bowl", "d_outer_mm": 4.0, "roc_mm": 5.0},
            "apex_mm": [6.0, 6.0, 3.0],
        },
        # 1e30 kPa: the same overflow as above, expressed in job units.
        "drive": {"f0_mhz": 1.0, "amplitude_kpa": 1e35},
        "run": {"spec": {"min_settle_periods": 1, "max_settle_periods": 2, "n_record_periods": 1}},
        "solver": "linear",
        "backend": "numpy",
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    out = tmp_path / "out"

    code = run_job_file(job_path, RunnerOptions(out=out, backend="numpy", progress=None))

    assert code == EXIT_SOLVER, "a NaN field must not be reported as a successful run"
    assert not (out / "result.h5").exists(), "a diverged run must not leave a result file"
    err = json.loads((out / "error.json").read_text(encoding="utf-8"))
    assert err["error_class"] == "SolverDivergedError"
    assert err["exit_code"] == EXIT_SOLVER
