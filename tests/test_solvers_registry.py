"""Registry + capability-declaration gate (the "fail loudly at setup" contract)."""

import numpy as np
import pytest

import caustica.solvers as solvers
from caustica import Grid, Medium
from caustica.materials import breast_default, water
from caustica.solvers import CWRunSpec, SolverCapabilityError
from caustica.sources import CWSource, plane_cw_source


def test_builtin_solvers_registered():
    names = solvers.available()
    assert "linear" in names
    assert "kwave" in names


def test_unknown_solver_lists_available():
    with pytest.raises(KeyError, match="linear"):
        solvers.get("does-not-exist")


def test_linear_rejects_nonlinear_medium_with_fixit():
    g = Grid(shape=(16, 16), dx=1e-3)
    id_map = np.full(g.shape, 2, dtype=np.uint8)  # fat: beta=4.5
    med = Medium.from_id_map(id_map, breast_default())
    src = plane_cw_source(g, f0=1e6, amplitude=1e5, position_vox=4)
    with pytest.raises(SolverCapabilityError, match="westervelt"):
        solvers.get("linear")().validate(g, med, src)


def test_kwave_rejects_1d():
    g = Grid(shape=(64,), dx=1e-3)
    med = Medium.homogeneous(g.shape, water())
    src = plane_cw_source(g, f0=1e6, amplitude=1e5, position_vox=4)
    with pytest.raises(SolverCapabilityError, match="supports"):
        solvers.get("kwave")().validate(g, med, src)


def test_shape_mismatch_rejected():
    g = Grid(shape=(16, 16), dx=1e-3)
    med = Medium.homogeneous((16, 12), water())
    src = plane_cw_source(g, f0=1e6, amplitude=1e5, position_vox=4)
    with pytest.raises(ValueError, match="medium shape"):
        solvers.get("linear")().validate(g, med, src)


def test_out_of_grid_source_rejected():
    g = Grid(shape=(16, 16), dx=1e-3)
    med = Medium.homogeneous(g.shape, water())
    src = CWSource(indices=np.array([[99, 0]]), phases=np.zeros(1), amplitude=1.0, f0=1e6)
    with pytest.raises(ValueError, match="outside grid"):
        solvers.get("linear")().validate(g, med, src)


def test_source_buried_in_the_pml_is_rejected():
    """The trap: a standoff exactly one PML thick.

    Every source voxel then sits in the sponge, is damped as fast as it is
    driven, and the run still converges — on a field that is quietly wrong.
    Nothing else in the native path notices, so ``validate`` must.
    """
    from caustica import PMLSpec

    g = Grid(shape=(64, 64, 96), dx=0.25e-3, pml=PMLSpec(thickness=5e-3))
    assert g.pml_vox == 20
    med = Medium.homogeneous(g.shape, water())

    buried = plane_cw_source(g, f0=1e6, amplitude=1e5, axis=2, position_vox=10)
    with pytest.raises(SolverCapabilityError, match="inside the PML band"):
        solvers.get("linear")().validate(g, med, buried)

    # ...but a plane that CLEARS the band is fine, even though its own edges
    # necessarily run into the lateral sponge (that is the plane-wave setup).
    ok = plane_cw_source(g, f0=1e6, amplitude=1e5, axis=2, position_vox=24)
    solvers.get("linear")().validate(g, med, ok)

    # and a grid with no PML has no band to fall into
    solvers.get("linear")().validate(
        Grid(shape=g.shape, dx=g.dx),
        med,
        buried,
    )


def test_run_spec_validation():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="min_settle_periods"):
        CWRunSpec(min_settle_periods=50, max_settle_periods=10)
    with pytest.raises(ValidationError, match="cfl_hard_max"):
        CWRunSpec(cfl=0.55, cfl_hard_max=0.50)
    spec = CWRunSpec()
    assert spec.n_record_periods == 2


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError, match="already registered"):

        @solvers.register
        class Fake(solvers.SolverBase):  # pragma: no cover - definition only
            name = "linear"
            caps = solvers.LinearKSpacePSTD.caps

            def run(self, *a, **k):
                raise NotImplementedError
