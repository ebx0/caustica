"""An axisymmetric scene must never be solved on a Cartesian grid.

The (r, z) half-plane of a revolved bowl is not a 2-D Cartesian setup: a
Cartesian 2-D solve integrates a LINE source, converges, and returns a field
that is quietly wrong. The guard is one attribute (``Medium.geometry``) plus
one capability (``SolverCaps.geometry``), checked in the solver and again
from the job config before anything is built.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import caustica.solvers as solvers
from caustica.config.job import JOB_FORMAT, medium_geometry, validate_job
from caustica.core.grid import Grid
from caustica.geometry.scene import Scene
from caustica.geometry.shapes import Ball
from caustica.io.medium_volume import load_medium_volume, write_medium_volume
from caustica.materials import Material, MaterialDB, water
from caustica.medium import GEOMETRIES, Medium
from caustica.solvers.base import SolverCapabilityError
from caustica.sources import plane_cw_source

WATER = {"name": "water", "alpha_np_m": 0.0, "rho": 1000.0, "c": 1500.0, "beta": 0.0}


def _db() -> MaterialDB:
    return MaterialDB(
        materials={
            0: water(),
            2: Material(name="fatty tissue", c=1450.0, rho=950.0, alpha_np_m=4.0, beta=0.0),
        }
    )


def _axi_scene() -> Scene:
    """A revolved ball on the (r, z) half-plane (axis 0 is the radius)."""
    return Scene(ndim=2, axisymmetric=True, background=0).add(Ball((0.0, 8e-3), 2e-3), 2)


def _cartesian_scene() -> Scene:
    return Scene(ndim=2, background=0).add(Ball((6e-3, 8e-3), 2e-3), 2)


def _grid() -> Grid:
    return Grid(shape=(24, 32), dx=0.5e-3)


# ------------------------------------------------------------- the attribute


def test_scene_stamps_its_geometry_on_the_medium():
    grid = _grid()
    assert _axi_scene().to_medium(grid, _db()).geometry == "axisymmetric"
    assert _cartesian_scene().to_medium(grid, _db()).geometry == "cartesian"


def test_medium_defaults_to_cartesian():
    assert Medium.homogeneous((8, 8), water()).geometry == "cartesian"
    assert Medium.from_id_map(np.zeros((4, 4), np.int64), _db()).geometry == "cartesian"
    zeros = np.zeros((4, 4), np.float32)
    assert Medium(alpha=zeros, rho=zeros, c=zeros, beta=zeros).geometry == "cartesian"


def test_medium_refuses_an_unknown_geometry():
    with pytest.raises(ValueError, match="geometry must be one of"):
        Medium.homogeneous((8, 8), water(), geometry="polar")


# ------------------------------------------------------- the solver refusal


def test_linear_2d_refuses_an_axisymmetric_scene():
    grid = _grid()
    medium = _axi_scene().to_medium(grid, _db())
    source = plane_cw_source(grid, f0=1e6, amplitude=1e5, axis=1, position_vox=4)
    with pytest.raises(SolverCapabilityError) as exc:
        solvers.get("linear")().validate(grid, medium, source)
    msg = str(exc.value)
    assert "axisymmetric" in msg
    assert "kspace-as" in msg


def test_linear_2d_refuses_it_at_run_too():
    """run() validates first, so the refusal precedes any stepping."""
    grid = _grid()
    medium = _axi_scene().to_medium(grid, _db())
    source = plane_cw_source(grid, f0=1e6, amplitude=1e5, axis=1, position_vox=4)
    with pytest.raises(SolverCapabilityError, match="kspace-as"):
        solvers.get("linear")().run(grid, medium, source)


@pytest.mark.parametrize("name", ["linear", "westervelt", "kwave"])
def test_every_built_in_solver_is_cartesian_only(name):
    """The built-ins, by name: an out-of-tree solver may declare anything."""
    assert solvers.get(name).caps.geometry == frozenset({"cartesian"})


def test_every_registered_solver_declares_a_known_geometry():
    for name in sorted(solvers.available()):
        geometry = solvers.get(name).caps.geometry
        assert geometry, f"solver '{name}' declares no geometry"
        assert set(geometry) <= set(GEOMETRIES), f"solver '{name}': {sorted(geometry)}"


def test_the_same_cartesian_scene_is_accepted():
    """The guard refuses the flag, not 2-D."""
    grid = _grid()
    medium = _cartesian_scene().to_medium(grid, _db())
    source = plane_cw_source(grid, f0=1e6, amplitude=1e5, axis=1, position_vox=4)
    solvers.get("linear")().validate(grid, medium, source)


# ---------------------------------------------------------- the CLI refusal


def _axi_job_dict(**over) -> dict:
    d = {
        "format": JOB_FORMAT,
        "kind": "explicit",
        "name": "axi-mini",
        "medium": {
            "kind": "scene",
            "scene": {"ndim": 2, "axisymmetric": True, "background": 0, "objects": []},
            "materials": {"0": WATER},
        },
        "grid": {"ndim": 2, "dx_mm": 0.5, "size_mm": [18, 24], "pml": {"thickness_mm": 2.0}},
        "source": {
            "kind": "array",
            "array": {"kind": "bowl", "d_outer_mm": 10.0, "roc_mm": 12.0},
            "apex_mm": [9, 9, 3.75],
        },
        "drive": {"f0_mhz": 1.0, "amplitude_kpa": 100.0},
        "run": {"spec": {"min_settle_periods": 3, "max_settle_periods": 8}, "harmonics": [1]},
        "solver": "linear",
    }
    d.update(over)
    return d


def _write_job(tmp_path: Path, d: dict) -> Path:
    p = tmp_path / "axi.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


@pytest.mark.parametrize("fast", [False, True])
def test_validate_job_reports_the_refusal(tmp_path, fast):
    report = validate_job(_write_job(tmp_path, _axi_job_dict()), fast=fast)
    assert not report.ok
    assert len(report.errors) == 1
    err = report.errors[0]
    assert err.startswith("SolverCapabilityError: ")
    assert "axisymmetric" in err
    assert "kspace-as" in err


def test_cli_validate_exits_2_with_the_message(tmp_path, capsys):
    from caustica.__main__ import main

    code = main(["validate", str(_write_job(tmp_path, _axi_job_dict()))])
    out = capsys.readouterr().out
    assert code == 2
    assert "axisymmetric" in out
    assert "kspace-as" in out


def test_a_3d_scene_carrying_the_flag_keeps_its_own_error(tmp_path):
    """axisymmetric on a 3-D scene is a broken scene, not a half-plane job.

    Telling that config to reach for the half-plane solver would be advice
    that cannot work, so the geometry check must not claim it.
    """
    d = _axi_job_dict()
    d["medium"]["scene"]["ndim"] = 3
    d["grid"] = {"ndim": 3, "dx_mm": 0.5, "size_mm": [18, 18, 24], "pml": {"thickness_mm": 2.0}}
    report = validate_job(_write_job(tmp_path, d))
    assert not report.ok
    assert any("axisymmetric scenes are 2-D" in e for e in report.errors)
    assert not any("kspace-as" in e for e in report.errors)


def test_an_unregistered_solver_name_is_still_reported(tmp_path):
    """The geometry check must not swallow the registry's own miss message."""
    report = validate_job(_write_job(tmp_path, _axi_job_dict(solver="nope")))
    assert not report.ok
    assert any("unknown solver 'nope'" in e for e in report.errors)


def test_medium_geometry_reads_the_config_without_building():
    from pydantic import TypeAdapter

    from caustica.config.job import MediumConfig

    adapter: TypeAdapter = TypeAdapter(MediumConfig)
    axi = adapter.validate_python(_axi_job_dict()["medium"])
    assert medium_geometry(axi) == "axisymmetric"
    assert medium_geometry(adapter.validate_python({"kind": "homogeneous"})) == "cartesian"
    three_d = _axi_job_dict()["medium"]
    three_d["scene"]["ndim"] = 3
    assert medium_geometry(adapter.validate_python(three_d)) == "cartesian"


# ------------------------------------ old medium_volume files stay Cartesian


def test_medium_volume_file_without_the_attribute_loads_as_cartesian(tmp_path):
    labels = np.zeros((6, 5, 4), dtype=np.int32)
    labels[2:5, 1:4, 1:3] = 2
    path = write_medium_volume(tmp_path / "vol.npz", dx=0.25e-3, labels=labels, materials=_db())
    with np.load(path) as npz:
        assert "geometry" not in npz.files
    medium = load_medium_volume(path).to_medium()
    assert medium.geometry == "cartesian"
