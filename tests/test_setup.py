"""Tests for :mod:`uwcem_phantoms.setup` — the stored run geometries.

The contract a setup file makes is narrow and checkable: reconstructing it must
give back the SAME transducer, in the same place, in front of the same phantom.
So the tests here mostly try to break that — mutate a recorded number, point a
setup at the wrong dataset, bury the array in the sponge — and require a loud
failure rather than a quietly different run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from uwcem_phantoms import catalog
from uwcem_phantoms.setup import (
    S1,
    SETUP_FORMAT,
    ArraySpec,
    SetupError,
    build_setups,
    load_setup,
    setup_names,
    verify_setups,
)

IDS = ("062204", "012304")


def _dataset_ready() -> bool:
    from uwcem_phantoms.paths import dataset_dir

    mp = dataset_dir() / "manifest.json"
    if not mp.exists():
        return False
    man = json.loads(mp.read_text(encoding="utf-8"))
    return all((dataset_dir() / e["file"]).exists() for e in man["phantoms"])


needs_dataset = pytest.mark.skipif(
    not _dataset_ready(),
    reason="the 0.25 mm dataset is not built (run: python -m uwcem_phantoms dataset)",
)
needs_archives = pytest.mark.skipif(
    not all(catalog.get(i).is_downloaded(with_pval=True) for i in IDS),
    reason="UWCEM archives not downloaded",
)


# --------------------------------------------------------------------------
# the array recipe (no phantom data needed)
# --------------------------------------------------------------------------


def test_s1_is_the_advertised_geometry():
    assert S1.n_elements == 64
    assert (S1.d_outer_mm, S1.d_inner_mm, S1.roc_mm) == (60.0, 26.4, 60.0)
    assert S1.f_number == pytest.approx(1.0)
    assert S1.half_angle_deg == pytest.approx(30.0)
    d = S1.derived()
    assert d["elem_radius_mm"] == pytest.approx(2.718368, abs=1e-5)
    assert d["shell_depth_mm"] == pytest.approx(6.5611, abs=1e-3)


def test_array_spec_json_round_trip():
    back = ArraySpec.from_json(S1.to_json())
    assert back == S1
    assert back.derived() == pytest.approx(S1.derived())


def test_unknown_array_kind_is_refused():
    bad = S1.to_json()
    bad["kind"] = "phased-toaster"
    with pytest.raises(SetupError, match="unknown array kind"):
        ArraySpec.from_json(bad)


def test_derived_values_are_recorded_not_recomputed_on_write():
    """The stored file must carry the numbers, or a reload cannot falsify them."""
    j = S1.to_json()
    assert set(j["derived"]) >= {"elem_radius_mm", "shell_depth_mm", "r_max_mm"}


# --------------------------------------------------------------------------
# building and loading against the real dataset
# --------------------------------------------------------------------------


@needs_dataset
@needs_archives
def test_build_and_load_round_trip(tmp_path):
    man = build_setups(IDS, out_dir=tmp_path)
    assert man["format"] == SETUP_FORMAT
    assert len(man["setups"]) == 2
    assert sorted(setup_names(tmp_path)) == sorted(f"s1-{i}" for i in IDS)

    for entry in man["setups"]:
        s = load_setup(entry["name"], out_dir=tmp_path, with_medium=False)
        # the array is REBUILT, not read: the voxel count must still match
        assert s.source.n_points == s.spec["placement"]["source_voxels"]
        assert s.spec["placement"]["elements_represented"] == 64
        # apex two voxels clear of the sponge, focus one ROC further on
        apex = s.spec["placement"]["apex_vox"]
        assert apex[2] == s.grid.pml_vox + 2
        assert s.focus_vox[2] == apex[2] + int(round(S1.roc_mm / s.spec["phantom"]["dx_mm"]))
        # no steering at all: this is the geometric focus
        assert s.spec["focus"]["steer_frac_roc"] == 0.0
        assert float(np.abs(s.source.phases).max()) == 0.0
        # the focus is in tissue and the shell is not
        assert s.spec["focus"]["tissue_class"] != 0
        assert s.spec["placement"]["min_clearance_mm"] > 0
        # the recorded box lies inside the grid and outside the sponge
        for r, n in zip(s.record_region, s.grid.shape, strict=True):
            assert 0 <= r.start < r.stop <= n
        assert s.record_region[2].start >= s.grid.pml_vox


@needs_dataset
@needs_archives
def test_loaded_setup_is_accepted_by_the_solver(tmp_path):
    build_setups(IDS[:1], out_dir=tmp_path)
    from caustica.solvers import registry

    s = load_setup(f"s1-{IDS[0]}", out_dir=tmp_path)
    assert s.medium is not None and s.medium.shape == s.grid.shape
    registry.get("westervelt")().validate(s.grid, s.medium, s.source)
    # ...and the linear solver correctly refuses it (tissue has beta != 0)
    from caustica.solvers.base import SolverCapabilityError

    with pytest.raises(SolverCapabilityError):
        registry.get("linear")().validate(s.grid, s.medium, s.source)


@needs_dataset
@needs_archives
def test_verify_setups_agrees_with_what_was_written(tmp_path):
    build_setups(IDS, out_dir=tmp_path)
    r = verify_setups(tmp_path)
    assert r["count"] == 2
    assert all(v["ok"] for v in r["setups"].values())


@needs_dataset
@needs_archives
def test_a_tampered_setup_fails_loudly(tmp_path):
    build_setups(IDS[:1], out_dir=tmp_path)
    path = tmp_path / f"s1-{IDS[0]}.json"
    good = json.loads(path.read_text(encoding="utf-8"))

    def write(mutate):
        d = json.loads(json.dumps(good))
        mutate(d)
        path.write_text(json.dumps(d), encoding="utf-8")

    # 1. a wrong format tag
    write(lambda d: d.update(format="bogus/9"))
    with pytest.raises(SetupError, match="format"):
        load_setup(f"s1-{IDS[0]}", out_dir=tmp_path, with_medium=False)

    # 2. a derived geometry the recipe does not reproduce — this is the guard
    #    against the array construction changing under a stored name
    write(lambda d: d["array"]["derived"].update(elem_radius_mm=2.9))
    with pytest.raises(SetupError, match="rebuilding the array"):
        load_setup(f"s1-{IDS[0]}", out_dir=tmp_path, with_medium=False)

    # 3. a voxel count that no longer matches the recipe
    write(lambda d: d["placement"].update(source_voxels=999))
    with pytest.raises(SetupError, match="source voxels"):
        load_setup(f"s1-{IDS[0]}", out_dir=tmp_path, with_medium=False)

    # 4. an apex shoved into the sponge — the trap the PML guard exists for
    write(lambda d: d["placement"].update(apex_vox=[280, 350, 2]))
    with pytest.raises(SetupError, match="PML band"):
        load_setup(f"s1-{IDS[0]}", out_dir=tmp_path, with_medium=False)

    # 5. claiming a clearance the phantom contradicts
    write(lambda d: d["placement"].update(min_clearance_vox=1))
    with pytest.raises(SetupError, match="measured clearance"):
        verify_setups(tmp_path)

    path.write_text(json.dumps(good), encoding="utf-8")
    verify_setups(tmp_path)  # unharmed by the refused loads


@needs_dataset
@needs_archives
def test_driving_at_a_frequency_the_dataset_does_not_bake_is_refused(tmp_path):
    """alpha is evaluated at the dataset's f0 and stored; driving elsewhere is
    not a resolution question, it is simply the wrong absorption."""
    with pytest.raises(SetupError, match="absorption would be wrong"):
        build_setups(IDS[:1], out_dir=tmp_path, f0_hz=2.0e6)


def test_load_setup_names_what_is_available(tmp_path):
    with pytest.raises(SetupError, match="no setup"):
        load_setup("s1-nope", out_dir=tmp_path)


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------


def test_setup_cli_parses_and_routes(monkeypatch):
    from uwcem_phantoms import cli
    from uwcem_phantoms import setup as st
    from uwcem_phantoms.cli import _cmd_setup, build_parser

    args = build_parser().parse_args(["setup"])
    assert args.func is _cmd_setup
    assert args.amplitude == 0.1 and args.pml == 5.0 and not args.verify

    calls = {}
    monkeypatch.setattr(
        st,
        "build_setups",
        lambda ids=None, out_dir=None, data_dir=None, amplitude_pa=None, pml_mm=None: (
            calls.update(ids=ids, amplitude_pa=amplitude_pa, pml_mm=pml_mm) or {"setups": []}
        ),
    )
    assert cli.main(["setup", "012304", "--amplitude", "0.5"]) == 0
    assert calls["ids"] == ["012304"]
    assert calls["amplitude_pa"] == pytest.approx(5.0e5)
