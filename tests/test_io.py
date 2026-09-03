"""M10 gates: quantization contract, atomic writes, result contract, resume.

Criteria encoded here:
- float16 round-trip max norm error <= 1e-3, verified; fallback to float32
  when the contract cannot be met;
- a writer killed mid-write leaves NO visible corrupt file (tmp swept, the
  target either absent or the previous complete version);
- resume skip-guard: in a 10-sample mini set, deleting one middle file makes
  exactly that one id regenerate;
- phase-convention and absorption-model attrs present in EVERY result file.
"""

import subprocess
import sys
import time

import numpy as np
import pytest

import caustica.solvers as solvers
from caustica import Grid, Medium, PMLSpec
from caustica.io import atomic_write, sweep_temp_debris, try_float16
from caustica.io.store import (
    RESULT_FORMAT,
    ResultStore,
    ensure_dir_verified,
    load_field,
    load_result,
    probe_writable,
    save_result,
    validate_result_file,
)
from caustica.materials import water
from caustica.solvers import CWRunSpec
from caustica.solvers.base import SolverResult
from caustica.sources import plane_cw_source

F0, C0 = 1.0e6, 1500.0


# ------------------------------------------------------------------ quantize


def test_float16_roundtrip_within_contract():
    rng = np.random.default_rng(0)
    arr = (rng.standard_normal((32, 24)) * 2.5e5).astype(np.float32)
    q = try_float16(arr, max_norm_err=1e-3)
    assert q.dtype_name == "float16"
    assert q.stored.dtype == np.float16
    peak = float(np.abs(arr).max())
    # The gate: measured AND actual round-trip error both within contract.
    assert q.norm_err <= 1e-3
    assert float(np.abs(q.restore() - arr).max()) <= 1e-3 * peak


def test_float16_falls_back_to_float32_when_contract_exceeded():
    rng = np.random.default_rng(1)
    arr = (rng.standard_normal(1000) * 1e6).astype(np.float32)
    # float16 carries ~4.9e-4 relative mantissa precision near peak — a 1e-5
    # contract is unmeetable, so the field must stay float32, verbatim.
    q = try_float16(arr, max_norm_err=1e-5)
    assert q.dtype_name == "float32"
    assert q.scale == 1.0 and q.norm_err == 0.0
    np.testing.assert_array_equal(q.restore(), arr)


def test_float16_zero_field_is_trivially_exact():
    q = try_float16(np.zeros((4, 4), np.float32))
    assert q.dtype_name == "float16" and q.norm_err == 0.0
    np.testing.assert_array_equal(q.restore(), np.zeros((4, 4), np.float32))


# -------------------------------------------------------------------- atomic


def test_atomic_write_lands_only_on_success(tmp_path):
    target = tmp_path / "out.bin"
    with atomic_write(target) as tmp:
        tmp.write_bytes(b"v1")
        assert not target.exists()  # invisible until the replace
    assert target.read_bytes() == b"v1"
    assert not list(tmp_path.glob("*.tmp"))  # tmp names are per-writer unique


def test_atomic_write_failure_keeps_old_version_and_cleans_tmp(tmp_path):
    target = tmp_path / "out.bin"
    target.write_bytes(b"old-complete-version")
    with pytest.raises(RuntimeError, match="boom"):
        with atomic_write(target) as tmp:
            tmp.write_bytes(b"half-writ")
            raise RuntimeError("boom")
    assert target.read_bytes() == b"old-complete-version"  # untouched
    assert not list(tmp_path.glob("*.tmp"))  # no debris on a caught failure


_KILL_SCRIPT = """
import sys, time
from pathlib import Path
from caustica.io.atomic import atomic_write

target, flag = Path(sys.argv[1]), Path(sys.argv[2])
with atomic_write(target) as tmp:
    with open(tmp, "wb") as fh:
        fh.write(b"x" * 4096)
        fh.flush()
    flag.write_text("mid-write")   # signal the parent we are inside the write
    time.sleep(60)                 # parent SIGKILLs us here
"""


def test_killed_writer_leaves_no_visible_file(tmp_path):
    """The M10 kill gate, with a REAL hard kill (no Python cleanup runs)."""
    script = tmp_path / "writer.py"
    script.write_text(_KILL_SCRIPT, encoding="utf-8")
    target, flag = tmp_path / "result.h5", tmp_path / "flag"
    proc = subprocess.Popen([sys.executable, str(script), str(target), str(flag)])
    try:
        deadline = time.time() + 30
        while not flag.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert flag.exists(), "writer never reached mid-write state"
    finally:
        proc.kill()
        proc.wait(timeout=30)
    # The invariant: the target NEVER exists half-written...
    assert not target.exists()
    # ...and the corpse is a .tmp that a STALE-aware sweep removes. A fresh
    # tmp may belong to a live sibling session, so the default threshold
    # leaves it alone; older_than_s=0 forces the single-owner semantics.
    debris = list(tmp_path.glob("*.tmp"))
    assert len(debris) == 1
    assert sweep_temp_debris(tmp_path) == []  # fresh -> spared by default
    removed = sweep_temp_debris(tmp_path, older_than_s=0)
    assert removed == debris and not debris[0].exists()


# ------------------------------------------------------ result file contract


@pytest.fixture(scope="module")
def mini_run():
    """One real (tiny, 1-D) solve: the contract is tested on honest data."""
    dx = C0 / (F0 * 4.0)
    grid = Grid(shape=(96,), dx=dx, pml=PMLSpec(thickness=12 * dx))
    med = Medium.homogeneous(grid.shape, water(c=C0))
    src = plane_cw_source(grid, f0=F0, amplitude=1e5, position_vox=18)
    spec = CWRunSpec(min_settle_periods=10, max_settle_periods=24)
    res = solvers.get("linear")().run(grid, med, src, spec, backend="numpy", harmonics=(1, 2))
    return grid, src, res


def _save(tmp_path, mini_run, name="run", **kw):
    grid, src, res = mini_run
    return save_result(
        tmp_path / f"{name}.h5",
        res,
        src,
        dx=grid.dx,
        grid_shape=grid.shape,
        pml_vox=grid.pml_vox,
        **kw,
    )


def test_save_load_roundtrip_within_quantization_contract(tmp_path, mini_run):
    grid, src, res = mini_run
    path = _save(tmp_path, mini_run)
    back = load_result(path)
    # Per stored component the contract is 1e-3 * its own peak.
    for h in res.phasors:
        for part in ("real", "imag"):
            a = getattr(res.phasors[h], part)
            b = getattr(back.phasors[h], part)
            assert float(np.abs(b - a).max()) <= 1e-3 * max(float(np.abs(a).max()), 1e-30)
    assert float(np.abs(back.p_max - res.p_max).max()) <= 1e-3 * float(res.p_max.max())
    # Run metadata survives exactly.
    assert back.region == res.region
    assert (back.dt, back.spp, back.steps_total) == (res.dt, res.spp, res.steps_total)
    assert back.converged_period == res.converged_period
    assert back.settle_capped == res.settle_capped
    assert back.convergence_history == pytest.approx(res.convergence_history, nan_ok=True)
    assert back.meta["solver"] == "linear"


def test_required_contract_attrs_present_in_every_file(tmp_path, mini_run):
    """The M10 downstream-contract gate: nothing may have to guess these."""
    import h5py

    path = _save(tmp_path, mini_run, extra_attrs={"job": "test-job"})
    with h5py.File(path, "r") as hf:
        for key in (
            "format",
            "caustica_version",
            "solver",
            "backend",
            "f0_hz",
            "dt_s",
            "spp",
            "harmonics",
            "dx_m",
            "grid_shape",
            "pml_vox",
            "region_start",
            "region_stop",
            "phase_convention",
            "absorption_model",
            "numerics_scheme",
            "source_discretization",
        ):
            assert key in hf.attrs, f"missing root attr {key!r}"
        assert hf.attrs["format"] == RESULT_FORMAT
        assert hf.attrs["job"] == "test-job"  # extra_attrs stamp through
        assert "phase_convention" in hf["output"].attrs
        assert "absorption_model" in hf["output"].attrs
        # amp/phase are DERIVED, never stored (project data contract).
        assert not any(n.startswith(("p_amp", "p_phase")) for n in hf["output"])
        # Every stored field carries its reload recipe.
        for name in hf["output"]:
            ds = hf[f"output/{name}"]
            for a in ("scale", "stored_dtype", "quant_norm_err", "reload"):
                assert a in ds.attrs, f"output/{name} missing {a!r}"


def test_load_field_honors_dynamic_scale(tmp_path, mini_run):
    grid, src, res = mini_run
    path = _save(tmp_path, mini_run)
    pmax = load_field(path, "p_max")
    assert pmax.dtype == np.float32
    assert float(np.abs(pmax - res.p_max).max()) <= 1e-3 * float(res.p_max.max())


def test_quantize_false_stores_float32_verbatim(tmp_path, mini_run):
    grid, src, res = mini_run
    path = _save(tmp_path, mini_run, name="exact", quantize=False)
    back = load_result(path)
    np.testing.assert_array_equal(back.p_max, res.p_max)
    np.testing.assert_array_equal(back.phasor, res.phasor)


def test_load_result_with_geometry_opens_the_file_once(tmp_path, mini_run, monkeypatch):
    """Fields AND the self-description out of a SINGLE open (janitor 02).

    ``caustica report`` used to call ``load_result`` and then re-open
    result.h5 to parse the same attrs from a second copy of the schema kept
    in the report module — a copy that could drift from the writer without
    anything noticing. The copy is gone; counting the opens keeps it gone.
    """
    import h5py

    grid, src, res = mini_run
    path = _save(
        tmp_path,
        mini_run,
        name="geo",
        extra_attrs={"job_name": "geo-job", "apex_vox": [18], "focus_vox": [64]},
    )
    opens = 0
    real_file = h5py.File

    def counting_file(*args, **kwargs):
        nonlocal opens
        opens += 1
        return real_file(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", counting_file)
    result, geo = load_result(path, with_geometry=True)

    assert opens == 1
    assert geo["dx"] == pytest.approx(grid.dx)
    assert geo["grid_shape"] == grid.shape
    assert geo["pml_vox"] == grid.pml_vox
    assert geo["f0_hz"] == pytest.approx(F0)
    assert geo["amplitude_pa"] == pytest.approx(src.amplitude)
    assert geo["solver"] == "linear" and geo["job_name"] == "geo-job"
    assert geo["apex_known"] and geo["apex_vox"] == (18,) and geo["focus_vox"] == (64,)
    np.testing.assert_array_equal(geo["source_indices"], src.indices)
    assert result.region == res.region  # the fields came back too


def test_result_geometry_falls_back_when_the_apex_stamp_is_absent(tmp_path, mini_run):
    """A pre-M10d file carries no apex/focus stamp: origin, and say so.

    Every output folder written before M10d is exactly this shape, and the
    report's "mm from the apex" caveat is driven by ``apex_known`` — so the
    fallback is contract, not convenience.
    """
    path = _save(tmp_path, mini_run, name="pre_m10d")
    _, geo = load_result(path, with_geometry=True)
    assert geo["apex_known"] is False
    assert geo["apex_vox"] == (0, 0, 0) and geo["focus_vox"] is None
    assert geo["job_name"] == "" and geo["git_commit"] == ""  # unstamped, not guessed
    assert isinstance(load_result(path), SolverResult)  # the default stays one object


# --------------------------------------------------------------------- store


def test_ensure_dir_verified_creates_nested(tmp_path):
    d = ensure_dir_verified(tmp_path / "a" / "b" / "c")
    assert d.is_dir()
    probe_writable(d)  # must not raise


def test_ensure_dir_verified_refuses_a_file(tmp_path):
    f = tmp_path / "notadir"
    f.write_text("x")
    with pytest.raises((OSError, RuntimeError)):
        ensure_dir_verified(f, retries=0)


def test_store_init_sweeps_temp_debris(tmp_path):
    import os as _os
    import time as _time

    root = tmp_path / "results"
    root.mkdir()
    debris = root / "run_00003.h5.tmp"
    debris.write_bytes(b"corpse of a killed writer")
    ResultStore(root)
    assert debris.exists()  # fresh tmp: may be a LIVE sibling's write -> spared
    old = _time.time() - 7200
    _os.utime(debris, (old, old))
    ResultStore(root)
    assert not debris.exists()  # stale corpse: swept


def test_resume_skip_guard_regenerates_only_the_missing_id(tmp_path, mini_run):
    """The M10 resume gate: 10 files, delete a middle one, only IT comes back."""
    grid, src, res = mini_run
    store = ResultStore(tmp_path / "mini")
    names = [f"sample_{i:05d}" for i in range(10)]

    writes = 0

    def produce(name: str) -> None:
        nonlocal writes
        writes += 1
        store.save(name, res, src, dx=grid.dx, grid_shape=grid.shape, pml_vox=grid.pml_vox)

    for name in store.missing(names):
        produce(name)
    assert writes == 10 and store.missing(names) == []

    store.path("sample_00005").unlink()  # the interruption
    assert store.missing(names) == ["sample_00005"]
    for name in store.missing(names):  # the resumed session
        produce(name)
    assert writes == 11  # exactly one regeneration, nothing else touched
    assert store.missing(names) == []


def test_torn_file_near_highwater_is_detected_and_regenerated(tmp_path, mini_run):
    grid, src, res = mini_run
    store = ResultStore(tmp_path / "mini2")
    names = [f"s{i}" for i in range(4)]
    for name in names:
        store.save(name, res, src, dx=grid.dx, grid_shape=grid.shape, pml_vox=grid.pml_vox)
    # Corrupt the newest file (inside the high-water validation zone). This
    # cannot happen through the atomic writer — it models external damage.
    store.path("s3").write_bytes(b"garbage that is not HDF5")
    assert not validate_result_file(store.path("s3"))
    assert store.missing(names) == ["s3"]


def test_deep_scan_catches_corruption_anywhere(tmp_path, mini_run):
    grid, src, res = mini_run
    store = ResultStore(tmp_path / "mini3")
    names = [f"s{i}" for i in range(6)]
    for name in names:
        store.save(name, res, src, dx=grid.dx, grid_shape=grid.shape, pml_vox=grid.pml_vox)
    store.path("s1").write_bytes(b"garbage")  # OUTSIDE the fast high-water zone
    assert store.missing(names) == []  # fast scan trusts the atomic invariant
    assert store.missing(names, deep=True) == ["s1"]  # deep scan does not


# ---------------------------------------------------- replace-retry (Windows)


def test_replace_retries_transient_permission_error(tmp_path, monkeypatch):
    """A reader holding the target open must not destroy the finished write."""
    import os as _os

    from caustica.io.atomic import replace_with_retry

    target = tmp_path / "locked.bin"
    tmp = tmp_path / "locked.bin.1234-abc.tmp"
    tmp.write_bytes(b"payload")
    real_replace, calls = _os.replace, {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("sharing violation")
        return real_replace(src, dst)

    monkeypatch.setattr(_os, "replace", flaky)
    replace_with_retry(tmp, target, attempts=5, delay_s=0.0)
    assert target.read_bytes() == b"payload" and calls["n"] == 3


def test_replace_final_failure_preserves_the_written_tmp(tmp_path, monkeypatch):
    import os as _os

    from caustica.io.atomic import replace_with_retry

    target = tmp_path / "locked.bin"
    tmp = tmp_path / "locked.bin.1234-abc.tmp"
    tmp.write_bytes(b"hours of solving")
    monkeypatch.setattr(
        _os, "replace", lambda s, d: (_ for _ in ()).throw(PermissionError("locked"))
    )
    with pytest.raises(PermissionError, match="preserved"):
        replace_with_retry(tmp, target, attempts=2, delay_s=0.0)
    assert tmp.read_bytes() == b"hours of solving"  # data survived the lock


def test_tmp_names_are_writer_unique(tmp_path):
    """Two writers to the SAME target must never share a tmp name.

    The 2026-08-19 review found deterministic names could promote a torn
    concurrent write to the final path; uniqueness is the fix's contract.
    """
    from caustica.io.atomic import tmp_path_for

    target = tmp_path / "result.h5"
    names = {tmp_path_for(target).name for _ in range(8)}
    assert len(names) == 8
    assert all(n.startswith("result.h5.") and n.endswith(".tmp") for n in names)


def test_a_result_says_which_numerics_and_which_source_made_it(tmp_path):
    """A pressure in pascals has to carry its own provenance.

    Two changes on 2026-08-24 moved absolute amplitudes by 13-18 %: zeroing
    the Nyquist wavenumber, and giving sources their own area instead of
    their voxel count. There is no backward compatibility to keep before
    v0.1, but a stored MPa still has to say which side of those it came
    from — inferring it from a commit date is not provenance.
    """
    import h5py

    from caustica.solvers.kspace.engine import NUMERICS_SCHEME
    from caustica.sources import bowl_cw_source

    grid = Grid(shape=(40, 40, 56), dx=0.5e-3, pml=PMLSpec(thickness=3e-3))
    medium = Medium.homogeneous(grid.shape, water())
    spec = CWRunSpec(min_settle_periods=2, max_settle_periods=6, n_record_periods=2)
    for mode in ("offgrid", "binary"):
        src = bowl_cw_source(grid, 1e6, 1e5, 4e-3, 10e-3, (20, 20, 8), discretization=mode)
        res = solvers.get("linear")().run(
            grid, medium, src, spec, backend="numpy", reference_point=(20, 20, 28)
        )
        path = save_result(
            tmp_path / f"{mode}.h5",
            res,
            src,
            dx=grid.dx,
            grid_shape=grid.shape,
            pml_vox=grid.pml_vox,
        )
        with h5py.File(path, "r") as hf:
            assert hf.attrs["numerics_scheme"] == NUMERICS_SCHEME
            assert hf.attrs["source_discretization"] == mode


def test_every_harmonic_is_stored_as_real_and_imaginary(tmp_path):
    """The recording contract a dataset is built on.

    Real and imaginary parts, separately, per harmonic; amplitude and phase
    never stored; and the float16 path only taken when the round-trip error
    it MEASURED is inside the contract, with that measurement written beside
    the data so a reader can check rather than trust.
    """
    import h5py

    from caustica.sources import bowl_cw_source

    # dx fine enough that the third harmonic clears the temporal Nyquist: the
    # exact-period step gives spp = floor(period / (cfl dx / c)) and the engine
    # needs 2h < spp, so 0.5 mm (spp = 6) is one harmonic short.
    grid = Grid(shape=(40, 40, 72), dx=0.25e-3, pml=PMLSpec(thickness=2e-3))
    medium = Medium.homogeneous(grid.shape, water())
    spec = CWRunSpec(min_settle_periods=2, max_settle_periods=6, n_record_periods=2)
    src = bowl_cw_source(grid, 1e6, 1e5, 3e-3, 8e-3, (20, 20, 10))
    res = solvers.get("linear")().run(
        grid,
        medium,
        src,
        spec,
        backend="numpy",
        reference_point=(20, 20, 42),
        harmonics=(1, 2, 3),
    )
    path = save_result(
        tmp_path / "h.h5", res, src, dx=grid.dx, grid_shape=grid.shape, pml_vox=grid.pml_vox
    )
    with h5py.File(path, "r") as hf:
        out = hf["output"]
        for h in (1, 2, 3):
            assert f"p_real_h{h}" in out and f"p_imag_h{h}" in out
            for name in (f"p_real_h{h}", f"p_imag_h{h}"):
                err = float(out[name].attrs["quant_norm_err"])
                assert err <= 1e-3, f"{name} stored outside the round-trip contract"
                assert out[name].attrs["stored_dtype"] in ("float16", "float32")
        assert list(hf.attrs["harmonics"]) == [1, 2, 3]
        assert not any(n.startswith(("p_amp", "p_phase")) for n in out)
