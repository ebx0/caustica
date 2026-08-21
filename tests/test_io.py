"""M10 gates: quantization contract, atomic writes, result contract, resume.

Criteria encoded here (MILESTONES M10):
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
