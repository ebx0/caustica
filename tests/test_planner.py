"""M8 planner gates (local half).

The planner must mirror the engine exactly where it can be checked without
hardware: same dt/spp (single source of truth), a VRAM inventory that
matches a hand count of engine.py's buffers, estimate sources labeled
db|calibrated|measured, and OOM verdicts that carry actionable advice.
The ±10% VRAM and ±25% calibrated-time gates are ON-DEVICE (Colab) gates —
tracked as open sub-criteria in MILESTONES.md, not testable here.
"""

from __future__ import annotations

import json
import math

import pytest

import caustica as hs
import caustica.solvers as solvers
from caustica import planner
from caustica.materials import water
from caustica.medium import Medium
from caustica.planner import calibration as cal
from caustica.planner import model
from caustica.solvers import CWRunSpec
from caustica.solvers.kspace.engine import cw_tof_periods
from caustica.sources import plane_cw_source


def tiny_setup(shape=(32, 32), dx=1e-3, f0=0.5e6):
    grid = hs.Grid(shape=shape, dx=dx, pml=hs.PMLSpec(thickness=4e-3))
    med = Medium.homogeneous(shape, water())
    src = plane_cw_source(grid, f0=f0, amplitude=1e5)
    return grid, med, src


# ---------------------------------------------------------------- memory model


def test_memory_inventory_matches_hand_count():
    # 60/50/40 are all 2,3,5-smooth -> padded shape must equal active shape.
    shape = (60, 50, 40)
    mm = model.kspace_memory(shape, nonlinear=True, n_harmonics=2, rec_elems=1000)
    assert mm.padded_shape == shape
    p = 60 * 50 * 40
    r = 60 * 50 * (40 // 2 + 1)
    b = mm.breakdown
    assert b["state (p + u)"] == (1 + 3) * 4 * p
    assert b["property maps"] == (3 + 1) * 4 * p
    assert b["sponge"] == 4 * p
    assert b["spectral factors (i*k*kappa)"] == 3 * 8 * r
    assert b["record buffers"] == 1000 * (8 * 2 + 4)
    assert b["step temporaries"] == 3 * 8 * r + (2 + 2) * 4 * p
    assert b["fft workspace"] == 2 * 8 * r
    assert mm.total_bytes == math.ceil(sum(b.values()) * model.ALLOCATOR_MARGIN)


def test_memory_model_uses_padded_fft_shape():
    # 61 is not 2,3,5-smooth: pads to 64 -> inventory must grow accordingly.
    small = model.kspace_memory((60, 50, 40), nonlinear=False, n_harmonics=1)
    padded = model.kspace_memory((61, 50, 40), nonlinear=False, n_harmonics=1)
    assert padded.padded_shape == (64, 50, 40)
    assert padded.total_bytes > small.total_bytes


def test_record_region_shrinks_the_estimate():
    grid, med, src = tiny_setup()
    full = planner.estimate(grid, med, src, gpu="A100")
    roi = planner.estimate(grid, med, src, gpu="A100", record_region=(slice(0, 8), slice(0, 8)))
    assert roi.vram_bytes < full.vram_bytes
    assert roi.vram_breakdown["record buffers"] == 8 * 8 * (8 * 1 + 4)


# ------------------------------------------------------------ engine mirroring


def test_estimate_mirrors_engine_discretization_and_bounds_steps():
    grid, med, src = tiny_setup()
    spec = CWRunSpec(min_settle_periods=2, max_settle_periods=12, n_record_periods=1)
    est = planner.estimate(grid, med, src, spec, solver="linear", gpu="T4")
    res = solvers.get("linear")().run(grid, med, src, spec)
    assert est.spp == res.spp
    assert est.dt == pytest.approx(res.dt, rel=1e-12)
    # engine may settle anywhere between the optimistic floor and the cap
    assert est.steps_expected <= res.steps_total <= est.steps_worst


def test_t_end_floor_enters_the_step_count():
    grid, med, src = tiny_setup()
    spec = CWRunSpec(
        min_settle_periods=1, max_settle_periods=4, n_record_periods=1, t_end_min_us=200.0
    )
    est = planner.estimate(grid, med, src, spec, gpu="T4")
    tof = cw_tof_periods(grid, med, src)
    assert tof + spec.max_settle_periods < 99  # the floor must dominate in this setup
    need = math.ceil(200e-6 * src.f0) - spec.n_record_periods
    assert est.steps_expected == (need + spec.n_record_periods) * est.spp
    assert est.steps_worst == est.steps_expected


# ------------------------------------------------------------ estimate sources


def test_estimate_source_labels_db_calibrated_measured(tmp_path):
    grid, med, src = tiny_setup()

    est = planner.estimate(grid, med, src, gpu="A100")
    assert est.source == "db"
    assert est.gpu == "A100-40GB"  # alias resolved
    assert est.fits
    assert any("calibrate" in w for w in est.warnings)

    calfile = tmp_path / "calibration.json"
    entry = {
        "a": 1e-9,
        "b": 2e-10,
        "backend": "cupy",
        "n_steps": 20,
        "nonlinear": False,
        "samples": [],
        "shapes": [],
        "vram_peak_bytes_by_shape": {},
        "calibrated_at": "test",
    }
    calfile.write_text(json.dumps({"version": 1, "devices": {"NVIDIA A100-SXM4-40GB": entry}}))
    est2 = planner.estimate(grid, med, src, gpu="A100", calibration_path=calfile)
    assert est2.source == "calibrated"
    _, p_elems, _ = model.fft_sizes(grid.shape)
    assert est2.t_step_s == pytest.approx(model.step_time(1e-9, 2e-10, p_elems))

    est3 = planner.estimate(grid, med, src, gpu="A100", measure=True)
    assert est3.source == "measured"
    assert est3.t_step_s > 0.0
    assert any("CPU" in w for w in est3.warnings)  # measured here on the numpy backend


def test_calibrate_cpu_roundtrip(tmp_path):
    calfile = tmp_path / "calibration.json"
    entry = planner.calibrate(shapes=((16, 16), (24, 24)), backend="numpy", n_steps=4, path=calfile)
    assert entry["a"] >= 0.0 and entry["b"] >= 0.0
    assert entry["a"] > 0.0 or entry["b"] > 0.0
    data = json.loads(calfile.read_text())
    assert "cpu" in data["devices"]
    assert cal.find_calibration_for("cpu", calfile) is not None
    # a cpu calibration must never masquerade as a GPU calibration
    assert cal.find_calibration_for("A100-40GB", calfile) is None


def test_fit_time_model_recovers_and_stays_nonnegative():
    a0, b0 = 3e-9, 5e-10
    samples = [(p, a0 * p * math.log2(p) + b0 * p) for p in (100_000, 300_000, 900_000)]
    a, b = cal.fit_time_model(samples)
    assert a == pytest.approx(a0, rel=1e-6)
    assert b == pytest.approx(b0, rel=1e-6)

    linear_only = [(p, 7e-10 * p) for p in (100_000, 900_000)]
    a, b = cal.fit_time_model(linear_only)
    assert a >= 0.0 and b >= 0.0
    predicted = model.step_time(a, b, 1_000_000)
    assert predicted == pytest.approx(7e-10 * 1_000_000, rel=0.05)


# -------------------------------------------------------------- OOM + verdicts


def test_oom_verdict_carries_actionable_advice(monkeypatch):
    fake = {
        "TINY": planner.GPUSpec(key="TINY", vram_gib=2, mem_bw_gbs=100, fp32_tflops=5.0),
        "A100-80GB": planner.GPUSpec(
            key="A100-80GB", vram_gib=80, mem_bw_gbs=1935, fp32_tflops=19.5
        ),
    }
    monkeypatch.setattr(planner, "load_gpu_db", lambda: (fake, {}))
    shape = (216, 216, 216)
    grid = hs.Grid(shape=shape, dx=0.5e-3, pml=hs.PMLSpec(thickness=3e-3))
    med = Medium.homogeneous(shape, water(beta=3.5))
    src = plane_cw_source(grid, f0=1e6, amplitude=1e5)
    est = planner.estimate(grid, med, src, solver="westervelt", gpu="TINY", harmonics=(1, 2))
    assert not est.fits
    assert any("increase dx" in a for a in est.advice)
    assert any("record region" in a for a in est.advice)
    assert any("harmonic" in a for a in est.advice)
    assert any("'linear' solver" in a for a in est.advice)
    assert any("A100-80GB" in a for a in est.advice)
    # the advice must be visible in the human summary too
    assert "DOES NOT FIT" in est.summary()


def test_unknown_gpu_and_unmodeled_solver_raise():
    grid, med, src = tiny_setup()
    with pytest.raises(ValueError, match="unknown gpu"):
        planner.estimate(grid, med, src, gpu="RTX9999")
    with pytest.raises(ValueError, match="k-space engine"):
        planner.estimate(grid, med, src, solver="kwave")


def test_compare_is_sorted_and_prints():
    grid, med, src = tiny_setup()
    comp = planner.compare(grid, med, src, gpus=("T4", "H100-SXM", "L4"))
    assert len(comp.estimates) == 3
    assert all(e.fits for e in comp.estimates)  # tiny grid fits everywhere
    times = [e.t_expected_s for e in comp.estimates]
    assert times == sorted(times)
    table = str(comp)
    assert "H100-SXM" in table and "db" in table


# ----------------------------------------------------- warmup as its own term


def test_expected_time_is_warmup_plus_steps_times_step_cost():
    """Fix A2: a GPU solve pays a one-time cost, and the model has to say so.

    The first real Colab session measured 26.6 ms/step against a 1.03 ms/step
    probe on the SAME shape in the SAME process — 2.66 s of cuFFT-plan and
    kernel-compilation cost that no per-step coefficient can absorb. The CPU
    control run came out at 0.96x, which is what makes it a missing constant
    rather than a broken model.
    """
    grid, med, src = tiny_setup()
    est = planner.estimate(grid, med, src, gpu="A100")
    assert est.warmup_s == model.GPU_WARMUP_S > 0.0
    assert est.t_expected_s == pytest.approx(est.warmup_s + est.t_step_s * est.steps_expected)
    assert est.t_worst_s == pytest.approx(est.warmup_s + est.t_step_s * est.steps_worst)
    assert "warmup" in est.summary()


def test_calibration_measures_a_warmup_and_the_estimate_uses_it(tmp_path):
    calfile = tmp_path / "calibration.json"
    entry = planner.calibrate(shapes=((16, 16), (24, 24)), backend="numpy", n_steps=4, path=calfile)
    assert entry["warmup_s"] >= 0.0
    assert entry["warmup_source"] == "probe"

    # The stored number is what an estimate against that device then uses,
    # instead of the datasheet constant.
    data = json.loads(calfile.read_text())
    data["devices"]["NVIDIA A100-SXM4-40GB"] = {**entry, "a": 1e-9, "b": 2e-10, "warmup_s": 7.5}
    calfile.write_text(json.dumps(data))
    grid, med, src = tiny_setup()
    est = planner.estimate(grid, med, src, gpu="A100", calibration_path=calfile)
    assert est.source == "calibrated"
    assert est.warmup_s == pytest.approx(7.5)


def test_a_pre_A2_calibration_entry_still_gets_the_constant(tmp_path):
    """Entries written before this fix carry no warmup key. Reading that as
    zero would silently reintroduce exactly the bug — a GPU entry without a
    measured warmup falls back to the datasheet constant."""
    calfile = tmp_path / "calibration.json"
    old_entry = {"a": 1e-9, "b": 2e-10, "backend": "cupy", "samples": [], "shapes": []}
    calfile.write_text(json.dumps({"version": 1, "devices": {"NVIDIA A100-SXM4-40GB": old_entry}}))
    grid, med, src = tiny_setup()
    est = planner.estimate(grid, med, src, gpu="A100", calibration_path=calfile)
    assert est.source == "calibrated"
    assert est.warmup_s == model.GPU_WARMUP_S


def test_record_warmup_writes_back_what_a_real_run_paid(tmp_path):
    """The probe replays the step composition, not a whole solve: it never
    builds the property maps or the source scatter, so it under-counts. The
    validation suite measures the real thing and writes it back here."""
    calfile = tmp_path / "calibration.json"
    planner.calibrate(shapes=((16, 16), (24, 24)), backend="numpy", n_steps=4, path=calfile)

    assert cal.record_warmup("no-such-device", 4.0, path=calfile) is None  # nothing to attach to
    updated = cal.record_warmup("cpu", 4.25, path=calfile)
    assert updated["warmup_s"] == pytest.approx(4.25)
    assert updated["warmup_source"] == "measured"
    assert cal.find_calibration_for("cpu", calfile)["warmup_s"] == pytest.approx(4.25)
    assert cal.record_warmup("cpu", -1.0, path=calfile)["warmup_s"] == 0.0  # never negative
