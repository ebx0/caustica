"""M8 planner gates (local half).

The planner must mirror the engine exactly where it can be checked without
hardware: same dt/spp (single source of truth), a VRAM inventory that
matches a hand count of engine.py's buffers, estimate sources labeled
db|calibrated|measured, and OOM verdicts that carry actionable advice.
The ±10% VRAM and ±25% calibrated-time gates are ON-DEVICE (Colab) gates —
tracked separately, not testable here.
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


def test_estimate_source_labels_db_calibrated_measured(tmp_path, no_gpu):
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
    # instead of the datasheet constant. An entry carrying only the flat
    # warmup (no two-term model) is used verbatim.
    data = json.loads(calfile.read_text())
    flat = {k: v for k, v in entry.items() if not k.startswith("warmup_")}
    data["devices"]["NVIDIA A100-SXM4-40GB"] = {**flat, "a": 1e-9, "b": 2e-10, "warmup_s": 7.5}
    calfile.write_text(json.dumps(data))
    grid, med, src = tiny_setup()
    est = planner.estimate(grid, med, src, gpu="A100", calibration_path=calfile)
    assert est.source == "calibrated"
    assert est.warmup_s == pytest.approx(7.5)


def test_the_calibrated_warmup_scales_with_the_grid(tmp_path):
    """Warmup is a per-process cost PLUS a per-shape one (cuFFT plans).

    Treating it as a constant under-predicted the 512^3 rung's 20.9 s by the
    whole plan-creation term, which is 46% of that run's wall time — enough
    on its own to miss M8's +/-25% gate (first gate session, 2026-08-22).
    """
    calfile = tmp_path / "calibration.json"
    entry = {
        "a": 0.0,
        "b": 2e-10,
        "backend": "cupy",
        "samples": [],
        "shapes": [],
        "warmup_context_s": 1.81,
        "warmup_per_elem_s": 1.6258e-07,
        "warmup_s": 4.54,
    }
    calfile.write_text(json.dumps({"version": 1, "devices": {"NVIDIA A100-SXM4-40GB": entry}}))
    # In the gate session all rungs shared one process, so only the first
    # paid the context and the later ones expose the per-shape term alone:
    # 400^3 paid 10.4 s and 512^3 paid 20.9 s. The slope fitted on the first
    # two reproduces the third to +4.3% — that is what makes this a model
    # rather than a guess.
    per_elem = entry["warmup_per_elem_s"]
    assert per_elem * 400**3 == pytest.approx(10.4, rel=0.02)
    assert per_elem * 512**3 == pytest.approx(20.9, rel=0.05)

    # A FRESH process — which is what a planned run actually is — pays the
    # context on top of that.
    assert cal.calibrated_warmup(entry, 400**3) == pytest.approx(1.81 + 10.4, rel=0.02)
    assert cal.calibrated_warmup(entry, 512**3) == pytest.approx(1.81 + 20.9, rel=0.05)

    # ... whereas a constant says the same thing at every size.
    assert cal.calibrated_warmup({"warmup_s": 4.54}, 512**3) == pytest.approx(4.54)
    assert cal.calibrated_warmup({"warmup_s": 4.54}, 128**3) == pytest.approx(4.54)


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


def cpu_target(**over) -> planner.GPUSpec:
    """The cpu planning target, built the way
    :func:`caustica.validation.analytic_suite.plan_target` builds it off the
    GPU: a key the calibration store matches, and throughput placeholders no
    honest number is ever derived from."""
    fields = {
        "key": "cpu",
        "vram_gib": 0.0,
        "mem_bw_gbs": 1.0,
        "fp32_tflops": 1.0,
        "device_name": "cpu",
        "source": "device",
    }
    return planner.GPUSpec(**{**fields, **over})


def test_a_cpu_plan_does_not_inherit_the_gpu_warmup_constant(tmp_path):
    """GPU_WARMUP_S is a CUDA measurement; numpy pays none of it.

    Inherited as a default it is not conservative, it is fiction: the
    analytic suite's planner table predicted 6.0 s for a pair of 1-D solves
    that cost 0.026 s, all but 0.002 s of it this constant (2026-08-23).
    Both non-measured paths are pinned, because the fallback lives in each.
    """
    grid, med, src = tiny_setup()

    # (a) calibrated, from an entry too old to carry a warmup at all — the
    # exact shape of the entry that caused the bug.
    calfile = tmp_path / "calibration.json"
    pre_a2 = {"a": 1e-9, "b": 2e-10, "backend": "numpy", "samples": [], "shapes": []}
    calfile.write_text(json.dumps({"version": 1, "devices": {"cpu": pre_a2}}))
    est = planner.estimate(grid, med, src, gpu=cpu_target(), calibration_path=calfile)
    assert est.source == "calibrated"
    assert est.warmup_s == 0.0
    assert est.t_expected_s == pytest.approx(est.t_step_s * est.steps_expected)

    # (b) db, with no entry for this machine at all.
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"version": 1, "devices": {}}))
    est_db = planner.estimate(grid, med, src, gpu=cpu_target(), calibration_path=empty)
    assert est_db.source == "db"
    assert est_db.warmup_s == 0.0

    # ... while the same two paths on a CARD are untouched: the constant is
    # the whole point of fix A2 there.
    gpu_db_path = planner.estimate(grid, med, src, gpu="A100", calibration_path=empty)
    assert (gpu_db_path.source, gpu_db_path.warmup_s) == ("db", model.GPU_WARMUP_S)


def test_a_measured_cpu_warmup_is_used_rather_than_zeroed(tmp_path):
    """Zero is the DEFAULT, not a rule: a cpu entry that measured its own
    warmup is believed, exactly like a GPU one. ``calibrate(backend="numpy")``
    has recorded those fields since fix A2 — the bug was only ever entries
    written before it."""
    calfile = tmp_path / "calibration.json"
    entry = planner.calibrate(shapes=((16, 16), (24, 24)), backend="numpy", n_steps=4, path=calfile)
    assert entry["warmup_context_s"] is not None and entry["warmup_source"] == "probe"

    data = json.loads(calfile.read_text())
    data["devices"]["cpu"] = {**entry, "warmup_context_s": 0.25, "warmup_per_elem_s": 0.0}
    calfile.write_text(json.dumps(data))
    grid, med, src = tiny_setup()
    est = planner.estimate(grid, med, src, gpu=cpu_target(), calibration_path=calfile)
    assert est.source == "calibrated"
    assert est.warmup_s == pytest.approx(0.25)


def test_the_cpu_target_is_recognised_by_the_same_rule_the_store_matches_on(tmp_path):
    """One rule, two readers: if these ever disagreed, a plan could take the
    cpu warmup branch while the calibration lookup went hunting for a card
    (or the reverse)."""
    assert planner.is_cpu_target(cpu_target())
    assert planner.is_cpu_target(cpu_target(key="unknown:cpu"))
    # Every real device in the database is a card, however it is spelled.
    devices, _aliases = planner.load_gpu_db()
    for key, spec in devices.items():
        assert not planner.is_cpu_target(spec), key
    assert not planner.is_cpu_target(planner.spec_for_device("NVIDIA A100-SXM4-40GB"))

    # And the store's own cpu rule agrees about which entry each key reaches.
    calfile = tmp_path / "calibration.json"
    calfile.write_text(
        json.dumps(
            {"version": 1, "devices": {"cpu": {"a": 1.0}, "NVIDIA A100-SXM4-40GB": {"a": 2.0}}}
        )
    )
    assert cal.find_calibration_for("cpu", calfile)["a"] == 1.0
    assert cal.find_calibration_for("A100-40GB", calfile)["a"] == 2.0


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


# ------------------------------------------- probe sizing (M8 time, fix F3)


def test_gpu_probe_shapes_saturate_the_device():
    """Bigger card, bigger probe -- and never one too small to be timed.

    The first A100 calibration probed 48^3 and 72^3. Both took ~1.0 ms/step
    because neither saturates the device: that measures kernel-launch
    latency, and extrapolating it to a real 512^3 run over-predicted the
    step cost by 6.8x (measured, 2026-08-22).
    """
    gib = 2**30
    largest = {}
    for name, free in (("T4", 14.6), ("L4", 21.5), ("A100-40", 39.1), ("A100-80", 79.0)):
        shapes = cal.probe_shapes_for_budget(int(free * gib))
        assert len(shapes) >= 2, f"{name}: the fit needs at least two sizes"
        n_elems = [s[0] ** 3 for s in shapes]
        assert n_elems == sorted(n_elems), f"{name}: probes must ascend"
        assert n_elems[-1] >= cal.MIN_GPU_PROBE_ELEMS, f"{name}: largest probe is latency-bound"
        biggest = max(shapes, key=lambda s: s[0])
        mem = model.kspace_memory(biggest, False, 1, rec_elems=biggest[0] ** 3)
        assert mem.total_bytes <= free * gib * max(cal.PROBE_VRAM_FRACTIONS) * 1.001, (
            f"{name}: the probe must not crowd out the run it is planning for"
        )
        largest[name] = n_elems[-1]
    assert largest["A100-80"] > largest["A100-40"] > largest["T4"]


def test_the_replaced_fixed_probe_pair_would_still_be_rejected():
    """Documents the defect: the old default is below the saturation floor."""
    assert 48**3 < cal.MIN_GPU_PROBE_ELEMS
    assert 72**3 < cal.MIN_GPU_PROBE_ELEMS


def test_a_fit_that_cannot_reproduce_its_samples_is_flagged():
    """The real A100 calibration, replayed.

    Two nearly-equal times at two sizes: the nonnegativity clip zeroes `a`,
    and the surviving single parameter cannot pass through both points. The
    residual is what makes that visible instead of silent.
    """
    samples = [(110_592, 0.0010611400000470894), (373_248, 0.0010005005001403333)]
    a, b = cal.fit_time_model(samples)
    assert a == 0.0, "the clip is what happened on the real device"
    assert cal.fit_max_rel_residual(samples, a, b) > cal.FIT_RESIDUAL_LIMIT

    # A healthy fit, by contrast, reproduces its own samples.
    good = [(4_096_000, 0.002), (16_777_216, 0.0082), (32_768_000, 0.0160)]
    ga, gb = cal.fit_time_model(good)
    assert cal.fit_max_rel_residual(good, ga, gb) < cal.FIT_RESIDUAL_LIMIT


# ------------------------------- the time curve is interpolated (M8.time)
#
# A device's per-element cost is a curve, and a two-parameter global law
# cannot bend the way a real one does. On an RTX PRO 6000 Blackwell (95 GiB,
# 2026-08-23) `calibrate()` probed three SATURATING sizes -- 216^3, 320^3,
# 432^3 -- and still could not fit a*P*log2(P) + b*P: residual 0.705, the
# throughput-anchored fallback engaged, and every plan carried "the
# calibration's time model misses its own samples by up to 70%".
#
# BLACKWELL_PROBES is that session's own curve, recovered from the report it
# wrote (benchmarks/reports/dev_validate/blackwell-20260823-131456):
#   432^3 -- t = b*P for the b it stored, which IS the anchored rate;
#   320^3 -- the stage-2 spot check measured this shape: 10.937 ms;
#   216^3 -- b*P / (1 + 0.7046), the only value consistent with the stored
#            residual once monotonicity settles the sign (the alternative
#            has a 216^3 step costing 12.6 ms while 320^3 costs 10.9).
# The rate RISES with size: small grids are disproportionately fast on this
# card, which is exactly the shape a*P*log2(P) + b*P cannot take.

BLACKWELL_B = 3.695926690488404e-10  # s/element, the anchored fallback it stored
BLACKWELL_PROBES = [
    (216**3, 0.002185085),  # rate 2.168e-10
    (320**3, 0.010937260),  # rate 3.338e-10  (measured, stage-2 spot check)
    (432**3, 0.029797141),  # rate 3.696e-10  (== BLACKWELL_B * P)
]
#: Per-step cost of the ladder rungs, wall minus the run's own measured
#: warmup: (13.66-4.389)/360, (34.97-12.838)/420, (83.93-32.375)/500. The
#: probes above never saw these sizes, so they grade the extrapolation.
BLACKWELL_LADDER = {400: 0.02575, 512: 0.05270, 640: 0.10311}


def test_a_curved_device_defeats_the_global_fit_but_not_the_interpolator():
    """The measured failure, and the fix, in one place."""
    rates = [t / p for p, t in BLACKWELL_PROBES]
    assert rates == sorted(rates)  # per-element cost RISES with size
    assert rates[-1] / rates[0] == pytest.approx(1.70, abs=0.01)

    # (a) no two-parameter global law survives this shape.
    a, b = cal.fit_time_model(BLACKWELL_PROBES)
    assert b == 0.0  # the nonnegativity clip, on the other coefficient this time
    assert cal.fit_max_rel_residual(BLACKWELL_PROBES, a, b) == pytest.approx(0.495, abs=0.01)
    anchored = cal.fit_max_rel_residual(BLACKWELL_PROBES, 0.0, BLACKWELL_B)
    assert anchored == pytest.approx(0.7045755942641264, abs=1e-4)  # the stored number
    assert min(anchored, cal.fit_max_rel_residual(BLACKWELL_PROBES, a, b)) > cal.FIT_RESIDUAL_LIMIT

    # (b) the interpolator passes through every measurement exactly.
    for p_elems, t in BLACKWELL_PROBES:
        assert cal.interp_step_time(BLACKWELL_PROBES, p_elems) == pytest.approx(t, rel=1e-12)

    # (c) an unprobed size in between lands between its neighbours' RATES.
    # 256^3 is 4.38 ms/step here; the anchored law says 6.20 ms, because it
    # applies the SLOWEST measured per-element rate at every size.
    p256 = 256**3
    t256 = cal.interp_step_time(BLACKWELL_PROBES, p256)
    assert t256 == pytest.approx(0.004383, rel=1e-3)
    assert rates[0] < t256 / p256 < rates[1]
    assert BLACKWELL_PROBES[0][1] < t256 < BLACKWELL_PROBES[1][1]
    assert model.step_time(0.0, BLACKWELL_B, p256) == pytest.approx(0.006200, rel=1e-3)

    # (d) the honest quality number is leave-one-out, not the residual
    # against knots it is guaranteed to hit. 12% held-out against a 70%
    # global miss -- inside M8's +/-25% gate, still above the recalibrate
    # threshold, and the plan says so rather than pretending.
    loo = cal.interp_loo_rel_err(BLACKWELL_PROBES)
    assert loo == pytest.approx(0.121, abs=0.001)
    assert cal.FIT_RESIDUAL_LIMIT < loo < 0.25
    assert cal.interp_loo_rel_err(BLACKWELL_PROBES[:2]) is None  # nothing to hold out


def test_the_ladder_the_gate_measured_is_predicted_within_the_m8_tolerance():
    """The rungs are genuinely out of sample: probes at 216/320/432^3,
    rungs at 400/512/640^3. Two of the three EXTRAPOLATE above the probes.

    This is the M8.time per-step half, which the session already passed on
    the anchored model (-6..-9%); interpolation must not lose that.
    """
    for side, steady in BLACKWELL_LADDER.items():
        interp = cal.interp_step_time(BLACKWELL_PROBES, side**3)
        assert abs(interp - steady) / steady < 0.25, side
    # measured: -10.5% at 400^3 (interpolated), -0.3% at 512^3 and +7.4% at
    # 640^3 (both extrapolated), against -8.1%/-5.9%/-6.0% for the anchor.
    assert cal.interp_step_time(BLACKWELL_PROBES, 512**3) == pytest.approx(0.05270, rel=0.01)


def test_a_well_behaved_device_gets_the_same_answer_as_before():
    """No regression where the global law was never the problem.

    Near-constant per-element rate (an A100's shape): fit and interpolation
    have to agree, or this change would trade one device's accuracy for
    another's.
    """
    a100 = [(4_096_000, 0.00205), (16_777_216, 0.0082), (32_768_000, 0.0158)]
    a, b = cal.fit_time_model(a100)
    assert cal.fit_max_rel_residual(a100, a, b) < cal.FIT_RESIDUAL_LIMIT  # a healthy fit

    for p_elems, tol in ((8_000_000, 0.03), (20_000_000, 0.03)):
        fitted = model.step_time(a, b, p_elems)
        interp = cal.interp_step_time(a100, p_elems)
        assert interp == pytest.approx(fitted, rel=tol)
    # measured: 3.870 vs 3.959 ms (2.3%) at 8.0e6, 9.675 vs 9.740 ms (0.7%) at 2.0e7
    assert cal.interp_loo_rel_err(a100) == pytest.approx(0.0016, abs=0.001)


def test_extrapolation_never_claims_a_rate_the_device_has_never_shown():
    """Above the top probe the last segment's slope continues -- until it
    would imply the card is faster per element than its best measurement."""
    p_top, t_top = BLACKWELL_PROBES[-1]
    best_rate = t_top / p_top
    for side in (440, 480, 512, 640, 800):
        p_elems = side**3
        t = cal.interp_step_time(BLACKWELL_PROBES, p_elems)
        assert t / p_elems >= best_rate - 1e-18, side
        assert t >= t_top
    # This curve's top segment is SUPERLINEAR (slope 1.11), so the floor does
    # not bind and the slope is what predicts -- 52.55 ms at 512^3 against a
    # measured 52.70. The rule is a floor on cost, never a cap.
    rising = [(1_000_000, 1e-3), (2_000_000, 3e-3)]
    assert cal.interp_step_time(rising, 4_000_000) == pytest.approx(9e-3, rel=1e-9)

    # A sublinear top segment IS floored, at the top sample's own rate:
    # continuing 'twice the work for 1.2x the time' would eventually claim an
    # unbounded throughput.
    flattening = [(1_000_000, 1e-3), (2_000_000, 1.2e-3)]
    assert cal.interp_step_time(flattening, 4_000_000) == pytest.approx(2.4e-3, rel=1e-9)
    assert cal.interp_step_time(flattening, 10_000_000) == pytest.approx(6.0e-3, rel=1e-9)


def test_below_the_smallest_probe_the_step_time_is_clamped():
    """Downward log-log extrapolation runs off toward absurdly fast.

    A step has a floor -- the launch, and the fixed per-kernel cost -- which
    the first A100 probe measured directly: 1.061 ms at 110,592 elements and
    1.001 ms at 373,248, 3.4x the work for the same time.
    """
    p_small, t_small = BLACKWELL_PROBES[0]
    for p_elems in (p_small // 2, p_small // 8, 1024, 1):
        assert cal.interp_step_time(BLACKWELL_PROBES, p_elems) == pytest.approx(t_small, rel=1e-12)
    # ... where continuing the first segment's slope (1.366) predicts 0.128 ms
    # for a grid an eighth the size: a 17x better per-element rate than
    # anything this device has ever shown.
    slope = math.log(BLACKWELL_PROBES[1][1] / t_small) / math.log(BLACKWELL_PROBES[1][0] / p_small)
    naive = t_small * ((p_small // 8) / p_small) ** slope
    assert naive * 1e3 == pytest.approx(0.128, abs=0.002)
    assert cal.interp_step_time(BLACKWELL_PROBES, p_small // 8) > 15 * naive


def test_a_plan_reads_the_samples_not_the_stored_coefficients(tmp_path):
    """The interpolator wins whenever the entry carries samples."""
    grid, med, src = tiny_setup()
    _, n, _ = model.fft_sizes(grid.shape)
    calfile = tmp_path / "calibration.json"
    entry = {
        # (a, b) that would predict something quite different, on purpose.
        "a": 1e-9,
        "b": 2e-10,
        "fit_mode": "interpolated",
        "global_fit_mode": "throughput-anchored",
        "fit_max_rel_residual": 0.705,  # the global law's number: must NOT warn now
        "interp_loo_rel_err": 0.04,
        "backend": "cupy",
        "warmup_s": 3.0,
        "p_max_probe": 4 * n,
        "samples": [[n // 2, 5e-4], [4 * n, 5e-3]],
        "shapes": [],
    }
    calfile.write_text(json.dumps({"version": 1, "devices": {"NVIDIA A100-SXM4-40GB": entry}}))
    est = planner.estimate(grid, med, src, gpu="A100", calibration_path=calfile)
    assert est.source == "calibrated"
    assert est.t_step_s == pytest.approx(cal.interp_step_time([(n // 2, 5e-4), (4 * n, 5e-3)], n))
    assert est.t_step_s != pytest.approx(model.step_time(1e-9, 2e-10, n))
    # the GLOBAL fit's 70% miss is a diagnostic about a model nothing is
    # using any more; the interpolator's own 4% held-out error is fine.
    assert not any("misses" in w for w in est.warnings), est.warnings

    # Raise the leave-one-out error past the limit and the plan says so.
    entry["interp_loo_rel_err"] = 0.31
    calfile.write_text(json.dumps({"version": 1, "devices": {"NVIDIA A100-SXM4-40GB": entry}}))
    est = planner.estimate(grid, med, src, gpu="A100", calibration_path=calfile)
    assert any("HELD-OUT probe by up to 31%" in w for w in est.warnings), est.warnings
    assert any("leave-one-out" in w for w in est.warnings)


def test_an_interpolating_entry_still_says_when_it_extrapolates(tmp_path):
    """Rule 3: the extrapolation warning is untouched by any of this."""
    grid, med, src = tiny_setup()
    _, n, _ = model.fft_sizes(grid.shape)
    calfile = tmp_path / "calibration.json"
    entry = {
        "a": 0.0,
        "b": 2e-10,
        "backend": "cupy",
        "warmup_s": 3.0,
        "samples": [[n // 32, 3.2e-8], [n // 16, 6.4e-8]],
        "shapes": [],
        "p_max_probe": n // 16,  # the plan is 16x the biggest probe
        "fit_mode": "interpolated",
        "interp_loo_rel_err": None,  # two samples: nothing to hold out
    }
    calfile.write_text(json.dumps({"version": 1, "devices": {"NVIDIA A100-SXM4-40GB": entry}}))
    est = planner.estimate(grid, med, src, gpu="A100", calibration_path=calfile)
    assert est.source == "calibrated"
    assert any("extrapolated" in w for w in est.warnings), est.warnings
    assert not any("misses" in w for w in est.warnings), est.warnings
    # ... and the extrapolated number is the floored one: the top segment is
    # sublinear (2x the size for 2x the time is slope 1.0 -- exactly the
    # floor), so the rate stays at the best measured.
    assert est.t_step_s / n == pytest.approx(6.4e-8 / (n // 16), rel=1e-9)

    entry["p_max_probe"] = n
    calfile.write_text(json.dumps({"version": 1, "devices": {"NVIDIA A100-SXM4-40GB": entry}}))
    est = planner.estimate(grid, med, src, gpu="A100", calibration_path=calfile)
    assert not any("extrapolated" in w for w in est.warnings), est.warnings


def test_an_old_style_entry_behaves_exactly_as_it_did(tmp_path):
    """Stores written before the interpolator carry (a, b) and no samples.

    Pinned byte for byte: same t_step, same warning, same everything. A
    silent change of meaning for entries already on disk would be the worst
    possible way to ship an accuracy fix.
    """
    grid, med, src = tiny_setup()
    _, n, _ = model.fft_sizes(grid.shape)
    calfile = tmp_path / "calibration.json"
    for samples in ([], None):  # missing key and empty list are both "old"
        entry = {
            "a": 1e-9,
            "b": 2e-10,
            "backend": "cupy",
            "warmup_s": 3.0,
            "fit_max_rel_residual": 0.42,
            "fit_mode": "throughput-anchored",
        }
        if samples is not None:
            entry["samples"] = samples
        calfile.write_text(json.dumps({"version": 1, "devices": {"NVIDIA A100-SXM4-40GB": entry}}))
        est = planner.estimate(grid, med, src, gpu="A100", calibration_path=calfile)
        assert est.source == "calibrated"
        assert est.t_step_s == model.step_time(1e-9, 2e-10, n)  # exactly, not approx
        assert cal.time_model_quality(entry) == ("fit", 0.42)
        assert any("misses its own samples by up to 42%" in w for w in est.warnings), est.warnings
        assert not any("leave-one-out" in w for w in est.warnings)

    # A single sample is not enough to interpolate either.
    assert cal.time_model_quality({"a": 1.0, "b": 2.0, "samples": [[1024, 1e-3]]})[0] == "fit"


def test_calibrate_records_the_interpolated_mode_and_its_loo(tmp_path):
    calfile = tmp_path / "calibration.json"
    entry = planner.calibrate(
        shapes=((16, 16), (24, 24), (32, 32)), backend="numpy", n_steps=4, path=calfile
    )
    assert entry["fit_mode"] == "interpolated"
    assert entry["global_fit_mode"] in ("nnls", "throughput-anchored")
    assert entry["a"] >= 0.0 and entry["b"] >= 0.0  # kept, for older readers
    assert entry["fit_max_rel_residual"] >= 0.0  # kept, as the global law's diagnostic
    loo = entry["interp_loo_rel_err"]
    assert loo is not None and math.isfinite(loo) and loo >= 0.0
    assert len(entry["samples"]) == 3

    # It survives the JSON round-trip and drives the plan.
    stored = json.loads(calfile.read_text())["devices"]["cpu"]
    assert cal.time_model_quality(stored) == ("interpolated", pytest.approx(loo))
    grid, med, src = tiny_setup()
    _, n, _ = model.fft_sizes(grid.shape)
    est = planner.estimate(grid, med, src, gpu=cpu_target(), calibration_path=calfile)
    assert est.source == "calibrated"
    assert est.t_step_s == pytest.approx(cal.predict_step_time(stored, n))

    # Two shapes -- the default cpu probe -- still calibrates; there is just
    # no interior sample to hold out, and it says None rather than 0.
    two = planner.calibrate(shapes=((16, 16), (24, 24)), backend="numpy", n_steps=4, path=calfile)
    assert two["fit_mode"] == "interpolated"
    assert two["interp_loo_rel_err"] is None
    assert cal.time_model_quality(two) == ("interpolated", None)


def test_a_bigger_grid_is_never_predicted_cheaper_than_a_smaller_one():
    """The first A100 calibration measured 1.061 ms at 110,592 elements and
    1.001 ms at 373,248 -- both latency-bound, the difference pure noise.
    Interpolating that verbatim would have a 3.4x bigger grid running faster."""
    noisy = [(110_592, 0.0010611400000470894), (373_248, 0.0010005005001403333)]
    predictions = [cal.interp_step_time(noisy, p) for p, _ in noisy]
    assert predictions[0] == pytest.approx(0.00106114, rel=1e-6)
    assert predictions[1] == pytest.approx(predictions[0])  # clamped up, not down
    assert cal.interp_step_time(noisy, 200_000) == pytest.approx(predictions[0])


# ----------------------------- the warmup curve is interpolated too (M8.time)
#
# Subtracting each rung's measured warmup from its wall time, the anchored
# per-step model was only -6..-9% out. M8.time failed on the OTHER half: the
# plans used the probe's warmup (0.2/0.6/1.2/2.3 s) while the real runs paid
# 4.389/12.838/32.375 s, and those are superlinear in P. record_warmup_model
# wrote them back, the linear refit landed on a NEGATIVE context -- clamped
# to zero -- and the next check (a real 192^3 run) caught it over-predicting
# 2x. Same disease as the step curve, same medicine.

BLACKWELL_WARMUPS = [(400**3, 4.389), (512**3, 12.838), (640**3, 32.375)]  # measured
BLACKWELL_WARMUP_SLOPE = 1.4264024314884076e-07  # what the linear refit stored


def test_the_warmup_curve_is_superlinear_and_the_linear_model_cannot_bend(tmp_path):
    device = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    calfile = tmp_path / "calibration.json"
    calfile.write_text(
        json.dumps({"version": 1, "devices": {device: {"a": 0.0, "b": BLACKWELL_B}}})
    )
    entry = cal.record_warmup_model(device, BLACKWELL_WARMUPS, path=calfile)

    rates = [w / p for p, w in BLACKWELL_WARMUPS]
    assert rates == sorted(rates)  # 0.069 -> 0.096 -> 0.124 us/element
    assert rates[-1] / rates[0] == pytest.approx(1.80, abs=0.01)

    # (a) the linear refit, and what it costs. Both numbers are the session's.
    assert entry["warmup_per_elem_s"] == pytest.approx(BLACKWELL_WARMUP_SLOPE, rel=1e-6)
    assert entry["warmup_context_s"] == 0.0  # the intercept went negative (-5.35 s)
    assert entry["warmup_fit_max_rel_residual"] == pytest.approx(1.080, abs=0.005)
    assert BLACKWELL_WARMUP_SLOPE * 400**3 == pytest.approx(9.129, abs=0.01)  # vs 4.389 measured

    # (b) interpolation reproduces every measured warmup exactly, and says so
    # with a held-out number instead of a residual it cannot fail.
    assert entry["warmup_mode"] == "interpolated"
    assert entry["warmup_loo_rel_err"] == pytest.approx(0.0235, abs=0.001)
    for p_elems, w in BLACKWELL_WARMUPS:
        assert cal.calibrated_warmup(entry, p_elems) == pytest.approx(w, rel=1e-12)
    assert cal.warmup_model_quality(entry) == ("interpolated", pytest.approx(0.0235, abs=0.001))


def test_below_the_smallest_warmup_sample_the_per_element_rate_is_held():
    """The one measurement below the sampled range decides this rule.

    A real 192^3 run paid 0.501 s of warmup. Holding the smallest sample's
    per-element rate predicts 0.485 s (-3.1%). Holding its VALUE -- the rule
    the step curve uses -- would say 4.389 s (+776%), and the linear two-term
    model said 1.010 s (+101%), which is what caught the defect.
    """
    p192 = 192**3
    predicted = cal.interp_warmup(BLACKWELL_WARMUPS, p192)
    assert predicted == pytest.approx(0.4854, abs=0.001)
    assert predicted == pytest.approx(0.501, rel=0.05)
    assert BLACKWELL_WARMUP_SLOPE * p192 == pytest.approx(1.010, abs=0.005)
    assert predicted < BLACKWELL_WARMUPS[0][1]  # never MORE than the smallest sample
    # proportional below the range, so half the grid is half the plan cost
    assert cal.interp_warmup(BLACKWELL_WARMUPS, p192 // 2) == pytest.approx(predicted / 2, rel=1e-9)
    assert cal.interp_warmup(BLACKWELL_WARMUPS, 0) == 0.0


def test_the_two_curves_clamp_differently_below_their_samples_on_purpose():
    """Step time is floored, warmup is not -- because they are different
    costs. A kernel launch costs what it costs however small the grid; a
    cuFFT plan for a smaller transform is simply a smaller plan."""
    tiny = 64**3
    assert cal.interp_step_time(BLACKWELL_PROBES, tiny) == BLACKWELL_PROBES[0][1]
    assert cal.interp_warmup(BLACKWELL_WARMUPS, tiny) < 0.1 * BLACKWELL_WARMUPS[0][1]
    # ... and above their samples they share one rule: never a better
    # per-element rate than the best measured.
    for samples, fn in (
        (BLACKWELL_PROBES, cal.interp_step_time),
        (BLACKWELL_WARMUPS, cal.interp_warmup),
    ):
        p_top, c_top = samples[-1]
        huge = 4 * p_top
        assert fn(samples, huge) / huge >= c_top / p_top - 1e-18


def test_a_plan_reads_the_warmup_samples_not_the_stored_two_term_model(tmp_path):
    grid, med, src = tiny_setup()
    _, n, _ = model.fft_sizes(grid.shape)
    calfile = tmp_path / "calibration.json"
    entry = {
        "a": 0.0,
        "b": 2e-10,
        "backend": "cupy",
        "samples": [],
        "shapes": [],
        # a two-term model that would say something quite different (7.5 s)
        "warmup_context_s": 5.0,
        "warmup_per_elem_s": 2.44e-3,
        "warmup_s": 99.0,
        "warmup_samples": [[n // 2, 0.4], [2 * n, 2.0]],
    }
    calfile.write_text(json.dumps({"version": 1, "devices": {"NVIDIA A100-SXM4-40GB": entry}}))
    est = planner.estimate(grid, med, src, gpu="A100", calibration_path=calfile)
    assert est.source == "calibrated"
    assert est.warmup_s == pytest.approx(cal.interp_warmup([(n // 2, 0.4), (2 * n, 2.0)], n))
    assert est.warmup_s == pytest.approx(0.8944, abs=0.001)  # not 7.5, and not 99.0
    assert est.t_expected_s == pytest.approx(est.warmup_s + est.t_step_s * est.steps_expected)


def test_asserting_one_flat_warmup_retires_the_curve_it_contradicts(tmp_path):
    """record_warmup() is the caller saying "it cost THIS". Leaving a stale
    multi-sample curve behind would silently outrank that assertion, since
    the samples are what predicts now."""
    device = "cpu"
    calfile = tmp_path / "calibration.json"
    calfile.write_text(json.dumps({"version": 1, "devices": {device: {"a": 0.0, "b": 1e-10}}}))
    cal.record_warmup_model(device, [(1_000_000, 1.0), (8_000_000, 12.0)], path=calfile)
    assert cal.find_calibration_for(device, calfile)["warmup_samples"] == [
        [1_000_000, 1.0],
        [8_000_000, 12.0],
    ]

    flat = cal.record_warmup(device, 3.0, path=calfile)
    assert "warmup_samples" not in flat
    assert cal.calibrated_warmup(flat, 8_000_000) == pytest.approx(3.0)  # the assertion wins

    sized = cal.record_warmup(device, 5.0, p_elems=4_000_000, path=calfile)
    assert sized["warmup_samples"] == [[4_000_000, 5.0]]  # one point is not a curve
    assert cal.calibrated_warmup(sized, 4_000_000) == pytest.approx(5.0)


def test_entry_samples_survives_whatever_is_on_disk():
    assert cal.entry_samples({}) == []
    assert cal.entry_samples({"samples": None}) == []
    assert cal.entry_samples({"samples": [[1024, 1e-3], [512, 2e-3]]}) == [
        (512, 2e-3),
        (1024, 1e-3),
    ]  # sorted by P
    junk = {"samples": [[0, 1e-3], [1024, 0.0], [2048, "x"], [4096], None, [8192, 1e-3]]}
    assert cal.entry_samples(junk) == [(8192, 1e-3)]
    # a size measured twice keeps the slower time
    assert cal.entry_samples({"samples": [[1024, 1e-3], [1024, 3e-3]]}) == [(1024, 3e-3)]


def test_a_calibrated_plan_says_when_it_is_extrapolating(tmp_path):
    """'calibrated' is a promise about accuracy; this is where it lapses."""
    calfile = tmp_path / "calibration.json"
    grid, med, src = tiny_setup()
    p_elems = model.kspace_memory(grid.shape, False, 1, rec_elems=1).padded_shape
    n = 1
    for x in p_elems:
        n *= x

    entry = {
        "a": 0.0,
        "b": 2e-10,
        "backend": "cupy",
        "samples": [],
        "shapes": [],
        "warmup_s": 3.0,
        "p_max_probe": max(1, n // 100),  # plan is 100x the probe
        "fit_max_rel_residual": 0.0,
        "fit_mode": "nnls",
    }
    calfile.write_text(json.dumps({"version": 1, "devices": {"NVIDIA A100-SXM4-40GB": entry}}))
    est = planner.estimate(grid, med, src, gpu="A100", calibration_path=calfile)
    assert est.source == "calibrated"
    assert any("extrapolated" in w for w in est.warnings), est.warnings

    # Same entry, probe as big as the plan: no extrapolation, no warning.
    entry["p_max_probe"] = n
    calfile.write_text(json.dumps({"version": 1, "devices": {"NVIDIA A100-SXM4-40GB": entry}}))
    est = planner.estimate(grid, med, src, gpu="A100", calibration_path=calfile)
    assert not any("extrapolated" in w for w in est.warnings), est.warnings


def test_a_calibration_that_misses_its_own_samples_warns_every_plan(tmp_path):
    calfile = tmp_path / "calibration.json"
    entry = {
        "a": 0.0,
        "b": 2e-10,
        "backend": "cupy",
        "samples": [],
        "shapes": [],
        "warmup_s": 3.0,
        "fit_max_rel_residual": 0.42,
        "fit_mode": "throughput-anchored",
    }
    calfile.write_text(json.dumps({"version": 1, "devices": {"NVIDIA A100-SXM4-40GB": entry}}))
    grid, med, src = tiny_setup()
    est = planner.estimate(grid, med, src, gpu="A100", calibration_path=calfile)
    assert any("misses its own samples" in w for w in est.warnings), est.warnings


def test_the_cpu_probe_stays_small(tmp_path):
    """M8's gate is about a GPU; a CPU calibration must not take minutes."""
    assert cal.default_probe_shapes("numpy") == ((48, 48, 48), (72, 72, 72))


def test_the_calibration_probe_does_not_overflow_float32():
    """A probe that overflows is not timing the arithmetic a solve does.

    The synthetic step multiplies by the spectral derivative, i.e. by |k| up
    to pi/dx; with the original 1e-3 coefficients the per-step gain was ~3
    and p reached inf inside the timed loop on every nonlinear probe.
    """
    import warnings as _warnings

    for nonlinear in (False, True):
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", RuntimeWarning)
            run = cal.measure_step_time(
                (32, 32, 32), nonlinear=nonlinear, backend="numpy", n_steps=4
            )
        assert run["t_step_s"] > 0.0


# -------------------------------------- unrecognised devices (M8, fix F4)


def test_an_unrecognised_gpu_is_not_silently_relabelled():
    assert planner.gpu_key_for_device("Acme Frobnicator 9000") is None
    spec = planner.spec_for_device(
        "NVIDIA RTX PRO 6000 Blackwell Server Edition", int(94.971 * 2**30)
    )
    assert spec.source == "device"
    assert spec.key.startswith("unknown:")
    assert spec.device_name == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert spec.vram_gib == pytest.approx(94.971, rel=1e-3)


def test_a_known_gpu_still_resolves_to_its_datasheet_row():
    spec = planner.spec_for_device("NVIDIA A100-SXM4-40GB", int(39.494 * 2**30))
    assert spec.key == "A100-40GB"
    assert spec.source == "db"
    assert spec.device_name == "NVIDIA A100-SXM4-40GB"  # carried, for calibration lookup


def test_an_unknown_device_reaches_its_own_calibration(tmp_path):
    """The Blackwell measured a calibration and then ignored it.

    ``find_calibration_for`` searched for the datasheet key's first token in
    the device name; "a100" is not in "NVIDIA RTX PRO 6000 Blackwell Server
    Edition", so the freshly measured entry was invisible and the plan fell
    back to ``db`` -- which is why M8's time gate could not even be graded
    on that device (2026-08-22).
    """
    device = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    calfile = tmp_path / "calibration.json"
    entry = {
        "a": 0.0,
        "b": 9.753e-10,
        "backend": "cupy",
        "samples": [],
        "shapes": [],
        "warmup_s": 1.52,
    }
    calfile.write_text(json.dumps({"version": 1, "devices": {device: entry}}))

    assert cal.find_calibration_for("A100-40GB", calfile) is None  # the old path: invisible
    assert cal.find_calibration_for("unknown:x", calfile, device_name=device) == entry

    spec = planner.spec_for_device(device, int(94.971 * 2**30))
    grid, med, src = tiny_setup()
    est = planner.estimate(grid, med, src, gpu=spec, calibration_path=calfile)
    assert est.source == "calibrated"


def test_a_big_unknown_card_is_not_judged_against_a100_vram():
    """A 95 GiB card was judged against an A100's 38.88 GiB of usable VRAM.

    Sizing the check from the two specs rather than hardcoding a shape: the
    guarantee is that a run BETWEEN the two capacities is refused on the
    A100 and accepted on the card that can actually hold it.
    """
    device = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    spec = planner.spec_for_device(device, int(94.971 * 2**30))
    a100 = planner._resolve_gpu("A100-40GB")
    assert spec.usable_bytes > 2 * a100.usable_bytes

    between = None
    for side in range(320, 1000, 16):
        mem = model.kspace_memory((side,) * 3, True, 1, rec_elems=side**3)
        if a100.usable_bytes < mem.total_bytes <= spec.usable_bytes:
            between = (side, mem)
            break
    assert between is not None, "no shape sits between the two capacities"
    side, mem = between
    assert mem.total_bytes > a100.usable_bytes
    assert mem.total_bytes <= spec.usable_bytes


def test_real_runs_refit_both_warmup_terms(tmp_path):
    """The gate suite's rungs are better warmup data than the probe.

    They are whole solves, at several sizes, each in its own process — so
    every sample paid the CUDA context and its own plan creation, which is
    exactly what the two terms mean. Writing back one flat number instead
    (what the suite used to do) collapses the model to a constant and
    re-loses the term that is 46% of a 512^3 run's wall time.
    """
    calfile = tmp_path / "calibration.json"
    device = "NVIDIA A100-SXM4-40GB"
    calfile.write_text(
        json.dumps(
            {
                "version": 1,
                "devices": {
                    device: {
                        "a": 0.0,
                        "b": 2e-10,
                        "warmup_context_s": 9.9,
                        "warmup_per_elem_s": 9.9e-7,
                    }
                },
            }
        )
    )

    c0, c1 = 1.81, 1.6258e-07
    samples = [(n**3, c0 + c1 * n**3) for n in (256, 400, 512)]
    entry = cal.record_warmup_model(device, samples, path=calfile)
    assert entry["warmup_context_s"] == pytest.approx(c0, rel=1e-3)
    assert entry["warmup_per_elem_s"] == pytest.approx(c1, rel=1e-3)
    assert entry["warmup_source"] == "measured"
    assert entry["warmup_samples"] == [[n**3, c0 + c1 * n**3] for n in (256, 400, 512)]

    # The samples are also what PREDICTS now (see the warmup-curve tests
    # below): on a device that really is linear the two agree, so recording
    # them costs nothing here. Above the top sample the interpolator's
    # per-element floor makes it the more conservative of the two.
    for n in (256, 400, 512):
        assert cal.calibrated_warmup(entry, n**3) == pytest.approx(c0 + c1 * n**3, rel=1e-9)
    assert cal.calibrated_warmup(entry, 640**3) == pytest.approx(c0 + c1 * 640**3, rel=0.05)
    assert cal.calibrated_warmup(entry, 640**3) >= c0 + c1 * 640**3

    # ... and the two-term model itself is untouched for an entry without
    # samples, which is what every store written before today looks like.
    two_term = {"warmup_context_s": c0, "warmup_per_elem_s": c1}
    assert cal.calibrated_warmup(two_term, 640**3) == pytest.approx(c0 + c1 * 640**3, rel=1e-12)


def test_one_warmup_sample_moves_the_constant_and_keeps_the_slope(tmp_path):
    calfile = tmp_path / "calibration.json"
    device = "cpu"
    calfile.write_text(
        json.dumps(
            {
                "version": 1,
                "devices": {
                    device: {
                        "a": 0.0,
                        "b": 1e-10,
                        "warmup_context_s": 0.5,
                        "warmup_per_elem_s": 2e-08,
                    }
                },
            }
        )
    )
    entry = cal.record_warmup_model(device, [(1_000_000, 3.0)], path=calfile)
    assert entry["warmup_per_elem_s"] == pytest.approx(2e-08)  # slope kept
    assert entry["warmup_context_s"] == pytest.approx(3.0 - 2e-08 * 1_000_000)


def test_a_falling_warmup_is_treated_as_noise_not_a_model(tmp_path):
    calfile = tmp_path / "calibration.json"
    device = "cpu"
    calfile.write_text(json.dumps({"version": 1, "devices": {device: {"a": 0.0, "b": 1e-10}}}))
    entry = cal.record_warmup_model(device, [(1_000_000, 4.0), (8_000_000, 2.0)], path=calfile)
    assert entry["warmup_per_elem_s"] == 0.0
    assert entry["warmup_context_s"] == pytest.approx(3.0)


def test_recording_a_warmup_for_an_uncalibrated_device_invents_nothing(tmp_path):
    calfile = tmp_path / "calibration.json"
    calfile.write_text(json.dumps({"version": 1, "devices": {}}))
    assert cal.record_warmup_model("no-such-device", [(10, 1.0)], path=calfile) is None
