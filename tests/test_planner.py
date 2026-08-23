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
    assert cal.calibrated_warmup(entry, 640**3) == pytest.approx(c0 + c1 * 640**3, rel=1e-3)


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
