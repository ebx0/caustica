"""Everything about the GPU gate suite that does NOT need a GPU.

The suite exists to close M7's and M8's on-device criteria, so its expensive
half runs exactly once, on somebody else's machine, in a session that is
awkward to repeat. That makes the cheap half — ladder sizing, the verdict
algebra, the report schema, the refusal branch, the no-GPU exit — the part
most worth pinning, and :class:`caustica.validation.gpu_gates.Harness` is the
seam that makes it possible: a fake device answers every question the suite
asks of hardware, with numbers this file chooses.

The one thing deliberately NOT done here is producing GPU numbers. A fake
that reported a passing A100 would make the milestone boxes tickable from a
laptop, which is the exact failure this suite exists to prevent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from caustica.validation import gpu_gates as gg

A100 = "NVIDIA A100-SXM4-40GB"


# --------------------------------------------------------------- fake device


@dataclass
class FakeDevice:
    """A device whose model error is a knob.

    ``run_job`` reads the job the suite wrote, prices it with the planner's
    OWN memory inventory (so the fake cannot flatter the model), then reports
    a "measured" value that is off by exactly ``vram_error`` / ``time_error``.
    A job priced above ``capacity_gib`` is refused the way the runner refuses
    one: exit 3, ``error.json``, advice.
    """

    capacity_gib: float = 39.0
    vram_error: float = 0.0  # measured = predicted / (1 + error)
    time_error: float = 0.0
    solve_oversized: bool = False  # a broken device that runs what it cannot fit
    steps_mismatch: bool = False
    plan_source: str = "calibrated"  # what plan.json says the timing came from
    calibration_raises: bool = False
    exit_code_after_stamp: int = 0  # a run that stamped everything and still failed
    warmup_s: float = 2.5
    calls: list[tuple[str, str]] = field(default_factory=list)
    recorded: list[tuple[str, float]] = field(default_factory=list)

    # -- the harness's questions -------------------------------------------
    def require_gpu(self) -> None:
        return None

    def env_report(self) -> dict:
        return {
            "caustica": "0.0.0-test",
            "python": "3.13.15",
            "resolved_backend": "cupy",
            "gpu_name": A100,
            "cupy_version": "14.0.1",
            "vram_total_gib": 39.5,
            "vram_free_gib": self.capacity_gib,
        }

    def device_name(self) -> str:
        return A100

    def free_vram_gib(self) -> float:
        return self.capacity_gib

    def calibrate(self, **kw) -> dict:
        if self.calibration_raises:
            raise RuntimeError("synthetic calibration failure")
        return {"a": 1e-11, "b": 2e-11, "warmup_s": self.warmup_s, "backend": "cupy"}

    def record_warmup(self, device: str, warmup_s: float):
        self.recorded.append((device, warmup_s))
        return {"warmup_s": warmup_s}

    # -- the expensive part, faked ------------------------------------------
    def run_job(self, job_path, options) -> int:
        job = json.loads(Path(job_path).read_text(encoding="utf-8"))
        out = Path(options.out)
        out.mkdir(parents=True, exist_ok=True)
        self.calls.append((job["name"], options.backend))

        shape = tuple(int(round(s / job["grid"]["dx_mm"])) for s in job["grid"]["size_mm"])
        predicted_gib = gg._predicted_gib(shape, job["solver"])
        steps = 128
        t_step = 4e-3
        predicted_s = self.warmup_s + steps * t_step

        if predicted_gib > self.capacity_gib and not self.solve_oversized:
            (out / "error.json").write_text(
                json.dumps(
                    {
                        "format": "caustica-error/1",
                        "stage": "gate",
                        "exit_code": 3,
                        "error_class": "VramRefusal",
                        "message": f"needs {predicted_gib:.2f} GiB, free is {self.capacity_gib}",
                        "advice": ["increase dx by >= x1.10", "or pick a larger device: H100-SXM"],
                    }
                ),
                encoding="utf-8",
            )
            return 3

        (out / "plan.json").write_text(
            json.dumps(
                {
                    "source": self.plan_source,
                    "spp": 16,
                    "t_step_s": t_step,
                    "warmup_s": self.warmup_s,
                    "steps_expected": steps,
                    "steps_worst": steps,
                    "t_expected_s": predicted_s,
                    "vram_gib": predicted_gib,
                }
            ),
            encoding="utf-8",
        )
        (out / "run_meta.json").write_text(
            json.dumps(
                {
                    "format": "caustica-run-meta/1",
                    "actual": {
                        "elapsed_solve_s": predicted_s / (1.0 + self.time_error),
                        "steps_total": steps + (7 if self.steps_mismatch else 0),
                        "t_step_measured_s": t_step,
                        "t_step_steady_s": t_step * 0.9,
                        "warmup_s": self.warmup_s + 1.0,
                        "steady_samples": 9,
                        "vram_pool_peak_gib": predicted_gib / (1.0 + self.vram_error),
                        "converged_period": 3,
                    },
                }
            ),
            encoding="utf-8",
        )
        return self.exit_code_after_stamp

    def harness(self) -> gg.Harness:
        return gg.Harness(
            require_gpu=self.require_gpu,
            env_report=self.env_report,
            device_name=self.device_name,
            free_vram_gib=self.free_vram_gib,
            calibrate=self.calibrate,
            run_job=self.run_job,
            record_warmup=self.record_warmup,
        )


def run_suite(tmp_path: Path, device: FakeDevice, **kw):
    kw.setdefault("parity", False)  # parity needs real fields; tested separately
    return gg.gpu_gates(
        out=tmp_path / "reports", harness=device.harness(), log=lambda _: None, **kw
    )


def gate_of(payload: dict, gate_id: str) -> dict:
    (g,) = [g for g in payload["gates"] if g["id"] == gate_id]
    return g


# -------------------------------------------------------------- the algebra


def test_a_check_that_could_not_be_evaluated_is_never_a_pass():
    """SKIP is the whole point: the easiest way for a suite like this to lie
    is to treat a missing measurement as agreement."""
    assert gg.Check.relative("x", None, 1.0, 10.0).verdict == "SKIP"
    assert gg.Check.relative("x", 1.0, None, 10.0).verdict == "SKIP"
    assert gg.Check.relative("x", 0.0, 0.0, 10.0).verdict == "SKIP"  # not "0 == 0, pass"
    assert gg.Check.at_most("x", None, 1e-5).verdict == "SKIP"
    assert gg.Check.at_most("x", float("nan"), 1e-5).verdict == "SKIP"
    assert gg.Check.happened("x", None, 3).verdict == "SKIP"

    assert gg.Check.relative("x", 1.05, 1.0, 10.0).verdict == "PASS"
    assert gg.Check.relative("x", 1.11, 1.0, 10.0).verdict == "FAIL"
    assert gg.Check.relative("x", 0.89, 1.0, 10.0).verdict == "FAIL"  # under-prediction too
    assert gg.Check.at_most("x", 1e-6, 1e-5).verdict == "PASS"
    assert gg.Check.happened("x", 3, 3).verdict == "PASS"


def test_a_gate_needs_its_milestones_COUNT_of_passes_and_no_failures():
    ok = gg.Check.relative("a", 1.0, 1.0, 10.0)
    bad = gg.Check.relative("b", 2.0, 1.0, 10.0)
    skipped = gg.Check.relative("c", None, 1.0, 10.0)

    assert gg.Gate("g", "c", 2, [ok, ok]).verdict == "PASS"
    assert gg.Gate("g", "c", 2, [ok]).verdict == "INCOMPLETE"  # "at least 2 grid sizes"
    assert gg.Gate("g", "c", 2, [ok, skipped]).verdict == "INCOMPLETE"
    assert gg.Gate("g", "c", 2, [ok, ok, bad]).verdict == "FAIL"  # one failure poisons it
    assert gg.Gate("g", "c", 1, []).verdict == "INCOMPLETE"  # nothing measured
    assert gg.Gate("g", "c", 0, []).verdict == "INCOMPLETE"  # required=0 cannot smuggle a PASS

    assert gg.overall_verdict([]) == "INCOMPLETE"
    assert gg.overall_verdict([gg.Gate("g", "c", 1, [ok])]) == "PASS"
    assert gg.overall_verdict([gg.Gate("g", "c", 1, [ok]), gg.Gate("h", "c", 2, [ok])]) == (
        "INCOMPLETE"
    )
    assert gg.overall_verdict([gg.Gate("g", "c", 1, [ok]), gg.Gate("h", "c", 1, [bad])]) == "FAIL"


# ---------------------------------------------------------------- the ladder


def test_ladder_sides_are_shapes_the_engine_will_not_pad():
    """A non-smooth side would be padded up by the engine, so the plan would
    describe a grid that never ran and the VRAM gate would measure the wrong
    thing."""
    from caustica.solvers.kspace import operators as ops

    for side in gg.smooth_sides(16, 700):
        assert ops.pad_shape((side, side, side)) == (side, side, side)
    assert 512 in gg.smooth_sides(16, 700)
    assert 511 not in gg.smooth_sides(16, 700)

    side = gg.side_for_target(8 * 2**30)
    assert side is not None
    assert gg._predicted_gib((side,) * 3, "westervelt") <= 8.0
    bigger = gg.smooth_sides(side + 2, side + 200)
    assert gg._predicted_gib((bigger[0],) * 3, "westervelt") > 8.0  # it really is the largest


def test_the_ladder_is_clipped_to_the_device_and_still_gives_two_points():
    big = gg.build_ladder(39.0)
    fitting = [r for r in big if r.expect_fit]
    assert len(fitting) >= 2
    assert all(r.predicted_gib <= 39.0 * gg.LADDER_HEADROOM for r in fitting)
    assert any(r.shape == (512, 512, 512) for r in fitting), "the M7 full-size rung is missing"

    # A 16 GiB T4 cannot hold the 28 GiB target; it must still produce the two
    # measurements M8's criterion counts.
    small = [r for r in gg.build_ladder(15.0) if r.expect_fit]
    assert len(small) >= 2
    assert all(r.predicted_gib <= 15.0 for r in small)

    # And a device with almost nothing free still yields a ladder rather than
    # an empty one that would silently make every gate INCOMPLETE for the
    # wrong reason.
    tiny = [r for r in gg.build_ladder(1.0) if r.expect_fit]
    assert len(tiny) >= 2


@pytest.mark.parametrize("free_gib", [15.0, 39.0, 79.0, 140.0])
def test_the_ladder_ends_with_a_rung_that_must_not_fit(free_gib):
    """Sized as "the smallest shape that does NOT fit", not "the largest under
    1.15x free" — the latter picked a 75 GiB rung on an 80 GiB device, which
    fits, and would have graded the refusal gate on a run nobody meant to
    refuse (found by parametrizing this test, 2026-08-23)."""
    ladder = gg.build_ladder(free_gib)
    oom = [r for r in ladder if not r.expect_fit]
    assert len(oom) == 1
    assert oom[0].predicted_gib > free_gib
    assert oom[0].shape[0] > max(r.shape[0] for r in ladder if r.expect_fit)
    assert gg.build_ladder(free_gib, oom_rung=False) == [r for r in ladder if r.expect_fit]


def test_max_vram_gib_shrinks_the_ladder_without_touching_the_device():
    ladder = [r for r in gg.build_ladder(39.0, max_vram_gib=6.0) if r.expect_fit]
    assert ladder and all(r.predicted_gib <= 6.0 for r in ladder)


def test_every_suite_job_is_self_contained_and_survives_the_real_validator(tmp_path):
    """A suite whose scenarios do not load is worth nothing on the night it
    is run, and a scenario that reaches for a file is not self-contained."""
    from caustica.config.job import validate_job

    jobs = [gg.rung_job(spec) for spec in gg.build_ladder(39.0)[:2]] + [gg.parity_job()]
    for job in jobs:
        assert "path" not in json.dumps(job)  # nothing points outside the library
        job_path = tmp_path / f"{job['name']}.json"
        job_path.write_text(json.dumps(job), encoding="utf-8")
        report = validate_job(job_path)
        assert report.ok, report.render()


def test_the_rung_medium_is_nonlinear_so_westervelt_buffers_are_real():
    """With beta = 0 the westervelt solver drops its nonlinear buffers, so the
    rung would measure a smaller run than the planner was asked about."""
    spec = gg.build_ladder(39.0)[0]
    assert gg.rung_job(spec)["medium"]["material"]["beta"] > 0.0


def test_gpu_key_is_detected_from_the_device_name():
    assert gg.gpu_key_for(A100, 39.5) == "A100-40GB"
    assert gg.gpu_key_for("NVIDIA A100-SXM4-80GB", 79.2) == "A100-80GB"
    assert gg.gpu_key_for("Tesla T4", 15.0) == "T4"
    assert gg.gpu_key_for("NVIDIA L4", 22.0) == "L4"
    assert gg.gpu_key_for("Some Future GPU", 1000.0) == "A100"  # falls back, never raises


# ------------------------------------------------------------- the protocol


def test_no_gpu_exits_2_and_names_the_fix(tmp_path):
    device = FakeDevice()

    def refuse():
        raise RuntimeError(
            "A GPU is required, but this Colab runtime has no CUDA device. "
            "Fix: Runtime -> Change runtime type -> Hardware accelerator: GPU"
        )

    harness = device.harness()
    harness.require_gpu = refuse
    code, payload = gg.gpu_gates(out=tmp_path, harness=harness, log=lambda _: None)

    assert code == gg.EXIT_ENV
    assert payload["verdict"] == "SKIPPED"
    assert "Runtime" in payload["reason"]
    assert device.calls == [], "nothing may be prepared before the environment verdict"


def test_the_real_cli_skips_cleanly_on_a_machine_without_a_gpu():
    """What CI runs. It must be a clean, actionable exit, not a traceback."""
    from caustica.core.backend import cupy_available
    from caustica.validation.__main__ import main

    if cupy_available():  # pragma: no cover - not this machine
        pytest.skip("this machine HAS a GPU; the real suite would run for minutes")
    assert main(["gpu-gates"]) == gg.EXIT_ENV


def test_a_faithful_model_passes_every_gate_it_measured(tmp_path):
    device = FakeDevice(vram_error=0.03, time_error=0.05)
    code, payload = run_suite(tmp_path, device)

    assert gate_of(payload, "M8.vram")["verdict"] == "PASS"
    assert gate_of(payload, "M8.time")["verdict"] == "PASS"
    assert gate_of(payload, "M7.fullsize")["verdict"] == "PASS"
    assert gate_of(payload, "M8.oom")["verdict"] == "PASS"
    # Parity was not run here, so its gate MUST stay open and the suite must
    # not claim overall success.
    assert gate_of(payload, "M7.parity")["verdict"] == "INCOMPLETE"
    assert payload["verdict"] == "INCOMPLETE"
    assert code == gg.EXIT_FAILED

    assert gate_of(payload, "M8.vram")["n_pass"] >= 2  # "at least 2 grid sizes"
    assert gate_of(payload, "M8.time")["n_pass"] >= 2  # "at least 2 scenarios"


def test_a_vram_model_off_by_twenty_percent_fails_the_vram_gate(tmp_path):
    _, payload = run_suite(tmp_path, FakeDevice(vram_error=0.20))
    vram = gate_of(payload, "M8.vram")
    assert vram["verdict"] == "FAIL"
    assert any(c["verdict"] == "FAIL" for c in vram["checks"])
    assert payload["verdict"] == "FAIL"


def test_a_time_model_off_by_forty_percent_fails_the_time_gate(tmp_path):
    _, payload = run_suite(tmp_path, FakeDevice(time_error=0.40))
    assert gate_of(payload, "M8.time")["verdict"] == "FAIL"
    assert gate_of(payload, "M8.vram")["verdict"] == "PASS"  # independent gates


def test_one_measurement_is_not_two(tmp_path):
    """M8 says "at least 2". A ladder with a single fitting rung leaves the
    gate INCOMPLETE — it does not get to pass on one good number."""
    _, payload = run_suite(tmp_path, FakeDevice(), targets_gib=(2.0,), oom_rung=False)
    fitting = [r for r in payload["rungs"] if r["expect_fit"]]
    assert len(fitting) == 1
    assert gate_of(payload, "M8.vram")["verdict"] == "INCOMPLETE"
    assert gate_of(payload, "M8.oom")["verdict"] == "INCOMPLETE"
    assert payload["verdict"] != "PASS"


def test_a_device_that_solves_what_it_cannot_fit_fails_the_refusal_gate(tmp_path):
    """The refusal is a gate, not a formality: a run that should have been
    refused and was solved instead is a FAIL, not a quietly-missing check."""
    _, payload = run_suite(tmp_path, FakeDevice(solve_oversized=True))
    oom = gate_of(payload, "M8.oom")
    assert oom["verdict"] == "FAIL"
    assert payload["verdict"] == "FAIL"


def test_a_step_count_that_does_not_match_the_plan_is_not_compared(tmp_path):
    """If the run took a different number of steps than the plan assumed, the
    wall-time comparison is measuring the convergence heuristic, not the
    timing model — so it is SKIPped rather than counted either way."""
    _, payload = run_suite(tmp_path, FakeDevice(steps_mismatch=True))
    timing = gate_of(payload, "M8.time")
    assert all(c["verdict"] == "SKIP" for c in timing["checks"])
    assert timing["verdict"] == "INCOMPLETE"
    assert gate_of(payload, "M8.vram")["verdict"] == "PASS"  # VRAM is still comparable


def test_a_datasheet_estimate_never_closes_the_post_calibration_time_gate(tmp_path):
    """M8's wording is "kalibrasyon SONRASI". A "db" plan is datasheet-coarse
    to about 2x, so it can land inside +/-25% by luck — and closing a
    calibration gate with an uncalibrated number is the quietest false PASS
    available to this suite."""
    _, payload = run_suite(tmp_path, FakeDevice(plan_source="db", time_error=0.01))
    timing = gate_of(payload, "M8.time")
    assert timing["verdict"] == "INCOMPLETE"
    assert all(c["verdict"] == "SKIP" for c in timing["checks"])
    assert any("not 'calibrated'" in c["detail"] for c in timing["checks"])
    # VRAM does not depend on calibration, so that gate is unaffected.
    assert gate_of(payload, "M8.vram")["verdict"] == "PASS"


def test_a_failed_calibration_costs_the_time_gate_not_the_session(tmp_path):
    """Fifteen minutes into a Colab session is the wrong moment to lose every
    measurement to a traceback — but it is also the wrong moment to grade
    M8's calibrated gate on plans that were never calibrated."""
    device = FakeDevice(calibration_raises=True, plan_source="db")
    _, payload = run_suite(tmp_path, device)

    assert "error" in payload["calibration"]
    assert any("calibration failed" in n for n in payload["notes"])
    assert gate_of(payload, "M8.time")["verdict"] == "INCOMPLETE"
    assert gate_of(payload, "M8.vram")["verdict"] == "PASS"  # still measured
    assert payload["rungs"], "the ladder still ran"


def test_a_rung_that_did_not_complete_contributes_nothing(tmp_path):
    """Exit code first, numbers second.

    A run can leave a full plan.json AND a run_meta.json behind and still have
    failed — the runner writes the plan before solving, and a future failure
    after stamping would leave both. Reading those numbers as a measurement
    would let a broken run close a milestone (mutation round, 2026-08-23:
    dropping this guard survived every other test).
    """
    _, payload = run_suite(tmp_path, FakeDevice(exit_code_after_stamp=4))
    for gate_id in ("M8.vram", "M8.time"):
        gate = gate_of(payload, gate_id)
        assert all(c["verdict"] == "SKIP" for c in gate["checks"]), gate
        assert gate["verdict"] == "INCOMPLETE"
    # M7's full-size criterion is different in kind: it asks whether the run
    # COMPLETED, and "it exited 4" is an answer, not a missing measurement.
    assert gate_of(payload, "M7.fullsize")["verdict"] == "FAIL"
    assert payload["verdict"] == "FAIL"
    # ... and a failed rung is not a step-time baseline either.
    assert payload["step_time_baseline"] == []


def test_the_report_is_written_stamped_and_readable(tmp_path):
    _, payload = run_suite(tmp_path, FakeDevice(vram_error=0.03))
    (folder,) = (tmp_path / "reports").iterdir()

    doc = json.loads((folder / "gpu_gates.json").read_text(encoding="utf-8"))
    assert doc["format"] == gg.FORMAT
    assert doc["device"] == A100
    assert doc["gpu_key"] == "A100-40GB"
    for key in ("generated", "caustica", "git_commit", "environment", "calibration"):
        assert doc[key], f"the report is not stamped with {key}"
    assert doc["gates"] and doc["rungs"]

    md = (folder / "REPORT.md").read_text(encoding="utf-8")
    assert gg.FORMAT in md and doc["verdict"] in md
    for g in doc["gates"]:
        assert g["id"] in md and g["criterion"] in md
    for r in doc["rungs"]:
        assert r["name"] in md
    assert "Step-time baseline" in md  # M19 reads this

    # Each rung keeps its own output folder next to the report: the evidence
    # is not a summary of evidence.
    for r in doc["rungs"]:
        assert (folder / "rungs" / r["name"] / "job_input.json").is_file()


def test_step_times_are_stamped_as_the_M19_baseline(tmp_path):
    _, payload = run_suite(tmp_path, FakeDevice())
    baseline = payload["step_time_baseline"]
    assert len(baseline) >= 2
    for row in baseline:
        assert row["voxels"] == row["shape"][0] ** 3
        assert row["t_step_steady_s"] > 0.0
        assert row["solver"] == "westervelt"
    # Only rungs that actually completed contribute a baseline row.
    assert len(baseline) == sum(1 for r in payload["rungs"] if r["exit_code"] == 0)


def test_the_measured_warmup_is_written_back_after_the_gates_not_before(tmp_path):
    """Feeding this run's measured warmup into the model that grades this run
    would be circular. It is recorded for the NEXT one."""
    device = FakeDevice(warmup_s=2.5)
    _, payload = run_suite(tmp_path, device)
    assert device.recorded == [(A100, 3.5)]  # the fake's runs pay warmup + 1.0
    assert payload["recorded_warmup_s"] == 3.5
    assert any("applies to the NEXT plan" in n for n in payload["notes"])
    # The graded numbers came from plan.json, which used the pre-run warmup.
    assert payload["rungs"][0]["plan"]["warmup_s"] == 2.5


# ----------------------------------------------------------------- parity


def test_field_diff_is_whole_field_not_a_headline():
    a = np.zeros((4, 4, 4), dtype=np.complex64)
    a[2, 2, 2] = 1.0  # the "peak"
    a[0, 0, 0] = 0.01

    same = gg.field_diff(a, a)
    assert same["rel_l2"] == 0.0 and same["rel_linf"] == 0.0

    # A field whose PEAK agrees exactly but which is wrong elsewhere must not
    # look like agreement.
    b = a.copy()
    b[0, 0, 0] = 0.02
    off = gg.field_diff(a, b)
    assert off["rel_linf"] == pytest.approx(0.01)
    assert off["rel_l2"] > 0.0

    assert gg.field_diff(a, a[:, :, :2])["rel_l2"] is None  # shapes disagree -> no number


def test_parity_verdicts_read_both_norms_of_both_fields():
    tight = {"rel_l2": 1e-9, "rel_linf": 1e-9}
    gates = gg.evaluate([], {"phasor": tight, "p_max": tight})
    parity = next(g for g in gates if g.id == "M7.parity")
    assert parity.verdict == "PASS" and len(parity.checks) == 4

    # L2 can hide a single bad voxel; L-infinity cannot, and one failing norm
    # fails the gate.
    gates = gg.evaluate([], {"phasor": {"rel_l2": 1e-9, "rel_linf": 1e-2}, "p_max": tight})
    assert next(g for g in gates if g.id == "M7.parity").verdict == "FAIL"

    # ALL FOUR norms have to be measured. Three good ones and one that could
    # not be evaluated is not "parity shown" — it is parity shown for three
    # quarters of the claim (mutation round: relaxing the count to 1 survived
    # every other assertion here).
    gates = gg.evaluate([], {"phasor": {"rel_l2": None, "rel_linf": 1e-9}, "p_max": tight})
    partial = next(g for g in gates if g.id == "M7.parity")
    assert partial.n_pass == 3 and partial.verdict == "INCOMPLETE"

    assert next(g for g in gg.evaluate([], None) if g.id == "M7.parity").verdict == "INCOMPLETE"


def test_the_parity_gate_is_measured_on_fp32_fields_not_on_a_stored_file():
    """The gate must not be graded through float16 storage.

    Operator measurement, 2026-08-23: the first Colab run's ``result.h5``
    agreed with a local CPU run to 3.6e-5 relative L2 and 4.883e-4 relative
    L-infinity — and 4.883e-4 is exactly 2^-11, ONE float16 ULP (99.17% of
    ``p_max`` bit-identical, 517 voxels off by one ULP, none by more). The
    fields agree below the resolution of the file. Gating M7's 1e-5 criterion
    on that round trip would fail a perfect solver, so this test reproduces
    the floor synthetically and pins that (a) storage alone exceeds the gate
    tolerance, and (b) the gate does not look there.
    """
    rng = np.random.default_rng(0)
    shape = (64, 64, 64)
    field = (rng.standard_normal(shape) * np.exp(-rng.random(shape) * 6)).astype(np.float32)
    peak = float(np.abs(field).max())

    # Two backends agreeing to ~1e-7 of the peak: two orders of magnitude
    # INSIDE the 1e-5 criterion, and what the first Colab session delivered.
    other = (field + (peak * 1e-7 * rng.standard_normal(shape)).astype(np.float32)).astype(
        np.float32
    )
    in_memory = gg.field_diff(field, other)
    assert in_memory["rel_l2"] < gg.PARITY_TOL and in_memory["rel_linf"] < gg.PARITY_TOL

    # The SAME two fields, seen through result.h5's float16 storage. The
    # deviation is now set by the storage grid, not by the solvers.
    through_storage = gg.field_diff(gg.stored_roundtrip(field), gg.stored_roundtrip(other))
    assert 2.0**-13 <= through_storage["rel_linf"] <= 2.0**-9, (
        "the storage floor should land on the float16 ULP scale (one ULP = 2^-11 = 4.883e-4)"
    )
    assert through_storage["rel_l2"] > gg.PARITY_TOL, (
        "storage noise alone must exceed the gate tolerance — that is the whole reason "
        "the gate is not measured on result.h5"
    )

    # The gate, fed the fp32 numbers, passes; fed the stored ones, it would not.
    passes = gg.evaluate([], {"phasor": in_memory, "p_max": in_memory})
    assert next(g for g in passes if g.id == "M7.parity").verdict == "PASS"
    fails = gg.evaluate([], {"phasor": through_storage, "p_max": through_storage})
    assert next(g for g in fails if g.id == "M7.parity").verdict == "FAIL"


def test_run_parity_gates_the_fp32_numbers_and_only_reports_the_stored_ones(tmp_path):
    """WHICH numbers land under the gated keys.

    Two backends that agree to ~1e-7 must produce a gated ``phasor`` block
    inside the 1e-5 criterion, while the informational block shows the
    float16 floor. Measuring the gate through storage instead — the exact
    mistake the operator's 2026-08-23 measurement caught — is invisible to a
    test that compares a run with itself, so this one feeds two DIFFERENT
    fields (mutation round: that mutation survived until this test existed).
    """

    class _Result:
        def __init__(self, phasor, p_max):
            self.phasor, self.p_max = phasor, p_max

    rng = np.random.default_rng(1)
    shape = (48, 48, 48)
    base = (rng.standard_normal(shape) * np.exp(-rng.random(shape) * 5)).astype(np.float32)
    peak = float(np.abs(base).max())
    nudged = (base + (peak * 1e-7 * rng.standard_normal(shape)).astype(np.float32)).astype(
        np.float32
    )
    legs = iter(
        [
            _Result(base.astype(np.complex64), base),
            _Result(nudged.astype(np.complex64), nudged),
        ]
    )

    data = gg.run_parity(
        tmp_path / "parity",
        backends=("numpy", "cupy"),
        solve=lambda job, backend: next(legs),
    )

    for key in ("phasor", "p_max"):
        assert data[key]["rel_l2"] < gg.PARITY_TOL, "the gated numbers are not the fp32 ones"
        assert data["stored_float16_reference"][key]["rel_l2"] > gg.PARITY_TOL, (
            "the informational block is not showing the storage floor"
        )
    assert next(g for g in gg.evaluate([], data) if g.id == "M7.parity").verdict == "PASS"


@pytest.mark.slow
def test_the_parity_path_runs_end_to_end_in_memory(tmp_path):
    """The whole comparison path — job, two real solves, both norms, the
    storage-floor line — without a GPU, by using numpy for both legs. Two
    identical solves must come out bit-identical; anything else means the
    path is broken before a GPU is ever involved."""
    data = gg.run_parity(tmp_path / "parity", backends=("numpy", "numpy"))

    assert "no result.h5" in data["measured_on"]
    for key in ("phasor", "p_max"):
        assert data[key]["rel_l2"] == 0.0
        assert data[key]["rel_linf"] == 0.0
        # Identical inputs quantize identically, so the storage line is 0 too;
        # what is pinned here is that the block EXISTS and is kept apart from
        # the gated numbers.
        assert data["stored_float16_reference"][key]["rel_l2"] == 0.0
    assert "not gated" in data["stored_float16_reference"]["note"]
    assert (tmp_path / "parity" / "fields.npz").is_file()  # evidence, not a summary


def test_the_report_keeps_the_storage_floor_apart_from_the_gate(tmp_path):
    """A reader who sees ~5e-4 in this report must be told it is the file's
    resolution, not a regression."""
    tight = {"rel_l2": 1e-9, "rel_linf": 1e-9}
    payload = {
        "format": gg.FORMAT,
        "generated": "now",
        "verdict": "PASS",
        "environment": {},
        "gates": [],
        "rungs": [],
        "parity": {
            "measured_on": "in-memory fp32 SolverResult fields",
            "phasor": tight,
            "p_max": tight,
            "stored_float16_reference": {"note": "informational", "phasor": {"rel_linf": 4.88e-4}},
        },
    }
    md = gg.render_markdown(payload)
    assert "in-memory fp32" in md
    assert "informational, not gated" in md.lower()
    assert "4.883e-4" in md  # the ULP is named, so the number is recognizable


# --------------------------------------------------------------- notebook

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "gpu_gates.ipynb"

#: The gate notebook's code cells, frozen byte for byte. Deliberately
#: stricter than ``colab_run.ipynb``'s contract: that one is allowed two
#: bridge calls, this one is allowed NO Python at all. It exists to invoke a
#: command, and a maintenance notebook that grows logic is a second
#: implementation of the suite that nobody tests.
NOTEBOOK_CODE_CELLS = (
    """\
# Setup. Colab GPU runtimes already ship cupy, so there is no GPU extra to install here.
# If pip upgrades numpy, Colab will ask you to restart the session: restart, then re-run this cell.
!pip install -q "caustica[report] @ git+https://github.com/ebx0/caustica\"""",
    """\
# The whole protocol. Nothing to edit: the ladder sizes itself from this device's free VRAM.
!python -m caustica.validation gpu-gates""",
    """\
# The same report the command just wrote, inline. Download the folder to keep the evidence:
# it holds the JSON, the Markdown, and every rung's own output folder.
!find benchmarks/reports/gpu_gates -name REPORT.md -exec cat {} +""",
)


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_the_gate_notebook_invokes_the_command_and_contains_no_logic():
    nb = _notebook()
    cells = nb["cells"]
    assert nb["nbformat"] == 4
    assert [c["cell_type"] for c in cells] == ["markdown", "code", "code", "code"]

    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    assert tuple(code) == NOTEBOOK_CODE_CELLS, "the gate notebook drifted from its template"
    for source in code:
        for line in source.splitlines():
            assert line.lstrip().startswith(("!", "#")), f"Python appeared in the notebook: {line}"


def test_the_gate_notebook_metadata_asks_for_a_gpu_and_stores_nothing():
    nb = _notebook()
    assert nb["metadata"]["accelerator"] == "GPU"  # the whole point of the file
    assert nb["metadata"]["kernelspec"] == {"display_name": "Python 3", "name": "python3"}
    for i, cell in enumerate(nb["cells"]):
        assert cell["metadata"] == {}, f"cell {i} grew metadata (cellView: form hides source)"
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_the_notebooks_command_line_is_one_the_cli_accepts():
    """A notebook that invokes a flag the parser does not have is a broken
    notebook that no test would otherwise see."""
    from caustica.validation.__main__ import build_parser

    (line,) = [
        ln.lstrip("! ").strip()
        for source in NOTEBOOK_CODE_CELLS
        for ln in source.splitlines()
        if "caustica.validation" in ln
    ]
    argv = line.split()
    assert argv[:4] == ["python", "-m", "caustica.validation", "gpu-gates"]
    assert build_parser().parse_args(argv[3:]).suite == "gpu-gates"


def test_the_gate_notebook_points_the_reader_at_the_user_notebook():
    """Two notebooks in one folder is a way to run the wrong one."""
    intro = "".join(_notebook()["cells"][0]["source"])
    assert "colab_run.ipynb" in intro
    assert "Hardware accelerator: GPU" in intro
    for code in ("0", "2", "4"):  # the exit codes an operator will see
        assert code in intro
