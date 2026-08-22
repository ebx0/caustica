"""M7 + M8's on-device criteria, measured in one self-contained command.

Both milestones stalled on the same thing: their remaining criteria can only
be answered by a real GPU, and a single session is not enough to answer them.
M8 asks for VRAM accuracy over **at least two grid sizes** and post-calibration
time accuracy over **at least two scenarios**; M7 asks for a full-size
(dx = 0.30 mm, 512-cubed FFT class) run that does not die of OOM, a numpy-vs-cupy
parity number, and a step-time baseline for M19. That is a protocol, not a
run — so it is written down here as one, and the operator's whole job is::

    python -m caustica.validation gpu-gates

What it does, in order:

1. Prints :func:`caustica.env.env_report`. No GPU: exit 2 with the fix for
   THIS machine (which is what a CI leg sees as a clean skip).
2. Runs :func:`caustica.planner.calibrate` on the target device — M8's gate
   is explicitly about the *calibrated* path, so the suite must produce that
   state rather than assume it.
3. Walks a **VRAM ladder**: several jobs of increasing size (~2, 8, 14, 28
   GiB by default, automatically clipped to the device's free VRAM). Each one
   contributes a predicted-vs-measured VRAM point and a predicted-vs-measured
   wall-time point. Large rungs run ``--preview-only``: a 28 GiB job would
   otherwise write a ~2.7 GB ``result.h5`` nobody asked for.
4. Adds one rung that deliberately **does not fit**, and records the refusal
   (exit 3) and its advice text as evidence that the planner refuses before
   paying rather than after.
5. Runs the **parity** scenario on numpy AND cupy and compares the two fields
   voxel by voxel — relative L2 and L-infinity over the whole volume, not a
   headline metric that can agree while the field disagrees.
6. Writes a stamped report (JSON + Markdown) under
   ``benchmarks/reports/gpu_gates/<gpu>-<timestamp>/``, with every rung's
   plan-vs-actual line, its deviation, and a PASS/FAIL/SKIP verdict per
   milestone criterion. Step times are stamped as the M19 baseline.

**A missing measurement is never a pass.** A check whose two sides are not
both present is ``SKIP``, a gate needs its milestone's required *count* of
PASSing checks and no FAILs, and a gate with nothing in it is ``INCOMPLETE``.
The suite's own exit code is 0 only when every gate is PASS.

Everything except the actual device work is exercised on CPU: the ladder
sizing, the verdict algebra, the report schema, the refusal branch and the
no-GPU exit all go through :class:`Harness`, which is the one seam where this
module touches hardware (see ``tests/test_validation_gpu_gates.py``).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Report format tag; the JSON companion of the Markdown page.
FORMAT = "caustica-gpu-gates/1"

#: Exit codes. 0/2/4 are borrowed from the runner's vocabulary on purpose —
#: 2 is "this environment cannot do it", 4 is "it ran and the answer is no".
EXIT_OK = 0
EXIT_ENV = 2
EXIT_FAILED = 4

#: Default VRAM ladder [GiB]. 14 is the 512-cubed / dx=0.30 mm full-size class
#: M7 names; the others bracket it so the model is checked over a decade of
#: sizes rather than at one point.
DEFAULT_TARGETS_GIB = (2.0, 8.0, 14.0, 28.0)

#: Never plan a rung against the last slice of free VRAM: the CUDA context
#: and the allocator's own fragmentation live there too.
LADDER_HEADROOM = 0.92

#: Rungs above this predicted size skip ``result.h5`` (``--preview-only``);
#: the preview package and metrics.json still land, so the run is still
#: inspectable, it just does not spend gigabytes proving it.
PREVIEW_ONLY_ABOVE_GIB = 4.0

#: The deliberately-refused rung is the SMALLEST shape that exceeds the
#: device, not a big round multiple of it. Two reasons: the job's medium is
#: built in HOST memory before the plan-first gate can refuse it (a rung at
#: 3x the device would cost tens of GB of RAM to prove a point), and "one
#: step above what fits" is the interesting case — a refusal that only
#: triggers at 3x would hide an off-by-a-lot gate.

#: M8's two tolerances and M7's parity tolerance, in one place.
VRAM_TOL_PCT = 10.0
TIME_TOL_PCT = 25.0
PARITY_TOL = 1e-5

_GIB = 2**30
_MAX_LADDER_SIDE = 2048


# --------------------------------------------------------------- verdicts


@dataclass(frozen=True)
class Check:
    """One PASS/FAIL/SKIP decision, with the numbers that produced it.

    Three constructors, because there are exactly three shapes of question
    the gates ask: is a prediction within a tolerance of a measurement
    (:meth:`relative`), is a measurement under a limit (:meth:`at_most`), and
    did something specific happen (:meth:`happened`).

    ``SKIP`` is load-bearing. A check that could not be evaluated — a rung
    that never ran, a stamp with a null field — must not silently become a
    pass, which is the single most likely way a suite like this lies.
    """

    name: str
    verdict: str  # "PASS" | "FAIL" | "SKIP"
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def relative(
        name: str,
        predicted: float | None,
        actual: float | None,
        tol_pct: float,
        unit: str = "",
    ) -> Check:
        """``|predicted - actual| / |actual| <= tol_pct`` — M8's gate shape."""
        data = {
            "predicted": predicted,
            "actual": actual,
            "tol_pct": tol_pct,
            "unit": unit,
            "deviation_pct": None,
        }
        if predicted is None or actual is None:
            return Check(name, "SKIP", "not measured", data)
        if actual == 0.0:
            # A zero measurement cannot carry a relative tolerance; saying so
            # is honest, calling it a pass because 0 == 0 is not.
            return Check(name, "SKIP", "measured value is zero", data)
        dev = 100.0 * (predicted - actual) / abs(actual)
        data["deviation_pct"] = dev
        ok = abs(dev) <= tol_pct
        return Check(
            name,
            "PASS" if ok else "FAIL",
            f"predicted {predicted:.4g}{unit} vs actual {actual:.4g}{unit} "
            f"({dev:+.1f}%, tolerance +/-{tol_pct:g}%)",
            data,
        )

    @staticmethod
    def at_most(name: str, value: float | None, limit: float, unit: str = "") -> Check:
        data = {"value": value, "limit": limit, "unit": unit}
        if value is None or not math.isfinite(value):
            return Check(name, "SKIP", "not measured", data)
        return Check(
            name,
            "PASS" if value <= limit else "FAIL",
            f"{value:.3e}{unit} (limit {limit:.3e}{unit})",
            data,
        )

    @staticmethod
    def happened(name: str, observed: Any, expected: Any, detail: str = "") -> Check:
        data = {"observed": observed, "expected": expected}
        if observed is None:
            return Check(name, "SKIP", "not observed", data)
        ok = observed == expected
        return Check(
            name,
            "PASS" if ok else "FAIL",
            detail or f"observed {observed!r}, expected {expected!r}",
            data,
        )


@dataclass
class Gate:
    """One milestone criterion and the checks that answer it.

    ``required`` is the milestone's own wording made countable: M8 says "at
    least 2 grid sizes" and "at least 2 scenarios", so one good measurement
    is not enough and the gate stays INCOMPLETE until the ladder has produced
    the second. A gate with no checks at all is INCOMPLETE, never PASS.
    """

    id: str
    criterion: str
    required: int
    checks: list[Check] = field(default_factory=list)

    @property
    def n_pass(self) -> int:
        return sum(c.verdict == "PASS" for c in self.checks)

    @property
    def verdict(self) -> str:
        if any(c.verdict == "FAIL" for c in self.checks):
            return "FAIL"
        return "PASS" if self.n_pass >= max(1, self.required) else "INCOMPLETE"

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "criterion": self.criterion,
            "required": self.required,
            "verdict": self.verdict,
            "n_pass": self.n_pass,
            "checks": [
                {"name": c.name, "verdict": c.verdict, "detail": c.detail, **c.data}
                for c in self.checks
            ],
        }


def overall_verdict(gates: list[Gate]) -> str:
    """PASS only when every gate passes; FAIL beats INCOMPLETE beats PASS."""
    verdicts = {g.verdict for g in gates}
    if not gates or "FAIL" in verdicts:
        return "FAIL" if "FAIL" in verdicts else "INCOMPLETE"
    return "INCOMPLETE" if "INCOMPLETE" in verdicts else "PASS"


# ------------------------------------------------------------- the ladder


def smooth_sides(lo: int, hi: int) -> list[int]:
    """Even 2,3,5-smooth grid sides in ``[lo, hi]``.

    The engine pads every axis up to the next 2,3,5-smooth length, so a
    ladder built out of non-smooth sides would plan one shape and run
    another — the predicted VRAM would be right about a grid that does not
    exist. Even, because the rfft's last axis is happier that way.
    """
    out = []
    for n in range(max(2, lo - lo % 2), hi + 1, 2):
        m = n
        for f in (2, 3, 5):
            while m % f == 0:
                m //= f
        if m == 1:
            out.append(n)
    return out


def side_for_target(
    target_bytes: float,
    *,
    nonlinear: bool = True,
    n_harmonics: int = 1,
    ndim: int = 3,
    max_side: int = _MAX_LADDER_SIDE,
) -> int | None:
    """Largest smooth cubic side whose predicted VRAM fits ``target_bytes``.

    Uses the planner's own inventory, so the ladder and the thing being
    tested cannot disagree about what a shape costs — the point of the rung
    is to check that inventory against the device, not to second-guess it.
    """
    from caustica.planner import model  # noqa: PLC0415 (keep import light)

    best: int | None = None
    for n in smooth_sides(16, max_side):
        shape = (n,) * ndim
        mem = model.kspace_memory(shape, nonlinear, n_harmonics, rec_elems=n**ndim)
        if mem.total_bytes <= target_bytes:
            best = n
        else:
            break
    return best


def side_above(
    limit_bytes: float,
    *,
    nonlinear: bool = True,
    n_harmonics: int = 1,
    ndim: int = 3,
    max_side: int = _MAX_LADDER_SIDE,
) -> int | None:
    """Smallest smooth cubic side whose predicted VRAM EXCEEDS ``limit_bytes``.

    This is the OOM rung. Sizing it as "the largest shape under 1.15x the
    device" was wrong in a way that only showed up on a big device: on 79 GiB
    free that picked a 75 GiB shape, which FITS, and the refusal gate would
    have been graded on a run that was never supposed to be refused.
    """
    from caustica.planner import model  # noqa: PLC0415

    for n in smooth_sides(16, max_side):
        shape = (n,) * ndim
        mem = model.kspace_memory(shape, nonlinear, n_harmonics, rec_elems=n**ndim)
        if mem.total_bytes > limit_bytes:
            return n
    return None


@dataclass(frozen=True)
class RungSpec:
    """One ladder step: a shape, what it should cost, and what it is for."""

    name: str
    shape: tuple[int, ...]
    dx_mm: float
    solver: str
    target_gib: float
    predicted_gib: float
    preview_only: bool
    expect_fit: bool
    note: str = ""

    @property
    def extent_mm(self) -> float:
        return self.shape[-1] * self.dx_mm

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dx_mm": self.dx_mm,
            "solver": self.solver,
            "target_gib": self.target_gib,
            "predicted_gib": self.predicted_gib,
            "preview_only": self.preview_only,
            "expect_fit": self.expect_fit,
            "note": self.note,
        }


def _predicted_gib(shape: tuple[int, ...], solver: str) -> float:
    from caustica.planner import model  # noqa: PLC0415

    n = 1
    for s in shape:
        n *= s
    mem = model.kspace_memory(shape, solver == "westervelt", 1, rec_elems=n)
    return mem.total_bytes / _GIB


def build_ladder(
    free_gib: float,
    *,
    targets_gib: tuple[float, ...] = DEFAULT_TARGETS_GIB,
    dx_mm: float = 0.30,
    solver: str = "westervelt",
    max_vram_gib: float | None = None,
    oom_rung: bool = True,
) -> list[RungSpec]:
    """Size the ladder for THIS device. Pure arithmetic — no hardware touched.

    The default targets are clipped to what the device can actually hold; a
    device too small for two of them gets a ladder derived from its own free
    VRAM instead, because M8's criterion is "at least two grid sizes", not
    "these particular sizes". The last entry is the rung that must NOT fit,
    which is evidence for the refusal path rather than for the model.
    """
    budget = free_gib if max_vram_gib is None else min(free_gib, max_vram_gib)
    budget = max(0.0, budget) * LADDER_HEADROOM
    kept = [t for t in sorted(targets_gib) if 0 < t <= budget]
    if len(kept) < 2 <= len(targets_gib):
        # CLIPPING left too few points: a small (or heavily occupied) device
        # still has to produce two, or M8's "at least 2 grid sizes" can never
        # close on it. An operator who explicitly asked for ONE target is not
        # overruled here — that is their ladder, and the gate will honestly
        # come out INCOMPLETE.
        kept = sorted({round(budget * f, 3) for f in (0.25, 0.6) if budget * f > 0.02})

    rungs: list[RungSpec] = []
    seen: set[tuple[int, ...]] = set()
    for target in kept:
        side = side_for_target(target * _GIB, nonlinear=solver == "westervelt")
        if side is None:
            continue
        shape = (side,) * 3
        if shape in seen:
            continue
        seen.add(shape)
        predicted = _predicted_gib(shape, solver)
        rungs.append(
            RungSpec(
                name=f"vram-{target:g}gib-{side}",
                shape=shape,
                dx_mm=dx_mm,
                solver=solver,
                target_gib=float(target),
                predicted_gib=predicted,
                preview_only=predicted > PREVIEW_ONLY_ABOVE_GIB,
                expect_fit=True,
                note="M7 full-size class (512^3)" if side == 512 else "",
            )
        )

    if oom_rung and budget > 0:
        side = side_above(free_gib * _GIB, nonlinear=solver == "westervelt")
        if side is not None and (not rungs or side > max(r.shape[0] for r in rungs)):
            shape = (side,) * 3
            predicted = _predicted_gib(shape, solver)
            rungs.append(
                RungSpec(
                    name=f"oom-{side}",
                    shape=shape,
                    dx_mm=dx_mm,
                    solver=solver,
                    target_gib=free_gib,
                    predicted_gib=predicted,
                    preview_only=True,
                    expect_fit=False,
                    note=(
                        f"{predicted:.1f} GiB against {free_gib:.1f} GiB free: "
                        f"must be REFUSED (exit 3)"
                    ),
                )
            )
    return rungs


def rung_job(spec: RungSpec, *, f0_mhz: float = 0.5, settle_periods: int = 2) -> dict:
    """A ``caustica-job/1`` dict for one rung. No external data, by design.

    Homogeneous nonlinear water plus a bowl scaled to the grid, so every rung
    is the same scene at a different resolution and the ladder measures size
    rather than a change of scenario. The settling window is pinned
    (``min = max``), which makes the step count deterministic — without that,
    a run that converges early would make the time gate a test of the
    convergence heuristic instead of the timing model.
    """
    extent = spec.extent_mm
    return {
        "format": "caustica-job/1",
        "kind": "explicit",
        "name": spec.name,
        "medium": {
            "kind": "homogeneous",
            # beta > 0: with linear water the westervelt solver drops its
            # nonlinear buffers and the rung would measure a smaller run
            # than the one the planner was asked about.
            "material": {
                "name": "water",
                "alpha_np_m": 0.5,
                "rho": 1000.0,
                "c": 1500.0,
                "beta": 3.5,
            },
        },
        "grid": {
            "ndim": 3,
            "dx_mm": spec.dx_mm,
            "size_mm": [s * spec.dx_mm for s in spec.shape],
            "pml": {"thickness_mm": max(3.0, 10 * spec.dx_mm)},
        },
        "source": {
            "kind": "array",
            "array": {
                "kind": "bowl",
                "d_outer_mm": round(0.30 * extent, 4),
                "roc_mm": round(0.30 * extent, 4),
            },
            "apex_mm": [round(0.5 * extent, 4), round(0.5 * extent, 4), round(0.06 * extent, 4)],
        },
        "drive": {"f0_mhz": f0_mhz, "amplitude_kpa": 200.0},
        "run": {
            "spec": {
                "min_settle_periods": settle_periods,
                "max_settle_periods": settle_periods,
                "n_record_periods": 1,
            },
            "harmonics": [1],
        },
        "solver": spec.solver,
    }


def parity_job(name: str = "parity-mini", dx_mm: float = 0.6, side: int = 40) -> dict:
    """The mini 3-D scenario M7's parity criterion is measured on.

    Small enough that the numpy leg is seconds, big enough to be a real 3-D
    focused field with a PML, a curved source and a settling history — the
    two backends have to agree on all of it, not on a peak.
    """
    extent = side * dx_mm
    return {
        "format": "caustica-job/1",
        "kind": "explicit",
        "name": name,
        "medium": {"kind": "homogeneous"},
        "grid": {
            "ndim": 3,
            "dx_mm": dx_mm,
            "size_mm": [extent, extent, extent],
            "pml": {"thickness_mm": 3.0},
        },
        "source": {
            "kind": "array",
            "array": {"kind": "bowl", "d_outer_mm": 0.35 * extent, "roc_mm": 0.40 * extent},
            "apex_mm": [0.5 * extent, 0.5 * extent, 0.1 * extent],
        },
        "drive": {"f0_mhz": 0.5, "amplitude_kpa": 100.0},
        "run": {
            "spec": {"min_settle_periods": 2, "max_settle_periods": 2, "n_record_periods": 1},
            "harmonics": [1],
        },
        # Linear: the parity criterion is about two backends computing the
        # same arithmetic, and float32 nonlinearity would add a divergence
        # that says nothing about the port.
        "solver": "linear",
    }


# -------------------------------------------------------------- the harness


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def gpu_key_for(device_name: str, total_gib: float | None = None) -> str:
    """Best ``gpu_db.json`` key for a CUDA device name.

    The suite configures itself: asking the operator to also pass ``--gpu
    A100`` on an A100 is a way to get a T4 report labeled A100. Falls back to
    the planner's own default rather than raising — a wrong datasheet label
    is cosmetic once calibration has run, which it has by then.
    """
    from caustica.planner import load_gpu_db  # noqa: PLC0415

    devices, _ = load_gpu_db()
    norm = _norm(device_name)
    best: tuple[int, float, str] | None = None
    for key, spec in devices.items():
        # .lower() BEFORE stripping punctuation, or "A100" becomes "100" and
        # "SXM" becomes "" — an empty token is a substring of everything, so
        # every device scored a match on it (caught by the T4/L4/H100 cases
        # in tests/test_validation_gpu_gates.py).
        tokens = [tok for tok in (_norm(part) for part in key.split("-")) if tok]
        if not tokens or tokens[0] not in norm:
            continue
        score = sum(1 for t in tokens if t in norm)
        gap = -abs(spec.vram_gib - total_gib) if total_gib is not None else 0.0
        cand = (score, gap, key)
        if best is None or cand > best:
            best = cand
    return best[2] if best is not None else "A100"


@dataclass
class Harness:
    """Every way this suite touches the world, in one substitutable object.

    The GPU half of the suite cannot run on the machine it is written on, so
    the alternative to this seam is code that is only ever exercised in the
    one place where it is expensive to be wrong. With it, the ladder, the
    verdict algebra, the refusal branch and the report are all driven from a
    CPU test with fabricated measurements.
    """

    require_gpu: Callable[[], None]
    env_report: Callable[[], dict]
    device_name: Callable[[], str]
    free_vram_gib: Callable[[], float | None]
    calibrate: Callable[..., dict]
    run_job: Callable[..., int]
    record_warmup: Callable[[str, float], Any]


def default_harness() -> Harness:
    """The real thing: cupy, the planner and the runner."""
    from caustica import env as env_mod  # noqa: PLC0415
    from caustica import planner  # noqa: PLC0415
    from caustica.runner import run_job_file  # noqa: PLC0415

    def _free_gib() -> float | None:
        info = env_mod.gpu_environment("cupy")
        return info.get("vram_free_gib")

    return Harness(
        require_gpu=lambda: env_mod.require_gpu("the GPU gate suite measures a real device"),
        # The TRUE resolved backend, not the one we hope for: an
        # env_report("cupy") on a machine without cupy would print
        # "resolved_backend: cupy" one line above the refusal.
        env_report=env_mod.env_report,
        device_name=lambda: planner.calibration.device_name("cupy"),
        free_vram_gib=_free_gib,
        calibrate=planner.calibrate,
        run_job=run_job_file,
        record_warmup=planner.record_warmup,
    )


# ------------------------------------------------------------- measurement


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@dataclass
class RungResult:
    """What one rung's output folder said, reduced to the numbers gates need."""

    spec: RungSpec
    exit_code: int | None = None
    plan: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    error: dict = field(default_factory=dict)

    @property
    def actual(self) -> dict:
        return self.meta.get("actual", {}) or {}

    @property
    def vram_actual_gib(self) -> float | None:
        return self.actual.get("vram_pool_peak_gib")

    @property
    def elapsed_s(self) -> float | None:
        return self.actual.get("elapsed_solve_s")

    def as_dict(self) -> dict:
        return {
            **self.spec.as_dict(),
            "exit_code": self.exit_code,
            "plan": {
                k: self.plan.get(k)
                for k in (
                    "source",
                    "spp",
                    "t_step_s",
                    "warmup_s",
                    "steps_expected",
                    "steps_worst",
                    "t_expected_s",
                    "vram_gib",
                )
            },
            "actual": {
                k: self.actual.get(k)
                for k in (
                    "elapsed_solve_s",
                    "steps_total",
                    "t_step_measured_s",
                    "t_step_steady_s",
                    "warmup_s",
                    "steady_samples",
                    "vram_pool_peak_gib",
                    "converged_period",
                )
            },
            "error": {k: self.error.get(k) for k in ("stage", "error_class", "message", "advice")},
        }


def run_rung(
    spec: RungSpec,
    outdir: Path,
    harness: Harness,
    *,
    gpu_key: str,
    settle_periods: int = 2,
) -> RungResult:
    """Run one rung and read its stamp back. Never raises on a refusal."""
    from caustica.runner import RunnerOptions  # noqa: PLC0415

    outdir.mkdir(parents=True, exist_ok=True)
    job_path = outdir / "job_input.json"
    job_path.write_text(
        json.dumps(rung_job(spec, settle_periods=settle_periods), indent=2), encoding="utf-8"
    )
    code = harness.run_job(
        job_path,
        RunnerOptions(
            out=outdir,
            backend="cupy",
            gpu=gpu_key,
            # No timing probe: calibration has already run (that IS M8's
            # gate), and the probe's ~20 steps would leave their own blocks
            # in the memory pool and inflate the VRAM measurement it is
            # about to be judged against.
            measure=False,
            preview_only=spec.preview_only,
            status_interval_s=0.0,
            progress="plain",
        ),
    )
    return RungResult(
        spec=spec,
        exit_code=code,
        plan=_read_json(outdir / "plan.json"),
        meta=_read_json(outdir / "run_meta.json"),
        error=_read_json(outdir / "error.json"),
    )


def field_diff(reference, compared) -> dict:
    """Relative L2 and L-infinity between two fields, normalized by the reference.

    Whole-field, deliberately: a peak-pressure comparison can agree to 1e-7
    while an entire plane is wrong, and M7's parity criterion is about the
    two backends computing the same FIELD.
    """
    import numpy as np  # noqa: PLC0415

    fa = np.asarray(reference)
    fb = np.asarray(compared)
    if fa.shape != fb.shape:
        return {"rel_l2": None, "rel_linf": None, "note": f"{fa.shape} != {fb.shape}"}
    fa64 = fa.astype(np.complex128)
    diff = np.abs(fa64 - fb.astype(np.complex128))
    denom_l2 = float(np.sqrt((np.abs(fa64) ** 2).sum()))
    denom_inf = float(np.abs(fa64).max())
    return {
        "rel_l2": (float(np.sqrt((diff**2).sum()) / denom_l2) if denom_l2 > 0 else None),
        "rel_linf": (float(diff.max() / denom_inf) if denom_inf > 0 else None),
    }


def field_parity(a_dir: Path, b_dir: Path) -> dict:
    """:func:`field_diff` over two runs' STORED ``result.h5`` fields.

    Not what M7's parity gate is measured on — see :func:`run_parity`. With
    the default float16 storage this bottoms out around one float16 ULP
    (~4.9e-4 relative L-infinity) no matter how well the solvers agree, so a
    1e-5 criterion applied here would fail a perfect port. Kept because
    comparing two archived runs is a real thing to want to do; the number it
    returns just has a floor.
    """
    from caustica.io.store import load_result  # noqa: PLC0415

    a = load_result(a_dir / "result.h5")
    b = load_result(b_dir / "result.h5")
    out: dict[str, Any] = {"reference": str(a_dir), "compared": str(b_dir)}
    for key in ("phasor", "p_max"):
        out[key] = field_diff(getattr(a, key), getattr(b, key))
    return out


# ------------------------------------------------------------------ gates


def evaluate(results: list[RungResult], parity: dict | None) -> list[Gate]:
    """Turn measurements into milestone verdicts. Pure — no I/O, no hardware."""
    fitting = [r for r in results if r.spec.expect_fit]
    refused = [r for r in results if not r.spec.expect_fit]

    vram = Gate(
        id="M8.vram",
        criterion=(
            "VRAM prediction within +/-10% of the measured mempool peak, on at least 2 grid sizes"
        ),
        required=2,
    )
    timing = Gate(
        id="M8.time",
        criterion=(
            "post-calibration wall-time prediction within +/-25% of actual, "
            "on at least 2 scenarios (same device); a plan whose source is not "
            "'calibrated' does not count"
        ),
        required=2,
    )
    fullsize = Gate(
        id="M7.fullsize",
        criterion="a full-size run (dx=0.30 mm, 512^3 FFT class) completes without OOM",
        required=1,
    )
    parity_gate = Gate(
        id="M7.parity",
        criterion=f"numpy vs cupy on a mini 3-D scenario: field relative error < {PARITY_TOL:g}",
        required=2,
    )
    oom = Gate(
        id="M8.oom",
        criterion="a run larger than the device is REFUSED before solving (exit 3) with advice",
        required=2,
    )

    for r in fitting:
        tag = r.spec.name
        completed = r.exit_code == 0
        vram.checks.append(
            Check.relative(
                f"{tag}: VRAM",
                r.plan.get("vram_gib") if completed else None,
                r.vram_actual_gib if completed else None,
                VRAM_TOL_PCT,
                " GiB",
            )
        )
        # Two things must hold before a wall-time number means anything.
        #
        # 1. M8's criterion is about the CALIBRATED path. If the plan says
        #    "db" — calibration failed, or the device did not match a
        #    gpu_db key — then a datasheet guess is being graded, and a
        #    datasheet guess is coarse to ~2x: it can land inside +/-25% by
        #    luck and close a milestone that was never measured.
        # 2. The suite pins min_settle == max_settle so the step count is
        #    deterministic. A run that took a different number of steps than
        #    planned is not comparable — that difference measures the
        #    convergence heuristic, not the timing model.
        planned_steps = r.plan.get("steps_expected")
        actual_steps = r.actual.get("steps_total")
        calibrated = r.plan.get("source") == "calibrated"
        comparable = (
            completed and calibrated and planned_steps is not None and planned_steps == actual_steps
        )
        check = Check.relative(
            f"{tag}: wall time",
            r.plan.get("t_expected_s") if comparable else None,
            r.elapsed_s if comparable else None,
            TIME_TOL_PCT,
            " s",
        )
        if not comparable and completed and not calibrated:
            check = Check(
                check.name,
                "SKIP",
                f"estimate source is {r.plan.get('source')!r}, not 'calibrated' — "
                f"M8's gate is about the calibrated path",
                {**check.data, "plan_source": r.plan.get("source")},
            )
        timing.checks.append(check)
        if r.spec.shape[0] >= 512 and r.spec.dx_mm <= 0.30:
            fullsize.checks.append(
                Check.happened(
                    f"{tag}: completed",
                    r.exit_code,
                    0,
                    f"shape {r.spec.shape} at dx={r.spec.dx_mm} mm exited {r.exit_code} "
                    f"(0 = solved, 3 = refused for memory)",
                )
            )

    for r in refused:
        oom.checks.append(
            Check.happened(
                f"{r.spec.name}: refused with exit 3",
                r.exit_code,
                3,
                f"a {r.spec.predicted_gib:.1f} GiB run on this device exited {r.exit_code}",
            )
        )
        advice = r.error.get("advice") or []
        oom.checks.append(
            Check.happened(
                f"{r.spec.name}: refusal carries advice",
                bool(advice),
                True,
                "; ".join(advice)[:400] or "error.json carried no advice",
            )
        )

    if parity:
        for key in ("phasor", "p_max"):
            block = parity.get(key) or {}
            parity_gate.checks.append(
                Check.at_most(f"parity {key}: relative L2", block.get("rel_l2"), PARITY_TOL)
            )
            parity_gate.checks.append(
                Check.at_most(f"parity {key}: relative Linf", block.get("rel_linf"), PARITY_TOL)
            )
        parity_gate.required = 4

    return [parity_gate, fullsize, vram, timing, oom]


# ----------------------------------------------------------------- report


def _fmt_check(c: Check) -> str:
    return f"| {c.name} | {c.verdict} | {c.detail} |"


def render_markdown(payload: dict) -> str:
    """The human half of the report. The JSON is the machine half."""
    env = payload.get("environment", {})
    lines = [
        "# caustica GPU gate suite",
        "",
        f"**Overall verdict: {payload['verdict']}** (`{FORMAT}`, generated {payload['generated']})",
        "",
        "| | |",
        "|---|---|",
        f"| device | {env.get('gpu_name', 'unknown')} |",
        f"| VRAM total / free at start | {env.get('vram_total_gib')} / "
        f"{env.get('vram_free_gib')} GiB |",
        f"| caustica | {payload.get('caustica')} @ {payload.get('git_commit')} |",
        f"| python / cupy | {env.get('python')} / {env.get('cupy_version')} |",
        f"| datasheet key | {payload.get('gpu_key')} |",
        "",
        "## Milestone gates",
        "",
        "| gate | verdict | criterion |",
        "|---|---|---|",
    ]
    for g in payload["gates"]:
        lines.append(f"| `{g['id']}` | **{g['verdict']}** | {g['criterion']} |")
    lines += ["", "### Every check", "", "| check | verdict | detail |", "|---|---|---|"]
    for g in payload["gates"]:
        for c in g["checks"]:
            lines.append(f"| {c['name']} | {c['verdict']} | {c['detail']} |")

    lines += [
        "",
        "## Ladder: plan vs actual",
        "",
        "| rung | shape | plan VRAM | actual VRAM | dev | plan time | actual time | dev | exit |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in payload["rungs"]:
        plan, act = r["plan"], r["actual"]
        lines.append(
            f"| {r['name']} | {'x'.join(str(s) for s in r['shape'])} "
            f"| {_g(plan.get('vram_gib'))} GiB | {_g(act.get('vram_pool_peak_gib'))} GiB "
            f"| {_pct(plan.get('vram_gib'), act.get('vram_pool_peak_gib'))} "
            f"| {_g(plan.get('t_expected_s'))} s | {_g(act.get('elapsed_solve_s'))} s "
            f"| {_pct(plan.get('t_expected_s'), act.get('elapsed_solve_s'))} "
            f"| {r['exit_code']} |"
        )

    lines += [
        "",
        "## Step-time baseline (M19 reads this)",
        "",
        "| shape | voxels | steady s/step | measured s/step (incl. warmup) | warmup s |",
        "|---|---|---|---|---|",
    ]
    for b in payload.get("step_time_baseline", []):
        lines.append(
            f"| {'x'.join(str(s) for s in b['shape'])} | {b['voxels']:,} "
            f"| {_g(b.get('t_step_steady_s'))} | {_g(b.get('t_step_measured_s'))} "
            f"| {_g(b.get('warmup_s'))} |"
        )

    parity = payload.get("parity") or {}
    if parity:
        lines += [
            "",
            "## numpy vs cupy parity (whole field)",
            "",
            f"Measured on: {parity.get('measured_on', 'unknown')}.",
            "",
            "| field | rel L2 | rel Linf |",
            "|---|---|---|",
        ]
        for key in ("phasor", "p_max"):
            block = parity.get(key) or {}
            lines.append(f"| {key} | {_g(block.get('rel_l2'))} | {_g(block.get('rel_linf'))} |")
        stored = parity.get("stored_float16_reference") or {}
        if stored:
            lines += [
                "",
                "Storage floor (**informational, not gated**) — the same two fields after "
                "the store's float16 quantization. One float16 ULP is 2^-11 = 4.883e-4, "
                "so a relative L-infinity of about that size here means the solvers agree "
                "below the resolution of the file:",
                "",
                "| field (stored) | rel L2 | rel Linf |",
                "|---|---|---|",
            ]
            for key in ("phasor", "p_max"):
                block = stored.get(key) or {}
                lines.append(f"| {key} | {_g(block.get('rel_l2'))} | {_g(block.get('rel_linf'))} |")

    notes = payload.get("notes") or []
    if notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    lines += [
        "",
        "---",
        "",
        "Reproduce: `python -m caustica.validation gpu-gates`. Every scenario in "
        "this report is generated from the library (homogeneous water + a bowl "
        "scaled to the grid); no external dataset is involved.",
        "",
    ]
    return "\n".join(lines)


def _g(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _pct(predicted: Any, actual: Any) -> str:
    if predicted is None or actual in (None, 0):
        return "--"
    return f"{100.0 * (predicted - actual) / abs(actual):+.1f}%"


def write_report(outdir: Path, payload: dict) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "gpu_gates.json"
    md_path = outdir / "REPORT.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, md_path


# ------------------------------------------------------------------ suite


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower() or "gpu"


def gpu_gates(
    *,
    out: str | Path | None = None,
    max_vram_gib: float | None = None,
    targets_gib: tuple[float, ...] = DEFAULT_TARGETS_GIB,
    dx_mm: float = 0.30,
    solver: str = "westervelt",
    settle_periods: int = 2,
    oom_rung: bool = True,
    parity: bool = True,
    gpu_key: str | None = None,
    harness: Harness | None = None,
    log: Callable[[str], None] = print,
) -> tuple[int, dict]:
    """Run the whole protocol. Returns ``(exit_code, report_payload)``."""
    from caustica import __version__ as version  # noqa: PLC0415
    from caustica.env import git_commit  # noqa: PLC0415

    harness = harness or default_harness()
    notes: list[str] = []

    env = harness.env_report()
    log("--- environment ----------------------------------------------")
    for key, value in env.items():
        log(f"  {key}: {value}")
    try:
        harness.require_gpu()
    except Exception as exc:
        log("")
        log(f"GPU GATES: cannot run here — {exc}")
        return EXIT_ENV, {"format": FORMAT, "verdict": "SKIPPED", "reason": str(exc)}

    device = harness.device_name()
    free_gib = harness.free_vram_gib()
    if free_gib is None:
        free_gib = float(env.get("vram_free_gib") or env.get("vram_total_gib") or 0.0)
        notes.append("free VRAM could not be probed; the ladder was sized from the env stamp")
    key = gpu_key or gpu_key_for(device, env.get("vram_total_gib"))

    stamp = datetime.now(timezone.utc)
    root = Path(out) if out is not None else Path("benchmarks/reports/gpu_gates")
    outdir = root / f"{_slug(device)}-{stamp.strftime('%Y%m%d-%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"\nreport folder: {outdir}")

    # ---- M8 says "post-calibration": produce that state, do not assume it --
    log(f"\ncalibrating {device} (datasheet key: {key}) ...")
    try:
        calibration = harness.calibrate(backend="cupy")
        log(
            f"  a={calibration.get('a'):.3e}  b={calibration.get('b'):.3e}  "
            f"warmup={calibration.get('warmup_s')}"
        )
    except Exception as exc:  # noqa: BLE001 - a lost calibration must not lose the session
        # The ladder still measures VRAM, which does not depend on this. The
        # time gate refuses itself: without calibration the runner's plans
        # come out labeled "db", and the check above SKIPs them by source.
        calibration = {"error": f"{type(exc).__name__}: {exc}"}
        log(f"  calibration FAILED: {calibration['error']}")
        notes.append(
            f"calibration failed ({calibration['error']}): every plan below is a datasheet "
            f"estimate, so M8's post-calibration time gate cannot be judged and stays open"
        )

    ladder = build_ladder(
        free_gib,
        targets_gib=targets_gib,
        dx_mm=dx_mm,
        solver=solver,
        max_vram_gib=max_vram_gib,
        oom_rung=oom_rung,
    )
    log(f"\nladder ({free_gib:.1f} GiB free):")
    for spec in ladder:
        log(
            f"  {spec.name:<22} {spec.shape}  plan {spec.predicted_gib:6.2f} GiB"
            f"  {'preview-only' if spec.preview_only else 'full result'}"
            f"  {'MUST BE REFUSED' if not spec.expect_fit else ''}"
        )

    results: list[RungResult] = []
    for spec in ladder:
        log(f"\n=== rung {spec.name} {spec.shape} ===")
        try:
            results.append(
                run_rung(
                    spec,
                    outdir / "rungs" / spec.name,
                    harness,
                    gpu_key=key,
                    settle_periods=settle_periods,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad rung must not lose the rest
            log(f"  rung raised {type(exc).__name__}: {exc}")
            notes.append(f"rung {spec.name} raised {type(exc).__name__}: {exc}")
            results.append(RungResult(spec=spec, exit_code=None))

    parity_data: dict | None = None
    if parity:
        log("\n=== parity: the same mini 3-D job on numpy and on cupy ===")
        try:
            parity_data = run_parity(outdir / "parity")
        except Exception as exc:  # noqa: BLE001
            log(f"  parity raised {type(exc).__name__}: {exc}")
            notes.append(f"parity raised {type(exc).__name__}: {exc}")

    gates = evaluate(results, parity_data)
    baseline = [
        {
            "shape": r.spec.shape,
            "voxels": int(r.spec.shape[0] ** 3),
            "solver": r.spec.solver,
            "t_step_steady_s": r.actual.get("t_step_steady_s"),
            "t_step_measured_s": r.actual.get("t_step_measured_s"),
            "warmup_s": r.actual.get("warmup_s"),
        }
        for r in results
        if r.exit_code == 0
    ]

    payload = {
        "format": FORMAT,
        "generated": stamp.isoformat(timespec="seconds"),
        "caustica": version,
        "git_commit": git_commit(),
        "device": device,
        "gpu_key": key,
        "free_vram_gib_at_start": free_gib,
        "environment": env,
        "calibration": calibration,
        "rungs": [r.as_dict() for r in results],
        "parity": parity_data,
        "gates": [g.as_dict() for g in gates],
        "step_time_baseline": baseline,
        "verdict": overall_verdict(gates),
        "notes": notes,
    }

    # Write the warmup BACK only after every gate has been judged: feeding a
    # number into the model that is then used to grade the same run would be
    # circular. The next suite (and every planner call after it) gets it.
    warmups = [r.actual.get("warmup_s") for r in results if r.actual.get("warmup_s") is not None]
    if warmups:
        harness.record_warmup(device, max(warmups))
        payload["recorded_warmup_s"] = max(warmups)
        notes.append(
            f"recorded warmup {max(warmups):.2f} s into the calibration for {device}; "
            f"it applies to the NEXT plan, not to the numbers graded above"
        )

    json_path, md_path = write_report(outdir, payload)
    log("\n--- verdict --------------------------------------------------")
    for g in gates:
        log(f"  {g.verdict:<11} {g.id}: {g.criterion}")
    log(f"\noverall: {payload['verdict']}")
    log(f"report: {md_path}")
    log(f"json:   {json_path}")
    return (EXIT_OK if payload["verdict"] == "PASS" else EXIT_FAILED), payload


def solve_job_dict(job: dict, backend: str):
    """Build and run a job dict IN MEMORY on one backend, no disk in between.

    The same three calls the runner makes, without the runner: the parity
    gate wants the solver's own fp32 arrays, and everything the runner adds
    (the output folder, the CPU-time gate that would refuse the numpy
    reference leg, the store) is between those arrays and the comparison.
    """
    import caustica.solvers as solvers  # noqa: PLC0415
    from caustica.config.job import build_job, parse_job  # noqa: PLC0415

    built = build_job(parse_job(job, job.get("name", "parity")), None, True)
    result = solvers.get(built.solver)().run(
        built.grid,
        built.medium,
        built.source,
        built.spec,
        backend=backend,
        record_region=built.record_region,
        reference_point=built.focus_vox,
        harmonics=built.harmonics,
    )
    return result


def stored_roundtrip(arr, max_norm_err: float = 1e-3):
    """One field as ``result.h5`` would give it back (float16 by default).

    Uses the store's own quantizer, so the number this produces is the real
    storage floor and not a re-derivation of it.
    """
    import numpy as np  # noqa: PLC0415

    from caustica.io.quantize import try_float16  # noqa: PLC0415

    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        re = try_float16(arr.real.astype(np.float32), max_norm_err).restore()
        im = try_float16(arr.imag.astype(np.float32), max_norm_err).restore()
        return (re + 1j * im).astype(np.complex64)
    return try_float16(arr.astype(np.float32), max_norm_err).restore()


def run_parity(
    outdir: Path,
    *,
    backends: tuple[str, str] = ("numpy", "cupy"),
    job: dict | None = None,
    solve: Callable[[dict, str], Any] | None = None,
) -> dict:
    """The same mini 3-D job on two backends, compared field by field.

    **Measured on the in-memory fp32 fields, deliberately not on a stored
    ``result.h5``.** Operator measurement, 2026-08-23: the first Colab run's
    file agreed with a local CPU run to 3.6e-5 relative L2 and 4.883e-4
    relative L-infinity — and 4.883e-4 is exactly 2^-11, one float16 ULP.
    Voxelwise, 99.17% of ``p_max`` was bit-identical, 517 voxels differed by
    one ULP and none by more; the handful of larger relative deviations were
    all sign-bit noise near zero. The fields agree BELOW the resolution of
    the file they were written to. Gating M7's 1e-5 criterion on a float16
    round trip would therefore have produced a confident FAIL for a perfect
    solver — which is the exact failure mode this suite must not have.

    The storage floor is still reported, under its own key, so a reader who
    compares two ``result.h5`` files and sees ~5e-4 knows what they are
    looking at instead of filing a regression.
    """
    solve = solve or solve_job_dict
    job = job or parity_job()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "job_input.json").write_text(json.dumps(job, indent=2), encoding="utf-8")

    results = [solve(job, backend) for backend in backends]
    data: dict[str, Any] = {
        "backends": list(backends),
        "measured_on": (
            "in-memory fp32 SolverResult fields, same process, no result.h5 round trip"
        ),
    }
    for key in ("phasor", "p_max"):
        data[key] = field_diff(getattr(results[0], key), getattr(results[1], key))
    data["stored_float16_reference"] = {
        "note": (
            "INFORMATIONAL, not gated: the same two fields after the store's float16 "
            "quantization. ~4.9e-4 relative L-infinity is one float16 ULP and is what "
            "comparing two result.h5 files looks like when the solvers agree exactly."
        ),
        **{
            key: field_diff(
                stored_roundtrip(getattr(results[0], key)),
                stored_roundtrip(getattr(results[1], key)),
            )
            for key in ("phasor", "p_max")
        },
    }
    try:
        import numpy as np  # noqa: PLC0415

        np.savez_compressed(
            outdir / "fields.npz",
            **{
                f"{name}_{key}": np.asarray(getattr(res, key))
                for name, res in zip(backends, results, strict=True)
                for key in ("phasor", "p_max")
            },
        )
    except Exception as exc:  # noqa: BLE001 - evidence is a bonus, not the gate
        data["fields_npz_error"] = f"{type(exc).__name__}: {exc}"
    return data
