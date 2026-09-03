"""Multi-engine comparison: ONE job, N registered solvers, one stamped table.

The library has three registered solvers today (``linear``, ``westervelt``,
``kwave``) and the plan adds more. Every cross-check between them
so far has lived inside a single pytest function:
``tests/test_kwave_adapter.py::test_kwave_vs_linear_2d_water`` builds one 2-D
scene, runs two engines, correlates the normalized fields and asserts
``r > 0.99``. That is a real validation, and it produces nothing: no
artifact, no environment stamp, no row a reader can compare against the row
they get on their own machine, and no way to ask the same question of a
fourth engine without writing a fourth test.

This module is that comparison turned into a command::

    python -m caustica.validation compare
    python -m caustica.validation compare --example water_bowl_mini --solvers linear,kwave

It builds the job ONCE and hands the *same* ``BuiltJob`` -- same grid, same
medium array, same source voxels and phases -- to each solver in turn, on one
backend, in memory. The first solver in the list is the reference; every
other one is compared against it.

Three rules the milestone spells out, and why each exists:

**The comparison is NORMALIZED.** Each field is divided by its own interior
peak before anything is measured. Absolute amplitude is a per-engine
convention, not physics: k-Wave normalizes its additive pressure source
internally while the native engine applies the equivalent ``2 c dt / dx``
mass-source scaling (see :mod:`caustica.solvers.kwave_adapter`, "Source
amplitude", 2026-08-11), so the two agree only "to first order" on absolute
Pa. What a cross-engine check can honestly assert is that the two engines
computed the same *shape*, which is exactly what the existing k-Wave test
does (``a = a / a.max(); b = b / b.max()``) and what is reproduced here. The
raw peaks and their ratio are still reported, under their own column, so the
amplitude convention stays visible instead of being normalized away.

**A T0 sanity gate runs first.** Before any engine's field enters a
comparison it must produce a sane field on a trivial job: finite everywhere,
and an interior peak within a decade of the focal gain
:func:`caustica.analytic.oneill.focal_gain` predicts for that bowl. A broken
k-Wave binary does not announce itself -- it can return zeros, or a truncated
buffer -- and a correlation coefficient computed against zeros is a number,
which is the dangerous part. With this gate the failure is labelled
``t0.kwave.* FAIL`` instead of appearing as a suspicious ``r``.

**A missing environment is a SKIP, stamped verbatim.** A machine without the
k-Wave binary, or without cupy, must not turn a physics report red -- and
must not quietly shrink the table either. Such a solver keeps its row, its
status is ``environment``, and the exception text is copied into the report
word for word so the reader can act on it. The classification is inherited,
not invented: the exception types treated as "this machine cannot run it" are
exactly the ones the existing k-Wave cross-check already skips on.

Every gated tolerance below names the test that established it -- the same
rule as :mod:`caustica.validation.analytic_suite`. Numbers with no such test
(the normalized relative L2, the focal-metric table) are REPORTED and not
gated: a limit invented here would be a limit chosen to pass.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from caustica.validation._verdict import (
    EXIT_ENV,
    EXIT_FAILED,
    EXIT_OK,
    Check,
    Gate,
    fmt_num,
    overall_verdict,
)

#: Report format tag; the JSON companion of the Markdown page.
FORMAT = "caustica-compare/1"


# ------------------------------------------------------------- tolerances
#
# INHERITED, NOT INVENTED. Same rule as the analytic suite: each gated limit
# is one an existing test already validated this library against, and each
# check carries that test's name into the report through ``Check.citing``.

#: The success criterion of the multi-engine comparison, verbatim from the
#: milestone ("mevcut linear-vs-kwave caprazlari harness'ten yeniden
#: uretilir (r>0.99)") and from the test that first measured it.
CROSS_CORR_MIN = 0.99
CROSS_CORR_SOURCE = "tests/test_kwave_adapter.py::test_kwave_vs_linear_2d_water"

#: The same test's second assertion: the peak of the normalized field must
#: land within one voxel per axis. It is what catches an index-order bug
#: (Fortran vs C) that a correlation over a symmetric field can survive.
PEAK_SHIFT_MAX_VOX = 1.0
PEAK_SHIFT_SOURCE = CROSS_CORR_SOURCE

#: T0's band, as a multiplicative factor on the analytic focal gain. This is
#: NOT a physics tolerance and must not be read as one -- the physics gate
#: for bowl agreement is ``oneill`` in the analytic suite, at 5%. This is
#: a garbage detector: a factor of five on each side is far too loose to
#: constrain a solver and far too tight to admit zeros, NaN-adjacent noise, a
#: silently damped source or a truncated buffer, which is the entire failure
#: set a dead external binary produces.
T0_PEAK_BAND = 5.0
T0_PEAK_SOURCE = (
    "tests/test_linear_oneill_3d.py::test_axial_profile_matches_oneill "
    "(via caustica.analytic.oneill.focal_gain)"
)

#: The finiteness half of T0. The NATIVE engines already refuse to return a
#: non-finite field -- that test is where the guarantee lives, after a 256^3
#: rung once returned NaN with exit 0 and was counted as evidence. An
#: EXTERNAL binary has no such guard, so T0 asks the question again at the
#: harness boundary, of every engine alike.
T0_FINITE_SOURCE = (
    "tests/test_divergence_guard.py::test_a_diverging_run_raises_instead_of_returning_nan"
)

#: Exception types that mean "this machine cannot run this solver" rather
#: than "this solver is wrong". Inherited: the live k-Wave cross-check
#: already skips (rather than fails) on exactly these, which is the project's
#: existing statement of where the line falls.
#:
#: ``SolverDivergedError`` is excluded explicitly -- it is a ``RuntimeError``
#: subclass, and a diverged run is a physics result, not a missing binary.
ENV_BROKEN_TYPES: tuple[type[BaseException], ...] = (
    ImportError,
    RuntimeError,
    FileNotFoundError,
    OSError,
)
ENV_BROKEN_SOURCE = CROSS_CORR_SOURCE


# ------------------------------------------------------------------- jobs
#
# Self-contained, like every other job in this package: homogeneous water
# from caustica.materials and a library-built bowl. No external data, so the
# command is runnable on a fresh `pip install caustica` with no downloads.

#: The T0 bowl, in metres, so the analytic gain can be computed from the same
#: numbers the job dict is built from (one source of truth, not two).
T0_DX_MM = 0.5
T0_D_OUTER_MM = 12.0
T0_ROC_MM = 9.0
T0_APEX_Z_MM = 4.0
T0_EXTENT_MM = (20.0, 20.0, 24.0)
T0_PML_MM = 3.0
T0_F0_MHZ = 0.5
T0_AMPLITUDE_KPA = 100.0
T0_C0 = 1500.0  # caustica.materials.water()'s default sound speed


def t0_job(name: str = "t0-sanity") -> dict:
    """The trivial job every solver must survive before it may be compared.

    Deliberately small (40x40x48 at 0.5 mm, ~0.3 s native / ~2 s k-Wave) and
    deliberately a FOCUSED bowl rather than a plane: a bowl has a closed-form
    focal gain in :mod:`caustica.analytic.oneill`, so the expected peak comes
    from the library's own validated physics instead of from a number typed
    into this file. 6 points per wavelength, matching the resolution of the
    k-Wave cross-check this module reproduces.
    """
    return {
        "format": "caustica-job/1",
        "kind": "explicit",
        "name": name,
        "medium": {"kind": "homogeneous"},
        "grid": {
            "ndim": 3,
            "dx_mm": T0_DX_MM,
            "size_mm": list(T0_EXTENT_MM),
            "pml": {"thickness_mm": T0_PML_MM},
        },
        "source": {
            "kind": "array",
            "array": {"kind": "bowl", "d_outer_mm": T0_D_OUTER_MM, "roc_mm": T0_ROC_MM},
            "apex_mm": [T0_EXTENT_MM[0] / 2, T0_EXTENT_MM[1] / 2, T0_APEX_Z_MM],
        },
        "drive": {"f0_mhz": T0_F0_MHZ, "amplitude_kpa": T0_AMPLITUDE_KPA},
        "run": {
            "spec": {"min_settle_periods": 4, "max_settle_periods": 16, "n_record_periods": 2},
            "harmonics": [1],
        },
        # Overridden per solver; the field only records what the dict says.
        "solver": "linear",
    }


def t0_expected_peak_pa() -> float:
    """Analytic focal pressure of the T0 bowl, from the job's own numbers.

    ``focal_gain`` returns ``|p(focus)| / (rho c u0) = k h``; the job drives a
    pressure source whose realized surface amplitude is ``p0`` (the mass-source
    normalization contract, tests/test_review_hardening.py), so the expected
    focal peak is ``p0 * k h``.
    """
    from caustica.analytic.oneill import focal_gain  # noqa: PLC0415

    gain = focal_gain(
        aperture_radius=T0_D_OUTER_MM / 2.0 * 1e-3,
        roc=T0_ROC_MM * 1e-3,
        f0=T0_F0_MHZ * 1e6,
        c0=T0_C0,
    )
    return float(gain * T0_AMPLITUDE_KPA * 1e3)


def mini_job(name: str = "compare-mini") -> dict:
    """The default scenario: a mini 3-D focused bowl in water.

    Small enough that three engines finish in seconds, real enough to be a
    cross-check: a curved source, a PML, a settling history and a focal lobe
    with measurable -6 dB extents. 0.5 mm at 0.5 MHz is 6 points per
    wavelength -- the SAME resolution as the k-Wave cross-check whose
    tolerance this harness inherits, so ``r > 0.99`` is being asked of a
    field the tolerance was actually established on.
    """
    extent = (24.0, 24.0, 30.0)
    return {
        "format": "caustica-job/1",
        "kind": "explicit",
        "name": name,
        "medium": {"kind": "homogeneous"},
        "grid": {
            "ndim": 3,
            "dx_mm": 0.5,
            "size_mm": list(extent),
            # 10 voxels, comfortably clear of the bowl on every face: k-Wave
            # puts its PML INSIDE the grid and refuses a source that touches
            # it, so a job usable by every engine must budget for that band.
            "pml": {"thickness_mm": 5.0},
        },
        "source": {
            "kind": "array",
            "array": {"kind": "bowl", "d_outer_mm": 12.0, "roc_mm": 10.0},
            "apex_mm": [extent[0] / 2, extent[1] / 2, 6.0],
        },
        "drive": {"f0_mhz": 0.5, "amplitude_kpa": 100.0},
        "run": {
            "spec": {"min_settle_periods": 4, "max_settle_periods": 20, "n_record_periods": 2},
            "harmonics": [1],
        },
        "solver": "linear",
    }


# --------------------------------------------------------- field comparison


def field_frame_of(built: Any) -> Any:
    """The built job's grid frame, as :class:`~caustica.report.metrics.FieldFrame`.

    One spelling for the harness: the T0 probe and the real comparison both
    measure in the frame ``caustica report`` measures in.
    """
    from caustica.report.metrics import FieldFrame  # noqa: PLC0415

    return FieldFrame(
        dx=built.grid.dx,
        grid_shape=built.grid.shape,
        pml_vox=built.grid.pml_vox,
        apex_vox=tuple(int(v) for v in built.derived.get("apex_vox", (0, 0, 0))),
        focus_vox=built.focus_vox,
    )


def interior_amp(result: Any, frame: Any) -> np.ndarray:
    """``|phasor|`` over the record region with the sponge shaved off.

    Uses :func:`caustica.report.metrics.interior_slices`, the library's own
    definition of "interior", so the window this harness compares in is the
    same window ``caustica report`` measures the focal metrics in -- a
    comparison and a metric table that disagreed about where the field is
    would be two reports, not one.
    """
    from caustica.report.metrics import interior_slices  # noqa: PLC0415

    return np.asarray(result.amp[interior_slices(result, frame)])


def compare_fields(reference: np.ndarray, compared: np.ndarray) -> dict:
    """Normalized agreement between two amplitude fields. Pure, no I/O.

    Each field is divided by its OWN peak first (see the module docstring:
    absolute amplitude is an engine convention, shape is the physics), then:

    * ``rel_l2`` -- ``||a_n - b_n||_2 / ||a_n||_2`` over every interior voxel.
      Whole-field on purpose: a peak-only comparison can agree to 1e-7 while
      a whole plane disagrees.
    * ``pearson_r`` -- the quantity the existing k-Wave cross-check gates.
    * ``peak_shift_vox`` -- the largest per-axis displacement of the interior
      peak, which is how that test writes its second assertion.

    Anything unmeasurable comes back ``None`` with a ``note``, never a zero:
    a null propagates into a SKIP downstream, and SKIP is never a pass.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(compared, dtype=np.float64)
    out: dict[str, Any] = {
        "n_voxels": int(a.size),
        "ref_peak_pa": None,
        "cmp_peak_pa": None,
        "peak_ratio": None,
        "rel_l2": None,
        "pearson_r": None,
        "peak_idx_ref": None,
        "peak_idx_cmp": None,
        "peak_shift_vox": None,
        "peak_shift_norm_vox": None,
        "note": None,
    }
    if a.shape != b.shape:
        out["note"] = f"shape mismatch: {a.shape} != {b.shape}"
        return out
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        out["note"] = "a field contains non-finite values; nothing normalizable"
        return out

    pa, pb = float(a.max()), float(b.max())
    out["ref_peak_pa"], out["cmp_peak_pa"] = pa, pb
    if pa <= 0.0 or pb <= 0.0:
        out["note"] = f"a field has no positive peak (reference {pa:g}, compared {pb:g})"
        return out
    out["peak_ratio"] = pb / pa

    idx_a = np.unravel_index(int(np.argmax(a)), a.shape)
    idx_b = np.unravel_index(int(np.argmax(b)), b.shape)
    shift = np.array(idx_b, dtype=float) - np.array(idx_a, dtype=float)
    out["peak_idx_ref"] = [int(i) for i in idx_a]
    out["peak_idx_cmp"] = [int(i) for i in idx_b]
    out["peak_shift_vox"] = float(np.abs(shift).max())
    out["peak_shift_norm_vox"] = float(np.linalg.norm(shift))

    an, bn = a / pa, b / pb
    denom = float(np.linalg.norm(an))
    out["rel_l2"] = float(np.linalg.norm(an - bn) / denom) if denom > 0 else None
    if an.std() == 0.0 or bn.std() == 0.0:
        # corrcoef would return nan here and nan reads as "measured" in a
        # table. A constant field has no shape to correlate; say that.
        out["note"] = "a normalized field is constant; Pearson r is undefined"
        return out
    out["pearson_r"] = float(np.corrcoef(an.ravel(), bn.ravel())[0, 1])
    return out


# ----------------------------------------------------------- solver driving


def is_environment_error(exc: BaseException) -> bool:
    """Is this exception "this machine cannot", rather than "this is wrong"?

    See :data:`ENV_BROKEN_TYPES`. The one carve-out is
    :class:`caustica.solvers.base.SolverDivergedError`, a ``RuntimeError``
    subclass that reports a physics outcome and must stay a failure.
    """
    from caustica.solvers.base import SolverDivergedError  # noqa: PLC0415

    if isinstance(exc, SolverDivergedError):
        return False
    return isinstance(exc, ENV_BROKEN_TYPES)


def verbatim(exc: BaseException) -> str:
    """``TypeName: message`` -- the error as the report must repeat it."""
    return f"{type(exc).__name__}: {exc}"


def uses_backend(solver_cls: type) -> bool:
    """Does this solver take caustica's ``backend=`` argument at all?

    An adapter declaring ``backends={"external"}`` (k-Wave) drives its own
    engine and rejects unknown ``run()`` options, so the argument must not be
    passed to it rather than passed and ignored.
    """
    return "external" not in solver_cls.caps.backends


def caps_refusal(solver_cls: type, built: Any, backend: str) -> str | None:
    """Why this solver cannot take this job, or ``None`` if it can.

    Delegates the physics half to the solver's own
    :meth:`~caustica.solvers.base.SolverBase.validate`, so the reason a job
    is refused here is word for word the reason a user would get from
    ``Simulation`` -- one capability contract, not a second opinion.
    """
    try:
        solver_cls().validate(built.grid, built.medium, built.source)
    except Exception as exc:  # noqa: BLE001 - any refusal is a refusal
        return verbatim(exc)
    if uses_backend(solver_cls) and backend not in solver_cls.caps.backends:
        return (
            f"solver '{getattr(solver_cls, 'name', solver_cls.__name__)}' declares backends "
            f"{sorted(solver_cls.caps.backends)}; this run resolved backend '{backend}'"
        )
    return None


def run_built(solver_cls: type, built: Any, backend: str) -> Any:
    """One solve of an already-built job on one solver. No disk in between.

    The same shape as :func:`caustica.validation.gpu_gates.solve_job_dict`,
    with the one difference this suite exists for: the SOLVER varies and the
    built job does not. Building once and reusing the object is the point --
    a second ``build_job`` would rasterize a second source and the comparison
    would silently include the builder.
    """
    kwargs: dict[str, Any] = {
        "record_region": built.record_region,
        "reference_point": built.focus_vox,
        "harmonics": built.harmonics,
    }
    if uses_backend(solver_cls):
        kwargs["backend"] = backend
    return solver_cls().run(built.grid, built.medium, built.source, built.spec, **kwargs)


# ------------------------------------------------------------- the harness


@dataclass
class Harness:
    """Every way this suite touches the world, in one substitutable object.

    Same seam, same reason, as the other two suites: a fake ``get_solver``
    lets the whole pipeline -- caps filtering, T0 grading, pairing, report
    schema, exit codes -- be exercised in milliseconds against engines that
    return canned fields, so a wrong verdict is caught without paying for a
    solve. The real path is still run for real, once, end to end.
    """

    env_report: Callable[[], dict]
    backend: Callable[[str], str]
    get_solver: Callable[[str], type]
    available: Callable[[], tuple[str, ...]]


def default_harness() -> Harness:
    """The real thing: the registry, on whatever backend the request resolves to."""
    import caustica.solvers as solver_pkg  # noqa: PLC0415
    from caustica import env as env_mod  # noqa: PLC0415
    from caustica.core.backend import get_backend  # noqa: PLC0415

    def _backend(requested: str) -> str:
        try:
            return str(get_backend(requested).name)
        except Exception:  # noqa: BLE001 - a broken CUDA stack must not lose the run
            return "numpy"

    return Harness(
        env_report=env_mod.env_report,
        backend=_backend,
        get_solver=solver_pkg.get,
        available=solver_pkg.available,
    )


# ------------------------------------------------------------- job sourcing


def resolve_job(
    job: dict | None = None,
    *,
    path: str | Path | None = None,
    example: str | None = None,
) -> tuple[dict, Path | None, str]:
    """The job dict to compare on, its base directory, and where it came from.

    A packaged example is READ, never run in place: this suite writes only
    into its own report folder, so the warning in :mod:`caustica.examples`
    (relative outputs resolving into ``site-packages``) does not apply -- the
    example's own directory is still handed back as the base dir so a job
    with relative medium paths resolves the way it would for its owner.
    """
    given = [x is not None for x in (job, path, example)]
    if sum(given) > 1:
        raise ValueError("give at most one of job=, path=, example=")
    if example is not None:
        from caustica import examples  # noqa: PLC0415

        p = examples.path(example)
        return json.loads(p.read_text(encoding="utf-8")), p.parent, f"example:{example}"
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise ValueError(f"no job file at {p}")
        return json.loads(p.read_text(encoding="utf-8")), p.parent, f"file:{p}"
    if job is not None:
        return job, None, "dict"
    return mini_job(), None, "builtin:compare-mini"


def choose_solvers(
    built: Any,
    backend: str,
    harness: Harness,
    requested: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """The solver list and, for the default list, why anything was left out.

    With no ``requested`` list the default is "every registered solver whose
    caps accept this job", ordered with the job's OWN solver first: the job
    names the engine it was written for, so that engine is the reference and
    the others are the cross-check. An explicit list is taken verbatim, first
    entry as reference -- including solvers whose caps refuse the job, which
    then appear in the report as an ``unsupported`` SKIP rather than being
    dropped behind the user's back.
    """
    known = tuple(harness.available())
    if requested is not None:
        unknown = [s for s in requested if s not in known]
        if unknown:
            raise ValueError(f"unknown solver(s) {unknown}; registered: {', '.join(known)}")
        return list(dict.fromkeys(requested)), {}

    excluded: dict[str, str] = {}
    accepted: list[str] = []
    for name in known:
        reason = caps_refusal(harness.get_solver(name), built, backend)
        if reason is None:
            accepted.append(name)
        else:
            excluded[name] = reason
    # The job's own solver leads; the registry's order decides the rest.
    if built.solver in accepted:
        accepted = [built.solver] + [n for n in accepted if n != built.solver]
    return accepted, excluded


# ------------------------------------------------------------------ gates


def evaluate(runs: dict[str, dict], pairs: list[dict], reference: str) -> list[Gate]:
    """Turn the measurements into milestone verdicts. Pure -- no I/O, no solve."""
    t0 = Gate(
        id="t0",
        criterion=(
            "every solver with a working environment produces a finite field on the "
            "trivial T0 job, with an interior peak within a factor "
            f"{T0_PEAK_BAND:g} of the analytic focal gain, BEFORE its field is allowed "
            "into any comparison"
        ),
        required=0,
    )
    attempted = 0
    for name, run in runs.items():
        data = run.get("t0") or {}
        status = data.get("status", "error")
        if status in ("environment", "unsupported"):
            # Stamped, not dropped: the row survives, carrying the reason.
            t0.checks.append(
                Check(
                    f"t0.{name}",
                    "SKIP",
                    data.get("message") or f"{status}: no field to sanity-check",
                    {"status": status, "source": ENV_BROKEN_SOURCE},
                )
            )
            continue
        if status == "error":
            # No ``source``: this check records that an engine RAISED, which
            # is not a tolerance and has no test to inherit one from. The
            # report prints "--" there rather than an irrelevant citation.
            t0.checks.append(
                Check(
                    f"t0.{name}",
                    "FAIL",
                    data.get("message") or "the T0 run raised",
                    {"status": status},
                )
            )
            continue
        attempted += 1
        t0.checks.append(
            Check.happened(
                f"t0.{name}.finite",
                data.get("finite"),
                True,
                detail=(
                    "phasor and p_max are finite everywhere"
                    if data.get("finite")
                    else "the T0 field contains NaN or inf"
                ),
            ).citing(T0_FINITE_SOURCE)
        )
        t0.checks.append(
            Check.in_range(
                f"t0.{name}.peak",
                data.get("peak_over_expected"),
                1.0 / T0_PEAK_BAND,
                T0_PEAK_BAND,
                unit=" x analytic",
            ).citing(T0_PEAK_SOURCE)
        )
    t0.required = 2 * attempted

    corr = Gate(
        id="cross",
        criterion=(
            f"normalized |phasor| correlation r >= {CROSS_CORR_MIN} between the reference "
            f"solver '{reference}' and every other compared solver"
        ),
        required=0,
    )
    focus = Gate(
        id="focus",
        criterion=(
            f"the interior peak of the normalized field agrees within "
            f"{PEAK_SHIFT_MAX_VOX:g} voxel per axis across engines"
        ),
        required=0,
    )
    measurable = 0
    for pair in pairs:
        label = f"{pair['reference']}-vs-{pair['compared']}"
        if pair.get("status") != "ok":
            reason = pair.get("message") or "not compared"
            corr.checks.append(
                Check(f"{label}.corr", "SKIP", reason, {"source": CROSS_CORR_SOURCE})
            )
            focus.checks.append(
                Check(f"{label}.peak_shift", "SKIP", reason, {"source": PEAK_SHIFT_SOURCE})
            )
            continue
        measurable += 1
        corr.checks.append(
            Check.at_least(f"{label}.corr", pair.get("pearson_r"), CROSS_CORR_MIN).citing(
                CROSS_CORR_SOURCE
            )
        )
        focus.checks.append(
            Check.at_most(
                f"{label}.peak_shift", pair.get("peak_shift_vox"), PEAK_SHIFT_MAX_VOX, unit=" vox"
            ).citing(PEAK_SHIFT_SOURCE)
        )
    corr.required = measurable
    focus.required = measurable
    return [t0, corr, focus]


def exit_code(runs: dict[str, dict], verdict: str) -> int:
    """0 pass, 2 this machine cannot, 4 it ran and the answer is no.

    The 2 is narrow on purpose: it means *nothing was compared and nothing
    failed* -- fewer than two solvers had a working environment, and no run
    that did happen went wrong. Anything else that is not a PASS is a 4, so a
    missing binary can never mask a real disagreement.
    """
    if verdict == "PASS":
        return EXIT_OK
    ran = [r for r in runs.values() if r.get("status") == "ok"]
    blocked = [r for r in runs.values() if r.get("status") in ("environment", "unsupported")]
    broke = [r for r in runs.values() if r.get("status") in ("sanity", "error")]
    if len(ran) < 2 and blocked and not broke:
        return EXIT_ENV
    return EXIT_FAILED


# ----------------------------------------------------------------- report


def _cell(text: Any) -> str:
    """One Markdown table cell, with the pipes that would split the row escaped."""
    return str(text).replace("|", "\\|")


def _get(d: Any, *keys: str) -> Any:
    """Nested lookup that answers ``None`` instead of raising on a gap."""
    for key in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def reproduce_command(payload: dict) -> str:
    """The exact command line that regenerates this report.

    Built from the payload rather than from the parsed arguments, so a report
    read months later names the job it actually ran instead of whatever the
    default would pick up on the day somebody retypes the command.
    """
    kind, _, value = str(payload.get("job_source", "")).partition(":")
    selector = {"example": f" --example {value}", "file": f" --job {value}"}.get(kind, "")
    return (
        "python -m caustica.validation compare"
        + selector
        + f" --solvers {','.join(payload.get('solvers', []))}"
        + f" --backend {payload.get('backend_requested')}"
    )


def render_markdown(payload: dict) -> str:
    """The human half of the report. The JSON is the machine half."""
    env = payload.get("environment", {})
    runs: dict[str, dict] = payload.get("runs", {})
    pairs: list[dict] = payload.get("pairs", [])
    lines = [
        "# caustica multi-engine comparison",
        "",
        f"**Overall verdict: {payload['verdict']}** (`{FORMAT}`, generated {payload['generated']})",
        "",
        "The SAME built job -- one grid, one medium, one source -- run on each registered "
        "solver below, in memory, on one backend. The first solver is the reference; every "
        "other is compared against it. Comparison is NORMALIZED (each field divided by its "
        "own interior peak): absolute amplitude is a per-engine source convention, field "
        "SHAPE is the cross-check. Gated tolerances name the test they are inherited from; "
        "ungated columns are reported because no test in this repository establishes a "
        "cross-engine limit for them.",
        "",
        "| | |",
        "|---|---|",
        f"| job | {_cell(payload.get('job_name'))} (`{_cell(payload.get('job_source'))}`) |",
        f"| grid | {_cell(payload.get('grid_shape'))} @ {fmt_num(payload.get('dx_mm'))} mm, "
        f"PML {payload.get('pml_vox')} vox |",
        f"| reference solver | `{_cell(payload.get('reference'))}` |",
        f"| backend | {payload.get('backend')} (requested `{payload.get('backend_requested')}`) |",
        f"| host | {payload.get('host')} |",
        f"| caustica | {payload.get('caustica')} @ {payload.get('git_commit')} |",
        f"| python / numpy | {env.get('python')} / {env.get('numpy')} |",
        f"| platform | {env.get('platform')} |",
        f"| suite runtime | {fmt_num(payload.get('elapsed_s'))} s |",
        "",
        "## solvers",
        "",
        "`environment` is a machine problem, never a physics verdict: the row stays, the "
        "error is copied verbatim, and its checks are SKIP. `unsupported` means the "
        "solver's own capability contract refused this job.",
        "",
        "| solver | status | T0 s | run s | steps | interior peak (MPa) | detail |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        peak = _get(run, "run", "peak_pa")
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                _cell(name),
                run.get("status", "--"),
                fmt_num(_get(run, "t0", "elapsed_s")),
                fmt_num(_get(run, "run", "elapsed_s")),
                fmt_num(_get(run, "run", "steps_total")),
                fmt_num(None if peak is None else peak / 1e6),
                _cell(run.get("message") or "--"),
            )
        )

    lines += [
        "",
        "## pairs (normalized |phasor|)",
        "",
        f"`r` and the peak shift are GATED at r >= {CROSS_CORR_MIN} and "
        f"<= {PEAK_SHIFT_MAX_VOX:g} voxel (`{CROSS_CORR_SOURCE}`). `rel L2` and the peak "
        "ratio are reported, not gated -- no test in this repository establishes a "
        "cross-engine limit for either, and one invented here would be one chosen to pass. "
        "The peak ratio is the UN-normalized amplitude comparison, kept visible so an "
        "engine's source convention cannot hide inside the normalization.",
        "",
        "| pair | rel L2 (norm) | Pearson r | peak shift (vox) | peak ratio cmp/ref | voxels |",
        "|---|---|---|---|---|---|",
    ]
    for pair in pairs:
        label = f"{pair['reference']} vs {pair['compared']}"
        if pair.get("status") != "ok":
            lines.append(
                f"| {_cell(label)} | -- | -- | -- | -- | "
                f"{_cell(pair.get('message') or 'not compared')} |"
            )
            continue
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                _cell(label),
                fmt_num(pair.get("rel_l2")),
                fmt_num(pair.get("pearson_r")),
                fmt_num(pair.get("peak_shift_vox")),
                fmt_num(pair.get("peak_ratio")),
                fmt_num(pair.get("n_voxels")),
            )
        )

    lines += [
        "",
        "## focal metrics",
        "",
        "`caustica.report.metrics.focus_metrics` -- the same function `caustica report` "
        "and `apps/focus_study` use, so these numbers are comparable to any run folder's "
        "`metrics.json`. Reported per engine, not gated.",
        "",
        "| solver | peak (MPa) | peak voxel | z from apex (mm) | axial -6 dB (mm) | "
        "lat-x -6 dB (mm) | lat-y -6 dB (mm) | vol >-6 dB (mm3) | ISPPA (W/cm2) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        m = run.get("metrics") or {}
        if not m or m.get("error"):
            lines.append(
                f"| `{_cell(name)}` | -- | -- | -- | -- | -- | -- | -- | "
                f"{_cell((m.get('error') if m else None) or 'not measured')} |"
            )
            continue
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _cell(name),
                fmt_num(_get(m, "peak", "p_mpa")),
                _cell(_get(m, "peak", "voxel_grid")),
                fmt_num(_get(m, "peak", "position_mm_from_apex", "z")),
                fmt_num(_get(m, "focal_spot", "axial_6db", "width_mm")),
                fmt_num(_get(m, "focal_spot", "lateral_x_6db", "width_mm")),
                fmt_num(_get(m, "focal_spot", "lateral_y_6db", "width_mm")),
                fmt_num(_get(m, "focal_spot", "volume_above_6db_mm3")),
                fmt_num(_get(m, "peak", "isppa_w_cm2")),
            )
        )

    lines += ["", "## gates", ""]
    for gate in payload.get("gates", []):
        lines += [
            f"### {gate['id']} -- {gate['verdict']}",
            "",
            # max(1, ...) mirrors Gate.verdict's own rule: a gate with nothing
            # to count still needs one passing check, and printing "needs 0"
            # next to INCOMPLETE would read as a contradiction.
            f"{gate['criterion']} (needs {max(1, gate['required'])} passing check(s), "
            f"has {gate['n_pass']}).",
            "",
            "| check | verdict | measured vs limit | source of the tolerance |",
            "|---|---|---|---|",
        ]
        for c in gate["checks"]:
            lines.append(
                f"| {_cell(c['name'])} | {c['verdict']} | {_cell(c['detail'])} | "
                f"{_cell(c.get('source', '--'))} |"
            )
        lines.append("")

    notes = payload.get("notes") or []
    if notes:
        lines += ["## notes", ""] + [f"* {_cell(n)}" for n in notes] + [""]

    lines += [
        f"Reproduce: `{_cell(reproduce_command(payload))}`. The job that was actually built "
        "is stored beside this file as `job_input.json`.",
        "",
    ]
    return "\n".join(lines)


def write_report(outdir: Path, payload: dict) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "compare.json"
    md_path = outdir / "REPORT.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, md_path


# ------------------------------------------------------------------ suite


def _t0_for(solver_cls: type, built_t0: Any, backend: str, expected_pa: float) -> dict:
    """Grade one solver on the trivial job. Never raises."""
    data: dict[str, Any] = {
        "status": "ok",
        "message": "",
        "expected_peak_pa": expected_pa,
        "finite": None,
        "peak_pa": None,
        "peak_over_expected": None,
        "elapsed_s": None,
        "steps_total": None,
    }
    reason = caps_refusal(solver_cls, built_t0, backend)
    if reason is not None:
        data["status"] = "unsupported"
        data["message"] = reason
        return data
    started = time.perf_counter()
    try:
        result = run_built(solver_cls, built_t0, backend)
    except Exception as exc:  # noqa: BLE001 - the classification IS the product here
        data["status"] = "environment" if is_environment_error(exc) else "error"
        data["message"] = verbatim(exc)
        data["elapsed_s"] = time.perf_counter() - started
        return data
    data["elapsed_s"] = time.perf_counter() - started
    data["steps_total"] = int(result.steps_total)
    data["finite"] = bool(
        np.isfinite(np.asarray(result.phasor)).all() and np.isfinite(np.asarray(result.p_max)).all()
    )
    try:
        amp = interior_amp(result, field_frame_of(built_t0))
        peak = float(amp.max())
    except Exception as exc:  # noqa: BLE001 - unmeasurable is a T0 failure, not a crash
        # Defensive: only reachable if the record region is swallowed by the
        # sponge. A peak that cannot be measured has not passed T0, so the
        # engine must not go on to the real job on the strength of it.
        data["status"] = "failed"
        data["message"] = f"T0 peak could not be measured: {verbatim(exc)}"
        return data
    data["peak_pa"] = peak
    if np.isfinite(peak) and expected_pa > 0:
        data["peak_over_expected"] = peak / expected_pa
    passed = bool(data["finite"]) and (
        data["peak_over_expected"] is not None
        and 1.0 / T0_PEAK_BAND <= data["peak_over_expected"] <= T0_PEAK_BAND
    )
    if not passed:
        data["status"] = "failed"
        data["message"] = (
            f"T0 sanity failed: finite={data['finite']}, peak={fmt_num(data['peak_pa'])} Pa "
            f"({fmt_num(data['peak_over_expected'])} x the analytic "
            f"{fmt_num(expected_pa)} Pa, band 1/{T0_PEAK_BAND:g}..{T0_PEAK_BAND:g})"
        )
    return data


def compare(
    *,
    job: dict | None = None,
    job_path: str | Path | None = None,
    example: str | None = None,
    solvers: Sequence[str] | None = None,
    backend: str = "auto",
    out: str | Path | None = None,
    harness: Harness | None = None,
    log: Callable[[str], None] = print,
) -> tuple[int, dict]:
    """Run one job on N solvers and write the stamped comparison.

    Returns ``(exit_code, report_payload)``.
    """
    from caustica import __version__ as version  # noqa: PLC0415
    from caustica.config.job import build_job, parse_job  # noqa: PLC0415
    from caustica.env import git_commit  # noqa: PLC0415

    harness = harness or default_harness()
    notes: list[str] = []
    started = time.perf_counter()

    job_dict, base_dir, job_source = resolve_job(job, path=job_path, example=example)
    resolved_backend = harness.backend(backend)

    env = harness.env_report()
    log("--- environment ----------------------------------------------")
    for key, value in env.items():
        log(f"  {key}: {value}")

    # Built ONCE. Every solver gets this object, so nothing about the scene
    # can differ between the engines being compared.
    built = build_job(parse_job(job_dict, job_dict.get("name", "compare")), base_dir, True)
    built_t0 = build_job(parse_job(t0_job(), "t0-sanity"), None, True)
    expected_pa = t0_expected_peak_pa()

    names, excluded = choose_solvers(built, resolved_backend, harness, solvers)
    for name, reason in excluded.items():
        notes.append(f"solver {name!r} left out of the default list: {reason}")
    if not names:
        raise ValueError(
            f"no registered solver accepts this job on backend {resolved_backend!r}: "
            + "; ".join(f"{n}: {r}" for n, r in excluded.items())
        )
    reference = names[0]

    stamp = datetime.now(timezone.utc)
    root = Path(out) if out is not None else Path("benchmarks/reports/compare")
    outdir = root / stamp.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "job_input.json").write_text(json.dumps(job_dict, indent=2), encoding="utf-8")

    log(
        f"\njob: {built.name} {built.grid.shape} @ {built.grid.dx * 1e3:g} mm"
        f"   backend: {resolved_backend} (requested {backend})"
        f"\nsolvers: {', '.join(names)}   reference: {reference}"
        f"\nreport folder: {outdir}"
    )

    frame = field_frame_of(built)

    runs: dict[str, dict] = {}
    amps: dict[str, np.ndarray] = {}
    for name in names:
        log(f"\n=== {name} ===")
        solver_cls = harness.get_solver(name)
        record: dict[str, Any] = {
            "solver": name,
            "status": "ok",
            "message": "",
            "takes_backend": uses_backend(solver_cls),
            "caps": {
                "ndim": sorted(solver_cls.caps.ndim),
                "nonlinear": bool(solver_cls.caps.nonlinear),
                "backends": sorted(solver_cls.caps.backends),
            },
            "t0": None,
            "run": None,
            "metrics": None,
        }
        runs[name] = record

        # --- T0 first: a field may not enter a comparison unsanitized. ---
        t0_data = _t0_for(solver_cls, built_t0, resolved_backend, expected_pa)
        record["t0"] = t0_data
        log(
            f"  T0: {t0_data['status']}"
            + (f" ({t0_data['message']})" if t0_data["message"] else "")
            + (
                f"  peak {fmt_num(t0_data['peak_pa'])} Pa = "
                f"{fmt_num(t0_data['peak_over_expected'])} x analytic"
                if t0_data["peak_pa"] is not None
                else ""
            )
        )
        if t0_data["status"] != "ok":
            record["status"] = {
                "environment": "environment",
                "unsupported": "unsupported",
                "failed": "sanity",
            }.get(t0_data["status"], "error")
            record["message"] = t0_data["message"]
            notes.append(f"solver {name!r}: {record['status']} -- {t0_data['message']}")
            continue

        # --- the real job ---
        t_start = time.perf_counter()
        try:
            result = run_built(solver_cls, built, resolved_backend)
        except Exception as exc:  # noqa: BLE001 - one bad engine must not lose the rest
            record["status"] = "environment" if is_environment_error(exc) else "error"
            record["message"] = verbatim(exc)
            record["run"] = {"elapsed_s": time.perf_counter() - t_start}
            log(f"  run raised {record['message']}")
            notes.append(f"solver {name!r}: {record['status']} -- {record['message']}")
            continue
        elapsed = time.perf_counter() - t_start

        amp = interior_amp(result, frame)
        amps[name] = amp
        record["run"] = {
            "elapsed_s": elapsed,
            "steps_total": int(result.steps_total),
            "spp": int(result.spp),
            "dt_s": float(result.dt),
            "converged_period": int(result.converged_period),
            "settle_capped": bool(result.settle_capped),
            "engine_backend": str(result.meta.get("backend", resolved_backend)),
            "peak_pa": float(amp.max()) if amp.size else None,
            "interior_shape": list(amp.shape),
        }
        log(
            f"  ran in {elapsed:.2f} s, {result.steps_total} steps, "
            f"interior peak {fmt_num(record['run']['peak_pa'])} Pa"
        )

        try:
            from caustica.report.metrics import focus_metrics  # noqa: PLC0415

            record["metrics"] = focus_metrics(
                result,
                frame,
                source_amplitude=built.source.amplitude,
                medium=built.medium,
                solver=name,
            )
        except Exception as exc:  # noqa: BLE001 - metrics are a column, not the gate
            # focus_metrics is 3-D (it indexes an apex z and three profiles);
            # a 2-D job loses the table, not the comparison.
            record["metrics"] = {"error": verbatim(exc)}
            notes.append(f"focal metrics unavailable for {name!r}: {verbatim(exc)}")

    # --------------------------------------------------------- pairing
    pairs: list[dict] = []
    for name in names[1:]:
        pair: dict[str, Any] = {"reference": reference, "compared": name, "status": "ok"}
        if reference not in amps:
            pair["status"] = "skipped"
            pair["message"] = (
                f"reference solver {reference!r} produced no field "
                f"({runs[reference]['status']}: {runs[reference]['message']})"
            )
        elif name not in amps:
            pair["status"] = "skipped"
            pair["message"] = f"{runs[name]['status']}: {runs[name]['message']}"
        else:
            pair.update(compare_fields(amps[reference], amps[name]))
            if pair.get("note"):
                notes.append(f"pair {reference} vs {name}: {pair['note']}")
        pairs.append(pair)

    gates = evaluate(runs, pairs, reference)
    verdict = overall_verdict(gates)
    payload = {
        "format": FORMAT,
        "generated": stamp.isoformat(timespec="seconds"),
        "caustica": version,
        "git_commit": git_commit(),
        "host": platform.node(),
        "backend": resolved_backend,
        "backend_requested": backend,
        "environment": env,
        "job_source": job_source,
        "job_name": built.name,
        "job": job_dict,
        "grid_shape": list(built.grid.shape),
        "dx_mm": built.grid.dx * 1e3,
        "pml_vox": built.grid.pml_vox,
        "t0_job": {"name": "t0-sanity", "expected_peak_pa": expected_pa, "band": T0_PEAK_BAND},
        "solvers": names,
        "reference": reference,
        "excluded": excluded,
        "runs": runs,
        "pairs": pairs,
        "gates": [g.as_dict() for g in gates],
        "verdict": verdict,
        "elapsed_s": time.perf_counter() - started,
        "notes": notes,
    }

    json_path, md_path = write_report(outdir, payload)
    log("\n--- verdict --------------------------------------------------")
    for g in gates:
        log(f"  {g.verdict:<11} {g.id}: {g.criterion}")
    log(f"\noverall: {verdict}  ({payload['elapsed_s']:.1f} s)")
    log(f"report: {md_path}")
    log(f"json:   {json_path}")
    return exit_code(runs, verdict), payload
