"""The PASS/FAIL/SKIP algebra every validation suite grades itself with.

Extracted from :mod:`caustica.validation.gpu_gates` when the analytic suite
arrived and needed exactly the same three things: a check that carries the
numbers that produced it, a gate that knows how MANY checks its milestone
criterion requires, and an overall verdict that is PASS only when nothing is
missing. Two copies of this would have been two chances to disagree about
what a missing measurement means.

The rule the whole package rests on: **a measurement that could not be taken
is never a pass.** Every constructor below answers ``SKIP`` when a side of its
comparison is absent, a gate with too few passes is ``INCOMPLETE`` rather than
PASS, and :func:`overall_verdict` refuses an empty gate list. A suite that
silently upgrades an unmeasured number to a green box is worse than no suite,
because it is believed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

#: Exit codes. 0/2/4 are borrowed from the runner's vocabulary on purpose --
#: 2 is "this environment cannot do it", 4 is "it ran and the answer is no".
EXIT_OK = 0
EXIT_ENV = 2
EXIT_FAILED = 4


@dataclass(frozen=True)
class Check:
    """One PASS/FAIL/SKIP decision, with the numbers that produced it.

    The constructors are the shapes of question a gate suite asks: is a
    prediction within a tolerance of a measurement (:meth:`relative`), is a
    measurement under a limit (:meth:`at_most`) or over a floor
    (:meth:`at_least`), does it sit inside a stated band (:meth:`in_range`),
    and did something specific happen (:meth:`happened`).

    ``SKIP`` is load-bearing. A check that could not be evaluated -- a rung
    that never ran, a scenario that raised, a stamp with a null field -- must
    not silently become a pass, which is the single most likely way a suite
    like this lies.
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
        """``|predicted - actual| / |actual| <= tol_pct`` -- the planner gate shape."""
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
    def at_least(name: str, value: float | None, floor: float, unit: str = "") -> Check:
        """``value >= floor`` -- a correlation or a margin that must not drop.

        The mirror of :meth:`at_most`, and not worth expressing as one: a
        correlation gate written ``at_most(1 - r, 0.01)`` grades the right
        thing and then reports a number no reader recognises as the gate.
        """
        data = {"value": value, "floor": floor, "unit": unit}
        if value is None or not math.isfinite(value):
            return Check(name, "SKIP", "not measured", data)
        return Check(
            name,
            "PASS" if value >= floor else "FAIL",
            f"{value:.6g}{unit} (floor {floor:.6g}{unit})",
            data,
        )

    @staticmethod
    def in_range(name: str, value: float | None, low: float, high: float, unit: str = "") -> Check:
        """``low <= value <= high`` -- for a contract stated as a BAND.

        A band is not a symmetric tolerance and must not be flattened into
        one. The contract that motivated this constructor is the realized
        drive amplitude, whose own test pins it to ``0.90 .. 1.12 x p0``
        (tests/test_review_hardening.py); restating that as "+/-10%" would
        invent a stricter gate than the one the library actually validated.
        """
        data = {"value": value, "low": low, "high": high, "unit": unit}
        if value is None or not math.isfinite(value):
            return Check(name, "SKIP", "not measured", data)
        return Check(
            name,
            "PASS" if low <= value <= high else "FAIL",
            f"{value:.6g}{unit} (band {low:g}..{high:g}{unit})",
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

    def citing(self, source: str) -> Check:
        """The same check, stamped with the test that established its limit.

        A validation suite free to pick its own tolerances is a suite that can
        be tuned until it passes. Every gated number in the analytic suite
        carries the test that already validated it, and the report prints that
        column, so a reader can go read the gate this one inherits.
        """
        return Check(self.name, self.verdict, self.detail, {**self.data, "source": source})


@dataclass
class Gate:
    """One milestone criterion and the checks that answer it.

    ``required`` is the criterion's own wording made countable: it says "at
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


def fmt_num(value: Any) -> str:
    """A table cell: 4 significant figures for a float, ``--`` for nothing."""
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def fmt_dev_pct(predicted: Any, actual: Any) -> str:
    """Signed deviation of a prediction from a measurement, as a table cell."""
    if predicted is None or actual in (None, 0):
        return "--"
    return f"{100.0 * (predicted - actual) / abs(actual):+.1f}%"
