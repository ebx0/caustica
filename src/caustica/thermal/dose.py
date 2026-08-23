"""CEM43 thermal dose (Sapareto-Dean) and the ITRUSST safety thresholds.

Ablation planning does not output pressure, it outputs DOSE. The accepted
currency is CEM43: the number of minutes at 43 C that would do the same
thermal damage as the temperature history the tissue actually saw
(Sapareto & Dean 1984)::

    t_CEM43 = integral R^(43 - T(t)) dt        [equivalent minutes at 43 C]
    R = 0.5   for T > 43 C
    R = 0.25  for T < 43 C

The two-valued ``R`` is the Arrhenius break at 43 C: above it, every extra
degree HALVES the time needed for the same damage; below it, every degree
lost buys a factor of four. The integral is in MINUTES — the unit is
"equivalent minutes", so a solve that steps in seconds must divide by 60,
which :func:`cem43_minutes` and the Pennes solver both do.

Two properties of this expression drive the API:

* **The integral is over the whole history, not the endpoint.** A voxel that
  spends 30 s at 50 C and then cools has a dose; the same voxel judged only
  by its final temperature has none. Dose must therefore be accumulated
  DURING the solve — which is why :class:`caustica.thermal.PennesSolver`
  integrates it step by step and this module only supplies the rate.
* **R^(43-T) overflows float32 above roughly T = 170 C** (2^127). A run that
  hot has left tissue physics behind long before it overflows; the solver's
  finite guard turns it into a refusal rather than an ``inf`` dose map.

Nothing here decides whether a dose is safe. The thresholds below are DATA;
rendering them next to a computed dose map, with a PASS/EXCEEDED next to
each one, is :mod:`caustica.thermal.report`'s job.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

#: The Arrhenius break temperature [C] of the Sapareto-Dean expression.
CEM43_REFERENCE_C = 43.0
#: R above the break: one degree hotter halves the equivalent time.
CEM43_R_ABOVE = 0.5
#: R below the break: one degree cooler quarters it.
CEM43_R_BELOW = 0.25

#: ITRUSST consensus thermal-safety limits, in CEM43 equivalent minutes, by
#: tissue. Source: ITRUSST consensus on biophysical safety for transcranial
#: ultrasound stimulation, Brain Stimulation 2025 — brain <= 2, bone <= 16,
#: skin <= 21 CEM43. These are the numbers the M18 report table will quote;
#: they are recorded here so exactly one copy of them exists in the library.
ITRUSST_CEM43_LIMITS: dict[str, float] = {"brain": 2.0, "bone": 16.0, "skin": 21.0}

#: The companion temperature-rise limit of the same consensus [C]: a
#: sonication that keeps dT <= 2 C is considered non-thermal.
ITRUSST_DELTA_T_LIMIT_C = 2.0

#: Provenance string for anything that prints the two constants above.
ITRUSST_SOURCE = (
    "ITRUSST consensus on biophysical safety for transcranial ultrasound "
    "stimulation, Brain Stimulation (2025): CEM43 <= 2 (brain), <= 16 (bone), "
    "<= 21 (skin); dT <= 2 C as the non-thermal criterion."
)

#: Research-use note that travels with every dose number (M18 criterion).
MEDICAL_DISCLAIMER = (
    "Research use only. caustica computes a modelled thermal dose from a "
    "modelled acoustic field; it is not a clinical decision tool and has no "
    "regulatory clearance."
)


def cem43_rate(temperature_c: Any, xp: ModuleType = np) -> Any:
    """``R^(43 - T)`` — equivalent minutes accrued per minute at ``T``.

    Backend-generic: pass ``backend.xp`` to evaluate on the GPU. The result
    keeps the dtype of ``temperature_c`` (float32 states stay float32).
    """
    t = xp.asarray(temperature_c)
    if t.dtype.kind != "f":
        # An integer temperature would make R^(43-T) an integer power and
        # floor the dose to 0 or overflow it; promote instead of surprising.
        t = t.astype(xp.float64)
    scalar = t.dtype.type
    # ``T == 43`` takes the below-break branch, where the exponent is 0 and
    # both branches give exactly 1.0 — the break value is not a discontinuity.
    r = xp.where(t > CEM43_REFERENCE_C, scalar(CEM43_R_ABOVE), scalar(CEM43_R_BELOW))
    return r ** (scalar(CEM43_REFERENCE_C) - t)


def cem43_minutes(temperature_c: Any, seconds: float, xp: ModuleType = np) -> Any:
    """Dose [CEM43 minutes] of a CONSTANT temperature held for ``seconds``.

    The closed form: 43 C for 60 s is exactly 1 equivalent minute, 44 C for
    60 s is 2, 42 C for 60 s is 0.25. Use it for hand checks and for the
    "what if it had held" comparison — never as a substitute for the
    accumulated dose of a transient, which is a different number.
    """
    if seconds < 0:
        raise ValueError(f"seconds must be >= 0, got {seconds}")
    return cem43_rate(temperature_c, xp) * (seconds / 60.0)
