"""``caustica.thermal`` — Pennes bioheat + CEM43 dose.

The output of ablation planning is not pressure, it is DOSE. This package is
the second half of that chain; the first half is
:class:`caustica.sensors.HeatingSource`, which turns a converged acoustic
result into volumetric heating ``Q`` [W/m^3]::

    from caustica.sensors import HeatingSource
    from caustica.thermal import PennesSolver, ThermalMedium

    heat = HeatingSource.from_result(result, medium, grid.dx, alpha_power_law_y=1.1)
    tmed = ThermalMedium.from_medium(medium, db, grid.dx)
    T0 = np.full(tmed.shape, 37.0, dtype=np.float32)

    hot = PennesSolver().solve(T0, heat, tmed, dt=0.05, n_steps=1200, dose=True)
    hot.peak_temperature_c, hot.peak_dose_cem43

and then the cooling phase, with the dose carried over::

    cool = PennesSolver().solve(hot.temperature, None, tmed, dt=0.05, n_steps=3600,
                                dose=True, dose0=hot.dose_cem43)

and finally the stamped dose/threshold summary of what that history did::

    from caustica.thermal import write_thermal_report

    write_thermal_report(cool, tmed, "benchmarks/reports/thermal/run", label="60 s sonication")

``dt`` is bounded by the explicit scheme's stability limit for the medium —
ask for it with ``PennesSolver().stable_dt(tmed)`` rather than guessing; a
step above it is refused, with the number, not silently integrated.

:mod:`caustica.thermal.dose` holds the CEM43 definition and the ITRUSST
safety thresholds as data; reading a PASS/EXCEEDED verdict off them is
:mod:`caustica.thermal.report`'s job. Every number here is research-only,
and the report says so in both the files it writes: see
``caustica.thermal.report.MEDICAL_LIABILITY_NOTE``.

Import discipline (same rule as :mod:`caustica.study` and
:mod:`caustica.validation`): the names below are PEP 562 lazy. This package
needs nothing heavier than numpy today, and the laziness is not about
weight — it keeps ONE rule at the package door (value types eager,
everything composed lazy) so a future dependency in the thermal chain
cannot quietly land in ``import caustica``.
"""

from __future__ import annotations

__all__ = [
    "ARTERIAL_TEMPERATURE_C",
    "BLOOD_DENSITY",
    "BLOOD_SPECIFIC_HEAT",
    "ITRUSST_CEM43_LIMITS",
    "ITRUSST_DELTA_T_LIMIT_C",
    "ITRUSST_SOURCE",
    "MEDICAL_DISCLAIMER",
    "MEDICAL_LIABILITY_NOTE",
    "PennesSolver",
    "ThermalDivergedError",
    "ThermalMedium",
    "ThermalPropertyError",
    "ThermalResult",
    "ThermalStabilityError",
    "cem43_minutes",
    "cem43_rate",
    "dose",
    "pennes",
    "properties",
    "report",
    "thermal_payload",
    "write_thermal_report",
]

_LAZY = {
    "ARTERIAL_TEMPERATURE_C": "caustica.thermal.pennes",
    "BLOOD_DENSITY": "caustica.thermal.pennes",
    "BLOOD_SPECIFIC_HEAT": "caustica.thermal.pennes",
    "ITRUSST_CEM43_LIMITS": "caustica.thermal.dose",
    "ITRUSST_DELTA_T_LIMIT_C": "caustica.thermal.dose",
    "ITRUSST_SOURCE": "caustica.thermal.dose",
    "MEDICAL_DISCLAIMER": "caustica.thermal.dose",
    "MEDICAL_LIABILITY_NOTE": "caustica.thermal.report",
    "PennesSolver": "caustica.thermal.pennes",
    "ThermalDivergedError": "caustica.thermal.pennes",
    "ThermalMedium": "caustica.thermal.properties",
    "ThermalPropertyError": "caustica.thermal.properties",
    "ThermalResult": "caustica.thermal.pennes",
    "ThermalStabilityError": "caustica.thermal.pennes",
    "cem43_minutes": "caustica.thermal.dose",
    "cem43_rate": "caustica.thermal.dose",
    "thermal_payload": "caustica.thermal.report",
    "write_thermal_report": "caustica.thermal.report",
}

#: Submodules reachable as attributes (``caustica.thermal.dose``) without an
#: explicit import, kept lazy for the same reason as the names above.
_SUBMODULES = ("dose", "pennes", "properties", "report")


def __getattr__(name: str):
    import importlib

    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    if name in _SUBMODULES:
        return importlib.import_module(f"caustica.thermal.{name}")
    raise AttributeError(f"module 'caustica.thermal' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
