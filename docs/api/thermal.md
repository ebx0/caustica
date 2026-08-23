# Thermal

An acoustic field is half of a HIFU calculation. The other half is what it does to tissue:
absorbed power becomes a volumetric heat source, Pennes bioheat carries it away, and CEM43
accumulates the dose.

!!! warning "Not a medical device"

    These numbers are a simulation of a physical model. They are not a treatment plan, and
    nothing in caustica is cleared for clinical use. The ITRUSST thresholds quoted by the
    report are literature reference values, not a safety certification.

## From field to heat

::: caustica.sensors.HeatingSource

## Bioheat

::: caustica.thermal.pennes.PennesSolver

::: caustica.thermal.pennes.ThermalResult

::: caustica.thermal.properties.ThermalMedium

## Dose

::: caustica.thermal.dose.cem43_minutes

::: caustica.thermal.dose.cem43_rate

## Reporting

::: caustica.thermal.report.write_thermal_report
