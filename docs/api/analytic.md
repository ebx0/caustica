# Analytic references

The ground-truth layer. These are not helpers around the solver — they are independent
closed-form solutions, and the test suite gates every solver milestone against them. When a
figure on this site says "against O'Neil", this is the O'Neil it means.

They share the library's phasor convention, `p(t) = Re{P·e^(−iωt)}`, so a phase from
`rayleigh_pressure` and a phase from a solve are directly comparable.

## Focused sources

::: caustica.analytic.oneill.axial_pressure

::: caustica.analytic.oneill.focal_gain

::: caustica.analytic.rayleigh.rayleigh_pressure

::: caustica.analytic.geometry.spherical_cap_points

## Nonlinearity

::: caustica.analytic.planewave.fubini_harmonic

::: caustica.analytic.planewave.shock_distance

## Absorption

::: caustica.analytic.planewave.attenuate
