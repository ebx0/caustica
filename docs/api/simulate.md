# Running a simulation

## The one call

::: caustica.facade.simulate

::: caustica.facade.SimulationRun

::: caustica.facade.SimulationError

## The solver registry

Solvers are looked up by name, not imported by path — which is what lets a third-party
package add one without touching caustica. See [extending](../extending.md).

::: caustica.solvers.registry.get

::: caustica.solvers.registry.available

::: caustica.solvers.registry.register

::: caustica.solvers.base.SolverBase

::: caustica.solvers.base.SolverCaps

::: caustica.solvers.base.CWRunSpec

::: caustica.solvers.base.SolverResult

::: caustica.solvers.base.interior_slices

## Parameter studies

::: caustica.study.core.Study
