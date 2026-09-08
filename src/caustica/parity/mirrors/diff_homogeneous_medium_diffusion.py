"""Mirror of ``example_diff_homogeneous_medium_diffusion`` (Thermal Diffusion).

k-Wave's own page: ``kWaveDiffusion`` evolves a Gaussian initial temperature
distribution in a homogeneous 2-D medium over 300 steps of 0.5 s with no heat
source, and the answer is checked against ``bioheatExact``.

That last clause is why this is the first page of the whole gallery. The
example carries its own closed form, so the figure grades BOTH engines
instead of grading caustica against the incumbent (comparison contract, rule
6). It is also the page where the honest thing to report is a refusal:
k-wave-python ships ``kspaceFirstOrder`` and no binding for ``kWaveDiffusion``
at all, so k-Wave's own thermal engine cannot run here. The exact solution
takes the reference role, which is what the k-Wave script itself compares
against, and the k-Wave row states the refusal rather than disappearing.

What is quoted and what is reconstructed
----------------------------------------
The page publishes the medium (density 1079, conductivity 0.52, specific heat
3540), the initial condition (``37 + 5 exp(-(x/w)^2 - (y/w)^2)`` with
``w = 4 dx``) and the schedule (``Nt = 300``, ``dt = 0.5``). It does not
publish ``Nx``, ``Ny`` or ``dx``. Those are reconstructed and say so, through
:class:`~caustica.parity.model.Setting`, with the reasoning next to the
number.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from caustica.materials import Material
from caustica.parity.model import (
    EngineRun,
    ParityExample,
    Setting,
    absent,
    holds,
)
from caustica.parity.runner import ExtraEngine, ParityMirror
from caustica.thermal.pennes import PennesSolver
from caustica.thermal.properties import ThermalMedium

NAME = "example_diff_homogeneous_medium_diffusion"

#: Quoted from the k-Wave page.
DENSITY = 1079.0
THERMAL_CONDUCTIVITY = 0.52
SPECIFIC_HEAT = 3540.0
BASE_TEMPERATURE_C = 37.0
PEAK_RISE_K = 5.0
WIDTH_IN_DX = 4.0
N_STEPS = 300
DT_S = 0.5

#: Reconstructed; see :meth:`HomogeneousDiffusionMirror.settings`.
NX = 128
DX = 1.0e-3

#: Thermal diffusivity of the k-Wave medium [m^2/s].
DIFFUSIVITY = THERMAL_CONDUCTIVITY / (DENSITY * SPECIFIC_HEAT)


def _material() -> Material:
    return Material(
        name="k-Wave diffusion example medium",
        alpha_np_m=0.0,
        rho=DENSITY,
        c=1540.0,
        beta=0.0,
        thermal_conductivity=THERMAL_CONDUCTIVITY,
        specific_heat=SPECIFIC_HEAT,
        perfusion_rate=0.0,
        source="k-wave.org example_diff_homogeneous_medium_diffusion",
    )


def _coordinates() -> tuple[np.ndarray, np.ndarray]:
    """Centred voxel coordinates [m], matching k-Wave's ``kgrid.x``."""
    axis = (np.arange(NX) - (NX - 1) / 2.0) * DX
    return np.meshgrid(axis, axis, indexing="ij")


def _initial_temperature() -> np.ndarray:
    x, y = _coordinates()
    w = WIDTH_IN_DX * DX
    return (BASE_TEMPERATURE_C + PEAK_RISE_K * np.exp(-((x / w) ** 2) - (y / w) ** 2)).astype(
        np.float32
    )


def exact_rise(t_s: float) -> np.ndarray:
    """The closed form the k-Wave example grades itself against.

    A Gaussian stays a Gaussian under the heat equation: in ``d`` dimensions
    its variance grows as ``sigma^2(t) = sigma0^2 + 2 D t`` and its peak falls
    as ``(sigma0 / sigma)^d``, which conserves the deposited energy. k-Wave
    writes the initial condition as ``exp(-(x/w)^2)``, so ``sigma0 = w /
    sqrt(2)`` and the temperature RISE above 37 C is what this returns.
    """
    x, y = _coordinates()
    sigma0_sq = (WIDTH_IN_DX * DX) ** 2 / 2.0
    sigma_sq = sigma0_sq + 2.0 * DIFFUSIVITY * t_s
    return PEAK_RISE_K * (sigma0_sq / sigma_sq) * np.exp(-(x**2 + y**2) / (2.0 * sigma_sq))


class HomogeneousDiffusionMirror(ParityMirror):
    """The k-Wave heat-diffusion example, run against its own closed form."""

    name = NAME

    def __init__(self, backend: str = "numpy") -> None:
        self.backend = backend

    # ---------- the example ----------

    def example(self) -> ParityExample:
        return ParityExample(
            name=NAME,
            title="Heat Diffusion In A Homogeneous Medium",
            category="Thermal Diffusion",
            demonstrates=(
                "kWaveDiffusion evolves a Gaussian initial temperature distribution in a "
                "homogeneous 2-D medium over 300 steps with no heat source, and the "
                "result is checked against the exact Pennes solution from bioheatExact."
            ),
            comparison_axis=(
                "the temperature map after diffusion and a profile through the centre, "
                "for caustica, for k-Wave and for the exact solution; peak temperature "
                "and the width of the Gaussian are the numbers"
            ),
            source_url=(
                "http://www.k-wave.org/documentation/example_diff_homogeneous_medium_diffusion.php"
            ),
            faithful_mirror=True,
            mirror_note=(
                "Faithful and free: it needs no new code and it carries its own closed "
                "form, so the figure grades both engines rather than grading caustica "
                "against the incumbent."
            ),
            graded_quantity="temperature rise above 37 C at t = 150 s",
            graded_units="K",
            traits={
                "field": holds(),
                "peak": holds(),
                "focused_beam": absent(
                    "the graded quantity is a diffusing hot spot, not a beam: it has no "
                    "aperture, no focus and no depth of field, so a -6 dB focal volume "
                    "would measure the initial condition's own width and call it a focus."
                ),
                "source_surface": absent(
                    "this example has no acoustic source at all; the run starts from an "
                    "initial temperature distribution and no energy enters afterwards."
                ),
                "beam_axis": holds(),
                "sidelobe": absent(
                    "a diffusing Gaussian falls monotonically from the centre in every "
                    "direction, so there is no sidelobe for either engine to resolve."
                ),
                "nonlinear": absent(
                    "there is no acoustic field here, and the heat equation is linear in "
                    "temperature, so no harmonic of a drive exists to compare."
                ),
                "third_harmonic": absent(
                    "there is no acoustic drive, so there is no fundamental for a third "
                    "harmonic to be measured against."
                ),
                "time_series": holds(),
                "shock": absent(
                    "diffusion smooths gradients and cannot steepen a front, so no "
                    "rise time is a property of this example."
                ),
                "closed_surface": absent(
                    "no acoustic power flows in this example, so a radiated power "
                    "through a closed surface is zero by construction rather than a "
                    "measurement of anything."
                ),
                "integrable_loss": absent(
                    "the loss term of an acoustic energy balance does not exist here; "
                    "the only transport is conduction, whose balance is the equation "
                    "itself."
                ),
                "absorbing": absent(
                    "there is no acoustic field for the medium to absorb; the initial "
                    "energy is already thermal."
                ),
                "varies_resolution": absent(
                    "the k-Wave example runs one grid and one time step; it does not "
                    "sweep resolution, so there is no sequence to fit an order to."
                ),
            },
        )

    def settings(self) -> Mapping[str, Setting]:
        return {
            "density_kg_m3": Setting(DENSITY, "k-Wave example page"),
            "thermal_conductivity_w_m_k": Setting(THERMAL_CONDUCTIVITY, "k-Wave example page"),
            "specific_heat_j_kg_k": Setting(SPECIFIC_HEAT, "k-Wave example page"),
            "initial_condition": Setting(
                "37 + 5 exp(-(x/w)^2 - (y/w)^2), w = 4 dx", "k-Wave example page"
            ),
            "n_steps": Setting(N_STEPS, "k-Wave example page"),
            "dt_s": Setting(DT_S, "k-Wave example page"),
            "grid_shape": Setting(
                [NX, NX],
                "inferred",
                "the page prints the medium, the initial condition and the schedule but "
                "not Nx or Ny. 128 by 128 is k-Wave's usual kWaveDiffusion grid and it "
                "is nine final standard deviations wide, so the insulated walls change "
                "nothing the comparison reads.",
            ),
            "dx_m": Setting(
                DX,
                "inferred",
                "the page does not print dx. At 1 mm the initial Gaussian (sigma = 2.83 "
                "voxels) is resolved and the spot spreads to sigma = 7.0 mm by 150 s, "
                "which is the spreading the example's figure shows.",
            ),
            "perfusion_rate_per_s": Setting(
                0.0,
                "caustica",
                "the k-Wave example passes [D, 0, 0] to bioheatExact, so its perfusion "
                "and its heat source are both zero; caustica states the zero rather "
                "than leaving the field undeclared.",
            ),
            "sound_speed_m_s": Setting(
                1540.0,
                "caustica",
                "caustica's Material carries acoustic fields whether or not a run reads "
                "them. The thermal medium reads density, conductivity, specific heat and "
                "perfusion only, so this number touches nothing on this page.",
            ),
            "boundary": Setting(
                "insulated",
                "caustica",
                "kWaveDiffusion's Fourier step is periodic; caustica's conservative "
                "stencil offers insulated or Dirichlet walls. At nine standard "
                "deviations from the spot neither choice moves the graded field.",
            ),
        }

    # ---------- the engines ----------

    def reference_engine(self) -> str:
        return "pennes-exact"

    def no_scene_reason(self) -> str:
        return (
            "this example has no acoustic scene: kWaveDiffusion evolves an initial "
            "temperature field with no source, no drive frequency and no wave. A wave "
            "solver has nothing to integrate here, which is a statement about the "
            "example and not a gap in the solver."
        )

    def extra_engines(self) -> tuple[ExtraEngine, ...]:
        return (
            ExtraEngine("pennes", "solver", "native"),
            ExtraEngine("pennes-exact", "closed form", "reference"),
            ExtraEngine(
                "kwave-diffusion",
                "solver",
                "native",
                refusal=(
                    "k-wave-python ships kspaceFirstOrder (2-D, 3-D and axisymmetric) "
                    "and no binding for kWaveDiffusion, so k-Wave's own thermal engine "
                    "cannot be driven from this harness. The exact solution takes the "
                    "reference role instead, which is what the k-Wave script itself "
                    "grades against."
                ),
            ),
        )

    def run_engine(self, name: str) -> EngineRun:
        if name == "pennes":
            return self._run_pennes()
        if name == "pennes-exact":
            return self._run_exact()
        raise KeyError(f"{NAME} has no engine '{name}'")

    def _thermal_medium(self) -> ThermalMedium:
        return ThermalMedium.homogeneous((NX, NX), _material(), DX)

    def _run_pennes(self) -> EngineRun:
        medium = self._thermal_medium()
        solver = PennesSolver(backend=self.backend, on_unstable="substep")
        result = solver.solve(_initial_temperature(), None, medium, dt=DT_S, n_steps=N_STEPS)
        rise = np.asarray(result.temperature, dtype=np.float64) - BASE_TEMPERATURE_C
        return EngineRun(
            engine="pennes",
            kind="solver",
            role="native",
            field=rise,
            dx=DX,
            beam_axis=0,
            scalars={
                "peak_rise_k": float(rise.max()),
                "t_end_s": float(result.t_end_s),
            },
            notes=(
                "the k-Wave example's dt of 0.5 s is above this scheme's explicit "
                f"stability bound of {solver.stable_dt(medium):.3g} s on a 1 mm grid, so "
                f"the solve sub-steps {result.substeps} times per outer step and lands "
                "on the example's own sampling instants.",
            ),
            provenance={
                "dt_s": DT_S,
                "steps_total": int(result.n_steps * result.substeps),
                "substeps": int(result.substeps),
                "scheme": result.meta.get("scheme"),
                "backend": result.meta.get("backend"),
            },
        )

    def _run_exact(self) -> EngineRun:
        t_end = N_STEPS * DT_S
        rise = exact_rise(t_end)
        return EngineRun(
            engine="pennes-exact",
            kind="closed form",
            role="reference",
            field=rise,
            dx=DX,
            beam_axis=0,
            scalars={"peak_rise_k": float(rise.max()), "t_end_s": float(t_end)},
            notes=(
                "the same closed form the k-Wave script calls bioheatExact with: a "
                "Gaussian stays Gaussian and its variance grows as sigma0^2 + 2 D t.",
            ),
            provenance={"dt_s": None, "steps_total": 0},
        )

    def notes(self) -> tuple[str, ...]:
        return (
            "The reference on this page is the closed form, not k-Wave. k-wave-python "
            "has no kWaveDiffusion binding, and the k-Wave example grades itself against "
            "bioheatExact anyway, so the exact solution is the honest third line the "
            "comparison contract asks for.",
            "'Axial' here means the profile along grid axis 0 through the peak. The "
            "spot is isotropic, so that cut is the k-Wave figure's own centre profile "
            "and not a beam axis; the metric registry has one profile trait and this is "
            "what it means on a diffusion page.",
            "The thermal solver records no point history in the committed tree, so the "
            "waveform metrics report 'not computed' rather than 'not applicable': a "
            "heating curve is meaningful here, this run simply did not record one.",
        )
