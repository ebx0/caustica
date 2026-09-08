"""Mirror of ``example_diff_focused_ultrasound_heating`` (Thermal Diffusion).

k-Wave's own page: run an acoustic simulation of a focused transducer, take
the steady-state pressure amplitude, turn it into a volume rate of heat
deposition with ``Q = alpha_np p^2 / (rho c)``, then run 10 seconds of heating
and 20 seconds of cooling and plot pressure, heat deposition, temperature,
thermal dose and the ablated volume. The page publishes the drive (1 MHz, 500
kPa at the surface), the CFL (0.3, which is where the parity contract's own
matched setting comes from), the ``Q`` formula and the two phase durations.

This is caustica's own subject written as a k-Wave example, and three things
about the comparison have to be said out loud on the page rather than
discovered by a reader.

**One thermal solver, two acoustic fields.** k-wave-python ships no
``kWaveDiffusion`` binding, so the heating half cannot be run twice. Each
engine's own pressure field is turned into ``Q`` and integrated by caustica's
Pennes solver, and the graded temperature therefore measures the ACOUSTIC
difference carried through one thermal scheme. It is not two thermal solvers
being compared, and the page must not be read as if it were.

**``Q`` uses the absorption at the fundamental only**, which is not yet
consistent with the absorption the field saw, and a later task in the thermal
phase makes the two consistent. Here the medium is linear and only the
fundamental exists, so the gap does not bite on this page; the note stays
because the formula is the same one that will be wrong on a nonlinear page.

**The ``Q`` formula assumes plane-wave intensity**, an approximation both
engines share. Agreement on ``Q`` is therefore agreement about the same
approximation, not evidence that the approximation is right.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

import caustica.solvers as solvers
from caustica.core.grid import Grid
from caustica.core.pml import PMLSpec
from caustica.materials import Material, db_cm_to_np_m
from caustica.medium import Medium
from caustica.parity.model import EngineRun, ParityExample, Setting, absent, holds
from caustica.parity.runner import AcousticScene, ParityMirror, parity_run_spec
from caustica.parity.scenes import arc_cw_source
from caustica.sensors import HeatingSource
from caustica.solvers.base import interior_slices
from caustica.thermal.pennes import ARTERIAL_TEMPERATURE_C, PennesSolver, ThermalResult
from caustica.thermal.properties import ThermalMedium

NAME = "example_diff_focused_ultrasound_heating"

#: Quoted from the k-Wave page.
FREQ_HZ = 1.0e6
SURFACE_PRESSURE_PA = 500.0e3
HEATING_S = 10.0
COOLING_S = 20.0

#: Reconstructed; see :meth:`FocusedUltrasoundHeatingMirror.settings`.
NX = 256
NY = 256
DX = 0.2e-3
PML_VOX = 20
ROC_VOX = 150.0
APERTURE_VOX = 150.0
APEX_AXIAL_VOX = 30
SOUND_SPEED = 1540.0
DENSITY = 1079.0
ALPHA_DB_CM_AT_F0 = 0.75
THERMAL_CONDUCTIVITY = 0.52
SPECIFIC_HEAT = 3540.0

#: The ablation threshold every clinical CEM43 curve is drawn at
#: [equivalent minutes at 43 C].
CEM43_ABLATION_THRESHOLD = 240.0


def _material() -> Material:
    return Material(
        name="soft tissue at 1 MHz",
        alpha_np_m=db_cm_to_np_m(ALPHA_DB_CM_AT_F0),
        rho=DENSITY,
        c=SOUND_SPEED,
        beta=0.0,
        thermal_conductivity=THERMAL_CONDUCTIVITY,
        specific_heat=SPECIFIC_HEAT,
        perfusion_rate=0.0,
        source="k-wave.org example_diff_focused_ultrasound_heating (reconstructed)",
    )


class FocusedUltrasoundHeatingMirror(ParityMirror):
    """Pressure, Q, temperature, dose and lesion, from three acoustic fields."""

    name = NAME

    def __init__(self, backend: str = "numpy") -> None:
        self.backend = backend
        self._scene: AcousticScene | None = None

    # ---------- the example ----------

    def example(self) -> ParityExample:
        return ParityExample(
            name=NAME,
            title="Heating By A Focused Ultrasound Transducer",
            category="Thermal Diffusion",
            demonstrates=(
                "Runs an acoustic simulation of a focused transducer, extracts the "
                "steady-state pressure amplitude, converts it to a volume rate of heat "
                "deposition with Q = alpha p^2 / (c0 rho0), then runs 10 seconds of "
                "heating and 20 seconds of cooling and plots pressure, heat deposition, "
                "temperature, thermal dose and the ablated volume."
            ),
            comparison_axis=(
                "the whole chain: pressure field, Q map, temperature map, CEM43 dose and "
                "ablated volume, caustica against k-Wave on the same grid; peak "
                "temperature rise and lesion volume are the two headline numbers"
            ),
            source_url=(
                "http://www.k-wave.org/documentation/example_diff_focused_ultrasound_heating.php"
            ),
            faithful_mirror=True,
            mirror_note=(
                "The flagship page: the library's own subject matter, and it runs today. "
                "Two honesty notes belong on it. Q uses the absorption at the "
                "fundamental only, which is not yet consistent with the absorption the "
                "field saw. And the Q formula assumes plane-wave intensity, an "
                "approximation both engines share, so agreement on Q is agreement about "
                "the same approximation."
            ),
            graded_quantity="peak temperature rise above 37 C over the whole exposure",
            graded_units="K",
            traits={
                "field": holds(),
                "peak": holds(),
                "focused_beam": holds(),
                "source_surface": absent(
                    "the graded quantity of this page is a temperature rise, and a focal "
                    "gain is a focal PRESSURE over the source-surface pressure. Kelvin "
                    "divided by pascal is not a gain, and the page carries the surface "
                    "pressure as a setting instead."
                ),
                "beam_axis": holds(),
                "sidelobe": absent(
                    "ten seconds of conduction smooths the deposition over several "
                    "millimetres, so the temperature map carries no resolved sidelobe. "
                    "The pressure field has them; the pressure field is not what this "
                    "page grades."
                ),
                "nonlinear": absent(
                    "the k-Wave example runs a linear acoustic simulation (it sets no "
                    "B/A), so the mirror sets beta = 0 and no harmonic is generated for "
                    "either engine to compare."
                ),
                "third_harmonic": absent(
                    "the acoustic run is linear, so there is no second harmonic and no third."
                ),
                "time_series": holds(),
                "shock": absent(
                    "a linear acoustic run cannot steepen a waveform, and a temperature "
                    "history rises on the scale of seconds; neither carries a shock."
                ),
                "closed_surface": holds(),
                "integrable_loss": holds(),
                "absorbing": holds(),
                "varies_resolution": absent(
                    "the k-Wave example runs one grid; it does not sweep resolution, so "
                    "there is no sequence to fit a convergence order to."
                ),
            },
        )

    def settings(self) -> Mapping[str, Setting]:
        return {
            "source_freq_hz": Setting(FREQ_HZ, "k-Wave example page"),
            "surface_pressure_pa": Setting(SURFACE_PRESSURE_PA, "k-Wave example page"),
            "cfl": Setting(
                0.3,
                "k-Wave example page",
                "this example states cfl = 0.3 in its own script, which is where the "
                "parity contract's matched setting comes from; caustica's default is "
                "0.48.",
            ),
            "heating_s": Setting(HEATING_S, "k-Wave example page"),
            "cooling_s": Setting(COOLING_S, "k-Wave example page"),
            "q_formula": Setting("Q = alpha_np p^2 / (rho c)", "k-Wave example page"),
            "dimensionality": Setting(
                "2-D",
                "caustica",
                "the page does not say whether its grid is 2-D or 3-D. A 2-D mirror is "
                "what the laptop tier can hold, and both engines run the same 2-D scene, "
                "so the comparison is unaffected. The absolute focal gain of a 2-D arc is "
                "not the absolute gain of a 3-D bowl, so no number here is a claim about "
                "the k-Wave example's own field.",
            ),
            "grid_shape_vox": Setting(
                [NX, NY],
                "inferred",
                "the page does not print the grid. 256 by 256 at 0.2 mm holds a 30 mm "
                "focal length, a 30 mm aperture, 15 mm beyond the focus and a 20-voxel "
                "PML on every face with the arc clear of it.",
            ),
            "dx_m": Setting(
                DX,
                "inferred",
                "the page does not print dx. 0.2 mm is 7.7 points per wavelength at 1 "
                "MHz in soft tissue, and it makes the transducer's radius and aperture "
                "whole numbers of voxels.",
            ),
            "roc_m": Setting(
                ROC_VOX * DX,
                "inferred",
                "the page does not print the transducer. A 30 mm radius of curvature "
                "with a 30 mm aperture is f/1, the geometry a focused-heating example "
                "uses to put a lesion at a stated depth.",
            ),
            "aperture_m": Setting(
                APERTURE_VOX * DX, "inferred", "as for the radius of curvature: f/1."
            ),
            "sound_speed_m_s": Setting(
                SOUND_SPEED, "inferred", "soft tissue; the page does not print the medium."
            ),
            "density_kg_m3": Setting(
                DENSITY,
                "inferred",
                "the density k-Wave's own kWaveDiffusion example uses, so the two thermal "
                "pages in this category describe the same tissue.",
            ),
            "alpha_db_cm_at_1mhz": Setting(
                ALPHA_DB_CM_AT_F0,
                "inferred",
                "0.75 dB/cm at 1 MHz is the soft-tissue value k-Wave's examples use. The "
                "run is single frequency, so the power-law exponent never enters and both "
                "engines are given the same frequency-independent number.",
            ),
            "thermal_conductivity_w_m_k": Setting(
                THERMAL_CONDUCTIVITY, "inferred", "as for the density: the same tissue."
            ),
            "specific_heat_j_kg_k": Setting(
                SPECIFIC_HEAT, "inferred", "as for the density: the same tissue."
            ),
            "perfusion_rate_per_s": Setting(
                0.0,
                "caustica",
                "the page's ablated-volume figure is drawn without perfusion; caustica "
                "states the zero rather than leaving the field undeclared.",
            ),
            "cem43_ablation_threshold_min": Setting(
                CEM43_ABLATION_THRESHOLD,
                "caustica",
                "the page plots an ablated volume without printing its threshold; 240 "
                "equivalent minutes at 43 C is the clinical convention and it is stated "
                "here so the lesion number can be reproduced.",
            ),
            "record_region": Setting(
                f"interior, {PML_VOX + 2} voxels shaved from every face",
                "caustica",
                "inside a PML the field is a sponge artefact, and turning it into Q would "
                "paint a ring of fictitious heating around the domain. Both engines are "
                "graded on the same interior.",
            ),
        }

    # ---------- the scene ----------

    def scene(self) -> AcousticScene:
        if self._scene is None:
            grid = Grid((NX, NY), DX, pml=PMLSpec(thickness=PML_VOX * DX))
            source = arc_cw_source(
                grid,
                f0=FREQ_HZ,
                amplitude=SURFACE_PRESSURE_PA,
                apex_vox=(float(APEX_AXIAL_VOX), (NY - 1) / 2.0),
                roc_vox=ROC_VOX,
                aperture_vox=APERTURE_VOX,
                axis=(1.0, 0.0),
                label="focused arc, f/1",
            )
            self._scene = AcousticScene(
                grid=grid,
                medium=Medium.homogeneous(grid.shape, _material()),
                source=source,
                spec=parity_run_spec(min_settle_periods=8, max_settle_periods=64),
                record_region=interior_slices(grid.shape, PML_VOX + 2),
            )
        return self._scene

    # ---------- the engines ----------

    def reference_engine(self) -> str:
        return "kwave"

    def run_engine(self, name: str) -> EngineRun:
        scene = self.scene()
        solver = solvers.get(name)()
        kwargs: dict[str, object] = {"record_region": scene.record_region}
        if name != "kwave":
            kwargs["backend"] = self.backend
        acoustic = solver.run(scene.grid, scene.medium, scene.source, scene.spec, **kwargs)

        heat = HeatingSource.from_result(acoustic, scene.medium, scene.grid.dx, harmonics=(1,))
        thermal_medium = ThermalMedium.homogeneous(heat.shape, _material(), DX)
        pennes = PennesSolver(backend=self.backend)
        dt = 0.9 * pennes.stable_dt(thermal_medium)
        t0 = np.full(thermal_medium.shape, ARTERIAL_TEMPERATURE_C, dtype=np.float32)
        hot = pennes.solve(t0, heat, thermal_medium, dt=dt, n_steps=int(HEATING_S / dt), dose=True)
        cool = pennes.solve(
            hot.temperature,
            None,
            thermal_medium,
            dt=dt,
            n_steps=int(COOLING_S / dt),
            dose=True,
            dose0=hot.dose_cem43,
        )
        exposure = ThermalResult.chain([hot, cool])

        rise = np.asarray(exposure.temperature_max, dtype=np.float64) - ARTERIAL_TEMPERATURE_C
        dose = np.asarray(exposure.dose_cem43, dtype=np.float64)
        ablated_mm2 = float((dose >= CEM43_ABLATION_THRESHOLD).sum()) * (DX * 1e3) ** 2
        amp = np.asarray(acoustic.amp, dtype=np.float64)
        return EngineRun(
            engine=name,
            kind="solver",
            role="reference" if name == self.reference_engine() else "native",
            field=rise,
            dx=DX,
            beam_axis=0,
            scalars={
                "peak_pressure_pa": float(amp.max()),
                "focal_gain": float(amp.max() / SURFACE_PRESSURE_PA),
                "peak_q_w_m3": float(heat.q_max),
                "peak_rise_k": float(rise.max()),
                "peak_cem43_min": float(dose.max()),
                "ablated_area_mm2": ablated_mm2,
            },
            source_surface_pa=SURFACE_PRESSURE_PA,
            absorbed_power_w=float(heat.total_power_w),
            provenance={
                "spp": int(acoustic.spp),
                "dt_s": float(acoustic.dt),
                "cfl_realized": float(acoustic.dt * scene.medium.c_max / DX),
                "steps_total": int(acoustic.steps_total),
                # What the run reported, not what it was asked for.
                "solver_backend": acoustic.meta.get("backend"),
                "thermal_backend": self.backend,
                "thermal_dt_s": float(dt),
                "thermal_steps": int(exposure.n_steps * exposure.substeps),
                "alpha_model": heat.alpha_model,
            },
            notes=(
                f"Q was built from the fundamental only ({heat.alpha_model}); the run is "
                f"linear, so no harmonic exists that the absorption model could get "
                f"wrong here.",
            ),
        )

    def notes(self) -> tuple[str, ...]:
        return (
            "The thermal half is caustica's Pennes solver for every engine, because "
            "k-wave-python ships no kWaveDiffusion binding. The graded temperature "
            "therefore measures the acoustic difference carried through one thermal "
            "scheme; it is not a comparison of two thermal solvers, and the page must "
            "not be read as one.",
            "Q uses the absorption at the fundamental only, which is not yet the "
            "absorption the field saw. The acoustic run here is linear, so only the "
            "fundamental exists and the gap does not bite on this page.",
            "The Q formula assumes plane-wave intensity. Both engines make the same "
            "assumption, so agreement on Q is agreement about the same approximation "
            "rather than evidence that the approximation holds at a focus.",
            "M-27's absorbed power is integrated over a 2-D plane, so it is a power per "
            "metre of the unmodelled third dimension. The metric is a ratio, so that "
            "convention cancels.",
            "The dose and lesion numbers travel as scalars, not as metrics: the registry "
            "M-01 to M-30 is acoustic and defines no thermal group, so CEM43 and the "
            "ablated area appear in each engine's block rather than in the metric table.",
        )
