"""Mirror of ``example_tvsp_transducer_field_patterns`` (Time Varying Source).

k-Wave's own page: a curved single-element arc transducer, driven with a
sinusoid in a homogeneous non-absorbing 2-D medium, with the whole domain's
final, maximum and RMS pressure recorded to show the focus and the side
lobes. The page publishes the arc verbatim::

    arc_pos = [20, 20];  radius = 60;  diameter = 81;
    focus_pos = [Nx/2, Nx/2];
    source_freq = 0.25e6;  source_mag = 0.5;
    sensor.mask = [1, 1, Nx, Ny].';
    sensor.record = {'p_final', 'p_max', 'p_rms'};

and publishes neither the grid nor the medium. Both are reconstructed here,
each with the reasoning beside the number.

Three mirroring decisions the page forces
----------------------------------------
**The domain is padded.** With ``arc_pos = [20, 20]`` on a 128 grid the arc's
ends land about two and a half voxels from two faces, which is inside
k-Wave's own 20-voxel PML; caustica refuses a source buried in the sponge
because such a run converges quietly on a wrong field. The scene is therefore
shifted 32 voxels away from both faces and the domain grows to match. Both
engines run the padded domain, so nothing about the comparison changes.

**One source, two propagators.** The arc is deposited band-limited once and
handed to every engine, the k-Wave binary included, which takes it as
per-voxel source weights on its own mask. Nothing here measures the
difference between a staircased ``makeArc`` shell and a band-limited deposit;
what it measures is two propagators driven identically.

**The beam runs diagonally, so the profile metrics do not apply.** The arc
faces from ``[20, 20]`` towards the grid centre, at 45 degrees to both axes.
The registry's profile group takes a beam axis as a grid axis, so on this
page M-10, M-11, M-13, M-14, M-15 and M-16 are "not applicable" with that
reason rather than sampling a line the beam does not lie on. It is the
example's own geometry that says so, not a limitation being hidden: the -6 dB
focal volume (M-09), which needs no axis, still reports.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

import caustica.solvers as solvers
from caustica.core.grid import Grid
from caustica.core.pml import PMLSpec
from caustica.materials import Material
from caustica.medium import Medium
from caustica.parity.model import EngineRun, ParityExample, Setting, absent, holds
from caustica.parity.runner import AcousticScene, ExtraEngine, ParityMirror, parity_run_spec
from caustica.parity.scenes import arc_cw_source
from caustica.solvers.base import interior_slices

NAME = "example_tvsp_transducer_field_patterns"

#: Quoted from the k-Wave page, in grid points.
ARC_POS = (20, 20)
ARC_RADIUS_VOX = 60.0
ARC_DIAMETER_VOX = 81.0
SOURCE_FREQ_HZ = 0.25e6
SOURCE_MAG_PA = 0.5

#: Reconstructed; see :meth:`TransducerFieldPatternsMirror.settings`.
NX_PAGE = 128
PAD_VOX = 32
NX = NX_PAGE + 2 * PAD_VOX
DX = 0.5e-3
PML_VOX = 20
SOUND_SPEED = 1500.0
DENSITY = 1000.0


def _apex_vox() -> tuple[float, float]:
    return (ARC_POS[0] + PAD_VOX, ARC_POS[1] + PAD_VOX)


def _axis() -> tuple[float, float]:
    """Unit vector from the arc's midpoint towards ``focus_pos``."""
    focus = np.array([NX_PAGE / 2.0 + PAD_VOX, NX_PAGE / 2.0 + PAD_VOX])
    direction = focus - np.array(_apex_vox())
    return tuple(direction / np.linalg.norm(direction))


def _material() -> Material:
    return Material(
        name="water, lossless",
        alpha_np_m=0.0,
        rho=DENSITY,
        c=SOUND_SPEED,
        beta=0.0,
        source="k-wave.org example_tvsp_transducer_field_patterns (reconstructed)",
    )


class TransducerFieldPatternsMirror(ParityMirror):
    """A driven arc transducer, caustica's steady state against k-Wave's."""

    name = NAME

    def __init__(self, backend: str = "numpy") -> None:
        self.backend = backend
        self._scene: AcousticScene | None = None

    # ---------- the example ----------

    def example(self) -> ParityExample:
        return ParityExample(
            name=NAME,
            title="Simulating Transducer Field Patterns",
            category="Time Varying Source Problems",
            demonstrates=(
                "A curved single-element arc transducer is driven with a sinusoid in a "
                "homogeneous non-absorbing 2-D medium, and the whole-domain final, "
                "maximum and RMS pressure are recorded to show the focus and the side "
                "lobes."
            ),
            comparison_axis=(
                "the maximum and RMS pressure maps and the axial and focal-plane "
                "profiles, caustica against k-Wave on the same grid; focal peak ratio "
                "and minus 6 dB widths"
            ),
            source_url=(
                "http://www.k-wave.org/documentation/example_tvsp_transducer_field_patterns.php"
            ),
            faithful_mirror=True,
            mirror_note=(
                "Faithful. The one page in this category the tree can publish before any "
                "transient source work lands, because a driven arc, a steady-state solve "
                "and a whole-grid p_max map are all in the library already."
            ),
            graded_quantity="maximum pressure over the record window",
            graded_units="Pa",
            traits={
                "field": holds(),
                "peak": holds(),
                "focused_beam": holds(),
                "source_surface": holds(),
                "beam_axis": absent(
                    "the arc faces from [20, 20] towards the grid centre, so its axis "
                    "runs at 45 degrees to both grid axes. A profile along a grid axis "
                    "through the peak would sample a line the beam does not lie on, and "
                    "the registry's profile group takes the beam axis as a grid axis."
                ),
                "sidelobe": holds(),
                "nonlinear": absent(
                    "the k-Wave example sets no B/A, so both engines integrate a linear "
                    "medium and no harmonic of the drive is generated to compare."
                ),
                "third_harmonic": absent(
                    "the medium is linear, so there is no second harmonic and no third."
                ),
                "time_series": holds(),
                "shock": absent(
                    "a linear lossless medium at 0.5 Pa cannot steepen a waveform; the "
                    "example drives half a pascal precisely to stay linear."
                ),
                "closed_surface": holds(),
                "integrable_loss": holds(),
                "absorbing": absent(
                    "the medium is non-absorbing by the example's own design (alpha = "
                    "0), so there is no absorbed power to integrate."
                ),
                "varies_resolution": absent(
                    "the k-Wave example runs one grid; it does not sweep resolution, so "
                    "there is no sequence to fit a convergence order to."
                ),
            },
        )

    def settings(self) -> Mapping[str, Setting]:
        return {
            "arc_pos_vox": Setting(list(ARC_POS), "k-Wave example page"),
            "arc_radius_vox": Setting(ARC_RADIUS_VOX, "k-Wave example page"),
            "arc_diameter_vox": Setting(ARC_DIAMETER_VOX, "k-Wave example page"),
            "focus_pos_vox": Setting([NX_PAGE / 2.0, NX_PAGE / 2.0], "k-Wave example page"),
            "source_freq_hz": Setting(SOURCE_FREQ_HZ, "k-Wave example page"),
            "source_mag_pa": Setting(SOURCE_MAG_PA, "k-Wave example page"),
            "recorded": Setting("p_max over the whole domain", "k-Wave example page"),
            "grid_shape_page_vox": Setting(
                [NX_PAGE, NX_PAGE],
                "inferred",
                "the page does not print Nx. focus_pos = [Nx/2, Nx/2] must sit about one "
                "radius from arc_pos = [20, 20]: at Nx = 128 that distance is 62.2 grid "
                "points against a radius of 60, which is the only round Nx that fits.",
            ),
            "grid_shape_vox": Setting(
                [NX, NX],
                "caustica",
                f"the page's own arc reaches 2.5 voxels from two faces, inside k-Wave's "
                f"20-voxel PML, and caustica refuses a source buried in the sponge "
                f"because that run converges quietly on a wrong field. The scene is "
                f"shifted {PAD_VOX} voxels off both faces and the domain grows to match. "
                f"Both engines run the padded domain.",
            ),
            "dx_m": Setting(
                DX,
                "inferred",
                "the page does not print dx. At 0.5 mm the 0.25 MHz drive has 12 points "
                "per wavelength in water, the arc's 60-voxel radius is 30 mm and its "
                "81-voxel chord is 40.5 mm, which is a focused single element of the "
                "size the example's figure shows.",
            ),
            "sound_speed_m_s": Setting(
                SOUND_SPEED,
                "inferred",
                "the page does not print the medium. 1500 m/s and 1000 kg/m^3 are "
                "k-Wave's water defaults and the example is explicitly non-absorbing.",
            ),
            "density_kg_m3": Setting(
                DENSITY, "inferred", "k-Wave's water default, as for the sound speed."
            ),
            "alpha_np_m": Setting(
                0.0,
                "k-Wave example page",
                "the page states a non-absorbing medium; caustica states the zero.",
            ),
            "pml_vox": Setting(
                PML_VOX,
                "inferred",
                "k-Wave's default PML in 2-D is 20 voxels inside the grid; the adapter "
                "passes the same thickness to both engines, so the damped band matches.",
            ),
            "cfl": Setting(
                0.3,
                "caustica",
                "the comparison setting of the parity contract, which is k-Wave's own "
                "makeTime default; caustica's default is 0.48.",
            ),
            "record_region": Setting(
                f"interior, {PML_VOX + 2} voxels shaved from every face",
                "caustica",
                "the k-Wave example records the whole domain including its PML, where "
                "the field is a sponge artefact rather than a field. Both engines are "
                "graded on the same interior.",
            ),
        }

    # ---------- the scene ----------

    def scene(self) -> AcousticScene:
        if self._scene is None:
            grid = Grid((NX, NX), DX, pml=PMLSpec(thickness=PML_VOX * DX))
            source = arc_cw_source(
                grid,
                f0=SOURCE_FREQ_HZ,
                amplitude=SOURCE_MAG_PA,
                apex_vox=_apex_vox(),
                roc_vox=ARC_RADIUS_VOX,
                aperture_vox=ARC_DIAMETER_VOX,
                axis=_axis(),
                label="k-Wave makeArc(radius=60, diameter=81)",
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

    def extra_engines(self) -> tuple[ExtraEngine, ...]:
        return ()

    def run_engine(self, name: str) -> EngineRun:
        scene = self.scene()
        solver = solvers.get(name)()
        kwargs: dict[str, object] = {"record_region": scene.record_region}
        if name != "kwave":
            kwargs["backend"] = self.backend
        result = solver.run(scene.grid, scene.medium, scene.source, scene.spec, **kwargs)
        field = np.asarray(result.p_max, dtype=np.float64)
        return EngineRun(
            engine=name,
            kind="solver",
            role="reference" if name == self.reference_engine() else "native",
            field=field,
            dx=DX,
            beam_axis=None,
            scalars={
                "peak_pa": float(field.max()),
                "phasor_peak_pa": float(np.abs(result.phasor).max()),
            },
            source_surface_pa=SOURCE_MAG_PA,
            provenance={
                "spp": int(result.spp),
                "dt_s": float(result.dt),
                "cfl_realized": float(result.dt * scene.medium.c_max / DX),
                "steps_total": int(result.steps_total),
                "settle_capped": bool(result.settle_capped),
                "converged_period": int(result.converged_period),
                "solver_backend": result.meta.get("backend"),
            },
        )

    def notes(self) -> tuple[str, ...]:
        return (
            "Both engines are driven by one source. The k-Wave example's own arc "
            "(radius 60, diameter 81 voxels) is sampled in arc length and deposited "
            "band-limited, and the k-Wave binary receives that same deposit as its "
            "per-voxel source weights rather than a staircased makeArc shell. This page "
            "therefore compares two propagators on an identical source; it does not "
            "compare k-Wave's binary voxel shell with caustica's band-limited deposit, "
            "which is a separate measurement and is not on this page.",
            "Both engines are graded on the interior, with the absorbing band and two "
            "further voxels shaved off. Inside a PML the field is a sponge artefact, and "
            "comparing two sponges is not a comparison of two solvers.",
            "The waveform metrics report 'not computed': this mirror records the "
            "steady-state maximum-pressure map, which is what the k-Wave example plots, "
            "and neither engine wrote a point history to correlate.",
        )
