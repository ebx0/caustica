"""Mirror of ``example_cpp_running_simulations`` (Using The C++ Code).

k-Wave's own page is about a mechanism, not a scene: write the simulation to
an HDF5 input file with ``SaveToDisk``, run the compiled
``kspaceFirstOrder-OMP`` binary from the command line, and read the HDF5
output back for plotting. The page carries no grid, no medium and no source;
it shows ``kspaceFirstOrder3D(kgrid, medium, source, sensor, 'SaveToDisk',
filename)`` and a command line, and nothing else. Every setting below is
therefore caustica's own, and says so.

Two things make this page worth its place in the gallery today.

**It is the only page whose comparison is between two implementations of the
same input file.** caustica's k-Wave adapter writes exactly that HDF5 and
drives exactly that binary, so the reference column IS the C++ code the
example is about; the native columns are caustica's own solvers on the same
setup. This is the cross-engine agreement claim, whose published gate is
whole-field ``r > 0.999`` and a peak within 1 %; the page's own note says
where it stands against that gate, which today is ``r`` just above 0.99 and a
peak about 2 % high on a flat piston, so it meets the claim's interim bar and
not its full one.

**Its second axis is cost**, which is the one axis the rest of the gallery
does not grade. That axis stays "not computed" here on purpose: M-29 and
M-30 refuse a timing that carries no load regime, and this run carries none.
A wall time is printed per engine as provenance, where a reader can see it is
an observation and not a benchmark.

Later the page changes shape: caustica will READ a k-Wave user's own input
file instead of writing one, which is the claim decision D-027 was taken for.
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
from caustica.sensors import HeatingSource
from caustica.solvers.base import interior_slices
from caustica.sources import disc_cw_source

NAME = "example_cpp_running_simulations"

#: Every number here is the mirror's own: the page carries none.
SHAPE = (64, 64, 96)
DX = 0.25e-3
PML_VOX = 10
FREQ_HZ = 1.0e6
SOURCE_PRESSURE_PA = 1.0e5
DISC_RADIUS_M = 5.0e-3
DISC_AXIAL_VOX = 14
SOUND_SPEED = 1500.0
DENSITY = 1000.0
ALPHA_DB_CM_AT_F0 = 0.75


def _material() -> Material:
    return Material(
        name="absorbing water-like medium at 1 MHz",
        alpha_np_m=db_cm_to_np_m(ALPHA_DB_CM_AT_F0),
        rho=DENSITY,
        c=SOUND_SPEED,
        beta=0.0,
        source="caustica; the k-Wave page carries no medium",
    )


class RunningSimulationsMirror(ParityMirror):
    """caustica's native solvers against the compiled k-Wave binary."""

    name = NAME

    def __init__(self, backend: str = "numpy") -> None:
        self.backend = backend
        self._scene: AcousticScene | None = None

    # ---------- the example ----------

    def example(self) -> ParityExample:
        return ParityExample(
            name=NAME,
            title="Running C++ Simulations",
            category="Using The C++ Code",
            demonstrates=(
                "Writes the simulation to an HDF5 input file with the SaveToDisk flag, "
                "runs the compiled kspaceFirstOrder-OMP binary from the command line, "
                "and reads the HDF5 output back for plotting."
            ),
            comparison_axis=(
                "the field from the binary against the field from caustica's native "
                "solver on the same input file, plus wall-clock time"
            ),
            source_url=("http://www.k-wave.org/documentation/example_cpp_running_simulations.php"),
            faithful_mirror=True,
            mirror_note=(
                "Faithful to the mechanism, which is all the page carries: it publishes "
                "no grid, no medium and no source. Today the page shows caustica driving "
                "the k-Wave binary; later it becomes caustica reading a k-Wave user's own "
                "input file and running it."
            ),
            graded_quantity="steady-state pressure amplitude at 1 MHz",
            graded_units="Pa",
            traits={
                "field": holds(),
                "peak": holds(),
                "focused_beam": absent(
                    "the source is a flat piston, so the field has no focus: its axial "
                    "maximum is the last near-field interference maximum of an unfocused "
                    "aperture, and calling that a focal volume would name a diffraction "
                    "pattern after something it is not."
                ),
                "source_surface": holds(),
                "beam_axis": holds(),
                "sidelobe": holds(),
                "nonlinear": absent(
                    "the mirror sets beta = 0, so both engines integrate a linear medium "
                    "and no harmonic of the drive exists to compare."
                ),
                "third_harmonic": absent(
                    "the medium is linear, so there is no second harmonic and no third."
                ),
                "time_series": holds(),
                "shock": absent(
                    "a linear medium cannot steepen a waveform, so no rise time is a "
                    "property of this example."
                ),
                "closed_surface": holds(),
                "integrable_loss": holds(),
                "absorbing": holds(),
                "varies_resolution": absent(
                    "this mirror runs one grid. The resolution sweep belongs on "
                    "example_na_optimising_performance, which is the page whose subject "
                    "it is."
                ),
            },
        )

    def settings(self) -> Mapping[str, Setting]:
        page = (
            "the k-Wave page publishes a mechanism and no scene: it shows SaveToDisk, a "
            "command line and a console log, and no grid, medium or source at all. "
        )
        return {
            "grid_shape_vox": Setting(
                list(SHAPE),
                "caustica",
                page + "64 by 64 by 96 at 0.25 mm is the largest 3-D scene that runs on "
                "both engines in seconds, which is what a page about driving a binary "
                "needs.",
            ),
            "dx_m": Setting(
                DX, "caustica", page + "0.25 mm is six points per wavelength at 1 MHz."
            ),
            "pml_vox": Setting(
                PML_VOX,
                "caustica",
                page + "10 voxels is k-Wave's own default PML in 3-D, and the adapter "
                "passes the same thickness to both engines.",
            ),
            "source_freq_hz": Setting(FREQ_HZ, "caustica", page + "1 MHz."),
            "source_pressure_pa": Setting(
                SOURCE_PRESSURE_PA, "caustica", page + "100 kPa, well inside the linear regime."
            ),
            "disc_radius_m": Setting(
                DISC_RADIUS_M,
                "caustica",
                page + "a 5 mm flat piston: the simplest 3-D source with a defined "
                "surface, so the focal-gain metric means something.",
            ),
            "sound_speed_m_s": Setting(SOUND_SPEED, "caustica", page + "water."),
            "density_kg_m3": Setting(DENSITY, "caustica", page + "water."),
            "alpha_db_cm_at_1mhz": Setting(
                ALPHA_DB_CM_AT_F0,
                "caustica",
                page + "the medium absorbs on purpose, so the absorbed-power invariant "
                "(M-27) has something to compare.",
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
                "inside a PML the field is a sponge artefact, not a field. Both engines "
                "are graded on the same interior.",
            ),
            "k_wave_route": Setting(
                "kspaceFirstOrder-OMP through k-wave-python's HDF5 input file",
                "k-Wave example page",
                "the page's own mechanism: caustica's adapter writes the input file the "
                "C++ binary reads and drives the binary, which is what makes the "
                "reference column here the C++ code the example is about.",
            ),
        }

    # ---------- the scene ----------

    def scene(self) -> AcousticScene:
        if self._scene is None:
            grid = Grid(SHAPE, DX, pml=PMLSpec(thickness=PML_VOX * DX))
            source = disc_cw_source(
                grid,
                f0=FREQ_HZ,
                amplitude=SOURCE_PRESSURE_PA,
                radius=DISC_RADIUS_M,
                center_vox=(SHAPE[0] // 2, SHAPE[1] // 2, DISC_AXIAL_VOX),
            )
            self._scene = AcousticScene(
                grid=grid,
                medium=Medium.homogeneous(grid.shape, _material()),
                source=source,
                spec=parity_run_spec(min_settle_periods=8, max_settle_periods=48),
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
        result = solver.run(scene.grid, scene.medium, scene.source, scene.spec, **kwargs)
        amp = np.asarray(result.amp, dtype=np.float64)
        heat = HeatingSource.from_result(result, scene.medium, scene.grid.dx, harmonics=(1,))
        return EngineRun(
            engine=name,
            kind="solver",
            role="reference" if name == self.reference_engine() else "native",
            field=amp,
            dx=DX,
            beam_axis=2,
            scalars={
                "peak_pa": float(amp.max()),
                "focal_gain": float(amp.max() / SOURCE_PRESSURE_PA),
            },
            source_surface_pa=SOURCE_PRESSURE_PA,
            absorbed_power_w=float(heat.total_power_w),
            provenance={
                "spp": int(result.spp),
                "dt_s": float(result.dt),
                "cfl_realized": float(result.dt * scene.medium.c_max / DX),
                "steps_total": int(result.steps_total),
                # What the run reported, not what it was asked for.
                "solver_backend": result.meta.get("backend"),
            },
        )

    def notes(self) -> tuple[str, ...]:
        return (
            "The reference column is the compiled k-Wave binary, reached through the "
            "same HDF5 input file the example's own command line takes. That is the "
            "point of the page, and it is why this comparison is between two "
            "implementations rather than between two libraries' conveniences.",
            "The cost axis reports 'not computed'. Each engine's wall time is in its "
            "provenance, but M-29 and M-30 refuse to publish a ratio without the load "
            "regime a timing needs before it is a measurement.",
            "This page carries the library's cross-engine agreement claim against the "
            "compiled binary, whose published gate is whole-field r > 0.999 with the "
            "peak within 1 %. A flat piston is the hardest field in the gallery for "
            "that gate: its axial maximum is a near-field ridge of nearly equal local "
            "maxima, so the two engines can agree closely and still put their peaks "
            "several voxels apart. The measured numbers are on this page rather than "
            "the gate's name, and they meet the parity gate of r > 0.99 with the peak "
            "within 3 % and not the tighter one.",
            "The waveform metrics report 'not computed': this mirror records the "
            "steady-state phasor, and neither engine wrote a point history to correlate.",
        )
