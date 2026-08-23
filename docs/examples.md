# Examples

Five worked examples. Every figure on this page was produced by running the job
beside it, and every number in the tables came back from that run — nothing here
is a sketch or a remembered result. `python scripts/make_examples.py` reproduces
the whole page.

!!! note "The grids are small on purpose"

    Each example is sized to solve in seconds on a laptop CPU, so the page can be
    regenerated on any machine. Real studies run at finer `dx` and larger boxes;
    nothing about the job changes except those numbers.


## A focused bowl in water

*The reference case, and the one with a closed-form answer to check against.*

A 20 mm bowl of 20 mm curvature — f/1.0 — driven at 1.2 MHz into water. This is the shape every other example is a variation on, and it is the one case where the answer is known independently: O'Neil's 1949 solution for a spherical cap. The right-hand panel is the solver's axis against that solution, both normalised — the analytic form is stated in surface velocity and the job in surface pressure, so only the shape is comparable, and the shape is what a solver gets wrong.

<div class="example-figure" markdown>
![A focused bowl in water: The reference case, and the one with a closed-form answer to check against.](assets/examples/bowl.svg#only-light)
![A focused bowl in water: The reference case, and the one with a closed-form answer to check against.](assets/examples/bowl-dark.svg#only-dark)
</div>

| what | measured |
|---|---|
| peak pressure | 1.62 MPa |
| focus offset from geometric | 1.08 mm |
| −6 dB axial × lateral | 9.57 mm × 1.63 mm |
| correlation with O'Neil on axis | r = 0.9958 |
| linear focal gain (analytic) | 13.5× |

??? example "The job — `bowl.json`"

    ```json
    {
      "format": "caustica-job/1",
      "kind": "explicit",
      "name": "focused_bowl",
      "medium": {
        "kind": "homogeneous"
      },
      "grid": {
        "ndim": 3,
        "dx_mm": 0.18,
        "size_mm": [
          25.92,
          25.92,
          32.4
        ],
        "pml": {
          "thickness_mm": 2.7
        }
      },
      "source": {
        "kind": "array",
        "array": {
          "kind": "bowl",
          "d_outer_mm": 20.0,
          "roc_mm": 20.0
        },
        "apex_mm": [
          12.96,
          12.96,
          4.5
        ]
      },
      "drive": {
        "f0_mhz": 1.2,
        "amplitude_kpa": 100.0
      },
      "run": {
        "spec": {
          "min_settle_periods": 2,
          "max_settle_periods": 8
        },
        "harmonics": [
          1
        ]
      },
      "solver": "linear"
    }
    ```

    ```python
    import caustica
    res = caustica.simulate("bowl.json")
    ```

<small>Solved in 8.6 s on one CPU core budget.</small>


## Nonlinear propagation, against Fubini

*Waveform steepening, harmonic by harmonic, checked against the closed form.*

A 2 MPa plane wave in water with β = 3.5, marched with `westervelt` until the harmonics stop moving. Fubini's series is the exact answer for exactly this problem up to shock formation, so the second panel is not an illustration — it is the gate the test suite enforces, drawn. The solver has to land inside 5 % of Fubini for A₂/A₁ everywhere along the σ range this box reaches, and the table says where it actually lands. The thin trace is the raw deviation: a standing-wave ripple at the acoustic wavelength rides on it, so the heavy line is that same deviation averaged over exactly one wavelength.

Sixteen points per wavelength, not the production four: harmonic-cascade accuracy needs headroom *above* f₀, and at 8 ppw the third harmonic sits at 2.7 and aliases into the second. That is a resolution rule about harmonic physics, separate from the one that bounds `p_max` capture — the kind of distinction that only shows up when you check against a closed form instead of against your own intuition.

<div class="example-figure" markdown>
![Nonlinear propagation, against Fubini: Waveform steepening, harmonic by harmonic, checked against the closed form.](assets/examples/harmonics.svg#only-light)
![Nonlinear propagation, against Fubini: Waveform steepening, harmonic by harmonic, checked against the closed form.](assets/examples/harmonics-dark.svg#only-dark)
</div>

| what | measured |
|---|---|
| drive | 2.01 MPa plane wave at 1 MHz, beta = 3.5 |
| resolution | 16 points per wavelength (3f0 at 5.3) |
| shock distance | 77 mm |
| sigma covered | 0.05 - 0.59 |
| A2/A1 vs Fubini, typical | 1.9 % |
| A2/A1 vs Fubini, worst single point | 4.0 % |
| the gate the tests enforce | 5 % |

```python
import caustica.solvers as solvers
from caustica import Grid, Medium, PMLSpec
from caustica.materials import water
from caustica.solvers import CWRunSpec
from caustica.sources import plane_cw_source

dx = 1500.0 / (1.0e6 * 16.0)                     # 16 points per wavelength
grid = Grid(shape=(600,), dx=dx, pml=PMLSpec(thickness=40 * dx))
med = Medium.homogeneous(grid.shape, water(c=1500.0, beta=3.5))
src = plane_cw_source(grid, f0=1.0e6, amplitude=2.0e6, position_vox=60)

res = solvers.get("westervelt")().run(
    grid, med, src,
    CWRunSpec(min_settle_periods=45, max_settle_periods=100, convergence_tol=0.003),
    backend="numpy", harmonics=(1, 2, 3),
)
a1, a2 = res.harmonic_amp(1), res.harmonic_amp(2)
```

<small>Solved in 0.1 s on one CPU core budget.</small>


## Focusing through bone

*Constructive solid geometry, painted onto the grid, with per-voxel physics.*

The medium becomes a scene: 2 mm of skin, 4 mm of bone at 2800 m/s, then brain — three boxes rasterized onto the same grid the solver runs on. Nothing else in the job changes. Running it twice, once against the layers and once against plain water, is what makes the figure a measurement rather than a picture: the axial pair shows what the barrier costs and where it puts the focus instead.

The first thing it shows is that the loudest point in the box stops being the focus. Bone reflects hard enough that the standing wave in front of it beats everything downstream, which is why the table compares the two runs *past* the barrier rather than comparing their maxima.

Read this one as a demonstration of the medium model, not as a transcranial result. Strong heterogeneity is not among the things this library has [gated itself against](validation.md) — a 1.9× jump in sound speed across one voxel is exactly where a solver earns or loses its accuracy, and caustica has not yet proved which.

<div class="example-figure" markdown>
![Focusing through bone: Constructive solid geometry, painted onto the grid, with per-voxel physics.](assets/examples/layered.svg#only-light)
![Focusing through bone: Constructive solid geometry, painted onto the grid, with per-voxel physics.](assets/examples/layered-dark.svg#only-dark)
</div>

| what | measured |
|---|---|
| layers | 2 mm skin, 4 mm bone (2800 m/s), then brain |
| focus in water | 1.62 MPa at z = 23.5 mm |
| focus through the barrier | 0.28 MPa at z = 19.7 mm |
| transmitted to the focus | 17 % |
| focus moved | -3.8 mm along z |
| loudest point in the box | z = 8.9 mm — in front of the bone, not the focus |

??? example "The job — `layered.json`"

    ```json
    {
      "format": "caustica-job/1",
      "kind": "explicit",
      "name": "layered",
      "medium": {
        "kind": "scene",
        "scene": {
          "ndim": 3,
          "background": 0,
          "objects": [
            {
              "shape": {
                "kind": "box",
                "center_mm": [
                  12.96,
                  12.96,
                  7.0
                ],
                "size_mm": [
                  25.92,
                  25.92,
                  2.0
                ]
              },
              "label": 1
            },
            {
              "shape": {
                "kind": "box",
                "center_mm": [
                  12.96,
                  12.96,
                  10.0
                ],
                "size_mm": [
                  25.92,
                  25.92,
                  4.0
                ]
              },
              "label": 2
            },
            {
              "shape": {
                "kind": "box",
                "center_mm": [
                  12.96,
                  12.96,
                  22.2
                ],
                "size_mm": [
                  25.92,
                  25.92,
                  20.4
                ]
              },
              "label": 3
            }
          ]
        },
        "materials": {
          "0": {
            "name": "water",
            "c": 1500.0,
            "rho": 1000.0,
            "alpha_np_m": 0.025,
            "beta": 0.0
          },
          "1": {
            "name": "skin",
            "c": 1610.0,
            "rho": 1090.0,
            "alpha_np_m": 21.0,
            "beta": 0.0
          },
          "2": {
            "name": "bone",
            "c": 2800.0,
            "rho": 1900.0,
            "alpha_np_m": 200.0,
            "beta": 0.0
          },
          "3": {
            "name": "brain",
            "c": 1550.0,
            "rho": 1040.0,
            "alpha_np_m": 8.0,
            "beta": 0.0
          }
        }
      },
      "grid": {
        "ndim": 3,
        "dx_mm": 0.18,
        "size_mm": [
          25.92,
          25.92,
          32.4
        ],
        "pml": {
          "thickness_mm": 2.7
        }
      },
      "source": {
        "kind": "array",
        "array": {
          "kind": "bowl",
          "d_outer_mm": 20.0,
          "roc_mm": 20.0
        },
        "apex_mm": [
          12.96,
          12.96,
          4.5
        ]
      },
      "drive": {
        "f0_mhz": 1.2,
        "amplitude_kpa": 100.0
      },
      "run": {
        "spec": {
          "min_settle_periods": 2,
          "max_settle_periods": 8
        },
        "harmonics": [
          1
        ]
      },
      "solver": "linear"
    }
    ```

    ```python
    import caustica
    res = caustica.simulate("layered.json")
    ```

<small>Solved in 5.1 s on one CPU core budget.</small>


## Steering a phased array

*A spiral element table, delay-and-sum phases, and a focus that moves off axis.*

A 32-element Archimedean spiral on the same shell as the bowl. `focus.mode: "steered"` asks for delay-and-sum phases toward a target 3 mm off the axis; the run records where the peak actually landed, which is not quite the same thing. The plane through the target shows the cost of steering a sparse array: the focus arrives, and so do the grating lobes that a 32-element aperture cannot suppress.

<div class="example-figure" markdown>
![Steering a phased array: A spiral element table, delay-and-sum phases, and a focus that moves off axis.](assets/examples/steered.svg#only-light)
![Steering a phased array: A spiral element table, delay-and-sum phases, and a focus that moves off axis.](assets/examples/steered-dark.svg#only-dark)
</div>

| what | measured |
|---|---|
| array | 128-element spiral, 20 mm, ROC 20 mm |
| steering target | 4 mm off axis, at the focal depth |
| where the peak landed | 3.60 mm off axis |
| miss from the requested target | 2.19 mm |
| peak, steered | 0.69 MPa |
| peak, unsteered on the same array | 0.74 MPa |

??? example "The job — `steered.json`"

    ```json
    {
      "format": "caustica-job/1",
      "kind": "explicit",
      "name": "steered_array",
      "medium": {
        "kind": "homogeneous"
      },
      "grid": {
        "ndim": 3,
        "dx_mm": 0.18,
        "size_mm": [
          25.92,
          25.92,
          32.4
        ],
        "pml": {
          "thickness_mm": 2.7
        }
      },
      "source": {
        "kind": "array",
        "array": {
          "kind": "archimedean_spiral",
          "n_elements": 128,
          "d_outer_mm": 20.0,
          "d_inner_mm": 7.0,
          "roc_mm": 20.0
        },
        "apex_mm": [
          12.96,
          12.96,
          4.5
        ],
        "focus": {
          "mode": "steered",
          "target_mm": [
            16.96,
            12.96,
            24.5
          ]
        }
      },
      "drive": {
        "f0_mhz": 1.2,
        "amplitude_kpa": 100.0
      },
      "run": {
        "spec": {
          "min_settle_periods": 2,
          "max_settle_periods": 8
        },
        "harmonics": [
          1
        ]
      },
      "solver": "linear"
    }
    ```

    ```python
    import caustica
    res = caustica.simulate("steered.json")
    ```

<small>Solved in 3.4 s on one CPU core budget.</small>


## From pressure to thermal dose

*Pennes bioheat and CEM43, driven by a real acoustic field.*

The acoustic field is only half of a HIFU calculation. This one turns a solved field into a volumetric heat source (`Q = 2αI`), marches Pennes bioheat through 20 s of sonication and cooling in perfused liver, and accumulates CEM43 across both phases. It uses the layer below `simulate()`, because the thermal chain consumes a solver result and a medium directly — which is also the honest way to show that the layer is there and is usable.

The right-hand panel is the reason a thermal model is not optional. The focal focal temperature reaches its steady state within the first second — heat leaves a focus this small as fast as it arrives — and then does not move again for the rest of the sonication. The dose does. CEM43 keeps accumulating for the whole half-minute the temperature is flat, which is why exposure is counted in dose rather than in degrees. Resolving that rise took a finer step than the plateau needs, so the run is three chained solves, each carrying the previous temperature *and* its accumulated dose forward.

<div class="example-figure" markdown>
![From pressure to thermal dose: Pennes bioheat and CEM43, driven by a real acoustic field.](assets/examples/thermal.svg#only-light)
![From pressure to thermal dose: Pennes bioheat and CEM43, driven by a real acoustic field.](assets/examples/thermal-dark.svg#only-dark)
</div>

| what | measured |
|---|---|
| sonication | 30 s on, then 30 s of cooling |
| peak acoustic pressure | 11.39 MPa |
| peak temperature | 50.2 °C |
| time to 90 % of that | 0.20 s |
| peak ΔT | 13.2 °C (ITRUSST flags above 2 °C) |
| peak CEM43 | 72 min (ablation is conventionally 240) |

```python
from caustica.sensors import HeatingSource
from caustica.thermal.pennes import PennesSolver
from caustica.thermal.properties import ThermalMedium

heat = HeatingSource.from_result(result, medium, grid.dx, harmonics=(1,))
tm = ThermalMedium.from_id_map(ids[region], db, grid.dx)

solver = PennesSolver(backend="numpy")
dt = 0.9 * solver.stable_dt(tm)
hot = solver.solve(t0, heat, tm, dt=dt, n_steps=int(20.0 / dt), dose=True)
cool = solver.solve(hot.temperature, None, tm, dt=dt,
                    n_steps=int(40.0 / dt), dose=True, dose0=hot.dose_cem43)
```

<small>Solved in 8.6 s on one CPU core budget.</small>


## Where these go next

- Every field above is available as `res.result.phasor` and saved by `res.save(...)` in the [`caustica-result/1`](gui_contract.md) layout.
- The numbers in the tables are the same ones the HTML report quotes — see [what has been measured](validation.md) for how they are gated.
- To write your own job, start from [the job reference](job_reference.md).
