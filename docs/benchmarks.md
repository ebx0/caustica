# Performance

Two things decide whether a job is worth starting: how long one step takes, and how
many of them there are. caustica measures the first on your machine before it commits
to anything — the numbers below are that same measurement, run here.

!!! warning "One laptop, one afternoon"

    This is a single CPU, measured in one session, and CPU performance is exactly the
    kind of thing a laptop lies about: thermal throttling and background load moved the
    step time on this machine by a factor of five during a single day of work. Read the
    **scaling** and the **relative cost of the two solvers** as meaningful; read the
    absolute throughput as one data point. `python scripts/make_benchmarks.py`
    reproduces the table on yours, which is the number that matters to you.

## The machine

| | |
|---|---|
| CPU | 13th Gen Intel(R) Core(TM) i5-13450HX |
| logical cores | 16 (FFT workers: 1) |
| platform | Windows-11-10.0.26200-SP0 |
| Python / NumPy | 3.12.10 / 2.2.6 |
| caustica | 0.1.0.dev0, `numpy` backend |

## Per-step cost

One step of the real op mix — forward FFT, k-space operator, inverse, the PML update,
and for `westervelt` the nonlinear term and harmonic accumulation. Minimum of
5 repeats of 14 steps each.

| grid | voxels | `linear` | `westervelt` | westervelt / linear | ns/voxel/step |
|---|---|---|---|---|---|
| 64×64×64 | 0.26 M | 12.7 ms | 12.8 ms | 1.00× | 48.5 |
| 96×96×96 | 0.88 M | 45.2 ms | 46.9 ms | 1.04× | 51.1 |
| 128×128×128 | 2.10 M | 115.6 ms | 119.8 ms | 1.04× | 55.1 |
| 128×128×256 | 4.19 M | 237.2 ms | 247.9 ms | 1.05× | 56.6 |
| 160×160×160 | 4.10 M | 246.8 ms | 268.0 ms | 1.09× | 60.2 |

Two things in that table are worth more than the absolute numbers. **Nonlinearity is nearly free.** The extra work is one multiply-add per voxel against a step whose cost is almost entirely FFTs, and it shows: the ratio here runs 1.00–1.09×. If you are unsure whether a job needs the nonlinear solver, cost is not the reason to skip it.

**And these are single-threaded FFTs.** caustica defaults to `cpu_fft_workers() == 1` on a 16-core machine, deliberately: a library that quietly grabs every core is a bad citizen inside someone else's parallel sweep. `caustica.set_cpu_fft_workers(n)` is the knob, and on this workload it is the first one to reach for.

<div class="benchmark-figure" markdown>
![Per-step wall time against grid size, for the linear and Westervelt k-space solvers, on a log-log axis](assets/benchmarks/scaling.svg#only-light)
![Per-step wall time against grid size, for the linear and Westervelt k-space solvers, on a log-log axis](assets/benchmarks/scaling-dark.svg#only-dark)
</div>

The cost is dominated by the FFTs, so it grows a little faster than the voxel count — the dotted line is what pure ∝ N would look like, anchored at the smallest grid. What that buys is a **spectral** spatial derivative: the gate is 4 points per wavelength, where a second-order finite-difference code wants 10 or more, and the cube of that ratio is the real comparison.

## What a whole run costs

A converged continuous-wave run on these grids is about 180 steps (15 per period, converged around period 12 — the solver stops when it has converged, so this varies with the job):

| grid | `linear` | `westervelt` |
|---|---|---|
| 64×64×64 | 2 s | 2 s |
| 96×96×96 | 8 s | 8 s |
| 128×128×128 | 21 s | 22 s |
| 128×128×256 | 43 s | 45 s |
| 160×160×160 | 44 s | 48 s |

This is why the planner exists. On a CPU, a native run prints the estimate first and
**refuses a job whose estimate exceeds 5 minutes** — `--allow-slow-cpu` accepts the
wait, a GPU backend avoids it. The refusal names the fix for the machine you are on
rather than telling you to buy a better one.

## Memory

The planner inventories the engine's actual buffers rather than multiplying a guess by
a fudge factor, which is what lets it answer *will this fit* instead of *this might
fit*. For `westervelt` with one harmonic:

| grid | voxels | planner says |
|---|---|---|
| 64×64×64 | 0.26 M | 0.02 GiB |
| 96×96×96 | 0.88 M | 0.08 GiB |
| 128×128×128 | 2.10 M | 0.19 GiB |
| 128×128×256 | 4.19 M | 0.38 GiB |
| 160×160×160 | 4.10 M | 0.37 GiB |

Each extra harmonic you ask `westervelt` to capture is another complex field over the
whole recorded region. Harmonics you do not ask for are not stored, for that reason.

## On a GPU

**There are no measured GPU numbers on this page, and there will not be until there
are.** The CuPy backend is packaged and has run on A100 hardware, but its parity and
full-size gates (milestone M7) are not closed, and a dedicated GPU performance round
(M19) has not happened yet. Quoting a speed-up before those two things would be
advertising, not measurement.

What exists today is the planner's device database — datasheet figures, from which it
estimates. It labels those estimates `db` and warns that they are datasheet-coarse,
roughly a factor of two. For the ±25 % path, calibrate on the device you actually have:

```python
from caustica.planner import calibrate

calibrate()          # times a few real shapes on this device, records the fit
```

After that the planner reports `calibrated` instead of `db`, and says so in the plan
block it prints before every run.

## Reproducing this

```bash
pip install -e '.[dev,report]'
python scripts/make_benchmarks.py
```

It takes about 2 minutes and rewrites this page in place with your
machine's numbers. If your ns/voxel/step is far off the table above, that is
information — most of the gap between two CPUs on this workload is FFT threading and
memory bandwidth, and `caustica.set_cpu_fft_workers()` is the first knob to try.
