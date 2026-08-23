#!/usr/bin/env python3
"""Measure per-step cost on this machine and write the performance page.

Everything on ``docs/benchmarks.md`` comes from ``planner.measure_step_time`` --
the same function the planner uses to decide whether to refuse a run, timing the
real op mix rather than a proxy. The script writes::

    docs/benchmarks.md
    docs/assets/benchmarks/scaling.svg   (and -dark)

    python scripts/make_benchmarks.py

Numbers are the MINIMUM over repeats. A minimum is the least-contaminated
estimate of what a machine can do; a mean on a laptop measures the laptop's other
tenants. The page says so, and says which laptop.
"""

from __future__ import annotations

import io
import json
import platform
import re
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
from make_howto import DARK, LIGHT, REPO, Theme, mix  # noqa: E402

matplotlib.rcParams["svg.fonttype"] = "path"

warnings.simplefilter("ignore")

OUT = REPO / "docs" / "assets" / "benchmarks"
PAGE = REPO / "docs" / "benchmarks.md"

#: cubes plus one long box, because HIFU grids are rarely cubic and the FFT
#: does not care about the aspect ratio but the cache does
SHAPES = [(64, 64, 64), (96, 96, 96), (128, 128, 128), (128, 128, 256), (160, 160, 160)]
REPEATS = 5
N_STEPS = 14

#: a representative run: 15 steps per period, converged around period 12
STEPS_PER_RUN = 15 * 12


def cpu_name() -> str:
    """The brand string, not the family/model/stepping platform.processor() gives."""
    try:
        if sys.platform == "win32":
            import winreg

            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0])
        if sys.platform == "darwin":
            import subprocess

            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001 - a nicer label is never worth a crash
        pass
    return platform.processor() or platform.machine()


def machine() -> dict:
    import caustica

    cpu = cpu_name()
    return {
        "caustica": caustica.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": re.sub(r"\s+", " ", cpu).strip(),
        "cores": __import__("os").cpu_count(),
        "fft_workers": caustica.cpu_fft_workers(),
        "numpy": np.__version__,
    }


def measure(shape, *, nonlinear: bool) -> float:
    """Minimum per-step wall time over REPEATS, in seconds."""
    from caustica.planner import measure_step_time

    best = float("inf")
    for _ in range(REPEATS):
        out = measure_step_time(shape, nonlinear=nonlinear, backend="numpy", n_steps=N_STEPS)
        best = min(best, float(out["t_step_s"]))
    return best


def plan_memory(shape) -> float:
    """GiB the planner says this run needs -- the number it actually gates on."""
    from caustica import Grid, Medium, PMLSpec
    from caustica.materials import Material, MaterialDB
    from caustica.planner import estimate
    from caustica.solvers import CWRunSpec
    from caustica.sources import bowl_cw_source

    dx = 0.25e-3
    grid = Grid(shape=shape, dx=dx, pml=PMLSpec(thickness=2.5e-3))
    db = MaterialDB(
        materials={1: Material(name="water", c=1500.0, rho=1000.0, alpha_np_m=0.0, beta=0.0)}
    )
    medium = Medium.from_id_map(np.ones(shape, np.int32), db)
    src = bowl_cw_source(
        grid,
        f0=1.0e6,
        amplitude=1.0e5,
        aperture_radius=6e-3,
        roc=15e-3,
        apex_vox=(shape[0] // 2, shape[1] // 2, grid.pml_vox + 4),
    )
    est = estimate(grid, medium, src, CWRunSpec(), solver="westervelt", measure=False)
    for name in ("bytes_total", "memory_bytes", "total_bytes", "vram_bytes"):
        if hasattr(est, name):
            return float(getattr(est, name)) / 2**30
    return float("nan")


# ---------------------------------------------------------------- the figure


def scaling_figure(rows, th: Theme) -> str:
    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=100)
    fig.patch.set_facecolor(th.fig_bg)
    ax.set_facecolor(th.fig_bg)
    for s in ax.spines.values():
        s.set_color(th.stroke)
    ax.tick_params(colors=th.muted, labelsize=8, length=3)

    n = np.array([r["voxels"] for r in rows], float)
    for key, label, color, marker in (
        ("linear", "linear", th.accent, "o"),
        ("westervelt", "westervelt", mix(th.fig_bg, th.accent, 0.55), "s"),
    ):
        t = np.array([r[key] for r in rows], float) * 1e3
        ax.plot(n / 1e6, t, color=color, lw=1.8, marker=marker, ms=5, label=label)

    # what perfectly linear scaling in voxel count would look like, anchored
    # at the smallest grid -- an FFT is N log N, so the real curves sit above it
    ref = rows[0]["linear"] * 1e3 * (n / n[0])
    ax.plot(n / 1e6, ref, color=th.muted, lw=1.0, ls=":", label="∝ N (for reference)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grid size [million voxels]", fontsize=8, color=th.muted)
    ax.set_ylabel("per-step wall time [ms]", fontsize=8, color=th.muted)
    ax.grid(True, which="both", color=mix(th.fig_bg, th.stroke, 0.7), lw=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(fontsize=8, frameon=True, facecolor=th.fig_bg, edgecolor=th.stroke)
    for t in leg.get_texts():
        t.set_color(th.text)
    fig.tight_layout()

    buf = io.StringIO()
    fig.savefig(buf, format="svg", facecolor=fig.get_facecolor())
    plt.close(fig)
    return re.sub(r"\d+\.\d{3,}", lambda m: f"{float(m.group()):.2f}", buf.getvalue())


# ------------------------------------------------------------------ the page


def page(rows, mach, taken_s) -> str:
    def fmt_shape(s):
        return "×".join(str(x) for x in s)

    def run_time(t):
        s = t * STEPS_PER_RUN
        return f"{s:.0f} s" if s < 90 else f"{s / 60:.1f} min"

    lines = [
        "# Performance",
        "",
        "Two things decide whether a job is worth starting: how long one step takes, and how",
        "many of them there are. caustica measures the first on your machine before it commits",
        "to anything — the numbers below are that same measurement, run here.",
        "",
        '!!! warning "One laptop, one afternoon"',
        "",
        "    This is a single CPU, measured in one session, and CPU performance is exactly the",
        "    kind of thing a laptop lies about: thermal throttling and background load moved the",
        "    step time on this machine by a factor of five during a single day of work. Read the",
        "    **scaling** and the **relative cost of the two solvers** as meaningful; read the",
        "    absolute throughput as one data point. `python scripts/make_benchmarks.py`",
        "    reproduces the table on yours, which is the number that matters to you.",
        "",
        "## The machine",
        "",
        "| | |",
        "|---|---|",
        f"| CPU | {mach['cpu']} |",
        f"| logical cores | {mach['cores']} (FFT workers: {mach['fft_workers']}) |",
        f"| platform | {mach['platform']} |",
        f"| Python / NumPy | {mach['python']} / {mach['numpy']} |",
        f"| caustica | {mach['caustica']}, `numpy` backend |",
        "",
        "## Per-step cost",
        "",
        "One step of the real op mix — forward FFT, k-space operator, inverse, the PML update,",
        "and for `westervelt` the nonlinear term and harmonic accumulation. Minimum of",
        f"{REPEATS} repeats of {N_STEPS} steps each.",
        "",
        "| grid | voxels | `linear` | `westervelt` | westervelt / linear | ns/voxel/step |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        ns = r["linear"] / r["voxels"] * 1e9
        lines.append(
            f"| {fmt_shape(r['shape'])} | {r['voxels'] / 1e6:.2f} M "
            f"| {r['linear'] * 1e3:.1f} ms | {r['westervelt'] * 1e3:.1f} ms "
            f"| {r['westervelt'] / r['linear']:.2f}× | {ns:.1f} |"
        )

    ratios = [r["westervelt"] / r["linear"] for r in rows]
    lines += [
        "",
        "Two things in that table are worth more than the absolute numbers. **Nonlinearity is "
        "nearly free.** The extra work is one multiply-add per voxel against a step whose cost "
        "is almost entirely FFTs, and it shows: the ratio here runs "
        f"{min(ratios):.2f}–{max(ratios):.2f}×"
        + (
            ", which straddles 1.0 — on this machine the difference is smaller than the "
            "measurement noise. "
            if min(ratios) < 1.0
            else ". "
        )
        + "If you are unsure whether a job needs the nonlinear solver, cost is not the reason "
        "to skip it.",
        "",
        f"**And these are single-threaded FFTs.** caustica defaults to "
        f"`cpu_fft_workers() == {mach['fft_workers']}` on a {mach['cores']}-core machine, "
        "deliberately: a library that quietly grabs every core is a bad citizen inside someone "
        "else's parallel sweep. `caustica.set_cpu_fft_workers(n)` is the knob, and on this "
        "workload it is the first one to reach for.",
        "",
        '<div class="benchmark-figure" markdown>',
        "![Per-step wall time against grid size, for the linear and Westervelt k-space solvers,"
        " on a log-log axis](assets/benchmarks/scaling.svg#only-light)",
        "![Per-step wall time against grid size, for the linear and Westervelt k-space solvers,"
        " on a log-log axis](assets/benchmarks/scaling.svg#only-dark)".replace(
            "scaling.svg", "scaling-dark.svg"
        ),
        "</div>",
        "",
        "The cost is dominated by the FFTs, so it grows a little faster than the voxel count —"
        " the dotted line is what pure ∝ N would look like, anchored at the smallest grid."
        " What that buys is a **spectral** spatial derivative: the gate is 4 points per"
        " wavelength, where a second-order finite-difference code wants 10 or more, and the"
        " cube of that ratio is the real comparison.",
        "",
        "## What a whole run costs",
        "",
        f"A converged continuous-wave run on these grids is about {STEPS_PER_RUN} steps"
        " (15 per period, converged around period 12 — the solver stops when it has"
        " converged, so this varies with the job):",
        "",
        "| grid | `linear` | `westervelt` |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {fmt_shape(r['shape'])} | {run_time(r['linear'])} | {run_time(r['westervelt'])} |"
        )

    lines += [
        "",
        "This is why the planner exists. On a CPU, a native run prints the estimate first and",
        "**refuses a job whose estimate exceeds 5 minutes** — `--allow-slow-cpu` accepts the",
        "wait, a GPU backend avoids it. The refusal names the fix for the machine you are on",
        "rather than telling you to buy a better one.",
        "",
        "## Memory",
        "",
        "The planner inventories the engine's actual buffers rather than multiplying a guess by",
        "a fudge factor, which is what lets it answer *will this fit* instead of *this might",
        "fit*. For `westervelt` with one harmonic:",
        "",
        "| grid | voxels | planner says |",
        "|---|---|---|",
    ]
    for r in rows:
        mem = r["memory_gib"]
        cell = "—" if not np.isfinite(mem) else f"{mem:.2f} GiB"
        lines.append(f"| {fmt_shape(r['shape'])} | {r['voxels'] / 1e6:.2f} M | {cell} |")

    lines += [
        "",
        "Each extra harmonic you ask `westervelt` to capture is another complex field over the",
        "whole recorded region. Harmonics you do not ask for are not stored, for that reason.",
        "",
        "## On a GPU",
        "",
        "**There are no measured GPU numbers on this page, and there will not be until there",
        "are.** The CuPy backend is packaged and has run on A100 hardware, but its parity and",
        "full-size gates (milestone M7) are not closed, and a dedicated GPU performance round",
        "(M19) has not happened yet. Quoting a speed-up before those two things would be",
        "advertising, not measurement.",
        "",
        "What exists today is the planner's device database — datasheet figures, from which it",
        "estimates. It labels those estimates `db` and warns that they are datasheet-coarse,",
        "roughly a factor of two. For the ±25 % path, calibrate on the device you actually have:",
        "",
        "```python",
        "from caustica.planner import calibrate",
        "",
        "calibrate()          # times a few real shapes on this device, records the fit",
        "```",
        "",
        "After that the planner reports `calibrated` instead of `db`, and says so in the plan",
        "block it prints before every run.",
        "",
        "## Reproducing this",
        "",
        "```bash",
        "pip install -e '.[dev,report]'",
        "python scripts/make_benchmarks.py",
        "```",
        "",
        f"It takes about {taken_s / 60:.0f} minutes and rewrites this page in place with your",
        "machine's numbers. If your ns/voxel/step is far off the table above, that is",
        "information — most of the gap between two CPUs on this workload is FFT threading and",
        "memory bandwidth, and `caustica.set_cpu_fft_workers()` is the first knob to try.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    t0 = time.perf_counter()
    mach = machine()
    print(f"  {mach['cpu']} / {mach['cores']} cores / numpy {mach['numpy']}")

    rows = []
    for shape in SHAPES:
        print(f"  {shape} ...", flush=True)
        row = {
            "shape": shape,
            "voxels": int(np.prod(shape)),
            "linear": measure(shape, nonlinear=False),
            "westervelt": measure(shape, nonlinear=True),
            "memory_gib": plan_memory(shape),
        }
        rows.append(row)
        print(
            f"    linear {row['linear'] * 1e3:7.1f} ms   "
            f"westervelt {row['westervelt'] * 1e3:7.1f} ms   "
            f"mem {row['memory_gib']:.2f} GiB"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    for th in (LIGHT, DARK):
        name = "scaling" + ("-dark" if th.name == "dark" else "") + ".svg"
        (OUT / name).write_text(scaling_figure(rows, th), encoding="utf-8")

    taken = time.perf_counter() - t0
    PAGE.write_text(page(rows, mach, taken), encoding="utf-8")
    print(f"  wrote {PAGE.relative_to(REPO)} ({taken / 60:.1f} min)")
    (OUT / "measured.json").write_text(
        json.dumps({"machine": mach, "rows": rows}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
