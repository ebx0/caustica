# Contributing to caustica

Thanks for looking. This is a scientific library, so the bar that matters most
here is not style, it is **evidence**.

## The one rule

> A claim is not true because the code looks right. It is true because
> something measured it.

Nothing in this project is ticked off without a test or a measurement to point
at. The same applies to a pull request: if it
changes numerics, it comes with the number that shows what changed. "Tests
pass" is necessary, never sufficient: a defect that is invisible on the
machine you ran on has happened here before (see the 256³ entry in
`CHANGELOG.md`), and the fix was to write the gate on the *input* to the
operation rather than on the output that looked fine.

## Setting up

```bash
git clone https://github.com/ebx0/caustica
cd caustica
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional extras, none of them installed implicitly: `[gpu]` (cupy, which you
must **not** install on Colab because the runtime already ships it),
`[kwave]`, `[report]`.

## Where the documentation is

<https://ebx0.github.io/caustica/> is built from
[`ebx0/ebx0.github.io`, branch `caustica-docs`](https://github.com/ebx0/ebx0.github.io/tree/caustica-docs),
not from this repository, and there are no pages here. Four of those pages are a
contract rather than prose (the GUI contract, the job format, the conventions,
the extension points) and that repository's build asserts them against the
caustica it installs from `master`. So a change here that moves one of those
surfaces turns the *site* build red, not this one: land the code, then open the
matching change there.

## Before you open a pull request

```bash
pytest                # the whole suite; some tests skip without a GPU or k-Wave
pytest -m "not slow"  # the fast pass: 72 s here, against 4 to 5 min for the whole suite
ruff check .          # lint; the tree is kept clean, not almost clean
```

Lint and the suite run in CI on Linux and Windows across Python 3.10 to 3.13
(CI takes the suite as `pytest -m "not kwave and not network"`), plus a
clean-environment leg that installs the built wheel and runs an example job
from it, so packaging breakage is caught before a release, not after.

For anything that touches the GPU path, `scripts/dev_validate.py` is the
development validator: `--profile local` is a light CPU pass, `--profile colab`
runs the full ladder on a hosted GPU and writes a stamped JSON report.

On a machine that has a CUDA device, `.\scripts\run_gpu_local.ps1` (Windows) and
`bash scripts/run_gpu_local.sh` (everywhere else) are the one command that
produces this machine's GPU evidence: the `gpu`-marked tests, then
`python -m caustica.validation gpu-gates`, with both logs, a `SUMMARY.md` and
the suite's own `REPORT.md` in one stamped folder under
`benchmarks/reports/gpu_gates/<device>-<date>/`. Call the shell one through
`bash`, since the repository is developed on Windows and git records no
executable bit here. Add `--quick` (`-Quick`) for a smaller ladder when you
only want to know that the device still works; its rungs are absolute VRAM
targets sized for a laptop card, so on a datacentre card they are short enough
that the `time` gate grades start-up rather than the model and only the other
four gates mean anything. The scripts exit 0 only when both legs are green,
1 when a gpu-marked test failed, 64 on a bad option, 69 when there is no
in-tree `.venv`, and otherwise pass the suite's own code through (2 no usable
GPU, 4 a gate failed or is incomplete). Expect 4 on a small card: the
`fullsize` criterion needs a 512-cubed run at dx = 0.30 mm, about 13.2 GiB, so
a device with less than roughly 15 GiB free leaves that one gate INCOMPLETE
however green the rest is.

Give the GPU the machine to itself while it runs, and read a single `time` FAIL
on the full ladder as a reason to run again rather than as a defect. Seventeen
runs on an idle RTX 5050 laptop, eleven full ladders and six quick ones, passed
the gate fourteen times; all three failures were full ladders and all six quick
runs passed. The solve itself is steady: the 324-cubed rung took 23.37 to
23.81 s in every full run, a spread of 1.9%. What moves is the plan built from
the calibration probe, whose 192-cubed sample ranged 8.85 to 9.31 ms, near
8.87 ms in twelve runs and near 9.30 ms in four. Because the graded rung is
4.8x the largest size the probe measures, its step cost is extrapolated and
that 5% becomes 10 to 20 points: the 324-cubed rung was planned at 27.7 to
28.6 s after a low probe, and every one of those runs passed at +18 to +22%,
while the three runs that drew a high probe planned it at 30.6, 32.4 and
34.1 s and all three failed, at +29.6%, +37.6% and +45.9% against a tolerance
of +/-25%. Quick mode is the more reproducible half for the same reason: its
largest rung is 270-cubed, only 2.9x the probe, so even the runs that drew a
high probe landed at +16.5% and +18.1%.

Load matters, but the mechanism is load that *changes* between the probe and
the rungs, not load as such. With a second process on the card throughout, the
probe is slowed along with the rungs and the gate still passes (-4.9% and
+16.5%); with the same process arriving after the probe, the 2 GiB rung came
out at -39.2% and the gate failed.

## Commit conventions

One trunk: `master`. Work in a topic branch, open a PR against `master`.

History is meant to be readable by someone who has never seen the project
ledger, so it follows [Conventional Commits](https://www.conventionalcommits.org/)
with a narrow set of types and real module names as scopes:

```
type(scope): imperative summary, lower case, no full stop
```

- **Types**, and nothing outside this list: `feat`, `fix`, `docs`, `test`,
  `refactor`, `perf`, `build`, `ci`, `chore`.
- **Scope** is the package or area the change lives in: `kspace`, `solvers`,
  `planner`, `config`, `runner`, `io`, `report`, `registry`, `validation`,
  `thermal`, `study`, `colab`, `packaging`, `cli`. Leave it out when the change
  really is repository-wide.
- **Subject** is at most 72 characters, imperative ("add", "drop", "zero"), and
  free of milestone codes, ticket numbers and dates. Internal bookkeeping lives
  in the (git-ignored) `archive/` ledgers, which is precisely why it does not
  belong in a subject line.
- **Breaking changes** take `type(scope)!:` plus a `BREAKING CHANGE:` footer
  saying what callers have to do.

**A body is the exception, not the rule.** Most commits are one subject line.
Write a body only when the change turns on a *why* the diff cannot show: a
measurement that motivated it, a mechanism that is not visible in the code, or
a decision a future reader would otherwise undo. Then wrap at 72 columns, lead
with the mechanism, and give the number:

```
fix(kspace): zero the Nyquist bin in the collocated first derivative

k_vectors passed the raw fftfreq ladder to spectral_derivative_factors,
leaving a live Nyquist bin on every even-length axis. numpy's pocketfft
projects that away; cuFFT documents its input as Hermitian and is free
not to, and at 256^3 the GPU run reached NaN by period 2 while the
identical CPU run sat at 45 kPa.
```

What to keep out of a message: em dashes, narration of the process that
produced the change ("the review round", "belt-and-braces"), running test
counts, and the file list the diff already carries.

## Reporting a problem

Open an issue with the environment block from any caustica report (or
`python -c "import caustica; print(caustica.__version__)"` plus your platform,
numpy and, if relevant, cupy versions), the job or script that reproduces it,
and what you expected instead. If a run diverged, the error message names the
period and step it happened at, and that line is the most useful thing you can
paste.

## Conduct

Participation is covered by the [Code of Conduct](CODE_OF_CONDUCT.md).
