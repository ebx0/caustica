# Contributing to caustica

Thanks for looking. This is a scientific library, so the bar that matters most
here is not style — it is **evidence**.

## The one rule

> A claim is not true because the code looks right. It is true because
> something measured it.

Nothing in this project is ticked off without a test or a measurement to point
at. The same applies to a pull request: if it
changes numerics, it comes with the number that shows what changed. "Tests
pass" is necessary, never sufficient — a defect that is invisible on the
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

Optional extras, none of them installed implicitly: `[gpu]` (cupy — do **not**
install it on Colab, the runtime already ships it), `[kwave]`, `[report]`.

## Where the documentation is

<https://ebx0.github.io/caustica/> is built from
[`ebx0/ebx0.github.io`, branch `caustica-docs`](https://github.com/ebx0/ebx0.github.io/tree/caustica-docs),
not from this repository — there are no pages here. Four of those pages are a
contract rather than prose (the GUI contract, the job format, the conventions,
the extension points) and that repository's build asserts them against the
caustica it installs from `master`. So a change here that moves one of those
surfaces turns the *site* build red, not this one: land the code, then open the
matching change there.

## Before you open a pull request

```bash
pytest              # the whole suite; some tests skip without a GPU or k-Wave
ruff check .        # lint; the tree is kept clean, not almost clean
```

Both run in CI on Linux and Windows across Python 3.10–3.13, plus a
clean-environment leg that installs the built wheel and runs an example job
from it — so packaging breakage is caught before a release, not after.

For anything that touches the GPU path, `scripts/dev_validate.py` is the
development validator: `--profile local` is a light CPU pass, `--profile colab`
runs the full ladder on a hosted GPU and writes a stamped JSON report.

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
- **Scope** is the package or area the change lives in — `kspace`, `solvers`,
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
period and step it happened at — that line is the most useful thing you can
paste.

## Conduct

Participation is covered by the [Code of Conduct](CODE_OF_CONDUCT.md).
