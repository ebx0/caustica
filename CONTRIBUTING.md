# Contributing to caustica

Thanks for looking. This is a scientific library, so the bar that matters most
here is not style — it is **evidence**.

## The one rule

> A claim is not true because the code looks right. It is true because
> something measured it.

`MILESTONES.md` is the project ledger, and no box in it is ticked without a
test or a measurement to point at. The same applies to a pull request: if it
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
install it on Colab, the runtime already ships it), `[kwave]`, `[report]`,
`[docs]`.

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

## Commit and branch conventions

- One trunk: `master`. Work in a topic branch, open a PR against `master`.
- Subject line: `kind(scope): what changed, in the imperative`, e.g.
  `fix(256^3): drop the Nyquist wavenumber from the collocated first derivative`.
  Common kinds: `feat`, `fix`, `docs`, `test`, `perf`, `chore`, `dev`.
- The body is where the reasoning goes. Say what was wrong, how you know, and
  what the numbers were. Future readers of a physics library need the *why*
  far more than they need the diff restated.

## Reporting a problem

Open an issue with the environment block from any caustica report (or
`python -c "import caustica; print(caustica.__version__)"` plus your platform,
numpy and, if relevant, cupy versions), the job or script that reproduces it,
and what you expected instead. If a run diverged, the error message names the
period and step it happened at — that line is the most useful thing you can
paste.

## Conduct

Participation is covered by the [Code of Conduct](CODE_OF_CONDUCT.md).
