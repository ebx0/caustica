# caustica — working notes for Claude

## Commit messages

Follow the **Commit conventions** section of [CONTRIBUTING.md](CONTRIBUTING.md)
exactly. The whole history was rewritten to this style on 2026-08-24; do not
reintroduce the old habits.

- `type(scope): imperative summary`. Types are limited to `feat`, `fix`,
  `docs`, `test`, `refactor`, `perf`, `build`, `ci`, `chore` — no `janitor`,
  `dev`, `evidence` or `cleanup`. Scope is a real module name.
- Subject at most 72 characters, lower case after the colon, no trailing
  period.
- **No milestone or ticket codes anywhere in the message** — not `M11`, `W2`,
  `M10i/W2`, `F1-F6`, `D34`, `K15`, `U1b`, `T9`, `janitor(03,08)`. That
  bookkeeping belongs in the git-ignored `archive/` ledgers.
- **No em dashes**, no process narration ("the review round", "the adversarial
  round", "belt-and-braces", "the seam answers from a cold import"), no running
  test counts (`637 -> 647 tests`), no restating the changed-file list.
- **Default to a subject line and no body.** Roughly one commit in eight earns
  a body: write one only for a *why* the diff cannot show — a measurement, a
  non-obvious mechanism, or a decision someone would otherwise undo. Wrap at
  72 columns.
- **No `Co-Authored-By` trailers** and no generated-with footer. The user's
  decision, 2026-08-24: AI use is disclosed in prose where it matters, not as
  noise on every commit.

## Git identity

Commits are authored as `ebx0 <97095537+ebx0@users.noreply.github.com>`. This
is set in the user's global git config; do not override it per-commit.

## Committing

Worker agents never stage, commit or push. The orchestrating session
commits and pushes `develop` with the user's standing permission
(given 2026-09-05), one task per commit, splitting overlapping files by
hunk when two tasks touched one, and only at a moment when no worker is
writing to the tree. Never `git add -A`.

## Planning documents

The plan to v0.1 lives in `planning/` (decision D-017). Read in this order:

1. `planning/STATUS.md` where we are, the next three tasks, the active lane.
2. `planning/briefs/README.md` the standing orders every implementation
   agent receives, then the phase file (`briefs/P<n>.md`) for the task.
3. `planning/PLAN.md` the phases, tasks, scope tiers and scenario gates.
4. `planning/VALIDATION.md` the claims and their numeric gates (`V-xx`).
5. `planning/ISSUES.md` defects and gaps (`I-xxx`) with the task that
   closes each.
6. `planning/DECISIONS.md` settled questions (`D-xxx`); do not re-argue
   them in a brief.
7. `planning/RESEARCH.md` outside reading with a verdict per item; a
   "verify" item is not a design input.

Per-task measurement reports go to `planning/reports/<task-id>.md`. The
old ledger under `archive/` is not a planning input (D-001).

## Orchestration

The user's standing arrangement: Fable directs, Opus subagents do the work.
The procedure is the local skill `.claude/skills/caustica-orchestrate/SKILL.md`
(invoke `/caustica-orchestrate`). Implementation tasks go through the saved
workflow `.claude/workflows/caustica-tasks.js`, launched by `scriptPath`
with `args: {phase, tasks}` (the name route fails on this host). A dead run
is recovered with `planning/tools/harvest.py <run-id>`. All three are
git-ignored, like `planning/`.

## Working rules for agents

- One task per agent, exactly the brief's scope. A mismatch between brief
  and code is reported, not improvised around.
- Evidence rule: a numerics change ships with a measurement against a
  named reference. Tests passing is necessary, never sufficient.
- The interpreter is the in-tree `.venv`, and `caustica` is installed in
  no other one. Either activate it or spell it out: from a fresh shell,
  bare `python -m pytest` collects 49 `ModuleNotFoundError: caustica`
  errors and exits non-zero, which reads as a broken tree and is not one.
  Use `./.venv/Scripts/python.exe -m ...` on Windows.
- Run `ruff check src tests apps`, `ruff format --check src tests apps`
  and `python -m pytest -m "not kwave and not network" -q` before
  reporting; run the `gpu` marker too when CuPy is available (this laptop
  has an RTX 5050).
- Piping pytest through `tail` hides its exit code: the pipeline reports
  `tail`'s status, so a collection error reads as a pass. Redirect to a
  file and check the exit code, or read the head as well as the tail.
- Bump `NUMERICS_SCHEME` when an existing solver's trajectory moves, and
  say so in CHANGELOG under Unreleased.
- English only in the tree; no em dashes; no milestone or ticket codes.
- Never `git add -A`; another session may be working in the same tree.
