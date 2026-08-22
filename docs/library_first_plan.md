# Library-First Conversion Plan

**Status:** written 2026-08-21 as the conversion plan; **partially executed** — see the
EXECUTION STATUS block below. §3's file:line anchors were verified on 2026-08-21 and have
drifted since; re-verify before relying on them.

**Goal.** `pip install caustica` → `import caustica` → a real simulation runs on a Colab GPU, with
no repository checkout, no notebook surgery, and no GUI anywhere in the dependency graph. The
library carries a **generic** volume-medium architecture; anything specific to one phantom source
(UWCEM) lives in its own repository. Any future GUI is a *consumer* of the library's contracts.

**How to use this document.** §5 is the work; each workstream lists the files to touch, the traps,
and the acceptance test that closes it. §7 is the trap list — read it before writing any code; it
contains eight ways to break working behaviour that are not obvious from the source. §12 is the
project's working agreement and is binding.

**EXECUTION STATUS (2026-08-22).** This document is now largely a *record*: W0 (=M10k),
W1 (=M10h) and W2 (=M10i) are **done and closed with evidence** — see `MILESTONES.md`.
Everything UWCEM now lives in **one file, `docs/uwcem.md`**, and its remaining tasks run LAST
(user decision 2026-08-22). Remaining live specs here: W5 (=M10m, next), W3+W4 (=M10j),
W8 (=M10l), W6 (=M10f), W7 (=M10g). Operating model: Fable 5 operates, Opus 5 subagents write
the code, criteria live in MILESTONES.md (K17). Full plugin architecture was added as M10n (K15).

**Language note.** English document (public-repo consistency; code and docstrings in this project
are English by convention). `MILESTONES.md` and `PLAN.md` are Turkish and stay that way.

---

## 1. Locked decisions

| # | Question | Decision | Consequence |
|---|---|---|---|
| D1 | Audience | **Real public library** — an outside researcher must succeed | Highest packaging bar; nothing may depend on your Drive, your paths, or your checkout |
| D2 | Install channel | **`git+https` now → PyPI at v0.1** | No release machinery yet; wheel correctness still verified from a clean env |
| D3 | Generated phantom dataset | **Never distributed; stays on your disk** | The library ships no anatomical data at all |
| D4 | API shape | **Three layers**: facade → objects → job JSON, all public | The facade must not become a second, divergent code path (R5) |
| D5 | No-GPU behaviour | **`auto` + loud warning + refuse large runs** | Planner-backed threshold gate |
| D6 | cupy installation | **Never auto-install; actionable error** | Zero pip-calling code in the library |
| D7 | GUI technology | **Undecided — freeze the contracts instead** | No GUI dependency chosen now |
| D8 | GUI location | **Separate repo (`caustica-gui`)** | One-way dependency, enforced by a test |
| D9 | Milestones | **New milestone group; M10f/M10g revised** | Numbering preserved; the criteria live in `MILESTONES.md` (§11) |
| D10 | API stability | **No guarantees before v1.0** | No API-snapshot test; `__all__` must stay honest |
| D11 | Progress feedback | **Callback + textual progress + mid-run preview** | One hook, reused by CLI, notebook and GUI |
| D12 | Document | **English, `docs/`** | This file |
| D13 | Packaged zero-data example | **A new synthetic job** modelled on `tests/test_runner.py::mini_job` | The nine stored setups reference a phantom file on a 560×700×480 grid — neither self-contained nor small |
| D14 | CI matrix | **Already correct; do not extend** | Only the clean-env wheel job and a network marker are missing |
| **D15** | **UWCEM code in the library** | **The library becomes completely UWCEM-free.** `phantom_dataset` medium kind, `stored_setup` job kind and `_require_uwcem` are removed | Licensing concern removed at the root; the import-direction rule becomes true instead of aspirational |
| **D16** | **Generic replacement** | **New medium kind `medium_volume`** — a caustica-owned format carrying either a label map + material DB or per-voxel property volumes, taking its grid from the file | Every phantom source enters through one door, not just UWCEM |
| **D17** | **Acoustic tissue values** | **Literature values move into `caustica.materials`**; the UWCEM media-number → tissue mapping stays in the UWCEM repo | `breast_default()` already exists there (`materials.py:67`) |
| **D18** | **UWCEM code + data** | **Own repository** (working name `uwcem-phantom`). Generated 4.5 GB and the nine setup JSONs stay local, out of git; caustica's `data/` empties | Consumes caustica, emits `medium_volume` files and explicit jobs |
| **D19** | **Sequencing of the split** | **Parallel with W1** | Packaging is independent of extraction; both start now |
| **D20** | **CPU refusal threshold** | **5 minutes**, `CAUSTICA_CPU_LIMIT_MIN`, `--allow-slow-cpu` escape | Small examples pass; a full-size job is refused |
| **D21** | **Mid-run preview** | **On by default** (every N periods, N=8 to match the checkpoint cadence); `progress=None` disables | One device→host copy of a coarse slice per callback |
| **D22** | **Colab output route** | **`/content` only — the library never mounts or writes Drive.** The user may pass any `--out`, including a Drive path they mounted themselves | Fewest assumptions; see R10 for the crash-loss risk |
| **D23** | **First handoff scope** | **W1 + W2 + W0 + W5**, in that order. W3 (facade) and W4 (progress) wait | W1/W2 collide with nothing; W5 needs W0c; W3 must land after W0c too (R13) but is cheapest paired with W4 |
| **D27** | **Arbitrary transducer geometry** | **New `elements` array kind** in the job schema: explicit positions + normals, inline or from `.npz`/`.csv` | `TransducerArray` is already generic; only the schema door is missing. Without it an outsider with their own array cannot use the job/queue/GUI path at all |
| **D28** | **`medium_volume` writer** | **The library writes the format too** — `write_medium_volume(...)` is public and the UWCEM repo calls it | A reader alone makes "bring your own volume" a dead promise; single source for the format |
| **D29** | **Outsider documentation** | **Minimal set lands in this handoff**: `caustica schema`, `docs/job_reference.md`, `docs/conventions.md`, README Colab quickstart | Nobody can author a job by reading pydantic source; conventions (phasor sign, Np/m, amplitude semantics) are how results get silently misread |
| **D30** | **GPU parity timing** | **Unchanged** — verified at the first Colab session, after M10j | User's call (2026-08-21). Until then README marks every GPU claim unverified |
| **D31** | **Low ppw** | **Loud warning, never a block.** Repeated in the plan output, `status.json`, `run_meta.json` and at the top of the report | A hard threshold would block the project's own production setting (1.88 ppw at 2f0 — a deliberate choice behind the dx=0.30 lock). Ignorable, but not scroll-past-able |
| **D32** | **CPU FFT threading** | ~~`workers=-1` by default~~ → **measured on an idle machine; default stayed 1** (M10i outcome — no gain on this hardware, `cpu_workers` knob added). Order held: measure → recalibrate → gate | The decision rule was "fit the plan to the measurement"; the measurement said single-thread |
| **D33** | **Visibility of critical events** | **`warnings.warn`** for backend fallback and low ppw; the library installs no logging handler on import; the CLI and facade enable logging | The library logs the numpy fallback at INFO with no handler configured — nobody has ever seen that message |
| **D34** | **Default output** | **Unchanged: full `result.h5` + preview.** Add a `--preview-only` flag and a predicted `result.h5` size line in the plan | Losing the field of a multi-hour run is unrecoverable; disk pressure is opt-in |
| **D35** | **Job format version** | **Stays `caustica-job/1`; no special error for removed kinds** | User's call. The only person who will meet the raw pydantic error is the author, while migrating the nine setups |
| **D24** | **New repository** | The **user creates an empty private repo**; the implementer fills it. It stays private until W0f's licence reading is done | Repo creation is outward-facing and not the implementer's call |
| **D25** | **Commit policy for this work** | Branch **`library-first`**, one **local** commit per workstream. **No push, `master` untouched** | A 137-file move without commits has no rollback path. Deviation from the standing no-commit rule is deliberate and scoped to this branch |
| **D26** | **Breakage window** | **A temporary break is accepted**: the nine local setups may stop working mid-extraction | Allows the clean ordering W0a → W0b → W0c → W0d. The *end* gate is unchanged: they must run bit-identically when W0 closes |

---

## 2. Framing

**The architecture is already library-first** in its solver, backend, job and result layers. Four
things are not:

1. **Packaging** (W1) — it cannot be installed and used without a checkout.
2. **The UWCEM entanglement** (W0) — the library imports a data-source-specific package.
3. **Reachability** (W5) — a stranger cannot describe their own transducer or their own medium, and
   has no schema reference to author against.
4. **Silent failure modes** (W2) — a numpy fallback nobody sees, a low-ppw warning nobody reads,
   single-threaded FFTs nobody measured.

Almost nothing here is a redesign, and any change that *feels* like a redesign is probably wrong.
Two exceptions are genuinely new contracts and should be treated as such: `medium_volume` (D16)
and the `elements` array kind (D27).

---

## 3. Ground truth (verified 2026-08-21)

**Backend / GPU**
- `src/caustica/core/backend.py` — lazy cupy import, cached CUDA probe, numpy fallback logged at
  INFO. `backend.fft` returns `scipy.fft` / `cupyx.scipy.fft` so float32 survives (numpy.fft would
  upcast and destroy parity).
- No compiled artifacts: numpy, scipy, pydantic, h5py. The k-Wave Colab failure mode is
  structurally impossible here.
- `runner.py:98 _gpu_environment(backend_name)` returns gpu_name, driver/runtime versions, cupy
  version, `vram_total_gib`, `vram_free_gib`; `{}` on non-cupy. Seed of the public `env_report()`.

**Solver loop**
- `solvers/kspace/engine.py:299 _period_boundary()` runs once per settled period (checkpoint
  cadence + graceful stop). **It returns immediately when `checkpoint is None`** (line 300–301) —
  trap T1.
- `engine.py:386` has a second stop poll at the settle→record transition.
- `kspace/linear.py:36` and `kspace/westervelt.py:40` accept `backend, record_region,
  reference_point, harmonics, checkpoint` then `raise TypeError(f"unknown run() options: ...")` —
  trap T2.
- `_NATIVE_SOLVERS = ("linear", "westervelt")` (`runner.py:71`); `runner.py:446-462` guards
  `backend=`/`checkpoint=` behind `if native:` because the kwave adapter rejects unknown kwargs —
  trap T3.

**Job / runner contracts**
- `config/job.py` — `StoredSetupJobConfig` (:498), `ExplicitJobConfig` (:513), discriminated on
  `kind`; `build_job(job, base_dir=None, with_medium=True)` (:791) is the single builder.
- `runner.py:315 run_job_file(job_path, opts)` — plan-first, disjoint exit codes
  (0 ok / 2 config / 3 OOM / 4 solver / 5 interrupted-resumable), heartbeat, stamp.
- VRAM refusal block: `runner.py:371-392`. The CPU gate belongs immediately after it.
- `_plan()` (`runner.py:222`) yields `t_expected_s`, `t_worst_s`, `vram_gib`, `steps_expected`,
  `spp`, `advice`, and `source` ∈ `db` | `calibrated` | `measured`. With `measure=True` (default)
  the estimate is **measured on the machine that will run it** — this is what makes the D20 gate
  trustworthy.

**UWCEM coupling — the thing D15 removes**
- **The import-direction rule is violated today.** `uwcem_phantoms/__init__.py` states "it consumes
  caustica, caustica never imports it", but the library imports it at four sites:
  `config/job.py:64 _require_uwcem()`, `:230-231` (`PhantomAsset` for `phantom_dataset`),
  `:618-619` (`load_setup` for `stored_setup`), `:955-956` (`dataset_dir`, `setups_dir`).
- `PhantomDatasetMediumConfig` (`job.py:216`) does two things `VolumeImportMediumConfig`
  (`job.py:193`) does not: it takes **grid shape and dx from the file** ("so an explicit job cannot
  silently run a resampled ghost of the dataset") and it loads **per-voxel property volumes** via
  `PhantomAsset`, not just a label map. Both must survive into `medium_volume` (D16).
- `caustica.materials:67 breast_default()` already exists — D17's destination is in place.
- `VolumeImportMediumConfig.build()` routes through `SceneConfig` so placement/resampling stay
  single-sourced. `medium_volume` should reuse that discipline.

**The UWCEM package (5575 lines) — what moves and what generalises**

| module | lines | nature |
|---|---|---|
| `dataset.py` | 1084 | UWCEM-specific (the nine-phantom aligned dataset) |
| `builder.py` | 712 | UWCEM-specific pipeline orchestration |
| `setup.py` | 629 | UWCEM-specific (the nine stored setups) |
| `tissue.py` | 523 | **split** — literature acoustic values are generic (D17); media-number mapping is UWCEM |
| `reader.py` | 446 | UWCEM-specific (mtype/pval ASCII decoders) |
| `processing.py` | 403 | generic label surgery ("deliberately free of physics") — stays with UWCEM for now (D16 scope) |
| `cli.py` | 402 | UWCEM-specific |
| `asset.py` | 335 | **split** — the *format* is caustica-owned and becomes `medium_volume` |
| `catalog.py` | 229 | UWCEM-specific (URLs `catalog.py:39`, IDs, ACR classes, citation `:44`) |
| `heterogeneity.py` | 220 | pval is UWCEM; synthetic correlated noise is generic |
| `spec.py` | 215 | UWCEM build recipe |
| `orientation.py` | 134 | UWCEM index-order derivation |
| `paths.py` | 103 | UWCEM-specific, and assumes a checkout (B4) |

- `apps/phantom_launcher.py` depends on UWCEM throughout (10 import sites) — it moves too.
- **137 tests move**: `test_phantoms.py` (93), `test_dataset.py` (28), `test_setup.py` (11),
  `test_phantom_launcher.py` (5) — about a third of the current 402. `test_job.py` (27) is
  partially affected and must be split, not moved.

**Data**
- `data/setups/*.json` (`caustica-setup/1`) each name a phantom `.npz` on a 560×700×480 grid, with
  `dataset_format: "hifusim-phantom-dataset/2"` (documented legacy alias). Not self-contained (D13).
- The generated 4.5 GB in `data/phantoms/` was written in that format. **`medium_volume` must read
  those existing files unchanged** — trap T7.
- `tests/test_runner.py:26 mini_job()` *is* self-contained: homogeneous water, 18×18×24 mm at
  dx 0.75 mm, bowl source, `linear`, 2–6 settle periods. Seconds on CPU.

**CI / tooling**
- `.github/workflows/ci.yml`: Linux+Windows × py3.12 plus a py3.10 Linux leg; `ruff check`,
  **`ruff format --check`**, `pytest -m "not kwave"`.
- 402 tests collect. **Use the repo venv** — the system Python has no caustica installed:
  `./.venv/Scripts/python.exe -m pytest -q`.

---

## 4. Blockers

| ID | Blocker | Severity | Fix cost |
|---|---|---|---|
| B12 | **The library imports `uwcem_phantoms`** at four sites — a licensing exposure and a violation of its own stated layering | **Critical** | Medium–Hard (W0) |
| B10 | **The GPU path has never run.** cupy is absent locally and CI has no GPU. Backend-generic code is an argument, not evidence | **Critical** | Hard (live Colab session) |
| B5 | Silent CPU fallback — a mis-set Colab runtime turns a 3-minute GPU run into an unnoticed multi-hour CPU run | High | Medium |
| B8 | No `caustica.colab`, no notebook in the repo | High | Medium |
| B4 | Phantom paths assume a checkout (cache inside the package dir; `dataset_dir()` → `<repo>/data/phantoms`) | High | Medium (in the new repo) |
| B6 | No facade — the notebook user writes ~10 lines of manual object assembly | Medium | Medium |
| B7 | No progress feedback in-notebook (`status.json` exists; nothing renders it) | Medium | Medium |
| B13 | **An outsider cannot describe their own transducer.** The job schema knows only `archimedean_spiral` and `bowl`; `TransducerArray` is generic but unreachable from a job file | **High** | Medium |
| B14 | **An outsider cannot describe their own medium.** After W0 the `medium_volume` *writer* would live in the UWCEM repo, and there is no job-schema reference or conventions document to author against | **High** | Medium |
| B9 | No packaged example that runs with zero external data | Medium | Easy |
| B1 | No console entry point | Low | Trivial |
| B2 | No `py.typed` | Low | Trivial |

---

## 5. Workstreams

Effort is in *working sessions*. Difficulty measures *uncertainty*, not typing volume.

### W0 — UWCEM extraction — **DONE (M10k, 2026-08-22)**

Executed as W0a–W0f; all gates closed with evidence (bit-identical Medium on the real dataset
file, nine setups 9/9, import-direction AST test green, licence terms recorded). The full
current state, locations, licence text and the few remaining tasks (repo publication — user's
call; maintenance rules) are consolidated in **`docs/uwcem.md`**. The detailed step plan that
used to live here was moved out when it stopped being a plan and became history.

### W1 — Packaging & distribution — **Trivial** — 0.5 session

**Files:** `pyproject.toml`, new `src/caustica/py.typed`, new `src/caustica/examples/`, CI.

- `[project.scripts]` → `caustica = "caustica.__main__:main"`. `main()` already returns an int and
  `__main__.py` ends with `raise SystemExit(main())`.
- `src/caustica/py.typed` (empty) + the `package-data` entry.
- **Packaged example (D13):** `src/caustica/examples/water_bowl_mini.json`, authored from
  `tests/test_runner.py:26`. No external files, `linear` solver, ≤ ~30 s on CPU, passes
  `caustica validate` unchanged. Add `caustica.examples.path("water_bowl_mini")` so nobody
  hand-builds `site-packages` paths.
- **The example must never be run in place** (found 2026-08-21). Relative output paths resolve
  against the *job file* (`runner.py:330-337`, trap T4), so running the packaged example directly
  would try to create `site-packages/caustica/examples/runs/<name>` — read-only in many
  environments, pollution in the rest. Ship `caustica example <name> [--to DIR]` that **copies**
  the job into the user's directory; the README quickstart runs the copy.
- **matplotlib stays an extra**, but the quickstart must say `pip install "caustica[report]"` —
  otherwise the `caustica report` step fails on a clean install. Preview writing is numpy-only and
  unaffected, and `report/figures.py:22` already raises an actionable message; the gap is purely
  the install line in the docs.
- **Clean-env wheel job in CI:** build → install into a fresh venv **from outside the repo** →
  `caustica --version`, `caustica validate <example>`, `caustica run --dry-run <example>`,
  `python -c "import caustica"`. Running from outside is the point: from inside, CWD masks a
  missing package-data entry (this is how the `gpu_db.json` bug survived).

**Acceptance:** `tests/test_packaging.py` asserts the wheel contains `py.typed`, `gpu_db.json` and
the example; the clean-env CI leg is green.

### W2 — Environment & safety policy (M10i) — **Medium** — 1.5 sessions

Common theme: **the user must not be burned silently.** Wrong backend, wrong resolution, an unseen
warning, a CPU crawling on one core — all of them look like they are working.

**Order inside this workstream is load-bearing:** `workers=-1` (D32) → recalibrate the planner's
CPU coefficients → *then* install the 5-minute gate. A threshold calibrated against
single-threaded FFTs measures nothing.

Additions beyond the original scope, all from the 2026-08-21 review:
- **`workers=-1` on every `scipy.fft` call (D32).** `workers` appears nowhere in the source today.
  Compare golden-regression fields before and after and record the measured speed-up in the devlog.
- **`warnings.warn` for critical events (D33)** — backend fallback and low ppw. No logging handler
  on import; `caustica run` and the facade enable logging at entry.
- **Low ppw gets louder, not stricter (D31)** — `job.py:944`'s warning repeats in the plan output,
  `status.json`, `run_meta.json` and at the head of the report.
- **`--preview-only` flag + a predicted `result.h5` size line in the plan (D34).** The default does
  not change: full field plus preview.



**Files:** `core/backend.py` or new `env.py`, `__init__.py`, `runner.py`.

- **`caustica.env_report() -> dict`** — promote `runner.py:98` and widen it: versions
  (python/numpy/scipy/pydantic/h5py/cupy), CUDA driver+runtime, GPU name, free/total VRAM,
  `caustica.__version__`, git commit when available, resolved backend. The runner keeps calling the
  same function so the stamp and the notebook cannot disagree. It must never raise.
- **`caustica.require_gpu(reason="")`** — returns the cupy backend or raises with an actionable
  message (D6). Branch it: on Colab (detect `google.colab` in `sys.modules` / `COLAB_*` env) name
  *Runtime → Change runtime type → GPU*, because the real failure there is almost always "not a GPU
  runtime", which no `pip install` fixes; elsewhere name `pip install cupy-cuda12x`. **No pip
  subprocess anywhere.**
- **CPU gate (D5/D20)** — immediately after the VRAM refusal at `runner.py:371-392`, mirrored in
  the facade:
  - only when the resolved backend is numpy and the solver is native (no estimate otherwise);
  - refuse when `est.t_expected_s > CAUSTICA_CPU_LIMIT_MIN * 60`, default **5 minutes**;
  - the message quotes the estimate **and** `est.source`, and names both escapes (a GPU backend, or
    `--allow-slow-cpu` / `allow_slow_cpu=True`);
  - below the threshold: run with exactly one WARNING;
  - **reuse `EXIT_CONFIG` (2)** — do not add a sixth exit code. The set is a documented contract the
    queue consumes.
- Ship the escape hatch in the same change as the gate.

**Acceptance:** `tests/test_env_gate.py` — full-size job on numpy refuses with the estimate quoted;
the packaged example still runs with one warning; `allow_slow_cpu=True` overrides; `env_report()`
returns a dict with no GPU present and does not raise.

### W3 — Facade API — **Medium** — 1.5 sessions

**Files:** new `src/caustica/facade.py`, `__init__.py`.

```python
res = caustica.simulate(
    setup="path/to/job.json",   # job path | job dict | ExplicitJobConfig | built objects
    solver="westervelt",
    backend="auto",
    harmonics=(1, 2),
    out=None,                   # None = in-memory; a path = full runner output folder
    progress="auto",            # W4; preview on by default (D21)
    allow_slow_cpu=False,       # W2
)
res.metrics      # caustica.report.metrics — the definitions focus_study uses
res.preview()    # the <=10 MB package, in memory
res.save(path)   # caustica-result/1
```

Hard constraints:

1. `simulate()` normalises every input into an `ExplicitJobConfig` and calls **`build_job`**
   (`job.py:791`). No parallel construction path. This rule is what keeps facade, CLI, queue and
   the future GUI describing the same world.
2. `out=None` does **not** disable plan-first: the planner still speaks, the gates still apply. It
   only means "write no files".
3. The return object *wraps* `SolverResult` (`solvers/base.py:76`) and the stamp; it does not
   reimplement it. `res.metrics` delegates to `caustica.report.metrics`.
4. Accepted input types are a short documented list; anything else raises `TypeError` naming the
   accepted forms. The polymorphism is where the difficulty actually lives.
5. `out=<path>` delegates to `run_job_file` — no duplicated output/stamp/resume logic (trap T4).
6. After W0c there is no `stored_setup`; `setup=` takes a job, not a setup name.

**Acceptance:** `tests/test_facade.py` — `simulate(job_dict)` and `caustica run job.json` produce
bit-identical phasor/p_max for the packaged example; an unsupported input raises usefully;
`out=None` writes nothing (assert against a tmp CWD).

### W4 — Progress hook + mid-run preview — **Easy–Medium** — 1 session

**Files:** `kspace/engine.py`, `kspace/linear.py`, `kspace/westervelt.py`, `runner.py`, new
`caustica/progress.py`, facade.

- One payload, reused everywhere (notebook, CLI, `status.json`, future GUI):
  `{period, periods_expected, step, steps_expected, peak, converge_delta, elapsed_s, eta_s, stage}`,
  `stage` ∈ `settle` | `record`. `_Heartbeat` (`runner.py:133`) already computes these — make it a
  *consumer* of the callback, not a second implementation.
- Fire from `_period_boundary()` (engine.py:299) **after fixing T1**, and once at the settle→record
  transition (engine.py:386). **Never per step** — that forces a device→host sync every step.
- Rendering lives outside the solver: `caustica.progress` picks `tqdm` when importable, else plain
  periodic lines. `tqdm` is an optional extra; Colab must work without it.
- **Mid-run preview is ON by default (D21):** every N periods (N=8, matching the default checkpoint
  cadence) render one coarse slice through the focus. One device→host copy per callback. Disable
  with `progress=None`. Keep the render itself outside the solver — the solver emits arrays, the
  renderer decides.
- **Callback exceptions must not kill a multi-hour solve** — wrap the call site, degrade to a
  warning (R9).

**Acceptance:** `tests/test_progress.py` — a CPU mini run **with no checkpoint** fires the callback
once per settled period (the T1 regression test); a `kwave` job with `progress=` set does not raise;
a callback that throws does not fail the run.

### W5 — Bring your own setup (M10m) — **Medium** — 1.5 sessions

The workstream that answers "would a stranger actually use this?". Needs W0c (it returns to
`config/job.py`).

**Files:** `config/job.py`, `arrays/`, `__main__.py`, new `docs/job_reference.md`, new
`docs/conventions.md`, README.

- **`elements` array kind (D27).** Explicit element positions + normals, inline in the job or from
  an `.npz`/`.csv`, plus element radius and focal length. `TransducerArray`
  (`arrays/transducer.py:29`) already accepts arbitrary `(n, 3)` positions and normals — only the
  schema door is missing. Keep the `derived()` pattern the spiral config uses (`job.py:271`): a
  reload re-derives geometry and the stored numbers exist to falsify a silent library change.
  Validate what the existing builders validate: matching shapes, no duplicate voxels after
  voxelization, source clear of the PML.
- **`caustica schema` (D29)** — emit the `caustica-job/1` JSON Schema, generated from the pydantic
  models. Never a hand-written second definition.
- **`docs/job_reference.md`** — every medium kind, every array kind, the drive/run/output sections,
  each with a working snippet.
- **`docs/conventions.md`** — the things that make results silently wrong when unknown: phasor
  convention `p(t)=Re{P·e^{-iωt}}` with outgoing `e^{+ikx}`; Np/m ↔ dB/cm; what `amplitude` means
  after the `2c·dt/dx` mass-source normalization; the coordinate frame (+z beam axis, apex frame);
  that the PML is part of the grid.
- **README Colab quickstart** — `pip install git+…` → packaged example → `caustica report`, no
  external data, top to bottom.

**Acceptance:** `tests/test_elements_array.py` — a job whose elements come from an `.npz` runs end
to end and its `derived()` matches on reload. `tests/test_schema_doc.py` — the kind list in
`caustica schema` output matches the headings in `docs/job_reference.md`, so the document cannot
rot silently. **Plus a manual outsider rehearsal**: in a clean environment outside the repo, using
only the README and the job reference, author and run a bowl-in-water scenario; write the steps
into the devlog. If that rehearsal needs a source dive, the workstream is not done.

### W6 — Colab bridge (revises M10f) — **Medium**, **Hard** dependency — 1.5 sessions + 1 live session

**Files:** new `src/caustica/colab.py`, new `notebooks/colab_run.ipynb`.

- `caustica.colab.run_job(...)`: environment check (`env_report()` + `require_gpu()` + planner VRAM
  estimate vs free VRAM — refuse *before* staging anything), then `run_job_file`, output under
  `/content`.
- **D22: the library never mounts or writes Drive.** No `drive.mount()`, no Drive paths, no
  Drive-aware retry logic. If the user wants persistence they mount Drive themselves and pass an
  `--out` under it — which the runner already supports. Delete the Drive clauses from the current
  M10f text rather than carrying them forward.
- **Dataset staging is gone from this module too.** After W0/D3/D18 the library has no anatomical
  data path at all; a Colab user brings a `medium_volume` file, or uses the synthetic example.
- Notebook: 4–5 cells, one editable CONFIG line, all logic imported from the package so updates
  arrive via `pip install -U` with a zero-diff `.ipynb`. Keep the M10f contract test that compares
  cell contents against a fixed template.
- Everything Colab-specific stops here (§6.3). No `google.colab` import below this module.

**The hard part is B10**, not the code: the first Colab session is the first execution of the GPU
path, ever. Budget it as its own event — it closes M7 (parity + full size), M8 (planner VRAM ±10%,
time ±25%) and this E2E at once. Arrive with everything else green.

### W7 — Queue (revises M10g) — **Hard (concurrency)** — 2 sessions

- The protocol takes a **shared folder path**; Drive is one instance and the library does not know
  about it (consistent with D22). This keeps the queue usable by an outside user and by a future
  GUI on a VM.
- The claim protocol (atomic rename + lock file with session id and heartbeat) is the single
  correctness-critical piece in this plan. Test it with simulated concurrent sessions on a local
  folder before trusting any network filesystem, whose rename semantics are weaker than POSIX.
- Runner exit codes are the queue's API: do not add or renumber them.

### W8 — GUI contract freeze (no GUI code) — **Easy** — 0.5 session

**Files:** new `docs/gui_contract.md`, new `tests/test_import_direction.py`.

- Document the exact surface a GUI may use: job schema + `validate`, run exit codes, `status.json`
  fields, preview layout (`caustica-preview/1`), result format (`caustica-result/1`),
  `env_report()`, the W4 progress payload. Anything unlisted is not a contract.
- **Import-direction test:** AST-scan every module under `src/caustica`; assert none imports `apps`,
  `uwcem_phantoms`, or `caustica_gui*`. This test currently **fails** — W0c is what makes it pass,
  so write it early and let it fail loudly until then.
- **Two contract gaps found 2026-08-21, close them here** (both small, both currently absent):
  - **Cancel signal.** `grep -rn "cancel" src/caustica` is empty; `stop_when` is time/period-based
    only. A GUI stop button has nothing to write. Add: a `cancel` file in the output folder makes
    the run stop at the next period boundary, write a checkpoint, and exit 5. The hook site already
    exists (`engine.py:299`) — only the file poll is missing.
  - **Structured errors.** Failures *before* solving (bad job, OOM refusal, checkpoint conflict)
    return `EXIT_CONFIG`/`EXIT_OOM` before the heartbeat object exists (`runner.py:315-392`), so no
    `status.json` is written at all and a GUI is left parsing stderr. Add `error.json`
    (`{stage, exit_code, error_class, message, advice[]}`) to the output folder. The planner already
    produces `est.advice`; today it is only printed.
- Do not pick a framework, add a `gui` extra, or create the second repo (D7/D8).

### W9 — Docs & examples — **Easy**, tedious — 1 session

- README: a Colab quickstart running top-to-bottom on the packaged example with **no external
  data**; a `medium_volume` section explaining how to bring your own volume; a pointer to the
  `uwcem-phantom` repo for that particular source.
- `examples/`: water bowl mini (seconds, CPU), layered tissue (minutes, CPU), medium_volume (GPU).
- Docstring pass over the newly public surface. `__all__` must list exactly what is documented.

### W10 — CI — **Easy** — 0.5 session

- The matrix is already correct (D14) — do not touch it.
- Add the clean-env wheel job (W1) and a `network` marker; run `-m "not kwave and not network"`.
- No GPU runner exists. Until B10 closes, GPU claims in the README stay explicitly marked
  unverified. Do not imply coverage that does not exist.

---

## 6. Target architecture

```
L5  caustica-gui              separate repo — talks ONLY to L3 artifacts
L4  caustica.colab            environment check + run under /content
L3  job JSON + CLI            caustica-job/1, runner, status.json, preview, exit codes
L2  facade                    caustica.simulate(...)
L1  objects                   Grid / Medium / Source / Solver / CWRunSpec
L0  core                      Backend, spectral ops, PML, materials, io, medium_volume

    uwcem-phantom (separate repo)  ──depends on──▶  caustica
    emits: medium_volume files, explicit job JSON
```

1. **Arrows point down only.** `caustica.*` never imports `apps`, `uwcem_phantoms`, or a GUI
   package. Enforced by W8's test — which fails today (B12) and passes after W0c.
2. **L2 is not a second code path.** The facade routes through `build_job`.
3. **L4 may be opinionated; L0–L3 may not.** Colab assumptions live in `caustica.colab` only — and
   after D22 that module's opinions stop at `/content`.
4. **L5 gets no special API.** If the GUI needs something, it becomes a documented L3 contract first.
5. **One door for volume media.** Every phantom source enters through `medium_volume`. If a source
   needs a special case in the library, the special case is wrong.

---

## 7. Traps (read before coding)

**T1 — `_period_boundary()` returns early with no checkpoint.** `engine.py:300-301` is
`if checkpoint is None: return`. A progress callback added naively there never fires for facade
runs — exactly the notebook case W4 exists for. Make progress and checkpointing independent
concerns sharing one call site.

**T2 — The native solvers reject unknown kwargs on purpose.** `linear.py` and `westervelt.py` both
`raise TypeError`. A new `progress=` must be declared in both and threaded through
`run_cw_kspace_pstd`. Three files, one change.

**T3 — Never forward native-only kwargs to the kwave adapter.** `runner.py:446-462` guards
`backend=`/`checkpoint=` behind `if native:` because the adapter rejects unknown kwargs by contract.
This exact bug (every kwave job crashing) was found in the M10c adversarial round. `progress=` goes
behind the same guard.

**T4 — Relative output paths resolve against the job file, not the CWD.** `runner.py:325-340`
explains why: CWD resolution makes `--resume` from another directory silently restart from period 0.
The facade must preserve this when delegating.

**T5 — Do not re-download the phantom archives.** `uwcem_phantoms/_data` holds already-fetched UWCEM
files. W0e's resolution order must find an existing checkout `_data` before the user cache.

**T6 — `report/__init__.py` is lazily wired (PEP 562)** so matplotlib and h5py stay optional. A new
eager import there breaks preview-writing on machines with neither.

**T7 — `medium_volume` must read the existing 4.5 GB unchanged.** The local dataset was written as
`caustica-phantom-dataset/2` (legacy alias `hifusim-phantom-dataset/2`). A format change that forces
a rebuild is a failed change, not a migration.

**T8 — W0 removes a third of the test suite from this repo.** 137 tests move to `uwcem-phantom`.
Expect the local count to drop from 402 to ~265 and do not read that as regression — but *do* verify
each moved test still runs in its new home before deleting it here.

---

## 8. Difficulty ledger

**Easy — about one day, low-risk**
Entry point, `py.typed`, packaged example + `caustica example`, `env_report()`, `caustica schema`,
README quickstart, GUI contract doc, import-direction test, CI wheel job.

**Medium — ~7 sessions**
Environment & safety policy (W2, 1.5), bring-your-own-setup (W5, 1.5), facade (W3, 1.5), progress +
preview (W4, 1), Colab module (W6, 1.5). Well-understood; the risk is drift and double code paths,
not feasibility. Mitigation is the same in all of them: route through the existing single sources of
truth — `build_job`, `report.metrics`, `_period_boundary`, the planner, `run_job_file`.

Two items inside W2 look trivial and are not: `workers=-1` invalidates the planner's CPU
calibration (redo it, in that order — D32), and moving critical events to `warnings.warn` changes
what every existing test sees on stderr.

**Hard — the real unknowns**
1. **W0, the UWCEM extraction.** Breadth, not depth: 5575 lines, 137 tests, a new repo, a new
   format contract, and a live local workflow that must not break. Most likely place for a silent
   regression.
2. **B10, the GPU path.** fp32 parity, cuFFT on padded non-power-of-two shapes, VRAM headroom
   against the planner's model — all inference today. Cannot be retired locally.
3. **Queue concurrency (W7)** over a filesystem with non-POSIX rename semantics.
4. **Long Colab runs** — 12 h ceiling, disconnects. Mitigated by checkpoint/resume and
   `--max-hours`; unproven, and now without a Drive safety net (D22 → R10).

---

## 9. Risk register

| ID | Risk | Mitigation |
|---|---|---|
| R11 | **W0 silently changes physics** — a moved tissue value, a re-derived orientation, a subtly different medium build | Bit-identity gates in W0a and W0d: same file → same `Medium`; nine local setups still run and match |
| R12 | **UWCEM licensing** — redistribution terms unclear | W0f: read the terms before the new repo goes public; the plan assumes *nothing* is redistributed and the user fetches from upstream; citation already wired (`catalog.py:44`) |
| R4 | fp32 CPU/GPU divergence | The M7 parity suite is the gate; claim no GPU support until it passes |
| R5 | Facade and job path diverge | Facade must call `build_job`; bit-identity test in W3 |
| R7 | The CPU gate refuses on a bad estimate | Low: with `measure=True` (default) the estimate is measured on the running machine and labelled `measured`. Ship `allow_slow_cpu` in the same change; quote `est.source` |
| R10 | **`/content`-only output (D22): a Colab crash loses the run** | Checkpoint/resume already survives a restart *within* a session but not a VM teardown. Document plainly: for long runs pass `--out` under a Drive folder you mounted yourself. Do not add Drive code to the library |
| R3 | cupy/CUDA drift on Colab | Never auto-install (D6); probe at runtime; distinguish "no GPU runtime" from "cupy missing" |
| R6 | GUI dependencies creep into the core | Separate repo (D8) + import-direction test (W8) |
| R8 | Queue double-runs a job | Claim-race test with simulated concurrent sessions before live use |
| R9 | A progress callback raising kills a multi-hour solve | Wrap the call site; degrade to a warning |
| R13 | Job-schema collisions | W0c, W5 and W3 all touch `config/job.py`. The order is fixed: W0c → W5, and W3 only next round. W1/W2 never touch that file |

---

## 10. Sequencing

Two chains run in parallel (D19):

**Current handoff (D23):** W1 (M10h) → W2 (M10i) → W0a…W0f (M10k) → W5 (M10m).
W1/W2 collide with nothing and could also run alongside W0; W5 must follow W0c.

**W3 facade and W4 progress (M10j) wait** — W3 must land after W0c or it gets written twice (R13),
and W4 is cheapest paired with it.

Then **M10j (W3 + W4)** → **W8 freeze (M10l)** → **Phase B: first Colab session (W6)** →
**Phase C: W7 queue, W9 docs**.

Note on GPU (D30): parity stays unverified until that Colab session. That is a deliberate choice —
it means everything shipped before it is CPU-honest, and the README must say so.

W8's import-direction test is written **first**, deliberately failing, as the gate that proves W0c
landed. W10 can go any time.

**Critical path:** W0a → W0c → W3 → W6 → (live session) → W7.

---

## 11. `MILESTONES.md` changes — **APPLIED 2026-08-21**

These edits are already in `MILESTONES.md`: M10f's dataset-staging and Drive clauses were deleted,
M10g became the shared-folder protocol, and **six** milestones were added after M10e —

| milestone | workstream | in this handoff? |
|---|---|---|
| **M10h** Kütüphane paketleme | W1 | yes |
| **M10i** Ortam ve güvenlik politikası | W2 | yes |
| **M10k** UWCEM ayrışımı | W0 | yes |
| **M10m** Dışarıdan kullanılabilirlik | W5 | yes |
| **M10j** Facade + ilerleme | W3 + W4 | no — next round |
| **M10l** GUI sözleşmesi | W8 | no — next round |

Environment policy and facade are separate milestones on purpose: M10i touches only `validate`'s
reporting and can close in parallel with the extraction, while M10j touches the job schema and
cannot. The sequencing line and the live-status section were updated too. What follows is kept as
the rationale record — the authoritative criteria live in `MILESTONES.md`.

The full milestone text is no longer duplicated here — it drifted once already (M10i was split into
M10i + M10j after this section was written) and a stale copy of a criteria list is worse than none.
`MILESTONES.md` is the single source; this document holds the *reasoning*, that one holds the
*criteria*.

## 12. Working agreement (binding)

- **Code and docstrings in English; devlog and milestone text in Turkish.** This document is the
  documented exception (D12).
- **`ruff format` is enforced by CI** (`ruff format --check src tests`).
- **Tests encode milestone criteria.** A milestone closes on a test that would fail if the criterion
  regressed, not on code that appears to work.
- **Commits:** the standing project rule is "never commit without the user asking". For this work
  it is scoped-overridden by D25: commit locally on the `library-first` branch, one per sub-step.
  **No push, no `master`.** Outside this branch the standing rule applies unchanged.
- Run tests with the repo venv: `./.venv/Scripts/python.exe -m pytest -q`.
- When a criterion cannot be met, record it in `MILESTONES.md` with the reason. A negative result
  recorded honestly is a valid outcome; a silently narrowed criterion is not.
- **W0 touches physics-adjacent code.** Every move in it is a *move*, not a rewrite: if a value, a
  formula or an axis convention changes, that is a bug, and the bit-identity gates exist to catch it.

---

## 13. Out of scope

GUI implementation, PyPI publication (v0.1 per D2), any distribution of anatomical data (D3/D18),
API stability guarantees (D10), Drive integration in the library (D22), and every milestone from M11
onward. This plan ends where the library is UWCEM-free, installable, importable, runnable on Colab,
and contract-complete for a GUI that does not exist yet.

## 14. Open questions

Every question raised during planning is decided (D1–D26). What remains does not block W0:

1. **When `uwcem-phantom` goes public** — it is created private (D24); W0f's licence reading is the
   gate. The name itself is still a working name and can change before it is public.
2. **Whether `processing.py` / `heterogeneity.py`'s generic halves eventually come back into
   caustica** as a `caustica.volumes` toolkit. Deliberately deferred: extract first, generalise
   later if a second phantom source appears. Do not pre-build it.
3. **`medium_volume` as the final kind name.** Chosen for being source-neutral and descriptive;
   renaming later is cheap while v1.0 is far off (D10), so it is not worth blocking on.
