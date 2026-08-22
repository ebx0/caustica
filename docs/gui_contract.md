# The GUI contract

caustica has no GUI, no GUI dependency and no `gui` extra, and this document
does not propose one. It exists because a GUI will eventually be written —
in a separate repository, in a technology nobody has picked yet (PLAN.md
§11 / D13) — and it must sit on a surface that is stable *before* it starts,
not one discovered by reading the library's internals.

So this page lists that surface: the file formats a GUI reads and writes, the
exit codes it routes on, the fields it may display, and the two signals it may
send. Every list here is either generated from the code or pinned by
`tests/test_gui_contract.py`, which fails if the code and this page disagree.

**Nothing that is not listed here is a contract.** Anything else in the
library — module layout, private helpers, log text, stdout prose, the shape of
`checkpoint.npz` — may change in any release without notice. If a GUI needs
something that is not on this page, the fix is to add it here first.

**What on this page is machine-checked, and what is not.** The bullet lists,
the tables, and the literal names, numbers and format strings on this page are
compared against the running code by `tests/test_gui_contract.py`; they cannot
rot silently. The prose between them is explanatory — reviewed, not executed.
Every finding of the M10l review, and one of the mutation round that followed
it, was a sentence rather than a list, which is the class a list comparison
cannot reach; where such a sentence makes a claim precise enough to test, the
answer is to add the test, not to trust the sentence. **If prose and a list
disagree, the list is the contract.**

The layering is enforced too: `tests/test_import_direction.py` AST-scans every
module under `src/caustica` and fails if any of them imports `apps`,
`uwcem_phantoms`, or a `caustica_gui*` package. Arrows point down only. A GUI
imports caustica; caustica never learns that a GUI exists.

---

## The shape of the whole thing

**Input is ONE file.** A `caustica-job/1` JSON document describes a run
completely. The single exception is a voxel medium, which references a volume
file by path — a 4.5 GB array does not belong inside a JSON document.

**Output is ONE folder.** Deliberately a folder and not a single file: resume
and live progress cannot work through one file. What a GUI needs in order to
*show* a finished run is nevertheless one small package — `preview.npz` plus
`metrics.json`, where the preview is budgeted at ≤ 10 MB and the metrics are a
couple of KB — so a remote run stays viewable over a slow link without
downloading a multi-GB `result.h5`.

```
caustica run job.json --out <folder>      # one job file in, one folder out
caustica validate job.json                # exit 0/2, before anything expensive
caustica schema                           # JSON Schema for caustica-job/1
caustica run job.json --out <f> --dry-run # plan.json only: VRAM, time, advice
caustica report <folder>                  # renders the preview package
```

The same three things are available in-process, with the same semantics:
`caustica.simulate(...)`, `caustica.config.job.validate_job(...)`,
`caustica.config.job.job_schema()`.

---

## Input: `caustica-job/1`

- **Schema.** `caustica schema` prints a JSON Schema (draft 2020-12) for the
  whole job format, with `additionalProperties: false` and discriminated
  unions for the medium and array kinds. It is generated from the same pydantic
  models the runner validates against, so an auto-generated form cannot drift
  from what the runner accepts. `caustica schema --kinds` lists the registered
  medium and array kind tags — including any a plugin added.
- **Validation.** `caustica validate job.json` exits 0 or 2 and prints a
  report; `validate_job(path)` returns a `JobReport` with `.ok`, `.errors`,
  `.warnings`, `.summary`, `.render()`. It checks everything that can be
  checked *without solving and without a GPU*: schema, referenced files,
  source-vs-PML clearance, focus placement, points-per-wavelength.
  `--fast` skips building the medium.
- **Field reference.** `docs/job_reference.md` documents every kind with a
  runnable snippet, and `tests/test_schema_doc.py` keeps it honest.

---

## Output: the run folder

### Output folder: written by every successful run

- `job.json` — the normalized copy of the job that actually ran
- `plan.json` — the planner's verdict, written BEFORE solving
- `plan.txt` — the same verdict as printed text
- `status.json` — the live heartbeat (see below)
- `result.h5` — the field, as `caustica-result/1`
- `preview.npz` — the ≤ 10 MB quick-look package, `caustica-preview/1`
- `metrics.json` — focal metrics, `caustica-metrics/1`
- `run_meta.json` — the stamp: environment, git commit, planner vs actual

With `--preview-only` there is no `result.h5`; the preview package is the
output. A non-native solver (`kwave`) writes no `plan.json`/`plan.txt`,
because the planner models the native engine only.

### Output folder: written only under some conditions

- `checkpoint.npz` — in-run state; present while a run is interrupted, removed
  on success. Its internal format is NOT contract: treat it as opaque
- `error.json` — present if and only if this folder's last attempt FAILED
- `cancel` — an input, not an output: a caller creates it to stop the run

---

## Exit codes

The exit-code set is the queue's API. It is disjoint so that a caller can
route without parsing text, and it is **closed** — a new failure mode reuses an
existing code rather than adding a sixth.

| code | constant | meaning |
| --- | --- | --- |
| `0` | `EXIT_OK` | success, or the folder already held a complete result |
| `2` | `EXIT_CONFIG` | config error: bad job, unknown backend/GPU, checkpoint conflict, CPU-time refusal |
| `3` | `EXIT_OOM` | refused before solving: the run does not fit in memory |
| `4` | `EXIT_SOLVER` | the solve or the store failed |
| `5` | `EXIT_INTERRUPTED` | stopped cleanly and resumably (`--max-hours`, or a `cancel` file) |

One code is deliberately absent from the table: **1**. It is what Python
returns when an exception escapes the runner's own classification — a bug, not
a verdict, and it comes with a traceback and no `error.json`. A caller that
sees 1 should report it, not retry it. (`caustica report` and
`caustica example` also use 2 for their own argument errors.)

Exit 5 is not a failure: a checkpoint is on disk and `--resume` finishes the
run. A resumed run reproduces the uninterrupted one bit for bit on one backend
(the documented band is rel < 1e-6).

In-process the same codes arrive as ONE exception carrying the same number:
`caustica.SimulationError`, with `.exit_code`. A pre-run gate raises it with
code 2 or 3, a failed solve with code 4 — the classification lives in the
attribute, not in the class hierarchy.

---

## `status.json` — the live heartbeat

Rewritten atomically at most every `--status-interval` seconds while solving,
and once on each terminal state. This is the file to poll: it survives a Drive
sync, needs no open connection, and its numbers are the ones M8's Colab gates
were calibrated against.

`state` is one of `solving`, `writing`, `done`, `interrupted`, `failed`.

### status.json: fields in every heartbeat

- `job` — the job's name
- `solver` — the solver that is running
- `backend` — the backend that was actually resolved
- `pid` — the process id writing this file
- `ppw_warnings` — low-resolution warnings, repeated here on purpose
- `state` — see above
- `periods_done` — acoustic periods completed (resume offset included)
- `steps_done` — derived: `periods_done * spp`
- `steps_expected` — the planner's expected step count (may be null)
- `steps_worst` — the worst-case step count (may be null)
- `eta_s` — seconds remaining, measured from THIS session's cadence
- `elapsed_s` — seconds since this session started solving
- `written_at` — UTC ISO-8601 timestamp of this write

### status.json: state-dependent extras

- `result` — `state: "done"` only: the path of the final artifact
- `detail` — `state: "interrupted"` only: why it stopped
- `error` — `state: "failed"` only: the classified failure text

Known and accepted: the period boundary fires once more just before the record
window, so a run that reaches recording reports `periods_done` one higher than
its settle count. Status numbers are telemetry, not provenance — the exact
counters live in `run_meta.json`.

---

## `error.json` — why a run failed

Written to the output folder on every non-zero exit that has a folder to write
into, including the failures that happen *before* solving starts and therefore
before `status.json` exists. It is an **addition** to the failure contract, never
a replacement: the exit code and the stderr message are unchanged and stay
authoritative. A GUI never has to parse stderr.

```json
{
  "format": "caustica-error/1",
  "stage": "gate",
  "exit_code": 3,
  "error_class": "VramRefusal",
  "message": "REFUSED before solving: this run needs 41.20 GiB but free device VRAM (NVIDIA A100-SXM4-40GB) is 38.88 GiB.",
  "advice": ["coarsen dx to 0.30 mm", "or record only the focal region"],
  "written_at": "2026-08-22T16:19:34+00:00"
}
```

### error.json: fields

- `format` — always `caustica-error/1`
- `stage` — where it failed; see below
- `exit_code` — the same code the process returned
- `error_class` — the exception type, or a gate's name (`VramRefusal`,
  `CpuTimeRefusal`, `CheckpointConflict`) — this is what a GUI switches on
- `message` — one human-readable line
- `advice` — a list of actionable strings, possibly empty. For a memory
  refusal these are the planner's own suggestions, not generic prose
- `written_at` — UTC ISO-8601 timestamp

### error.json: stages

- `config` — the job would not load or build; nothing was planned
- `plan` — the planner itself failed (e.g. an unknown `--gpu` name)
- `gate` — a pre-run gate refused: memory, or CPU wall time
- `checkpoint` — a checkpoint exists and `--resume` was not given
- `solve` — the solver raised
- `store` — the solve finished and writing the result failed

Two rules a GUI may rely on:

1. A **successful** run writes no `error.json`, and a new attempt on a folder
   deletes the previous one. Its presence means "this folder's last real
   attempt failed".
2. An **interrupted** run (exit 5) writes no `error.json` either. Stopping on
   request is not failing; `status.json` says `interrupted` and the exit code
   says 5.
3. `--dry-run` touches it in neither direction: a fit-check is a probe, not an
   attempt, so it never deletes an existing record and never writes one.

The converse does **not** hold, and a GUI must not assume it: a process that is
killed (the ordinary Colab session death) writes nothing at all, and leaves
`status.json` reading `solving` with no process behind it. Absence of
`error.json` means "no *classified* failure was recorded", not "the run is
fine". The exit code, when there is one, is the authority.

There is one case with nowhere to write: the job would not load *and* no
explicit `--out` was given, so the folder's own name — which comes from the
job — was never known. Nothing is invented in that case; the stderr message
and exit code are all there is. A GUI that always passes `--out` never meets
it.

---

## `cancel` — stopping a run without killing it

A GUI's Stop button creates an (empty) file named `cancel` in the run's output
folder. Nothing else is required, and no signal, socket or process handle is
involved.

- The solve polls for it **once per acoustic period boundary** — one `stat`,
  never per step. A per-step poll would put a filesystem round-trip between
  GPU kernels. (There is one extra poll just before the record window, so the
  count is periods + 1.) The poll asks whether `cancel` is a *file*: a
  directory of that name is not a request and is ignored.
- On seeing it the run writes a checkpoint, stops, and exits **5**.
- The runner then **deletes** the file, so a stopped folder never advertises
  a stop request nobody will honor: once the process has exited, `cancel` is
  gone. This is belt-and-braces rather than the load-bearing part — the clear
  at the start of every real run (next bullet) already carries the `--resume`
  on its own, measured with the handler's delete removed — but a GUI may rely
  on both. The resumed run finishes bit-identically to an uninterrupted one.
- A `cancel` left behind by a process that was killed before it could honor it
  is cleared when the next attempt on that folder starts, rather than
  cancelling every resume forever. This clear is the one that matters, and it
  is what makes `--resume` safe after any kind of death. The consequence is
  honest: `cancel` is a signal to a run that is *already going*, not a way to
  pre-cancel one.
- Only the native solvers (`linear`, `westervelt`) can be cancelled, because
  only they take checkpoints. A `kwave` job says so on stdout and ignores the
  file — stopping it would lose the run rather than pause it, which is the
  opposite of what the file is for.

`--max-hours` is the same stop on a timer, and produces the same exit 5.

---

## Planning without running: `--dry-run` and `plan.json`

`caustica run job.json --out <folder> --dry-run` exits 0 having written
`job.json`, `plan.json` and `plan.txt` and nothing else — no solve, no result,
no status. This is how a GUI answers "will this fit, and how long will it
take?" before committing a GPU. A non-native solver writes only `job.json`:
there is nothing to plan, so there is no `plan.json` to read.

Two things a caller must expect. A dry run exits **0** when the run fits — but
a memory refusal still exits **3**, because "it does not fit" is the answer to
the question, not a malfunction. (The CPU-time gate is different: under
`--dry-run` it prints its verdict and still exits 0, since planning a
Colab-bound job from a laptop is a normal thing to do.) And a dry run writes
neither `error.json` nor anything else beyond the plan — it will not disturb
the record of a real run in the same folder.

### plan.json: fields

- `source` — where the timing came from: `db`, `calibrated` or `measured`
- `spp` — steps per acoustic period
- `dt_s` — the timestep
- `t_step_s` — seconds per step
- `warmup_s` — the ONE-TIME cost before steady stepping (cuFFT plans, kernel
  compilation, first allocations). `t_expected_s = warmup_s + steps_expected *
  t_step_s`, so a short GPU run can be mostly warmup and still be healthy
- `steps_expected` — expected step count
- `steps_worst` — worst-case step count
- `t_expected_s` — expected wall time on THIS machine
- `t_worst_s` — worst-case wall time
- `vram_gib` — device memory this run needs
- `vram_breakdown_bytes` — what that memory is spent on
- `result_size_mb_expected` — how big `result.h5` will be
- `gpu` — the datasheet GPU the second estimate targets
- `gpu_t_expected_s` — expected wall time on that GPU
- `gpu_fits` — whether it fits on that GPU
- `warnings` — planner warnings
- `advice` — actionable suggestions (the same strings `error.json` carries)
- `ppw_warnings` — low-resolution warnings

A run that does not fit is refused with exit 3 *before* anything expensive
happens, and the same `advice` list is written to `error.json`.

---

## `caustica-preview/1` — the ≤ 10 MB quick-look

`preview.npz` is the package a GUI displays. It holds peak-plane slices per
harmonic, one `p_max` plane, a block-mean coarse amplitude volume, mm axes
measured from the transducer apex, the convergence history, and a `meta_json`
entry carrying the format tag, grid geometry, the realized peak voxel and the
coarsening step.
The 10 MB budget is *measured* on the compressed bytes, not estimated.

Read it with `caustica.report.preview.load_preview(path)`; render it with
`caustica report <folder>`. In-process, `simulate(...).preview()` returns the
identical package without touching disk.

`metrics.json` is the numeric companion, `caustica-metrics/1`.

### metrics.json: top-level fields

- `format` — always `caustica-metrics/1`
- `job` — the job's name
- `generated` — UTC ISO-8601 timestamp
- `peak` — peak pressure and where it is
- `focal_spot` — the −6 dB focal spot
- `target` — the requested focus, and the miss distance
- `run` — solver/steps/convergence provenance

---

## `caustica-result/1` — the field itself

An HDF5 file. Root attributes carry the format tag, caustica version, solver,
backend, `f0_hz`, `dt_s`, `spp`, `steps_total`, `t_end_s`, `tof_periods`,
`converged_period`, `settle_capped`, `harmonics`, `dx_m`, `grid_shape`,
`pml_vox`, `region_start`, `region_stop`, plus the runner's stamp
(`job_name`, `job_kind`, `git_commit`, `runner`, `apex_vox`, `focus_vox`) and
two conventions in prose: `phase_convention` and `absorption_model`.

`input/` holds the source geometry and drive; `output/` holds
`p_real_h{n}` / `p_imag_h{n}` per harmonic and `p_max`, each with the `scale`
attribute needed to restore it from float16; `convergence/history` holds
`(period, peak_pa, rel_change)` rows.

Amplitude and phase are never stored — they are always recomputed. Read the
file with `caustica.io.store.load_result(path)`; check one with
`validate_result_file(path)`.

---

## `env_report()` — what machine is this

`caustica.env_report()` never raises, on any machine, with any CUDA stack in
any state. The runner stamps its output into `run_meta.json` and a notebook
prints the same function's output, so the two cannot disagree.

### env_report(): keys on every machine

- `caustica` — the library version
- `python` — the interpreter version
- `platform` — the platform string
- `numpy` — the numpy version
- `scipy` — the scipy version, or null
- `pydantic` — the pydantic version, or null
- `h5py` — the h5py version, or null
- `resolved_backend` — the backend this run actually resolved (for a bare
  `env_report()` call, what `auto` would pick here); on a failed probe it
  reads `probe_error: <ExceptionType>` instead of a backend name

### env_report(): keys added on the cupy backend

- `gpu_name` — the device name
- `driver_version` — the CUDA driver version
- `cuda_runtime_version` — the CUDA runtime version
- `cupy_version` — the cupy version
- `vram_total_gib` — total device memory
- `vram_free_gib` — free device memory; this is what the memory gate uses

A failed GPU probe replaces those six with a single `gpu_probe_error` string
instead of raising.

`caustica.require_gpu()` returns the cupy backend or raises with the fix for
*this* machine. It never calls pip.

---

## The progress payload

One dict is emitted per acoustic period boundary — never per step, because a
per-step device→host sync destroys GPU throughput. Pass a callable as
`progress=` to `caustica.simulate(...)` or `RunnerOptions`, or the strings
`"auto"` / `"plain"` for the built-in console display.

### Progress payload: keys

- `period` — periods completed
- `periods_expected` — how many are expected in total
- `step` — steps completed
- `steps_expected` — how many are expected in total
- `peak` — this period's peak pressure, Pa
- `converge_delta` — relative change against the previous period
- `elapsed_s` — seconds since this session started
- `eta_s` — seconds remaining, or null
- `stage` — `settle` or `record`
- `snapshot` — a zero-argument CALLABLE returning a 2-D slice through the focus

The first nine keys are the serializable contract. `snapshot` is the tenth and
is **not** serializable: a consumer that serializes the payload — a socket, a
`status.json` writer — MUST drop it. It is lazy on purpose, so a consumer that
draws a preview every eight periods pays exactly one device→host copy on those
periods and none on the others.

A consumer that raises does not kill the solve; the engine warns once and
carries on.

---

## `run_meta.json` — the audit stamp

### run_meta.json: top-level fields

- `format` — always `caustica-run-meta/1`
- `job` — the job's name
- `job_kind` — the job kind that ran
- `solver` — the solver that ran
- `backend` — the backend that was resolved
- `generated` — UTC ISO-8601 timestamp
- `git_commit` — which caustica ran: the checkout's commit when the package
  is imported from a git work tree, otherwise the commit frozen into the
  package at build time (so a wheel install — Colab — is traceable too),
  and `unknown` only when neither exists
- `environment` — `env_report()` output
- `ppw_warnings` — low-resolution warnings
- `planner` — the `plan.json` payload, or null for a non-native solver
- `actual` — measured wall time, steps, convergence, resume offset, VRAM peak.
  Inside it, `t_step_measured_s` is `elapsed / steps` and therefore bundles
  the one-time warmup into a per-step average; `t_step_steady_s` is the
  median rate BETWEEN period boundaries and `warmup_s` is what the total
  does not explain. Both are `null` on a run too short to measure them
  (fewer than three period boundaries). No existing key changed meaning
- `derived` — re-derivable geometry (apex voxel, f-number, …)

---

## Format identifiers

| format | written by |
| --- | --- |
| `caustica-job/1` | the input job document |
| `caustica-result/1` | `result.h5` |
| `caustica-preview/1` | `preview.npz` |
| `caustica-metrics/1` | `metrics.json` |
| `caustica-run-meta/1` | `run_meta.json` |
| `caustica-error/1` | `error.json` |
| `caustica-checkpoint/1` | `checkpoint.npz` (opaque; listed so a stale one is recognizable) |

---

## Explicitly not contract

- The Python module layout below the documented entry points, and every
  underscore-prefixed name.
- stdout prose. stderr messages are stable enough to show a user, but a GUI
  routes on the exit code and `error.json`, never on the text.
- The internal structure of `checkpoint.npz`, and `plan.txt`'s layout.
- Log records and their wording.
- Any framework, transport, socket or IPC mechanism. There is none, on
  purpose: the contract is files in a folder, which works identically for a
  local run, an SSH'd machine and a Colab session writing into a Drive folder
  the *user* mounted. caustica never mounts one itself.
- `caustica.colab` (M10f). It is a convenience entry point for a human in a
  notebook: it prints an environment verdict, requires a GPU, picks a default
  folder under `/content` and hands the runner's own failures back readably.
  Every decision in it is an opinion about where it is running, which is
  exactly what a contract must not be. It adds nothing a program cannot get
  from `caustica run` / `caustica.simulate` plus this page, and a driving
  program should use those.
