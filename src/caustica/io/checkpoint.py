"""In-run checkpoints: a killed session loses periods, not the run.

A standard stored-setup solve is ~3 minutes on an A100, but finer dx and
queued dataset runs stretch a Colab session into hours, and Colab sessions
die mid-run. File-level resume (skip finished files) cannot help the run
that was in flight — this module can: the engine writes its complete time-
stepping state every N acoustic periods, atomically, and a rerun of the SAME
call continues from the last checkpoint instead of from silence.

Correctness contract: a resumed run reproduces the
uninterrupted run's phasor to rel < 1e-6. On one backend this holds trivially
— the saved state is bit-exact (float32 arrays through uncompressed ``.npz``,
scalars through JSON) and the step loop is deterministic — so the resumed
trajectory is the SAME floating-point trajectory, not a nearby one.

Safety contract: a checkpoint is only ever resumed into the run that wrote
it. The engine fingerprints everything that shapes the trajectory (grid,
medium hash, source hash, dt/spp, spec, solver, backend, record region) and
:func:`load_checkpoint` refuses on any difference, naming the keys that
moved. A library upgrade that changes the numerics should bump the engine's
fingerprint ``scheme`` tag so stale checkpoints are refused rather than
silently continued.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from caustica.io.atomic import atomic_write

log = logging.getLogger("caustica")

CHECKPOINT_FORMAT = "caustica-checkpoint/1"

#: Engine stages a checkpoint can be taken in. "settle" covers both the
#: convergence loop and the t_end padding loop (the saved counters encode
#: which); "record" is the snapshot taken right before the record window, so
#: a kill DURING recording resumes by redoing the (short) window from its
#: exact start state.
STAGES = ("settle", "record")


class RunInterrupted(RuntimeError):
    """Raised by the engine when ``stop_when`` asked for a graceful stop.

    The run state is already on disk at ``checkpoint_path``; rerunning the
    same call with the same :class:`CheckpointSpec` continues it.
    """

    def __init__(self, checkpoint_path: Path, stage: str, periods_done: int, steps_done: int):
        super().__init__(
            f"run interrupted on request at period {periods_done} (stage '{stage}', "
            f"{steps_done} steps); state saved to {checkpoint_path} — rerun the same "
            f"call to resume."
        )
        self.checkpoint_path = Path(checkpoint_path)
        self.stage = stage
        self.periods_done = periods_done
        self.steps_done = steps_done


class CheckpointMismatch(RuntimeError):
    """The checkpoint on disk belongs to a DIFFERENT run than the one asked."""


@dataclass
class CheckpointSpec:
    """Checkpoint policy for one engine run.

    Attributes
    ----------
    path:
        Checkpoint file (``.npz``). If it exists when the run starts, the
        run RESUMES from it (after a fingerprint check); it is deleted when
        the run completes.
    every_periods:
        Write cadence, in acoustic periods. The trade is plain: a checkpoint
        write costs one host round-trip of the full field state (4 padded
        float32 volumes in 3-D), and a kill loses at most this many periods.
    stop_when:
        Optional callable polled once per period boundary (and once more just
        before the record window). When it returns True the engine writes a
        checkpoint and raises :class:`RunInterrupted` — this is the graceful
        session-budget stop (the runner's ``--max-hours``). It is NOT polled
        inside the record window, which is only ``n_record_periods`` long.
    """

    path: Path
    every_periods: int = 8
    stop_when: Callable[[], bool] | None = field(default=None, compare=False)
    #: When True the engine does NOT delete the checkpoint on completion —
    #: the CALLER owns cleanup and removes it only after the result is safely
    #: stored, so a post-solve store failure stays resumable from the
    #: pre-record snapshot instead of discarding the whole solve
    #: (adversarial review, 2026-08-19).
    keep_on_success: bool = False

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.every_periods < 1:
            raise ValueError(f"every_periods must be >= 1, got {self.every_periods}")


def arrays_sha1(*arrays: np.ndarray) -> str:
    """Order-sensitive SHA-1 over the raw bytes of the given arrays."""
    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _fingerprint_json(fingerprint: dict[str, Any]) -> str:
    return json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))


def write_checkpoint(
    path: str | Path,
    *,
    stage: str,
    n: int,
    periods_done: int,
    prev_peak: float | None,
    converged: bool,
    converged_period: int,
    history: list[tuple[int, float, float]],
    fields: dict[str, np.ndarray],
    fingerprint: dict[str, Any],
) -> Path:
    """Persist one engine state atomically (uncompressed ``.npz``).

    Uncompressed on purpose: the fields are near-random float32 volumes that
    deflate poorly, and the write happens inside the solve loop where seconds
    matter. Written through :func:`~caustica.io.atomic.atomic_write`, so a
    kill mid-write leaves the previous checkpoint intact.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    path = Path(path)
    meta = {
        "format": CHECKPOINT_FORMAT,
        "stage": stage,
        "n": int(n),
        "periods_done": int(periods_done),
        "prev_peak": None if prev_peak is None else float(prev_peak),
        "converged": bool(converged),
        "converged_period": int(converged_period),
        "fingerprint": fingerprint,
    }
    hist = np.asarray(history, dtype=np.float64).reshape(len(history), 3)
    payload = {
        "meta_json": np.asarray(json.dumps(meta)),
        "history": hist,
        **{f"field_{name}": np.ascontiguousarray(arr) for name, arr in fields.items()},
    }
    with atomic_write(path) as tmp:
        # np.savez(str_path) appends ".npz" to names missing it (the notebook's
        # ".tmp.npy" orphan bug); an open file handle sidesteps that entirely.
        with open(tmp, "wb") as fh:
            np.savez(fh, **payload)
    log.info(
        "checkpoint written: %s (stage %s, period %d, %d steps)", path.name, stage, periods_done, n
    )
    return path


def load_checkpoint(path: str | Path, fingerprint: dict[str, Any]) -> dict[str, Any] | None:
    """Load a checkpoint for the run described by ``fingerprint``.

    Returns None when no checkpoint exists (fresh run). Raises
    :class:`CheckpointMismatch` when one exists but belongs to a different
    run — resuming it would silently splice two different simulations.
    """
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as npz:
        meta = json.loads(str(npz["meta_json"]))
        if meta.get("format") != CHECKPOINT_FORMAT:
            raise CheckpointMismatch(
                f"{path.name}: format {meta.get('format')!r} != {CHECKPOINT_FORMAT!r}"
            )
        stored_fp, want_fp = meta["fingerprint"], fingerprint
        if _fingerprint_json(stored_fp) != _fingerprint_json(want_fp):
            keys = sorted(set(stored_fp) | set(want_fp))
            diff = [
                f"  {k}: checkpoint={stored_fp.get(k)!r} vs run={want_fp.get(k)!r}"
                for k in keys
                if stored_fp.get(k) != want_fp.get(k)
            ]
            raise CheckpointMismatch(
                f"{path.name} was written by a DIFFERENT run than the one asked to "
                f"resume — refusing to splice trajectories. Differing keys:\n"
                + "\n".join(diff)
                + f"\nDelete {path} to start this run from scratch."
            )
        hist = npz["history"]
        fields = {key[len("field_") :]: npz[key] for key in npz.files if key.startswith("field_")}
    state = {
        "stage": meta["stage"],
        "n": int(meta["n"]),
        "periods_done": int(meta["periods_done"]),
        "prev_peak": meta["prev_peak"],
        "converged": bool(meta["converged"]),
        "converged_period": int(meta["converged_period"]),
        "history": [(int(p), float(pk), float(r)) for p, pk, r in hist],
        "fields": fields,
    }
    log.info(
        "checkpoint loaded: %s (stage %s, period %d, %d steps)",
        path.name,
        state["stage"],
        state["periods_done"],
        state["n"],
    )
    return state
