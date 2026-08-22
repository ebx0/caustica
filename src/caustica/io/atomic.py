"""Atomic file writes: a visible file is a completely written file.

The invariant everything in :mod:`caustica.io` leans on (checkpoints, results,
resume scans) is the notebook's hard-won rule: every write goes to a ``.tmp``
sibling first and lands under its final name only through ``os.replace``.
A reader can then trust existence alone — no structural validation is needed
to know a file is not a torn write, which is what makes resume scans cheap
over a Google Drive FUSE mount (one directory listing instead of a thousand
HDF5 opens).

Concurrency hardening (adversarial review, 2026-08-19): the temp name is
UNIQUE PER WRITER (pid + random token), so two sessions racing on the same
result name can never truncate each other's in-flight temp and publish a
torn file under the final name — each ``os.replace`` promotes only bytes its
own writer produced. The counterpart is debris: a killed writer leaves a
``*.tmp`` behind. Those are never valid, so :func:`sweep_temp_debris`
removes them — but only STALE ones (mtime older than a threshold), because
on a shared Drive folder a fresh ``.tmp`` may belong to a live sibling
session whose finished solve must not be discarded.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("caustica")

#: Suffix appended to a path while it is being written.
TMP_SUFFIX = ".tmp"

#: Default sweep threshold: a ``.tmp`` younger than this may belong to a
#: LIVE writer in another session and is left alone.
DEFAULT_STALE_S = 3600.0

#: ``os.replace`` retry policy for Windows sharing violations: a reader
#: holding the target open (h5py, a viewer) makes the rename fail
#: transiently; retrying beats destroying the finished write.
_REPLACE_ATTEMPTS = 6
_REPLACE_DELAY_S = 0.15


def tmp_path_for(path: str | Path) -> Path:
    """A writer-unique temporary sibling for ``path`` (new name every call)."""
    p = Path(path)
    return p.with_name(f"{p.name}.{os.getpid():x}-{secrets.token_hex(3)}{TMP_SUFFIX}")


def replace_with_retry(
    tmp: Path,
    path: Path,
    attempts: int = _REPLACE_ATTEMPTS,
    delay_s: float = _REPLACE_DELAY_S,
) -> None:
    """``os.replace`` with a short retry on Windows sharing violations.

    If the rename still fails, the fully written ``tmp`` is deliberately
    KEPT (and named in the error): the data survived, only the promotion
    failed — deleting it would turn a transient lock into data loss.
    """
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise PermissionError(
                    f"could not replace {path} (target locked by another process?). "
                    f"The completed write is preserved at {tmp} — close the reader "
                    f"and retry, or rename it manually."
                ) from None
            time.sleep(delay_s * (attempt + 1))


@contextmanager
def atomic_write(path: str | Path) -> Iterator[Path]:
    """Yield a temporary path; on clean exit, ``os.replace`` it onto ``path``.

    If the CALLER raises, the temporary file is removed and ``path`` is left
    as it was (absent, or the previous complete version). If only the final
    rename fails (locked target on Windows), the written temp is kept — see
    :func:`replace_with_retry`. The caller writes to the yielded path with
    whatever library it likes (h5py, np.savez, open())::

        with atomic_write(out / "result.h5") as tmp:
            with h5py.File(tmp, "w") as hf:
                ...
        # only now does result.h5 exist / change
    """
    path = Path(path)
    tmp = tmp_path_for(path)
    try:
        yield tmp
    except BaseException:
        # BaseException on purpose: KeyboardInterrupt mid-write must also
        # clean up. A SIGKILL cannot be caught by anyone — that case is what
        # sweep_temp_debris() exists for.
        try:
            tmp.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - FS-dependent
            log.warning("could not remove temp file %s: %s", tmp, exc)
        raise
    replace_with_retry(tmp, path)


def sweep_temp_debris(directory: str | Path, older_than_s: float = DEFAULT_STALE_S) -> list[Path]:
    """Remove STALE ``*.tmp`` files in ``directory`` (non-recursive).

    Only files whose mtime is older than ``older_than_s`` go: a fresh temp
    may be a live sibling session's in-flight write on a shared folder, and
    unlinking it would discard that session's finished solve at its
    ``os.replace``. Pass ``0`` to sweep unconditionally (single-owner dirs).
    Returns the paths removed.
    """
    directory = Path(directory)
    removed: list[Path] = []
    if not directory.is_dir():
        return removed
    cutoff = time.time() - older_than_s
    for junk in directory.glob(f"*{TMP_SUFFIX}"):
        try:
            if junk.stat().st_mtime > cutoff:
                continue
            junk.unlink()
            removed.append(junk)
            log.info("removed stale temp artifact %s", junk.name)
        except OSError as exc:  # pragma: no cover - FS-dependent
            log.warning("could not remove %s: %s", junk, exc)
    return removed
