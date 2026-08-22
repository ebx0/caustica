"""IO: the result contract, atomic writes, resume, and in-run checkpoints (M10).

Four deliberately separate concerns:

* :mod:`~caustica.io.atomic` — tmp -> ``os.replace`` writes + debris sweep;
* :mod:`~caustica.io.quantize` — float16 dynamic quantization with a measured
  error contract;
* :mod:`~caustica.io.store` — the ``caustica-result/1`` HDF5 contract, the
  Drive-proof :class:`~caustica.io.store.ResultStore`, and the resume
  skip-guard;
* :mod:`~caustica.io.checkpoint` — in-run engine state checkpoints so a killed
  session loses periods, not the run.

``store`` (and with it h5py) loads lazily: the engine only needs
``checkpoint``, and ``import caustica`` must stay numpy-only.
"""

from caustica.io.atomic import atomic_write, sweep_temp_debris
from caustica.io.checkpoint import (
    CheckpointMismatch,
    CheckpointSpec,
    RunInterrupted,
    load_checkpoint,
    write_checkpoint,
)
from caustica.io.medium_volume import (
    MediumVolume,
    MediumVolumeError,
    load_medium_volume,
    write_medium_volume,
)
from caustica.io.quantize import DEFAULT_MAX_NORM_ERR, Quantized, try_float16

_STORE_NAMES = frozenset(
    {
        "ABSORPTION_MODEL",
        "PHASE_CONVENTION",
        "RESULT_FORMAT",
        "ResultStore",
        "ensure_dir_verified",
        "load_field",
        "load_result",
        "probe_writable",
        "save_result",
        "validate_result_file",
    }
)

__all__ = [
    "DEFAULT_MAX_NORM_ERR",
    "CheckpointMismatch",
    "CheckpointSpec",
    "MediumVolume",
    "MediumVolumeError",
    "Quantized",
    "RunInterrupted",
    "atomic_write",
    "load_checkpoint",
    "load_medium_volume",
    "sweep_temp_debris",
    "try_float16",
    "write_checkpoint",
    "write_medium_volume",
    *sorted(_STORE_NAMES),
]


def __getattr__(name: str):  # PEP 562: keep h5py out of `import caustica.solvers`
    if name in _STORE_NAMES:
        from caustica.io import store

        return getattr(store, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
