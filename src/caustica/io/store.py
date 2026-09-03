"""The HDF5 result contract (``caustica-result/1``) and a Drive-proof store.

Port of the notebook's data contract, generalized from "one phase-map
sample" to "one CW solve":

* STORED  : ``output/p_real_h{n}``, ``output/p_imag_h{n}`` for every recorded
  harmonic, plus ``output/p_max`` — float16 dynamically quantized when the
  measured round-trip error meets the contract, else float32 (see
  :mod:`caustica.io.quantize`).
* DERIVED : amplitude and phase are NEVER stored — always recomputed from
  real/imag, exactly like :class:`~caustica.solvers.base.SolverResult` does.
* Every write is atomic (:mod:`caustica.io.atomic`), so a file that exists is
  a file that was completely written — the invariant the resume scan trusts.
* The phase convention and the absorption model travel as root/output attrs
  in EVERY file (a downstream contract, and a release gate). Nothing reading
  a result a year from now should have to guess either.

:class:`ResultStore` wraps a directory of such files with the notebook's
Drive survival kit: verified mkdir (a Drive FUSE mount can accept a mkdir
and not materialize it), a write probe (an unwritable directory must fail
NOW, not after the solve), temp-debris sweep, and the resume skip-guard
(:meth:`ResultStore.missing`).
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import caustica
from caustica.io.atomic import atomic_write, sweep_temp_debris
from caustica.io.quantize import DEFAULT_MAX_NORM_ERR, restore, try_float16
from caustica.solvers.base import SolverResult
from caustica.sources import CWSource

log = logging.getLogger("caustica")

RESULT_FORMAT = "caustica-result/1"

#: Library-wide phasor convention (engine docstring is the source of truth).
PHASE_CONVENTION = (
    "p(t) = Re{P exp(-i omega t)}; outgoing wave = exp(+ikx) (matches the "
    "analytic O'Neil/Rayleigh references). The absolute phase zero is not "
    "source-referenced; relative phase (voxel-to-voxel) is exact."
)
#: Library-wide absorption model (engine docstring is the source of truth).
ABSORPTION_MODEL = (
    "frequency-INDEPENDENT exponential, exp(-alpha*c*dt) per step, applied "
    "symmetrically to p and u so the spatial decay rate equals alpha (NOT "
    "the source notebook's p-only alpha/2). Harmonics above f0 are therefore "
    "under-absorbed relative to power-law tissue."
)

_H5_COMPRESSION = "gzip"
_H5_COMPRESSION_OPTS = 4


# --------------------------------------------------------------------- write


def _quantized_dataset(
    grp: h5py.Group, name: str, arr: np.ndarray, quantize: bool, max_norm_err: float
) -> None:
    """One compressed dataset + the metadata needed to reload it losslessly-enough."""
    if quantize:
        q = try_float16(arr, max_norm_err)
    else:
        from caustica.io.quantize import Quantized

        q = Quantized(np.asarray(arr, dtype=np.float32), 1.0, "float32", 0.0)
    ds = grp.create_dataset(
        name,
        data=q.stored,
        chunks=True,
        compression=_H5_COMPRESSION,
        compression_opts=_H5_COMPRESSION_OPTS,
    )
    ds.attrs["scale"] = float(q.scale)
    ds.attrs["stored_dtype"] = q.dtype_name
    ds.attrs["quant_norm_err"] = float(q.norm_err)
    ds.attrs["reload"] = 'arr = ds[:].astype("float32") * ds.attrs["scale"]'


def save_result(
    path: str | Path,
    result: SolverResult,
    source: CWSource,
    *,
    dx: float,
    grid_shape: tuple[int, ...],
    pml_vox: int = 0,
    quantize: bool = True,
    max_norm_err: float = DEFAULT_MAX_NORM_ERR,
    extra_attrs: dict[str, Any] | None = None,
) -> Path:
    """Write one solve result atomically under the ``caustica-result/1`` contract.

    ``dx``/``grid_shape``/``pml_vox`` are recorded so the file is
    self-describing (a reader can place ``result.region`` in space without
    the original Grid object). ``extra_attrs`` land on the root group —
    the runner stamps job/commit/environment through it.
    """
    path = Path(path)
    with atomic_write(path) as tmp:
        with h5py.File(tmp, "w") as hf:
            hf.attrs["format"] = RESULT_FORMAT
            hf.attrs["caustica_version"] = caustica.__version__
            hf.attrs["solver"] = str(result.meta.get("solver", ""))
            hf.attrs["backend"] = str(result.meta.get("backend", ""))
            hf.attrs["f0_hz"] = float(source.f0)
            hf.attrs["dt_s"] = float(result.dt)
            hf.attrs["spp"] = int(result.spp)
            hf.attrs["steps_total"] = int(result.steps_total)
            hf.attrs["t_end_s"] = float(result.t_end_s)
            hf.attrs["tof_periods"] = int(result.tof_periods)
            hf.attrs["converged_period"] = int(result.converged_period)
            hf.attrs["settle_capped"] = bool(result.settle_capped)
            hf.attrs["harmonics"] = sorted(result.phasors)
            hf.attrs["dx_m"] = float(dx)
            hf.attrs["grid_shape"] = list(grid_shape)
            hf.attrs["pml_vox"] = int(pml_vox)
            hf.attrs["region_start"] = [s.start for s in result.region]
            hf.attrs["region_stop"] = [s.stop for s in result.region]
            hf.attrs["phase_convention"] = PHASE_CONVENTION
            hf.attrs["absorption_model"] = ABSORPTION_MODEL
            # Which numerics produced this field, and how the source that fed
            # them was discretized. Both moved in 2026-08-24 by 13-18 % of an
            # absolute amplitude; a stored pressure has to say which side of
            # that it came from rather than leaving it to a commit date.
            hf.attrs["numerics_scheme"] = str(result.meta.get("numerics_scheme", "unknown"))
            hf.attrs["source_discretization"] = str(
                getattr(source, "discretization", "") or "unknown"
            )
            for key, val in (extra_attrs or {}).items():
                hf.attrs[key] = val

            gin = hf.create_group("input")
            gin.create_dataset("source_indices", data=np.asarray(source.indices, dtype=np.int32))
            gin.create_dataset("source_phases_rad", data=np.asarray(source.phases, np.float32))
            gin.attrs["f0_hz"] = float(source.f0)
            gin.attrs["amplitude_pa"] = float(source.amplitude)
            gin.attrs["ramp_periods"] = float(source.ramp_periods)
            gin.attrs["label"] = str(source.label)
            gin.attrs["n_points"] = int(source.n_points)

            gout = hf.create_group("output")
            for h in sorted(result.phasors):
                ph = result.phasors[h]
                _quantized_dataset(gout, f"p_real_h{h}", ph.real, quantize, max_norm_err)
                _quantized_dataset(gout, f"p_imag_h{h}", ph.imag, quantize, max_norm_err)
            _quantized_dataset(gout, "p_max", result.p_max, quantize, max_norm_err)
            gout.attrs["stored"] = (
                "p_real_h{n}, p_imag_h{n} (harmonic phasors), p_max (time peak "
                "over the record window)"
            )
            gout.attrs["derived"] = (
                "p_amp = sqrt(p_real^2 + p_imag^2); p_phase = atan2(p_imag, p_real) "
                "— never stored, always recomputed"
            )
            gout.attrs["phase_convention"] = PHASE_CONVENTION
            gout.attrs["absorption_model"] = ABSORPTION_MODEL
            gout.attrs["pmax_window"] = (
                "time-domain peak over the RECORD window only (steady state); "
                "transient/settling peaks are excluded by design"
            )

            hist = np.asarray(result.convergence_history, dtype=np.float64).reshape(-1, 3)
            gconv = hf.create_group("convergence")
            gconv.create_dataset("history", data=hist)
            gconv.attrs["columns"] = "period, peak_pa, rel_change"
    return path


# ---------------------------------------------------------------------- read


def load_field(path: str | Path, name: str) -> np.ndarray:
    """Reload one stored ``output/`` field, honoring the dynamic float16 scale."""
    with h5py.File(path, "r") as hf:
        ds = hf[f"output/{name}"]
        return restore(ds[:], ds.attrs["scale"])


def _geometry(hf: h5py.File) -> dict:
    """The file's self-description: where the fields sit and what drove them.

    Parsed HERE, beside the :func:`save_result` that writes these attrs.
    ``caustica report`` used to keep its own copy of this parsing —
    and open the file a SECOND time to run it — which is exactly how a
    schema and its reader drift apart (janitor ticket 02).

    Missing optional stamps degrade the way a reader needs them to: an
    older file has no ``apex_vox``, so positions fall back to the grid
    origin and ``apex_known`` says so instead of the caller guessing.
    """
    a = hf.attrs
    return {
        "dx": float(a["dx_m"]),
        "grid_shape": tuple(int(v) for v in a["grid_shape"]),
        "pml_vox": int(a["pml_vox"]),
        "apex_vox": tuple(int(v) for v in a["apex_vox"]) if "apex_vox" in a else (0, 0, 0),
        "focus_vox": tuple(int(v) for v in a["focus_vox"]) if "focus_vox" in a else None,
        "apex_known": "apex_vox" in a,
        "job_name": str(a.get("job_name", "")),
        "solver": str(a.get("solver", "")),
        "backend": str(a.get("backend", "")),
        "git_commit": str(a.get("git_commit", "")),
        "f0_hz": float(a["f0_hz"]),
        "amplitude_pa": float(hf["input"].attrs["amplitude_pa"]),
        "source_indices": np.asarray(hf["input/source_indices"]),
    }


def load_result(
    path: str | Path, *, with_geometry: bool = False
) -> SolverResult | tuple[SolverResult, dict]:
    """Reconstruct a :class:`SolverResult` from a ``caustica-result/1`` file.

    Values differ from the originals by at most the recorded quantization
    error (``<= max_norm_err * |peak|`` per field, 0 when stored float32).

    ``with_geometry=True`` returns ``(result, geometry)`` — the fields AND
    the self-description (dx, grid shape, PML, apex/focus voxels, drive,
    source voxels, provenance) from ONE open, which is all a report needs to
    place the numbers in space.
    """
    path = Path(path)
    with h5py.File(path, "r") as hf:
        if hf.attrs.get("format") != RESULT_FORMAT:
            raise ValueError(f"{path.name}: format {hf.attrs.get('format')!r} != {RESULT_FORMAT!r}")
        phasors: dict[int, np.ndarray] = {}
        for h in (int(x) for x in hf.attrs["harmonics"]):
            re_ds, im_ds = hf[f"output/p_real_h{h}"], hf[f"output/p_imag_h{h}"]
            re_arr = restore(re_ds[:], re_ds.attrs["scale"])
            im_arr = restore(im_ds[:], im_ds.attrs["scale"])
            phasors[h] = (re_arr + 1j * im_arr).astype(np.complex64)
        pmax_ds = hf["output/p_max"]
        p_max = restore(pmax_ds[:], pmax_ds.attrs["scale"])
        region = tuple(
            slice(int(a), int(b))
            for a, b in zip(hf.attrs["region_start"], hf.attrs["region_stop"], strict=True)
        )
        history = [(int(p), float(pk), float(r)) for p, pk, r in hf["convergence/history"][:]]
        meta = {
            "solver": str(hf.attrs["solver"]),
            "backend": str(hf.attrs["backend"]),
            "loaded_from": str(path),
            "caustica_version": str(hf.attrs["caustica_version"]),
        }
        result = SolverResult(
            phasor=phasors[1],
            p_max=p_max,
            region=region,
            dt=float(hf.attrs["dt_s"]),
            spp=int(hf.attrs["spp"]),
            steps_total=int(hf.attrs["steps_total"]),
            t_end_s=float(hf.attrs["t_end_s"]),
            tof_periods=int(hf.attrs["tof_periods"]),
            converged_period=int(hf.attrs["converged_period"]),
            settle_capped=bool(hf.attrs["settle_capped"]),
            convergence_history=history,
            phasors=phasors,
            meta=meta,
        )
        return (result, _geometry(hf)) if with_geometry else result


def validate_result_file(path: str | Path) -> bool:
    """Cheap structural check: opens, declares the format, has the outputs.

    Used only near the resume high-water mark — the atomic-write invariant
    makes existence sufficient everywhere else.
    """
    try:
        with h5py.File(path, "r") as hf:
            if hf.attrs.get("format") != RESULT_FORMAT:
                return False
            harmonics = [int(x) for x in hf.attrs["harmonics"]]
            names = [f"p_real_h{h}" for h in harmonics]
            names += [f"p_imag_h{h}" for h in harmonics]
            names.append("p_max")
            return all(f"output/{n}" in hf for n in names)
    except OSError:
        return False


# --------------------------------------------------------------------- store


def ensure_dir_verified(directory: str | Path, retries: int = 3) -> Path:
    """``mkdir -p`` and PROVE the directory exists (Drive FUSE lesson, v12.3).

    ``os.makedirs`` returning without raising is not proof on a Google Drive
    FUSE mount — the mkdir can be accepted and then not materialize, which is
    exactly how a 64-second solve once got thrown away. Verified with
    ``is_dir()`` and retried with a short backoff.
    """
    directory = Path(directory)
    for attempt in range(retries + 1):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            if attempt == retries:
                raise
            log.warning("mkdir %s failed (%s) — retry %d/%d", directory, exc, attempt + 1, retries)
        if directory.is_dir():
            return directory
        time.sleep(0.2 * (attempt + 1))
    raise RuntimeError(
        f"could not create/verify directory {directory}. On a Drive FUSE mount this "
        f"usually means the mount went stale — remount with force_remount=True."
    )


def probe_writable(directory: str | Path) -> None:
    """Write and delete a probe file so an unwritable path fails NOW.

    Without this, an unwritable output directory is only discovered after a
    full solve, discarding the result. One millisecond here is worth an hour
    there. Raises ``OSError`` on failure.
    """
    directory = Path(directory)
    # Real per-process entropy: an address-derived name can collide between
    # two sessions probing the same Drive folder, and one probe's unlink
    # would make the other's exists() check report a healthy mount as stale
    # (adversarial review, 2026-08-19).
    probe = directory / f".write_probe_{os.getpid():x}-{secrets.token_hex(3)}"
    with open(probe, "wb") as fh:
        fh.write(b"ok")
    vanished = not probe.exists()
    probe.unlink(missing_ok=True)
    if vanished:
        raise OSError(f"write probe vanished after write in {directory} — stale mount?")


class ResultStore:
    """A directory of ``caustica-result/1`` files with Drive-proof plumbing.

    Construction verifies the directory (mkdir + probe) and sweeps temp
    debris; :meth:`missing` is the resume skip-guard: give it the full list
    of names a session should produce and it returns only the ones that
    still need generating — a deleted or torn file re-enters the list, a
    complete one never does.
    """

    SUFFIX = ".h5"

    def __init__(self, root: str | Path, *, probe: bool = True):
        self.root = ensure_dir_verified(root)
        if probe:
            probe_writable(self.root)
        swept = sweep_temp_debris(self.root)
        if swept:
            log.info("swept %d temp file(s) in %s", len(swept), self.root)

    def path(self, name: str) -> Path:
        return self.root / f"{name}{self.SUFFIX}"

    def save(self, name: str, result: SolverResult, source: CWSource, **kwargs: Any) -> Path:
        """Atomic :func:`save_result` under ``root/<name>.h5``.

        Re-verifies the parent directory at WRITE time, not just at store
        construction: minutes of solving separate the two moments and a
        Drive FUSE mount can lose a directory in between (v12.3).
        """
        ensure_dir_verified(self.root)
        return save_result(self.path(name), result, source, **kwargs)

    def load(self, name: str) -> SolverResult:
        return load_result(self.path(name))

    def is_complete(self, name: str) -> bool:
        """Exists — which, under the atomic-write invariant, means complete."""
        return self.path(name).exists()

    def completed(self, pattern: str = "*", deep: bool = False) -> list[str]:
        """Names present in the store, cheapest-honest validation applied.

        One directory listing; structural validation only near the high-water
        mark (the last 3 names, where an interrupted session could
        conceivably have left something odd) unless ``deep=True`` validates
        every file (slow over Drive). Invalid files are dropped from the
        listing — i.e. they will be regenerated.
        """
        rx = re.escape(self.SUFFIX) + "$"
        names = sorted(re.sub(rx, "", p.name) for p in self.root.glob(f"{pattern}{self.SUFFIX}"))
        to_check = names if deep else names[-3:]
        bad = [n for n in to_check if not validate_result_file(self.path(n))]
        if bad:
            log.warning(
                "%d file(s) failed structural validation and will be regenerated: %s",
                len(bad),
                bad,
            )
            names = [n for n in names if n not in set(bad)]
        return names

    def missing(self, names: list[str], deep: bool = False) -> list[str]:
        """The resume skip-guard: which of ``names`` still need generating."""
        have = set(self.completed(deep=deep))
        return [n for n in names if n not in have]
