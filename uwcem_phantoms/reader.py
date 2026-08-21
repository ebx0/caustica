"""Decoding the UWCEM ASCII volumes — fast, and faithful to the file.

The repository ships two ~15-30 million line ASCII columns per phantom
(``mtype.txt``, ``pval.txt``). ``np.loadtxt`` needs minutes on those; a GUI
that reloads a phantom on every parameter change cannot afford that. So both
files get a **vectorized byte-level decoder** exploiting their very rigid
formats, each guarded by a validator that falls back to ``np.loadtxt`` the
moment the file does not look the way we assume:

* ``mtype.txt`` — every line is one of exactly ten tokens (``-1.0 -2.0 -4.0
  1.1 1.2 1.3 2.0 3.1 3.2 3.3``). The triple (byte0, byte1, byte2) is unique
  across those ten, so one gather + one ``searchsorted`` turns the whole file
  into ``int8`` class codes without ever parsing a float.
* ``pval.txt`` — printed as ``%1.5f``, so every record is exactly 7 characters
  plus a newline. That is a FIXED STRIDE: reshape the raw bytes to (n, 8) and
  do the decimal arithmetic in numpy.

Both decoders verify their assumption on the real bytes before trusting it
(token set for mtype, stride + separator positions for pval), and both handle
CRLF and a missing final newline. A file that violates the assumption is
decoded slowly and correctly rather than quickly and wrongly.

Faithfulness rule: this module returns the volume in the file's OWN
``(s1, s2, s3)`` index order. Anatomical reorientation is a separate,
documented transform (:mod:`uwcem_phantoms.orientation`) — mixing the two
is how axis bugs become invisible.
"""

from __future__ import annotations

import os
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uwcem_phantoms import catalog
from uwcem_phantoms.paths import cache_dir, ensure_dir

#: Media numbers from Table 1 of the repository instruction manual, in the
#: canonical order used for the integer class codes stored by this module.
MEDIA_NUMBERS: tuple[float, ...] = (
    -1.0,  # 0 immersion medium (outside the breast)
    -2.0,  # 1 skin
    -4.0,  # 2 muscle (chest wall)
    1.1,  # 3 fibroconnective/glandular-1 (highest water content)
    1.2,  # 4 fibroconnective/glandular-2
    1.3,  # 5 fibroconnective/glandular-3
    2.0,  # 6 transitional
    3.1,  # 7 fatty-1
    3.2,  # 8 fatty-2
    3.3,  # 9 fatty-3 (lowest water content)
)

#: ``MEDIA_TOKENS[i]`` is the exact ASCII spelling written by the repository.
MEDIA_TOKENS: tuple[str, ...] = (
    "-1.0",
    "-2.0",
    "-4.0",
    "1.1",
    "1.2",
    "1.3",
    "2.0",
    "3.1",
    "3.2",
    "3.3",
)

N_CLASSES = len(MEDIA_NUMBERS)

#: Line terminator of the repository files.
NEWLINE_BYTE = bytes([10])


class PhantomFormatError(RuntimeError):
    """A downloaded phantom file does not match the documented format."""


# ------------------------------------------------------------- breastInfo


def parse_breast_info(text: str) -> dict:
    """Parse ``breastInfo.txt`` into ``{id, s1, s2, s3, classification}``."""
    out: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key == "breast ID":
            out["id"] = value
        elif key in ("s1", "s2", "s3"):
            out[key] = int(value)
        elif key == "classification":
            out["classification"] = int(value)
    missing = {"id", "s1", "s2", "s3", "classification"} - set(out)
    if missing:
        raise PhantomFormatError(f"breastInfo.txt is missing {sorted(missing)}")
    return out


def read_breast_info(phantom_id: str) -> dict:
    """Read the on-disk ``breastInfo.txt`` for ``phantom_id``."""
    path = catalog.get(phantom_id).local("breastInfo.txt")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run uwcem_phantoms.fetch({phantom_id!r}) first"
        )
    return parse_breast_info(path.read_text(encoding="ascii", errors="replace"))


# ------------------------------------------------------------- zip access


def _read_zip_member(zip_path: Path, expect_suffix: str = ".txt") -> bytes:
    """Return the bytes of the single text member of ``zip_path``."""
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found — run uwcem_phantoms.fetch(...) to download it"
        )
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(expect_suffix)]
        if not members:
            raise PhantomFormatError(f"{zip_path} holds no {expect_suffix} member")
        return zf.read(members[0])


# ------------------------------------------------------------ mtype decode


def _line_starts(raw: np.ndarray) -> np.ndarray:
    """Start offset of every non-empty line in a byte array (int64 view).

    Handles CRLF (the ``\\r`` simply lands at the end of the previous line,
    which the token decoder never reads) and a missing trailing newline.
    """
    nl = np.flatnonzero(raw == 10)
    if nl.size == 0:
        return np.zeros(1, dtype=np.int64)
    # A start follows every newline, plus offset 0. Drop a start that would
    # sit past the end (file ends with a newline — the common case).
    starts = np.empty(nl.size + 1, dtype=np.int64)
    starts[0] = 0
    starts[1:] = nl + 1
    if starts[-1] >= raw.size:
        starts = starts[:-1]
    return starts


def decode_mtype(raw_bytes: bytes, n_expected: int) -> np.ndarray:
    """Decode ``mtype.txt`` bytes into ``int8`` class codes (0..9), flat.

    Codes index :data:`MEDIA_NUMBERS`. Raises :class:`PhantomFormatError`
    when the file holds a token outside Table 1 or the wrong voxel count.
    """
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    starts = _line_starts(raw)
    if starts.size != n_expected:
        # Trailing blank lines are benign; anything else is a real mismatch.
        if starts.size > n_expected and np.all(raw[starts[n_expected] :] <= 32):
            starts = starts[:n_expected]
        else:
            raise PhantomFormatError(
                f"mtype.txt holds {starts.size} lines but the grid needs {n_expected}"
            )
    if starts[-1] + 2 >= raw.size:
        raise PhantomFormatError("mtype.txt ends mid-token")

    key = raw[starts].astype(np.uint32) << 16
    key |= raw[starts + 1].astype(np.uint32) << 8
    key |= raw[starts + 2]
    starts_kept = starts
    del starts

    table = _token_keys()
    order = np.argsort(table)
    pos = np.searchsorted(table[order], key)
    np.clip(pos, 0, table.size - 1, out=pos)
    codes = order[pos].astype(np.int8)
    bad = table[order][pos] != key
    if not bad.any():
        # The key is only the first THREE bytes, so "1.15" and "3.25" match
        # the prefixes of "1.1" and "3.2" and would decode to a real class
        # without ever raising. Confirm each token actually ENDS where its
        # Table-1 spelling does (review finding, 2026-08-18).
        lengths = np.array([len(t) for t in MEDIA_TOKENS], dtype=np.int64)
        ends = starts_kept + lengths[codes]
        terminator = np.where(ends < raw.size, raw[np.minimum(ends, raw.size - 1)], 10)
        bad = (terminator != 10) & (terminator != 13)
    if bad.any():
        where = np.flatnonzero(bad)[:5]
        spelled = [
            raw_bytes[int(o) : int(o) + 6].split(NEWLINE_BYTE)[0].decode("ascii", "replace")
            for o in starts_kept[where]
        ]
        raise PhantomFormatError(
            f"mtype.txt holds tokens {spelled} that are not in Table 1 "
            f"{MEDIA_TOKENS}; refusing to guess a tissue type"
        )
    return codes


def _token_keys() -> np.ndarray:
    """The (byte0, byte1, byte2) key of each Table-1 token, as uint32."""
    keys = np.empty(N_CLASSES, dtype=np.uint32)
    for i, tok in enumerate(MEDIA_TOKENS):
        b = tok.encode("ascii")
        keys[i] = (b[0] << 16) | (b[1] << 8) | b[2]
    return keys


def _decode_mtype_slow(raw_bytes: bytes, n_expected: int) -> np.ndarray:
    """Reference decoder: parse floats, then match against Table 1."""
    values = np.asarray(raw_bytes.decode("ascii", errors="replace").split(), dtype=np.float64)
    if values.size != n_expected:
        raise PhantomFormatError(
            f"mtype.txt holds {values.size} values but the grid needs {n_expected}"
        )
    codes = np.full(values.size, -1, dtype=np.int8)
    for i, media in enumerate(MEDIA_NUMBERS):
        codes[np.isclose(values, media, atol=1e-6)] = i
    if (codes < 0).any():
        offenders = np.unique(values[codes < 0])[:5]
        raise PhantomFormatError(f"mtype.txt holds media numbers {offenders} outside Table 1")
    return codes


# ------------------------------------------------------------- pval decode


def decode_pval(raw_bytes: bytes, n_expected: int) -> np.ndarray:
    """Decode ``pval.txt`` bytes into ``float32`` values in [0, 1], flat.

    Fast path assumes the documented ``%1.5f`` fixed-width records; falls
    back to a full float parse when the bytes disagree.
    """
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    size = raw.size
    for stride, sep_off in ((8, 7), (9, 8)):  # LF, CRLF
        usable = size - (size % stride)
        if usable // stride != n_expected:
            continue
        block = raw[:usable].reshape(-1, stride)
        if not np.all(block[:, sep_off] == 10):
            continue
        if not np.all(block[:, 1] == ord(".")):
            continue
        # Validate column-wise, and accumulate the fraction one digit at a
        # time. Materializing block[:, 2:7] as int32 costs 20 bytes per voxel
        # and the five-term sum adds four more full-size temporaries — on the
        # biggest phantom that was ~1.4 GB of scratch to produce a 111 MB
        # result, and it dominated the peak of any build that used pval
        # (review finding, 2026-08-18). One accumulator plus one live digit
        # temporary is 8 bytes per voxel.
        digit_cols = (0, 2, 3, 4, 5, 6)
        if any(block[:, c].min() < 48 or block[:, c].max() > 57 for c in digit_cols):
            continue
        frac = (block[:, 2].astype(np.int32) - 48) * 10000
        for col, weight in ((3, 1000), (4, 100), (5, 10), (6, 1)):
            frac += (block[:, col].astype(np.int32) - 48) * weight
        out = block[:, 0].astype(np.float32) - np.float32(48.0)
        out += frac.astype(np.float32) * np.float32(1e-5)
        del frac
        return np.ascontiguousarray(out)
    return _decode_pval_slow(raw_bytes, n_expected)


def _decode_pval_slow(raw_bytes: bytes, n_expected: int) -> np.ndarray:
    values = np.asarray(raw_bytes.decode("ascii", errors="replace").split(), dtype=np.float32)
    if values.size != n_expected:
        raise PhantomFormatError(
            f"pval.txt holds {values.size} values but the grid needs {n_expected}"
        )
    return values


# --------------------------------------------------------------- the model


@dataclass(frozen=True, eq=False)
class RawPhantom:
    """One UWCEM phantom exactly as the repository stores it.

    ``codes`` is an ``int8`` volume of indices into :data:`MEDIA_NUMBERS`,
    shaped ``(s1, s2, s3)`` — the file's own axis order, NOT reoriented.
    ``pval`` (optional) is the matching ``float32`` within-class position in
    [0, 1]; it is 0 for skin, muscle and the immersion medium, which have no
    within-class spread in the source data.

    Both arrays are FORTRAN-contiguous. That is not an accident: the file is
    written with ``s1`` varying fastest, so ``flat.reshape((s3, s2, s1))``
    followed by a ``(2, 1, 0)`` transpose is a zero-copy view, while forcing
    C order here would cost a full strided copy of up to 28 million voxels
    (~4 s) that :mod:`~uwcem_phantoms.orientation` is about to redo anyway.
    Reorientation makes exactly one contiguous copy, at the end.
    """

    phantom_id: str
    acr_class: int
    codes: np.ndarray
    dx: float = catalog.NATIVE_DX
    pval: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.codes.shape

    @property
    def n_voxels(self) -> int:
        return int(self.codes.size)

    @property
    def extent(self) -> tuple[float, ...]:
        return tuple(n * self.dx for n in self.shape)

    @property
    def has_pval(self) -> bool:
        return self.pval is not None

    def class_counts(self) -> dict[int, int]:
        """Voxel count per class code (0..9), including absent classes."""
        counts = np.bincount(self.codes.reshape(-1).astype(np.int64), minlength=N_CLASSES)
        return {i: int(counts[i]) for i in range(N_CLASSES)}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        mm = tuple(round(e * 1e3, 1) for e in self.extent)
        return (
            f"RawPhantom(id={self.phantom_id}, ACR={self.acr_class}, "
            f"shape={self.shape}, {mm} mm, pval={self.has_pval})"
        )


def _as_native_volume(flat: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Reshape a flat column into ``(s1, s2, s3)`` without copying.

    The repository's write loop is ``for kk: for jj: for ii: print A(ii,jj,kk)``,
    i.e. column-major over ``(s1, s2, s3)``. Reading the same bytes row-major
    as ``(s3, s2, s1)`` and transposing back is therefore an exact, free
    relabeling of the axes — no data moves.
    """
    s1, s2, s3 = shape
    return flat.reshape((s3, s2, s1)).transpose(2, 1, 0)


def _cache_path(phantom_id: str) -> Path:
    return cache_dir() / f"{phantom_id}-mtype.npz"


def load_raw(
    phantom_id: str,
    with_pval: bool = False,
    cache: bool = True,
    fetch_missing: bool = True,
) -> RawPhantom:
    """Load one phantom's voxel data, downloading and caching as needed.

    The ``mtype`` volume is cached as ``int8`` in ``_data/cache`` (a few
    hundred kB — it compresses ~1000:1), so the second load is instant.
    ``pval`` is deliberately NOT cached: as float data it compresses badly
    (tens of MB per phantom) while decoding straight from the shipped zip
    costs well under a second.
    """
    info = catalog.get(phantom_id)
    if fetch_missing and not info.is_downloaded(with_pval=with_pval):
        catalog.fetch(phantom_id, with_pval=with_pval)

    meta = read_breast_info(phantom_id)
    shape = (meta["s1"], meta["s2"], meta["s3"])
    if shape != info.shape:
        raise PhantomFormatError(
            f"breastInfo.txt for {phantom_id} says {shape} but the catalog expects "
            f"{info.shape}; the download may be corrupt or the repository changed"
        )
    n = shape[0] * shape[1] * shape[2]

    codes = None
    cpath = _cache_path(phantom_id)
    zip_path = info.local("mtype.zip")
    # Fingerprint the SOURCE, not just the cache: a re-download (or a
    # --force fetch that fixed a truncated archive) must not keep serving the
    # array decoded from the old bytes.
    try:
        stat = zip_path.stat()
        fingerprint = f"{phantom_id}|{stat.st_size}|{int(stat.st_mtime)}"
    except OSError:
        fingerprint = phantom_id
    if cache and cpath.exists():
        try:
            with np.load(cpath) as data:
                cached = data["codes"]
                stamp = str(data["fingerprint"]) if "fingerprint" in data.files else None
                if cached.shape == shape and stamp == fingerprint:
                    codes = cached
        except Exception:  # noqa: BLE001 - any unreadable cache re-decodes
            # zipfile.BadZipFile (a truncated npz) derives from Exception, not
            # from OSError/ValueError/KeyError, so a narrower clause let a
            # damaged cache escape as a hard failure with no way to clear it
            # (review finding, 2026-08-18). The cache is an optimization; a
            # bad one costs a re-decode, never a failed load.
            codes = None

    if codes is None:
        raw_bytes = _read_zip_member(info.local("mtype.zip"))
        try:
            flat = decode_mtype(raw_bytes, n)
        except PhantomFormatError:
            flat = _decode_mtype_slow(raw_bytes, n)
        codes = _as_native_volume(flat, shape)
        if cache:
            ensure_dir(cpath.parent)
            # Unique per writer. The studio server is threaded and its /api/plan
            # (180 ms debounce) and /api/build (420 ms) both decode the same
            # phantom on first use, a ~1.4 s window: a shared ".part" name made
            # them collide — WinError 32 on replace, ENOENT on POSIX (review
            # finding, 2026-08-18). Publishing is best effort; losing the race
            # must not fail a load that already succeeded.
            tmp = cpath.with_name(f"{cpath.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
            try:
                # Write through a file OBJECT: np.savez_compressed(path)
                # appends ".npz" to any name that lacks it, so the atomic
                # rename would otherwise lose its temp file.
                with open(tmp, "wb") as fh:
                    np.savez_compressed(
                        fh, codes=codes, phantom_id=phantom_id, fingerprint=fingerprint
                    )
                os.replace(tmp, cpath)
            except OSError:
                Path(tmp).unlink(missing_ok=True)

    pval = None
    if with_pval:
        raw_bytes = _read_zip_member(info.local("pval.zip"))
        pval = _as_native_volume(decode_pval(raw_bytes, n), shape)
        np.clip(pval, 0.0, 1.0, out=pval)

    return RawPhantom(
        phantom_id=phantom_id,
        acr_class=meta["classification"],
        codes=codes,
        dx=catalog.NATIVE_DX,
        pval=pval,
    )
