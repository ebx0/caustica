"""Explicit element tables: bring your own transducer geometry.

:class:`~caustica.arrays.transducer.TransducerArray` has always accepted an
arbitrary ``(n, 3)`` set of element centers and normals — the only thing
missing was a door for people whose layout is neither an Archimedean spiral
nor a bowl (a manufacturer's element table, a CAD export, an optimizer's
output). This module is that door: a tiny reader for ``.npz`` / ``.csv``
element tables, plus the builder that turns a table into a
:class:`TransducerArray`.

Units follow the layer they belong to: this module is L0/L1 and speaks
**metres**, like every other caustica geometry API. The ``elements`` *job*
kind (:mod:`caustica.config.job`) speaks **millimetres**, like every other
job field — including inside the referenced file. The file itself carries no
unit tag, so the schema fixes it: an element table read by a job is in mm.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from caustica.arrays.transducer import TransducerArray

__all__ = ["element_table_digest", "elements_array", "read_element_file"]

#: Rounding applied before hashing an element table [m]. 1 pm is ~1e-10 of a
#: therapy aperture — far below anything physical, far above float64 noise —
#: so the digest is stable across the inline and file paths.
_DIGEST_DECIMALS = 12

#: Largest plausible distance of an element from the beam axis [m]. A table
#: that exceeds it is almost always a unit mistake, and a silently 1000x
#: transducer would "run" and produce meaningless numbers.
_MAX_APERTURE_R = 1.0


def read_element_file(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read an element table; returns ``(positions, normals_or_None)``.

    Two formats, chosen by suffix:

    ``.npz``
        ``positions`` — ``(n, 3)``, required; ``normals`` — ``(n, 3)``,
        optional. Any other array in the file is ignored.
    ``.csv``
        3 or 6 numeric columns (``x,y,z`` or ``x,y,z,nx,ny,nz``),
        comma- or whitespace-separated. One optional header line, plus
        ``#`` comment lines, are skipped.

    Values are returned exactly as stored — this reader assigns no units.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"element table not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            keys = list(data.files)
            if "positions" not in keys:
                raise ValueError(
                    f"{path.name}: npz element table needs a 'positions' array "
                    f"(found: {', '.join(keys) or '(empty)'}); optional: 'normals'"
                )
            pos = np.asarray(data["positions"], np.float64)
            nrm = np.asarray(data["normals"], np.float64) if "normals" in keys else None
        return pos, nrm
    if suffix == ".csv":
        rows: list[list[float]] = []
        header_used = False
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split(",") if "," in line else line.split()
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                # The first non-numeric line before any data is the header —
                # however many comment/blank lines preceded it.
                if not rows and not header_used:
                    header_used = True
                    continue
                raise ValueError(
                    f"{path.name}:{lineno}: not a numeric element row: {raw.strip()!r}"
                ) from None
        if not rows:
            raise ValueError(f"{path.name}: no element rows found")
        widths = {len(r) for r in rows}
        if widths not in ({3}, {6}):
            raise ValueError(
                f"{path.name}: csv element table needs 3 columns (x,y,z) or 6 "
                f"(x,y,z,nx,ny,nz) on every row; found row widths {sorted(widths)}"
            )
        table = np.asarray(rows, np.float64)
        return (table[:, :3], table[:, 3:6] if table.shape[1] == 6 else None)
    raise ValueError(
        f"{path.name}: unsupported element-table format '{suffix or path.name}'; use .npz or .csv"
    )


def element_table_digest(array: TransducerArray) -> str:
    """A short content hash of an array's element table (positions + normals).

    Summary statistics cannot falsify an element table. Aperture numbers —
    element count, maximum radius, shell depth — are order statistics: they
    survive mirroring the array, rotating it, re-scattering every element but
    the outermost, swapping two elements' radii, or changing every normal.
    Each of those changes the radiated field by tens of per cent while every
    recorded number stays put (measured, M10m review). So a *table*-backed
    kind records a digest of the table itself, and a reload that reads a
    different table says so.

    Deliberately not a checksum of the FILE: the same geometry given inline,
    as ``.npz`` or as ``.csv`` must produce the same digest, and a file whose
    bytes changed without its geometry changing is not drift.
    """
    h = hashlib.sha256()
    for a in (array.positions, array.normals):
        rounded = np.round(np.asarray(a, np.float64), _DIGEST_DECIMALS) + 0.0  # kill -0.0
        h.update(np.ascontiguousarray(rounded).tobytes())
    return h.hexdigest()[:16]


def elements_array(
    positions: np.ndarray,
    elem_radius: float,
    focal_length: float,
    normals: np.ndarray | None = None,
) -> TransducerArray:
    """Build a :class:`TransducerArray` from explicit element centers [m].

    ``normals=None`` points every element at the geometric focus
    ``(0, 0, focal_length)`` — the apex-frame convention shared with
    :func:`~caustica.arrays.transducer.archimedean_spiral` and
    :mod:`caustica.analytic`. Supplied normals are normalized here, so a
    table of un-normalized direction vectors is accepted as-is.
    """
    pos = np.asarray(positions, np.float64)
    if pos.size == 0:
        raise ValueError("element table is empty: need at least one element")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (n, 3), got {pos.shape}")
    if not np.isfinite(pos).all():
        raise ValueError("element positions contain NaN/inf")
    r_max = float(np.linalg.norm(pos[:, :2], axis=1).max())
    if r_max > _MAX_APERTURE_R:
        raise ValueError(
            f"elements sit up to {r_max:.4g} m from the beam axis — that is not a "
            f"transducer, it is a unit mistake. caustica geometry APIs take METRES; "
            f"a job file's element table takes MILLIMETRES."
        )

    if normals is None:
        nrm = np.array([0.0, 0.0, focal_length]) - pos
    else:
        nrm = np.asarray(normals, np.float64)
        if nrm.shape != pos.shape:
            raise ValueError(
                f"normals {nrm.shape} do not match positions {pos.shape}: give one "
                f"unit vector per element, or omit them to aim every element at the focus"
            )
        if not np.isfinite(nrm).all():
            raise ValueError("element normals contain NaN/inf")
    norm = np.linalg.norm(nrm, axis=1, keepdims=True)
    if float(norm.min()) <= 0.0:
        bad = int(np.argmin(norm))
        raise ValueError(
            f"element {bad} has a zero-length normal"
            + ("" if normals is not None else " (it sits exactly on the geometric focus)")
        )
    return TransducerArray(
        positions=pos,
        normals=nrm / norm,
        elem_radius=float(elem_radius),
        focal_length=float(focal_length),
    )
