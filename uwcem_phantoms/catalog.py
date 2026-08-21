"""The UWCEM phantom catalog: what exists, and how to get it.

Source: University of Wisconsin Computational Electromagnetics Laboratory,
*Numerical Breast Phantom Repository*,
https://uwcem.ece.wisc.edu/phantomRepository.html

Nine MRI-derived phantoms, 0.5 mm isotropic, classified by ACR radiographic
density (1 = almost entirely fat ... 4 = extremely dense). Each phantom ships
three ASCII files: ``breastInfo.txt`` (grid dims + class), ``mtype.txt``
(media number per voxel) and ``pval.txt`` (within-class interpolation value
per voxel). The two big ones are distributed zipped.

Design decisions worth knowing:

* Grid dimensions are NOT hardcoded here — they are read from the phantom's
  own ``breastInfo.txt``. The values in :data:`CATALOG` are documentation
  (and let ``list()`` work offline before anything is downloaded); the reader
  cross-checks them against the downloaded file and raises on a mismatch
  rather than reshaping 15 million voxels into a lie.
* Archives are kept ZIPPED on disk and decoded straight out of the zip. The
  nine ``mtype.txt``/``pval.txt`` files total ~3.5 GB unpacked versus 179 MB
  zipped, and the decoded form we actually reuse is the far smaller
  ``.npz`` cache, so unpacking the text would cost 3.5 GB to gain nothing.

Citation (required by the repository): users must acknowledge the University
of Wisconsin-Madison; see :data:`CITATION`.
"""

from __future__ import annotations

import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from uwcem_phantoms.paths import ensure_dir, raw_dir

BASE_URL = "https://uwcem.ece.wisc.edu/MRIdatabase"

CITATION = (
    "Numerical breast phantoms courtesy of the University of Wisconsin-Madison "
    "Computational Electromagnetics Laboratory "
    "(https://uwcem.ece.wisc.edu/phantomRepository.html). Cite: "
    "M. J. Burfeindt et al., 'MRI-derived 3-D-printed breast phantom for "
    "microwave breast imaging validation', IEEE Antennas and Wireless "
    "Propagation Letters, vol. 11, pp. 1610-1613, 2012; and "
    "E. Zastrow et al., 'Development of anatomically realistic numerical breast "
    "phantoms with accurate dielectric properties for modeling microwave "
    "interactions with the human breast', IEEE Trans. Biomed. Eng., vol. 55, 2008."
)

#: ACR breast-composition classes used by the repository.
ACR_CLASSES = {
    1: "almost entirely fat (<25% glandular)",
    2: "scattered fibroglandular (25-50% glandular)",
    3: "heterogeneously dense (51-75% glandular)",
    4: "very dense (>75% glandular)",
}

#: Native voxel size of every phantom in the repository [m].
NATIVE_DX = 0.5e-3

FILES = ("breastInfo.txt", "mtype.zip", "pval.zip")


@dataclass(frozen=True)
class PhantomInfo:
    """One catalog entry (documentation + expected download sizes)."""

    phantom_id: str
    acr_class: int
    shape: tuple[int, int, int]  # (s1, s2, s3) as written in breastInfo.txt
    mtype_bytes: int
    pval_bytes: int

    @property
    def acr_description(self) -> str:
        return ACR_CLASSES[self.acr_class]

    @property
    def n_voxels(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        return tuple(round(n * NATIVE_DX * 1e3, 1) for n in self.shape)  # type: ignore[return-value]

    def url(self, filename: str) -> str:
        return f"{BASE_URL}/{self.phantom_id}/{filename}"

    def local(self, filename: str) -> Path:
        return raw_dir(self.phantom_id) / filename

    def is_downloaded(self, with_pval: bool = False) -> bool:
        names = ["breastInfo.txt", "mtype.zip"] + (["pval.zip"] if with_pval else [])
        return all(self.local(n).exists() and self.local(n).stat().st_size > 0 for n in names)


#: Every phantom in the repository, ordered by ACR class then id.
CATALOG: dict[str, PhantomInfo] = {
    p.phantom_id: p
    for p in (
        PhantomInfo("071904", 1, (310, 355, 253), 931955, 26616117),
        PhantomInfo("012804", 1, (267, 375, 288), 1115856, 26347637),
        PhantomInfo("012204", 2, (316, 352, 307), 1408613, 27479249),
        PhantomInfo("070604PA1", 2, (300, 383, 270), 1269770, 26647550),
        PhantomInfo("010204", 2, (258, 253, 251), 820465, 14554071),
        PhantomInfo("080304", 3, (269, 332, 202), 1099814, 15664025),
        PhantomInfo("070604PA2", 3, (258, 365, 248), 1306087, 18872103),
        PhantomInfo("062204", 3, (219, 243, 273), 889278, 10163335),
        PhantomInfo("012304", 4, (215, 328, 212), 945232, 11019155),
    )
}

PHANTOM_IDS: tuple[str, ...] = tuple(CATALOG)


def get(phantom_id: str) -> PhantomInfo:
    """Catalog entry for ``phantom_id`` (raises with the valid ids listed)."""
    try:
        return CATALOG[phantom_id]
    except KeyError:
        raise KeyError(
            f"unknown phantom id {phantom_id!r}; available: {', '.join(PHANTOM_IDS)}"
        ) from None


def by_class(acr_class: int) -> tuple[PhantomInfo, ...]:
    """Every phantom of one ACR density class."""
    return tuple(p for p in CATALOG.values() if p.acr_class == acr_class)


# ---------------------------------------------------------------- download


class DownloadError(RuntimeError):
    """A phantom file could not be fetched or arrived corrupt."""


def _download(url: str, dest: Path, timeout: float, progress) -> None:
    """Fetch ``url`` to ``dest`` atomically (never leave a half file behind)."""
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed host
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(dest.name, done, total)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"failed to download {url}: {exc}") from exc
    tmp.replace(dest)


def fetch(
    phantom_id: str,
    with_pval: bool = True,
    force: bool = False,
    timeout: float = 300.0,
    progress=None,
) -> PhantomInfo:
    """Ensure the raw UWCEM files for ``phantom_id`` are on disk.

    Already-present files are kept unless ``force``; a zip that fails its
    CRC check is deleted and re-fetched once, because a truncated download
    that silently decodes to garbage voxels is the worst outcome here.

    ``progress`` is called as ``progress(filename, bytes_done, bytes_total)``.
    """
    info = get(phantom_id)
    names = ["breastInfo.txt", "mtype.zip"] + (["pval.zip"] if with_pval else [])
    for name in names:
        dest = info.local(name)
        for attempt in (0, 1):
            if dest.exists() and dest.stat().st_size > 0 and not (force and attempt == 0):
                if name.endswith(".zip") and not _zip_ok(dest):
                    dest.unlink(missing_ok=True)
                else:
                    break
            _download(info.url(name), dest, timeout, progress)
            if name.endswith(".zip") and not _zip_ok(dest):
                dest.unlink(missing_ok=True)
                if attempt == 1:
                    raise DownloadError(f"{info.url(name)} downloaded but is not a readable zip")
            else:
                break
    return info


def _zip_ok(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.namelist() != [] and zf.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def fetch_all(with_pval: bool = True, timeout: float = 300.0, progress=None) -> list[PhantomInfo]:
    """Download every phantom in the catalog (~179 MB with pval, ~10 MB without)."""
    return [fetch(pid, with_pval=with_pval, timeout=timeout, progress=progress) for pid in CATALOG]


def status() -> list[dict]:
    """One dict per phantom describing what is present locally (for UIs/CLI)."""
    rows = []
    for info in CATALOG.values():
        rows.append(
            {
                "id": info.phantom_id,
                "acr_class": info.acr_class,
                "acr_description": info.acr_description,
                "shape": info.shape,
                "extent_mm": info.extent_mm,
                "n_voxels": info.n_voxels,
                "has_mtype": info.local("mtype.zip").exists(),
                "has_pval": info.local("pval.zip").exists(),
                "downloaded": info.is_downloaded(),
            }
        )
    return rows
