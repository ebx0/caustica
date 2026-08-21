"""Anatomical phantom import: UWCEM breast phantoms -> simulation-ready media.

A standalone side package (repo root, next to ``apps/`` — deliberately OUTSIDE
the ``caustica`` wheel: it consumes caustica, caustica never imports it). It turns
the University of Wisconsin numerical breast phantom repository
(https://uwcem.ece.wisc.edu/phantomRepository.html) into a single file the
caustica solvers can consume, with the knobs a study actually needs:
tissue-model detail, resolution, cropping, geometric simplification, measured
within-tissue heterogeneity and synthetic scatterer noise.

Typical use — build once, simulate many times::

    from uwcem_phantoms import PhantomSpec, build

    spec = PhantomSpec(
        phantom_id="012304",                       # ACR class 4, very dense
        f0_mhz=1.0,
        resolution={"dx_mm": 0.3, "method": "smooth"},
        crop={"mode": "tissue", "margin_mm": 8},
        domain={"standoff_mm": 15, "fft_friendly": True},
        heterogeneity={"use_pval": True, "noise_pct": 2.0, "correlation_mm": 0.6},
    )
    asset = build(spec)
    print(asset.summary())
    path = asset.save()                            # -> _data/exports/<name>.npz

Then, in the simulation::

    from uwcem_phantoms import load_phantom

    ph     = load_phantom(path)
    grid   = ph.grid(pml=hs.PMLSpec(thickness=5e-3))
    medium = ph.to_medium()

The export is ALSO a plain caustica label volume, so code that predates this
module keeps working::

    from caustica.geometry import LabelVolume
    vol = LabelVolume.load_npz(path)               # same file, no new API

For comparative studies there is the STANDARD DATASET (:mod:`.dataset`): every
phantom at 0.25 mm on one common grid, front faces on one z-plane, transverse
centres aligned, pval on — built into ``data/phantoms/`` at the repo root::

    python -m uwcem_phantoms dataset            # build all nine
    python -m uwcem_phantoms dataset --verify   # re-check what is on disk

Data placement: everything downloaded, cached or ad-hoc exported stays inside
the package (``uwcem_phantoms/_data``); override with ``CAUSTICA_PHANTOM_DATA``.
Downloads are lazy — :func:`build` fetches what it needs on first use.

Attribution is not optional: see :data:`uwcem_phantoms.catalog.CITATION`,
which is copied into every export's metadata.
"""

from uwcem_phantoms.asset import PhantomAsset, load_phantom
from uwcem_phantoms.builder import build, plan
from uwcem_phantoms.catalog import (
    ACR_CLASSES,
    CATALOG,
    CITATION,
    NATIVE_DX,
    PHANTOM_IDS,
    PhantomInfo,
    fetch,
    fetch_all,
    status,
)
from uwcem_phantoms.dataset import (
    DATASET_DX_MM,
    DatasetPlan,
    build_dataset,
    dataset_filename,
    plan_dataset,
    verify_dataset,
)
from uwcem_phantoms.heterogeneity import PropertyVolumes
from uwcem_phantoms.orientation import CANONICAL_AXES, detect_propagation_axis, to_canonical
from uwcem_phantoms.reader import MEDIA_NUMBERS, MEDIA_TOKENS, RawPhantom, load_raw
from uwcem_phantoms.spec import (
    CropSpec,
    DomainSpec,
    HeterogeneitySpec,
    PhantomSpec,
    ResolutionSpec,
    SimplifySpec,
)
from uwcem_phantoms.tissue import (
    DEFAULT_TISSUES,
    TISSUE_MODELS,
    AcousticTissue,
    TissueTable,
    tissue_table,
)

__all__ = [
    "ACR_CLASSES",
    "CANONICAL_AXES",
    "CATALOG",
    "CITATION",
    "DATASET_DX_MM",
    "DEFAULT_TISSUES",
    "MEDIA_NUMBERS",
    "MEDIA_TOKENS",
    "NATIVE_DX",
    "PHANTOM_IDS",
    "TISSUE_MODELS",
    "AcousticTissue",
    "CropSpec",
    "DatasetPlan",
    "DomainSpec",
    "HeterogeneitySpec",
    "PhantomAsset",
    "PhantomInfo",
    "PhantomSpec",
    "PropertyVolumes",
    "RawPhantom",
    "ResolutionSpec",
    "SimplifySpec",
    "TissueTable",
    "build",
    "plan",
    "build_dataset",
    "dataset_filename",
    "detect_propagation_axis",
    "fetch",
    "fetch_all",
    "load_phantom",
    "load_raw",
    "plan_dataset",
    "status",
    "tissue_table",
    "to_canonical",
    "verify_dataset",
]
