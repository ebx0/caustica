"""Pydantic configuration models (JSON-serializable simulation setup).

``job`` (the ``caustica-job/1`` tree) loads lazily: it pulls in geometry,
arrays and sources, and eager-importing it here would create an import cycle
(geometry.configs itself imports ``caustica.config.models``).
"""

from caustica.config.models import CausticaModel, GridConfig, PMLConfig

_JOB_NAMES = frozenset(
    {
        "JOB_FORMAT",
        "ArraySourceConfig",
        "BowlArrayConfig",
        "BuiltJob",
        "DriveConfig",
        "ExplicitJobConfig",
        "FocusConfig",
        "HomogeneousMediumConfig",
        "JobError",
        "JobReport",
        "MediumVolumeConfig",
        "OutputConfig",
        "RunConfig",
        "SceneMediumConfig",
        "SpiralArrayConfig",
        "VolumeImportMediumConfig",
        "build_job",
        "dump_job",
        "load_job",
        "validate_job",
    }
)

__all__ = ["GridConfig", "CausticaModel", "PMLConfig", *sorted(_JOB_NAMES)]


def __getattr__(name: str):  # PEP 562: break the geometry <-> config cycle
    if name in _JOB_NAMES:
        from caustica.config import job

        return getattr(job, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
