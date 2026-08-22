"""M10m/K15 gate: medium and array kinds are a registry, not a closed union.

The load-bearing test is the plugin one: a fake *installed* distribution
declares a medium kind AND an array kind through the entry-point groups, and
a job using both passes ``validate`` — with no caustica source change. The
rest guards the properties that make the seam safe: core kinds go through
the same door, a broken plugin is skipped rather than fatal, an unknown kind
says what IS registered, and ``import caustica`` never pays for the scan.
"""

import contextlib
import importlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Literal

import pytest

from caustica.config import job as jobmod
from caustica.config.kinds import (
    ARRAY_GROUP,
    MEDIUM_GROUP,
    MediumKindConfig,
    array_kinds,
    medium_kinds,
)

PLUGIN_SRC = '''
"""A pretend third-party package: one medium kind, one array kind.

Note what it imports: caustica.config.kinds (the seam) and the public array
helpers — never caustica.config.job, which is still being imported when the
entry points are scanned.
"""

from typing import Literal

import numpy as np
from pydantic import Field

from caustica.arrays import elements_array
from caustica.config.kinds import ArrayKindConfig, MediumKindConfig
from caustica.materials import Material
from caustica.medium import Medium


class GelMediumConfig(MediumKindConfig):
    """Uniform coupling gel, defined entirely outside caustica."""

    kind: Literal["test_gel"] = "test_gel"
    c: float = Field(1520.0, gt=0.0)

    def c_min(self) -> float:
        return self.c

    def build(self, grid):
        mat = Material(name="test_gel", c=self.c, rho=1020.0, alpha_np_m=1.0, beta=0.0)
        return Medium.homogeneous(grid.shape, mat)


class RingArrayConfig(ArrayKindConfig):
    """A flat-ring transducer nobody in caustica has heard of."""

    kind: Literal["test_ring"] = "test_ring"
    n_elements: int = Field(6, ge=1)
    ring_radius_mm: float = Field(4.0, gt=0.0)
    elem_radius_mm: float = Field(1.2, gt=0.0)
    roc_mm: float = Field(12.0, gt=0.0)

    def focal_length_mm(self) -> float:
        return self.roc_mm

    def _array(self):
        th = np.linspace(0.0, 2.0 * np.pi, self.n_elements, endpoint=False)
        r = self.ring_radius_mm * 1e-3
        pos = np.column_stack((r * np.cos(th), r * np.sin(th), np.zeros_like(th)))
        return elements_array(
            positions=pos,
            elem_radius=self.elem_radius_mm * 1e-3,
            focal_length=self.roc_mm * 1e-3,
        )

    def derived(self) -> dict:
        return {
            "n_elements": float(self.n_elements),
            "ring_radius_mm": self.ring_radius_mm,
        }

    def build_source(self, grid, drive, apex_vox, focus, phases_rad):
        arr = self._array()
        asrc = arr.voxelize(
            grid, apex_vox, f0=drive.f0_hz, amplitude=drive.amplitude_pa, phases=None
        )
        extra = dict(self.derived())
        extra["source_voxels"] = int(asrc.source.n_points)
        return asrc.source, extra
'''

BROKEN_SRC = "raise RuntimeError('this plugin is broken on purpose')\n"


def _install_fake_dist(root: Path, module: str, source: str, entry_points: str) -> None:
    """Write an importable module + a .dist-info so importlib.metadata sees it."""
    (root / f"{module}.py").write_text(source, encoding="utf-8")
    info = root / f"{module}-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {module.replace('_', '-')}\nVersion: 0.1\n",
        encoding="utf-8",
    )
    (info / "entry_points.txt").write_text(entry_points, encoding="utf-8")


@contextlib.contextmanager
def plugin_on_path(root: Path, modules: tuple[str, ...]):
    """Make ``root`` importable, force a re-scan, and undo all of it after."""
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    medium_kinds._loaded = False
    array_kinds._loaded = False
    try:
        yield
    finally:
        for name in ("test_gel",):
            medium_kinds._forget(name)
        for name in ("test_ring",):
            array_kinds._forget(name)
        with contextlib.suppress(ValueError):
            sys.path.remove(str(root))
        for module in modules:
            sys.modules.pop(module, None)
        medium_kinds._loaded = True
        array_kinds._loaded = True
        importlib.invalidate_caches()
        jobmod._rebuild_kind_unions()


def plugin_job_dict() -> dict:
    return {
        "format": jobmod.JOB_FORMAT,
        "kind": "explicit",
        "name": "plugin-mini",
        "medium": {"kind": "test_gel", "c": 1520.0},
        "grid": {
            "ndim": 3,
            "dx_mm": 0.5,
            "size_mm": [18, 18, 24],
            "pml": {"thickness_mm": 3.0},
        },
        "source": {
            "kind": "array",
            "array": {"kind": "test_ring", "n_elements": 6, "roc_mm": 12.0},
            "apex_mm": [9.0, 9.0, 6.0],
        },
        "drive": {"f0_mhz": 0.8, "amplitude_kpa": 100.0},
        "run": {"spec": {"min_settle_periods": 2, "max_settle_periods": 6}, "harmonics": [1]},
        "solver": "linear",
    }


# ------------------------------------------------------------- the core kinds


def test_core_kinds_register_through_the_same_door():
    """No private path: caustica's own kinds ARE the registry's first clients."""
    assert medium_kinds.available() == (
        "homogeneous",
        "medium_volume",
        "scene",
        "volume_import",
    )
    assert array_kinds.available() == ("archimedean_spiral", "bowl", "elements")
    assert medium_kinds.get("scene") is jobmod.SceneMediumConfig
    assert array_kinds.get("elements") is jobmod.ElementsArrayConfig


def test_unknown_kind_lists_what_is_registered():
    with pytest.raises(KeyError, match="medium_volume") as exc:
        medium_kinds.get("phantom_dataset")  # the kind M10k removed
    assert MEDIUM_GROUP in str(exc.value)  # ...and how to add your own
    with pytest.raises(KeyError, match="archimedean_spiral") as exc:
        array_kinds.get("linear_array")
    assert ARRAY_GROUP in str(exc.value)


def test_unknown_kind_in_a_job_file_names_the_expected_tags(tmp_path):
    """The schema-level error keeps its M10b wording (registration order)."""
    from pydantic import ValidationError

    d = plugin_job_dict()
    d["medium"] = {"kind": "test_gel"}  # plugin NOT installed here
    with pytest.raises(ValidationError, match="does not match any of the expected tags"):
        jobmod._JOB_ADAPTER.validate_python(d)
    p = tmp_path / "job.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    report = jobmod.validate_job(p)
    assert not report.ok
    assert any(
        "'homogeneous', 'scene', 'volume_import', 'medium_volume'" in e for e in report.errors
    )


def test_registration_refusals_teach():
    class NoTag(MediumKindConfig):
        pass

    with pytest.raises(ValueError, match="must be annotated"):
        medium_kinds.register(NoTag)

    with pytest.raises(TypeError, match="must subclass"):
        medium_kinds.register(jobmod.BowlArrayConfig)  # an ARRAY kind

    class Clash(MediumKindConfig):
        kind: Literal["scene"] = "scene"

    with pytest.raises(ValueError, match="already registered by SceneMediumConfig"):
        medium_kinds.register(Clash)

    # a mistyped default silently constructs the WRONG kind in Python
    class Mismatch(MediumKindConfig):
        kind: Literal["mismatch"] = "mismatched"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="must default to"):
        medium_kinds.register(Mismatch)

    assert medium_kinds.available() == ("homogeneous", "medium_volume", "scene", "volume_import")


def test_registration_is_all_or_nothing_when_wiring_fails():
    """A kind that cannot be wired into the job models must not linger.

    Otherwise `available()` and `caustica schema` advertise a kind that
    `validate` refuses — and inside discover() the failure is logged as
    "plugin failed to load", which makes the inconsistency look explained.
    """

    class Late(MediumKindConfig):
        kind: Literal["late_kind"] = "late_kind"

    boom = RuntimeError("rebuild exploded")

    def bad_hook() -> None:
        raise boom

    medium_kinds._hooks.append(bad_hook)
    try:
        with pytest.raises(RuntimeError, match="rebuild exploded"):
            medium_kinds.register(Late)
    finally:
        medium_kinds._hooks.remove(bad_hook)
    assert "late_kind" not in medium_kinds.available()
    assert "late_kind" not in json.dumps(jobmod.job_schema())


# ------------------------------------------------------------------ plugins


def test_entry_point_plugin_adds_a_medium_and_an_array_kind(tmp_path):
    """The M10m acceptance: a stranger's package extends the job schema."""
    _install_fake_dist(
        tmp_path,
        "caustica_test_plugin",
        PLUGIN_SRC,
        textwrap.dedent(
            f"""\
            [{MEDIUM_GROUP}]
            gel = caustica_test_plugin:GelMediumConfig

            [{ARRAY_GROUP}]
            ring = caustica_test_plugin:RingArrayConfig
            """
        ),
    )
    with plugin_on_path(tmp_path, ("caustica_test_plugin",)):
        assert "test_gel" in medium_kinds.available()
        assert "test_ring" in array_kinds.available()

        p = tmp_path / "job.json"
        p.write_text(json.dumps(plugin_job_dict()), encoding="utf-8")
        report = jobmod.validate_job(p)
        assert report.ok, report.render()

        # the medium really came from the plugin (gel, not water)
        built = jobmod.build_job(*jobmod.load_job(p))
        assert float(built.medium.c.max()) == pytest.approx(1520.0)
        assert built.derived["ring_radius_mm"] == 4.0

        # and `caustica schema` grew both branches
        schema = jobmod.job_schema()
        assert "test_gel" in json.dumps(schema)
        assert "test_ring" in json.dumps(schema)

    # ...and after teardown the core schema is exactly what it was
    assert "test_gel" not in medium_kinds.available()
    assert "test_gel" not in json.dumps(jobmod.job_schema())


def test_a_broken_plugin_is_skipped_not_fatal(tmp_path, caplog):
    _install_fake_dist(
        tmp_path,
        "caustica_broken_plugin",
        BROKEN_SRC,
        f"[{MEDIUM_GROUP}]\nboom = caustica_broken_plugin:Nope\n",
    )
    with plugin_on_path(tmp_path, ("caustica_broken_plugin",)):
        with caplog.at_level("WARNING", logger="caustica"):
            names = medium_kinds.available()
        assert names == ("homogeneous", "medium_volume", "scene", "volume_import")
        assert "failed to load" in caplog.text


# -------------------------------------------------------------------- lazy


def test_import_caustica_does_not_scan_entry_points():
    """The scan costs an importlib.metadata sweep; `import caustica` must not.

    Structural, not timing-based: the job module (which builds the unions and
    therefore triggers discovery) must be absent from a fresh interpreter
    that only did ``import caustica``.
    """
    code = textwrap.dedent(
        """
        import sys
        import caustica
        assert "caustica.config.job" not in sys.modules, "job.py imported eagerly"
        scanned = []
        if "caustica.config.kinds" in sys.modules:
            k = sys.modules["caustica.config.kinds"]
            scanned = [r.label for r in (k.medium_kinds, k.array_kinds) if r._loaded]
        assert not scanned, f"entry points scanned on import: {scanned}"
        print("clean")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout
