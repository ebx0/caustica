"""Every extensible axis is a registry, not a closed list.

The load-bearing test is the plugin one: ONE fake *installed* distribution
declares an extension for all FIVE entry-point groups — solver, medium kind,
array kind, backend, report renderer — and a job uses them end to end, with
no caustica source change. The rest guards the properties that make the seam
safe: core implementations go through the same door, a broken plugin is
skipped rather than fatal, an unknown name says what IS registered, and
``import caustica`` never pays for the scan.
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

import caustica.solvers as solvers
from caustica.config import job as jobmod
from caustica.config.kinds import (
    ARRAY_GROUP,
    MEDIUM_GROUP,
    MediumKindConfig,
    array_kinds,
    medium_kinds,
)
from caustica.core.backend import backends, get_backend
from caustica.registry import (
    ARRAY_KIND_GROUP,
    BACKEND_GROUP,
    ENTRY_POINT_GROUPS,
    MEDIUM_KIND_GROUP,
    REPORT_RENDERER_GROUP,
    SOLVER_GROUP,
    same_definition,
)
from caustica.report.renderers import render_report, report_renderers
from caustica.solvers.registry import solver_registry

PLUGIN_SRC = '''
"""A pretend third-party package extending caustica on all five axes.

One medium kind, one array kind, one solver, one backend, one report
renderer — every one of them registered from OUTSIDE caustica, through the
entry-point groups alone.

Note what it imports: caustica.config.kinds (the seam) and the public array,
solver and backend names — never caustica.config.job, which is still being
imported when the entry points are scanned.
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from caustica.arrays import elements_array
from caustica.config.kinds import ArrayKindConfig, MediumKindConfig
from caustica.core.backend import Backend
from caustica.materials import Material
from caustica.medium import Medium
from caustica.solvers.kspace.linear import LinearKSpacePSTD


#: Proof-of-execution logs: the plugin's own code saying it ran. A stamp in
#: result.h5 can be produced by a look-alike; these cannot.
MEDIUM_BUILDS = []
SOLVER_RUNS = []
BACKEND_CALLS = []
RENDER_CALLS = []


class GelMediumConfig(MediumKindConfig):
    """Uniform coupling gel, defined entirely outside caustica."""

    kind: Literal["test_gel"] = "test_gel"
    c: float = Field(1520.0, gt=0.0)

    def c_min(self) -> float:
        return self.c

    def build(self, grid):
        MEDIUM_BUILDS.append(self.c)
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


class RelabelledLinearSolver(LinearKSpacePSTD):
    """A third party's solver.

    Deliberately a subclass of a built-in: the point under test is the
    REGISTRATION path, not new physics, and a genuinely new solver would
    make the test about the solver instead of about the seam. The registry
    key is the class's own `name`, so the entry-point line cannot disagree.
    """

    name = "test_linear"

    def run(self, *args, **kwargs):
        SOLVER_RUNS.append(self.name)
        return super().run(*args, **kwargs)


def make_test_backend():
    """A CPU backend under a name caustica has never heard of.

    A backend is a factory, not a class: `get_backend("test_numpy")` calls
    this. Anything lazy or expensive (a device probe, a CUDA import) belongs
    inside the call, exactly like caustica's own cupy factory.
    """
    BACKEND_CALLS.append("test_numpy")
    return Backend("test_numpy", np)


def render_text_report(outdir, *, preview_only=False):
    """A report renderer that owes nothing to matplotlib.

    Contract: (outdir, *, preview_only) -> the path a reader should open.
    """
    outdir = Path(outdir)
    if not (outdir / "result.h5").exists() and not (outdir / "preview.npz").exists():
        raise FileNotFoundError(f"nothing to report in {outdir}")
    metrics = {}
    mp = outdir / "metrics.json"
    if mp.exists():
        metrics = json.loads(mp.read_text(encoding="utf-8"))
    RENDER_CALLS.append(str(outdir))
    out = outdir / "REPORT.txt"
    peak = (metrics.get("peak") or {}).get("p_pa")  # the real caustica-metrics/1 key
    lines = [
        "third-party report",
        f"job: {metrics.get('job', outdir.name)}",
        f"preview_only: {preview_only}",
        f"peak_pa: {peak}",
    ]
    out.write_text("\\n".join(lines), encoding="utf-8")
    return out
'''

BROKEN_SRC = "raise RuntimeError('this plugin is broken on purpose')\n"

PLUGIN_MODULE = "caustica_test_plugin"

#: What the fake distribution declares — one line per axis. Two of the five
#: keys are read from the implementation (a kind's ``kind`` field, a solver's
#: ``name``); for the other three the entry-point NAME is the key, which is
#: why the backend and renderer lines are spelled as their runtime names.
PLUGIN_ENTRY_POINTS = f"""\
[{MEDIUM_KIND_GROUP}]
gel = {PLUGIN_MODULE}:GelMediumConfig

[{ARRAY_KIND_GROUP}]
ring = {PLUGIN_MODULE}:RingArrayConfig

[{SOLVER_GROUP}]
relabelled = {PLUGIN_MODULE}:RelabelledLinearSolver

[{BACKEND_GROUP}]
test_numpy = {PLUGIN_MODULE}:make_test_backend

[{REPORT_RENDERER_GROUP}]
test_text = {PLUGIN_MODULE}:render_text_report
"""

#: registry -> the names the fake plugin adds to it (teardown must undo them).
PLUGIN_NAMES = (
    (medium_kinds, ("test_gel",)),
    (array_kinds, ("test_ring",)),
    (solver_registry, ("test_linear",)),
    (backends, ("test_numpy",)),
    (report_renderers, ("test_text",)),
)


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
    """Make ``root`` importable, force a re-scan, and undo all of it after.

    All five registries are re-armed and torn down together: they share one
    ``discover()``, and a registry left holding a test entry would leak into
    every later test's ``available()``.
    """
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    for registry, _ in PLUGIN_NAMES:
        registry._loaded = False
    # Discover NOW rather than at the first question a test happens to ask:
    # re-arming `_loaded` alone leaves `_JOB_ADAPTER` built from the old kind
    # set, so a test that ran a job before touching `available()` died with
    # `union_tag_invalid`. A real install is discovered at
    # job.py import time; the fixture has to match that, not the test order.
    for registry, _ in PLUGIN_NAMES:
        registry.discover()
    jobmod._rebuild_kind_unions()
    try:
        yield
    finally:
        for registry, names in PLUGIN_NAMES:
            for name in names:
                registry._forget(name)
        with contextlib.suppress(ValueError):
            sys.path.remove(str(root))
        for module in modules:
            sys.modules.pop(module, None)
        for registry, _ in PLUGIN_NAMES:
            registry._loaded = True
        importlib.invalidate_caches()
        jobmod._rebuild_kind_unions()


@contextlib.contextmanager
def five_axis_plugin(root: Path):
    """Install the fake distribution under ``root`` and activate all of it."""
    _install_fake_dist(root, PLUGIN_MODULE, PLUGIN_SRC, PLUGIN_ENTRY_POINTS)
    with plugin_on_path(root, (PLUGIN_MODULE,)):
        yield


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


def core_kinds(registry) -> tuple[str, ...]:
    """The kinds caustica itself ships, ignoring any installed plugin.

    A third-party kind in the same environment is the whole point of the
    registry — it must not turn caustica's own suite red (found by the
    skeptical review: installing a plugin failed six tests that asserted the
    registry's exact contents).
    """
    return tuple(
        n for n in registry.available() if registry.get(n).__module__.startswith("caustica.")
    )


# ------------------------------------------------------------- the core kinds


def test_core_kinds_register_through_the_same_door():
    """No private path: caustica's own kinds ARE the registry's first clients."""
    assert core_kinds(medium_kinds) == (
        "homogeneous",
        "medium_volume",
        "scene",
        "volume_import",
    )
    assert core_kinds(array_kinds) == ("archimedean_spiral", "bowl", "elements")
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
    """The schema-level error keeps its original wording (registration order)."""
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

    assert core_kinds(medium_kinds) == ("homogeneous", "medium_volume", "scene", "volume_import")


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


def _restore_job_module(original: dict) -> None:
    """Undo a reload of ``caustica.config.job`` by putting the FIRST classes back.

    Reloading again does not undo a reload: it produces a THIRD set of class
    objects, and every module that did ``from caustica.config.job import X``
    at collection time still points at the first one. Anything built after
    that fails ``isinstance`` against the name the test imported -- a failure
    that lands in an unrelated file, several tests later.

    So: restore the module namespace verbatim, then re-register the original
    kind classes (the collision guard treats a same-module/qualname class as
    a redefinition, which is exactly what this is) and rebuild the models.
    """
    first = {
        obj.__qualname__: obj
        for obj in original.values()
        if isinstance(obj, type) and obj.__module__ == jobmod.__name__
    }
    vars(jobmod).clear()
    vars(jobmod).update(original)
    for registry in (medium_kinds, array_kinds):
        for name, reloaded in list(registry._items.items()):
            was = first.get(reloaded.__qualname__)
            if was is not None and same_definition(was, reloaded):
                registry.add(name, was)  # a redefinition, per the collision guard
    jobmod._rebuild_kind_unions()


def test_reloading_the_job_module_still_works():
    """`%autoreload 2` re-runs job.py, producing NEW classes with the SAME tags.

    Identity-only collision detection called that a name clash — reporting a
    class as colliding with itself, halfway through the reload, leaving the
    module a mix of new and stale objects with no error at the point of use.
    Editing caustica inside a notebook is a first-class workflow, so a reload
    has to survive.
    """
    before = medium_kinds.available()
    # Everything job.py defined on its FIRST import: other test modules (and
    # any live user code) hold these exact objects, so the reload has to be
    # handed back at the end -- see _restore_job_module.
    original = dict(vars(jobmod))
    importlib.reload(jobmod)
    try:
        assert medium_kinds.available() == before
        assert array_kinds.available() == array_kinds.available()
        # the registry now holds the RELOADED classes, and the schema follows
        assert medium_kinds.get("homogeneous") is jobmod.HomogeneousMediumConfig
        assert array_kinds.get("elements") is jobmod.ElementsArrayConfig
        job = jobmod._JOB_ADAPTER.validate_python(
            {
                **plugin_job_dict(),
                "medium": {"kind": "homogeneous"},
                "source": {
                    "kind": "array",
                    "array": {"kind": "bowl", "d_outer_mm": 10.0, "roc_mm": 12.0},
                    "apex_mm": [9.0, 9.0, 6.0],
                },
            }
        )
        assert isinstance(job.medium, jobmod.HomogeneousMediumConfig)
        # ...and the reload did not stack a second rebuild hook per registry
        assert len(medium_kinds._hooks) == 1
        assert len(array_kinds._hooks) == 1
    finally:
        _restore_job_module(original)


# ------------------------------------------------------------------ plugins


def test_entry_point_plugin_adds_a_medium_and_an_array_kind(tmp_path):
    """The acceptance: a stranger's package extends the job schema."""
    with five_axis_plugin(tmp_path):
        assert "test_gel" in medium_kinds.available()
        assert "test_ring" in array_kinds.available()
        # ...and caustica's own kind set is untouched, which is what keeps
        # caustica's suite green in an environment that has plugins installed.
        assert core_kinds(medium_kinds) == (
            "homogeneous",
            "medium_volume",
            "scene",
            "volume_import",
        )
        assert core_kinds(array_kinds) == ("archimedean_spiral", "bowl", "elements")

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


def test_entry_point_plugin_extends_all_five_axes(tmp_path):
    """The acceptance: ONE outside package, all five seams, end to end.

    Registration -> discovery -> execution for each axis, from a job file:

    * medium kind and array kind describe the setup,
    * the plugin SOLVER runs it (run A) and its name lands in the result,
    * the plugin BACKEND drives the arithmetic of a native solve (run B),
    * the plugin RENDERER turns the output folder into a report.

    Run B uses caustica's own solver on purpose. The runner only forwards
    ``backend=`` to the solvers in ``runner._NATIVE_SOLVERS`` (the kwave
    adapter rejects unknown kwargs by contract), so a THIRD-PARTY
    solver silently solves on the default backend today — see the
    review notes. Splitting run A and run B keeps this test honest about
    what each one proves rather than asserting a stamp nothing produced.
    """
    from caustica.runner import EXIT_OK, RunnerOptions, run_job_file

    h5py = pytest.importorskip("h5py")

    with five_axis_plugin(tmp_path):
        # ---- registration + discovery, on all five ----
        assert "test_gel" in medium_kinds.available()
        assert "test_ring" in array_kinds.available()
        assert "test_linear" in solvers.available()
        assert "test_numpy" in backends.available()
        assert "test_text" in report_renderers.available()
        # the plugin backend is a real, usable Backend value object
        assert get_backend("test_numpy").name == "test_numpy"

        def write(name: str, **over) -> Path:
            d = plugin_job_dict()
            d["name"] = name
            d.update(over)
            path = tmp_path / f"{name}.json"
            path.write_text(json.dumps(d), encoding="utf-8")
            return path

        def run(path: Path, out: Path) -> None:
            report = jobmod.validate_job(path)
            assert report.ok, report.render()
            code = run_job_file(path, RunnerOptions(out=out, measure=False, status_interval_s=0.0))
            assert code == EXIT_OK

        # The plugin's own bookkeeping: a stamp in result.h5 only echoes the
        # job file, so each axis also has to show that its code executed.
        plug = importlib.import_module(PLUGIN_MODULE)
        for log in (plug.MEDIUM_BUILDS, plug.SOLVER_RUNS, plug.BACKEND_CALLS, plug.RENDER_CALLS):
            log.clear()

        # ---- run A: the plugin's solver, on the plugin's medium + array ----
        out_a = tmp_path / "out_a"
        run(write("plugin-solver", solver="test_linear"), out_a)
        with h5py.File(out_a / "result.h5", "r") as hf:
            assert hf.attrs["solver"] == "test_linear"
        assert plug.SOLVER_RUNS == ["test_linear"]  # the plugin CLASS solved it
        # ...on the plugin's gel, not water (built twice: validate, then run)
        assert plug.MEDIUM_BUILDS and set(plug.MEDIUM_BUILDS) == {1520.0}
        meta_a = json.loads((out_a / "run_meta.json").read_text(encoding="utf-8"))
        assert meta_a["backend"] in ("numpy", "cupy")  # run A says nothing about backends
        # the plugin's medium and array really built the setup
        assert meta_a["derived"]["ring_radius_mm"] == 4.0
        assert not plug.BACKEND_CALLS

        # ---- run B: the plugin's backend, driving a native solve ----
        out_b = tmp_path / "out_b"
        run(write("plugin-backend", solver="linear", backend="test_numpy"), out_b)
        with h5py.File(out_b / "result.h5", "r") as hf:
            assert hf.attrs["backend"] == "test_numpy"
        # the runner resolved it, and so did the engine that did the arithmetic
        assert len(plug.BACKEND_CALLS) >= 2

        # ---- the plugin's renderer, on a real output folder ----
        rendered = render_report(out_b, renderer="test_text")
        assert rendered == out_b / "REPORT.txt"
        assert plug.RENDER_CALLS == [str(out_b)]
        text = rendered.read_text(encoding="utf-8")
        assert "third-party report" in text and "plugin-backend" in text
        # ...and it read a real number out of metrics.json, not a missing key
        peak = json.loads((out_b / "metrics.json").read_text(encoding="utf-8"))["peak"]["p_pa"]
        assert f"peak_pa: {peak}" in text
        # ...and caustica's own renderer is still the default
        assert report_renderers.get("matplotlib").__module__ == "caustica.report.renderers"

    # after teardown every axis is back to caustica's own set
    assert "test_linear" not in solvers.available()
    assert "test_numpy" not in backends.available()
    assert "test_text" not in report_renderers.available()


def test_unregistered_names_are_actionable_on_every_axis():
    """A name nobody registered must say what IS registered, and where to add."""
    with pytest.raises(ValueError, match="Unknown backend") as backend_exc:
        get_backend("torch")
    assert "numpy" in str(backend_exc.value)
    assert BACKEND_GROUP in str(backend_exc.value)

    with pytest.raises(KeyError) as renderer_exc:
        render_report(".", renderer="latex")
    assert "matplotlib" in str(renderer_exc.value)
    assert REPORT_RENDERER_GROUP in str(renderer_exc.value)
    # ...and it prints as prose, not as a quoted KeyError repr (the CLI
    # forwards this text straight to the user)
    assert not str(renderer_exc.value).startswith(chr(34) + "unknown")

    with pytest.raises(KeyError) as solver_exc:
        solvers.get("kzk")
    assert "westervelt" in str(solver_exc.value)
    assert SOLVER_GROUP in str(solver_exc.value)

    for registry, group in ((medium_kinds, MEDIUM_GROUP), (array_kinds, ARRAY_GROUP)):
        with pytest.raises(KeyError) as kind_exc:
            registry.get("no_such_kind")
        assert group in str(kind_exc.value)


def test_a_job_naming_an_unregistered_backend_is_a_config_error(tmp_path):
    """The Literal used to catch this; the registry has to keep catching it."""
    d = plugin_job_dict()
    d["medium"] = {"kind": "homogeneous"}
    d["source"]["array"] = {"kind": "bowl", "d_outer_mm": 10.0, "roc_mm": 12.0}
    d["backend"] = "test_numpy"  # real, but this process has no such plugin
    path = tmp_path / "job.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    report = jobmod.validate_job(path)
    assert not report.ok
    assert any("Unknown backend" in e and "cupy, numpy" in e for e in report.errors)


def test_entry_point_group_names_are_frozen():
    """Renaming a group breaks every plugin already installed. Pin them."""
    assert ENTRY_POINT_GROUPS == (
        "caustica.solvers",
        "caustica.medium_kinds",
        "caustica.array_kinds",
        "caustica.backends",
        "caustica.report_renderers",
    )
    # ...and every registry actually scans the group it claims to
    assert solver_registry.group == SOLVER_GROUP
    assert medium_kinds.group == MEDIUM_KIND_GROUP == MEDIUM_GROUP
    assert array_kinds.group == ARRAY_KIND_GROUP == ARRAY_GROUP
    assert backends.group == BACKEND_GROUP
    assert report_renderers.group == REPORT_RENDERER_GROUP


def test_core_implementations_come_from_the_registries():
    """No private path on the two axes the registries added.

    ``get_backend`` and ``caustica report`` must reach caustica's own
    implementations the same way they reach a plugin's — otherwise the seam
    is decoration and only the plugin path is ever exercised.
    """
    import caustica.core.backend as B

    assert set(backends.available()) >= {"numpy", "cupy"}
    assert backends.get("numpy") is B._numpy_backend
    assert backends.get("cupy") is B._cupy_backend
    assert get_backend("numpy") == B._numpy_backend()
    # "auto" is a POLICY over the registered backends, not a registered one
    assert "auto" not in backends.available()
    assert get_backend("auto").name in ("numpy", "cupy")

    assert report_renderers.available() == ("matplotlib",)
    assert render_report.__module__ == "caustica.report.renderers"


def test_listing_renderers_does_not_import_matplotlib():
    """Preview-writing (and now renderer discovery) stays matplotlib-free."""
    code = (
        "import sys\n"
        "from caustica.report import report_renderers, write_preview\n"
        "assert report_renderers.available() == ('matplotlib',), report_renderers.available()\n"
        "assert 'matplotlib' not in sys.modules, 'matplotlib imported by the registry'\n"
        "assert 'h5py' not in sys.modules, 'h5py imported by the registry'\n"
        "assert write_preview is not None\n"
        "print('clean')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout


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
        assert "boom" not in names
        assert core_kinds(medium_kinds) == (
            "homogeneous",
            "medium_volume",
            "scene",
            "volume_import",
        )
        assert "failed to load" in caplog.text


# ------------------------------------------------ what a factory must return


def test_a_backend_factory_is_held_to_its_registry_key():
    """The key and ``Backend.name`` must agree, or a run is mislabelled.

    Everything downstream reads ``Backend.name``, never the name that was
    asked for: the ``run_meta.json`` stamp, the ``backend`` attr in
    ``result.h5``, and the checkpoint fingerprint that decides whether a
    resume is the SAME run. The closed ``Literal`` used to make the mismatch
    unreachable; the registry does not, so ``get_backend`` checks.
    """
    import numpy as np

    from caustica.core.backend import Backend

    @backends.register("test_liar")
    def _liar() -> Backend:
        return Backend("numpy", np)  # not "test_liar"

    @backends.register("test_not_a_backend")
    def _not_a_backend():
        return "I am a string"

    try:
        with pytest.raises(ValueError, match="registry key and Backend.name must match"):
            get_backend("test_liar")
        with pytest.raises(TypeError, match="not a caustica.core.backend.Backend"):
            get_backend("test_not_a_backend")
    finally:
        backends._forget("test_liar")
        backends._forget("test_not_a_backend")


def test_two_anonymous_factories_still_collide():
    """The reload-tolerant guard must not become a free pass for lambdas.

    ``same_definition`` calls two objects the same definition when module and
    qualname match — true for a reloaded class, and ALSO true for any two
    module-level lambdas, which all answer to ``<lambda>``. Without this the
    second silently replaced the first under one name.
    """
    import numpy as np

    from caustica.core.backend import Backend

    first = lambda: Backend("test_anon", np)  # noqa: E731 - the point of the test
    second = lambda: Backend("test_anon", np)  # noqa: E731
    assert not same_definition(first, second)

    backends.add("test_anon", first)
    try:
        with pytest.raises(ValueError, match="already registered"):
            backends.add("test_anon", second)
    finally:
        backends._forget("test_anon")


def test_collision_messages_kept_their_pre_m10n_wording():
    """The shared base must not quietly reword an existing refusal."""
    with pytest.raises(ValueError, match=r"^solver name 'linear' already registered by "):

        @solvers.register
        class Duplicate(solvers.SolverBase):  # pragma: no cover - definition only
            name = "linear"
            caps = solvers.LinearKSpacePSTD.caps

            def run(self, *a, **k):
                raise NotImplementedError

    class Clash(MediumKindConfig):
        kind: Literal["scene"] = "scene"

    with pytest.raises(ValueError, match=r"^medium kind 'scene' already registered by Scene"):
        medium_kinds.register(Clash)


def test_the_cli_parser_stays_free_of_the_report_package():
    """Building the parser must not import numpy metrics for one default name.

    It reads one string; `caustica.report.__init__` eagerly imports the
    metrics and preview modules, so doing it at parser-build time doubled
    startup for EVERY command, `--help` included.
    """
    code = (
        "import sys\n"
        "from caustica.__main__ import build_parser\n"
        "build_parser()\n"
        "leaked = [m for m in ('caustica.report', 'scipy', 'h5py', 'matplotlib') "
        "if m in sys.modules]\n"
        "assert not leaked, leaked\n"
        "print('clean')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout


def test_a_misspelled_backend_is_refused_before_the_medium_is_built(tmp_path, monkeypatch):
    """`--backend` was an argparse `choices=`: a typo cost nothing.

    Now the registry owns the name, so the refusal has to happen before
    `build_job(with_medium=True)` — otherwise a misspelling is paid for with
    a full (possibly multi-GB) medium build.
    """
    from caustica.config import job as jm
    from caustica.runner import EXIT_CONFIG, RunnerOptions, run_job_file

    built = []
    real_build = jm.build_job
    monkeypatch.setattr(
        "caustica.runner.build_job",
        lambda *a, **k: (built.append(1), real_build(*a, **k))[1],
    )
    path = tmp_path / "job.json"
    path.write_text(json.dumps(plugin_job_dict() | {"medium": {"kind": "homogeneous"}}), "utf-8")
    code = run_job_file(path, RunnerOptions(out=tmp_path / "out", backend="torch", measure=False))
    assert code == EXIT_CONFIG
    assert not built, "the medium was built before the backend name was refused"


def test_the_kind_registries_answer_from_a_cold_import():
    """`from caustica.config.kinds import medium_kinds` is the documented seam.

    The core kinds are registered by caustica.config.job as it is imported,
    which nothing else forces — so a plugin author who imported only the seam
    used to be told "Available: (none)", or shown their own kind and nothing
    else. The registry imports job.py itself now, and only when asked
    (`import caustica` still does not).
    """
    code = textwrap.dedent(
        """
        import sys
        from caustica.config.kinds import array_kinds, medium_kinds
        assert "caustica.config.job" not in sys.modules, "the seam imported job.py eagerly"
        assert medium_kinds.available() == (
            "homogeneous", "medium_volume", "scene", "volume_import"
        ), medium_kinds.available()
        assert array_kinds.available() == ("archimedean_spiral", "bowl", "elements")
        try:
            medium_kinds.get("nope")
        except KeyError as exc:
            assert "homogeneous" in str(exc), str(exc)
        else:
            raise AssertionError("unknown kind did not raise")
        print("clean")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout


# -------------------------------------------------------------------- lazy


def test_import_caustica_does_not_scan_entry_points():
    """The scan costs an importlib.metadata sweep; `import caustica` must not.

    Structural, not timing-based, and covering all five axes: the modules
    that would trigger a scan must be absent from a fresh interpreter that
    only did ``import caustica`` — and the one registry that IS constructed
    eagerly (backends, because `caustica.core.backend` is on the import
    path) must not have scanned.

    ``"importlib.metadata" not in sys.modules`` would be the sharper probe
    and is NOT usable: pydantic pulls it in from ``pydantic.plugin._loader``
    as soon as a model class is built — which ``import caustica`` does — so
    it is already there whatever caustica's registries do. (Bare
    ``import pydantic`` alone does not; measured 2026-08-22.)
    """
    code = textwrap.dedent(
        """
        import sys
        import caustica
        for lazy in ("caustica.config.job", "caustica.solvers", "caustica.report.renderers"):
            assert lazy not in sys.modules, f"{lazy} imported eagerly"
        registries = []
        for mod, names in (
            ("caustica.config.kinds", ("medium_kinds", "array_kinds")),
            ("caustica.core.backend", ("backends",)),
            ("caustica.solvers.registry", ("solver_registry",)),
            ("caustica.report.renderers", ("report_renderers",)),
        ):
            m = sys.modules.get(mod)
            if m is not None:
                registries += [getattr(m, n) for n in names]
        assert registries, "no registry was even constructed - the probe is broken"
        scanned = [r.label for r in registries if r._loaded]
        assert not scanned, f"entry points scanned on import: {scanned}"
        print("clean")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout
