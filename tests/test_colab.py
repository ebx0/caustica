"""M10f: the Colab bridge, and the notebook that must never need editing.

Two halves. The first freezes ``notebooks/colab_run.ipynb`` cell by cell: the
notebook is a *thin* front for :mod:`caustica.colab`, so a logic change has to
arrive through ``pip install -U`` with a notebook diff of exactly zero. The
template below is the lock — editing the notebook without editing this file
fails, which is the point.

The second half is everything about the bridge that a machine with no GPU can
still prove: the fake-Colab detection, the two separate refusals, where output
lands, and — the one that matters most — that ``run_job`` adds nothing to and
takes nothing from the runner's contract (exit codes, ``error.json``,
``cancel``). The Colab-gated half (a real GPU, a real session) is deliberately
NOT simulated here; it is a live-session criterion in MILESTONES M10f.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import types
from pathlib import Path

import pytest
from tests.test_runner import mini_job

import caustica.colab as colab
from caustica.facade import SimulationError
from caustica.runner import CANCEL_FILE, ERROR_FILE, EXIT_OK

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "colab_run.ipynb"

# --------------------------------------------------------------------------
# The frozen notebook template. Byte-for-byte what each cell must contain.
# --------------------------------------------------------------------------

CELL_MARKDOWN_INTRO = """# caustica on Colab

Run one `caustica-job/1` file on a Colab GPU and look at the result — without editing any logic.

**You edit exactly one line: `CONFIG`, in the third cell.** Everything else lives in the
`caustica` package (`caustica.colab`), so improvements arrive with `pip install -U` and this
notebook's diff stays empty.

Before you start: **Runtime -> Change runtime type -> Hardware accelerator: GPU.** A CPU runtime
is refused with the fix, before anything is downloaded, created or built.

Results land in `/content/runs/<job>` — Colab's session disk, which survives a runtime restart
but not a VM teardown. caustica mounts no cloud storage of its own: if you want a run to outlive
the session, mount your own in a cell and pass that folder as `run_job(CONFIG, out=...)`.

Docs: [job format](https://github.com/ebx0/caustica/blob/master/docs/job_reference.md) ·
[conventions](https://github.com/ebx0/caustica/blob/master/docs/conventions.md) ·
[what a program may rely on](https://github.com/ebx0/caustica/blob/master/docs/gui_contract.md)"""

CELL_INSTALL = """\
# Setup. Colab GPU runtimes already ship cupy, so there is no GPU extra to install here.
# If pip upgrades numpy, Colab will ask you to restart the session: restart, then re-run this cell.
!pip install -q "caustica[report] @ git+https://github.com/ebx0/caustica\""""

# The URL is split across two source lines only to stay inside the line limit; the
# JOINED text is what the notebook must contain, character for character.
CELL_CONFIG = (
    "# The ONLY line you edit: a caustica-job/1 file — a path under /content, or an https URL.\n"
    'CONFIG = "https://raw.githubusercontent.com/ebx0/caustica/master'
    '/src/caustica/examples/water_bowl_mini.json"'
)

CELL_RUN = """from caustica.colab import run_job

# Environment verdict first (a GPU, or an actionable refusal that prepares nothing),
# then the planner, the pre-run gates, the solve and the audit stamp.
outdir = run_job(CONFIG)"""

CELL_SHOW = """from caustica.colab import show

show(outdir)  # the run's own metrics.json numbers, then the report figures, inline"""

TEMPLATE: tuple[tuple[str, str], ...] = (
    ("markdown", CELL_MARKDOWN_INTRO),
    ("code", CELL_INSTALL),
    ("code", CELL_CONFIG),
    ("code", CELL_RUN),
    ("code", CELL_SHOW),
)


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _python_cells() -> list[str]:
    """Code cells that are plain Python — the shell/magic cell is not parsable."""
    return [
        _source(c)
        for c in _notebook()["cells"]
        if c["cell_type"] == "code"
        and not any(line.lstrip().startswith(("!", "%")) for line in _source(c).splitlines())
    ]


# ------------------------------------------------- the notebook is frozen


def test_notebook_cells_match_the_frozen_template():
    """THE contract test: the notebook is a template, not a workspace.

    Any behaviour change must land in ``caustica.colab`` and reach the user
    through ``pip install -U``. If this fails, the notebook grew logic —
    move it into the package, or change the template here on purpose.
    """
    nb = _notebook()
    cells = nb["cells"]
    assert nb["nbformat"] == 4
    assert len(cells) == len(TEMPLATE), f"cell count changed: {len(cells)} != {len(TEMPLATE)}"
    for i, (cell, (kind, text)) in enumerate(zip(cells, TEMPLATE, strict=True)):
        assert cell["cell_type"] == kind, f"cell {i} is a {cell['cell_type']}, expected {kind}"
        assert _source(cell) == text, f"cell {i} content drifted from the template"


def test_notebook_carries_no_outputs_and_no_execution_counts():
    """Stored outputs are how a notebook's diff stops being zero."""
    for cell in _notebook()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_notebook_has_exactly_one_editable_line():
    """One assignment a user is meant to touch, and it is CONFIG.

    ``outdir = run_job(CONFIG)`` is an assignment too, so the rule is
    narrower and checkable: exactly one cell assigns a *literal*, and that
    literal is the job the run consumes.
    """
    literal_assignments = []
    for source in _python_cells():
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                literal_assignments.append(node.targets[0].id)
    assert literal_assignments == ["CONFIG"]


def test_notebook_holds_no_logic_beyond_the_two_bridge_calls():
    """Everything the notebook does is: import two names, call them.

    A second caustica import, a helper function or an inline loop would mean
    logic living where ``pip install -U`` cannot reach it.
    """
    imported: set[str] = set()
    called: set[str] = set()
    defined = 0
    for source in _python_cells():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                assert node.module == "caustica.colab", f"unexpected import: {node.module}"
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.Import):
                raise AssertionError("plain imports keep logic in the notebook")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.For, ast.While, ast.If)):
                defined += 1
    assert imported == {"run_job", "show"}
    assert called == {"run_job", "show"}
    assert defined == 0


def test_notebook_never_mounts_anything():
    """PLAN K12, at the notebook level: no mount, no cloud-drive path."""
    text = NOTEBOOK.read_text(encoding="utf-8").lower()
    for forbidden in ("drive.mount", "/content/drive", "import google", "from google"):
        assert forbidden not in text, f"the notebook mentions {forbidden!r}"


def test_notebook_installs_from_the_public_repo_and_never_a_gpu_extra():
    """Colab ships cupy; installing a GPU extra there is wasted minutes (K6)."""
    install = _source(_notebook()["cells"][1])
    assert "pip install" in install and "git+https://github.com/ebx0/caustica" in install
    assert "[gpu]" not in install
    assert "cupy" not in install.split("!pip", 1)[1]  # only the comment may say cupy


# ------------------------------------------------ the library carries no Drive


def test_the_library_has_no_drive_code_and_never_imports_google_colab():
    """M10f's grep criterion, encoded.

    Two separate rules, because they are two separate risks:

    * **Google Drive**: zero occurrences anywhere under ``src/caustica``. Not
      a mount, not a path, not a retry loop (PLAN K12).
    * **``google.colab``**: may be *probed* (is it already in ``sys.modules``?)
      but never imported, and only in the modules that legitimately answer
      "where am I": the environment policy, the progress renderer, and the
      bridge's own prose.

    (Note that a job's ``drive`` section — f0 and amplitude — is the acoustic
    drive and has nothing to do with any of this; the patterns below are
    deliberately specific so the two never get confused.)
    """
    drive = re.compile(r"drive\.mount|/content/drive", re.IGNORECASE)
    imports_colab = re.compile(r"^\s*(?:import|from)\s+google\b")
    drive_hits, import_hits, mentions = [], [], set()
    for path in sorted((REPO / "src" / "caustica").rglob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if drive.search(line):
                drive_hits.append(f"{path.name}:{n}: {line.strip()}")
            if imports_colab.match(line):
                import_hits.append(f"{path.name}:{n}: {line.strip()}")
            if "google.colab" in line:
                mentions.add(path.name)
    assert drive_hits == [], f"Google Drive leaked into the library: {drive_hits}"
    assert import_hits == [], f"google.colab is imported, not probed: {import_hits}"
    assert mentions <= {"env.py", "progress.py", "colab.py"}, f"new colab assumption: {mentions}"


def test_the_bridge_reuses_the_one_colab_probe():
    """One definition of "are we on Colab", shared with ``require_gpu``.

    A second probe here could disagree with the machine-specific message
    :mod:`caustica.env` picks for the same runtime.
    """
    from caustica import env

    assert colab._on_colab is env._on_colab


# ------------------------------------------------------- fake-Colab detection


@pytest.fixture
def fake_colab(monkeypatch):
    """Make :func:`caustica.env._on_colab` say yes, the way Colab does."""
    monkeypatch.setitem(sys.modules, "google.colab", types.ModuleType("google.colab"))
    return True


@pytest.fixture
def not_colab(monkeypatch):
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    for key in [k for k in list(sys.modules) if k.startswith("google.colab")]:
        monkeypatch.delitem(sys.modules, key, raising=False)
    for key in [k for k in dict(__import__("os").environ) if k.startswith("COLAB_")]:
        monkeypatch.delenv(key, raising=False)
    return False


def test_detects_a_colab_runtime_through_sys_modules(fake_colab):
    assert colab.on_colab() is True
    assert colab.session_root() == colab.CONTENT_ROOT


def test_detects_a_colab_runtime_through_the_env_var(not_colab, monkeypatch):
    assert colab.on_colab() is False
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-x")
    assert colab.on_colab() is True


def test_output_defaults_under_content_on_colab(fake_colab):
    assert colab.default_out("/content/my_job.json") == Path("/content/runs/my_job")


def test_output_defaults_under_the_cwd_locally(not_colab, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert colab.default_out(tmp_path / "elsewhere" / "mini.json") == tmp_path / "runs" / "mini"


def test_output_default_from_a_url_uses_the_url_stem(fake_colab):
    url = "https://example.org/a/b/water_bowl_mini.json?ref=main"
    assert colab.default_out(url) == Path("/content/runs/water_bowl_mini")


# ------------------------------------------- the two refusals are two messages


@pytest.fixture
def no_cupy_at_all(monkeypatch):
    """cupy is not installed: nothing is known about the device yet."""
    monkeypatch.setattr(colab, "cupy_available", lambda: False)
    monkeypatch.setattr(colab, "_cupy_installed", lambda: False)


@pytest.fixture
def cupy_but_no_device(monkeypatch):
    """cupy imports fine; the runtime simply has no GPU."""
    monkeypatch.setattr(colab, "cupy_available", lambda: False)
    monkeypatch.setattr(colab, "_cupy_installed", lambda: True)
    # env.require_gpu asks the backend module itself, not the bridge's copy.
    monkeypatch.setattr("caustica.env.cupy_available", lambda: False)


def test_missing_cupy_and_a_cpu_runtime_are_two_different_messages(
    fake_colab, no_cupy_at_all, monkeypatch
):
    """K6: they have different fixes, so they must not share a sentence."""
    with pytest.raises(RuntimeError) as missing:
        colab.require_gpu_here()
    monkeypatch.setattr(colab, "_cupy_installed", lambda: True)
    monkeypatch.setattr("caustica.env.cupy_available", lambda: False)
    with pytest.raises(RuntimeError) as no_device:
        colab.require_gpu_here()

    a, b = str(missing.value), str(no_device.value)
    assert a != b
    # (a) the cupy fact, and NOT a claim about the device
    assert "cupy is not installed" in a
    assert "no CUDA device" not in a
    # (b) the device fact, and NOT an install instruction (no pip can fix it)
    assert "no CUDA device" in b
    assert "cupy is not installed" not in b
    assert "pip install cupy" not in b
    # both name the Runtime menu, because on Colab that is where both are fixed
    assert "Change runtime type" in a and "Change runtime type" in b


def test_missing_cupy_off_colab_names_the_install_the_user_runs(not_colab, no_cupy_at_all):
    with pytest.raises(RuntimeError) as exc:
        colab.require_gpu_here()
    message = str(exc.value)
    assert "cupy is not installed on this machine" in message
    assert "pip install cupy-cuda12x" in message
    assert "caustica never installs it for you" in message
    assert "Change runtime type" not in message  # there is no Runtime menu here


def test_the_device_refusal_off_colab_is_env_policys_own_message(not_colab, cupy_but_no_device):
    with pytest.raises(RuntimeError) as exc:
        colab.require_gpu_here()
    message = str(exc.value)
    assert "no usable CUDA device was found on this machine" in message
    assert "cupy is not installed" not in message


def test_a_usable_gpu_passes_the_gate(monkeypatch):
    monkeypatch.setattr(colab, "cupy_available", lambda: True)
    assert colab.require_gpu_here() is None


# --------------------------------------------- refuse BEFORE preparing anything


def test_a_gpu_less_runtime_is_refused_before_anything_is_prepared(
    fake_colab, no_cupy_at_all, tmp_path, capsys
):
    """The order is the feature: a bad runtime costs a message, not a download."""

    def never(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("the bridge prepared something before checking the runtime")

    out = tmp_path / "runs" / "nope"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(colab, "_fetch", never)
        mp.setattr(colab, "run_job_file", never)
        with pytest.raises(RuntimeError, match="cupy is not installed"):
            colab.run_job("https://example.invalid/job.json", out=out)
    assert not out.exists()  # no folder, no download, no medium build
    # ...and the environment was printed first, so the user sees what it saw.
    assert "caustica" in capsys.readouterr().out


# --------------------------------------- run_job leaves the runner contract alone


@pytest.fixture
def gpu_present(monkeypatch):
    """Pass the bridge's gate; the RUN still resolves numpy underneath.

    Patching the bridge's own name (not the backend module's) is deliberate:
    the job then really executes on this CPU, so the tests below exercise the
    genuine runner path rather than a mock of it.
    """
    monkeypatch.setattr(colab, "cupy_available", lambda: True)


def test_run_job_produces_the_ordinary_run_folder_and_returns_it(gpu_present, tmp_path):
    out = colab.run_job(mini_job(tmp_path), out=tmp_path / "out", measure=False, progress=None)
    assert out == tmp_path / "out"
    for name in ("job.json", "plan.json", "status.json", "result.h5", "metrics.json"):
        assert (out / name).is_file(), f"the runner's {name} is missing"
    assert not (out / ERROR_FILE).exists()  # a successful run writes no failure record


def test_run_job_builds_the_runner_options_a_notebook_wants(
    gpu_present, not_colab, tmp_path, monkeypatch
):
    seen: dict = {}

    def capture(job_path, opts):
        seen["job"], seen["opts"] = Path(job_path), opts
        return EXIT_OK

    monkeypatch.setattr(colab, "run_job_file", capture)
    monkeypatch.chdir(tmp_path)
    job = mini_job(tmp_path)
    out = colab.run_job(job, gpu="H100")
    assert seen["job"] == job
    assert seen["opts"].gpu == "H100"
    assert seen["opts"].progress == "auto"  # an entry point for a human, like simulate()
    assert Path(seen["opts"].out) == tmp_path / "runs" / "mini" == out


def test_a_caller_option_wins_over_the_bridges_default(gpu_present, tmp_path, monkeypatch):
    seen: dict = {}

    def capture(job_path, opts):
        seen["opts"] = opts
        return EXIT_OK

    monkeypatch.setattr(colab, "run_job_file", capture)
    colab.run_job(mini_job(tmp_path), out=tmp_path / "o", progress=None, resume=True)
    assert seen["opts"].progress is None
    assert seen["opts"].resume is True


def test_an_unknown_option_names_the_accepted_ones(gpu_present, tmp_path):
    with pytest.raises(TypeError) as exc:
        colab.run_job(mini_job(tmp_path), out=tmp_path / "o", nonesuch=1)
    message = str(exc.value)
    assert "nonesuch" in message
    assert "resume" in message and "max_hours" in message and "preview_only" in message


def test_run_job_passes_dry_run_through_untouched(gpu_present, tmp_path):
    out = colab.run_job(mini_job(tmp_path), out=tmp_path / "o", dry_run=True, measure=False)
    assert (out / "plan.json").is_file()
    assert not (out / "result.h5").exists()  # a probe, not an attempt


def test_a_failed_run_raises_with_the_runners_own_exit_code(gpu_present, tmp_path):
    """Exit codes are the queue's API — the bridge carries them, never remaps."""
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "caustica-job/1", "kind": "explicit"}', encoding="utf-8")
    out = tmp_path / "o"
    with pytest.raises(SimulationError) as exc:
        colab.run_job(bad, out=out)
    assert exc.value.exit_code == 2  # EXIT_CONFIG
    assert "exited 2" in str(exc.value)
    record = json.loads((out / ERROR_FILE).read_text(encoding="utf-8"))
    assert record["format"] == "caustica-error/1" and record["exit_code"] == 2
    assert record["message"] in str(exc.value)  # quoted, not re-diagnosed


def test_a_cancelled_run_reports_exit_5_and_how_to_continue(gpu_present, tmp_path):
    """The GUI stop signal, driven from a notebook: pause, do not lose."""
    out = tmp_path / "o"

    def press_stop_at_period_3(event):
        if event["period"] >= 3:
            (out / CANCEL_FILE).touch()

    with pytest.raises(SimulationError) as exc:
        colab.run_job(mini_job(tmp_path), out=out, measure=False, progress=press_stop_at_period_3)
    assert exc.value.exit_code == 5  # EXIT_INTERRUPTED
    assert (out / "checkpoint.npz").is_file()
    assert not (out / ERROR_FILE).exists()  # stopping is not failing
    assert "resume=True" in str(exc.value)
    # and the advice is true: the resume finishes the run.
    assert colab.run_job(mini_job(tmp_path), out=out, measure=False, progress=None, resume=True)
    assert (out / "result.h5").is_file()


def test_an_oom_refusal_adds_the_colab_lever(gpu_present, tmp_path):
    with pytest.raises(SimulationError) as exc:
        colab.run_job(mini_job(tmp_path), out=tmp_path / "o", measure=False, vram_limit_gib=1e-6)
    assert exc.value.exit_code == 3
    message = str(exc.value)
    assert "REFUSED before solving" in message  # the runner's own headline
    assert "Change runtime type" in message  # the bridge's Colab-specific lever


# --------------------------------------------------------- URL jobs and output


def test_a_url_job_is_downloaded_next_to_the_run(gpu_present, not_colab, tmp_path, monkeypatch):
    payload = mini_job(tmp_path).read_bytes()

    class FakeResponse:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(colab.urllib.request, "urlopen", lambda url, timeout=0: FakeResponse())
    monkeypatch.chdir(tmp_path)
    out = colab.run_job(
        "https://example.org/jobs/mini.json", out=tmp_path / "o", measure=False, progress=None
    )
    assert (tmp_path / colab.JOBS_DIRNAME / "mini.json").is_file()
    assert (out / "result.h5").is_file()


def test_an_explicit_out_wins_including_a_folder_the_user_mounted(gpu_present, tmp_path):
    """K12: persistence is the user's mount plus ``out=``; we only write there."""
    elsewhere = tmp_path / "somebody_elses_mount" / "runs"
    out = colab.run_job(mini_job(tmp_path), out=elsewhere, measure=False, progress=None)
    assert out == elsewhere and (elsewhere / "result.h5").is_file()


# --------------------------------------------------------------- the result view


def test_summary_quotes_the_run_folders_own_numbers(gpu_present, tmp_path):
    out = colab.run_job(mini_job(tmp_path), out=tmp_path / "o", measure=False, progress=None)
    text = colab.summary(out)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert str(metrics["peak"]["p_mpa"]) in text
    assert "solver linear" in text
    assert "result.h5" in text and "preview.npz" in text


def test_summary_of_an_empty_folder_says_something_useful(tmp_path):
    assert str(tmp_path) in colab.summary(tmp_path)


def test_show_prints_the_numbers_even_when_it_cannot_draw(tmp_path, capsys):
    assert colab.show(tmp_path, figures=True) is None
    printed = capsys.readouterr().out
    assert str(tmp_path) in printed
    assert "no figures" in printed


def test_show_can_skip_figures_entirely(gpu_present, tmp_path, capsys):
    out = colab.run_job(mini_job(tmp_path), out=tmp_path / "o", measure=False, progress=None)
    assert colab.show(out, figures=False) is None
    assert "peak" in capsys.readouterr().out
