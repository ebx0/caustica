"""M10h gates: the wheel is complete and the packaged example is safe.

Wheel content is pinned by test because a missing package-data entry is
invisible from inside a checkout (the CWD masks it — exactly how the
``gpu_db.json`` bug survived until a real ``pip install``). The example
tests encode the other M10h rule: packaged jobs are *copied out* before
running, so nothing ever tries to write into the install directory.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import build_stamp
import pytest

import caustica
from caustica import examples

REPO = Path(__file__).resolve().parents[1]


def _git_commit_all(root: Path) -> list[str]:
    """``git commit`` that works on a throwaway repo with no user config."""
    return [
        "git",
        "-c",
        "user.email=t@example.invalid",
        "-c",
        "user.name=t",
        "-C",
        str(root),
        "commit",
        "-q",
        "--allow-empty",
        "-am",
        "stamp test",
    ]


def _dir_snapshot(d: Path) -> set[tuple[str, int, int]]:
    """(path, size, mtime_ns) per file — names alone would miss an in-place
    overwrite. ``__pycache__`` is excluded: interpreter byte-code cache is
    not a library write (and is skipped silently on read-only installs)."""
    return {
        (p.relative_to(d).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
        for p in d.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }


def _install_root() -> Path:
    """The installed ``caustica`` package directory (site-packages in a wheel
    install, ``src/caustica`` under the dev editable install)."""
    import caustica

    return Path(caustica.__file__).resolve().parent


# -------------------------------------------------------------------- wheel


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> zipfile.ZipFile:
    out = tmp_path_factory.mktemp("wheel")
    # Build from a PRISTINE copy of the source tree. Building in the checkout
    # reuses a stale ``build/lib`` — files copied there by an earlier build
    # ship again even after their package-data entry (or the file itself) is
    # gone, so regressions are invisible (found 2026-08-21 by mutation-testing
    # this fixture: deleting package-data lines left the wheel green).
    srctree = tmp_path_factory.mktemp("srctree")
    for f in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copyfile(REPO / f, srctree / f)
    shutil.copytree(
        REPO / "src",
        srctree / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "*.pyc"),
    )
    # --no-build-isolation: use the dev env's setuptools instead of
    # downloading one, so this test passes offline.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(out),
            str(srctree),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:  # surface the build backend's actual error
        raise RuntimeError(f"wheel build failed:\n{proc.stdout}\n{proc.stderr}")
    (wheel_path,) = out.glob("caustica-*.whl")
    return zipfile.ZipFile(wheel_path)


def test_wheel_contains_the_package_data(wheel):
    names = set(wheel.namelist())
    assert "caustica/py.typed" in names
    assert "caustica/planner/gpu_db.json" in names  # the regression that started this
    assert "caustica/examples/__init__.py" in names
    assert "caustica/examples/water_bowl_mini.json" in names


def test_wheel_declares_the_console_script(wheel):
    (ep_name,) = (n for n in wheel.namelist() if n.endswith(".dist-info/entry_points.txt"))
    text = wheel.read(ep_name).decode()
    assert "[console_scripts]" in text
    assert "caustica = caustica.__main__:main" in text


def test_wheel_ships_no_repo_side_packages(wheel):
    top_level = {n.split("/", 1)[0] for n in wheel.namelist()}
    assert not any(t.startswith(("uwcem_phantoms", "apps", "tests")) for t in top_level)


def test_wheel_ships_no_file_absent_from_src(wheel):
    """Every packaged file must exist in ``src/`` — a stale build artifact
    must not resurrect deleted modules (matters for the M10k removals).

    ``_build_info.py`` is the one exemption: the build GENERATES it (fix A1),
    it is git-ignored, and a checkout that has never been built does not have
    it. Its own presence in the wheel is asserted separately, below.
    """
    generated = {"caustica/_build_info.py"}
    ghosts = [
        n
        for n in wheel.namelist()
        if n.startswith("caustica/") and n not in generated and not (REPO / "src" / n).is_file()
    ]
    assert ghosts == []


# ------------------------------------------------------- build provenance (A1)


def test_wheel_carries_a_build_info_module(wheel):
    """The wheel ships the generated stamp, whatever the build tree knew.

    This fixture builds from a copy with no ``.git``, so the commit here is
    honestly ``"unknown"`` — what is pinned is that the MODULE exists and has
    the three names ``caustica.env.build_info`` reads. The commit-carrying
    path is proven end to end in ``test_wheel_built_from_a_checkout_...``.
    """
    text = wheel.read("caustica/_build_info.py").decode()
    ns: dict = {}
    exec(compile(text, "_build_info.py", "exec"), ns)  # noqa: S102 - our own generated file
    assert set(("VERSION", "COMMIT", "BUILT_AT")) <= set(ns)
    assert ns["VERSION"] == caustica.__version__


def test_wheel_ships_the_validation_suite(wheel):
    """Colab installs a WHEEL and then runs ``python -m caustica.validation
    gpu-gates``. A subpackage missing from the distribution is invisible from
    inside a checkout — the exact shape of the gpu_db.json bug."""
    names = set(wheel.namelist())
    for mod in ("__init__", "__main__", "gpu_gates"):
        assert f"caustica/validation/{mod}.py" in names


def test_build_stamp_keeps_an_existing_stamp_when_git_is_absent(tmp_path):
    """The sdist rule: no git + a stamp already there -> do not clobber it.

    A wheel built FROM an sdist has no repository, and overwriting the
    sdist's stamp with ``"unknown"`` would discard the only provenance that
    distribution will ever have.
    """
    root = tmp_path / "tree"  # a plain directory: git rev-parse fails here
    pkg = root / "src" / "caustica"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('__version__ = "9.9.9"\n', encoding="utf-8")

    assert build_stamp.head_commit(root) is None
    first = build_stamp.stamp(root)
    assert first is not None and 'COMMIT = "unknown"' in first.read_text(encoding="utf-8")

    first.write_text(build_stamp.render("9.9.9", "d" * 40, "then"), encoding="utf-8")
    assert build_stamp.stamp(root) is None  # kept, not rewritten
    assert "d" * 40 in first.read_text(encoding="utf-8")


def test_build_stamp_refuses_a_commit_from_a_surrounding_repository(tmp_path):
    """``git rev-parse`` walks UP. A source tree unpacked inside somebody
    else's repository must not be stamped with that repository's commit."""
    outer = tmp_path / "outer"
    (outer / "inner").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(outer)], check=True, capture_output=True)
    (outer / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(_git_commit_all(outer), check=True, capture_output=True, cwd=outer)

    assert build_stamp.head_commit(outer) is not None  # the repo itself: fine
    assert build_stamp.head_commit(outer / "inner") is None  # a subdirectory: refused


def test_env_and_build_stamp_agree_on_this_checkouts_head():
    """The two copies of the "is this really our repository" rule (build time
    cannot import caustica — numpy is not installed yet) must not drift."""
    from caustica import env

    assert env._checkout_head() == build_stamp.head_commit(REPO)


@pytest.mark.slow
def test_wheel_built_from_a_checkout_stamps_run_meta_outside_the_repo(tmp_path):
    """Fix A1, end to end: Colab installs a WHEEL, and a wheel has no git.

    Before this, every ``run_meta.json`` written on Colab carried
    ``"git_commit": "unknown"`` — the traceability hole the first real GPU
    session exposed. Here a wheel is built from a real (temporary) checkout,
    installed into an empty directory, and a job is run from a working
    directory that is not inside any repository. The stamp must carry the
    commit the wheel was built from.
    """
    src = tmp_path / "checkout"
    src.mkdir()
    for f in ("pyproject.toml", "README.md", "LICENSE", "setup.py", "build_stamp.py"):
        shutil.copyfile(REPO / f, src / f)
    shutil.copytree(
        REPO / "src",
        src / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "*.pyc", "_build_info.py"),
    )
    subprocess.run(["git", "init", "-q", str(src)], check=True, capture_output=True)
    subprocess.run(_git_commit_all(src), check=True, capture_output=True, cwd=src)
    head = subprocess.run(
        ["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert len(head) == 40

    wheels = tmp_path / "wheels"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(wheels),
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"wheel build failed:\n{proc.stdout}\n{proc.stderr}")
    (whl,) = wheels.glob("caustica-*.whl")
    with zipfile.ZipFile(whl) as z:
        assert f'COMMIT = "{head}"' in z.read("caustica/_build_info.py").decode()

    # A CLEAN install target: only caustica lands here, dependencies come from
    # the test interpreter. PYTHONPATH precedes site-packages, so this install
    # — not the editable checkout — is the one that gets imported.
    site = tmp_path / "site"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(site), str(whl)],
        check=True,
        capture_output=True,
    )
    cwd = tmp_path / "elsewhere"  # not inside any git work tree
    cwd.mkdir()
    env = {**os.environ, "PYTHONPATH": str(site)}

    where = subprocess.run(
        [sys.executable, "-c", "import caustica; print(caustica.__file__)"],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        check=True,
    ).stdout.strip()
    assert Path(where).is_relative_to(site), f"the editable checkout leaked in: {where}"

    job = examples.copy("water_bowl_mini", cwd)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "caustica",
            "run",
            str(job),
            "--no-measure",
            "--out",
            str(cwd / "out"),
            "--no-progress",
        ],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    meta = json.loads((cwd / "out" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["git_commit"] == head != "unknown"


# ----------------------------------------------------------------- examples


def test_examples_api_lists_and_resolves():
    assert "water_bowl_mini" in examples.available()
    p = examples.path("water_bowl_mini")
    assert p.is_file()
    with pytest.raises(KeyError, match="water_bowl_mini"):  # names the available ones
        examples.path("no_such_example")


def test_example_copy_refuses_overwrite(tmp_path):
    first = examples.copy("water_bowl_mini", tmp_path)
    assert first.is_file()
    with pytest.raises(FileExistsError):
        examples.copy("water_bowl_mini", tmp_path)


def test_packaged_example_validates_clean():
    """The shipped JSON passes ``caustica validate`` as-is (W1/D13) with ZERO
    warnings — the quickstart's first-contact output must not open with an
    ignored warning, and the example sits at ppw 3.0 where the low-ppw
    warning (< 3.0) is one dx edit away."""
    from caustica.config.job import validate_job

    root = _install_root()
    before = _dir_snapshot(root)
    report = validate_job(examples.path("water_bowl_mini"))
    assert report.ok, report.render()
    assert report.warnings == [], report.render()
    assert _dir_snapshot(root) == before  # validate never writes


def test_copied_example_runs_without_touching_the_install_dir(tmp_path):
    """The copy runs end to end; outputs land next to the COPY, and the whole
    installed package tree shows no surviving write — (path, size, mtime)
    snapshot over the package root, before vs after. (A create-then-delete
    transient would evade any snapshot; the dominant T4 failure — a runs/
    tree created under the install dir — cannot.)"""
    from caustica.runner import EXIT_OK, RunnerOptions, run_job_file

    root = _install_root()
    before = _dir_snapshot(root)

    job_copy = examples.copy("water_bowl_mini", tmp_path)
    t0 = time.perf_counter()
    code = run_job_file(job_copy, RunnerOptions(measure=False, status_interval_s=0.0))
    elapsed = time.perf_counter() - t0

    assert code == EXIT_OK
    out = tmp_path / "runs" / "water_bowl_mini"  # T4: relative to the job file
    assert (out / "result.h5").is_file()
    assert _dir_snapshot(root) == before
    assert elapsed < 30.0  # the packaged example stays a QUICKstart


# ---------------------------------------------------------------------- CLI


def test_example_cli_lists_copies_and_errors(tmp_path, capsys):
    from caustica.__main__ import main

    assert main(["example"]) == 0
    assert "water_bowl_mini" in capsys.readouterr().out

    assert main(["example", "water_bowl_mini", "--to", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert str(tmp_path / "water_bowl_mini.json") in out
    assert (tmp_path / "water_bowl_mini.json").is_file()

    assert main(["example", "water_bowl_mini", "--to", str(tmp_path)]) == 2  # overwrite
    assert main(["example", "no_such_example"]) == 2
