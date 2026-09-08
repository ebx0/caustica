"""The two local GPU runners must stay twins.

``scripts/run_gpu_local.ps1`` and ``scripts/run_gpu_local.sh`` are the same
procedure written twice, and the failure mode of that arrangement is silent
drift: someone changes the marker expression, the quick preset or the report
root in one of them, and two machines then produce folders that cannot be
compared. Nothing about these checks needs a GPU, so they run everywhere the
suite runs, which is exactly where the drift would otherwise go unnoticed.

Every check here extracts the value it grades out of the script rather than
searching the file for a phrase. A search over the whole text is satisfied by a
comment or a banner, so it passes on a script whose real invocation has already
drifted; that is how the first version of this file came to be green on a
mutant that reintroduced the pipeline trap it names.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SH = ROOT / "scripts" / "run_gpu_local.sh"
PS1 = ROOT / "scripts" / "run_gpu_local.ps1"
WORKFLOW = ROOT / ".github" / "workflows" / "gpu-selfhosted.yml"

#: The marker expression both runners select the gpu leg with. It is spelled
#: out here so a change has to be made in three places on purpose.
MARKER = "gpu and not kwave and not network"
#: The ladder targets `--quick` / `-Quick` passes to the gate suite, in GiB.
QUICK_TARGETS = "1.0,2.0"
#: Where a run lands when neither runner is given an explicit output root.
DEFAULT_OUT = "benchmarks/reports/gpu_gates"

#: label -> (regex over the sh source, regex over the ps1 source, agreed value).
#: Each regex captures the value itself, so a changed invocation cannot hide
#: behind an unchanged comment.
SHARED = {
    "the marker expression the gpu leg selects": (
        r'(?m)^marker="([^"]*)"',
        r"(?m)^\$marker = '([^']*)'",
        MARKER,
    ),
    "the --quick ladder targets": (
        r'(?m)^quick_targets="([^"]*)"',
        r"(?m)^\$quickTargets = '([^']*)'",
        QUICK_TARGETS,
    ),
    "the default report root": (
        r'(?m)^default_out="([^"]*)"',
        r'(?m)^\s*\[string\]\$Out = "([^"]*)"',
        DEFAULT_OUT,
    ),
}

#: The lines that must appear verbatim, because they are where each script
#: spends the shared value above instead of a literal of its own.
USES = {
    SH: (
        '-m pytest -m "$marker"',
        '--targets "$quick_targets"',
        '"${out:-$default_out}"',
    ),
    PS1: (
        "'pytest', '-m', $marker",
        "'--targets', $quickTargets",
        "'--out', $Out",
    ),
}

#: A pipeline reports the LAST command's status, so any of these downstream of
#: the interpreter turns a collection error into a pass.
FILTERS = re.compile(r"\|\s*(tail|head|select-object|select\b|sort|grep|findstr)", re.I)
#: What the output is allowed to flow into instead.
KEEPERS = re.compile(r"\|\s*(tee|Tee-Object|ForEach-Object)", re.I)


@pytest.fixture(scope="module")
def texts() -> dict[Path, str]:
    return {p: p.read_text(encoding="utf-8") for p in (SH, PS1)}


def _capture(pattern: str, text: str, what: str, path: Path) -> str:
    m = re.search(pattern, text)
    assert m is not None, f"{path.name} no longer defines {what} ({pattern})"
    return m.group(1)


def _interpreter_lines(text: str) -> list[tuple[int, str]]:
    """The lines that run the interpreter with its stderr merged into stdout.

    That merge is the signature of a captured leg: the runs whose output
    becomes a log and whose exit code becomes the script's own. Comments are
    skipped, since both files explain the merge in prose beside it.
    """
    return [
        (i, ln)
        for i, ln in enumerate(text.splitlines(), 1)
        if "2>&1" in ln and not ln.lstrip().startswith("#")
    ]


def test_both_runners_exist() -> None:
    for path in (SH, PS1):
        assert path.is_file(), f"{path} is missing"
        assert path.stat().st_size > 0


@pytest.mark.parametrize("label", sorted(SHARED))
def test_the_twins_agree(label: str, texts: dict[Path, str]) -> None:
    sh_pattern, ps_pattern, expected = SHARED[label]
    sh_value = _capture(sh_pattern, texts[SH], label, SH)
    ps_value = _capture(ps_pattern, texts[PS1], label, PS1)
    assert sh_value == ps_value, f"{label} drifted: sh {sh_value!r} vs ps1 {ps_value!r}"
    assert sh_value == expected, f"{label} is {sh_value!r}, this test expects {expected!r}"


def test_each_runner_spends_the_shared_value(texts: dict[Path, str]) -> None:
    """Agreeing on a definition is worth nothing if the invocation ignores it."""
    for path, needles in USES.items():
        for needle in needles:
            assert needle in texts[path], f"{path.name} no longer uses {needle!r}"


def test_neither_runner_pipes_the_interpreter_into_a_filter(texts: dict[Path, str]) -> None:
    """A pipeline reports the LAST command's status, so `pytest | tail` passes."""
    for path, text in texts.items():
        lines = _interpreter_lines(text)
        assert lines, f"{path.name} runs nothing with its stderr merged in"
        for number, line in lines:
            where = f"{path.name}:{number}"
            assert not FILTERS.search(line), f"{where} filters the output: {line.strip()}"
            assert KEEPERS.search(line), f"{where} does not tee the output: {line.strip()}"
            # A redirect would swallow the console output the same way, and on
            # the shell side it also drops PIPESTATUS.
            assert ">" not in line.replace("2>&1", ""), (
                f"{path.name}:{number} redirects instead of teeing: {line.strip()}"
            )


def test_each_runner_keeps_the_interpreters_own_exit_code(texts: dict[Path, str]) -> None:
    """The code that matters is the interpreter's, never the pipeline's.

    The window is generous because the PowerShell side reads the code after a
    multi-line ForEach-Object block, but it still has to be read before the
    script does anything else that would overwrite it.
    """
    wanted = {SH: "PIPESTATUS", PS1: "LASTEXITCODE"}
    for path, text in texts.items():
        lines = text.splitlines()
        for number, _ in _interpreter_lines(text):
            window = " ".join(lines[number : number + 15])
            assert wanted[path] in window, (
                f"{path.name}:{number} does not take the exit code from {wanted[path]}"
            )


def test_the_powershell_twin_writes_utf8_logs(texts: dict[Path, str]) -> None:
    """Tee-Object -FilePath writes UTF-16 on Windows PowerShell 5.1.

    The shell twin's logs are UTF-8 with LF, and the whole point of the pair is
    that two machines leave comparable folders, so the PowerShell side must not
    let a cmdlet choose the encoding for it.
    """
    text = texts[PS1]
    assert "Tee-Object -FilePath" not in text, "Tee-Object would own the file, in UTF-16"
    assert "UTF8Encoding($false)" in text, "logs and SUMMARY.md must be UTF-8 without a BOM"
    assert "WriteAllText" in text, "WriteAllLines would end every line with CRLF"


def test_the_runners_are_ascii(texts: dict[Path, str]) -> None:
    """English only, and no em dashes: the tree's rule, checked where ruff cannot."""
    for path, text in texts.items():
        offenders = [c for c in text if ord(c) > 126]
        assert not offenders, f"{path.name} carries non-ascii: {sorted(set(offenders))}"


def _block(text: str, key: str, indent: int) -> list[str]:
    """The lines nested under `key:` at `indent` columns, comments included.

    A YAML subset reader, enough for a workflow file and with no dependency:
    PyYAML is not a caustica dependency, and a check that skips when an import
    is missing is a check that does not run on the machines that matter.
    """
    opener = " " * indent + key + ":"
    out: list[str] = []
    inside = False
    for line in text.splitlines():
        if not inside:
            inside = line.rstrip() == opener or line.startswith(opener + " ")
            continue
        if not line.strip():
            out.append(line)
            continue
        if len(line) - len(line.lstrip(" ")) <= indent:
            break
        out.append(line)
    return out


def _scalars(lines: list[str], indent: int) -> dict[str, str]:
    """`key: value` pairs at exactly `indent`, values unquoted and untyped."""
    found: dict[str, str] = {}
    for line in lines:
        if len(line) - len(line.lstrip(" ")) != indent:
            continue
        m = re.match(r"^\s*([A-Za-z_][\w-]*):\s*(.*?)\s*$", line)
        if m:
            found[m.group(1)] = m.group(2)
    return found


def test_the_selfhosted_workflow_cannot_run_by_accident() -> None:
    """No runner has this label yet; the guard is what keeps the leg dormant."""
    text = WORKFLOW.read_text(encoding="utf-8")

    triggers = _scalars(_block(text, "on", 0), 2)
    assert set(triggers) == {"workflow_dispatch"}, (
        f"the dormant workflow gained a trigger: {sorted(triggers)}"
    )

    job = _block("\n".join(_block(text, "jobs", 0)), "gpu-gates", 2)
    assert job, "the gpu-gates job is gone or renamed"
    keys = _scalars(job, 4)
    assert keys.get("if") == "false", f"the `if: false` guard is gone: if = {keys.get('if')!r}"

    body = "\n".join(job)
    assert "bash scripts/run_gpu_local.sh" in body, "CI must run the same script a developer runs"
    # Without the gpu extra the fresh venv has no CuPy, every gpu-marked test
    # skips and the gate suite exits 2, so waking the job would prove nothing.
    assert '".[dev,gpu]"' in body, "the install must pull the gpu extra"


def test_the_workflow_parses_as_yaml_when_pyyaml_is_available() -> None:
    """A second opinion on the reader above, on any machine that has PyYAML."""
    yaml = pytest.importorskip("yaml", reason="PyYAML is not a caustica dependency")
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML reads the bare `on` key as the boolean True.
    triggers = doc.get("on", doc.get(True))
    assert set(triggers) == {"workflow_dispatch"}
    assert doc["jobs"]["gpu-gates"]["if"] is False
