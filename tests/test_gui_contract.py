"""M10l gate: ``docs/gui_contract.md`` cannot rot.

The contract page exists so a GUI — written later, elsewhere, in a technology
nobody has picked — can be built against a surface that is frozen *before* it
starts. A page like that is worth exactly as much as its accuracy, and every
field list on it is one refactor away from being a lie.

So nothing on that page is trusted here. Each list is compared against the
thing it describes: the exit codes against the runner's constants, the
``status.json`` fields against the keys a real run writes, the progress payload
against a real payload, the output-folder listing against a real folder. The
one concession is the cupy-only ``env_report`` block, which cannot be produced
on a CPU box — those names are checked against the source text of
``caustica/env.py`` instead, which still catches a rename.
"""

import json
import re
from pathlib import Path

import pytest
from tests.test_runner import mini_job, opts

import caustica.runner as runner_mod
from caustica.config.job import JOB_FORMAT
from caustica.env import env_report
from caustica.io.checkpoint import CHECKPOINT_FORMAT
from caustica.io.store import RESULT_FORMAT
from caustica.report.preview import PREVIEW_FORMAT
from caustica.runner import (
    CANCEL_FILE,
    ERROR_FILE,
    ERROR_FORMAT,
    ERROR_KEYS,
    ERROR_STAGES,
    EXIT_CONFIG,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_OOM,
    EXIT_SOLVER,
    run_job_file,
)

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "gui_contract.md"


# ------------------------------------------------------------------ parsing


def doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


def section(heading: str) -> str:
    """The body under an exact heading, up to the next heading of any level."""
    match = re.search(rf"^#+ {re.escape(heading)}\n(.*?)(?=^#+ |\Z)", doc_text(), re.S | re.M)
    assert match, f"docs/gui_contract.md has no heading {heading!r}"
    return match.group(1)


def bullets(heading: str) -> list[str]:
    """Every ``- `name` ...`` bullet under a heading, in order."""
    found = re.findall(r"^- `([^`]+)`", section(heading), re.M)
    assert found, f"no `name` bullets under {heading!r}"
    return found


def table(heading: str) -> list[tuple[str, ...]]:
    """Rows of a markdown table under a heading, as tuples of cell text."""
    rows = []
    for line in section(heading).splitlines():
        if not line.startswith("|") or set(line) <= set("| -"):
            continue
        cells = tuple(c.strip() for c in line.strip("|").split("|"))
        if cells[0].lower() in ("code", "format"):  # header row
            continue
        rows.append(cells)
    assert rows, f"no table rows under {heading!r}"
    return rows


def unbacktick(cell: str) -> str:
    return cell.strip().strip("`")


# ---------------------------------------------------------------- the runs
# Four real runs produce every artifact the page describes. Module-scoped:
# it is the same seconds-scale mini job the runner suite uses.


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("gui_contract")
    job = mini_job(root)
    payloads: list[dict] = []

    ok = root / "ok"
    assert run_job_file(job, opts(out=ok, progress=payloads.append)) == EXIT_OK

    stopped = root / "stopped"
    assert run_job_file(job, opts(out=stopped, stop_after_periods=2)) == EXIT_INTERRUPTED

    refused = root / "refused"
    assert run_job_file(job, opts(out=refused, vram_limit_gib=1e-5)) == EXIT_OOM

    broken = root / "broken"
    with pytest.MonkeyPatch.context() as mp:

        def boom(*a, **kw):
            raise OSError("synthetic store failure")

        mp.setattr(runner_mod, "save_result", boom)
        assert run_job_file(job, opts(out=broken)) == EXIT_SOLVER

    def status(d: Path) -> dict:
        return json.loads((d / "status.json").read_text(encoding="utf-8"))

    return {
        "ok": ok,
        "stopped": stopped,
        "refused": refused,
        "broken": broken,
        "progress": payloads[0],
        "status_done": status(ok),
        "status_interrupted": status(stopped),
        "status_failed": status(broken),
    }


# ------------------------------------------------------------------- checks


def test_the_page_says_what_is_not_a_contract():
    assert "**Nothing that is not listed here is a contract.**" in doc_text()
    # input = one job file, output = one defined folder (PLAN.md §11)
    assert "**Input is ONE file.**" in doc_text()
    assert "**Output is ONE folder.**" in doc_text()


def test_documented_exit_codes_are_the_runners_exit_codes():
    documented = {unbacktick(name): int(unbacktick(code)) for code, name, _ in table("Exit codes")}
    real = {n: getattr(runner_mod, n) for n in dir(runner_mod) if n.startswith("EXIT_")}
    assert documented == real
    # and the codes really are disjoint (the queue's whole premise)
    assert len(set(real.values())) == len(real)


def test_documented_output_folder_matches_a_real_one(runs):
    always = set(bullets("Output folder: written by every successful run"))
    assert always == {f.name for f in runs["ok"].iterdir()}

    conditional = set(bullets("Output folder: written only under some conditions"))
    seen = {f.name for f in runs["stopped"].iterdir()} | {f.name for f in runs["refused"].iterdir()}
    assert conditional == (seen - always) | {CANCEL_FILE}
    assert conditional == {"checkpoint.npz", ERROR_FILE, CANCEL_FILE}


def test_documented_status_fields_match_a_real_status_json(runs):
    done = set(runs["status_done"])
    interrupted = set(runs["status_interrupted"])
    failed = set(runs["status_failed"])
    base = done & interrupted & failed
    assert set(bullets("status.json: fields in every heartbeat")) == base
    extras = (done | interrupted | failed) - base
    assert set(bullets("status.json: state-dependent extras")) == extras
    # the documented states are the ones the runner actually writes
    states = {runs[k]["state"] for k in ("status_done", "status_interrupted", "status_failed")}
    listed = set(re.findall(r"`(\w+)`", section("`status.json` — the live heartbeat")))
    assert states <= listed


def test_documented_error_json_matches_the_runners_contract():
    assert tuple(bullets("error.json: fields")) == ERROR_KEYS
    assert tuple(bullets("error.json: stages")) == ERROR_STAGES
    # the example block on the page is a real caustica-error/1 document
    block = re.search(r"```json\n(\{.*?\})\n```", doc_text(), re.S)
    assert block, "the error.json section no longer shows an example"
    example = json.loads(block.group(1))
    assert tuple(example) == ERROR_KEYS
    assert example["format"] == ERROR_FORMAT
    assert example["stage"] in ERROR_STAGES
    assert example["exit_code"] == EXIT_OOM


def test_documented_plan_and_meta_and_metrics_fields_match_real_files(runs):
    def keys(name: str) -> set:
        return set(json.loads((runs["ok"] / name).read_text(encoding="utf-8")))

    assert set(bullets("plan.json: fields")) == keys("plan.json")
    assert set(bullets("run_meta.json: top-level fields")) == keys("run_meta.json")
    assert set(bullets("metrics.json: top-level fields")) == keys("metrics.json")


def test_documented_progress_payload_matches_a_real_payload(runs):
    assert bullets("Progress payload: keys") == list(runs["progress"])
    assert callable(runs["progress"]["snapshot"])  # the tenth key, as documented


def test_documented_env_report_keys_match_env_report():
    assert set(bullets("env_report(): keys on every machine")) == set(env_report("numpy"))
    # The GPU block cannot be produced on a CPU box; check the names against
    # the source that would produce them, which still catches a rename.
    src = (REPO / "src" / "caustica" / "env.py").read_text(encoding="utf-8")
    for key in bullets("env_report(): keys added on the cupy backend"):
        assert f'"{key}"' in src, f"env.py no longer produces {key!r}"
    assert '"gpu_probe_error"' in src


def test_documented_format_identifiers_are_the_real_ones():
    documented = {unbacktick(fmt) for fmt, _ in table("Format identifiers")}
    assert documented == {
        JOB_FORMAT,
        RESULT_FORMAT,
        PREVIEW_FORMAT,
        ERROR_FORMAT,
        CHECKPOINT_FORMAT,
        "caustica-run-meta/1",
        "caustica-metrics/1",
    }


def test_the_cancel_and_error_file_names_on_the_page_are_the_real_ones():
    body = section("`cancel` — stopping a run without killing it")
    assert f"`{CANCEL_FILE}`" in body
    assert f"exits **{EXIT_INTERRUPTED}**" in body
    assert f"## `{ERROR_FILE}` — why a run failed" in doc_text()


def test_the_cli_lines_on_the_page_actually_parse():
    """Every ``caustica ...`` line in the overview block is a real command."""
    from caustica.__main__ import build_parser

    parser = build_parser()
    block = re.search(r"```\ncaustica run.*?```", doc_text(), re.S).group(0)
    lines = [ln.split("#")[0].strip() for ln in block.splitlines() if ln.startswith("caustica ")]
    assert len(lines) >= 4
    for line in lines:
        args = [("job.json" if a.endswith(".json") else a) for a in line.split()[1:]]
        args = [("out" if a.startswith("<") else a) for a in args]
        parsed = parser.parse_args(args)  # argparse exits 2 on an unknown flag
        assert parsed.command in ("run", "validate", "schema", "report")


def test_config_error_advice_points_at_commands_the_page_documents(tmp_path):
    """The advice a failed job hands a GUI must name real entry points."""
    p = tmp_path / "broken.json"
    p.write_text("{not json")
    out = tmp_path / "out"
    assert run_job_file(p, opts(out=out)) == EXIT_CONFIG
    advice = " ".join(json.loads((out / ERROR_FILE).read_text(encoding="utf-8"))["advice"])
    assert "caustica validate" in advice and "caustica schema" in advice
    assert "caustica validate" in doc_text() and "caustica schema" in doc_text()


def test_the_page_points_at_the_import_direction_gate():
    assert "tests/test_import_direction.py" in doc_text()
    assert (REPO / "tests" / "test_import_direction.py").exists()


def test_freezing_the_contract_added_no_gui_dependency():
    """M10l's own rule: writing the contract down must not smuggle a GUI in."""
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "gui" not in pyproject
    src = REPO / "src" / "caustica"
    hits = [p.name for p in src.rglob("*.py") if "caustica_gui" in p.read_text(encoding="utf-8")]
    assert hits == []
