"""M10m gate: ``caustica schema`` is real JSON Schema, and the docs cannot rot.

Two documents describe the job format — the generated schema and
``docs/job_reference.md``. Only one of them is written by hand, so the risk is
that the hand-written one silently falls behind a kind that was added or
renamed. These tests make that impossible: the kind headings in the reference
must be exactly the registered kinds, and each kind's first JSON snippet must
validate against that kind's model.
"""

import json
import re
from pathlib import Path

import pytest

from caustica.config.job import JOB_FORMAT, job_schema, validate_job
from caustica.config.kinds import array_kinds, medium_kinds

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "job_reference.md"
CONVENTIONS = REPO / "docs" / "conventions.md"


def core_kinds(registry) -> tuple[str, ...]:
    """The kinds caustica itself ships, ignoring any installed plugin.

    A third-party kind in the same environment is the whole point of the
    registry — it must not turn caustica's own suite red (found by the M10m
    skeptical review: installing a plugin failed six tests that asserted the
    registry's exact contents).
    """
    return tuple(
        n for n in registry.available() if registry.get(n).__module__.startswith("caustica.")
    )


def doc_sections() -> dict[str, list[str]]:
    """`## heading` -> the `### \\`kind\\`` names beneath it."""
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in DOC.read_text(encoding="utf-8").splitlines():
        h2 = re.fullmatch(r"## (.+)", line)
        if h2:
            current = sections.setdefault(h2.group(1).strip(), [])
            continue
        h3 = re.fullmatch(r"### `([a-z_]+)`", line)
        if h3 and current is not None:
            current.append(h3.group(1))
    return sections


def doc_snippets_all() -> dict[str, list[dict]]:
    """`### \\`kind\\`` -> every ```json block under that heading, in order."""
    text = DOC.read_text(encoding="utf-8")
    out: dict[str, list[dict]] = {}
    for match in re.finditer(r"### `([a-z_]+)`\n(.*?)(?=\n## |\n### |\Z)", text, re.S):
        kind, body = match.group(1), match.group(2)
        blocks = [json.loads(b) for b in re.findall(r"```json\n(.*?)\n```", body, re.S)]
        if blocks:
            out[kind] = blocks
    return out


def doc_snippets() -> dict[str, dict]:
    """`### \\`kind\\`` -> the first ```json block under that heading."""
    return {k: v[0] for k, v in doc_snippets_all().items()}


def minimal_job() -> dict:
    """The complete job the reference opens with."""
    block = re.search(r"```json\n(\{\n  \"format\".*?)\n```", DOC.read_text(encoding="utf-8"), re.S)
    assert block, "the reference no longer opens with a complete job snippet"
    return json.loads(block.group(1))


# ------------------------------------------------------------------- schema


def test_schema_is_valid_json_schema():
    schema = job_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == JOB_FORMAT
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False  # extra="forbid", in the schema too
    assert set(schema["required"]) >= {"name", "medium", "source", "drive"}

    # every $ref resolves inside $defs (a dangling ref is an unusable schema)
    text = json.dumps(schema)
    refs = set(re.findall(r'"\$ref": "#/\$defs/([^"]+)"', text))
    assert refs, "schema has no $defs references at all"
    missing = sorted(refs - set(schema["$defs"]))
    assert missing == [], f"dangling $ref: {missing}"

    # and it round-trips as JSON text (the CLI prints it)
    assert json.loads(json.dumps(schema)) == schema


def test_schema_discriminators_match_the_registries():
    schema = job_schema()
    medium = schema["properties"]["medium"]["discriminator"]
    array = schema["$defs"]["ArraySourceConfig"]["properties"]["array"]["discriminator"]
    assert medium["propertyName"] == "kind"
    assert array["propertyName"] == "kind"
    assert tuple(sorted(medium["mapping"])) == medium_kinds.available()
    assert tuple(sorted(array["mapping"])) == array_kinds.available()


def test_cli_schema_prints_parseable_json(capsys):
    from caustica.__main__ import main

    assert main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out) == job_schema()

    assert main(["schema", "--compact"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 1  # one line
    assert json.loads(out) == job_schema()

    assert main(["schema", "--kinds"]) == 0
    lines = capsys.readouterr().out.split()
    assert "homogeneous" in lines
    assert "elements" in lines


# ---------------------------------------------------------------------- docs


def test_reference_documents_exactly_the_registered_kinds():
    """The doc-rot gate: headings vs the registry, both directions."""
    sections = doc_sections()
    # caustica's reference documents caustica's kinds; a third-party kind
    # documents itself in its own package.
    assert tuple(sorted(sections["Medium kinds"])) == core_kinds(medium_kinds)
    assert tuple(sorted(sections["Array kinds"])) == core_kinds(array_kinds)


@pytest.mark.parametrize("kind", core_kinds(medium_kinds))
def test_each_medium_snippet_validates(kind):
    snippet = doc_snippets().get(kind)
    assert snippet is not None, f"docs/job_reference.md has no JSON snippet for '{kind}'"
    assert snippet["kind"] == kind
    medium_kinds.get(kind).model_validate(snippet)  # extra="forbid" catches stale keys


@pytest.mark.parametrize("kind", core_kinds(array_kinds))
def test_each_array_snippet_validates(kind):
    snippet = doc_snippets().get(kind)
    assert snippet is not None, f"docs/job_reference.md has no JSON snippet for '{kind}'"
    assert snippet["kind"] == kind
    array_kinds.get(kind).model_validate(snippet)


def test_the_documented_minimal_job_actually_runs_validate(tmp_path):
    """The first snippet a stranger copies must pass validate as printed."""
    job = minimal_job()
    assert job["format"] == JOB_FORMAT
    p = tmp_path / "doc_job.json"
    p.write_text(json.dumps(job), encoding="utf-8")
    report = validate_job(p)
    assert report.ok, report.render()
    assert report.warnings == [], report.render()


@pytest.mark.parametrize("kind", core_kinds(array_kinds))
def test_every_array_kind_has_a_snippet_that_fits_the_documented_grid(kind, tmp_path):
    """At least one snippet per array kind must RUN, not merely parse.

    Model validation alone let the spiral section ship the 100 mm production
    array as its only example — schema-valid, and a focus outside the grid of
    the job this page opens with (found by the M10m skeptical review). A kind
    may show a full-size recipe, but it owes the reader one that works here.
    """
    snippets = doc_snippets_all().get(kind, [])
    assert snippets, f"docs/job_reference.md has no JSON snippet for '{kind}'"
    failures = []
    for i, snippet in enumerate(snippets):
        if snippet.get("kind") != kind:
            continue
        job = minimal_job()
        job["source"]["array"] = snippet
        p = tmp_path / f"{kind}_{i}.json"
        p.write_text(json.dumps(job), encoding="utf-8")
        report = validate_job(p)
        if report.ok:
            return
        failures.append(f"snippet #{i}: {report.errors}")
    raise AssertionError(
        f"no '{kind}' snippet in docs/job_reference.md validates inside the "
        f"page's own minimal job:\n" + "\n".join(failures)
    )


def test_conventions_covers_the_five_silent_wrongness_traps():
    """conventions.md exists to stop misread results; keep its five topics."""
    text = CONVENTIONS.read_text(encoding="utf-8")
    for needle in (
        "e^(-i ω t)",  # phasor convention
        "Np/m",  # absorption units
        "db_cm_to_np_m",  # ...and the conversion helper that ships
        "+z",  # coordinate frame
        "2·c·dt/dx",  # what amplitude means
        "part of the grid",  # PML is inside size_mm
    ):
        assert needle in text, f"conventions.md no longer explains {needle!r}"
