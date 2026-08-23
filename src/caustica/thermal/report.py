"""``THERMAL.md`` + ``thermal.json``: the dose/threshold summary of one solve.

Nothing in here integrates anything. The temperature history comes from
:class:`~caustica.thermal.pennes.PennesSolver` (which accumulates CEM43
during the solve, because dose is a history integral), the thresholds come
from :mod:`caustica.thermal.dose` (which holds exactly one copy of the
ITRUSST numbers), and this module only decides which of them go in which
table and what the word next to each row is. A report that recomputed a
dose would be a second definition of it.

What the report answers
-----------------------
Three questions, in the order a reader asks them:

1. **How hot did it get?** Peak temperature over the WHOLE history (the
   solver's ``temperature_max``, not its endpoint) and the rise above the
   baseline, because the ITRUSST non-thermal line is stated as ``dT <= 2 C``.
2. **How much dose landed, and where?** Peak CEM43 and the VOLUME above each
   ITRUSST limit — a peak alone cannot distinguish one over-cooked voxel
   from a cubic centimetre of it.
3. **Is that inside the limits?** Every threshold row carries ``PASS`` or
   ``EXCEEDED``. When the medium carries an ``id_map`` and the caller names
   the tissues, the per-tissue table grades each tissue against ITS OWN limit
   (brain 2, bone 16, skin 21 CEM43-min); without labels the report says so
   and falls back to reading the whole volume against the strictest limit,
   which is the conservative answer and is labelled as such.

Two deliberate honesty rules
----------------------------
* **A report with no dose map is refused.** :meth:`PennesSolver.solve` only
  accumulates CEM43 when asked (``dose=True``); rendering a "dose report"
  from a run that never integrated one would have to leave every threshold
  row blank, and a blank row in a safety table is read as a pass.
* **The tissue -> ITRUSST class mapping is PRINTED, not assumed.** The class
  is guessed from the tissue label by substring ("Cortical bone" -> bone),
  which is a naming heuristic and can be wrong; the table shows the class it
  used for every row, an unmatched label is graded ``NOT GRADED`` rather than
  against some default limit, and ``tissue_classes=`` overrides the guess.

And one rule that is not negotiable: :data:`MEDICAL_LIABILITY_NOTE` appears
verbatim in BOTH files. It is not a formatting option and there is no flag
that removes it.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from caustica.io.atomic import atomic_write
from caustica.thermal.dose import (
    ITRUSST_CEM43_LIMITS,
    ITRUSST_DELTA_T_LIMIT_C,
    ITRUSST_SOURCE,
    MEDICAL_DISCLAIMER,
)
from caustica.validation._verdict import fmt_num

#: Format tag of both files (bumped when the payload shape changes).
FORMAT = "caustica-thermal/1"

#: The two files every thermal report writes.
MD_NAME = "THERMAL.md"
JSON_NAME = "thermal.json"

#: Verdict vocabulary. Deliberately not the validation suites' PASS/FAIL:
#: a threshold that is crossed is not a broken test, it is an exposure over
#: a limit, and an ablation run is SUPPOSED to cross the 2 C line.
VERDICT_PASS = "PASS"
VERDICT_EXCEEDED = "EXCEEDED"
VERDICT_NOT_GRADED = "NOT GRADED"

#: The note that travels with every number this module prints, in both the
#: Markdown and the JSON. It opens with
#: :data:`caustica.thermal.dose.MEDICAL_DISCLAIMER` verbatim so the library
#: has exactly one definition of the research-use sentence, and adds what a
#: reader of a dose TABLE specifically has to be told.
MEDICAL_LIABILITY_NOTE = (
    f"{MEDICAL_DISCLAIMER} No number in this report is a medical claim, a "
    f"treatment plan, or a safety clearance for an exposure of any living "
    f"subject. Every value is the output of a numerical model of an idealised "
    f"medium and is bounded by the tissue properties, the acoustic field and "
    f"the discretisation it was given; the ITRUSST limits are quoted as "
    f"published thresholds, and quoting them is not a statement that this "
    f"simulation is an adequate basis for any decision about a person."
)


# ------------------------------------------------------------------ helpers


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp(backend_name: str | None = None) -> dict:
    """Who ran this, where, with which caustica — never raises.

    The SAME composition as :func:`caustica.study.core.stamp` and the
    runner's ``run_meta.json`` (:func:`caustica.env.env_report` plus
    :func:`caustica.env.git_commit`), built here rather than imported so a
    thermal report does not drag the runner into ``import caustica.thermal``.
    """
    from caustica import __version__  # noqa: PLC0415 (cycle-safe at call time)
    from caustica.env import env_report, git_commit  # noqa: PLC0415

    return {
        "generated": _now_iso(),
        "caustica": __version__,
        "git_commit": git_commit(),
        "host": platform.node(),
        "environment": env_report(backend_name),
    }


def _cell(text: Any) -> str:
    """One Markdown table cell: escape the pipes that would split the row."""
    return str(text).replace("|", "\\|")


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _kv(rows: Sequence[tuple[str, Any]]) -> list[str]:
    return _table(["", ""], [[label, value] for label, value in rows])


def timestamped_dir(root: str | Path = "benchmarks/reports/thermal") -> Path:
    """``<root>/<UTC timestamp>`` — where an evidence run leaves its report.

    Same shape as the validation suites' report folders, so a milestone's
    thermal evidence is found the same way its analytic evidence is.
    """
    return Path(root) / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def labels_from_db(db: Any, ids: Sequence[int] | None = None) -> dict[int, str]:
    """``{tissue id: Material.name}`` for a :class:`~caustica.materials.MaterialDB`.

    The convenience that keeps the report's tissue names and the medium's
    tissue names the same strings — passing a hand-written dict is the
    fastest way to label a dose table with the wrong tissue.
    """
    wanted = tuple(db.ids) if ids is None else tuple(int(i) for i in ids)
    return {int(i): str(db[int(i)].name) for i in wanted if int(i) in db}


def itrusst_class(label: str | None) -> str | None:
    """Which ITRUSST limit a tissue label falls under, or ``None``.

    A substring match against the three published classes
    (:data:`caustica.thermal.dose.ITRUSST_CEM43_LIMITS`), lowercased:
    ``"Cortical bone"`` -> ``"bone"``, ``"Brain grey matter"`` -> ``"brain"``.
    It is a NAMING heuristic and nothing more, which is why every report
    prints the class it landed on for each tissue and grades an unmatched
    label ``NOT GRADED`` instead of picking a limit for it.
    """
    if not label:
        return None
    text = str(label).lower()
    hits = [name for name in ITRUSST_CEM43_LIMITS if name in text]
    if len(hits) != 1:
        # Zero matches: an unlisted tissue (fat, muscle, gel). More than one:
        # a label like "skin over bone" that names two classes with different
        # limits — picking either would be an invention.
        return None
    return hits[0]


def _above(field: np.ndarray, limit: float, voxel_volume_mm3: float | None) -> dict:
    """How much of ``field`` sits strictly above ``limit`` (the ITRUSST ``<=``)."""
    mask = np.asarray(field) > limit
    n = int(mask.sum())
    return {
        "limit": float(limit),
        "n_voxels": n,
        "n_voxels_total": int(np.asarray(field).size),
        "volume_mm3": None if voxel_volume_mm3 is None else n * voxel_volume_mm3,
        "fraction": n / float(np.asarray(field).size),
        "verdict": VERDICT_EXCEEDED if n else VERDICT_PASS,
    }


def _worst(verdicts: Sequence[str]) -> str:
    """EXCEEDED beats NOT GRADED beats PASS — the safety-table ordering."""
    if VERDICT_EXCEEDED in verdicts:
        return VERDICT_EXCEEDED
    if not verdicts:
        return VERDICT_NOT_GRADED
    return VERDICT_PASS if all(v == VERDICT_PASS for v in verdicts) else VERDICT_NOT_GRADED


# ------------------------------------------------------------------ payload


def thermal_payload(
    result: Any,
    medium: Any,
    *,
    label: str = "thermal solve",
    baseline_temperature_c: float | None = None,
    tissue_labels: Mapping[int, str] | None = None,
    tissue_classes: Mapping[int, str | None] | None = None,
    notes: Sequence[str] = (),
    stamp: Mapping[str, Any] | None = None,
) -> dict:
    """``thermal.json`` for one :class:`~caustica.thermal.pennes.ThermalResult`.

    Parameters
    ----------
    result:
        A finished solve. It MUST carry a dose map (``solve(..., dose=True)``);
        see the module docstring for why a doseless dose report is refused.
    medium:
        The :class:`~caustica.thermal.properties.ThermalMedium` it ran on —
        the source of ``dx`` (so voxel counts become a volume) and of the
        ``id_map`` the per-tissue table needs.
    label:
        What this run was, in the report's title.
    baseline_temperature_c:
        The zero of ``dT``. Defaults to the solve's own arterial temperature,
        which is the body baseline the ITRUSST 2 C line is stated against;
        override it for a phantom that starts somewhere else.
    tissue_labels:
        ``{tissue id: name}``. Without it there is no per-tissue table (an
        id map alone names nothing); :func:`labels_from_db` builds one from
        the same ``MaterialDB`` the medium was built from.
    tissue_classes:
        ``{tissue id: "brain"|"bone"|"skin"|None}`` — overrides
        :func:`itrusst_class` where the label heuristic guesses wrong.
    notes:
        Caveats to print. The report adds its own.
    stamp:
        Provenance dict; built with :func:`_stamp` when not given.
    """
    if getattr(result, "dose_cem43", None) is None:
        raise ValueError(
            "this ThermalResult carries no CEM43 map, so there is no dose to report and "
            "every threshold row would be blank — which a reader of a safety table reads "
            "as a pass. Re-run the solve with dose=True (and pass the previous phase's "
            "dose_cem43 as dose0 when you are chaining a heat-up and a cool-down), then "
            "write the report."
        )
    t_max = np.asarray(result.temperature_max, dtype=np.float64)
    t_final = np.asarray(result.temperature, dtype=np.float64)
    dose = np.asarray(result.dose_cem43, dtype=np.float64)
    shape = tuple(medium.shape)
    for name, arr in (("temperature_max", t_max), ("dose_cem43", dose)):
        if arr.shape != shape:
            raise ValueError(
                f"result.{name} has shape {arr.shape} but the thermal medium is {shape}; "
                f"the report would put a dose number on the wrong voxel."
            )

    meta = dict(getattr(result, "meta", {}) or {})
    baseline = (
        float(meta.get("arterial_temperature_c", 37.0))
        if baseline_temperature_c is None
        else float(baseline_temperature_c)
    )
    delta = t_max - baseline

    ndim = len(shape)
    dx_mm = float(medium.dx) * 1e3
    # A voxel volume is only a volume in 3-D; in 1-D/2-D the same count is a
    # length or an area per unit of the missing axes, so it is reported as a
    # voxel count and the volume column says so. (Same rule as
    # HeatingSource.total_power_w.)
    voxel_volume_mm3 = dx_mm**3 if ndim == 3 else None

    thresholds = [
        {
            "name": f"CEM43 ({tissue})",
            "quantity": "cem43",
            "tissue_class": tissue,
            "unit": "equivalent minutes at 43 C",
            **_above(dose, limit, voxel_volume_mm3),
        }
        for tissue, limit in sorted(ITRUSST_CEM43_LIMITS.items(), key=lambda kv: kv[1])
    ]
    thresholds.append(
        {
            "name": "temperature rise (non-thermal line)",
            "quantity": "delta_t",
            "tissue_class": None,
            "unit": "C",
            **_above(delta, ITRUSST_DELTA_T_LIMIT_C, voxel_volume_mm3),
        }
    )

    id_map = getattr(medium, "id_map", None)
    tissues: list[dict] = []
    report_notes = list(notes)
    if id_map is None:
        report_notes.append(
            "the thermal medium carries no id_map, so there is no per-tissue table; the "
            "threshold rows above read the WHOLE volume against each limit."
        )
    elif not tissue_labels:
        report_notes.append(
            "the medium has an id_map but no tissue_labels were given, so no tissue can "
            "be matched to an ITRUSST class; pass labels_from_db(db) to get the "
            "per-tissue table."
        )
    else:
        id_map = np.asarray(id_map)
        classes = dict(tissue_classes or {})
        for tissue_id in (int(i) for i in np.unique(id_map)):
            mask = id_map == tissue_id
            name = tissue_labels.get(tissue_id)
            # An explicit override wins even when it says None ("this tissue
            # has no published limit"), which is why it is a membership test
            # and not a .get() with a default.
            cls = classes[tissue_id] if tissue_id in classes else itrusst_class(name)
            limit = None if cls is None else float(ITRUSST_CEM43_LIMITS[cls])
            n_over = None if limit is None else int((dose[mask] > limit).sum())
            n_vox = int(mask.sum())
            tissues.append(
                {
                    "id": tissue_id,
                    "label": name,
                    "itrusst_class": cls,
                    "limit_cem43": limit,
                    "n_voxels": n_vox,
                    "volume_mm3": None if voxel_volume_mm3 is None else n_vox * voxel_volume_mm3,
                    "peak_temperature_c": float(t_max[mask].max()),
                    "peak_delta_t_c": float(delta[mask].max()),
                    "peak_cem43": float(dose[mask].max()),
                    "n_voxels_over_limit": n_over,
                    "volume_over_limit_mm3": (
                        None
                        if (n_over is None or voxel_volume_mm3 is None)
                        else n_over * voxel_volume_mm3
                    ),
                    "verdict": (
                        VERDICT_NOT_GRADED
                        if limit is None
                        else (VERDICT_EXCEEDED if n_over else VERDICT_PASS)
                    ),
                }
            )
        ungraded = [t["label"] for t in tissues if t["itrusst_class"] is None]
        if ungraded:
            report_notes.append(
                f"no ITRUSST class matches {ungraded}; those tissues are NOT GRADED "
                f"(the consensus publishes limits for brain, bone and skin only). Pass "
                f"tissue_classes={{id: 'brain'|'bone'|'skin'}} where a label should map "
                f"to one of them."
            )

    graded = [t["verdict"] for t in tissues if t["verdict"] != VERDICT_NOT_GRADED]
    delta_row = thresholds[-1]
    if graded:
        basis = "per-tissue ITRUSST classes, each tissue against its own CEM43 limit"
        dose_verdict = _worst(graded)
    else:
        strictest = min(ITRUSST_CEM43_LIMITS.items(), key=lambda kv: kv[1])
        basis = (
            f"no graded tissue: the whole volume read against the STRICTEST ITRUSST "
            f"limit ({strictest[0]}, {strictest[1]:g} CEM43-min), which is the "
            f"conservative reading, not a claim about what this tissue is"
        )
        dose_verdict = thresholds[0]["verdict"]

    payload = {
        "format": FORMAT,
        "label": label,
        **dict(stamp if stamp is not None else _stamp(meta.get("backend"))),
        # Not optional, not behind a flag, and first among the content keys so
        # anything that truncates this file still carries it.
        "medical_liability_note": MEDICAL_LIABILITY_NOTE,
        "itrusst_source": ITRUSST_SOURCE,
        "verdict": _worst([dose_verdict, delta_row["verdict"]]),
        "verdict_dose": dose_verdict,
        "verdict_delta_t": delta_row["verdict"],
        "verdict_basis": basis,
        "run": {
            "scheme": meta.get("scheme"),
            "backend": meta.get("backend"),
            "boundary": meta.get("boundary"),
            "q": meta.get("q"),
            "perfusion_active": meta.get("perfusion_active"),
            "dt_s": float(result.dt),
            "n_steps": int(result.n_steps),
            "substeps": int(result.substeps),
            "t_end_s": float(result.t_end_s),
            "dt_stable_s": meta.get("dt_stable_s"),
            "phases": meta.get("chained_phases"),
            "shape": list(shape),
            "dx_m": float(medium.dx),
            "n_voxels": int(dose.size),
            "voxel_volume_mm3": voxel_volume_mm3,
        },
        "temperature": {
            "baseline_c": baseline,
            "peak_c": float(t_max.max()),
            "peak_delta_t_c": float(delta.max()),
            "final_peak_c": float(t_final.max()),
            "final_mean_c": float(t_final.mean()),
            "peak_is_over_history": True,
        },
        "dose": {
            "peak_cem43": float(dose.max()),
            "mean_cem43": float(dose.mean()),
            "n_voxels_with_dose": int((dose > 0.0).sum()),
        },
        "thresholds": thresholds,
        "tissues": tissues,
        "notes": report_notes,
    }
    return payload


# ---------------------------------------------------------------- rendering


def _volume_cell(row: Mapping[str, Any]) -> str:
    """The volume-above column, or the voxel count when a volume is not one."""
    vol = row.get("volume_mm3")
    if vol is None:
        return f"{row['n_voxels']} vox (not 3-D: no volume)"
    return f"{vol:.4g} mm³ ({row['n_voxels']} vox)"


def render_markdown(payload: Mapping[str, Any]) -> str:
    """The human half. Every number here is a key of ``thermal.json``."""
    env = payload.get("environment", {}) or {}
    run = payload["run"]
    temp = payload["temperature"]
    dose = payload["dose"]

    lines = [
        f"# caustica thermal report — {payload['label']}",
        "",
        f"CEM43 dose and ITRUSST threshold summary of one Pennes solve "
        f"(`{payload['format']}`), generated {payload.get('generated', '?')}.",
        "",
        f"> **{MEDICAL_LIABILITY_NOTE}**",
        "",
        "## Verdict",
        "",
        *_kv(
            [
                ("overall", f"**{payload['verdict']}**"),
                ("dose (CEM43)", payload["verdict_dose"]),
                (f"rise ≤ {ITRUSST_DELTA_T_LIMIT_C:g} °C", payload["verdict_delta_t"]),
                ("basis", payload["verdict_basis"]),
            ]
        ),
        "",
        "## Temperature",
        "",
        *_kv(
            [
                ("baseline [°C]", fmt_num(temp["baseline_c"])),
                ("peak over the whole history [°C]", fmt_num(temp["peak_c"])),
                ("peak rise ΔT [°C]", fmt_num(temp["peak_delta_t_c"])),
                ("peak at the END of the solve [°C]", fmt_num(temp["final_peak_c"])),
                ("mean at the END of the solve [°C]", fmt_num(temp["final_mean_c"])),
            ]
        ),
        "",
        "The peak is the per-voxel maximum over every step INCLUDING the internal "
        "sub-steps, not the endpoint: a sonication that has already cooled still "
        "did its damage on the way.",
        "",
        "## Dose",
        "",
        *_kv(
            [
                ("peak CEM43 [equivalent minutes]", fmt_num(dose["peak_cem43"])),
                ("mean CEM43 [equivalent minutes]", fmt_num(dose["mean_cem43"])),
                (
                    "voxels with any dose",
                    f"{dose['n_voxels_with_dose']} / {run['n_voxels']}",
                ),
            ]
        ),
        "",
        "## ITRUSST thresholds",
        "",
        *_table(
            ["threshold", "limit", "volume above", "fraction", "verdict"],
            [
                [
                    row["name"],
                    f"{row['limit']:g} {row['unit']}",
                    _volume_cell(row),
                    f"{100.0 * row['fraction']:.3g}%",
                    row["verdict"],
                ]
                for row in payload["thresholds"]
            ],
        ),
        "",
        f"*{ITRUSST_SOURCE}* Each CEM43 row above reads the WHOLE volume against that "
        f'limit — it is the as-if question ("if this were brain, would 2 CEM43 be '
        f'crossed anywhere?"). The per-tissue table below is the answer for the tissue '
        f"that is actually there. The rise row is the consensus' NON-THERMAL criterion: "
        f"a deliberate ablation is expected to cross it.",
        "",
    ]

    if payload["tissues"]:
        lines += [
            "## Per tissue",
            "",
            *_table(
                [
                    "id",
                    "tissue",
                    "ITRUSST class",
                    "size",
                    "peak T [°C]",
                    "peak ΔT [°C]",
                    "peak CEM43",
                    "limit",
                    "over limit",
                    "verdict",
                ],
                [
                    [
                        t["id"],
                        t["label"] or "—",
                        t["itrusst_class"] or "— (unlisted)",
                        _volume_cell(t),
                        fmt_num(t["peak_temperature_c"]),
                        fmt_num(t["peak_delta_t_c"]),
                        fmt_num(t["peak_cem43"]),
                        "--" if t["limit_cem43"] is None else f"{t['limit_cem43']:g}",
                        "--" if t["n_voxels_over_limit"] is None else f"{t['n_voxels_over_limit']}",
                        t["verdict"],
                    ]
                    for t in payload["tissues"]
                ],
            ),
            "",
            "The ITRUSST class column is a match on the tissue NAME, printed so it can "
            "be checked; override it with `tissue_classes={id: 'brain'|'bone'|'skin'}`.",
            "",
        ]

    phases = run.get("phases") or []
    lines += [
        "## Run",
        "",
        *_kv(
            [
                ("scheme", f"`{run['scheme']}`"),
                ("backend", str(run["backend"])),
                ("boundary", str(run["boundary"])),
                ("heat source Q", str(run["q"])),
                ("perfusion", "on" if run["perfusion_active"] else "off"),
                (
                    "time",
                    f"{fmt_num(run['t_end_s'])} s = {run['n_steps']} × "
                    f"{fmt_num(run['dt_s'])} s (× {run['substeps']} sub-step(s))",
                ),
                *(
                    [
                        (
                            "phases (source on / off)",
                            " → ".join(
                                f"{fmt_num(p.get('t_end_s'))} s with Q={p.get('q')}" for p in phases
                            ),
                        )
                    ]
                    if phases
                    else []
                ),
                ("stability bound dt [s]", fmt_num(run["dt_stable_s"])),
                (
                    "grid",
                    f"{'×'.join(str(n) for n in run['shape'])} @ {run['dx_m'] * 1e3:.4g} mm",
                ),
            ]
        ),
        "",
        "## Environment",
        "",
        *_kv(
            [
                ("caustica", f"{payload.get('caustica')} @ {str(payload.get('git_commit'))[:12]}"),
                ("host", str(payload.get("host", "?"))),
                (
                    "python / numpy / scipy",
                    f"{env.get('python')} / {env.get('numpy')} / {env.get('scipy')}",
                ),
                ("platform", str(env.get("platform", "?"))),
                ("resolved backend", str(env.get("resolved_backend", "?"))),
                (
                    "GPU",
                    f"{env.get('gpu_name')} ({env.get('vram_total_gib')} GiB)"
                    if env.get("gpu_name")
                    else "— (no CUDA device)",
                ),
            ]
        ),
        "",
    ]

    if payload["notes"]:
        lines += ["## Caveats", "", *[f"- {n}" for n in payload["notes"]], ""]

    lines += [
        "## Files",
        "",
        f"- `{JSON_NAME}` — this report, machine-readable (`{payload['format']}`), "
        f"including the liability note above verbatim",
        "",
        "---",
        "",
        MEDICAL_LIABILITY_NOTE,
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------- writing


def write_thermal_report(
    result: Any,
    medium: Any,
    outdir: str | Path,
    **kwargs: Any,
) -> Path:
    """Write ``THERMAL.md`` + ``thermal.json``; returns the folder.

    Both files go through :func:`caustica.io.atomic.atomic_write`: the JSON is
    the machine half, something else will read it, and a half-written safety
    report that merely EXISTS is exactly the failure atomic writes exist to
    prevent. ``kwargs`` are :func:`thermal_payload`'s.
    """
    payload = thermal_payload(result, medium, **kwargs)
    return write_payload(payload, outdir)


def write_payload(payload: Mapping[str, Any], outdir: str | Path) -> Path:
    """Write an already-built payload (the split :func:`write_thermal_report` uses)."""
    if payload.get("medical_liability_note") != MEDICAL_LIABILITY_NOTE:
        # The one invariant this module refuses to let a caller edit out.
        raise ValueError(
            "the payload does not carry the medical liability note verbatim; it is not "
            "an optional field of a caustica thermal report."
        )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with atomic_write(outdir / JSON_NAME) as tmp:
        tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    with atomic_write(outdir / MD_NAME) as tmp:
        tmp.write_text(render_markdown(payload), encoding="utf-8")
    return outdir
