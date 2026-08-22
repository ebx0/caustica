"""M10m gate: bring your own transducer — the ``elements`` array kind.

The acceptance question is a stranger's: *my* element table, from *my* file,
runs end to end and the reload can still falsify the geometry. Everything
here is a mini CPU job (seconds).
"""

import json
from pathlib import Path

import numpy as np
import pytest

import caustica.solvers as solvers
from caustica.arrays import elements_array, read_element_file
from caustica.config.job import (
    JOB_FORMAT,
    ElementsArrayConfig,
    JobError,
    build_job,
    load_job,
    validate_job,
)

# A tiny 8-element ring on a spherical shell, apex frame, MILLIMETRES.
ROC_MM = 12.0
N_ELEM = 8
RING_R_MM = 4.0


def ring_positions_mm() -> np.ndarray:
    th = np.linspace(0.0, 2.0 * np.pi, N_ELEM, endpoint=False)
    r = RING_R_MM
    z = ROC_MM - np.sqrt(ROC_MM**2 - r**2)  # on the shell of radius ROC
    return np.column_stack((r * np.cos(th), r * np.sin(th), np.full(N_ELEM, z)))


def elements_job_dict(array: dict, **over) -> dict:
    d = {
        "format": JOB_FORMAT,
        "kind": "explicit",
        "name": "elements-mini",
        "medium": {"kind": "homogeneous"},
        "grid": {
            "ndim": 3,
            "dx_mm": 0.5,
            "size_mm": [18, 18, 24],
            "pml": {"thickness_mm": 3.0},
        },
        "source": {"kind": "array", "array": array, "apex_mm": [9.0, 9.0, 6.0]},
        "drive": {"f0_mhz": 0.8, "amplitude_kpa": 100.0},
        "run": {"spec": {"min_settle_periods": 2, "max_settle_periods": 6}, "harmonics": [1]},
        "solver": "linear",
    }
    d.update(over)
    return d


def write_job(tmp_path: Path, d: dict, name: str = "job.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


# ----------------------------------------------------------------- the reader


def test_read_npz_and_csv_agree(tmp_path):
    pos = ring_positions_mm()
    nrm = np.array([0.0, 0.0, ROC_MM]) - pos
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

    npz = tmp_path / "ring.npz"
    np.savez(npz, positions=pos, normals=nrm)
    csv = tmp_path / "ring.csv"
    csv.write_text(
        "x,y,z,nx,ny,nz\n"
        + "\n".join(",".join(repr(float(v)) for v in row) for row in np.hstack((pos, nrm))),
        encoding="utf-8",
    )
    p_npz, n_npz = read_element_file(npz)
    p_csv, n_csv = read_element_file(csv)
    np.testing.assert_allclose(p_npz, p_csv)
    np.testing.assert_allclose(n_npz, n_csv)

    # 3-column csv: no normals, and a header is optional.
    bare = tmp_path / "bare.csv"
    bare.write_text(
        "\n".join(" ".join(repr(float(v)) for v in row) for row in pos), encoding="utf-8"
    )
    p_bare, n_bare = read_element_file(bare)
    np.testing.assert_allclose(p_bare, pos)
    assert n_bare is None


def test_reader_refusals_are_actionable(tmp_path):
    npz = tmp_path / "wrong.npz"
    np.savez(npz, centers=ring_positions_mm())
    with pytest.raises(ValueError, match="needs a 'positions' array"):
        read_element_file(npz)
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("1,2,3\n4,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row widths"):
        read_element_file(ragged)
    txt = tmp_path / "x.txt"
    txt.write_text("1,2,3", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported element-table format"):
        read_element_file(txt)
    with pytest.raises(FileNotFoundError):
        read_element_file(tmp_path / "nope.npz")


def test_missing_normals_aim_at_the_focus():
    pos = ring_positions_mm() * 1e-3
    arr = elements_array(positions=pos, elem_radius=1e-3, focal_length=ROC_MM * 1e-3)
    to_focus = arr.focus - arr.positions
    to_focus /= np.linalg.norm(to_focus, axis=1, keepdims=True)
    np.testing.assert_allclose(arr.normals, to_focus, atol=1e-12)


def test_unit_mistake_is_refused_not_run():
    """Metres-vs-millimetres is the failure that would otherwise run happily."""
    with pytest.raises(ValueError, match="unit mistake"):
        elements_array(  # positions passed in mm to a metres API
            positions=ring_positions_mm() * 1e3, elem_radius=1e-3, focal_length=ROC_MM * 1e-3
        )


# ------------------------------------------------------------------ the kind


def test_elements_job_from_npz_runs_end_to_end(tmp_path):
    """The M10m headline: my element table, my file, a real solve."""
    np.savez(tmp_path / "ring.npz", positions=ring_positions_mm())
    d = elements_job_dict(
        {
            "kind": "elements",
            "file": "ring.npz",  # relative -> resolved against the JOB file (T4)
            "elem_radius_mm": 1.2,
            "roc_mm": ROC_MM,
        }
    )
    job, base = load_job(write_job(tmp_path, d))
    built = build_job(job, base_dir=base)
    assert built.derived["n_elements"] == float(N_ELEM)
    assert built.derived["elements_represented"] == N_ELEM
    assert built.derived["phases"] == "zeros"

    res = solvers.get(built.solver)().run(
        built.grid,
        built.medium,
        built.source,
        built.spec,
        backend="numpy",
        record_region=built.record_region,
        reference_point=built.focus_vox,
        harmonics=built.harmonics,
    )
    # The ring focuses: the geometric focus beats the source plane's own level.
    assert float(res.amp[built.focus_vox]) > built.source.amplitude


def test_elements_derived_matches_on_reload(tmp_path):
    """A reload re-derives the geometry from the file and must agree."""
    np.savez(tmp_path / "ring.npz", positions=ring_positions_mm())
    d = elements_job_dict(
        {"kind": "elements", "file": "ring.npz", "elem_radius_mm": 1.2, "roc_mm": ROC_MM}
    )
    path = write_job(tmp_path, d)
    built = build_job(*load_job(path), with_medium=False)

    job2, base2 = load_job(path)  # a fresh load, as a report/resume would do
    job2.source.check_derived(built.derived, base_dir=base2)

    # ...and it is genuinely falsifiable: shift the table on disk by 1 mm.
    np.savez(tmp_path / "ring.npz", positions=ring_positions_mm() + [1.0, 0.0, 0.0])
    with pytest.raises(JobError, match="r_max_mm"):
        job2.source.check_derived(built.derived, base_dir=base2)


def test_inline_elements_match_the_same_table_from_file(tmp_path):
    pos = ring_positions_mm()
    np.savez(tmp_path / "ring.npz", positions=pos)
    common = {"kind": "elements", "elem_radius_mm": 1.2, "roc_mm": ROC_MM}
    from_file = build_job(
        *load_job(write_job(tmp_path, elements_job_dict({**common, "file": "ring.npz"}), "f.json")),
        with_medium=False,
    )
    inline = build_job(
        *load_job(
            write_job(
                tmp_path,
                elements_job_dict({**common, "positions_mm": pos.tolist()}),
                "i.json",
            )
        ),
        with_medium=False,
    )
    assert from_file.derived == inline.derived
    np.testing.assert_array_equal(from_file.source.indices, inline.source.indices)


def test_elements_can_be_steered_and_phased(tmp_path):
    """Unlike a bowl, an element table is a real phased array."""
    pos = ring_positions_mm().tolist()
    d = elements_job_dict(
        {"kind": "elements", "positions_mm": pos, "elem_radius_mm": 1.2, "roc_mm": ROC_MM}
    )
    d["source"]["focus"] = {"mode": "steered", "target_mm": [10.5, 9.0, 17.0]}
    built = build_job(*load_job(write_job(tmp_path, d, "steer.json")), with_medium=False)
    assert built.derived["phases"].startswith("das")
    assert float(np.abs(built.source.phases).max()) > 0.1

    d["source"].pop("focus")
    d["source"]["phases_rad"] = [0.1] * N_ELEM
    built2 = build_job(*load_job(write_job(tmp_path, d, "phased.json")), with_medium=False)
    assert built2.derived["phases"] == "explicit"

    d["source"]["phases_rad"] = [0.1] * (N_ELEM - 1)
    with pytest.raises(JobError, match=f"entries for {N_ELEM} elements"):
        build_job(*load_job(write_job(tmp_path, d, "bad.json")), with_medium=False)


def test_schema_refusals_teach(tmp_path):
    """Every impossible combination dies at load, saying what to do instead."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="exactly one of 'file' or 'positions_mm'"):
        ElementsArrayConfig(elem_radius_mm=1.0, roc_mm=ROC_MM)
    with pytest.raises(ValidationError, match="exactly one of 'file' or 'positions_mm'"):
        ElementsArrayConfig(
            elem_radius_mm=1.0, roc_mm=ROC_MM, file="a.npz", positions_mm=[(0, 0, 0)]
        )
    with pytest.raises(ValidationError, match="belongs with inline"):
        ElementsArrayConfig(elem_radius_mm=1.0, roc_mm=ROC_MM, file="a.npz", normals_mm=[(0, 0, 1)])
    with pytest.raises(ValidationError, match="1 entries for 2 positions_mm"):
        ElementsArrayConfig(
            elem_radius_mm=1.0,
            roc_mm=ROC_MM,
            positions_mm=[(0, 0, 0), (1, 0, 0)],
            normals_mm=[(0, 0, 1)],
        )
    with pytest.raises(ValidationError, match="size of the whole bowl"):
        ElementsArrayConfig(elem_radius_mm=ROC_MM, roc_mm=ROC_MM, positions_mm=[(0, 0, 0)])


def test_missing_element_file_names_the_path(tmp_path):
    d = elements_job_dict(
        {"kind": "elements", "file": "nowhere.npz", "elem_radius_mm": 1.2, "roc_mm": ROC_MM}
    )
    report = validate_job(write_job(tmp_path, d))
    assert not report.ok
    assert any("nowhere.npz" in e for e in report.errors), report.render()


def test_elements_round_trip_through_json():
    cfg = ElementsArrayConfig(
        elem_radius_mm=1.2,
        roc_mm=ROC_MM,
        positions_mm=tuple(map(tuple, ring_positions_mm())),
    )
    assert ElementsArrayConfig.model_validate_json(cfg.model_dump_json()) == cfg


def test_existing_validations_apply_to_elements(tmp_path):
    """Shape match, the dedup refusal and the source-PML gate are SHARED code."""
    pos = ring_positions_mm().tolist()

    # two elements inside one voxel -> voxelize's dedup refusal (dx = 0.5 mm)
    collided = [*pos, [pos[0][0] + 0.05, pos[0][1], pos[0][2]]]
    d = elements_job_dict(
        {"kind": "elements", "positions_mm": collided, "elem_radius_mm": 0.2, "roc_mm": ROC_MM}
    )
    report = validate_job(write_job(tmp_path, d, "coarse.json"))
    assert not report.ok
    assert any("deduplication" in e for e in report.errors), report.render()

    # apex inside the sponge -> the shared source-clears-PML gate
    d = elements_job_dict(
        {"kind": "elements", "positions_mm": pos, "elem_radius_mm": 1.2, "roc_mm": ROC_MM}
    )
    d["source"]["apex_mm"] = [9.0, 9.0, 1.0]
    report = validate_job(write_job(tmp_path, d, "pml.json"))
    assert not report.ok
    assert any("PML" in e for e in report.errors), report.render()
