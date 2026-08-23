"""M10d: shared focal metrics, the preview package and `caustica report`.

The criteria under test:
* metric definitions are single-sourced — focus_study and the library
  compute IDENTICAL numbers from the same solve;
* the preview package stays <= 10 MB measured on a full-size grid, and its
  contents reproduce the stored field within the float16 contract;
* `caustica report` renders from result.h5 AND from the preview alone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from caustica.__main__ import main as cli_main
from caustica.report.metrics import FieldFrame, focus_metrics
from caustica.report.preview import (
    DEFAULT_MAX_BYTES,
    block_mean,
    load_preview,
    write_preview,
)
from caustica.runner import EXIT_OK, RunnerOptions, run_job_file
from caustica.solvers.base import SolverResult

JOB_FORMAT = "caustica-job/1"


# ------------------------------------------------------------- shared solves


@pytest.fixture(scope="module")
def water_setup_result():
    """One coarse focus_study water-bowl solve, shared by the parity tests."""
    from apps.focus_study import scenarios

    knobs = scenarios.Knobs(dx=0.6e-3, min_settle_periods=2, max_settle_periods=6)
    setup = scenarios.water_bowl(knobs)
    import caustica.solvers as solvers

    result = solvers.get(knobs.solver)().run(
        setup.grid,
        setup.medium,
        setup.source,
        setup.spec,
        backend="numpy",
        reference_point=setup.focus_vox,
        harmonics=knobs.harmonics,
    )
    return setup, result


@pytest.fixture(scope="module")
def runner_outdir(tmp_path_factory) -> Path:
    """One mini runner job (float32 store, h1+h2), shared by the report tests."""
    tmp = tmp_path_factory.mktemp("report_run")
    job = {
        "format": JOB_FORMAT,
        "kind": "explicit",
        "name": "mini-report",
        "medium": {"kind": "homogeneous"},
        "grid": {"ndim": 3, "dx_mm": 0.5, "size_mm": [18, 18, 24], "pml": {"thickness_mm": 3.0}},
        "source": {
            "kind": "array",
            "array": {"kind": "bowl", "d_outer_mm": 10.0, "roc_mm": 12.0},
            "apex_mm": [9, 9, 6.0],
        },
        "drive": {"f0_mhz": 1.0, "amplitude_kpa": 100.0},
        "run": {"spec": {"min_settle_periods": 2, "max_settle_periods": 6}, "harmonics": [1, 2]},
        "solver": "westervelt",
        "output": {"quantize": False},
    }
    job_path = tmp / "mini.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    out = tmp / "out"
    code = run_job_file(job_path, RunnerOptions(out=out, measure=False, status_interval_s=0.0))
    assert code == EXIT_OK
    return out


# --------------------------------------------- single source of truth (M10d)


def test_focus_study_and_library_compute_identical_metrics(water_setup_result):
    """analyze() and focus_metrics() must agree to the last rounded digit."""
    from apps.focus_study import analysis

    setup, result = water_setup_result
    via_app = analysis.analyze(setup, result)
    via_lib = focus_metrics(
        result,
        FieldFrame(
            dx=setup.grid.dx,
            grid_shape=setup.grid.shape,
            pml_vox=setup.grid.pml_vox,
            apex_vox=setup.apex_vox,
            focus_vox=setup.focus_vox,
        ),
        source_amplitude=setup.knobs.amplitude,
        medium=setup.medium,
        solver=setup.knobs.solver,
    )
    for section in ("peak", "target", "focal_spot", "run", "harmonics"):
        assert via_app[section] == via_lib[section], section


def test_analyze_keeps_the_pre_refactor_shape(water_setup_result):
    """Key order and presence — metrics.json byte-stability depends on it."""
    from apps.focus_study import analysis

    setup, result = water_setup_result
    m = analysis.analyze(setup, result)
    assert list(m) == [
        "peak",
        "target",
        "focal_spot",
        "run",
        "harmonics",
        "warnings",  # janitor ticket 09: the metrics' own caveat channel
        "vs_oneill",
    ]
    assert m["peak"]["gain_vs_source"] is not None
    assert m["peak"]["isppa_w_cm2"] is not None  # medium was available


def test_metrics_from_result_file_match_the_runner_metrics(runner_outdir):
    """The h5 path (no medium) reproduces the runner's numbers.

    quantize=False in the job, so the only difference between the in-memory
    field and the reloaded one is the float32 cast — metric-level agreement
    must be tight; isppa is the one honest exception (needs the medium).
    """
    import h5py

    from caustica.io.store import load_result

    stored = json.loads((runner_outdir / "metrics.json").read_text(encoding="utf-8"))
    result = load_result(runner_outdir / "result.h5")
    with h5py.File(runner_outdir / "result.h5", "r") as hf:
        a = dict(hf.attrs)
        amplitude = float(hf["input"].attrs["amplitude_pa"])
    m = focus_metrics(
        result,
        FieldFrame(
            dx=float(a["dx_m"]),
            grid_shape=tuple(int(v) for v in a["grid_shape"]),
            pml_vox=int(a["pml_vox"]),
            apex_vox=tuple(int(v) for v in a["apex_vox"]),
            focus_vox=tuple(int(v) for v in a["focus_vox"]),
        ),
        source_amplitude=amplitude,
        medium=None,
    )
    assert m["peak"]["voxel_grid"] == stored["peak"]["voxel_grid"]
    assert m["peak"]["p_pa"] == pytest.approx(stored["peak"]["p_pa"], rel=1e-5)
    assert m["peak"]["gain_vs_source"] == pytest.approx(stored["peak"]["gain_vs_source"], abs=0.01)
    assert m["peak"]["isppa_w_cm2"] is None and stored["peak"]["isppa_w_cm2"] is not None
    assert m["target"]["hit_ratio"] == pytest.approx(stored["target"]["hit_ratio"], abs=1e-3)
    assert m["focal_spot"]["axial_6db"]["width_mm"] == pytest.approx(
        stored["focal_spot"]["axial_6db"]["width_mm"], abs=1e-2
    )
    assert m["harmonics"]["a2_over_a1_at_peak_pct"] == pytest.approx(
        stored["harmonics"]["a2_over_a1_at_peak_pct"], abs=0.05
    )
    assert m["run"] == stored["run"]


# ----------------------------------------------------------- preview package


def _synthetic_result(shape: tuple[int, int, int]) -> SolverResult:
    """A full-grid-sized fake solve: smooth field with a known focal blob."""
    rng = np.random.default_rng(7)
    zz = np.linspace(-1, 1, shape[2])[None, None, :]
    xx = np.linspace(-1, 1, shape[0])[:, None, None]
    yy = np.linspace(-1, 1, shape[1])[None, :, None]
    amp = 1e6 * np.exp(-8.0 * (xx**2 + yy**2 + (zz - 0.2) ** 2)).astype(np.float32)
    amp += rng.random(shape, dtype=np.float32) * 1e3
    phasor = (amp * np.exp(1j * 0.3)).astype(np.complex64)
    return SolverResult(
        phasor=phasor,
        p_max=amp * 1.1,
        region=tuple(slice(0, n) for n in shape),
        dt=1e-7,
        spp=10,
        steps_total=100,
        t_end_s=1e-5,
        tof_periods=5,
        converged_period=8,
        settle_capped=False,
        convergence_history=[(i, 1e5 * i, 0.1 / (i + 1)) for i in range(8)],
        phasors={1: phasor},
        meta={"solver": "synthetic"},
    )


def test_preview_stays_under_10mb_on_a_full_grid(tmp_path):
    """The M10d gate, measured: a 256^3 full-grid field -> <= 10 MB package."""
    shape = (256, 256, 256)
    result = _synthetic_result(shape)
    path = write_preview(
        tmp_path,
        result,
        FieldFrame(
            dx=0.25e-3,
            grid_shape=shape,
            pml_vox=8,
            apex_vox=(128, 128, 12),
            focus_vox=(128, 128, 150),
        ),
        metrics={"format": "caustica-metrics/1", "job": "synthetic"},
    )
    size = path.stat().st_size
    assert size <= DEFAULT_MAX_BYTES, f"preview is {size} bytes"
    pre = load_preview(path)
    step = pre["meta"]["coarse_step"]
    assert step > 1  # a full grid MUST have been coarsened to fit
    # The coarse volume + slices reproduce the field within the f16 contract.
    peak = float(result.amp.max())
    assert pre["coarse_amp"].shape == tuple((n // step) for n in shape)
    np.testing.assert_allclose(
        pre["amp_h1_slice_y"],
        result.amp[:, pre["meta"]["peak_voxel_grid"][1], :],
        atol=1.1e-3 * peak,
    )
    assert (tmp_path / "metrics.json").exists()


def test_block_mean_is_a_true_block_average():
    arr = np.arange(4 * 4 * 4, dtype=np.float64).reshape(4, 4, 4)
    out = block_mean(arr, 2)
    assert out.shape == (2, 2, 2)
    assert out[0, 0, 0] == pytest.approx(arr[:2, :2, :2].mean())
    assert out[1, 1, 1] == pytest.approx(arr[2:, 2:, 2:].mean())
    # step 1 is a no-op (float32 cast aside)
    np.testing.assert_array_equal(block_mean(arr, 1), arr.astype(np.float32))


def test_runner_writes_the_preview_package(runner_outdir):
    """`caustica run` leaves preview.npz + metrics.json next to result.h5."""
    from caustica.io.store import load_result

    pre = load_preview(runner_outdir / "preview.npz")
    result = load_result(runner_outdir / "result.h5")
    pk = pre["meta"]["peak_voxel_grid"]
    org = [s.start for s in result.region]
    peak = float(result.amp.max())
    np.testing.assert_allclose(
        pre["amp_h1_slice_z"],
        result.amp[:, :, pk[2] - org[2]],
        atol=1.1e-3 * peak,
    )
    assert pre["meta"]["harmonics"] == [1, 2]
    assert "amp_h2_slice_y" in pre and "p_max_slice_y" in pre
    metrics = json.loads((runner_outdir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["job"] == "mini-report"
    assert metrics["peak"]["isppa_w_cm2"] is not None  # runner had the medium


# ---------------------------------------------------------- caustica report


def test_report_from_result_h5(runner_outdir):
    code = cli_main(["report", str(runner_outdir)])
    assert code == 0
    html = (runner_outdir / "index.html").read_text(encoding="utf-8")
    assert "mini-report" in html
    assert (runner_outdir / "REPORT.md").exists()
    for fig in ("fig_field.png", "fig_profiles.png", "fig_harmonics.png", "fig_convergence.png"):
        assert (runner_outdir / fig).exists(), fig
    # figures referenced by the page actually exist next to it
    assert "fig_field.png" in html


def test_report_from_preview_alone(runner_outdir, tmp_path):
    """Copy ONLY the preview package elsewhere — report must still render."""
    import shutil

    lone = tmp_path / "lone"
    lone.mkdir()
    for name in ("preview.npz", "metrics.json", "run_meta.json"):
        shutil.copy2(runner_outdir / name, lone / name)
    code = cli_main(["report", str(lone)])
    assert code == 0
    html = (lone / "index.html").read_text(encoding="utf-8")
    assert "preview" in html
    assert (lone / "fig_preview.png").exists()
    assert not (lone / "fig_field.png").exists()  # full figures need result.h5


def test_report_on_an_empty_folder_is_a_clean_error(tmp_path):
    assert cli_main(["report", str(tmp_path)]) == 2


def test_coarse_step_formula_targets_the_budget():
    """Sanity on the sizing math: the chosen step fits the float16 volume."""
    from caustica.report.preview import _coarse_step

    shape = (512, 512, 512)
    budget = 6 * 1024 * 1024
    s = _coarse_step(shape, budget)
    assert 2 * math.prod(n // s for n in shape) <= budget
    assert _coarse_step((32, 32, 32), budget) == 1  # small grids stay exact


# ------------------------------------------- janitor round (2026-08-21)


def test_report_on_pre_m10d_result_without_apex_attrs(runner_outdir, tmp_path):
    """An M10c-era result.h5 (no apex_vox/focus_vox stamp) must still report.

    Every pre-M10d output folder on a Drive is exactly this shape; the report
    falls back to the grid origin and says so in a caveat.
    """
    from caustica.io.store import load_result, save_result
    from caustica.sources import CWSource

    result = load_result(runner_outdir / "result.h5")
    src = CWSource(
        indices=np.array([[2, 2, 2], [3, 3, 3]], dtype=np.int32),
        phases=np.zeros(2, dtype=np.float32),
        amplitude=1e5,
        f0=1e6,
    )
    old = tmp_path / "old"
    old.mkdir()
    save_result(
        old / "result.h5",
        result,
        src,
        dx=0.5e-3,
        grid_shape=(36, 36, 48),
        pml_vox=6,
        quantize=False,
        # NO apex_vox / focus_vox extra_attrs — the pre-M10d layout.
    )
    assert cli_main(["report", str(old)]) == 0
    report = (old / "REPORT.md").read_text(encoding="utf-8")
    assert "predates the apex_vox stamp" in report
    assert "I_sppa needs the medium" in report  # metrics were recomputed
    metrics = json.loads((old / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["peak"]["isppa_w_cm2"] is None


def test_preview_roundtrip_with_offset_record_region(tmp_path):
    """Region start != 0: grid-frame bookkeeping through metrics + preview."""
    from caustica.report.metrics import focus_metrics

    shape = (40, 44, 60)
    off = (10, 12, 20)
    grid_shape = tuple(o + n + 8 for o, n in zip(off, shape, strict=True))
    base = _synthetic_result(shape)
    result = SolverResult(
        phasor=base.phasor,
        p_max=base.p_max,
        region=tuple(slice(o, o + n) for o, n in zip(off, shape, strict=True)),
        dt=base.dt,
        spp=base.spp,
        steps_total=base.steps_total,
        t_end_s=base.t_end_s,
        tof_periods=base.tof_periods,
        converged_period=base.converged_period,
        settle_capped=base.settle_capped,
        convergence_history=base.convergence_history,
        phasors=base.phasors,
        meta=base.meta,
    )
    apex = (off[0] + shape[0] // 2, off[1] + shape[1] // 2, off[2] + 2)
    frame = FieldFrame(dx=0.5e-3, grid_shape=grid_shape, pml_vox=4, apex_vox=apex)
    m = focus_metrics(result, frame)
    pk_grid = m["peak"]["voxel_grid"]
    # The stored voxel is in the GRID frame: region-frame argmax + offset.
    pk_region = np.unravel_index(int(np.argmax(result.amp)), shape)
    assert pk_grid == [int(i) + o for i, o in zip(pk_region, off, strict=True)]

    path = write_preview(
        tmp_path, result, frame, metrics={"format": "caustica-metrics/1", "job": "offset"}
    )
    pre = load_preview(path)
    assert pre["meta"]["peak_voxel_grid"] == pk_grid
    assert pre["meta"]["region_start"] == list(off)
    pk_r = [g - o for g, o in zip(pk_grid, off, strict=True)]
    peak = float(result.amp.max())
    np.testing.assert_allclose(pre["amp_h1_slice_y"], result.amp[:, pk_r[1], :], atol=1.1e-3 * peak)
    # focus_vox=None: no target section, meta records null.
    assert "target" not in m
    assert pre["meta"]["focus_vox"] is None


def test_load_preview_rejects_wrong_format(tmp_path):
    import numpy as _np

    bad = tmp_path / "preview.npz"
    with open(bad, "wb") as fh:
        _np.savez(fh, meta_json=_np.str_(json.dumps({"format": "other/1"})))
    with pytest.raises(ValueError, match="caustica-preview/1"):
        load_preview(bad)


def test_report_cli_on_truncated_preview_is_a_clean_error(runner_outdir, tmp_path):
    """A half-synced preview.npz (THE Drive failure mode) exits 2, no traceback."""
    lone = tmp_path / "half"
    lone.mkdir()
    data = (runner_outdir / "preview.npz").read_bytes()
    (lone / "preview.npz").write_bytes(data[: len(data) // 2])
    assert cli_main(["report", str(lone)]) == 2


def test_report_preview_flag_without_package_is_a_clear_error(runner_outdir, tmp_path, capsys):
    """--preview with only result.h5 present: exit 2 AND point at the fix."""
    import shutil

    lone = tmp_path / "onlyresult"
    lone.mkdir()
    shutil.copy2(runner_outdir / "result.h5", lone / "result.h5")
    assert cli_main(["report", str(lone), "--preview"]) == 2
    err = capsys.readouterr().err
    assert "no preview.npz" in err and "result.h5 exists" in err


def test_render_html_with_no_rows_emits_no_stray_table_close(tmp_path):
    from caustica.report.html import render_html

    out = render_html(
        tmp_path / "index.html",
        page_title="t",
        title="t",
        description="d",
        rows=[],
    )
    html = out.read_text(encoding="utf-8")
    assert "</table>" not in html and "<table>" not in html


# ------------------------------- janitor ticket 01 (2026-08-23): the gaps left


def test_preview_report_without_metrics_json(runner_outdir, tmp_path):
    """The FIRST-sync state on a Drive: preview.npz landed, nothing else has.

    That is the whole reason the preview package exists — judge a Colab run
    before the rest of the folder syncs — so the report must render from it
    alone: header + quicklook, and no metrics section conjured out of a file
    that is not there.
    """
    import shutil

    lone = tmp_path / "first_sync"
    lone.mkdir()
    shutil.copy2(runner_outdir / "preview.npz", lone / "preview.npz")
    assert cli_main(["report", str(lone), "--preview"]) == 0
    assert (lone / "fig_preview.png").exists()
    report = (lone / "REPORT.md").read_text(encoding="utf-8")
    assert "first_sync" in report  # no job name anywhere: the folder names it
    assert "preview package" in report
    assert "peak pressure" not in report and "Focal spot" not in report
    assert not (lone / "metrics.json").exists()  # the preview path invents nothing


def test_extent_6db_reports_truncation_instead_of_a_boundary_number():
    """A profile that never reaches -6 dB inside the recorded region.

    The honest answer is "cannot measure": pinning the crossing to the
    region boundary would quote a focal width the run never resolved.
    """
    from caustica.report.metrics import extent_6db

    coord = np.linspace(0.0, 1e-2, 21)
    prof = np.full(21, 1e6, dtype=np.float32)  # flat: never falls below half
    assert extent_6db(coord, prof, 10) == {
        "left_mm": None,
        "right_mm": None,
        "width_mm": None,
        "truncated": True,
    }


def test_target_outside_the_recorded_region_has_no_hit_ratio():
    """A requested focus outside the record region: no pressure to report.

    The record region may be a focal box; a target outside it is a real
    (mis)configuration, and reading it would index the wrong voxel rather
    than fail. p_pa/hit_ratio go None, the displacement stays honest.
    """
    shape = (24, 24, 32)
    result = _synthetic_result(shape)
    m = focus_metrics(
        result,
        FieldFrame(
            dx=0.5e-3,
            grid_shape=(40, 40, 60),
            pml_vox=2,
            apex_vox=(12, 12, 2),
            focus_vox=(12, 12, shape[2] + 10),
        ),
    )
    assert m["target"]["p_pa"] is None and m["target"]["hit_ratio"] is None
    assert m["target"]["displacement_norm_mm"] > 0.0  # geometry still measurable


def test_write_preview_coarsens_further_until_the_package_actually_fits(tmp_path, monkeypatch):
    """The budget is a MEASUREMENT, not an estimate — the retry loop proves it.

    ``_coarse_step`` sizes the volume from raw float16 bytes; when
    compression undershoots that guess the package is rebuilt one step
    coarser. Here the estimate is pinned to a step whose package is measured
    first, then the budget is set one byte below it — so the loop MUST run.
    """
    from caustica.report import preview as pv

    shape = (48, 48, 48)
    result = _synthetic_result(shape)
    frame = FieldFrame(dx=0.5e-3, grid_shape=shape, pml_vox=2, apex_vox=(24, 24, 4))
    monkeypatch.setattr(pv, "_coarse_step", lambda shape, budget: 2)
    too_big = len(pv.build_preview(result, frame, max_bytes=10**9))

    path = write_preview(tmp_path, result, frame, metrics={"job": "tight"}, max_bytes=too_big - 1)
    assert path.stat().st_size <= too_big - 1
    assert load_preview(path)["meta"]["coarse_step"] > 2  # step+1 until it fit


def _capture_figures(monkeypatch) -> dict:
    """Keep the Figure objects ``figures._save`` writes out.

    A decoration that was NOT drawn is invisible in the PNG; the artists on
    the axes are the only place the contract can be read back.
    """
    from caustica.report import figures as hfig

    kept: dict = {}
    real_save = hfig._save

    def spy(fig, outdir, name):
        kept[name] = fig
        return real_save(fig, outdir, name)

    monkeypatch.setattr(hfig, "_save", spy)
    return kept


def test_field_maps_without_source_indices_draws_no_source_dots(tmp_path, monkeypatch):
    """FigureContext's optional fields DROP a decoration, never fail.

    ``caustica report`` always has the source voxels (result.h5 stores
    them), but a context built by hand — a GUI, an app, a preview-only
    caller — may not, and the figure must still render.
    """
    from dataclasses import replace

    from caustica.report import figures as hfig
    from caustica.report.metrics import axial_profiles

    shape = (24, 24, 32)
    result = _synthetic_result(shape)
    frame = FieldFrame(dx=0.5e-3, grid_shape=shape, pml_vox=2, apex_vox=(12, 12, 2))
    prof = axial_profiles(result, frame)
    kept = _capture_figures(monkeypatch)

    ctx = hfig.FigureContext(frame=frame, title="no source")  # no source_indices, no focus_vox
    assert hfig.field_maps(ctx, result, prof, tmp_path) == "fig_field.png"
    assert [ln.get_marker() for ln in kept["fig_field"].axes[0].lines] == ["x"]  # peak only

    src = np.array([[12, 12, 4], [13, 12, 4]], dtype=np.int32)
    hfig.field_maps(replace(ctx, source_indices=src), result, prof, tmp_path)
    assert sorted(ln.get_marker() for ln in kept["fig_field"].axes[0].lines) == [".", "x"]


def test_medium_figure_drops_the_sound_speed_panel_when_there_is_none(tmp_path, monkeypatch):
    """Labels alone still earn a figure; no labels earn none at all."""
    from dataclasses import replace

    from caustica.report import figures as hfig

    shape = (16, 16, 20)
    labels = np.zeros(shape, np.int32)
    labels[:, :, 10:] = 1
    kept = _capture_figures(monkeypatch)
    ctx = hfig.FigureContext(
        frame=FieldFrame(dx=0.5e-3, grid_shape=shape, pml_vox=2, apex_vox=(8, 8, 2)),
        title="labels only",
        labels=labels,
        label_names={0: "water", 1: "fat"},
    )
    assert hfig.medium_plot(ctx, tmp_path) == "fig_medium.png"
    fig = kept["fig_medium"]
    assert fig.axes[0].images and not fig.axes[1].images  # c panel never drawn
    assert len(fig.axes) == 2  # ...so no colorbar axes came with it either
    assert hfig.medium_plot(replace(ctx, labels=None), tmp_path) is None


def test_convergence_figure_is_skipped_when_no_history_was_recorded(tmp_path):
    """No recorded periods (a resumed run, a single-period solve) -> no figure.

    ``make_all`` filters the None out, so an empty axes pair never reaches
    the report; that filtering only works if the plot admits it has nothing.
    """
    from dataclasses import replace

    from caustica.report import figures as hfig

    result = replace(_synthetic_result((16, 16, 20)), convergence_history=[])
    assert hfig.convergence_plot(result, tmp_path) is None
    assert not (tmp_path / "fig_convergence.png").exists()


def test_record_region_entirely_inside_the_pml_is_refused():
    """Nothing to analyze: the peak hunt must not report a sponge artefact.

    A tiny record region on a big grid (a mis-set record_region_vox) lands
    completely inside the PML margin — the honest answer is a refusal, not
    an argmax over damped voxels.
    """
    from caustica.report.metrics import interior_slices

    result = _synthetic_result((4, 4, 4))
    with pytest.raises(ValueError, match="entirely inside the PML"):
        interior_slices(
            result,
            FieldFrame(dx=0.5e-3, grid_shape=(40, 40, 40), pml_vox=10, apex_vox=(20, 20, 20)),
        )


# ------------------------ janitor ticket 09 (2026-08-23): the a2 edge caveat


def _result_with_a2_spike_at(shape: tuple[int, int, int], spike: tuple[int, int, int]):
    """A two-harmonic solve whose a2 maximum sits exactly on ``spike``.

    The fundamental keeps its own focal blob in the middle, so the a2 peak and
    the fundamental peak are deliberately different voxels — which is the whole
    situation the caveat is about.
    """
    from dataclasses import replace

    base = _synthetic_result(shape)
    a2 = np.full(shape, 1.0, dtype=np.float32)
    a2[spike] = 5e4
    return replace(base, phasors={1: base.phasor, 2: a2.astype(np.complex64)})


def test_pml_edge_distance_is_measured_to_the_nearest_sponge_face():
    """Distance 0 means "on the first non-absorbing plane", and it is a MIN.

    One axis sitting against the sponge is enough to make the voxel suspect,
    however deep in the grid the other two are.
    """
    from caustica.report.metrics import pml_edge_distance

    frame = FieldFrame(dx=0.5e-3, grid_shape=(40, 40, 60), pml_vox=6, apex_vox=(20, 20, 2))
    assert pml_edge_distance((6, 20, 30), frame) == 0  # first interior plane
    assert pml_edge_distance((10, 20, 30), frame) == 4
    assert pml_edge_distance((20, 20, 53), frame) == 0  # far face: 60 - 6 - 1
    # x is 13 from its nearest face, z is 23 from its own: the MIN is reported.
    assert pml_edge_distance((20, 20, 30), frame) == 13


def test_a2_peak_on_the_pml_edge_is_flagged_without_changing_its_value():
    """The number is contractual; the caveat next to it is the fix (ticket 09).

    A beta=0 run reported an a2 maximum at 6.5% of the fundamental from a
    voxel 4 steps inside the sponge — harmonic-DFT edge residue, not physics.
    """
    shape = (30, 30, 40)
    frame = FieldFrame(dx=0.5e-3, grid_shape=shape, pml_vox=6, apex_vox=(15, 15, 2))
    near = _result_with_a2_spike_at(shape, (15, 15, 9))  # 3 voxels from the PML
    m = focus_metrics(near, frame)

    assert m["harmonics"]["a2_peak_voxel_grid"] == [15, 15, 9]
    assert m["harmonics"]["a2_peak_pa"] == pytest.approx(5e4, rel=1e-6)  # untouched
    assert m["harmonics"]["a2_peak_distance_to_pml_vox"] == 3
    assert len(m["warnings"]) == 1
    assert "PML edge" in m["warnings"][0] and "a2_at_fundamental_peak_pa" in m["warnings"][0]


def test_an_a2_peak_in_the_middle_of_the_grid_earns_no_caveat():
    """Far from the sponge: the distance is still reported, the channel is empty.

    The spike sits as deep as this grid allows — exactly ON the threshold,
    which is the boundary the rule must not warn about (``< 8``, not ``<=``).
    """
    from caustica.report.metrics import A2_PML_MARGIN_WARN_VOX

    shape = (30, 30, 40)
    frame = FieldFrame(dx=0.5e-3, grid_shape=shape, pml_vox=6, apex_vox=(15, 15, 2))
    far = _result_with_a2_spike_at(shape, (15, 15, 20))
    m = focus_metrics(far, frame)

    dist = m["harmonics"]["a2_peak_distance_to_pml_vox"]
    assert dist == A2_PML_MARGIN_WARN_VOX == 8
    assert m["warnings"] == []


def test_the_a2_caveat_reaches_the_report_row_and_only_when_it_applies():
    """REPORT.md/index.html carry the caveat on the A2 maximum row itself."""
    from caustica.report.html import harmonics_rows

    shape = (30, 30, 40)
    frame = FieldFrame(dx=0.5e-3, grid_shape=shape, pml_vox=6, apex_vox=(15, 15, 2))
    near = focus_metrics(_result_with_a2_spike_at(shape, (15, 15, 9)), frame)
    far = focus_metrics(_result_with_a2_spike_at(shape, (15, 15, 20)), frame)

    def a2_max_row(m: dict) -> str:
        return next(v for sec, label, v in harmonics_rows(m) if label == "A2 maximum")

    assert "edge artifact" in a2_max_row(near)
    assert "edge artifact" not in a2_max_row(far)
    # A metrics.json written before the field existed still renders, uncaveated.
    legacy = {"harmonics": {k: v for k, v in near["harmonics"].items() if "distance" not in k}}
    assert "edge artifact" not in a2_max_row(legacy)


def test_the_new_harmonics_field_is_purely_additive(runner_outdir):
    """A real run's metrics.json gains the field and nothing else moves.

    The pre-ticket key order and values are pinned here so a later edit to
    focus_metrics cannot quietly reshuffle a published contract.
    """
    metrics = json.loads((runner_outdir / "metrics.json").read_text(encoding="utf-8"))
    assert list(metrics["harmonics"]) == [
        "a2_at_fundamental_peak_pa",
        "a2_over_a1_at_peak_pct",
        "a2_peak_pa",
        "a2_peak_voxel_grid",
        "a2_peak_distance_to_pml_vox",  # the only addition, and it is last
    ]
    assert list(metrics)[:-1] == [
        "format",
        "job",
        "generated",
        "peak",
        "target",
        "focal_spot",
        "run",
        "harmonics",
    ]
    assert list(metrics)[-1] == "warnings"
    assert isinstance(metrics["warnings"], list)
