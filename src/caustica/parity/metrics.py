"""The thirty parity metrics of the comparison contract, M-01 to M-30.

Ratios are the engine over the reference unless the metric says otherwise,
and every metric reads exactly one :class:`~caustica.parity.model.ParityCase`
(the example, one engine's run, the reference run).

This module EXTENDS ``caustica.report.metrics`` rather than replacing it: the
-6 dB extents come from :func:`caustica.report.metrics.extent_6db`, so a
focal width on a parity page and a focal width in a run report are the same
definition. What is new here is the comparison itself, which the report
module never needed: agreement, offset, ratio and the applicability of each.

Absence is never a number. A metric that cannot answer raises
:class:`~caustica.parity.model.NotComputed` (this run did not record it) or
:class:`~caustica.parity.model.NotApplicable` (the physics gives it no
meaning), with the sentence a reader will see in place of the value.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.ndimage import uniform_filter

from caustica.parity.model import NotApplicable, NotComputed, ParityCase
from caustica.parity.registry import metric
from caustica.report.metrics import extent_6db

#: SSIM window edge in voxels, and the two stabilizing constants of the
#: standard formulation (Wang et al. 2004) on images normalized to [0, 1].
SSIM_WINDOW = 7
SSIM_K1 = 0.01
SSIM_K2 = 0.03


# ---------------------------------------------------------------------------
# Applicability predicates
# ---------------------------------------------------------------------------


def always(case: ParityCase) -> str | None:
    """A metric that means something for every example."""
    return None


def needs(*trait_names: str) -> Callable[[ParityCase], str | None]:
    """Applicable only while every named trait holds, else the first reason."""

    def pred(case: ParityCase) -> str | None:
        for name in trait_names:
            t = case.example.trait(name)
            if not t.holds:
                return t.reason
        return None

    return pred


def needs_engine_reference(*trait_names: str) -> Callable[[ParityCase], str | None]:
    """As :func:`needs`, and only when the reference is an engine run."""
    base = needs(*trait_names)

    def pred(case: ParityCase) -> str | None:
        if case.reference.kind != "solver":
            return (
                f"the reference for this example is {case.reference.engine} "
                f"({case.reference.kind}), which takes no wall time and holds no device "
                f"memory, so a ratio against it would compare a solve with an evaluation."
            )
        return base(case)

    return pred


# ---------------------------------------------------------------------------
# Field access
# ---------------------------------------------------------------------------


def _field(run, what: str) -> np.ndarray:
    if run.field is None:
        raise NotComputed(
            f"the {run.engine} run recorded no {what} field, so there is nothing to "
            f"compare; record it in the example's mirror before publishing this row."
        )
    return np.asarray(run.field, dtype=np.float64)


def graded_pair(case: ParityCase) -> tuple[np.ndarray, np.ndarray]:
    """This engine's graded field and the reference's, shapes checked."""
    what = case.example.graded_quantity
    a = _field(case.run, what)
    b = _field(case.reference, what)
    if a.shape != b.shape:
        raise NotComputed(
            f"the {case.run.engine} field is {a.shape} and the {case.reference.engine} "
            f"reference is {b.shape}; the two were not graded on the same region."
        )
    return a, b


def _reference_scale(b: np.ndarray, case: ParityCase) -> float:
    peak = float(np.abs(b).max())
    if peak <= 0.0:
        raise NotComputed(
            f"the {case.reference.engine} reference field is identically zero, so a "
            f"relative error against it has no denominator."
        )
    return peak


def peak_index(field: np.ndarray) -> tuple[int, ...]:
    return tuple(int(i) for i in np.unravel_index(int(np.argmax(field)), field.shape))


def profile(field: np.ndarray, axis: int, through: tuple[int, ...]) -> np.ndarray:
    """The 1-D line along ``axis`` through voxel ``through``."""
    idx: list[object] = list(through)
    idx[axis] = slice(None)
    return np.asarray(field[tuple(idx)], dtype=np.float64)


def _beam_axis(case: ParityCase) -> int:
    axis = case.run.beam_axis
    ref_axis = case.reference.beam_axis
    if axis is None or ref_axis is None:
        raise NotComputed(
            "the mirror did not name a beam axis for both engines, so a profile through "
            "the peak has no direction to run along."
        )
    if axis != ref_axis:
        raise NotComputed(
            f"the {case.run.engine} run names axis {axis} as the beam axis and the "
            f"reference names axis {ref_axis}; the two profiles are not the same line."
        )
    return int(axis)


def _lateral_axes(case: ParityCase, field: np.ndarray) -> tuple[int, ...]:
    """The transverse axes, or a stated absence when the field has none.

    A 1-D field is all beam axis. Returning an empty tuple there would build
    a metric value out of an empty mapping, which is an absence wearing the
    shape of an answer, so the geometry says so instead.
    """
    beam = _beam_axis(case)
    lateral = tuple(d for d in range(field.ndim) if d != beam)
    if not lateral:
        raise NotApplicable(
            f"the graded field is {field.ndim}-D and its only axis is the beam axis, so "
            f"this example has no transverse direction to measure across."
        )
    return lateral


def pearson(a: np.ndarray, b: np.ndarray, what: str) -> float:
    """Pearson ``r`` between two arrays, with a stated refusal on a flat one."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size < 2:
        raise NotComputed(f"{what} holds {x.size} sample(s); a correlation needs at least two.")
    sx, sy = float(x.std()), float(y.std())
    if sx <= 0.0 or sy <= 0.0:
        raise NotComputed(
            f"{what} is constant in at least one engine (std {sx:.3g} and {sy:.3g}), so "
            f"Pearson r is undefined rather than perfect."
        )
    return float(np.corrcoef(x, y)[0, 1])


def relative_l2(a: np.ndarray, b: np.ndarray, what: str) -> float:
    denom = float(np.linalg.norm(np.asarray(b, dtype=np.float64).ravel()))
    if denom <= 0.0:
        raise NotComputed(
            f"{what}: the reference is identically zero, so a relative L2 has no denominator."
        )
    return float(np.linalg.norm((a - b).ravel()) / denom)


def ssim(a: np.ndarray, b: np.ndarray, window: int = SSIM_WINDOW) -> float:
    """Mean structural similarity of two images normalized to their own peak.

    The standard local formulation with a uniform window: local means, local
    variances and the local covariance, stabilized by ``C1`` and ``C2``. A
    uniform window rather than a Gaussian one keeps the dependency list at
    ``scipy.ndimage`` and moves the answer by well under a percent on the
    smooth fields a parity page compares.
    """
    if min(a.shape) < window:
        raise NotComputed(
            f"the graded field is {a.shape}, smaller than the {window}-voxel SSIM window; "
            f"a local similarity needs a window that fits."
        )
    c1 = SSIM_K1**2
    c2 = SSIM_K2**2
    size = (window,) * a.ndim
    mu_a = uniform_filter(a, size=size)
    mu_b = uniform_filter(b, size=size)
    saa = uniform_filter(a * a, size=size) - mu_a * mu_a
    sbb = uniform_filter(b * b, size=size) - mu_b * mu_b
    sab = uniform_filter(a * b, size=size) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (saa + sbb + c2)
    return float(np.mean(num / den))


def _six_db_width_m(field: np.ndarray, axis: int, dx: float) -> float:
    """-6 dB extent [m] of the profile along ``axis`` through the peak."""
    pk = peak_index(field)
    prof = profile(field, axis, pk)
    coord = np.arange(prof.size, dtype=np.float64) * dx
    ext = extent_6db(coord, prof, int(pk[axis]))
    if ext["truncated"]:
        raise NotComputed(
            f"the -6 dB extent along axis {axis} runs off the recorded region (the profile "
            f"never falls to half the peak on at least one side), so its width is a lower "
            f"bound and not a measurement."
        )
    return float(ext["width_mm"]) * 1e-3


def _dx(run) -> float:
    """The run's voxel pitch [m], or a stated absence when it recorded none.

    ``EngineRun.dx`` defaults to zero, which is not a pitch. A metric that
    turns voxels into metres would otherwise publish a zero length, which is
    a missing input dressed as a measurement.
    """
    dx = float(run.dx)
    if dx <= 0.0:
        raise NotComputed(
            f"the {run.engine} run carries no voxel pitch (dx = {dx:g} m), so a length "
            f"in voxels cannot be turned into a length in metres."
        )
    return dx


def _ratio(run_value: float, ref_value: float, what: str) -> float:
    if ref_value == 0.0:
        raise NotComputed(f"{what}: the reference value is zero, so the ratio is undefined.")
    return float(run_value / ref_value)


def _harmonic(run, n: int) -> np.ndarray:
    if n not in run.harmonics:
        raise NotComputed(
            f"the {run.engine} run recorded harmonics {sorted(run.harmonics)}; {n}f0 was "
            f"not among them, so its amplitude was never computed."
        )
    return np.asarray(run.harmonics[n], dtype=np.float64)


def _series(run, name: str):
    if name not in run.series:
        raise NotComputed(
            f"the {run.engine} run recorded the series {sorted(run.series)}; '{name}' was "
            f"not among them."
        )
    return run.series[name]


def _shared_series(case: ParityCase) -> tuple[str, ...]:
    both = sorted(set(case.run.series) & set(case.reference.series))
    if not both:
        raise NotComputed(
            f"the {case.run.engine} run recorded {sorted(case.run.series)} and the "
            f"{case.reference.engine} reference {sorted(case.reference.series)}; no sensor "
            f"point is recorded by both, so no waveform can be compared."
        )
    return tuple(both)


def _scalar(run, name: str, what: str) -> float:
    value = getattr(run, name)
    if value is None:
        raise NotComputed(f"the {run.engine} run does not carry {what}.")
    return float(value)


# ---------------------------------------------------------------------------
# Field agreement (M-01 to M-06)
# ---------------------------------------------------------------------------


@metric(
    "M-01", "Pearson r over the whole graded field", "field agreement", "always", applies=always
)
def m01_field_pearson(case: ParityCase) -> float:
    a, b = graded_pair(case)
    return pearson(a, b, "the graded field")


@metric(
    "M-02", "Relative L2 error of the field", "field agreement", "always", applies=always, units="-"
)
def m02_field_relative_l2(case: ParityCase) -> float:
    a, b = graded_pair(case)
    return relative_l2(a, b, "the graded field")


@metric(
    "M-03",
    "Relative L-infinity error of the field",
    "field agreement",
    "always",
    applies=always,
    units="-",
)
def m03_field_relative_linf(case: ParityCase) -> float:
    a, b = graded_pair(case)
    return float(np.abs(a - b).max() / _reference_scale(b, case))


@metric(
    "M-04",
    "Normalized RMSE, by the reference peak",
    "field agreement",
    "always",
    applies=always,
    units="-",
)
def m04_field_nrmse(case: ParityCase) -> float:
    a, b = graded_pair(case)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return rmse / _reference_scale(b, case)


@metric(
    "M-05",
    "Mean signed bias, separating a systematic offset from scatter",
    "field agreement",
    "always",
    applies=always,
)
def m05_field_bias(case: ParityCase) -> dict[str, float]:
    a, b = graded_pair(case)
    scale = _reference_scale(b, case)
    d = a - b
    return {
        "bias": float(d.mean()),
        "bias_over_reference_peak": float(d.mean() / scale),
        "scatter_over_reference_peak": float(d.std() / scale),
    }


@metric(
    "M-06",
    "Structural similarity on the normalized magnitude image",
    "field agreement",
    "a 2-D or 3-D field exists",
    applies=needs("field"),
    units="-",
)
def m06_field_ssim(case: ParityCase) -> float:
    a, b = graded_pair(case)
    if a.ndim < 2:
        raise NotApplicable(
            "the graded quantity of this example is one dimensional, and structural "
            "similarity is a statement about an image."
        )
    scale_a = float(np.abs(a).max())
    scale_b = _reference_scale(b, case)
    if scale_a <= 0.0:
        raise NotComputed(
            f"the {case.run.engine} field is identically zero, so its normalized image "
            f"does not exist."
        )
    return ssim(np.abs(a) / scale_a, np.abs(b) / scale_b)


# ---------------------------------------------------------------------------
# Focus (M-07 to M-12)
# ---------------------------------------------------------------------------


@metric("M-07", "Peak ratio", "focus", "a peak exists", applies=needs("peak"), units="-")
def m07_peak_ratio(case: ParityCase) -> float:
    a, b = graded_pair(case)
    return _ratio(float(a.max()), float(b.max()), "the peak of the graded field")


@metric(
    "M-08",
    "Peak position offset, in voxels and in mm",
    "focus",
    "a peak exists",
    applies=needs("peak"),
)
def m08_peak_offset(case: ParityCase) -> dict[str, float]:
    a, b = graded_pair(case)
    pa = np.array(peak_index(a), dtype=np.float64)
    pb = np.array(peak_index(b), dtype=np.float64)
    d = pa - pb
    out: dict[str, float] = {f"axis_{i}_vox": float(v) for i, v in enumerate(d)}
    out["norm_vox"] = float(np.linalg.norm(d))
    out["norm_mm"] = float(np.linalg.norm(d) * _dx(case.reference) * 1e3)
    return out


@metric(
    "M-09",
    "Minus 6 dB focal volume ratio",
    "focus",
    "a focused beam",
    applies=needs("focused_beam"),
    units="-",
)
def m09_focal_volume_ratio(case: ParityCase) -> float:
    a, b = graded_pair(case)
    va = float((a >= 0.5 * a.max()).sum()) * _dx(case.run) ** a.ndim
    vb = float((b >= 0.5 * b.max()).sum()) * _dx(case.reference) ** b.ndim
    return _ratio(va, vb, "the -6 dB focal volume")


@metric(
    "M-10",
    "Minus 6 dB axial length ratio",
    "focus",
    "a focused beam",
    applies=needs("focused_beam", "beam_axis"),
    units="-",
)
def m10_axial_length_ratio(case: ParityCase) -> float:
    a, b = graded_pair(case)
    axis = _beam_axis(case)
    return _ratio(
        _six_db_width_m(a, axis, _dx(case.run)),
        _six_db_width_m(b, axis, _dx(case.reference)),
        "the -6 dB axial length",
    )


@metric(
    "M-11",
    "Minus 6 dB lateral width ratio, both transverse axes",
    "focus",
    "a focused beam",
    applies=needs("focused_beam", "beam_axis"),
    units="-",
)
def m11_lateral_width_ratio(case: ParityCase) -> dict[str, float]:
    a, b = graded_pair(case)
    out: dict[str, float] = {}
    for axis in _lateral_axes(case, a):
        out[f"axis_{axis}"] = _ratio(
            _six_db_width_m(a, axis, _dx(case.run)),
            _six_db_width_m(b, axis, _dx(case.reference)),
            f"the -6 dB lateral width along axis {axis}",
        )
    return out


@metric(
    "M-12",
    "Focal gain, peak over source-surface pressure",
    "focus",
    "a source with a defined surface",
    applies=needs("source_surface", "peak"),
)
def m12_focal_gain(case: ParityCase) -> dict[str, float]:
    a, b = graded_pair(case)
    p0_run = _scalar(case.run, "source_surface_pa", "a source-surface pressure")
    p0_ref = _scalar(case.reference, "source_surface_pa", "a source-surface pressure")
    gain_run = _ratio(float(a.max()), p0_run, "the focal gain")
    gain_ref = _ratio(float(b.max()), p0_ref, "the reference focal gain")
    return {
        "run": gain_run,
        "reference": gain_ref,
        "ratio": _ratio(gain_run, gain_ref, "the focal gain ratio"),
    }


# ---------------------------------------------------------------------------
# Profiles (M-13 to M-16)
# ---------------------------------------------------------------------------


@metric(
    "M-13",
    "Axial profile Pearson r through the realized peak",
    "profiles",
    "a beam axis exists",
    applies=needs("beam_axis"),
)
def m13_axial_pearson(case: ParityCase) -> float:
    a, b = graded_pair(case)
    axis = _beam_axis(case)
    return pearson(
        profile(a, axis, peak_index(a)),
        profile(b, axis, peak_index(b)),
        "the axial profile",
    )


@metric(
    "M-14",
    "Lateral profile Pearson r through the realized peak",
    "profiles",
    "a beam axis exists",
    applies=needs("beam_axis"),
)
def m14_lateral_pearson(case: ParityCase) -> dict[str, float]:
    a, b = graded_pair(case)
    out: dict[str, float] = {}
    for axis in _lateral_axes(case, a):
        out[f"axis_{axis}"] = pearson(
            profile(a, axis, peak_index(a)),
            profile(b, axis, peak_index(b)),
            f"the lateral profile along axis {axis}",
        )
    return out


@metric(
    "M-15",
    "Axial profile relative L2",
    "profiles",
    "a beam axis exists",
    applies=needs("beam_axis"),
    units="-",
)
def m15_axial_relative_l2(case: ParityCase) -> float:
    a, b = graded_pair(case)
    axis = _beam_axis(case)
    return relative_l2(
        profile(a, axis, peak_index(a)),
        profile(b, axis, peak_index(b)),
        "the axial profile",
    )


def _first_sidelobe_db(prof: np.ndarray, i_peak: int, engine: str) -> float:
    """Level of the highest first sidelobe [dB below the peak].

    Walk outward from the peak on each side to the first local minimum and
    then to the first local maximum after it; the larger of the two lobes is
    the answer. A side with no resolved lobe contributes nothing, and when
    neither side has one the metric is not computed rather than pinned to the
    profile's edge.
    """
    peak = float(prof[i_peak])
    if peak <= 0.0:
        raise NotComputed(
            f"the {engine} lateral profile peaks at {peak:.3g}; no sidelobe "
            f"level can be referred to it."
        )
    lobes: list[float] = []
    for direction in (1, -1):
        i = i_peak
        falling = True
        while 0 < i + direction < prof.size - 1:
            i += direction
            if falling:
                if prof[i + direction] > prof[i]:
                    falling = False
            elif prof[i + direction] < prof[i]:
                lobes.append(float(prof[i]))
                break
    if not lobes:
        raise NotApplicable(
            f"no sidelobe is resolved in the {engine} lateral profile: it falls "
            f"monotonically to the edge of the recorded region on both sides."
        )
    return float(20.0 * np.log10(max(lobes) / peak))


@metric(
    "M-16",
    "First sidelobe level difference",
    "profiles",
    "a sidelobe is resolved",
    applies=needs("sidelobe", "beam_axis"),
    units="dB",
)
def m16_sidelobe_difference(case: ParityCase) -> float:
    a, b = graded_pair(case)
    axis = _lateral_axes(case, a)[0]
    run_db = _first_sidelobe_db(
        profile(a, axis, peak_index(a)), peak_index(a)[axis], case.run.engine
    )
    ref_db = _first_sidelobe_db(
        profile(b, axis, peak_index(b)), peak_index(b)[axis], case.reference.engine
    )
    return float(run_db - ref_db)


# ---------------------------------------------------------------------------
# Spectral (M-17 to M-20)
# ---------------------------------------------------------------------------


def _harmonic_ratio_at_peak(case: ParityCase, n: int) -> dict[str, float]:
    a, b = graded_pair(case)
    pk_a, pk_b = peak_index(a), peak_index(b)
    an = _harmonic(case.run, n)[pk_a]
    a1 = _harmonic(case.run, 1)[pk_a]
    bn = _harmonic(case.reference, n)[pk_b]
    b1 = _harmonic(case.reference, 1)[pk_b]
    run = _ratio(float(an), float(a1), f"the {n}f0 ratio")
    ref = _ratio(float(bn), float(b1), f"the reference {n}f0 ratio")
    return {"run": run, "reference": ref, "ratio": _ratio(run, ref, f"the {n}f0 ratio")}


@metric(
    "M-17",
    "Second-harmonic ratio A2/A1 at the peak",
    "spectral",
    "the example is nonlinear",
    applies=needs("nonlinear"),
    units="-",
)
def m17_second_harmonic_ratio(case: ParityCase) -> dict[str, float]:
    return _harmonic_ratio_at_peak(case, 2)


@metric(
    "M-18",
    "Third-harmonic ratio A3/A1 at the peak",
    "spectral",
    "nonlinear and 3f0 is resolved",
    applies=needs("nonlinear", "third_harmonic"),
    units="-",
)
def m18_third_harmonic_ratio(case: ParityCase) -> dict[str, float]:
    return _harmonic_ratio_at_peak(case, 3)


@metric(
    "M-19",
    "Relative L2 of the 2f0 field",
    "spectral",
    "the example is nonlinear",
    applies=needs("nonlinear"),
    units="-",
)
def m19_second_harmonic_field_l2(case: ParityCase) -> float:
    a2 = _harmonic(case.run, 2)
    b2 = _harmonic(case.reference, 2)
    if a2.shape != b2.shape:
        raise NotComputed(
            f"the 2f0 fields are {a2.shape} and {b2.shape}; the two engines did not "
            f"record the same region."
        )
    return relative_l2(a2, b2, "the 2f0 field")


@metric(
    "M-20",
    "Total harmonic distortion at the focal point",
    "spectral",
    "the example is nonlinear",
    applies=needs("nonlinear"),
    units="-",
)
def m20_thd(case: ParityCase) -> dict[str, float]:
    a, b = graded_pair(case)

    def thd(run, pk: tuple[int, ...]) -> float:
        harmonics = sorted(run.harmonics)
        if harmonics[:1] != [1] or len(harmonics) < 2:
            raise NotComputed(
                f"the {run.engine} run recorded harmonics {harmonics}; a total harmonic "
                f"distortion needs the fundamental and at least one harmonic above it."
            )
        a1 = float(_harmonic(run, 1)[pk])
        above = np.array([float(_harmonic(run, n)[pk]) for n in harmonics[1:]])
        return _ratio(float(np.sqrt((above**2).sum())), a1, "the total harmonic distortion")

    run = thd(case.run, peak_index(a))
    ref = thd(case.reference, peak_index(b))
    return {"run": run, "reference": ref, "ratio": _ratio(run, ref, "the THD ratio")}


# ---------------------------------------------------------------------------
# Time domain (M-21 to M-24)
# ---------------------------------------------------------------------------


@metric(
    "M-21",
    "Waveform Pearson r at each named sensor point",
    "time domain",
    "a time series is recorded",
    applies=needs("time_series"),
)
def m21_waveform_pearson(case: ParityCase) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in _shared_series(case):
        sa = _series(case.run, name)
        sb = _series(case.reference, name)
        if sa.value.shape != sb.value.shape:
            raise NotComputed(
                f"the '{name}' series is {sa.value.shape} in {case.run.engine} and "
                f"{sb.value.shape} in {case.reference.engine}; the two were not sampled "
                f"on the same schedule."
            )
        out[name] = pearson(sa.value, sb.value, f"the '{name}' waveform")
    return out


@metric(
    "M-22",
    "Time-of-flight error at the peak",
    "time domain",
    "a time series is recorded",
    applies=needs("time_series"),
    units="s",
)
def m22_time_of_flight_error(case: ParityCase) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in _shared_series(case):
        sa = _series(case.run, name)
        sb = _series(case.reference, name)
        ta = float(sa.t_s[int(np.argmax(sa.value))])
        tb = float(sb.t_s[int(np.argmax(sb.value))])
        out[name] = ta - tb
    return out


@metric(
    "M-23",
    "Peak-positive and peak-negative pressure ratios",
    "time domain",
    "a time series is recorded",
    applies=needs("time_series"),
    units="-",
)
def m23_peak_pressure_ratios(case: ParityCase) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in _shared_series(case):
        sa = _series(case.run, name).value
        sb = _series(case.reference, name).value
        out[f"{name}_positive"] = _ratio(
            float(sa.max()), float(sb.max()), f"the peak-positive value of '{name}'"
        )
        out[f"{name}_negative"] = _ratio(
            float(sa.min()), float(sb.min()), f"the peak-negative value of '{name}'"
        )
    return out


def _rise_time(t: np.ndarray, y: np.ndarray) -> float:
    """10 to 90 percent rise time [s] of the steepest rising edge."""
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo
    if span <= 0.0:
        raise NotComputed("the waveform is flat, so it has no rising edge to time.")
    i_peak = int(np.argmax(y))
    y10, y90 = lo + 0.1 * span, lo + 0.9 * span
    i90 = next((i for i in range(i_peak, -1, -1) if y[i] <= y90), None)
    i10 = next((i for i in range(i_peak, -1, -1) if y[i] <= y10), None)
    if i90 is None or i10 is None:
        raise NotComputed(
            "the recorded waveform never crosses the 10 and 90 percent levels before its "
            "peak, so a rise time would be an extrapolation."
        )
    return float(t[i90] - t[i10])


@metric(
    "M-24",
    "Shock rise-time ratio",
    "time domain",
    "the waveform steepens",
    applies=needs("shock", "time_series"),
    units="-",
)
def m24_rise_time_ratio(case: ParityCase) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in _shared_series(case):
        sa = _series(case.run, name)
        sb = _series(case.reference, name)
        out[name] = _ratio(
            _rise_time(sa.t_s, sa.value),
            _rise_time(sb.t_s, sb.value),
            f"the rise time of '{name}'",
        )
    return out


# ---------------------------------------------------------------------------
# Invariants (M-25 to M-27)
# ---------------------------------------------------------------------------


@metric(
    "M-25",
    "Radiated power through a closed surface, ratio",
    "invariants",
    "a closed surface fits in the domain",
    applies=needs("closed_surface"),
    units="-",
)
def m25_radiated_power_ratio(case: ParityCase) -> float:
    return _ratio(
        _scalar(case.run, "radiated_power_w", "a radiated power"),
        _scalar(case.reference, "radiated_power_w", "a radiated power"),
        "the radiated power",
    )


@metric(
    "M-26",
    "Energy balance residual, against the energy conservation gate",
    "invariants",
    "the loss is zero or integrable",
    applies=needs("integrable_loss", "closed_surface"),
    units="-",
)
def m26_energy_balance_residual(case: ParityCase) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, run in (("run", case.run), ("reference", case.reference)):
        radiated = _scalar(run, "radiated_power_w", "a radiated power")
        driven = _scalar(run, "source_power_w", "a source power")
        out[label] = _ratio(radiated - driven, driven, "the energy balance residual")
    return out


@metric(
    "M-27",
    "Absorbed power integral ratio",
    "invariants",
    "the medium absorbs",
    applies=needs("absorbing"),
    units="-",
)
def m27_absorbed_power_ratio(case: ParityCase) -> float:
    return _ratio(
        _scalar(case.run, "absorbed_power_w", "an absorbed power integral"),
        _scalar(case.reference, "absorbed_power_w", "an absorbed power integral"),
        "the absorbed power integral",
    )


# ---------------------------------------------------------------------------
# Numerical (M-28) and cost (M-29, M-30)
# ---------------------------------------------------------------------------


@metric(
    "M-28",
    "Fitted convergence order across the example's own resolutions",
    "numerical",
    "the example varies resolution",
    applies=needs("varies_resolution"),
    units="-",
)
def m28_convergence_order(case: ParityCase) -> dict[str, float]:
    sweep = case.run.resolution_sweep
    if not sweep or len(sweep) < 3:
        raise NotComputed(
            f"the {case.run.engine} run carries {0 if not sweep else len(sweep)} "
            f"resolution(s); a fitted order needs at least three."
        )
    dx = np.array([s[0] for s in sweep], dtype=np.float64)
    err = np.array([s[1] for s in sweep], dtype=np.float64)
    if (err <= 0).any():
        raise NotComputed(
            "one of the sweep's errors is zero or negative, so the log-log fit that gives "
            "the order has no slope."
        )
    slope, _ = np.polyfit(np.log(dx), np.log(err), 1)
    return {"order": float(slope), "resolutions": float(len(sweep))}


@metric(
    "M-29",
    "Wall time ratio at matched accuracy",
    "cost",
    "one machine, one stated load regime",
    applies=needs_engine_reference(),
    units="-",
)
def m29_wall_time_ratio(case: ParityCase) -> float:
    regime = case.run.provenance.get("load_regime")
    if not regime:
        raise NotComputed(
            "this run carries no load regime, and a timing taken while other work shared "
            "the machine is not a measurement. Re-run with the load regime stamped."
        )
    return _ratio(
        _scalar(case.run, "wall_time_s", "a wall time"),
        _scalar(case.reference, "wall_time_s", "a wall time"),
        "the wall time",
    )


@metric(
    "M-30",
    "Peak device memory ratio",
    "cost",
    "one machine, one stated load regime",
    applies=needs_engine_reference(),
    units="-",
)
def m30_peak_memory_ratio(case: ParityCase) -> float:
    regime = case.run.provenance.get("load_regime")
    if not regime:
        raise NotComputed(
            "this run carries no load regime, so a peak memory taken beside unknown other "
            "work is not a measurement. Re-run with the load regime stamped."
        )
    return _ratio(
        _scalar(case.run, "peak_device_bytes", "a peak device memory"),
        _scalar(case.reference, "peak_device_bytes", "a peak device memory"),
        "the peak device memory",
    )
