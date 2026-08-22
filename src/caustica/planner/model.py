"""Static VRAM and wall-time models for the k-space CW engine (M8 planner).

The memory model is a literal inventory of what ``run_cw_kspace_pstd``
allocates — ``engine.py`` is the ground truth, and ``tests/test_planner.py``
pins this inventory so any new persistent buffer added to the engine breaks
a test here instead of silently invalidating the planner. On top of the
inventory sit an FFT-workspace share and a flat allocator margin
(fragmentation + cuFFT plan cache), per the M8 contract.

The "db" time model is deliberately coarse (datasheet numbers, expected
within ~2x): its job is device *comparison* before any hardware is touched.
The accuracy path is calibration (``calibrate.py``), which fits the same
``t_step = a*N*log2(N) + b*N`` form to ~20 measured steps on the device —
the M8 gate (±25%) applies to the calibrated path only.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2, prod

from caustica.solvers.kspace import operators as ops

#: Flat margin over the buffer inventory: allocator fragmentation, cuFFT plan
#: cache, and small unlisted temporaries (M8 contract: +15%).
ALLOCATOR_MARGIN = 1.15

#: Bytes reserved on-device before any caustica allocation: CUDA context,
#: cupy runtime, and the kernel module cache (~1 GB empirically on Colab).
CONTEXT_OVERHEAD_BYTES = int(1.2e9)

#: Fraction of datasheet fp32 TFLOPS that large real-to-complex cuFFT
#: transforms actually sustain (empirically ~10%; the calibrated path
#: replaces this entirely).
FFT_FLOP_EFF = 0.10

#: Fraction of datasheet memory bandwidth sustained by fused elementwise
#: kernels (reads+writes of large float32 volumes).
BW_EFF = 0.75

#: One-time GPU warmup per solve [s] — cuFFT plan creation for the engine's
#: padded shape, kernel compilation for its specific fusions, and the first
#: device allocations. It is NOT per-step work, so a run that pays it is not
#: "slower per step"; a model without it is simply missing a constant.
#:
#: The number comes from the first real GPU session (A100-SXM4-40GB, Colab,
#: 2026-08-22): a 104-step solve took 2.77 s while the planner's own probe
#: measured 1.03 ms/step on the same shape in the same process — 2.66 s that
#: no per-step model can explain, and which the same job on the CPU (0.96x
#: predicted) does not pay. Rounded up to 3.0 as a single-device default;
#: :func:`caustica.planner.calibrate` overwrites it with a measured value on
#: the target device, and ``caustica.validation`` writes back what real runs
#: actually paid.
GPU_WARMUP_S = 3.0

_F32 = 4
_C64 = 8


@dataclass(frozen=True)
class MemoryModel:
    """VRAM inventory for one CW solve on the padded FFT domain."""

    breakdown: dict[str, int]  # component -> bytes (pre-margin)
    total_bytes: int  # sum(breakdown) * ALLOCATOR_MARGIN
    padded_shape: tuple[int, ...]


def fft_sizes(active_shape: tuple[int, ...]) -> tuple[tuple[int, ...], int, int]:
    """Padded shape and element counts ``(padded, P, R)``.

    ``P`` is real-volume elements, ``R`` is rfft-spectrum elements
    (last axis halved: ``n//2 + 1``).
    """
    padded = ops.pad_shape(tuple(active_shape))
    p_elems = prod(padded)
    r_elems = prod(padded[:-1]) * (padded[-1] // 2 + 1)
    return padded, p_elems, r_elems


def kspace_memory(
    active_shape: tuple[int, ...],
    nonlinear: bool,
    n_harmonics: int,
    rec_elems: int | None = None,
) -> MemoryModel:
    """Byte-level inventory of ``run_cw_kspace_pstd`` for ``active_shape``.

    Mirrors engine.py line by line: state (p + nd velocity components),
    property maps (dt_over_rho, rhoc2_dt, absorb, + beta2_dt when
    nonlinear), the sponge volume, per-axis spectral factors, record
    buffers (one complex64 per harmonic + float32 pmax over the record
    region), the step-loop temporaries that coexist at the peak, and an
    FFT workspace share (in+out complex copies).
    """
    padded, p_elems, r_elems = fft_sizes(active_shape)
    nd = len(padded)
    rec = int(rec_elems) if rec_elems is not None else prod(active_shape)

    breakdown = {
        "state (p + u)": (1 + nd) * _F32 * p_elems,
        "property maps": (3 + (1 if nonlinear else 0)) * _F32 * p_elems,
        "sponge": _F32 * p_elems,
        "spectral factors (i*k*kappa)": nd * _C64 * r_elems,
        "record buffers": rec * (_C64 * n_harmonics + _F32),
        "step temporaries": 3 * _C64 * r_elems + (2 + (2 if nonlinear else 0)) * _F32 * p_elems,
        "fft workspace": 2 * _C64 * r_elems,
    }
    total = ceil(sum(breakdown.values()) * ALLOCATOR_MARGIN)
    return MemoryModel(breakdown=breakdown, total_bytes=total, padded_shape=padded)


def n_ffts_per_step(nd: int) -> int:
    """FFT invocations per engine step: rfftn(p), nd gradient irfftn,
    nd rfftn(u_i), one divergence irfftn."""
    return 2 + 2 * nd


def pointwise_bytes_per_elem(nd: int, nonlinear: bool) -> float:
    """Approximate bytes moved per padded-volume element per step by the
    elementwise (non-FFT) work.

    Counted from engine.step(): per velocity axis ~10 float32 passes
    (update, absorb, sponge), ~10 for the pressure update (+6 nonlinear),
    and the k-space multiplies (~6 complex passes per axis on the
    half-spectrum, R/P ~= 0.5).
    """
    f32_passes = 10.0 * (nd + 1) + (6.0 if nonlinear else 0.0)
    kspace_bytes = 6.0 * nd * _C64 * 0.5
    return _F32 * f32_passes + kspace_bytes


def db_time_coeffs(
    fp32_tflops: float, mem_bw_gbs: float, nd: int, nonlinear: bool
) -> tuple[float, float]:
    """Datasheet-derived ``(a, b)`` for ``t_step = a*P*log2(P) + b*P``.

    ``a`` carries the FFT flop cost (~2.5*N*log2(N) per real transform),
    ``b`` the bandwidth-bound elementwise cost. Coarse by design — see
    module docstring.
    """
    a = 2.5 * n_ffts_per_step(nd) / (fp32_tflops * 1e12 * FFT_FLOP_EFF)
    b = pointwise_bytes_per_elem(nd, nonlinear) / (mem_bw_gbs * 1e9 * BW_EFF)
    return a, b


def step_time(a: float, b: float, p_elems: int) -> float:
    """Evaluate the M8 time model ``t_step = a*P*log2(P) + b*P`` [s]."""
    return a * p_elems * log2(p_elems) + b * p_elems
