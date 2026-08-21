"""Turning a class map into per-voxel acoustic properties — with texture.

A label map alone gives a piecewise-constant medium: every fat voxel is
*identical*. Real tissue is not, and a perfectly uniform medium produces no
backscatter at all. Two independent mechanisms add texture here, and they
answer different questions:

**1. pval (measured heterogeneity).** The repository ships a per-voxel scalar
``p in [0, 1]`` saying where that voxel sits between the LOW and HIGH property
bounds of its own tissue class. It is derived from the patient's MRI, so it
is spatially structured the way the real breast is — the lipid-rich core of a
fat lobule really does differ from its edge. Using it costs nothing in
physics honesty: it interpolates inside the class's own literature spread.

**2. Scatterer noise (synthetic).** A pseudo-random field added on top, with a
tunable correlation length, standing in for sub-voxel structure the 0.5 mm
MRI cannot resolve. This is the standard way to give a numerical phantom
speckle. It is NOT measured — it is a modelling choice, so it is off by
default, always seeded, and the seed is recorded in the export.

Which properties should carry the noise? Backscatter comes from ACOUSTIC
IMPEDANCE ``z = rho*c`` and compressibility contrast, so the default perturbs
``rho`` and ``c``. Perturbing ``alpha`` instead only makes the medium lossy in
a blotchy way; perturbing ``beta`` mostly does nothing at diagnostic levels.

Correlation length is specified in MILLIMETRES, not voxels, on purpose: the
same spec must mean the same physical medium after a change of ``dx``,
otherwise a resolution study silently changes the scattering model too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PROPERTY_NAMES = ("alpha", "rho", "c", "beta")


@dataclass
class PropertyVolumes:
    """The four dense volumes a :class:`caustica.medium.Medium` is built from."""

    alpha: np.ndarray
    rho: np.ndarray
    c: np.ndarray
    beta: np.ndarray

    def __post_init__(self) -> None:
        shapes = {n: getattr(self, n).shape for n in PROPERTY_NAMES}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"property volumes disagree in shape: {shapes}")
        for name in PROPERTY_NAMES:
            setattr(self, name, np.ascontiguousarray(getattr(self, name), dtype=np.float32))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.alpha.shape

    def asdict(self) -> dict[str, np.ndarray]:
        return {n: getattr(self, n) for n in PROPERTY_NAMES}

    def copy(self) -> PropertyVolumes:
        return PropertyVolumes(**{n: getattr(self, n).copy() for n in PROPERTY_NAMES})

    def impedance(self) -> np.ndarray:
        """Acoustic impedance ``rho * c`` [Rayl] — what reflections care about."""
        return self.rho * self.c

    def stats(self) -> dict[str, dict[str, float]]:
        out = {}
        for name in PROPERTY_NAMES:
            v = getattr(self, name)
            out[name] = {
                "min": float(v.min()),
                "max": float(v.max()),
                "mean": float(v.mean()),
                "std": float(v.std()),
            }
        return out


def correlated_noise(
    shape: tuple[int, ...],
    correlation_vox: float,
    rng: np.random.Generator,
    dtype=np.float32,
) -> np.ndarray:
    """Zero-mean, unit-variance Gaussian field with a given correlation length.

    ``correlation_vox`` is the Gaussian smoothing sigma in voxels; 0 gives
    white noise (fully uncorrelated between neighbouring voxels). After
    smoothing, the field is renormalized to unit standard deviation —
    smoothing otherwise shrinks the variance by ~``(4*pi*sigma^2)^(-ndim/4)``,
    which would make "3% noise" mean 0.3% as soon as a correlation length was
    set. Renormalizing keeps the knob honest.
    """
    field = rng.standard_normal(shape).astype(dtype, copy=False)
    if correlation_vox > 0:
        from scipy import ndimage

        # 'wrap' keeps the field statistically stationary at the borders;
        # 'reflect' would double-count edge voxels and raise variance there.
        field = ndimage.gaussian_filter(field, sigma=float(correlation_vox), mode="wrap")
    # Demean, THEN normalize — in that order, unconditionally.
    #
    # The mean matters as much as the variance: a nonzero sample mean is not
    # noise, it is a systematic bias applied to every tissue voxel at once
    # ("3% scatterer noise" quietly also making the whole breast 1% denser).
    # Smoothing makes it worse, because only ~(N/sigma)^3 independent samples
    # survive — measured at 10% of the noise amplitude for a 4-voxel
    # correlation length on a 40^3 domain.
    field -= field.mean()
    std = float(field.std())
    if std > 0:
        field /= std
    return field


def apply_pval_interpolation(
    props: PropertyVolumes,
    codes: np.ndarray,
    pval: np.ndarray,
    low: dict[str, np.ndarray],
    high: dict[str, np.ndarray],
) -> PropertyVolumes:
    """Blend each property between its class LOW and HIGH bound using ``p``.

    ``low``/``high`` map a property name to a per-class lookup array indexed
    by class code. Following the repository manual, ``value = p*high +
    (1-p)*low``. Classes whose bounds coincide (skin, muscle, the immersion
    medium — the source data gives them ``p = 0``) come out unchanged, so no
    special-casing is needed.
    """
    if codes.shape != pval.shape:
        raise ValueError(f"codes {codes.shape} and pval {pval.shape} disagree")
    idx = codes.astype(np.intp)
    p = pval.astype(np.float32, copy=False)
    for name in PROPERTY_NAMES:
        lo = low[name][idx]
        hi = high[name][idx]
        setattr(props, name, np.ascontiguousarray(lo + (hi - lo) * p, dtype=np.float32))
    return props


def apply_scatterer_noise(
    props: PropertyVolumes,
    mask: np.ndarray,
    amplitude_pct: float,
    correlation_vox: float,
    properties: tuple[str, ...],
    seed: int,
    coupled: bool = True,
) -> tuple[PropertyVolumes, dict]:
    """Multiply selected properties by ``1 + s*n(x)`` inside ``mask``.

    ``coupled=True`` drives every selected property from ONE noise field, so
    density and sound speed rise and fall together and the medium stays
    physically coherent. ``coupled=False`` draws an independent field per
    property.

    Mind the impedance arithmetic, which is not intuitive: with a shared
    field ``z = rho*c`` scales as ``(1 + s*n)^2``, so ``dz/z ~= 2*s*n``; with
    independent fields ``dz/z ~= s*(n1 + n2)``, i.e. ``s*sqrt(2)``. **Coupled
    therefore gives sqrt(2) times MORE impedance contrast, not less** —
    measured on a uniform medium at 3% noise: coupled 6.00% impedance CV,
    uncoupled 4.25%. Since backscatter comes from impedance, a speckle study
    targeting 2% should ask for ``noise_pct=1`` coupled, not 2 (review
    finding, 2026-08-18 — this docstring previously claimed the opposite).

    Values are clipped so no voxel can go non-positive: a negative density or
    sound speed does not raise here, it produces NaNs a thousand time steps
    into the solve.
    """
    unknown = [p for p in properties if p not in PROPERTY_NAMES]
    if unknown:
        raise ValueError(f"unknown properties {unknown}; pick from {PROPERTY_NAMES}")
    if amplitude_pct <= 0 or not properties:
        return props, {"applied": False}
    if mask.shape != props.shape:
        raise ValueError(f"mask {mask.shape} does not match volumes {props.shape}")

    rng = np.random.default_rng(seed)
    s = float(amplitude_pct) / 100.0
    shared = correlated_noise(props.shape, correlation_vox, rng) if coupled else None
    realized = {}
    has_mask = bool(mask.any())
    for name in properties:
        field = shared
        if field is None:
            field = correlated_noise(props.shape, correlation_vox, rng)
        # Work on the SELECTED voxels only, in place. A full-volume `factor`
        # array plus `v * factor` would hold two extra property-sized buffers
        # per property — 2.4 GB of scratch on a 79 Mvox build, to perturb the
        # 40% of voxels that are tissue.
        v = getattr(props, name)
        if not has_mask:
            realized[name] = 0.0
            continue
        factor = 1.0 + s * field[mask]
        # alpha and beta may legitimately be 0 (lossless / linear water);
        # 0.05 keeps a positive floor without inventing loss where there is none.
        np.clip(factor, 0.05, None, out=factor)
        v[mask] *= factor
        # The std of the FACTOR, not of the resulting property. Measuring the
        # property mixes in the (much larger) contrast BETWEEN tissues: on a
        # two-tissue volume a 3.0% request reported 6.76%, so a user checking
        # their knob would have halved it and delivered 1.5% (review finding,
        # 2026-08-18).
        realized[name] = float(factor.std())
    return props, {
        "applied": True,
        "amplitude_pct": amplitude_pct,
        "correlation_vox": correlation_vox,
        "properties": list(properties),
        "coupled": coupled,
        "seed": seed,
        # std of the multiplicative factor actually applied, per property
        "realized_cv": realized,
    }
