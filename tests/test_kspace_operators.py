"""k-space operator gates, with the 256^3 divergence pinned as a CPU test.

The defect these tests guard used to be invisible on the CPU, which is why it
survived three GPU sessions: the array the engine handed to ``irfftn`` was not
a legal half-spectrum, numpy's pocketfft quietly symmetrized it and returned
something sane, and cuFFT -- whose C2R contract says the input MUST be
Hermitian -- resolved the same ambiguity differently at 256^3 and diverged to
NaN by the second period.

So the gate here is deliberately NOT "does the CPU answer look right". It is
the CONTRACT itself: the input array is measured against the Hermitian
symmetry ``irfftn`` requires, which is a property of the input alone and
therefore visible on any machine, and two legal-but-different C2R readings are
compared against each other. Both are exact zeros once the operator is right.
"""

import numpy as np
import pytest

from caustica.solvers.kspace import operators as ops

DX = 3e-4
DT = 1e-7
C_REF = 1500.0


def _factors(shape, *, dt=DT, zero_nyquist=True):
    ks = ops.k_vectors(shape, DX, np)
    kappa = ops.kappa_sinc(ks, c_ref=C_REF, dt=dt, xp=np)
    if zero_nyquist:
        return ops.spectral_derivative_factors(ks, kappa, shape, np), ks, kappa
    # The pre-fix operator, kept here so the gates below cannot go vacuous:
    # if the contract were satisfied by construction these tests would pass
    # against anything.
    return [(1j * k * kappa).astype(np.complex64) for k in ks], ks, kappa


def _mirror(plane):
    """``plane[-j]`` for every leading index, i.e. the j -> -j partner."""
    if plane.ndim == 0:
        return plane
    return plane[np.ix_(*[(-np.arange(n)) % n for n in plane.shape])]


def _self_paired_planes(n_last):
    """Last-axis bins that pair with THEMSELVES, so must be conjugate-even."""
    return [0] + ([n_last // 2] if n_last % 2 == 0 else [])


def hermitian_defect(x, shape):
    """How far ``x`` is from being a legal ``irfftn`` input, relative to |x|."""
    scale = float(np.abs(x).max()) or 1.0
    worst = 0.0
    for j in _self_paired_planes(shape[-1]):
        pl = np.asarray(x[..., j])
        worst = max(worst, float(np.abs(pl - np.conj(_mirror(pl))).max()))
    return worst / scale


def round_trip_loss(x, shape):
    """Fraction of ``x`` that ``irfftn`` cannot carry, relative to |x|.

    ``rfftn(irfftn(X)) == X`` holds exactly for a legal half-spectrum and
    cannot hold otherwise: the transform pair projects onto the Hermitian
    subspace, and what the projection drops is precisely the part no C2R
    contract defines. numpy discards it; cuFFT folds some of it back into the
    real output. Whichever a backend picks, handing it something with a
    non-zero loss here is asking two libraries to agree on undefined
    behaviour -- which at 256^3 they did not.
    """
    axes = tuple(range(len(shape)))
    back = np.fft.rfftn(np.fft.irfftn(x, s=shape, axes=axes), axes=axes)
    return float(np.abs(back - x).max()) / (float(np.abs(x).max()) or 1.0)


def _random_field(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


# Cubic-even is the failing rung's shape class; the two mixed shapes are the
# U1b matrix's discriminators (256 on the beam axis alone vs on the transverse
# axes alone) shrunk to a size a test suite can afford; odd has no Nyquist bin
# at all and must therefore be untouched.
SHAPES = [(8, 8, 8), (7, 7, 7), (8, 8, 6), (6, 6, 8), (9, 8, 7), (10,), (8, 6)]


@pytest.mark.parametrize("shape", SHAPES)
def test_derivative_factors_drop_the_nyquist_on_even_axes(shape):
    deriv, _, _ = _factors(shape)
    for ax, n in enumerate(shape):
        idx = [0] * len(shape)
        if n % 2 == 0:
            idx[ax] = n // 2
            assert deriv[ax][tuple(idx)] == 0.0, f"axis {ax} kept its Nyquist bin"
        # every other wavenumber on that axis is still carried
        idx[ax] = 1 if n > 2 else 0
        assert deriv[ax][tuple(idx)] != 0.0 or idx[ax] == 0


@pytest.mark.parametrize("shape", SHAPES)
def test_derivative_factors_are_exactly_conjugate_symmetric(shape):
    """The operator alone, with no field in it, carries no defect at all."""
    deriv, _, _ = _factors(shape)
    for ax in range(len(shape)):
        assert hermitian_defect(deriv[ax], shape) == 0.0


# ``rfftn`` of a real field is only Hermitian to its own float roundoff --
# pocketfft's non-power-of-two radices are not bit-symmetric -- so the engine's
# actual input inherits a ~1e-8 floor that belongs to the transform, not to the
# operator. Six orders below the ~1.5 the broken operator produced.
ROUNDOFF = 1e-6


@pytest.mark.parametrize("shape", SHAPES)
def test_gradient_spectrum_honors_the_c2r_hermitian_contract(shape):
    """The gate the 256^3 rung needed: measured on the INPUT, not the output."""
    deriv, _, _ = _factors(shape)
    pk = np.fft.rfftn(_random_field(shape))
    for ax in range(len(shape)):
        assert hermitian_defect(deriv[ax] * pk, shape) < ROUNDOFF


@pytest.mark.parametrize("shape", SHAPES)
def test_divergence_spectrum_honors_the_contract_too(shape):
    """``sum_i deriv_i * rfftn(u_i)`` is the engine's other irfftn input."""
    deriv, _, _ = _factors(shape)
    acc = None
    for ax in range(len(shape)):
        term = deriv[ax] * np.fft.rfftn(_random_field(shape, seed=10 + ax))
        acc = term if acc is None else acc + term
    assert hermitian_defect(acc, shape) < ROUNDOFF


@pytest.mark.parametrize("shape", [(8, 8, 8), (8, 8, 6), (6, 6, 8)])
def test_the_contract_was_actually_broken_before(shape):
    """Without the fix the input violates the contract by ~its own magnitude.

    Not a historical note: it is what keeps the two tests above from passing
    for free. A purely imaginary factor sitting on a bin that has to be
    conjugate-even is off by twice itself, hence a defect of order 1.
    """
    deriv, _, _ = _factors(shape, zero_nyquist=False)
    pk = np.fft.rfftn(_random_field(shape))
    worst = max(hermitian_defect(deriv[ax] * pk, shape) for ax in range(len(shape)))
    assert worst > 0.5


@pytest.mark.parametrize("shape", [(8, 8, 8), (8, 8, 6), (6, 6, 8)])
def test_the_transform_carries_everything_we_hand_it(shape):
    """The CPU-side stand-in for "cuFFT and pocketfft disagreed at 256^3".

    Before the fix a fifth of the gradient spectrum fell through ``irfftn``
    without a contract saying where it went.
    """
    pk = np.fft.rfftn(_random_field(shape))
    last = len(shape) - 1

    broken, _, _ = _factors(shape, zero_nyquist=False)
    assert round_trip_loss(broken[last] * pk, shape) > 0.1

    fixed, _, _ = _factors(shape)
    assert round_trip_loss(fixed[last] * pk, shape) < ROUNDOFF


def test_kappa_keeps_the_true_nyquist():
    """The dispersion correction is an even function of |k| -- do not touch it.

    kappa is not an odd derivative and has no Hermitian problem; zeroing its
    Nyquist would change the physics instead of repairing a contract.
    """
    shape = (8, 8, 8)
    _, ks, kappa = _factors(shape)
    k_nyq = np.pi / DX
    assert max(float(np.abs(k).max()) for k in ks) == pytest.approx(k_nyq, rel=1e-6)
    expected = float(np.sinc(C_REF * np.sqrt(3.0) * k_nyq * DT / (2.0 * np.pi)))
    assert float(kappa.min()) == pytest.approx(expected, rel=1e-6)


def test_k_vectors_are_not_mutated_by_the_factors():
    """Callers report ``max|k|`` from ``ks`` after building the factors."""
    shape = (8, 8, 8)
    ks = ops.k_vectors(shape, DX, np)
    before = [k.copy() for k in ks]
    kappa = ops.kappa_sinc(ks, c_ref=C_REF, dt=DT, xp=np)
    ops.spectral_derivative_factors(ks, kappa, shape, np)
    for k, ref in zip(ks, before, strict=True):
        np.testing.assert_array_equal(k, ref)


def test_resolved_modes_still_differentiate_exactly():
    """Dropping the Nyquist must cost nothing on modes the grid resolves."""
    shape = (32, 32, 32)
    # dt -> 0 makes kappa 1, isolating the derivative from the k-space
    # dispersion correction the factors also carry.
    deriv, _, _ = _factors(shape, dt=1e-14)
    x = np.arange(shape[2]) * DX
    m = 3  # well below the 16-cycle Nyquist
    kx = 2.0 * np.pi * m / (shape[2] * DX)
    p = np.broadcast_to(np.sin(kx * x), shape).astype(np.float32).copy()
    grad = np.fft.irfftn(deriv[2] * np.fft.rfftn(p), shape)
    analytic = np.broadcast_to(kx * np.cos(kx * x), shape)
    assert np.abs(grad - analytic).max() / kx < 1e-4
