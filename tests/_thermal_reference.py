"""An independent Pennes solver, assembled and implicit — TEST-ONLY on purpose.

The criterion is an *independent-implementation* cross-check under 5%. The
milestone originally named MATLAB k-Wave's ``kWaveDiffusion``; it does not
exist in ``k-wave-python`` 0.6.2 (verified in the installed environment), so
the ratified replacement is this: a second solver that shares as little
machinery as possible with :class:`caustica.thermal.PennesSolver` and lives
only in the test tree.

What is actually independent here
---------------------------------
================  ==============================  ============================
                  ``caustica.thermal.PennesSolver``  this module
================  ==============================  ============================
time integrator   forward Euler (explicit)        backward Euler (implicit)
spatial operator  7-point stencil applied in      one assembled
                  place, per axis, on the backend ``scipy.sparse`` matrix
linear algebra    none — the step IS the stencil  SuperLU factorisation, reused
                                                  across steps
precision         float32, state carried as a     float64 throughout, absolute
                  rise above a reference          temperatures
stability         conditional (``dt`` refused     unconditional
                  above a Gershgorin bound)
================  ==============================  ============================

The two paths therefore agree only if the PHYSICS agrees; a transcription
error in either stencil, a factor-of-two in the perfusion sink, or a wrong
sign on a boundary flux shows up as disagreement.

One convention is deliberately SHARED: the face conductivity is the harmonic
mean ``2 k_i k_j / (k_i + k_j)``, exactly as ``PennesSolver`` builds it. That
is on purpose. A reference that discretised the conductivity jump differently
(arithmetic mean, say) would disagree with the library at a two-layer
interface for a reason that has nothing to do with the thing under test — the
cross-check is about the TIME INTEGRATOR and the assembled operator, not about
re-litigating which face average is right (that question is already settled
against a closed form in ``tests/test_thermal.py``'s two-layer gate).

Boundaries are insulated (zero flux), the solver's default. Nothing here is
importable from ``caustica``: shipping a second solver would create a second
answer for users to choose between, and the library's answer is the explicit
one.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu

from caustica.thermal.pennes import ARTERIAL_TEMPERATURE_C, BLOOD_DENSITY, BLOOD_SPECIFIC_HEAT


def _ax(arr: np.ndarray, axis: int, sl: slice) -> np.ndarray:
    idx = [slice(None)] * arr.ndim
    idx[axis] = sl
    return arr[tuple(idx)]


def assemble(
    medium,
    *,
    blood_density: float = BLOOD_DENSITY,
    blood_specific_heat: float = BLOOD_SPECIFIC_HEAT,
    arterial_temperature_c: float = ARTERIAL_TEMPERATURE_C,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """The Pennes operator as a sparse matrix.

    Returns ``(L, w_rho_c, rho_c)`` with ``L`` [W/m^3/K] the assembled
    conduction-plus-perfusion operator, so that the semi-discrete system is::

        rho_c * dT/dt = L @ T + Q + w_rho_c * T_a

    ``L`` is symmetric negative semi-definite (it is a weighted graph
    Laplacian minus the perfusion diagonal), which is what lets the backward
    Euler matrix be factorised once and reused.
    """
    k = np.asarray(medium.k, dtype=np.float64)
    shape = k.shape
    n = k.size
    ids = np.arange(n).reshape(shape)
    inv_dx2 = 1.0 / (float(medium.dx) ** 2)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    diag = np.zeros(n, dtype=np.float64)
    for axis in range(k.ndim):
        lo = _ax(k, axis, slice(None, -1))
        hi = _ax(k, axis, slice(1, None))
        # Harmonic mean = two half-cells of conductor in SERIES. Shared with
        # the library on purpose; see the module docstring.
        face = (2.0 * lo * hi / (lo + hi) * inv_dx2).ravel()
        i = _ax(ids, axis, slice(None, -1)).ravel()
        j = _ax(ids, axis, slice(1, None)).ravel()
        rows += [i, j]
        cols += [j, i]
        vals += [face, face]
        # i and j are each free of repeats, so plain fancy-index subtraction
        # accumulates correctly (no np.add.at needed).
        diag[i] -= face
        diag[j] -= face

    w_rho_c = np.asarray(medium.perfusion, dtype=np.float64).ravel() * (
        blood_density * blood_specific_heat
    )
    diag -= w_rho_c
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)

    operator = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    ).tocsr()
    rho_c = np.asarray(medium.volumetric_heat_capacity, dtype=np.float64).ravel()
    return operator, w_rho_c, rho_c


def _source_vector(q, shape) -> np.ndarray:
    if q is None:
        return np.zeros(int(np.prod(shape)), dtype=np.float64)
    arr = np.asarray(q, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(shape, float(arr))
    if arr.shape != tuple(shape):
        raise ValueError(f"Q has shape {arr.shape}, expected {tuple(shape)}")
    return arr.ravel()


def backward_euler(
    temperature0,
    q,
    medium,
    dt: float,
    n_steps: int,
    *,
    record_every: int | None = None,
    arterial_temperature_c: float = ARTERIAL_TEMPERATURE_C,
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """Integrate ``n_steps`` implicit steps; returns ``(T_final, samples, times)``.

    ``(rho_c/dt - L) T^{n+1} = (rho_c/dt) T^n + Q + w_rho_c T_a`` — one
    SuperLU factorisation, reused for every step because ``dt`` is fixed.
    Unconditionally stable: no stability bound is consulted anywhere, which
    is precisely what makes it an independent check on a scheme that has one.
    """
    shape = tuple(medium.shape)
    operator, w_rho_c, rho_c = assemble(medium, arterial_temperature_c=arterial_temperature_c)
    b = _source_vector(q, shape) + w_rho_c * float(arterial_temperature_c)
    mass = rho_c / float(dt)
    lu = splu((sparse.diags(mass) - operator).tocsc())

    t = np.asarray(temperature0, dtype=np.float64).ravel().copy()
    samples: list[np.ndarray] = []
    times: list[float] = []
    if record_every is not None:
        samples.append(t.reshape(shape).copy())
        times.append(0.0)
    for step in range(1, int(n_steps) + 1):
        t = lu.solve(mass * t + b)
        if record_every is not None and step % record_every == 0:
            samples.append(t.reshape(shape).copy())
            times.append(step * float(dt))
    return t.reshape(shape), samples, times


def steady_state(
    q, medium, *, arterial_temperature_c: float = ARTERIAL_TEMPERATURE_C
) -> np.ndarray:
    """Solve ``-L T = Q + w_rho_c T_a`` directly — the t -> infinity answer.

    No time stepping at all, so it is the sharpest independent statement
    available about the steady state: the explicit solver can only approach
    it, one conditionally-stable step at a time.

    Requires perfusion somewhere (otherwise the insulated operator is
    singular: a floating temperature with no sink has no unique steady
    state, only a ramp).
    """
    operator, w_rho_c, _ = assemble(medium, arterial_temperature_c=arterial_temperature_c)
    if not (w_rho_c > 0).any():
        raise ValueError(
            "steady_state needs perfusion somewhere: with insulated walls and no sink "
            "the operator is singular and the temperature just ramps."
        )
    b = _source_vector(q, tuple(medium.shape)) + w_rho_c * float(arterial_temperature_c)
    lu = splu((-operator).tocsc())
    return lu.solve(b).reshape(tuple(medium.shape))
