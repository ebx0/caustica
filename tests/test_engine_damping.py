"""The engine's damping volume: one buffer, one multiply per field per step.

Absorption (``e^{-alpha c dt}``) and the sponge are both per-voxel
multipliers of the pressure and of every velocity component. The engine
forms their float32 product once at setup and applies that single volume,
so the two factors cost one buffer and one pass instead of two.

What has to stay true, and is measured here:

* the volume the engine actually builds is absorption times the sponge,
  and it is the sponge bit for bit when the medium is lossless;
* both factors are still there -- an absorbing run still decays at the
  configured alpha, and the sponge still swallows the wave at the edge;
* the reassociation the fusion performs is worth a couple of units in the
  last place, which is a property of float32 and is characterised, not a
  guard on the engine.

The alpha and PML tolerances themselves live in
``tests/test_linear_planewave.py``; this file guards the fusion, not the
numbers it must not disturb.
"""

from __future__ import annotations

import numpy as np

import caustica as hs
from caustica.core.pml import sponge_profile_1d
from caustica.materials import water
from caustica.medium import Medium
from caustica.solvers import CWRunSpec, get
from caustica.solvers.kspace import operators as ops
from caustica.solvers.kspace.engine import cw_discretization
from caustica.sources import plane_cw_source

C0 = 1500.0
F0 = 1.0e6
DX = C0 / (F0 * 4.0)  # 4 points per wavelength, the 1-D validation setup


def test_fused_damping_is_within_two_ulp_of_the_two_pass_form():
    """``u * (sponge * absorb)`` vs ``(u * absorb) * sponge``, float32.

    Both factors are in [0, 1] and the operands are the engine's own: a
    real sponge volume and a real ``exp(-alpha c dt)`` map. Each form
    rounds twice, so the reassociation can separate them by two units in
    the last place; measured over this sample the worst element moves by
    1.73e-07, which is 1.46 * 2**-23.

    This characterises float32 arithmetic and calls no engine code. What
    the engine actually builds and applies is graded by the two tests
    below.
    """
    shape = (24, 20, 16)
    rng = np.random.default_rng(20260904)
    sponge = ops.sponge_volume(shape, 6, 2.0, np)
    alpha_c_dt = rng.uniform(0.0, 0.05, size=shape)
    absorb = np.exp(-alpha_c_dt).astype(np.float32)
    field = rng.uniform(-1.0e5, 1.0e5, size=shape).astype(np.float32)

    two_pass = (field * absorb) * sponge
    damp = sponge * absorb
    fused = field * damp

    scale = np.maximum(np.abs(two_pass), np.finfo(np.float32).tiny)
    rel = float(np.max(np.abs(fused - two_pass) / scale))
    assert rel <= 2.0**-22, f"fused damping moved the field by {rel:.3e} relative"
    assert damp.dtype == np.float32
    assert float(damp.max()) <= 1.0


def _run_1d(alpha: float, *, pml_vox: int, n: int, min_settle: int, max_settle: int):
    grid = hs.Grid(shape=(n,), dx=DX, pml=hs.PMLSpec(thickness=pml_vox * DX))
    medium = Medium.homogeneous((n,), water(c=C0, alpha_np_m=alpha))
    source = plane_cw_source(grid, f0=F0, amplitude=1.0e5, position_vox=40)
    spec = CWRunSpec(
        min_settle_periods=min_settle,
        max_settle_periods=max_settle,
        n_record_periods=2,
    )
    res = get("linear")().run(grid, medium, source, spec, backend="numpy")
    return grid, medium, spec, res


def test_the_fused_volume_still_carries_both_factors():
    """One run, both factors graded: interior decay and sponge swallow.

    Dropping the absorption factor would flatten the measured alpha to
    zero; dropping the sponge would leave the wave alive where the PML
    band ends. Either mistake survives the ulp test above, so it is
    caught here instead.
    """
    alpha = 30.0  # Np/m, e^{-1.35} over the measurement window
    n, pml_vox = 288, 32
    grid, _, _, res = _run_1d(alpha, pml_vox=pml_vox, n=n, min_settle=45, max_settle=95)
    assert grid.pml_vox == pml_vox

    span = slice(80, 200)
    x = np.arange(*span.indices(n)) * DX
    slope = np.polyfit(x, np.log(res.amp[span]), 1)[0]
    rel_err = abs(-slope - alpha) / alpha
    assert rel_err < 0.01, f"measured alpha off by {rel_err * 100:.2f}%"

    # The sponge profile falls to ~e^{-edge} at the outermost voxel; the
    # field there must fall with it rather than ride out to the boundary.
    edge_amp = float(res.amp[-1])
    interior_amp = float(res.amp[n - pml_vox - 4])
    assert edge_amp < 0.2 * interior_amp, (
        f"edge amplitude {edge_amp:.1f} Pa is not damped against "
        f"{interior_amp:.1f} Pa inside the sponge"
    )
    assert float(sponge_profile_1d(n, pml_vox, 2.0)[-1]) < 0.2


def test_the_engine_builds_damp_as_absorption_times_the_sponge(monkeypatch):
    """The volume the engine applies is the product, graded on a real run.

    ``run_cw_kspace_pstd`` builds its damping volume by multiplying the
    absorption map into the array ``ops.sponge_volume`` returns, in place,
    so holding on to that array holds the fused result. It is compared
    against a freshly built sponge and the ``exp(-alpha c dt)`` implied by
    the discretization helper the engine and the planner share.

    In lossless water absorption is exactly 1.0f everywhere, so the fused
    volume must be the sponge bit for bit. That is why the 3-D validation
    bowl does not move at all under this fusion.

    The in-place build is pinned on purpose and not only for convenience:
    an out-of-place product would let both factors coexist with the result
    and raise the setup peak by one float32 volume, which is the memory
    the fusion was meant to save. If this test starts failing on the ratio
    while the engine looks right, that is what changed.
    """
    n, pml_vox = 128, 16
    real_sponge = ops.sponge_volume
    captured: list = []

    def capture(shape, width, edge, xp):
        vol = real_sponge(shape, width, edge, xp)
        captured.append((vol, (shape, width, edge, xp)))
        return vol

    monkeypatch.setattr(ops, "sponge_volume", capture)

    for alpha in (0.0, 30.0):
        captured.clear()
        grid, medium, spec, _ = _run_1d(alpha, pml_vox=pml_vox, n=n, min_settle=2, max_settle=2)
        assert len(captured) == 1, (
            f"the engine built {len(captured)} sponge volumes, expected exactly one"
        )
        damp, args = captured[0]
        fresh = real_sponge(*args)
        assert damp.dtype == np.float32
        assert float(damp.max()) <= 1.0

        _, dt = cw_discretization(grid, medium, spec, F0)
        expected = np.float32(np.exp(-alpha * C0 * dt))
        if alpha == 0.0:
            assert np.array_equal(damp, fresh), "lossless damping is not the sponge itself"
        ratio = damp / fresh
        assert np.allclose(ratio, expected, rtol=1e-6, atol=0.0), (
            f"damping volume is not sponge * exp(-alpha c dt): ratio spans "
            f"{float(ratio.min()):.8f} to {float(ratio.max()):.8f}, expected "
            f"{float(expected):.8f}"
        )
