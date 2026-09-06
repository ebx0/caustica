"""numpy and cupy have to compute the same field, on today's engine.

The parity number the README quotes was measured on an A100 in August 2026,
and the k-space engine has changed since: the collocated first derivative now
zeroes the Nyquist bin (a fix whose old operator diverged on cupy and not on
numpy), the settle loop stops on the harmonics rather than a fixed count, and
absorption and the sponge are fused into one damping volume. A number from
before those changes cannot certify the port today, and re-running the whole
GPU gate suite needs a rented device.

So this file runs the gate's own mini 3-D parity scenario, the same job on
both backends in one process with no file round trip in between, and holds it
to the same limit the suite gates on. It skips without a CUDA device, so a CPU
checkout is unaffected; on a machine with a GPU it is seconds.
"""

import pytest

from caustica.core.backend import cupy_available
from caustica.validation.gpu_gates import PARITY_TOL, parity_job, run_parity

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not cupy_available(), reason="needs a CUDA device"),
]


@pytest.fixture(scope="module")
def parity(tmp_path_factory):
    """One mini 3-D job solved on numpy and on cupy, compared in memory."""
    outdir = tmp_path_factory.mktemp("parity")
    return run_parity(outdir, job=parity_job(side=40))


@pytest.mark.parametrize("field", ["phasor", "p_max"])
@pytest.mark.parametrize("norm", ["rel_l2", "rel_linf"])
def test_the_two_backends_agree_on_the_whole_field(parity, field, norm):
    measured = parity[field][norm]
    assert measured < PARITY_TOL, (
        f"numpy vs cupy {field} {norm} = {measured:.3e}, limit {PARITY_TOL:.3e}"
    )


def test_parity_is_measured_before_any_file_round_trip(parity):
    """float16 storage is one ULP at 4.9e-4, which would swamp a 1e-5 limit."""
    assert "no result.h5 round trip" in parity["measured_on"]
    stored = parity["stored_float16_reference"]
    assert stored["phasor"]["rel_linf"] >= parity["phasor"]["rel_linf"]
