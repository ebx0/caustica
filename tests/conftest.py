"""Fixtures shared by the whole suite.

The one thing worth centralising here is the *absence* of a GPU.

Large parts of this suite were written on a machine that had none, so they
assert the CPU path head-on: the slow-CPU gate refuses, ``auto`` resolves to
numpy, ``env_report`` says ``resolved_backend == "numpy"``. Every one of
those assertions is about an environment, not about the library, and the
first time the suite ran on a CUDA box thirteen of them failed — not because
anything regressed but because ``auto`` had picked cupy and the gate they
were testing never fired.

Skipping them on a GPU machine would be the cheap repair and the wrong one:
it drops the coverage exactly where the developer's machine has stopped
being the typical one. :func:`no_gpu` instead makes the availability probe
answer "none", so the tests run the path they describe on either kind of
machine. ``_CUPY_STATE`` is the single place the answer is cached, and both
``get_backend("auto")`` and ``caustica.env`` read it through
``cupy_available()``, so patching it is enough.
"""

from __future__ import annotations

import pytest


def force_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``cupy_available()`` report no device for this test."""
    from caustica.core import backend as B

    monkeypatch.setitem(B._CUPY_STATE, "checked", True)
    monkeypatch.setitem(B._CUPY_STATE, "available", False)
    monkeypatch.setitem(B._CUPY_STATE, "module", None)


@pytest.fixture
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run this test as though the machine had no CUDA device.

    For tests whose subject *is* the CPU environment. A test that wants a
    real GPU asks for one with ``@pytest.mark.skipif(not cupy_available())``
    instead; the two are complements, not alternatives.
    """
    force_no_gpu(monkeypatch)
