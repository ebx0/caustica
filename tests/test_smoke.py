"""M0 gate: the package installs, imports and exposes its public surface."""

import hifusim


def test_import_and_version():
    assert isinstance(hifusim.__version__, str)
    assert hifusim.__version__.count(".") >= 2


def test_public_surface():
    for name in ("Grid", "PMLSpec", "Medium", "Material", "MaterialDB", "get_backend"):
        assert hasattr(hifusim, name), f"missing public export: {name}"
