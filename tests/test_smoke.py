"""M0 gate: the package installs, imports and exposes its public surface."""

import caustica


def test_import_and_version():
    assert isinstance(caustica.__version__, str)
    assert caustica.__version__.count(".") >= 2


def test_public_surface():
    for name in ("Grid", "PMLSpec", "Medium", "Material", "MaterialDB", "get_backend"):
        assert hasattr(caustica, name), f"missing public export: {name}"
