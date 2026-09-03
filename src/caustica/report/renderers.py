"""Report renderers: the plugin seam for turning a run folder into a report.

``caustica report <out-dir>`` does not know how to draw anything. It resolves
a *renderer* by name and calls it; caustica's own matplotlib renderer is
registered here under the name ``"matplotlib"`` and is reached through the
same door a third party's renderer would be (no private path).

A renderer is any callable::

    def render(outdir: Path, *, preview_only: bool = False) -> Path: ...

``outdir`` is a runner output folder (``result.h5`` and/or ``preview.npz``,
plus ``metrics.json`` / ``run_meta.json`` when the runner wrote them);
``preview_only`` asks for the quick look even when the full field is there.
The return value is the path a reader should open. Renderers should raise a
plain exception with a readable message when the folder holds nothing they
can render — the CLI prints it and exits 2.

Import discipline: this module is numpy-free and matplotlib-free, so
``caustica.report`` can list the available renderers — and the runner can
write previews — on a machine with neither matplotlib nor h5py installed.
The core renderer's own imports happen inside its body.
"""

from __future__ import annotations

from pathlib import Path

from caustica.registry import REPORT_RENDERER_GROUP, FactoryRegistry

#: The renderer used when nobody asks for one.
DEFAULT_RENDERER = "matplotlib"

#: name -> renderer callable.
report_renderers: FactoryRegistry = FactoryRegistry(
    "report renderer", REPORT_RENDERER_GROUP, plural="report renderers"
)


@report_renderers.register(DEFAULT_RENDERER)
def _matplotlib_renderer(outdir: Path, *, preview_only: bool = False) -> Path:
    """caustica's own renderer: REPORT.md + index.html + PNG figures.

        A thin registered wrapper so the matplotlib import stays inside the call
    : merely *listing* the renderers must not drag matplotlib in.
    """
    from caustica.report.run_report import report_out_dir  # noqa: PLC0415 (matplotlib lazy)

    return report_out_dir(outdir, preview_only=preview_only)


def render_report(
    outdir: str | Path,
    *,
    preview_only: bool = False,
    renderer: str = DEFAULT_RENDERER,
) -> Path:
    """Render ``outdir`` with the named renderer; returns the path to open.

    Raises :class:`caustica.registry.UnknownPluginError` (a ``KeyError``)
    listing the registered names when ``renderer`` is not one of them.
    """
    render = report_renderers.get(renderer)
    return Path(render(Path(outdir), preview_only=preview_only))
