"""Presentation for the progress payload (M10j) — rendering lives OUTSIDE the solver.

The engine emits one dict per period boundary (see
:func:`caustica.solvers.kspace.engine.run_cw_kspace_pstd`); everything that
turns that dict into something a human looks at lives here, so the solver
never grows a display dependency and a GUI can ignore this module entirely.

Two consumers ship:

* a progress line — a ``tqdm`` bar when tqdm happens to be importable AND
  something interactive is watching (a tty, or a notebook kernel), plain
  periodic lines otherwise. **tqdm is never a requirement**: Colab and a bare
  ``pip install caustica`` must both work, so the import is attempted once and
  its absence is not an error. Piped into a log, a rewriting bar is noise, so
  the plain renderer takes over.
* a mid-run preview — ON by default (decision D21): every
  :data:`DEFAULT_PREVIEW_EVERY` periods the payload's lazy ``snapshot`` is
  called ONCE and rendered as a coarse ASCII map of the field through the
  focus. Coarse and text-only on purpose: it answers "is this run going
  somewhere sane?" in any terminal, over ssh, and in a Colab output cell,
  with no plotting library and no image bytes. A notebook that wants a real
  figure passes its own callable — that is what ``progress=<callable>`` is for.

Everything writes to **stderr**, which keeps the stdout contract of
``caustica run`` (plan text, result path) intact for anything parsing it.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, TextIO

import numpy as np

#: Preview cadence in acoustic periods — matches the default checkpoint
#: cadence so the two rhythms of a long run stay in phase.
DEFAULT_PREVIEW_EVERY = 8

#: Accepted ``progress=`` values, quoted verbatim in the error message.
ACCEPTED = ("None (off)", "'auto'", "'plain'", "a callable taking one dict")

_RAMP = " .:-=+*#"
_MAP_COLS = 56
_MAP_ROWS = 16


def _fmt_pressure(pa: float | None) -> str:
    if pa is None:
        return "--"
    if abs(pa) >= 1e5:
        return f"{pa / 1e6:.3f} MPa"
    return f"{pa / 1e3:.1f} kPa"


def _fmt_seconds(s: float | None) -> str:
    if s is None:
        return "--"
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.2f}h"


def _coarsen(arr: np.ndarray, rows: int = _MAP_ROWS, cols: int = _MAP_COLS) -> np.ndarray:
    """Block-MAX downsample |arr| to at most ``rows x cols``.

    Max, not mean: a preview exists to show where the energy went, and a mean
    over a coarse block hides a tight focus behind the water around it.
    """
    a = np.abs(np.asarray(arr, dtype=np.float32))
    if a.ndim == 1:
        a = a[None, :]
    r_step = max(1, -(-a.shape[0] // rows))
    c_step = max(1, -(-a.shape[1] // cols))
    r_at = np.arange(0, a.shape[0], r_step)
    c_at = np.arange(0, a.shape[1], c_step)
    return np.maximum.reduceat(np.maximum.reduceat(a, r_at, axis=0), c_at, axis=1)


def ascii_map(arr: np.ndarray) -> list[str]:
    """A coarse |field| map as text rows (empty when the field is still zero)."""
    small = _coarsen(arr)
    peak = float(small.max())
    if not np.isfinite(peak) or peak <= 0.0:
        return []
    levels = np.clip((small / peak * (len(_RAMP) - 1) + 0.5).astype(np.int32), 0, len(_RAMP) - 1)
    return ["".join(_RAMP[v] for v in row) for row in levels]


class ConsoleProgress:
    """Default ``progress='auto'`` consumer: one line per period + a periodic map.

    Stateless with respect to the solve — it only reads the payload, so
    attaching or dropping it cannot change a single float of the result.
    """

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        preview: bool = True,
        preview_every: int = DEFAULT_PREVIEW_EVERY,
        use_tqdm: bool = True,
        label: str = "",
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.preview = preview
        self.preview_every = max(1, int(preview_every))
        self.label = label
        self._bar: Any = None
        self._tqdm = _load_tqdm() if use_tqdm and _interactive(self.stream) else None
        self._last_preview: int | None = None

    # -- output ----------------------------------------------------------
    def _write(self, text: str) -> None:
        """Print without fighting the bar for the same terminal line."""
        if self._bar is not None and self._tqdm is not None:
            self._tqdm.write(text, file=self.stream)
        else:
            print(text, file=self.stream, flush=True)

    def _line(self, ev: dict) -> str:
        return (
            f"[{ev['stage']:6s}] period {ev['period']}/{ev['periods_expected']}  "
            f"step {ev['step']}/{ev['steps_expected']}  "
            f"peak {_fmt_pressure(ev['peak'])}  "
            f"d={'--' if ev['converge_delta'] is None else format(ev['converge_delta'], '.2e')}  "
            f"eta {_fmt_seconds(ev['eta_s'])}"
        )

    def _render_preview(self, ev: dict) -> None:
        snapshot = ev.get("snapshot")
        if not callable(snapshot):
            return
        rows = ascii_map(snapshot())  # THE one device->host copy of this callback
        if not rows:
            return
        self._write(
            f"  preview @ period {ev['period']} — |p| through the focus "
            f"(beam left to right), peak {_fmt_pressure(ev['peak'])}"
        )
        for row in rows:
            self._write(f"  |{row}|")

    # -- the callback ----------------------------------------------------
    def __call__(self, ev: dict) -> None:
        if self._tqdm is None:
            self._write(self._line(ev))
        else:
            if self._bar is None:
                self._bar = self._tqdm(
                    total=ev["steps_expected"],
                    initial=ev["step"],
                    unit="step",
                    desc=self.label or "solving",
                    file=self.stream,
                    leave=False,
                )
            self._bar.update(max(0, ev["step"] - self._bar.n))
            self._bar.set_postfix_str(
                f"{ev['stage']} p{ev['period']}/{ev['periods_expected']} "
                f"peak {_fmt_pressure(ev['peak'])}",
                refresh=False,
            )
            if ev["stage"] == "record":
                self._write(self._line(ev))

        if self.preview and ev["period"] != self._last_preview:
            if ev["period"] % self.preview_every == 0 or ev["stage"] == "record":
                self._last_preview = ev["period"]
                self._render_preview(ev)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def _interactive(stream: TextIO) -> bool:
    """Is there a human watching this stream redraw itself?

    A tqdm bar rewrites one line; piped into a log file or a CI transcript
    that turns into thousands of carriage returns and the periodic preview
    becomes unreadable. A notebook is interactive even though its stream is
    not a tty, so both are asked.
    """
    if "ipykernel" in sys.modules or "google.colab" in sys.modules:
        return True
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _load_tqdm():
    """``tqdm.auto.tqdm`` if it is installed, else None. Never an error."""
    try:  # pragma: no cover - depends on the environment, both paths are fine
        from tqdm.auto import tqdm

        return tqdm
    except Exception:
        return None


def resolve(
    progress: str | Callable[[dict], None] | None,
    **kwargs: Any,
) -> Callable[[dict], None] | None:
    """Turn a ``progress=`` argument into a callback (or None for "off").

    Accepted: ``None``, ``"auto"``, ``"plain"`` (never a bar, useful for logs
    and tests), or any callable taking the payload dict. Anything else raises
    with the list — a silently ignored ``progress="yes"`` would look exactly
    like a hung run.
    """
    if progress is None:
        return None
    if callable(progress):
        return progress
    if isinstance(progress, str):
        if progress == "auto":
            return ConsoleProgress(**kwargs)
        if progress == "plain":
            return ConsoleProgress(use_tqdm=False, **kwargs)
    raise ValueError(f"progress={progress!r} is not one of: {', '.join(ACCEPTED)}")


def chain(*callbacks: Callable[[dict], None] | None) -> Callable[[dict], None] | None:
    """Fan one payload out to several consumers (heartbeat + display + user).

    There is still ONE instrumentation site in the engine; this only splits
    the stream afterwards. A consumer that raises is contained by the engine,
    which warns once — but it would take its siblings down with it, so each
    one is isolated here too.
    """
    live = [cb for cb in callbacks if cb is not None]
    if not live:
        return None
    if len(live) == 1:
        return live[0]
    return _FanOut(tuple(live))


class _FanOut:
    """One payload, several consumers; one failing consumer does not mute the rest."""

    def __init__(self, consumers: tuple[Callable[[dict], None], ...]) -> None:
        self.consumers = consumers

    def __call__(self, ev: dict) -> None:
        first: BaseException | None = None
        for cb in self.consumers:
            try:
                cb(ev)
            except Exception as exc:  # noqa: BLE001 - report one, run all
                first = first or exc
        if first is not None:
            raise first  # the engine warns once and keeps solving

    def close(self) -> None:
        for cb in self.consumers:
            close(cb)


def close(callback: Callable[[dict], None] | None) -> None:
    """Release a consumer's display resources, if it has any."""
    closer = getattr(callback, "close", None)
    if callable(closer):
        closer()
