"""Interactive launcher for the phantom submodule — flags without memorising them.

``python -m uwcem_phantoms build ...`` has twenty flags, which is the right
shape for a script and the wrong shape for "I just want to look at a phantom".
This is the same functionality behind a menu: pick, answer a few questions with
sane defaults, see what it will cost *before* it runs, then build.

Two things it deliberately does beyond wrapping the CLI:

* it suggests a ``dx`` from the chosen ``f0`` instead of letting the user find
  out after a 4-minute build that they asked for 1.3 points per wavelength;
* it runs :func:`~uwcem_phantoms.builder.plan` and shows the grid, the voxel
  count and the peak RAM before committing.

Every wizard ends by printing the equivalent CLI line, so this stays a way to
LEARN the flags rather than a substitute for them.

Run it with ``phantoms.bat`` (Windows), ``./phantoms.sh`` (POSIX), or::

    python -m apps.phantom_launcher            # menu
    python -m apps.phantom_launcher gui        # straight to one action
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- presentation


def _enable_ansi() -> bool:
    """Turn on VT processing on Windows; report whether colour is usable."""
    if not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if not ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:  # noqa: BLE001 - colour is a nicety, never a failure
            return False
    return True


for _stream in (sys.stdout, sys.stderr):
    # Never let an unrepresentable character in a console code page raise
    # UnicodeEncodeError mid-prompt: output is ASCII by policy, and this is the
    # backstop for anything that arrives from a phantom name or an exception.
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001 - not a console, or already wrapped
        pass


COLOR = _enable_ansi()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def bold(t: str) -> str:
    return _c(t, "1")


def dim(t: str) -> str:
    return _c(t, "2")


def cyan(t: str) -> str:
    return _c(t, "36")


def warn(t: str) -> str:
    return _c(t, "33")


def bad(t: str) -> str:
    return _c(t, "31")


def rule(char: str = "-") -> None:
    print(dim(char * 72))


def header(title: str) -> None:
    print()
    rule("=")
    print(f"  {bold(title)}")
    rule("=")


# -------------------------------------------------------------------- prompting


class Abort(Exception):
    """The user pressed Ctrl-C / Ctrl-D - back to the menu, not out of the app."""


def _read(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        raise Abort from None


def ask(prompt: str, default=None, cast=str, choices=None):
    """One question, with its default shown and its answer validated."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = _read(f"  {prompt}{dim(suffix)}: ")
        if not raw:
            if default is None:
                print(bad("    an answer is required"))
                continue
            return default
        try:
            value = cast(raw)
        except (TypeError, ValueError) as exc:
            print(bad(f"    {exc}"))
            continue
        if choices is not None and value not in choices:
            print(bad(f"    pick one of {list(choices)}"))
            continue
        return value


def ask_bool(prompt: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = _read(f"  {prompt} {dim('[' + hint + ']')}: ").lower()
        if not raw:
            return default
        if raw in ("y", "yes", "e", "evet", "1"):
            return True
        if raw in ("n", "no", "h", "hayir", "hayır", "0"):
            return False
        print(bad("    answer y or n"))


def ask_float(prompt: str, default: float, lo: float | None = None, hi: float | None = None):
    def cast(raw: str) -> float:
        v = float(raw.replace(",", "."))
        if lo is not None and v < lo:
            raise ValueError(f"must be >= {lo}")
        if hi is not None and v > hi:
            raise ValueError(f"must be <= {hi}")
        return v

    return ask(prompt, default, cast)


def ask_int(prompt: str, default: int, lo: int | None = None):
    def cast(raw: str) -> int:
        v = int(raw)
        if lo is not None and v < lo:
            raise ValueError(f"must be >= {lo}")
        return v

    return ask(prompt, default, cast)


def choose(prompt: str, options: list[str], default: str, labels: list[str] | None = None) -> str:
    """Numbered pick-list; the number OR the value itself is accepted."""
    for i, opt in enumerate(options, 1):
        label = f"  {dim('-')} {labels[i - 1]}" if labels else ""
        mark = "*" if opt == default else " "
        print(f"    {mark} {i}) {opt}{label}")

    def cast(raw: str) -> str:
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        if raw in options:
            return raw
        raise ValueError(f"pick 1-{len(options)} or a name")

    return ask(prompt, default, cast)


# ------------------------------------------------------------------- helpers


def human_bytes(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1024
    return f"{n:.1f} TB"


def as_snippet_path(path: Path) -> str:
    """A path safe to paste into Python source: forward slashes, no escapes."""
    return str(path).replace("\\", "/")


def slowest_sound_speed(model: str, f0_mhz: float) -> float:
    """Lowest LOW-endpoint sound speed in the table - where lambda is shortest."""
    from uwcem_phantoms import tissue_table

    table = tissue_table(model, f0=f0_mhz * 1e6)
    return float(table.lookup("lo")["c"].min())


def ppw_line(c_min: float, f0_mhz: float, dx_mm: float) -> str:
    ppw = c_min / (f0_mhz * 1e6 * dx_mm * 1e-3)
    text = f"{ppw:.1f} points per wavelength in the slowest tissue ({c_min:.0f} m/s)"
    if ppw < 2.0:
        return bad(f"    UNUSABLE: {text} - the solve will produce no focus at all")
    if ppw < 4.0:
        return warn(f"    marginal: {text} (design rule is ~4.4)")
    return dim(f"    ok: {text}")


# --------------------------------------------------------------------- actions


def action_gui() -> None:
    """Launch the local web studio."""
    header("phantom studio (web GUI)")
    port = ask_int("port", 8765, lo=1)
    open_browser = ask_bool("open a browser window", True)
    preview = ask_float("preview budget (Mvox) - bigger is sharper and slower", 10.0, lo=0.5)
    print()
    print(dim("  Ctrl-C in this window stops the server and returns to the menu."))

    from apps.phantom_studio.server import serve

    try:
        serve(port=port, preview_mvox=preview, open_browser=open_browser)
    except OSError as exc:
        print(bad(f"  could not start on port {port}: {exc}"))
        print(dim("  another studio is probably already running - try a different port"))


def action_catalog() -> None:
    """Print the repository catalog and what is downloaded locally."""
    header("catalog")
    from uwcem_phantoms.cli import main as cli_main

    cli_main(["list"])


def action_tissues() -> None:
    """Print the acoustic property table at a chosen frequency."""
    header("tissue table")
    model = choose(
        "tissue model",
        ["detailed", "grouped", "simple"],
        "detailed",
        ["10 classes, every fatty/glandular sub-group", "6 classes", "4 classes"],
    )
    f0 = ask_float("f0 (MHz)", 1.0, lo=0.01, hi=50.0)
    print()
    from uwcem_phantoms.cli import main as cli_main

    cli_main(["tissues", "--model", model, "--f0", str(f0)])


def action_fetch() -> None:
    """Download raw phantom archives."""
    header("download raw archives")
    from uwcem_phantoms import PHANTOM_IDS, catalog

    rows = catalog.status()
    missing = [r["id"] for r in rows if not (r["has_mtype"] and r["has_pval"])]
    if not missing:
        print(dim("  every phantom is already downloaded (mtype + pval)."))
        if not ask_bool("re-download anyway", False):
            return
        targets = list(PHANTOM_IDS)
        force = True
    else:
        print(f"  missing or incomplete: {', '.join(missing)}")
        which = choose("download", ["missing", "all", "one"], "missing")
        force = which == "all"
        if which == "one":
            targets = [choose("phantom", list(PHANTOM_IDS), missing[0])]
        else:
            targets = missing if which == "missing" else list(PHANTOM_IDS)
    with_pval = ask_bool("include the large pval archives", True)

    last = [""]

    def progress(name: str, done: int, total: int) -> None:
        tag = f"{name} {done / 1e6:6.1f}/{total / 1e6:6.1f} MB"
        if tag != last[0]:
            print(f"\r  {tag}", end="", flush=True)
            last[0] = tag

    for pid in targets:
        print(f"  {pid}:")
        catalog.fetch(pid, with_pval=with_pval, force=force, progress=progress)
        print("\r  done" + " " * 40)


def action_info() -> None:
    """Describe an exported .npz."""
    header("inspect an export")
    from uwcem_phantoms.asset import PhantomAsset
    from uwcem_phantoms.paths import export_dir

    files = sorted(export_dir().glob("*.npz"))
    if not files:
        print(dim(f"  no exports yet in {export_dir()}"))
        return
    for i, f in enumerate(files, 1):
        print(f"    {i}) {f.name}  {dim(human_bytes(f.stat().st_size))}")

    def cast(raw: str) -> Path:
        if raw.isdigit() and 1 <= int(raw) <= len(files):
            return files[int(raw) - 1]
        # A bare name means "the one in the export directory" -- that is the
        # list the user is looking at, and Path(raw) alone would resolve it
        # against the CWD instead and reject it.
        for candidate in (export_dir() / raw, export_dir() / f"{raw}.npz", Path(raw)):
            if candidate.is_file():
                return candidate
        raise ValueError("pick a number from the list, or give a path")

    path = Path(ask("file", files[-1].name, cast))
    if not path.is_file():
        path = export_dir() / path.name
    asset = PhantomAsset.load(path)
    print()
    print(asset.summary())
    if asset.meta.get("warnings"):
        print()
        for w in asset.meta["warnings"]:
            print(warn(f"  ! {w}"))
    if ask_bool("\n  show the build log", False):
        for line in asset.meta.get("log", []):
            print(dim(f"    - {line}"))
    print()
    print(dim("  import it with:"))
    print("    from uwcem_phantoms import load_phantom")
    print(f'    ph = load_phantom("{as_snippet_path(path)}")')
    print("    grid, medium = ph.grid(), ph.to_medium()")


def action_dataset() -> None:
    """Build (or verify) the standard aligned dataset in data/phantoms."""
    header("standard dataset - every phantom on ONE aligned grid")
    from uwcem_phantoms.dataset import (
        DATASET_DX_MM,
        DEPTH_LIMIT_MM,
        build_dataset,
        plan_dataset,
        verify_dataset,
    )
    from uwcem_phantoms.paths import dataset_dir

    print(dim(f"  output: {dataset_dir()}"))
    if ask_bool("verify what is on disk instead of building", False):
        report = verify_dataset(progress=lambda m: print(dim(f"  {m}")))
        print(f"  OK: {len(report['files'])} file(s) passed every check")
        return

    dx = ask_float("dx (mm)", DATASET_DX_MM, lo=0.05, hi=2.0)
    depth = ask_float("depth limit (mm, 0 = no limit)", DEPTH_LIMIT_MM or 0.0, lo=0.0, hi=500.0)
    print(dim("  surveying the phantoms for the common box..."))
    plan = plan_dataset(dx_mm=dx, depth_limit_mm=depth or None)
    print(f"  {plan.summary()}")
    print(dim("  files are compressed on disk; the peak above is transient build RAM."))
    if plan.depth_capped and plan.clipped_ids():
        print(
            warn(
                "  the depth limit CUTS tissue off the back of the phantoms listed "
                "above — this is a destructive crop, not padding."
            )
        )
    if not ask_bool("build all nine now (this takes a while)", True):
        print(dim("  nothing built."))
        return
    build_dataset(plan=plan, progress=lambda m: print(f"  {m}"))
    print()
    from uwcem_phantoms.dataset import dataset_filename

    print(dim("  load one in a simulation:"))
    print("    from uwcem_phantoms import load_phantom")
    print(f"    ph = load_phantom('data/phantoms/{dataset_filename('012304', dx)}')")
    print("    grid, medium = ph.grid(), ph.to_medium()")


# ------------------------------------------------------------- the build wizard

# Flags cli.py's `build` subcommand does not expose; setting one of these means
# the printed CLI equivalent would be a lie, so the wizard prints the spec JSON
# instead. Better a longer snippet than a command that silently builds something
# else.
CLI_CANNOT_EXPRESS = ("keep_largest_only", "close_skin_iterations")


def cli_equivalent(spec, out: str | None, store_properties: str) -> str | None:
    """The `python -m uwcem_phantoms build ...` line for this spec, or None."""
    s, r, c, d, h = spec.simplify, spec.resolution, spec.crop, spec.domain, spec.heterogeneity
    if s.keep_largest_only or s.close_skin_iterations:
        return None
    parts = ["python -m uwcem_phantoms build", spec.phantom_id]
    parts += [f"--f0 {spec.f0_mhz:g}"]
    if s.tissue_model != "detailed":
        parts += [f"--model {s.tissue_model}"]
    if r.dx_mm is not None:
        parts += [f"--dx {r.dx_mm:g}"]
    if r.method != "smooth":
        parts += [f"--resample {r.method}"]
    if c.mode != "breast":
        parts += [f"--crop {c.mode}"]
    if c.margin_mm != 5.0:
        parts += [f"--margin {c.margin_mm:g}"]
    for flag, value, default in (
        ("--standoff", d.standoff_mm, 0.0),
        ("--backing", d.backing_mm, 0.0),
        ("--lateral", d.lateral_margin_mm, 0.0),
    ):
        if value != default:
            parts += [f"{flag} {value:g}"]
    if not d.fft_friendly:
        parts += ["--no-fft-pad"]
    if d.max_voxels:
        parts += [f"--max-voxels {d.max_voxels}"]
    if s.drop_muscle:
        parts += ["--drop-muscle"]
    if s.drop_skin:
        parts += ["--drop-skin"]
    if s.fill_holes:
        parts += ["--fill-holes"]
    if s.remove_islands_vox:
        parts += [f"--remove-islands {s.remove_islands_vox}"]
    if s.smooth_iterations:
        parts += [f"--smooth {s.smooth_iterations}"]
    if h.use_pval:
        parts += ["--pval"]
    if h.noise_pct:
        parts += [f"--noise {h.noise_pct:g}"]
        if h.correlation_mm:
            parts += [f"--correlation {h.correlation_mm:g}"]
        if h.seed:
            parts += [f"--seed {h.seed}"]
    if spec.name:
        parts += [f"--name {spec.name}"]
    if out:
        parts += [f'--out "{out}"']
    if store_properties != "auto":
        parts += [f"--store-properties {store_properties}"]
    return " ".join(parts)


def action_build(state: dict) -> None:
    """Guided build: questions -> plan -> confirm -> export."""
    header("build a phantom")
    from uwcem_phantoms import PhantomSpec, build, status
    from uwcem_phantoms.builder import plan

    # --- which phantom ---------------------------------------------------
    rows = status()
    print(dim("    #  id           ACR  grid              local   density"))
    for i, r in enumerate(rows, 1):
        have = "yes" if r["has_mtype"] else dim("no")
        grid = "x".join(str(n) for n in r["shape"])
        mark = "*" if r["id"] == state.get("phantom_id") else " "
        print(
            f"   {mark}{i:>2}) {r['id']:<12} {r['acr_class']:>2}   "
            f"{grid:<16}  {have:<6}  {r['acr_description']}"
        )
    ids = [r["id"] for r in rows]

    def cast_id(raw: str) -> str:
        if raw.isdigit() and 1 <= int(raw) <= len(ids):
            return ids[int(raw) - 1]
        if raw in ids:
            return raw
        raise ValueError(f"pick 1-{len(ids)} or an id")

    phantom_id = ask("phantom", state.get("phantom_id", "012304"), cast_id)

    # --- physics ---------------------------------------------------------
    f0 = ask_float(
        "f0 (MHz) - the attenuation power law is evaluated here",
        state.get("f0", 1.0),
        lo=0.01,
        hi=50.0,
    )
    print()
    model = choose(
        "tissue model",
        ["detailed", "grouped", "simple"],
        state.get("model", "detailed"),
        ["10 classes", "6 classes", "4 classes (fat / gland / skin / bath)"],
    )

    c_min = slowest_sound_speed(model, f0)
    design_mm = c_min / (f0 * 1e6 * 4.4) * 1e3
    floor_mm = c_min / (f0 * 1e6 * 2.0) * 1e3
    suggested = float(f"{design_mm:.2g}")
    print()
    print(dim(f"    at {f0:g} MHz: dx <= {design_mm:.2f} mm for the 4.4 ppw design rule,"))
    print(dim(f"    {floor_mm:.2f} mm is the hard floor below which no focus forms."))
    dx = ask_float("dx (mm)", state.get("dx", suggested), lo=0.05, hi=10.0)
    print(ppw_line(c_min, f0, dx))

    # --- geometry --------------------------------------------------------
    print()
    crop = choose(
        "crop",
        ["breast", "tissue", "none"],
        state.get("crop", "breast"),
        ["the protruding breast only", "bounding box of all tissue", "keep the whole phantom"],
    )
    margin = 5.0
    if crop != "none":
        margin = ask_float("crop margin (mm)", state.get("margin", 5.0), lo=0.0)
    standoff = ask_float(
        "standoff (mm) - coupling medium in front for the transducer",
        state.get("standoff", 15.0),
        lo=0.0,
    )

    # --- heterogeneity ---------------------------------------------------
    print()
    use_pval = ask_bool(
        "use the repository's measured per-voxel heterogeneity (pval)", state.get("pval", True)
    )
    noise = ask_float("scatterer noise (%) - 0 for none", state.get("noise", 0.0), lo=0.0, hi=50.0)
    correlation, seed = 0.0, 0
    if noise > 0:
        correlation = ask_float(
            "noise correlation length (mm)", state.get("correlation", 0.6), lo=0.0, hi=20.0
        )
        seed = ask_int("seed", state.get("seed", 0), lo=0)
        print(dim("    note: coupled rho+c noise gives ~2x this in impedance contrast"))

    # --- the long tail ---------------------------------------------------
    simplify = {"tissue_model": model}
    domain = {"standoff_mm": standoff}
    store_properties = "auto"
    print()
    if ask_bool("edit advanced options", False):
        simplify.update(
            drop_muscle=ask_bool("  replace the chest wall with coupling medium", False),
            drop_skin=ask_bool("  remove the skin layer", False),
            keep_largest_only=ask_bool("  discard tissue disconnected from the breast", False),
            fill_holes=ask_bool("  fill coupling-medium pockets inside tissue", False),
            remove_islands_vox=ask_int("  dissolve components smaller than N voxels", 0, lo=0),
            close_skin_iterations=ask_int("  skin closing passes", 0, lo=0),
            smooth_iterations=ask_int("  majority-filter passes", 0, lo=0),
        )
        domain.update(
            backing_mm=ask_float("  coupling medium behind the chest wall (mm)", 0.0, lo=0.0),
            lateral_margin_mm=ask_float(
                "  transverse margin added on all four sides (mm)", 0.0, lo=0.0
            ),
            fft_friendly=ask_bool("  pad each axis to a 2/3/5/7-smooth size (faster FFTs)", True),
        )
        store_properties = choose(
            "  store dense property volumes",
            ["auto", "always", "never"],
            "auto",
            ["only when pval/noise make them necessary", "force (big files)", "labels only"],
        )

    resample = "smooth"
    name = ask("export name (blank = descriptive default)", "", str) or None

    spec = PhantomSpec(
        phantom_id=phantom_id,
        f0_mhz=f0,
        simplify=simplify,
        resolution={"dx_mm": dx, "method": resample},
        crop={"mode": crop, "margin_mm": margin},
        domain=domain,
        heterogeneity={
            "use_pval": use_pval,
            "noise_pct": noise,
            "correlation_mm": correlation,
            "seed": seed,
        },
        name=name,
    )
    state.update(
        phantom_id=phantom_id,
        f0=f0,
        model=model,
        dx=dx,
        crop=crop,
        margin=margin,
        standoff=standoff,
        pval=use_pval,
        noise=noise,
        correlation=correlation,
        seed=seed,
    )

    # --- what it will cost, before it costs it ---------------------------
    print()
    print(dim("  planning..."))
    p = plan(spec)
    bound = "" if p.exact else warn("  (upper bound - simplification may shrink it)")
    print()
    print(
        f"  grid    : {bold('x'.join(str(n) for n in p.shape))} @ {p.dx * 1e3:g} mm"
        f"  =  {'x'.join(f'{e:g}' for e in p.extent_mm)} mm{bound}"
    )
    print(f"  voxels  : {p.n_voxels:,}  ({p.n_voxels / 1e6:.1f} Mvox)")
    print(
        f"  export  : labels {human_bytes(p.label_bytes)}"
        + (f" + properties {human_bytes(p.property_bytes)}" if p.property_bytes else "")
        + dim("  (before compression)")
    )
    peak = f"  peak RAM: ~{human_bytes(p.peak_bytes)}"
    print(bad(peak) if p.peak_bytes > 8e9 else warn(peak) if p.peak_bytes > 3e9 else peak)
    print(ppw_line(c_min, f0, dx))
    print(f"  name    : {spec.export_name()}.npz")

    line = cli_equivalent(spec, None, store_properties)
    print()
    if line:
        print(dim("  same thing from the command line:"))
        print(f"    {cyan(line)}")
    else:
        print(dim("  this spec uses options the CLI has no flag for; as JSON:"))
        print(dim("    from uwcem_phantoms import PhantomSpec, build"))
        print(dim(f"    build(PhantomSpec.model_validate_json(r'''{spec.model_dump_json()}'''))"))

    print()
    if not ask_bool("build it now", True):
        print(dim("  nothing built."))
        return

    print()
    asset = build(spec, progress=lambda m, f: print(f"  [{f:4.0%}] {m}"))
    print()
    print(asset.summary())
    for w in asset.meta.get("warnings", []):
        print()
        print(warn(f"  ! {w}"))
    path = asset.save(store_properties=store_properties)
    print()
    print(f"  wrote {bold(str(path))}  {dim(human_bytes(path.stat().st_size))}")
    print()
    print(dim("  use it in a simulation:"))
    print("    from uwcem_phantoms import load_phantom")
    print(f'    ph     = load_phantom("{as_snippet_path(path)}")')
    print("    grid   = ph.grid(pml=PMLSpec(thickness=10))")
    print("    medium = ph.to_medium()          # linear solver: to_medium(linear=True)")


# ----------------------------------------------------------------------- menu

MENU = [
    ("gui", "Studio GUI", "the web app: 3-D + slices, every knob live"),
    ("build", "Build a phantom", "guided flags -> a simulation-ready .npz"),
    ("dataset", "Standard dataset", "all nine on ONE aligned grid -> data/phantoms"),
    ("catalog", "Catalog", "the 9 phantoms and what is downloaded"),
    ("tissues", "Tissue table", "acoustic properties at a chosen f0"),
    ("info", "Inspect an export", "summary, warnings, build log"),
    ("fetch", "Download archives", "raw phantom data from UWCEM"),
]


def run_action(key: str, state: dict) -> None:
    if key == "gui":
        action_gui()
    elif key == "build":
        action_build(state)
    elif key == "dataset":
        action_dataset()
    elif key == "catalog":
        action_catalog()
    elif key == "tissues":
        action_tissues()
    elif key == "info":
        action_info()
    elif key == "fetch":
        action_fetch()
    else:
        raise KeyError(key)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    state: dict = {}

    if argv:
        key = argv[0].lstrip("-")
        aliases = {"studio": "gui", "web": "gui", "list": "catalog", "download": "fetch"}
        key = aliases.get(key, key)
        if key in ("help", "h", "?"):
            print(f"usage: python -m apps.phantom_launcher [{'|'.join(k for k, _, _ in MENU)}]")
            return 0
        if key not in {k for k, _, _ in MENU}:
            print(bad(f"unknown action {argv[0]!r}"))
            return 2
        try:
            run_action(key, state)
        except Abort:
            print(dim("cancelled"))
        except Exception as exc:  # noqa: BLE001 - one-shot mode gets the same
            # friendly failure the menu loop gives, not a raw traceback
            # (e.g. DatasetError from `dataset` verify with nothing on disk).
            print(bad(f"{type(exc).__name__}: {exc}"))
            if "--traceback" in sys.argv:
                import traceback

                traceback.print_exc()
            else:
                print(dim("(re-run with --traceback for the full stack)"))
            return 1
        return 0

    while True:
        header("caustica  -  phantom launcher")
        for i, (_, title, blurb) in enumerate(MENU, 1):
            print(f"   {i})  {bold(title):<26} {dim(blurb)}")
        print(f"   0)  {dim('Quit')}")
        print()
        try:
            raw = _read("  choice: ")
        except Abort:
            print(dim("bye"))
            return 0
        if raw in ("0", "q", "quit", "exit"):
            print(dim("bye"))
            return 0
        if not (raw.isdigit() and 1 <= int(raw) <= len(MENU)):
            print(bad(f"  pick 1-{len(MENU)} or 0"))
            continue
        try:
            run_action(MENU[int(raw) - 1][0], state)
        except Abort:
            print(dim("  cancelled - back to the menu"))
        except Exception as exc:  # noqa: BLE001 - a menu must survive a bad build
            print()
            print(bad(f"  {type(exc).__name__}: {exc}"))
            if "--traceback" in sys.argv:
                import traceback

                traceback.print_exc()
            else:
                print(dim("  (re-run with --traceback for the full stack)"))
        try:
            _read(dim("\n  press Enter for the menu "))
        except Abort:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
