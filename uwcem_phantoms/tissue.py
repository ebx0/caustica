"""What the UWCEM media numbers sound like.

The repository is an ELECTROMAGNETIC dataset: its tissue classes are defined
by dielectric properties, and it ships no acoustic numbers at all. This
module supplies the missing half, and is explicit about how much of it is
measurement and how much is interpolation:

* **skin, muscle, fat, fibroglandular, water** — literature values (Duck,
  *Physical Properties of Tissue*, 1990; IT'IS Foundation Tissue Properties
  Database; Mast, ARLO 1(2), 2000; Duric et al. ultrasound-tomography breast
  measurements). Each entry carries a ``[lo, hi]`` range that is the reported
  spread, not a guess bracket.
* **the 1.1/1.2/1.3 and 3.1/3.2/3.3 sub-groups** — INTERPOLATED. The UWCEM
  sub-groups are ordered by water content (1.1 highest, 3.3 lowest) and water
  content is the dominant driver of sound speed and attenuation in soft
  tissue, so the sub-groups are placed on a monotone ramp between the
  "most glandular" and "most lipid" literature endpoints. This is a modelling
  choice; it is flagged as such on every row and can be replaced wholesale by
  passing a custom :class:`TissueTable`.

Frequency: attenuation is stored in power-law form ``alpha0 * f^b``
[dB/(cm MHz^b)] and converted to the single ``Np/m`` value the v1 solver
consumes at the export frequency ``f0``. Storing only Np/m would silently
pin a phantom to one frequency — the number is right at 1 MHz and wrong at
3 MHz, with nothing in the file to say so.

Three tissue models are offered:

``detailed`` (10 ids)
    Every media number kept separate. Ids equal the class codes.
``grouped`` (5 ids)
    coupling / skin / muscle / fibroglandular / fat — the resolution most
    HIFU studies actually need.
``simple`` (5 ids, 0-4)
    The same ID SPACE as :func:`caustica.materials.breast_default` (0 PML,
    1 skin, 2 fat/glandular, 3 muscle, 4 gel), so an export drops into
    existing caustica scripts unchanged. The property VALUES are this
    module's literature table, NOT breast_default's — skin absorption in
    particular is 2.2x higher (32.8 vs 15.0 Np/m at 1 MHz). Results are
    therefore not comparable to a breast_default run without saying so
    (review finding, 2026-08-18).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The literature acoustic values and their carrier live in caustica.materials
# since M10k/W0b (D17): they are generic soft-tissue literature, not UWCEM's.
# What stays HERE is everything UWCEM-specific — the media-number ordering,
# the interpolated sub-group ramp, the tissue models and the pval rules.
from caustica.materials import (  # noqa: F401 (conversion helpers re-exported)
    DB_CM_TO_NP_M,
    TISSUE_LIBRARY,
    AcousticTissue,
    Material,
    MaterialDB,
    db_cm_to_np_m,
    np_m_to_db_cm,
)
from uwcem_phantoms.reader import MEDIA_NUMBERS, MEDIA_TOKENS, N_CLASSES

TISSUE_MODELS = ("detailed", "grouped", "simple")


# ---------------------------------------------------------------------------
# The table. Endpoint rows come from caustica's literature library; sub-group
# rows are interpolated along the water-content axis the UWCEM numbering
# encodes (a UWCEM modelling choice, so they live here).
# ---------------------------------------------------------------------------

_RAMP = "interpolated along the UWCEM water-content ordering (see module docstring)"

#: Acoustic properties per class code (index = code in ``reader.MEDIA_NUMBERS``).
DEFAULT_TISSUES: tuple[AcousticTissue, ...] = (
    # 0 : media -1, immersion medium.
    TISSUE_LIBRARY["water_37c"],
    # 1 : media -2, skin (~1.5 mm dermis layer).
    TISSUE_LIBRARY["skin"],
    # 2 : media -4, pectoral muscle (chest wall).
    TISSUE_LIBRARY["muscle"],
    # 3 : media 1.1, fibroconnective/glandular-1 — HIGHEST water content.
    #     The glandular ENDPOINT of the ramp (literature-anchored).
    TISSUE_LIBRARY["fibroglandular"],
    # 4 : media 1.2
    AcousticTissue(
        name="Fibroglandular-2",
        c=(1515.0, 1560.0),
        rho=(1025.0, 1048.0),
        alpha0=(0.78, 1.05),
        b=1.1,
        bona=(7.3, 8.4),
        source=_RAMP,
        interpolated=True,
    ),
    # 5 : media 1.3
    AcousticTissue(
        name="Fibroglandular-3",
        c=(1500.0, 1540.0),
        rho=(1012.0, 1036.0),
        alpha0=(0.71, 0.95),
        b=1.1,
        bona=(7.6, 8.8),
        source=_RAMP,
        interpolated=True,
    ),
    # 6 : media 2, transitional tissue — the mid-point of the ramp by
    #     construction; the repository defines it as intermediate.
    AcousticTissue(
        name="Transitional",
        c=(1480.0, 1520.0),
        rho=(985.0, 1015.0),
        alpha0=(0.62, 0.85),
        b=1.1,
        bona=(8.2, 9.4),
        source=_RAMP,
        interpolated=True,
    ),
    # 7 : media 3.1, fatty-1 — highest water content of the fatty group.
    AcousticTissue(
        name="Fatty-1 (highest water)",
        c=(1455.0, 1495.0),
        rho=(950.0, 985.0),
        alpha0=(0.55, 0.75),
        b=1.1,
        bona=(8.8, 10.0),
        source=_RAMP,
        interpolated=True,
    ),
    # 8 : media 3.2
    AcousticTissue(
        name="Fatty-2",
        c=(1440.0, 1475.0),
        rho=(925.0, 955.0),
        alpha0=(0.48, 0.66),
        b=1.1,
        bona=(9.4, 10.7),
        source=_RAMP,
        interpolated=True,
    ),
    # 9 : media 3.3, fatty-3 — LOWEST water content, essentially lipid.
    #     The fat ENDPOINT of the ramp (literature-anchored).
    TISSUE_LIBRARY["fat"],
)

assert len(DEFAULT_TISSUES) == N_CLASSES  # noqa: S101 - table/code-space contract

#: Display colour per class code, for every renderer (web GUI, reports, any
#: future exporter) so one tissue is always one colour.
#:
#: Structure, not decoration. The ten classes are FIVE tissue families, two of
#: which are internally ORDERED (fibroglandular 1->3 and fatty 1->3 run down
#: the water-content axis). So the palette is five categorical hues carrying
#: an ordinal lightness ramp inside the two graded families — encoding
#: identity with hue and degree with lightness, which is what the data is.
#:
#: The five family hues were selected against the colour-vision gates on this
#: viewer's dark surface (#12181f) and clear them ALL-PAIRS, which is the
#: right test for a map where every tissue can touch every other: worst pair
#: CVD dE 9.0 (target >= 8), worst normal-vision dE 15.4 (floor 15), every hue
#: >= 3:1 contrast against the surface. Both sub-ramps pass the ordinal check
#: (monotone lightness, single hue, visible steps). Class 0 is the immersion
#: bath: it is the image's background, deliberately recessive and near
#: transparent in 3-D, so it is a SURFACE colour rather than a sixth series
#: and is excluded from those gates.
#:
#: Hue assignment follows imaging habit where the gates allowed it: warm red
#: for the thin skin shell, crimson for muscle, teal for fibroglandular
#: density, violet for the transitional class that sits between gland and
#: fat, gold for fat (palest step = purest lipid).
DEFAULT_COLORS: tuple[str, ...] = (
    "#101a26",  # 0 coupling medium  (recessive surface, not a series)
    "#c6360c",  # 1 skin             family: warm red
    "#b93c96",  # 2 muscle           family: crimson
    "#128396",  # 3 fibroglandular-1 teal ramp, darkest = highest water
    "#1ca4bb",  # 4 fibroglandular-2
    "#45c1d9",  # 5 fibroglandular-3
    "#b759fa",  # 6 transitional     family: violet
    "#85761e",  # 7 fatty-1          gold ramp, darkest = highest water
    "#a0913e",  # 8 fatty-2
    "#bfb05e",  # 9 fatty-3          palest = purest lipid
)

#: The dark surface the palette above was validated against.
VIEWER_SURFACE = "#12181f"


#: ``detailed`` model: material id == class code.
DETAILED_IDS: dict[int, int] = {i: i for i in range(N_CLASSES)}

#: ``grouped`` model: 0 coupling, 1 skin, 2 muscle, 3 fibroglandular, 4 fat.
GROUPED_IDS: dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3, 7: 4, 8: 4, 9: 4}

#: ``simple`` model: the ids of :func:`caustica.materials.breast_default`
#: (0 PML, 1 skin, 2 fat/glandular, 3 muscle, 4 gel), so exports drop into
#: existing scripts. The ids match; the property values do not (see the module
#: docstring). All breast interior tissue collapses onto id 2.
SIMPLE_IDS: dict[int, int] = {0: 4, 1: 1, 2: 3, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2}

CODE_TO_ID: dict[str, dict[int, int]] = {
    "detailed": DETAILED_IDS,
    "grouped": GROUPED_IDS,
    "simple": SIMPLE_IDS,
}


#: Class codes the repository actually ships a ``p`` value for: "each voxel
#: that represents normal breast tissue or subcutaneous fat has a value of p
#: within [0, 1] ... the values of p for all other voxels are set to zero"
#: (instruction manual, pval section). Skin, muscle and the immersion medium
#: are therefore NOT p-interpolated — their stored 0 means "no data", not
#: "lowest properties", and reading it as the latter would drag every skin
#: voxel in the phantom down to its literature minimum.
PVAL_CODES: frozenset[int] = frozenset({3, 4, 5, 6, 7, 8, 9})


@dataclass(frozen=True)
class TissueTable:
    """A tissue model: how class codes map to material ids, and their physics."""

    model: str
    code_to_id: dict[int, int]
    tissues: dict[int, AcousticTissue]  # keyed by MATERIAL id
    f0: float

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.tissues))

    @property
    def coupling_id(self) -> int:
        """Material id of the immersion/coupling medium (class code 0)."""
        return self.code_to_id[0]

    @property
    def pval_ids(self) -> frozenset[int]:
        """Material ids whose voxels carry a meaningful ``p``.

        A merged id qualifies only when EVERY code folded into it has pval
        data; a mixed id (none exist in the shipped models, but a custom
        mapping could make one) is excluded, because applying the
        interpolation to half its voxels would be worse than not at all.
        """
        by_id: dict[int, list[int]] = {}
        for code, mat_id in self.code_to_id.items():
            by_id.setdefault(mat_id, []).append(code)
        return frozenset(
            mat_id for mat_id, codes in by_id.items() if all(c in PVAL_CODES for c in codes)
        )

    def material_db(self, which: str = "mid") -> MaterialDB:
        """The :class:`~caustica.materials.MaterialDB` for this table at ``f0``."""
        return MaterialDB(
            materials={i: t.to_material(self.f0, which) for i, t in self.tissues.items()}
        )

    def lookup(self, which: str, pval_only: bool = False) -> dict[str, np.ndarray]:
        """Per-property arrays indexed by MATERIAL id (dense, id 0..max).

        With ``pval_only=True``, ids that carry no ``p`` data collapse to
        their midpoint regardless of ``which`` — so the caller can blend
        ``lo -> hi`` everywhere and still leave skin, muscle and the coupling
        bath at their nominal values.
        """
        n = max(self.tissues) + 1
        out = {name: np.zeros(n, dtype=np.float32) for name in ("alpha", "rho", "c", "beta")}
        allowed = self.pval_ids
        for i, t in self.tissues.items():
            w = which if (not pval_only or i in allowed) else "mid"
            out["alpha"][i] = t.alpha_np_m(self.f0, w)
            out["rho"][i] = AcousticTissue._pick(t.rho, w)
            out["c"][i] = AcousticTissue._pick(t.c, w)
            out["beta"][i] = t.beta(w)
        return out

    def lookup_by_code(self, which: str) -> dict[str, np.ndarray]:
        """Per-property arrays indexed by CLASS CODE (0..N_CLASSES-1).

        ``p`` is a WITHIN-CLASS position -- the UWCEM manual defines it
        against the bounding curves of "the given tissue type", i.e. the
        media number, not whatever material id the tissue model merged that
        media number into. Blending over a merged id's UNION range would
        rescale ``p`` onto a band 3-4x too wide and hand a pure-lipid voxel a
        glandular sound speed, so the pval blend indexes here instead.

        Codes the repository ships no ``p`` for (skin, muscle, the bath)
        collapse to their midpoint, exactly as :meth:`lookup` does.
        """
        names = ("alpha", "rho", "c", "beta")
        out = {name: np.zeros(N_CLASSES, dtype=np.float32) for name in names}
        members: dict[int, int] = {}
        for mat_id in self.code_to_id.values():
            members[mat_id] = members.get(mat_id, 0) + 1
        for code in range(N_CLASSES):
            mat_id = self.code_to_id[code]
            # A material id built from ONE code keeps the table's own row, so
            # overrides survive; a merged id has to fall back to the per-class
            # row, which is what p was measured against.
            t = self.tissues[mat_id] if members[mat_id] == 1 else DEFAULT_TISSUES[code]
            w = which if code in PVAL_CODES else "mid"
            out["alpha"][code] = t.alpha_np_m(self.f0, w)
            out["rho"][code] = AcousticTissue._pick(t.rho, w)
            out["c"][code] = AcousticTissue._pick(t.c, w)
            out["beta"][code] = t.beta(w)
        return out

    def color(self, mat_id: int) -> str:
        """Display colour of a material id (its lowest constituent class code).

        An id that no class code maps to — the ``simple`` model's unused PML
        slot at id 0 — takes the bath colour rather than a grey fallback: it
        is a duplicate of the coupling medium, and a stray neutral in a
        five-colour palette reads as a real tissue.
        """
        codes = sorted(c for c, m in self.code_to_id.items() if m == mat_id)
        if codes:
            return DEFAULT_COLORS[codes[0]]
        return DEFAULT_COLORS[0]

    def describe(self) -> list[dict]:
        """One JSON-ready row per material id (for the CLI table and the GUI)."""
        rows = []
        for i in self.ids:
            t = self.tissues[i]
            rows.append(
                {
                    "id": i,
                    "name": t.name,
                    "color": self.color(i),
                    "has_pval": i in self.pval_ids,
                    "c_mid": round(AcousticTissue._pick(t.c, "mid"), 1),
                    "c_range": list(t.c),
                    "rho_mid": round(AcousticTissue._pick(t.rho, "mid"), 1),
                    "rho_range": list(t.rho),
                    "alpha_np_m": round(t.alpha_np_m(self.f0), 3),
                    "alpha_db_cm": round(np_m_to_db_cm(t.alpha_np_m(self.f0)), 3),
                    "alpha0_db_cm_mhz": list(t.alpha0),
                    "alpha_power": t.b,
                    "beta_mid": round(t.beta(), 3),
                    "bona_range": list(t.bona),
                    "interpolated": t.interpolated,
                    "source": t.source,
                    "codes": [c for c, m in self.code_to_id.items() if m == i],
                    "media_numbers": [
                        MEDIA_NUMBERS[c] for c, m in self.code_to_id.items() if m == i
                    ],
                }
            )
        return rows


def _merge(tissues: list[AcousticTissue], name: str) -> AcousticTissue:
    """Merge several tissues into one by taking the union of their ranges.

    The union describes the MERGED MATERIAL — "fat, somewhere between pure
    lipid and the wettest fatty tissue" — which is what a piecewise-constant
    build should report. It is NOT the band the pval blend may use: ``p`` is
    measured within one media number, so blending it across the union would
    stretch it over a range twice as wide and give a pure-lipid voxel a
    fatty-1 sound speed. That is why the blend goes through
    :meth:`TissueTable.lookup_by_code` instead (review finding, 2026-08-18).
    """
    return AcousticTissue(
        name=name,
        c=(min(t.c[0] for t in tissues), max(t.c[1] for t in tissues)),
        rho=(min(t.rho[0] for t in tissues), max(t.rho[1] for t in tissues)),
        alpha0=(min(t.alpha0[0] for t in tissues), max(t.alpha0[1] for t in tissues)),
        b=float(np.mean([t.b for t in tissues])),
        bona=(min(t.bona[0] for t in tissues), max(t.bona[1] for t in tissues)),
        source="merged: " + "; ".join(dict.fromkeys(t.source for t in tissues)),
        interpolated=any(t.interpolated for t in tissues),
    )


def tissue_table(
    model: str = "detailed",
    f0: float = 1.0e6,
    overrides: dict[int, AcousticTissue] | None = None,
) -> TissueTable:
    """Build the :class:`TissueTable` for one of :data:`TISSUE_MODELS`.

    Merged models take the UNION of their members' ranges, so a "fat" id in
    the grouped model spans everything from fatty-1 to fatty-3 — the
    heterogeneity the merge threw away reappears as a wider pval spread
    instead of vanishing.
    """
    if model not in CODE_TO_ID:
        raise ValueError(f"unknown tissue model {model!r}; pick from {TISSUE_MODELS}")
    if f0 <= 0:
        raise ValueError(f"f0 must be > 0 Hz, got {f0}")
    mapping = CODE_TO_ID[model]
    members: dict[int, list[AcousticTissue]] = {}
    for code, mat_id in mapping.items():
        members.setdefault(mat_id, []).append(DEFAULT_TISSUES[code])
    tissues = {
        mat_id: (group[0] if len(group) == 1 else _merge(group, _merged_name(model, mat_id)))
        for mat_id, group in members.items()
    }
    if model == "simple":
        # id 0 is breast_default's PML slot: unused by any voxel, but present
        # so the emitted MaterialDB stays key-compatible with existing code.
        tissues[0] = DEFAULT_TISSUES[0]
    if overrides:
        unknown = [k for k in overrides if k not in tissues]
        if unknown:
            raise ValueError(f"override ids {unknown} are not in the {model!r} model")
        tissues = {**tissues, **overrides}
    return TissueTable(model=model, code_to_id=dict(mapping), tissues=tissues, f0=float(f0))


_MERGED_NAMES = {
    "grouped": {3: "Fibroglandular (merged)", 4: "Fat (merged)"},
    "simple": {2: "Fat/glandular (merged breast interior)"},
}


def _merged_name(model: str, mat_id: int) -> str:
    return _MERGED_NAMES.get(model, {}).get(mat_id, f"{model} id {mat_id}")


def media_number_label(code: int) -> str:
    """Human label for a class code, e.g. ``"3.2 Fatty-2"``."""
    return f"{MEDIA_TOKENS[code]} {DEFAULT_TISSUES[code].name}"
