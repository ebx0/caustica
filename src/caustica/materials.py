"""Acoustic material definitions and the id -> material database.

A :class:`Material` declares its absorption in ONE of two forms:

* **power law** (preferred), ``alpha0_db_cm_mhz_y`` [dB/(cm MHz^y)] with the
  exponent ``y``, so ``alpha(f) = alpha0 f^y`` and no frequency is baked into
  the table;
* **legacy single frequency**, ``alpha_np_m`` [Np/m], frequency independent.
  It is the v1 form and stays valid: it is what every stored job and every
  ``medium_volume`` file on disk carries, and it means "y = 0", the same
  number at every frequency.

:meth:`Material.alpha_np_m_at` answers both forms at a named frequency, and
:meth:`Material.at_frequency` bakes a power-law material down to the legacy
form once a run has chosen its ``f0``.

The rest of the acoustic set is unchanged: ``rho`` [kg/m^3], ``c`` [m/s],
``beta`` [-] (nonlinearity coefficient, beta = 1 + B/2A). Thermal fields
(conductivity [W/m/K], specific heat [J/kg/K], perfusion [1/s]) stay optional
on a material because an acoustic run does not read them; every material this
module ships fills them from the IT'IS database, with the source string on
the row.

``absorbed_fraction`` [-] is the fraction of the attenuation that is ABSORBED
(turned into heat) rather than scattered. It is optional for the same reason
the thermal fields are: only a thermal run reads it. When it is not declared,
:data:`DEFAULT_ABSORBED_FRACTION` (1.0, all attenuation heats) is what a
heating calculation assumes, which makes the resulting ``Q`` an UPPER BOUND
wherever scattering is a real part of the loss. Every material shipped here
declares the value out loud instead of leaving it to that default, and says
in its source string whether the number is a measurement or an assumption.

``breast_default()`` is a verbatim port of the notebook's TISSUE_PROPS table
(v6-v12, unchanged): the acoustic numbers are pinned by tests and must not
drift silently, because they define what the existing dataset means. Its rows
now also carry the IT'IS thermal fields of the tissue each one names, which
adds nothing an acoustic run reads and lets the same table feed a thermal
run. ``breast_default_power_law()`` is its power-law sibling, built from
:data:`TISSUE_LIBRARY` rather than from the notebook literals.

:func:`tissue_db` is the shortest path from a segmentation to a run: name a
library tissue per label and get a :class:`MaterialDB` that
``Medium.from_id_map`` and ``ThermalMedium.from_labels`` both read, with no
hand-typed number anywhere in the chain.

The table is citable as a whole: :func:`tissue_library_stamp` returns
``"caustica-tissue-library/<version> sha256:<digest>"``, the version being a
declared release of the shipped rows and the digest a hash over every value
and citation in them. A result that records the stamp names the exact
properties it ran on; the suite pins the pair, so a changed number fails
until the version moves with it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, fields

from pydantic import Field, model_validator

from caustica.config.models import CausticaModel

#: 1 dB/cm = 100/(20*log10(e)) Np/m.
DB_CM_TO_NP_M = 100.0 / 8.685889638065035

#: What a heating calculation assumes when a material does not declare its
#: ``absorbed_fraction``: ALL of the attenuation is absorbed. The sign of that
#: assumption is stated once here and repeated wherever it is applied. It
#: over-states ``Q``, hence the temperature and the dose, by exactly the
#: scattered share of the loss, so a run that uses it is an upper bound and
#: never an unsafe under-estimate.
DEFAULT_ABSORBED_FRACTION = 1.0


def db_cm_to_np_m(alpha_db_cm: float) -> float:
    """Convert an attenuation in dB/cm to Np/m."""
    return alpha_db_cm * DB_CM_TO_NP_M


def np_m_to_db_cm(alpha_np_m: float) -> float:
    """Convert an attenuation in Np/m to dB/cm."""
    return alpha_np_m / DB_CM_TO_NP_M


class MixedAbsorptionExponentError(ValueError):
    """A material table carries more than one power-law exponent.

    Its own type because the fix is a decision, never a default: the solvers
    integrate ONE exponent per run (a single pair of fractional Laplacians),
    so a table that mixes water (y = 2) with soft tissue (y ~ 1.1) has to be
    told which exponent the run means.
    """


class Material(CausticaModel):
    """One acoustic (and optionally thermal) material.

    Give the absorption in exactly one form::

        Material(alpha0_db_cm_mhz_y=0.5, y=1.1, rho=911, c=1440, beta=6.0)
        Material(alpha_np_m=6.0, rho=932, c=1450, beta=4.5)   # legacy

    The second form is frequency independent by construction: it is the
    number a solver applies at every frequency it carries, which is why a
    file storing it also stores the frequency it was baked at.
    """

    name: str = ""
    alpha_np_m: float | None = Field(
        None,
        ge=0.0,
        description="Legacy absorption [Np/m], frequency independent (equivalent to y = 0)",
    )
    alpha0_db_cm_mhz_y: float | None = Field(
        None,
        ge=0.0,
        description="Power-law absorption prefactor [dB/(cm MHz^y)]; needs y",
    )
    y: float | None = Field(
        None,
        ge=0.0,
        le=3.0,
        description="Power-law absorption exponent [-]; needs alpha0_db_cm_mhz_y",
    )
    rho: float = Field(..., gt=0.0, description="Density [kg/m^3]")
    c: float = Field(..., gt=0.0, description="Sound speed [m/s]")
    beta: float = Field(..., ge=0.0, description="Nonlinearity coefficient (1 + B/2A)")
    # --- thermal fields (optional: an acoustic run does not read them) ---
    thermal_conductivity: float | None = Field(None, gt=0.0, description="k [W/m/K]")
    specific_heat: float | None = Field(None, gt=0.0, description="C [J/kg/K]")
    perfusion_rate: float | None = Field(None, ge=0.0, description="w_b [1/s]")
    absorbed_fraction: float | None = Field(
        None,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of the attenuation that is absorbed (heats the tissue) [-]; "
            "None means undeclared and a heating calculation assumes 1.0"
        ),
    )
    source: str = Field("", description="Where these numbers come from (free text)")

    @model_validator(mode="after")
    def _one_absorption_form(self) -> Material:
        legacy = self.alpha_np_m is not None
        power = self.alpha0_db_cm_mhz_y is not None or self.y is not None
        if power and (self.alpha0_db_cm_mhz_y is None or self.y is None):
            raise ValueError(
                "the power-law absorption form needs BOTH alpha0_db_cm_mhz_y "
                "[dB/(cm MHz^y)] and y; got "
                f"alpha0_db_cm_mhz_y={self.alpha0_db_cm_mhz_y!r}, y={self.y!r}"
            )
        if legacy and power:
            raise ValueError(
                "a material declares its absorption in ONE form: alpha_np_m (legacy, "
                "frequency independent) or alpha0_db_cm_mhz_y with y (power law). Both "
                "were given, and nothing can check that they agree, because they agree "
                "at one frequency only. Keep the power law and call "
                "material.at_frequency(f0) where a run needs the baked number."
            )
        if not legacy and not power:
            raise ValueError(
                "a material must declare its absorption: alpha0_db_cm_mhz_y=... with "
                "y=... (power law) or alpha_np_m=... (legacy, frequency independent). "
                "A lossless medium is alpha_np_m=0.0 stated explicitly."
            )
        return self

    # ---------- absorption ----------

    @property
    def absorption_model(self) -> str:
        """``"power_law"`` or ``"single_frequency"``."""
        return "single_frequency" if self.alpha0_db_cm_mhz_y is None else "power_law"

    @property
    def y_exponent(self) -> float:
        """The power-law exponent this material implies; legacy means 0.0."""
        return 0.0 if self.y is None else float(self.y)

    def alpha_np_m_at(self, f_hz: float) -> float:
        """Absorption [Np/m] at ``f_hz``.

        The legacy form ignores the frequency by definition and answers at
        any value, 0 Hz included. The power-law form evaluates
        ``alpha0 (f / 1 MHz)^y`` and converts dB/cm to Np/m, so it needs a
        positive frequency and refuses anything else rather than returning
        the 0 that ``0^y`` would give for a lossy tissue.
        """
        if self.alpha0_db_cm_mhz_y is None:
            return float(self.alpha_np_m)  # type: ignore[arg-type]
        if not f_hz > 0.0:
            raise ValueError(
                f"a power-law material needs a frequency > 0 Hz to answer with, got {f_hz}"
            )
        return db_cm_to_np_m(self.alpha0_db_cm_mhz_y * (f_hz / 1e6) ** float(self.y))

    @property
    def heating_fraction(self) -> float:
        """The absorbed fraction a heating calculation will use [-].

        The declared :attr:`absorbed_fraction`, or
        :data:`DEFAULT_ABSORBED_FRACTION` when the material does not declare
        one. Read :attr:`absorbed_fraction` instead when the question is
        whether the number was declared at all: this property cannot tell a
        measured 1.0 from an assumed one, and the provenance of that
        difference is the whole point of carrying the field.
        """
        return (
            DEFAULT_ABSORBED_FRACTION if self.absorbed_fraction is None else self.absorbed_fraction
        )

    def alpha_absorption_np_m_at(self, f_hz: float) -> float:
        """The part of ``alpha(f)`` [Np/m] that heats the tissue.

        ``heating_fraction * alpha_np_m_at(f_hz)``. This, not the attenuation,
        is the coefficient in ``Q = 2 alpha I``: the scattered share of the
        loss leaves the voxel as sound.
        """
        return self.heating_fraction * self.alpha_np_m_at(f_hz)

    def at_frequency(self, f_hz: float) -> Material:
        """This material in the legacy form, with alpha baked at ``f_hz``.

        A legacy material is returned unchanged. The power law is dropped
        from the copy on purpose: carrying both forms would let the two
        disagree at every frequency but one, and the law itself stays in the
        job file and in the table this copy was made from.
        """
        if self.alpha0_db_cm_mhz_y is None:
            return self
        return Material(
            name=self.name,
            alpha_np_m=self.alpha_np_m_at(f_hz),
            rho=self.rho,
            c=self.c,
            beta=self.beta,
            thermal_conductivity=self.thermal_conductivity,
            specific_heat=self.specific_heat,
            perfusion_rate=self.perfusion_rate,
            absorbed_fraction=self.absorbed_fraction,
            source=self.source,
        )


class MaterialDB(CausticaModel):
    """Integer tissue-id -> :class:`Material` mapping (JSON-serializable)."""

    materials: dict[int, Material]

    def __getitem__(self, tissue_id: int) -> Material:
        return self.materials[tissue_id]

    def __contains__(self, tissue_id: int) -> bool:
        return tissue_id in self.materials

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.materials))

    @property
    def has_power_law(self) -> bool:
        """True when any material declares its absorption as a power law."""
        return any(m.absorption_model == "power_law" for m in self.materials.values())

    def y(self) -> float:
        """The one power-law exponent this table implies.

        Raises :class:`MixedAbsorptionExponentError`, naming every offending
        material, when the table carries more than one. A legacy material
        counts as ``y = 0``.
        """
        by_y: dict[float, list[str]] = {}
        for tissue_id in self.ids:
            m = self.materials[tissue_id]
            by_y.setdefault(m.y_exponent, []).append(f"id {tissue_id} ({m.name or 'unnamed'})")
        if len(by_y) == 1:
            return next(iter(by_y))
        listing = "; ".join(f"y = {y:g}: {', '.join(names)}" for y, names in sorted(by_y.items()))
        raise MixedAbsorptionExponentError(
            f"this material table mixes absorption exponents ({listing}). A run carries "
            f"ONE exponent, so it has to be named: set the job's absorption.y, or give "
            f"every material the same y. A legacy material (alpha_np_m only) counts as "
            f"y = 0, frequency independent."
        )

    def at_frequency(self, f_hz: float) -> MaterialDB:
        """This table with every power law baked to Np/m at ``f_hz``."""
        return MaterialDB(materials={i: m.at_frequency(f_hz) for i, m in self.materials.items()})


def _library_thermal(key: str, acoustic_source: str) -> dict[str, object]:
    """Thermal fields, absorbed fraction and a two-part source string.

    The acoustic half of a table can come from anywhere (the notebook rows,
    a user file); the thermal half is always a named IT'IS row. Composing the
    source here keeps the two halves separable in the string a result file
    stores.

    The borrowed values are the database row's own, including ``w_b``, which
    the database's heat transfer rate fixes together with the density the
    database quotes for that row. That density is in the string and it is not
    always the acoustic density beside it: the notebook's Muscle is
    1050 kg/m^3 against the row's 1090.4, its Fat 932 against 911. Keeping
    the row's ``w_b`` keeps the volumetric perfusion the database measured;
    recomputing it at the acoustic density would move it 3.7 % and 2.3 %
    away from the database, so the borrowed number is deliberate and the
    string says which density it was derived at.
    """
    t = TISSUE_LIBRARY[key]
    rest = t.thermal_provenance()
    return {
        **t._extra_material_fields(),
        "source": f"{acoustic_source}; {rest}" if rest else acoustic_source,
    }


def water(
    alpha_np_m: float = 0.0,
    c: float = 1500.0,
    rho: float = 1000.0,
    beta: float = 0.0,
    name: str = "water",
) -> Material:
    """Water; defaults are the LINEAR LOSSLESS validation medium.

    Note: beta defaults to 0.0 (linear) on purpose, because this is the
    medium the O'Neil/Rayleigh validation chain assumes. Physical water would
    be beta=3.5; pass it explicitly when you mean nonlinear water.

    The thermal fields are the IT'IS ``Water`` row, so this material can also
    be the coupling bath of a thermal run. Its perfusion is 0.0 stated out
    loud, which is what a non-perfused medium declares; a missing field would
    be refused by :class:`~caustica.thermal.properties.ThermalMedium`.

    The ``source`` string quotes the arguments this call actually used, not
    the defaults, so a caller that passes ``beta=3.5`` or a non-zero alpha
    carries a citation that matches the numbers it got. The acoustic density
    stays the caller's (1000 kg/m^3 by default) while the thermal half is the
    database row at 994 kg/m^3; both densities appear in the string.
    """
    return Material(
        name=name,
        alpha_np_m=alpha_np_m,
        rho=rho,
        c=c,
        beta=beta,
        **_library_thermal(  # type: ignore[arg-type]
            "water_37c",
            # Built from the arguments, never from a literal: four of the five
            # shipped callers pass their own alpha, beta or c, and a source
            # string quoting the defaults would be a false citation on exactly
            # the materials whose provenance matters most.
            f"acoustic: caustica water validation medium (alpha {alpha_np_m:g} Np/m, "
            f"rho {rho:g} kg/m^3, c {c:g} m/s, beta {beta:g})",
        ),
    )


def breast_default() -> MaterialDB:
    """The notebook's breast-phantom tissue table (TISSUE_PROPS), verbatim.

    ids: 0=PML/matching, 1=skin, 2=fat/glandular, 3=muscle, 4=coupling gel.
    Legacy form throughout: these five acoustic rows ARE what the existing
    dataset means, and they are pinned to the digit by test.

    Each row also carries the thermal fields and the absorbed fraction of one
    library row, so the same table feeds a thermal run. Which row, and why:
    ids 0 and 4 take ``water_37c``, id 1 ``skin``, id 3 ``muscle``, and id 2,
    which the notebook calls fat/glandular, takes the ``fat`` endpoint,
    because its acoustic numbers (alpha 6 Np/m, rho 932, c 1450) are the fat
    end of that mixture and not the glandular one. The choice is visible in
    the dose: fat perfusion is 7.136e-4 1/s against fibroglandular
    2.601e-3 1/s, a factor 3.65, and the perfusion-limited steady rise is
    ``Q / (w_b rho_b C_b)``, so a caller who means glandular tissue should
    build the table from :func:`tissue_db` with ``"fibroglandular"`` rather
    than reuse this one. No acoustic number moves: a solver reads alpha, rho,
    c and beta and nothing else.
    """
    notebook = "acoustic: notebook TISSUE_PROPS v6-v12, verbatim"
    #: id -> (name, alpha [Np/m], rho, c, beta, library key for the thermal half)
    rows = {
        0: ("PML", 0.1, 1000.0, 1500.0, 3.5, "water_37c"),
        1: ("Skin", 15.0, 1109.0, 1600.0, 4.0, "skin"),
        2: ("Fat", 6.0, 932.0, 1450.0, 4.5, "fat"),
        3: ("Muscle", 10.0, 1050.0, 1580.0, 4.5, "muscle"),
        4: ("Gel", 0.1, 1000.0, 1500.0, 3.5, "water_37c"),
    }
    return MaterialDB(
        materials={
            i: Material(
                name=name,
                alpha_np_m=alpha,
                rho=rho,
                c=c,
                beta=beta,
                **_library_thermal(key, notebook),  # type: ignore[arg-type]
            )
            for i, (name, alpha, rho, c, beta, key) in rows.items()
        }
    )


def breast_default_power_law(which: str = "mid") -> MaterialDB:
    """The same five ids as :func:`breast_default`, in power-law form.

    Built from :data:`TISSUE_LIBRARY` (literature values with their sources
    and their thermal properties), not from the notebook literals: the
    matching layer and the coupling gel are water, and skin, fat and muscle
    are the library rows.

    The exponents deliberately differ (water 2.0, skin and muscle 1.0, fat
    1.1), so :meth:`MaterialDB.y` refuses this table and a run using it has
    to name its own exponent. That is the state of the literature, not an
    oversight.
    """
    return MaterialDB(
        materials={
            0: TISSUE_LIBRARY["water_37c"].to_power_law_material(which),
            1: TISSUE_LIBRARY["skin"].to_power_law_material(which),
            2: TISSUE_LIBRARY["fat"].to_power_law_material(which),
            3: TISSUE_LIBRARY["muscle"].to_power_law_material(which),
            4: TISSUE_LIBRARY["water_37c"].to_power_law_material(which),
        }
    )


def tissue_db(
    names: Mapping[int, str],
    *,
    which: str = "mid",
    f0_hz: float | None = None,
) -> MaterialDB:
    """A :class:`MaterialDB` for a label volume, out of :data:`TISSUE_LIBRARY`.

    ``names`` maps each integer label to a library key::

        db = tissue_db({0: "water_37c", 1: "skin", 2: "liver"}, f0_hz=1e6)
        medium = Medium.from_id_map(labels, db)
        tmed = ThermalMedium.from_medium(medium, db, grid.dx)

    Every material it returns carries the acoustic four, the thermal three,
    the absorbed fraction and the citation of each, so a chain built this way
    has no hand-typed number in it. That is the point: a table typed into a
    script is a table with no provenance, and the thermal half is exactly
    where an invented number is invisible until it changes a dose.

    Parameters
    ----------
    names:
        label -> library key. An unknown key is refused with the available
        keys listed; guessing the nearest name would silently simulate a
        different tissue.
    which:
        ``"lo"``, ``"mid"`` (default) or ``"hi"``: which end of each library
        row's reported spread to take.
    f0_hz:
        When given, every material is baked to the legacy single-frequency
        form at this frequency (:meth:`Material.at_frequency`), which is what
        the k-space solvers accept today. Left out, the table stays in
        power-law form and a run using it has to name its own exponent.
    """
    unknown = sorted({key for key in names.values() if key not in TISSUE_LIBRARY})
    if unknown:
        raise KeyError(
            f"tissue_db: no such tissue in TISSUE_LIBRARY: {unknown}. "
            f"Known keys: {sorted(TISSUE_LIBRARY)}. Refusing to guess which tissue "
            f"was meant; pass a Material of your own for anything the library does "
            f"not carry."
        )
    db = MaterialDB(
        materials={
            int(label): TISSUE_LIBRARY[key].to_power_law_material(which)
            for label, key in names.items()
        }
    )
    return db if f0_hz is None else db.at_frequency(f0_hz)


BACKGROUND_ID = 4
PML_ID = 0


# --------------------------------------------------------------------------
# Literature acoustic tissue values. MOVED verbatim from the
# phantom package's tissue table: these numbers are generic soft-tissue
# literature, not that source's (an electromagnetic repository shipping no
# acoustic values). The source's media-number -> tissue-class mapping and
# its interpolated sub-group ramp stay with the phantom package; only the
# literature-anchored endpoints live here. A moved value that changes is a
# bug (pinned to the digit by test).
# --------------------------------------------------------------------------


def perfusion_ml_min_kg_to_per_s(ml_min_kg: float, rho_kg_m3: float) -> float:
    """IT'IS heat-transfer rate [mL blood/(min kg tissue)] -> Pennes w_b [1/s].

    ``w_b`` is a volume of blood per volume of tissue per second, so the
    per-kilogram rate is multiplied by the tissue density and divided by
    1e6 mL/m^3 and by 60 s/min. Skin, for example: 106.3813131 mL/min/kg at
    1109 kg/m^3 gives 1.9663e-3 1/s.
    """
    return ml_min_kg * rho_kg_m3 * 1e-6 / 60.0


@dataclass(frozen=True)
class AcousticTissue:
    """One acoustic tissue: nominal values plus their literature spread.

    Every acoustic quantity is a ``(low, high)`` pair, the reported spread
    rather than a guess bracket. The nominal value is the midpoint; per-voxel
    data (like a repository's ``p``) can blend ``low + p*(high - low)``.
    Attenuation is stored in power-law form ``alpha0 * f^b``
    [dB/(cm MHz^b)] and evaluated at an export frequency, because storing
    only Np/m would silently pin a medium to one frequency.

    The thermal fields are single values (the databases quote one), in the
    units :class:`Material` uses: k [W/m/K], C [J/kg/K], w_b [1/s].
    """

    name: str
    c: tuple[float, float]  # sound speed [m/s]
    rho: tuple[float, float]  # density [kg/m^3]
    alpha0: tuple[float, float]  # attenuation prefactor [dB/(cm MHz^b)]
    b: float  # attenuation power-law exponent
    bona: tuple[float, float]  # nonlinearity parameter B/A
    source: str
    interpolated: bool = False
    thermal_conductivity: float | None = None  # k [W/m/K]
    specific_heat: float | None = None  # C [J/kg/K]
    perfusion_rate: float | None = None  # w_b [1/s]
    thermal_source: str = ""
    #: Absorbed share of the attenuation [-]. ``None`` means UNDECLARED and
    #: travels to the material as ``None``: a row that says nothing about
    #: scattering must not arrive on the other side looking like a row that
    #: declared 1.0, because the whole use of the field is telling the two
    #: apart. Every row this module ships declares it.
    absorbed_fraction: float | None = None
    absorption_source: str = ""

    def alpha_np_m(self, f0_hz: float, which: str = "mid") -> float:
        """Absorption [Np/m] at ``f0_hz`` from the power law."""
        f_mhz = f0_hz / 1e6
        a0 = self._pick(self.alpha0, which)
        return db_cm_to_np_m(a0 * f_mhz**self.b)

    def beta(self, which: str = "mid") -> float:
        """Nonlinearity coefficient ``1 + B/2A``."""
        return 1.0 + 0.5 * self._pick(self.bona, which)

    @staticmethod
    def _pick(pair: tuple[float, float], which: str) -> float:
        if which == "lo":
            return pair[0]
        if which == "hi":
            return pair[1]
        if which == "mid":
            return 0.5 * (pair[0] + pair[1])
        raise ValueError(f"which must be 'lo', 'mid' or 'hi', got {which!r}")

    def thermal_provenance(self) -> str:
        """Where the thermal fields and the absorbed fraction come from.

        Separate from :attr:`source`, which is the acoustic provenance, so a
        table whose acoustic numbers come from somewhere else (the notebook
        rows of :func:`breast_default`, say) can borrow these values and still
        say where each half was measured.
        """
        parts = []
        if self.thermal_source:
            parts.append(f"thermal: {self.thermal_source}")
        if self.absorption_source:
            parts.append(f"absorbed fraction: {self.absorption_source}")
        return "; ".join(parts)

    def _provenance(self) -> str:
        rest = self.thermal_provenance()
        return f"{self.source}; {rest}" if rest else self.source

    def _extra_material_fields(self) -> dict[str, float | None]:
        """The fields a thermal run reads, carried onto the ``Material``."""
        return {
            "thermal_conductivity": self.thermal_conductivity,
            "specific_heat": self.specific_heat,
            "perfusion_rate": self.perfusion_rate,
            "absorbed_fraction": self.absorbed_fraction,
        }

    def to_material(self, f0_hz: float, which: str = "mid") -> Material:
        """A legacy-form material with alpha baked at ``f0_hz``."""
        return Material(
            name=self.name,
            alpha_np_m=self.alpha_np_m(f0_hz, which),
            rho=self._pick(self.rho, which),
            c=self._pick(self.c, which),
            beta=self.beta(which),
            source=self._provenance(),
            **self._extra_material_fields(),
        )

    def to_power_law_material(self, which: str = "mid") -> Material:
        """A power-law material: no frequency is baked in."""
        return Material(
            name=self.name,
            alpha0_db_cm_mhz_y=self._pick(self.alpha0, which),
            y=self.b,
            rho=self._pick(self.rho, which),
            c=self._pick(self.c, which),
            beta=self.beta(which),
            source=self._provenance(),
            **self._extra_material_fields(),
        )


_DUCK = "Duck, Physical Properties of Tissue (1990)"
_ITIS = "IT'IS Foundation Tissue Properties Database (acoustic)"
#: The exact release every thermal number below is quoted from, read from the
#: shipped ASCII table rather than from a secondary citation.
ITIS_DB = "IT'IS Tissue Properties Database V4.2 (2024-06-04, DOI 10.13099/VIP21000-04-2)"

#: Blood density [kg/m^3] as the same release quotes it. The Pennes perfusion
#: sink needs ``rho_b C_b`` and the solver carries its own defaults; these two
#: constants are the table's own numbers, so the two can be compared by test
#: rather than by reading two docstrings.
ITIS_BLOOD_DENSITY = 1049.75
#: Blood specific heat [J/kg/K], same release, same row.
ITIS_BLOOD_SPECIFIC_HEAT = 3617.0

#: What the shipped soft-tissue rows declare for the absorbed fraction, and
#: why it is 1.0. This is an ASSUMPTION and says so: no absorption-to-
#: attenuation ratio is quoted per tissue by the databases these rows come
#: from (IT'IS states outright that it makes no distinction between the two),
#: so inventing a per-tissue split would be a number with no measurement
#: behind it. 1.0 is the conservative end of the range: it turns every decibel
#: of loss into heat, so Q, the temperature and the dose are upper bounds.
#: A user with a measured ratio for their tissue sets it per material.
ASSUMED_ABSORBED_FRACTION_SOURCE = (
    "assumption, not a measurement: all attenuation is taken to be absorbed "
    "(absorbed fraction 1.0), because no per-tissue absorption-to-attenuation "
    "ratio is quoted by the sources these acoustic values come from. It makes "
    "the heating an upper bound; set absorbed_fraction from a measurement to "
    "lower it."
)
#: Water is the one row where 1.0 is a fact rather than an assumption.
WATER_ABSORBED_FRACTION_SOURCE = (
    "1.0 by construction: a homogeneous liquid has no scatterers, so the "
    "attenuation of degassed water IS its absorption"
)


def _itis_thermal(
    tissue: str, k: float, cp: float, htr_ml_min_kg: float, rho: float
) -> dict[str, object]:
    """The three thermal values plus the source string for one IT'IS row."""
    return {
        "thermal_conductivity": k,
        "specific_heat": cp,
        "perfusion_rate": perfusion_ml_min_kg_to_per_s(htr_ml_min_kg, rho),
        "thermal_source": (
            f"{ITIS_DB}, row '{tissue}': k = {k:g} W/m/K, C = {cp:g} J/kg/K, heat "
            f"transfer rate = {htr_ml_min_kg:g} mL/min/kg at rho = {rho:g} kg/m^3"
        ),
    }


#: Literature-anchored tissues (names kept verbatim from the pre-split table,
#: because the name travels inside exported MaterialDB JSON and renaming
#: would be a value change). ``fibroglandular`` and ``fat`` are the
#: highest-water and lowest-water ENDPOINTS the phantom package's sub-group
#: ramp interpolates between; the interpolated in-between rows are its
#: modelling choice.
TISSUE_LIBRARY: dict[str, AcousticTissue] = {
    # Degassed water at body temperature, the standard HIFU coupling bath.
    # Absorption is tiny but NOT zero.
    "water_37c": AcousticTissue(
        name="Coupling medium (water, 37 C)",
        c=(1515.0, 1530.0),
        rho=(992.0, 995.0),
        # 37 C, not the textbook 20 C figure. Water absorption falls with
        # temperature: alpha/f^2 is ~25e-15 Np m^-1 s^2 at 20 C (the familiar
        # 0.0022 dB/(cm MHz^2)) but ~16-18e-15 at body temperature. The c and
        # rho above are already the 37 C values, so quoting the 20 C alpha
        # made this row ~1.6x too lossy for the temperature it names (review
        # finding, 2026-08-18).
        alpha0=(0.0014, 0.0017),
        b=2.0,
        bona=(5.0, 5.4),
        source=f"{_DUCK}, ch. 4 (water B/A = 5.2; alpha(37 C) ~ 0.0015 f^2 dB/cm)",
        **_itis_thermal("Water", 0.6045, 4178.0, 0.0, 994.035466),  # type: ignore[arg-type]
        absorbed_fraction=1.0,
        absorption_source=WATER_ABSORBED_FRACTION_SOURCE,
    ),
    # Skin (~1.5 mm dermis layer).
    "skin": AcousticTissue(
        name="Skin",
        c=(1595.0, 1660.0),
        rho=(1085.0, 1125.0),
        alpha0=(2.20, 3.50),
        b=1.0,
        bona=(7.5, 8.3),
        source=f"{_ITIS} (skin c=1624, rho=1109); {_DUCK} (alpha 2.2-3.5 dB/cm/MHz, B/A 7.9)",
        **_itis_thermal("Skin", 0.3721835, 3390.5, 106.3813131, 1109.0),  # type: ignore[arg-type]
        absorbed_fraction=1.0,
        absorption_source=ASSUMED_ABSORBED_FRACTION_SOURCE,
    ),
    # Pectoral muscle (chest wall).
    "muscle": AcousticTissue(
        name="Muscle (chest wall)",
        c=(1545.0, 1600.0),
        rho=(1050.0, 1095.0),
        alpha0=(0.57, 1.09),
        b=1.0,
        bona=(7.0, 7.8),
        source=f"{_ITIS} (c=1588, rho=1090); {_DUCK} "
        "(alpha 0.57 along / 1.09 across fibres, B/A 7.4)",
        **_itis_thermal("Muscle", 0.49496875, 3421.2, 36.7382931, 1090.4),  # type: ignore[arg-type]
        absorbed_fraction=1.0,
        absorption_source=ASSUMED_ABSORBED_FRACTION_SOURCE,
    ),
    # Fibroconnective/glandular tissue at its HIGHEST water content: the
    # glandular endpoint (literature-anchored).
    "fibroglandular": AcousticTissue(
        name="Fibroglandular-1 (highest water)",
        c=(1530.0, 1580.0),
        rho=(1035.0, 1060.0),
        alpha0=(0.85, 1.15),
        b=1.1,
        bona=(7.0, 8.0),
        source=f"{_ITIS} breast gland (c=1505, rho=1041); Duric et al. UST breast (c 1500-1580)",
        **_itis_thermal("Breast Gland", 0.3345, 2960.0, 150.0, 1040.5),  # type: ignore[arg-type]
        absorbed_fraction=1.0,
        absorption_source=ASSUMED_ABSORBED_FRACTION_SOURCE,
    ),
    # Fat at its LOWEST water content, essentially lipid: the fat endpoint
    # (literature-anchored).
    "fat": AcousticTissue(
        name="Fatty-3 (lowest water, lipid)",
        c=(1425.0, 1460.0),
        rho=(900.0, 930.0),
        alpha0=(0.40, 0.60),
        b=1.1,
        bona=(9.6, 11.3),
        source=f"{_ITIS} breast fat (c=1440, rho=911); {_DUCK} (fat B/A 9.6-11.3, alpha 0.4-0.6)",
        **_itis_thermal("Breast Fat", 0.209, 2348.333333, 47.0, 911.0),  # type: ignore[arg-type]
        absorbed_fraction=1.0,
        absorption_source=ASSUMED_ABSORBED_FRACTION_SOURCE,
    ),
    # The three organ rows below are IT'IS V4.2 throughout, acoustic and
    # thermal: sound speed, density and B/A are the database's min/max
    # spread, and the attenuation prefactor is its alpha0 [Np/m/MHz^b]
    # converted to dB/cm (no spread is reported, so the pair is one value
    # twice).
    "liver": AcousticTissue(
        name="Liver",
        c=(1541.5, 1611.0),
        rho=(1050.0, 1158.0),
        alpha0=(np_m_to_db_cm(6.915),) * 2,
        b=1.0,
        bona=(6.54, 8.72),
        source=f"{ITIS_DB}, row 'Liver' (c 1541.5-1611 m/s, rho 1050-1158 kg/m^3, "
        f"alpha0 6.915 Np/m/MHz, b = 1, B/A 6.54-8.72)",
        **_itis_thermal("Liver", 0.519111111, 3540.2, 860.456666, 1078.75),  # type: ignore[arg-type]
        absorbed_fraction=1.0,
        absorption_source=ASSUMED_ABSORBED_FRACTION_SOURCE,
    ),
    "brain": AcousticTissue(
        name="Brain (average)",
        c=(1506.0, 1565.0),
        rho=(1041.0, 1050.0),
        alpha0=(np_m_to_db_cm(6.8032),) * 2,
        b=1.3,
        bona=(6.55, 7.05),
        source=f"{ITIS_DB}, row 'Brain' (c 1506-1565 m/s, rho 1041-1050 kg/m^3, "
        f"alpha0 6.8032 Np/m/MHz^1.3, b = 1.3, B/A 6.55-7.05)",
        **_itis_thermal("Brain", 0.51325, 3630.0, 558.6063123, 1045.5),  # type: ignore[arg-type]
        absorbed_fraction=1.0,
        absorption_source=ASSUMED_ABSORBED_FRACTION_SOURCE,
    ),
    "blood": AcousticTissue(
        name="Blood",
        c=(1559.2, 1590.0),
        rho=(1025.0, 1060.0),
        alpha0=(np_m_to_db_cm(2.3676),) * 2,
        b=1.0498,
        bona=(6.0, 6.3),
        source=f"{ITIS_DB}, row 'Blood' (c 1559.2-1590 m/s, rho 1025-1060 kg/m^3, "
        f"alpha0 2.3676 Np/m/MHz^1.0498, b = 1.0498, B/A 6.0-6.3)",
        # IT'IS quotes 10000 mL/min/kg for blood itself, the sentinel that
        # says "this voxel IS the perfusing fluid": in Pennes it pins a blood
        # voxel to the arterial temperature, which is what a large vessel
        # does to a nearby focus.
        **_itis_thermal(  # type: ignore[arg-type]
            "Blood",
            0.516857143,
            ITIS_BLOOD_SPECIFIC_HEAT,
            10000.0,
            ITIS_BLOOD_DENSITY,
        ),
        absorbed_fraction=1.0,
        absorption_source=ASSUMED_ABSORBED_FRACTION_SOURCE,
    ),
}


#: The version of the shipped tissue table, as a whole. It moves when ANY
#: number, exponent or source string in :data:`TISSUE_LIBRARY` moves, which is
#: what makes the table separately citable: a result that records this string
#: names the exact set of properties it ran on, and two results carrying
#: different stamps ran on different tissue.
#:
#: The rule that keeps it honest is a test, not a habit:
#: ``tests/test_thermal_bridge.py`` pins the version together with
#: :func:`tissue_library_digest`, so a changed number fails the suite until
#: the version is bumped in the same commit.
TISSUE_LIBRARY_VERSION = "caustica-tissue-library/1"


def tissue_library_digest() -> str:
    """A short content hash over every value and citation in the table.

    The digest, not the version, is what notices a change: the version is a
    human decision and a human can forget to move it, while this is computed
    from the rows themselves. Pinned beside
    :data:`TISSUE_LIBRARY_VERSION` in the suite so the two cannot drift.

    Every field is included, sources among them: a value whose citation
    silently changed is a different claim about the same number. The fields
    are read from the dataclass rather than listed here, so a field added to
    :class:`AcousticTissue` later enters the digest by itself; a list would
    have to be remembered, and the one change it would be forgotten on is a
    new property nobody hashed.
    """
    h = hashlib.sha256()
    for key in sorted(TISSUE_LIBRARY):
        t = TISSUE_LIBRARY[key]
        h.update(key.encode())
        for f in fields(t):
            h.update(f"|{f.name}={getattr(t, f.name)!r}".encode())
    return h.hexdigest()[:16]


def tissue_library_stamp() -> str:
    """The one string a result file records to name this table.

    ``"caustica-tissue-library/1 sha256:...."``. A run that stores it can be
    replayed against the same properties years later, and a run that stores
    nothing cannot: which tissue table produced a dose is not recoverable
    from the dose.
    """
    return f"{TISSUE_LIBRARY_VERSION} sha256:{tissue_library_digest()}"
