"""Acoustic material definitions and the id -> material database.

The acoustic property set matches the production notebook exactly:
``alpha`` [Np/m] (frequency-independent exponential absorption in v1),
``rho`` [kg/m^3], ``c`` [m/s], ``beta`` [-] (nonlinearity coefficient,
beta = 1 + B/2A). Thermal fields are declared now (optional) so the M18
thermal module will not need a schema migration, but nothing reads them yet.

``breast_default()`` is a verbatim port of the notebook's TISSUE_PROPS table
(v6-v12, unchanged): the numbers are pinned by tests and must not drift
silently — they define what the existing dataset means.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field

from caustica.config.models import CausticaModel


class Material(CausticaModel):
    """One acoustic (and later thermal) material."""

    name: str = ""
    alpha_np_m: float = Field(..., ge=0.0, description="Absorption [Np/m] at f0 (v1 model)")
    rho: float = Field(..., gt=0.0, description="Density [kg/m^3]")
    c: float = Field(..., gt=0.0, description="Sound speed [m/s]")
    beta: float = Field(..., ge=0.0, description="Nonlinearity coefficient (1 + B/2A)")
    # --- M18 thermal hooks (declared, unused in v1) ---
    thermal_conductivity: float | None = Field(None, description="[W/m/K] (M18)")
    specific_heat: float | None = Field(None, description="[J/kg/K] (M18)")
    perfusion_rate: float | None = Field(None, description="[1/s] (M18)")


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


def water(
    alpha_np_m: float = 0.0,
    c: float = 1500.0,
    rho: float = 1000.0,
    beta: float = 0.0,
    name: str = "water",
) -> Material:
    """Water; defaults are the LINEAR LOSSLESS validation medium.

    Note: beta defaults to 0.0 (linear) on purpose — this is the medium the
    O'Neil/Rayleigh validation chain assumes. Physical water would be
    beta=3.5; pass it explicitly when you mean nonlinear water.
    """
    return Material(name=name, alpha_np_m=alpha_np_m, rho=rho, c=c, beta=beta)


def breast_default() -> MaterialDB:
    """The notebook's breast-phantom tissue table (TISSUE_PROPS), verbatim.

    ids: 0=PML/matching, 1=skin, 2=fat/glandular, 3=muscle, 4=coupling gel.
    """
    t = {
        0: Material(name="PML", alpha_np_m=0.1, rho=1000.0, c=1500.0, beta=3.5),
        1: Material(name="Skin", alpha_np_m=15.0, rho=1109.0, c=1600.0, beta=4.0),
        2: Material(name="Fat", alpha_np_m=6.0, rho=932.0, c=1450.0, beta=4.5),
        3: Material(name="Muscle", alpha_np_m=10.0, rho=1050.0, c=1580.0, beta=4.5),
        4: Material(name="Gel", alpha_np_m=0.1, rho=1000.0, c=1500.0, beta=3.5),
    }
    return MaterialDB(materials=t)


BACKGROUND_ID = 4
PML_ID = 0


# --------------------------------------------------------------------------
# Literature acoustic tissue values (M10k/W0b, D17). MOVED verbatim from the
# phantom package's tissue table — these numbers are generic soft-tissue
# literature, not that source's (an electromagnetic repository shipping no
# acoustic values). The source's media-number -> tissue-class mapping and
# its interpolated sub-group ramp stay with the phantom package; only the
# literature-anchored endpoints live here. A moved value that changes is a
# bug (pinned to the digit by test).
# --------------------------------------------------------------------------

#: 1 dB/cm = 100/(20*log10(e)) Np/m.
DB_CM_TO_NP_M = 100.0 / 8.685889638065035


def db_cm_to_np_m(alpha_db_cm: float) -> float:
    """Convert an attenuation in dB/cm to Np/m."""
    return alpha_db_cm * DB_CM_TO_NP_M


def np_m_to_db_cm(alpha_np_m: float) -> float:
    return alpha_np_m / DB_CM_TO_NP_M


@dataclass(frozen=True)
class AcousticTissue:
    """One acoustic tissue: nominal values plus their literature spread.

    Every quantity is a ``(low, high)`` pair — the reported spread, not a
    guess bracket. The nominal value is the midpoint; per-voxel data (like a
    repository's ``p``) can blend ``low + p*(high - low)``. Attenuation is
    stored in power-law form ``alpha0 * f^b`` [dB/(cm MHz^b)] and evaluated
    at an export frequency — storing only Np/m would silently pin a medium
    to one frequency.
    """

    name: str
    c: tuple[float, float]  # sound speed [m/s]
    rho: tuple[float, float]  # density [kg/m^3]
    alpha0: tuple[float, float]  # attenuation prefactor [dB/(cm MHz^b)]
    b: float  # attenuation power-law exponent
    bona: tuple[float, float]  # nonlinearity parameter B/A
    source: str
    interpolated: bool = False

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

    def to_material(self, f0_hz: float, which: str = "mid") -> Material:
        return Material(
            name=self.name,
            alpha_np_m=self.alpha_np_m(f0_hz, which),
            rho=self._pick(self.rho, which),
            c=self._pick(self.c, which),
            beta=self.beta(which),
        )


_DUCK = "Duck, Physical Properties of Tissue (1990)"
_ITIS = "IT'IS Foundation Tissue Properties Database (acoustic)"

#: Literature-anchored tissues (names kept verbatim from the pre-split table
#: — the name travels inside exported MaterialDB JSON, so renaming would be a
#: value change). ``fibroglandular`` and ``fat`` are the highest-water and
#: lowest-water ENDPOINTS the phantom package's sub-group ramp interpolates
#: between; the interpolated in-between rows are its modelling choice.
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
    ),
    # Fibroconnective/glandular tissue at its HIGHEST water content — the
    # glandular endpoint (literature-anchored).
    "fibroglandular": AcousticTissue(
        name="Fibroglandular-1 (highest water)",
        c=(1530.0, 1580.0),
        rho=(1035.0, 1060.0),
        alpha0=(0.85, 1.15),
        b=1.1,
        bona=(7.0, 8.0),
        source=f"{_ITIS} breast gland (c=1505, rho=1041); Duric et al. UST breast (c 1500-1580)",
    ),
    # Fat at its LOWEST water content, essentially lipid — the fat endpoint
    # (literature-anchored).
    "fat": AcousticTissue(
        name="Fatty-3 (lowest water, lipid)",
        c=(1425.0, 1460.0),
        rho=(900.0, 930.0),
        alpha0=(0.40, 0.60),
        b=1.1,
        bona=(9.6, 11.3),
        source=f"{_ITIS} breast fat (c=1440, rho=911); {_DUCK} (fat B/A 9.6-11.3, alpha 0.4-0.6)",
    ),
}
