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
