"""ThermalMedium: dense per-voxel thermal property volumes.

Why a parallel container instead of four more volumes on
:class:`~caustica.medium.Medium`
--------------------------------------------------------
``Medium`` is documented as *the four property volumes the solvers consume*
(alpha, rho, c, beta) and every acoustic path — the k-space engine, the
k-Wave adapter, the checkpoint fingerprint that hashes exactly those four
arrays, the planner's VRAM inventory — is built on that being the whole set.
Bolting optional thermal volumes onto it would put four maps that no
acoustic solver reads inside every acoustic run's memory budget, and would
make ``Medium`` half-valid whenever the material table has no thermal
fields (which is the normal case: they are ``Optional`` on ``Material`` and
most tables leave them empty).

So the thermal solve gets its own container, built the same way, with the
same two constructors and the same refusal-to-guess rule — plus a third,
:meth:`ThermalMedium.from_medium`, which reuses the id map an acoustic
``Medium`` already carries so the two media cannot describe different
tissue layouts.

One deliberate difference from ``Medium``: a ``ThermalMedium`` carries
``dx``. The acoustic solvers always receive a ``Grid`` next to the medium,
but the Pennes solver's public API is ``solve(T0, Q, medium, dt, n_steps)``
— a finite-difference stencil cannot be built without the spacing, so the
spacing belongs to the object that owns the stencil's coefficients.

Units: k [W/m/K], rho [kg/m^3], specific heat [J/kg/K], perfusion [1/s]
(volume of blood per volume of tissue per second, the ``w_b`` of Pennes).
"""

from __future__ import annotations

import numpy as np

from caustica.materials import Material, MaterialDB

#: The three ``Material`` fields a thermal solve needs, and what they mean.
REQUIRED_THERMAL_FIELDS: dict[str, str] = {
    "thermal_conductivity": "k [W/m/K]",
    "specific_heat": "C [J/kg/K]",
    "perfusion_rate": "w_b [1/s]",
}


class ThermalPropertyError(ValueError):
    """A material cannot answer a thermal solve's question about itself.

    Its own type because the fix is always in the material table, never in
    the solve: some tissue in the medium has no thermal properties, and
    inventing them (k of water? perfusion of zero?) would silently decide
    the answer the run exists to compute.
    """


def _missing_thermal_fields(material: Material) -> list[str]:
    return [name for name in REQUIRED_THERMAL_FIELDS if getattr(material, name) is None]


def _require_thermal(material: Material, tissue_id: int | None = None) -> None:
    """Refuse a material that has not declared its thermal properties.

    Mirrors the M6f frequency guard: name the thing, name the missing field,
    say what silently using a default would cost, and give the fix.
    """
    missing = _missing_thermal_fields(material)
    if not missing:
        return
    where = f"tissue id {tissue_id} " if tissue_id is not None else ""
    fields = ", ".join(f"{name} ({REQUIRED_THERMAL_FIELDS[name]})" for name in missing)
    raise ThermalPropertyError(
        f"material {where}{material.name!r} is missing the thermal field(s): {fields}. "
        f"They are Optional on Material because an acoustic run does not read them — "
        f"a thermal run does, and guessing them would decide the temperature this "
        f"solve exists to compute (k sets how fast the focus spreads, C how much "
        f"energy one degree costs, w_b how much blood carries away). Set them on the "
        f"material, e.g. Material(..., thermal_conductivity=0.52, specific_heat=3600, "
        f"perfusion_rate=0.0). A non-perfused medium (water, gel, an ex-vivo phantom) "
        f"is perfusion_rate=0.0 stated explicitly, not a missing field."
    )


class ThermalMedium:
    """Per-voxel k / rho / C / w_b volumes plus the grid spacing.

    All volumes are float32, C-contiguous and in HOST memory; the solver
    uploads them once at the start of a run (same policy as ``Medium``).
    """

    def __init__(
        self,
        k: np.ndarray,
        rho: np.ndarray,
        specific_heat: np.ndarray,
        perfusion: np.ndarray,
        dx: float,
        id_map: np.ndarray | None = None,
    ):
        vols = {
            "k": k,
            "rho": rho,
            "specific_heat": specific_heat,
            "perfusion": perfusion,
        }
        shapes = {name: np.shape(v) for name, v in vols.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"thermal property volumes disagree in shape: {shapes}")
        for name, v in vols.items():
            arr = np.ascontiguousarray(v, dtype=np.float32)
            if not np.isfinite(arr).all():
                raise ValueError(f"thermal volume '{name}' contains non-finite values")
            setattr(self, name, arr)
        for name in ("k", "rho", "specific_heat"):
            arr = getattr(self, name)
            if (arr <= 0).any():
                raise ValueError(
                    f"thermal volume '{name}' must be > 0 everywhere "
                    f"(min = {float(arr.min())}); zero conductivity, density or heat "
                    f"capacity makes the Pennes equation undefined."
                )
        if (self.perfusion < 0).any():
            raise ValueError(
                f"perfusion must be >= 0 (min = {float(self.perfusion.min())}); a "
                f"negative w_b would make blood a heat SOURCE."
            )
        if not (np.isfinite(dx) and dx > 0):
            raise ValueError(f"dx must be a positive finite spacing in meters, got {dx}")
        self.dx = float(dx)
        if id_map is not None and np.shape(id_map) != self.k.shape:
            raise ValueError(f"id_map shape {np.shape(id_map)} != property shape {self.k.shape}")
        self.id_map = id_map

    # ---------- constructors ----------

    @classmethod
    def homogeneous(cls, shape: tuple[int, ...], material: Material, dx: float) -> ThermalMedium:
        """Uniform thermal medium of one material (the analytic gates)."""
        _require_thermal(material)
        full = np.ones(shape, dtype=np.float32)
        return cls(
            k=full * material.thermal_conductivity,
            rho=full * material.rho,
            specific_heat=full * material.specific_heat,
            perfusion=full * material.perfusion_rate,
            dx=dx,
        )

    @classmethod
    def from_id_map(cls, id_map: np.ndarray, db: MaterialDB, dx: float) -> ThermalMedium:
        """Dense thermal volumes from an integer tissue-id map.

        Unknown ids and materials without thermal fields are both refused,
        with every offender named — a thermal run that quietly defaulted one
        tissue would put a cold (or a boiling) island in the dose map.
        """
        id_map = np.asarray(id_map)
        if not np.issubdtype(id_map.dtype, np.integer):
            raise TypeError(f"id_map must be integer-typed, got {id_map.dtype}")
        present = [int(i) for i in np.unique(id_map)]
        unknown = [i for i in present if i not in db]
        if unknown:
            raise ValueError(
                f"id_map contains ids {unknown} that are missing from the MaterialDB "
                f"(known ids: {list(db.ids)}). Refusing to guess tissue properties."
            )
        incomplete = {i: _missing_thermal_fields(db[i]) for i in present}
        incomplete = {i: miss for i, miss in incomplete.items() if miss}
        if incomplete:
            # Report ALL of them, then raise on the first: fixing a table one
            # error message at a time is how a five-tissue phantom costs five runs.
            listing = ", ".join(
                f"id {i} ({db[i].name!r}): {', '.join(miss)}" for i, miss in incomplete.items()
            )
            first = next(iter(incomplete))
            try:
                _require_thermal(db[first], first)
            except ThermalPropertyError as exc:
                raise ThermalPropertyError(
                    f"{exc}\nAll materials present in this id map that are missing "
                    f"thermal fields: {listing}."
                ) from None
        shp = id_map.shape
        out = {name: np.zeros(shp, np.float32) for name in ("k", "rho", "cp", "w")}
        for tissue_id in present:
            m = db[tissue_id]
            mask = id_map == tissue_id
            out["k"][mask] = m.thermal_conductivity
            out["rho"][mask] = m.rho
            out["cp"][mask] = m.specific_heat
            out["w"][mask] = m.perfusion_rate
        return cls(
            k=out["k"],
            rho=out["rho"],
            specific_heat=out["cp"],
            perfusion=out["w"],
            dx=dx,
            id_map=id_map,
        )

    @classmethod
    def from_medium(cls, medium, db: MaterialDB, dx: float) -> ThermalMedium:
        """The thermal twin of an acoustic :class:`~caustica.medium.Medium`.

        Uses the medium's own ``id_map``, so the two media cannot disagree
        about where the tissues are.
        """
        if getattr(medium, "id_map", None) is None:
            raise ValueError(
                "this Medium carries no id_map (it was built from raw volumes or with "
                "Medium.homogeneous), so its tissue layout cannot be reused. Build the "
                "thermal medium with ThermalMedium.from_id_map(id_map, db, dx) or "
                "ThermalMedium.homogeneous(shape, material, dx)."
            )
        thermal = cls.from_id_map(medium.id_map, db, dx)
        if not np.allclose(thermal.rho, medium.rho):
            raise ValueError(
                "the density this MaterialDB gives disagrees with the acoustic "
                "Medium's rho — the two were built from different tables, and the "
                "thermal solve would run on a different phantom than the acoustic one."
            )
        return thermal

    # ---------- diagnostics ----------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.k.shape

    @property
    def ndim(self) -> int:
        return self.k.ndim

    @property
    def volumetric_heat_capacity(self) -> np.ndarray:
        """``rho * C`` [J/m^3/K] — the energy one degree costs per voxel."""
        return (self.rho.astype(np.float64) * self.specific_heat).astype(np.float32)

    @property
    def diffusivity(self) -> np.ndarray:
        """``k / (rho C)`` [m^2/s]."""
        return (self.k.astype(np.float64) / self.volumetric_heat_capacity).astype(np.float32)

    @property
    def is_perfused(self) -> bool:
        return bool((self.perfusion > 0).any())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        d = self.diffusivity
        return (
            f"ThermalMedium(shape={self.shape}, dx={self.dx * 1e3:.3g} mm, "
            f"D=[{float(d.min()):.3g},{float(d.max()):.3g}] m^2/s, "
            f"perfused={self.is_perfused})"
        )
