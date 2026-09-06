"""Medium: dense per-voxel acoustic property volumes.

A Medium holds the four property volumes the solvers consume (alpha, rho,
c, beta), always float32 and C-contiguous, always in HOST (numpy) memory.
Moving them to the GPU is the solver's job at init time (one transfer per
run), mirroring the notebook design where the property maps were uploaded
once and reused for the whole session.

A Medium also carries the ``geometry`` it was sampled on, ``"cartesian"``
(the default, and what every solver in the tree integrates) or
``"axisymmetric"`` (the 2-D (r, z) half-plane). The attribute exists so a
Cartesian solver can refuse an axisymmetric setup instead of solving it as a
line source; :meth:`caustica.geometry.scene.Scene.to_medium` sets it.

Two constructors cover the current use cases:
* ``Medium.homogeneous(shape, material)``: validation media (water).
* ``Medium.from_id_map(id_map, db)``: phantom media; every voxel id must
  exist in the MaterialDB, unknown ids raise with the offending id list
  (silently defaulting a tissue would corrupt physics).

Both constructors read ``material.alpha_np_m``, so both refuse a power-law
material (``alpha0_db_cm_mhz_y`` with ``y``, whose ``alpha_np_m`` is None):
a Medium holds ONE absorption number per voxel, which only exists once a
frequency has been named. Bake the table first with
:meth:`caustica.materials.MaterialDB.at_frequency`.
"""

from __future__ import annotations

import numpy as np

from caustica.materials import Material, MaterialDB

#: Grid geometries a medium can be sampled on. ``"cartesian"`` is the plain
#: 1/2/3-D grid every solver here integrates; ``"axisymmetric"`` is the 2-D
#: (r, z) half-plane, where axis 0 is the radius (r >= 0) and the physics is
#: that of a revolved 3-D body, NOT of a 2-D Cartesian line source.
GEOMETRIES = ("cartesian", "axisymmetric")


def _require_single_frequency(material: Material, tissue_id: int | None = None) -> None:
    """Refuse a material whose absorption is still a power law.

    A ``Medium`` holds one alpha [Np/m] per voxel, so the law has to be
    evaluated at the drive frequency before the volume can be filled. Written
    as a refusal because the silent alternative is worse than a crash:
    ``material.alpha_np_m`` is None for a power-law material, and numpy
    writes None into a float32 volume as NaN, which produces a medium whose
    c, rho and beta are correct and whose absorption is NaN everywhere.
    """
    if material.absorption_model != "power_law":
        return
    where = f"tissue id {tissue_id} " if tissue_id is not None else ""
    raise ValueError(
        f"material {where}{material.name!r} declares power-law absorption "
        f"(alpha0_db_cm_mhz_y={material.alpha0_db_cm_mhz_y}, y={material.y}) and so "
        f"has no single alpha_np_m. A Medium carries ONE absorption number per "
        f"voxel, and alpha0 f^y is a number only once a frequency is named. Bake "
        f"the table at the drive frequency first, then hand the baked table or "
        f"material to the builder: "
        f"Medium.from_id_map(id_map, db.at_frequency(f0_hz)), "
        f"Medium.homogeneous(shape, material.at_frequency(f0_hz)), "
        f"scene.to_medium(grid, db.at_frequency(f0_hz)). Filling the "
        f"volume as it stands would write None into a float32 array, which numpy "
        f"turns into NaN, silently."
    )


class Medium:
    def __init__(
        self,
        alpha: np.ndarray,
        rho: np.ndarray,
        c: np.ndarray,
        beta: np.ndarray,
        id_map: np.ndarray | None = None,
        geometry: str = "cartesian",
    ):
        if geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
        self.geometry = geometry
        vols = {"alpha": alpha, "rho": rho, "c": c, "beta": beta}
        shapes = {name: v.shape for name, v in vols.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"property volumes disagree in shape: {shapes}")
        for name, v in vols.items():
            arr = np.ascontiguousarray(v, dtype=np.float32)
            setattr(self, name, arr)
        if id_map is not None and id_map.shape != self.alpha.shape:
            raise ValueError(f"id_map shape {id_map.shape} != property shape {self.alpha.shape}")
        self.id_map = id_map

    # ---------- constructors ----------

    @classmethod
    def homogeneous(
        cls, shape: tuple[int, ...], material: Material, geometry: str = "cartesian"
    ) -> Medium:
        """Uniform medium of one material (validation runs, water tanks).

        The material's absorption must already be in the legacy
        frequency-independent form; a power-law material is refused, naming
        ``Material.at_frequency(f0_hz)`` as the fix.
        """
        _require_single_frequency(material)
        full = np.full(shape, 1.0, dtype=np.float32)
        return cls(
            alpha=full * material.alpha_np_m,
            rho=full * material.rho,
            c=full * material.c,
            beta=full * material.beta,
            geometry=geometry,
        )

    @classmethod
    def from_id_map(cls, id_map: np.ndarray, db: MaterialDB, geometry: str = "cartesian") -> Medium:
        """Dense property volumes from an integer tissue-id map.

        Unknown ids and power-law materials are both refused, with every
        offender named; the second because a power law has no single alpha to
        write into the volume, and None written into a float32 array is a
        silent NaN.
        """
        id_map = np.asarray(id_map)
        if not np.issubdtype(id_map.dtype, np.integer):
            raise TypeError(f"id_map must be integer-typed, got {id_map.dtype}")
        present = np.unique(id_map)
        unknown = [int(i) for i in present if int(i) not in db]
        if unknown:
            raise ValueError(
                f"id_map contains ids {unknown} that are missing from the MaterialDB "
                f"(known ids: {list(db.ids)}). Refusing to guess tissue properties."
            )
        power_law = [i for i in present if db[int(i)].absorption_model == "power_law"]
        if power_law:
            # Name ALL of them, then raise on the first: fixing a table one error
            # message at a time is how a five-tissue phantom costs five runs.
            listing = ", ".join(f"id {int(i)} ({db[int(i)].name!r})" for i in power_law)
            first = int(power_law[0])
            try:
                _require_single_frequency(db[first], first)
            except ValueError as exc:
                raise ValueError(
                    f"{exc}\nAll materials present in this id map that declare a "
                    f"power law: {listing}."
                ) from None
        shp = id_map.shape
        alpha = np.zeros(shp, np.float32)
        rho = np.zeros(shp, np.float32)
        c = np.zeros(shp, np.float32)
        beta = np.zeros(shp, np.float32)
        for tissue_id in present:
            m = db[int(tissue_id)]
            mask = id_map == tissue_id
            alpha[mask] = m.alpha_np_m
            rho[mask] = m.rho
            c[mask] = m.c
            beta[mask] = m.beta
        return cls(alpha=alpha, rho=rho, c=c, beta=beta, id_map=id_map, geometry=geometry)

    # ---------- diagnostics ----------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.alpha.shape

    @property
    def c_min(self) -> float:
        return float(self.c.min())

    @property
    def c_max(self) -> float:
        return float(self.c.max())

    @property
    def is_linear(self) -> bool:
        """True when the whole volume has beta == 0 (no nonlinear term)."""
        return bool((self.beta == 0).all())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Medium(shape={self.shape}, c=[{self.c_min:.0f},{self.c_max:.0f}] m/s, "
            f"linear={self.is_linear}, geometry={self.geometry})"
        )
