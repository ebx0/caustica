"""Medium: dense per-voxel acoustic property volumes.

A Medium holds the four property volumes the solvers consume — alpha, rho,
c, beta — always float32 and C-contiguous, always in HOST (numpy) memory.
Moving them to the GPU is the solver's job at init time (one transfer per
run), mirroring the notebook design where the property maps were uploaded
once and reused for the whole session.

Two constructors cover the current use cases:
* ``Medium.homogeneous(shape, material)`` — validation media (water).
* ``Medium.from_id_map(id_map, db)`` — phantom media; every voxel id must
  exist in the MaterialDB, unknown ids raise with the offending id list
  (silently defaulting a tissue would corrupt physics).
"""

from __future__ import annotations

import numpy as np

from hifusim.materials import Material, MaterialDB


class Medium:
    def __init__(
        self,
        alpha: np.ndarray,
        rho: np.ndarray,
        c: np.ndarray,
        beta: np.ndarray,
        id_map: np.ndarray | None = None,
    ):
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
    def homogeneous(cls, shape: tuple[int, ...], material: Material) -> Medium:
        """Uniform medium of one material (validation runs, water tanks)."""
        full = np.full(shape, 1.0, dtype=np.float32)
        return cls(
            alpha=full * material.alpha_np_m,
            rho=full * material.rho,
            c=full * material.c,
            beta=full * material.beta,
        )

    @classmethod
    def from_id_map(cls, id_map: np.ndarray, db: MaterialDB) -> Medium:
        """Dense property volumes from an integer tissue-id map."""
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
        return cls(alpha=alpha, rho=rho, c=c, beta=beta, id_map=id_map)

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
            f"linear={self.is_linear})"
        )
