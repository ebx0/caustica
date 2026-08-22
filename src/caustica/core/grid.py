"""Cartesian computation grid (1/2/3-D, isotropic spacing) and its k-space.

Scope decision (load-bearing): ``Grid`` owns *geometry only* — shape, spacing,
coordinates and spatial-frequency (k) vectors. It deliberately does NOT own
the time step, CFL number or the kappa dispersion correction: those depend on
the medium's maximum sound speed and on solver policy (exact-period dt), so
they live with the solver. Keeping Grid medium-free lets one grid be reused
across media, solvers and sweeps.

Conventions
-----------
* Spacing is isotropic (single ``dx``) — a k-space PSTD requirement in this
  codebase; anisotropic grids would silently break the kappa correction.
* SI units internally (meters). The config layer (`caustica.config`) converts
  user-facing millimeters; display helpers expose mm for humans.
* k-vectors follow numpy FFT layout: full ``fftfreq`` on every axis, plus the
  ``rfftfreq`` variant for the LAST axis (solvers use real-to-complex FFTs
  with Hermitian symmetry on the last axis to halve memory/bandwidth).
"""

from __future__ import annotations

import numpy as np

from caustica.core.pml import PMLSpec


class Grid:
    """Isotropic Cartesian grid.

    Parameters
    ----------
    shape:
        Voxel counts per axis; length 1, 2 or 3. Every axis must have at
        least 4 voxels (below that, FFT-based derivatives are meaningless).
    dx:
        Isotropic spacing in meters.
    pml:
        Optional sponge-PML specification attached to this grid. The grid
        only converts its thickness to voxels; application is solver policy.
    """

    def __init__(self, shape: tuple[int, ...] | list[int], dx: float, pml: PMLSpec | None = None):
        shape = tuple(int(s) for s in shape)
        if not 1 <= len(shape) <= 3:
            raise ValueError(f"Grid supports 1..3 dimensions, got shape {shape}")
        if any(s < 4 for s in shape):
            raise ValueError(f"every axis needs >= 4 voxels, got shape {shape}")
        if not (np.isfinite(dx) and dx > 0):
            raise ValueError(f"dx must be a positive finite number in meters, got {dx}")
        self.shape = shape
        self.dx = float(dx)
        self.pml = pml

    # ---------- basic geometry ----------

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dx_mm(self) -> float:
        return self.dx * 1e3

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape))

    @property
    def extent(self) -> tuple[float, ...]:
        """Physical size per axis in meters (shape * dx)."""
        return tuple(s * self.dx for s in self.shape)

    @property
    def pml_vox(self) -> int:
        """Attached PML thickness in voxels (0 when no PML is attached)."""
        return 0 if self.pml is None else self.pml.thickness_vox(self.dx)

    def axis_coords(self, axis: int, centered: bool = True) -> np.ndarray:
        """Voxel-center coordinates along ``axis`` in meters.

        ``centered=True`` puts 0 at the grid center (the natural frame for
        focused-transducer setups); ``False`` starts at 0 on the first voxel.
        """
        n = self.shape[axis]
        x = np.arange(n, dtype=np.float64) * self.dx
        if centered:
            x -= x[n // 2]
        return x

    # ---------- k-space ----------

    def k_axis(self, axis: int) -> np.ndarray:
        """Angular spatial frequencies (rad/m) for a full FFT along ``axis``."""
        return 2.0 * np.pi * np.fft.fftfreq(self.shape[axis], d=self.dx)

    def k_axis_r(self, axis: int) -> np.ndarray:
        """Angular spatial frequencies for a REAL FFT along ``axis`` (rfft layout)."""
        return 2.0 * np.pi * np.fft.rfftfreq(self.shape[axis], d=self.dx)

    def k_vectors(self, rfft_last: bool = True) -> tuple[np.ndarray, ...]:
        """Broadcast-ready k-vectors, one per axis.

        Each returned array has ``ndim`` axes with length 1 everywhere except
        its own axis, so ``sum(k*k for k in ks)`` broadcasts to the (r)FFT
        field shape without allocating dense meshes.
        """
        ks = []
        for ax in range(self.ndim):
            k = self.k_axis_r(ax) if (rfft_last and ax == self.ndim - 1) else self.k_axis(ax)
            shape = [1] * self.ndim
            shape[ax] = k.size
            ks.append(k.reshape(shape))
        return tuple(ks)

    def k_squared(self, rfft_last: bool = True) -> np.ndarray:
        """|k|^2 on the (r)FFT layout (float64; solvers cast as needed)."""
        ks = self.k_vectors(rfft_last=rfft_last)
        out = ks[0] ** 2
        for k in ks[1:]:
            out = out + k**2
        return out

    @property
    def k_max(self) -> float:
        """Nyquist angular frequency pi/dx (rad/m)."""
        return np.pi / self.dx

    # ---------- resolution diagnostics ----------

    def ppw(self, f0: float, c: float) -> float:
        """Points-per-wavelength at frequency ``f0`` [Hz] and speed ``c`` [m/s].

        The k-space PSTD scheme is accurate down to ~2 ppw; the dataset
        design rule of thumb (dx=0.30 mm, f0=1.1 MHz, c=1450) gives 4.39 ppw
        at f0 and 2.20 ppw at the second harmonic.
        """
        if f0 <= 0 or c <= 0:
            raise ValueError("f0 and c must be > 0")
        return c / (f0 * self.dx)

    def voxels_for(self, length_m: float) -> int:
        """Round a physical length to whole voxels (mm-to-voxel one-way rule)."""
        return int(round(length_m / self.dx))

    # ---------- repr ----------

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        pml = f", pml={self.pml_vox}vox" if self.pml is not None else ""
        return f"Grid(shape={self.shape}, dx={self.dx_mm:.3g} mm{pml})"
