"""Scene: ordered label assignment of shapes/volumes + rasterization.

COMSOL-like construction flow: start from a background label, add objects
in paint order (later assignments override earlier ones, like drawing),
then rasterize onto a grid at YOUR dx. Supports 2-D, 3-D and 2-D
axisymmetric (the (r, z) half-plane convention: axis 0 is the radius,
r >= 0; the future axisymmetric solver consumes the same scenes).

Edge quality: ``supersample=s`` evaluates s^ndim sub-points per voxel and
takes the majority label — the standard cure for staircased interfaces on
coarse grids. Rasterization is chunked along axis 0 so 3-D grids never
materialize a full supersampled point cloud.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from caustica.core.grid import Grid
from caustica.geometry.shapes import Shape
from caustica.geometry.volumes import LabelVolume


@dataclass(frozen=True)
class _ShapeEntry:
    shape: Shape
    label: int


@dataclass(frozen=True)
class _VolumeEntry:
    volume: LabelVolume
    position: tuple[float, ...]  # physical offset of the volume's origin [m]
    ignore: tuple[int, ...]  # labels treated as transparent


class Scene:
    """Ordered geometry assignments producing an integer label map."""

    def __init__(self, ndim: int, background: int = 0, axisymmetric: bool = False):
        if ndim not in (2, 3):
            raise ValueError(f"ndim must be 2 or 3, got {ndim}")
        if axisymmetric and ndim != 2:
            raise ValueError("axisymmetric scenes are 2-D (r, z half-plane)")
        self.ndim = ndim
        self.background = int(background)
        self.axisymmetric = bool(axisymmetric)
        self._entries: list[_ShapeEntry | _VolumeEntry] = []

    def add(self, shape: Shape, label: int) -> Scene:
        """Paint ``shape`` with ``label`` (later adds override earlier ones)."""
        if shape.ndim != self.ndim:
            raise ValueError(f"{shape.ndim}-D shape in a {self.ndim}-D scene")
        self._entries.append(_ShapeEntry(shape, int(label)))
        return self

    def add_volume(
        self,
        volume: LabelVolume,
        position: tuple[float, ...] | None = None,
        ignore: tuple[int, ...] = (),
    ) -> Scene:
        """Place an imported label volume; its labels paint over the scene.

        ``position`` is the physical location of the volume's first voxel
        center [m]. When omitted it defaults to ``volume.origin`` — so a
        volume produced by :meth:`rasterize` (or round-tripped through npz)
        lands back where it was, instead of silently at zero (review
        finding, 2026-08-11). Labels listed in ``ignore`` are transparent
        (typically the phantom's own background).
        """
        if volume.ndim != self.ndim:
            raise ValueError(f"{volume.ndim}-D volume in a {self.ndim}-D scene")
        pos = tuple(float(p) for p in (position if position is not None else volume.origin))
        if len(pos) != self.ndim:
            raise ValueError(f"position needs {self.ndim} entries, got {position}")
        self._entries.append(_VolumeEntry(volume, pos, tuple(int(i) for i in ignore)))
        return self

    @property
    def n_objects(self) -> int:
        return len(self._entries)

    # ---------------- rasterization ----------------

    def rasterize(
        self,
        grid: Grid,
        origin: tuple[float, ...] | None = None,
        supersample: int = 1,
        chunk_voxels: int = 2_000_000,
    ) -> LabelVolume:
        """Evaluate the scene on ``grid`` voxel centers -> :class:`LabelVolume`.

        ``origin`` is the physical coordinate of voxel (0, ..., 0)'s center
        (default zeros). ``supersample >= 2`` takes a majority vote over
        sub-voxel samples (odd values recommended; big win on curved
        interfaces at coarse dx).
        """
        if grid.ndim != self.ndim:
            raise ValueError(f"{grid.ndim}-D grid for a {self.ndim}-D scene")
        if supersample < 1:
            raise ValueError(f"supersample must be >= 1, got {supersample}")
        org = np.asarray(origin if origin is not None else (0.0,) * self.ndim, np.float64)
        if org.size != self.ndim:
            raise ValueError(f"origin needs {self.ndim} entries")
        if self.axisymmetric and org[0] < -1e-12:
            # Voxel CENTERS must sit at r >= 0; the first center on the axis
            # (r = 0) is the standard axisymmetric convention.
            raise ValueError(
                f"axisymmetric scene: radial axis must start at r >= 0 (origin[0] = {org[0]:g} m)"
            )

        s = int(supersample)
        dx = grid.dx
        # Sub-voxel offsets, centered within the voxel.
        sub = (np.arange(s) - (s - 1) / 2.0) * (dx / s)

        # Per-axis supersampled coordinate vectors.
        axes = [
            org[d] + np.arange(grid.shape[d])[:, None] * dx + sub[None, :] for d in range(self.ndim)
        ]
        axes = [a.reshape(-1) for a in axes]  # length shape[d] * s each

        out = np.full(grid.shape, self.background, dtype=np.int32)
        # Sub-points per voxel-row of axis 0: s sub-rows x (each later axis
        # contributes n*s samples) -> prod(shape[1:]) * s^ndim total.
        n_rows_chunk = max(
            1, int(chunk_voxels // max(1, int(np.prod(grid.shape[1:])) * s**self.ndim))
        )

        for row0 in range(0, grid.shape[0], n_rows_chunk):
            row1 = min(row0 + n_rows_chunk, grid.shape[0])
            ax0 = axes[0][row0 * s : row1 * s]
            mesh = np.meshgrid(ax0, *axes[1:], indexing="ij")
            pts = np.column_stack([m.reshape(-1) for m in mesh])
            if self.axisymmetric:
                # Sub-voxel samples of the on-axis voxel row reach r < 0;
                # physically those are the mirror points |r| (review
                # finding, 2026-08-11 — unmirrored sampling misclassified
                # the axis row, exactly where the focus sits).
                pts[:, 0] = np.abs(pts[:, 0])
            labels = np.full(pts.shape[0], self.background, dtype=np.int32)
            for entry in self._entries:
                if isinstance(entry, _ShapeEntry):
                    labels[entry.shape.contains(pts)] = entry.label
                else:
                    self._paint_volume(entry, pts, labels)
            ss_shape = [(row1 - row0) * s] + [n * s for n in grid.shape[1:]]
            labels = labels.reshape(ss_shape)
            out[row0:row1] = self._majority(labels, s)
        return LabelVolume(labels=out.astype(np.int32), dx=dx, origin=tuple(org.tolist()))

    def _paint_volume(self, entry: _VolumeEntry, pts: np.ndarray, labels: np.ndarray) -> None:
        vol = entry.volume
        idx = np.round((pts - np.asarray(entry.position)) / vol.dx).astype(np.int64)
        inside = np.ones(pts.shape[0], dtype=bool)
        for d in range(self.ndim):
            inside &= (idx[:, d] >= 0) & (idx[:, d] < vol.shape[d])
        vals = vol.labels[tuple(idx[inside, d] for d in range(self.ndim))].astype(np.int32)
        keep = np.ones(vals.shape[0], dtype=bool)
        for ig in entry.ignore:
            keep &= vals != ig
        target = np.where(inside)[0][keep]
        labels[target] = vals[keep]

    def _paint_precedence(self) -> dict[int, int]:
        """Later-painted labels get higher precedence (background lowest)."""
        prec = {self.background: 0}
        for i, entry in enumerate(self._entries, start=1):
            if isinstance(entry, _ShapeEntry):
                prec[entry.label] = i
            else:
                for lab in entry.volume.label_set:
                    if lab not in entry.ignore:
                        prec[lab] = i
        return prec

    def _majority(self, ss_labels: np.ndarray, s: int) -> np.ndarray:
        """Majority vote over s^ndim sub-samples per voxel.

        Vote ties break toward the most recently painted label (true
        paint-order semantics; review finding 2026-08-11 — np.unique order
        used to make the smallest label win regardless of paint order).
        """
        if s == 1:
            return ss_labels
        shape = tuple(n // s for n in ss_labels.shape)
        # -> (vox..., sub...) then flatten sub axes.
        resh: list[int] = []
        for n in shape:
            resh.extend((n, s))
        moved = ss_labels.reshape(resh)
        order = list(range(0, 2 * len(shape), 2)) + list(range(1, 2 * len(shape), 2))
        blocks = moved.transpose(order).reshape(*shape, s ** len(shape))
        prec = self._paint_precedence()
        # argmax returns the FIRST maximum -> order candidates by
        # descending precedence so ties resolve to the latest paint.
        present = np.array(
            sorted(np.unique(ss_labels), key=lambda lab: (-prec.get(int(lab), -1), int(lab)))
        )
        counts = np.zeros((*shape, present.size), dtype=np.int32)
        for i, lab in enumerate(present):
            counts[..., i] = (blocks == lab).sum(axis=-1)
        return present[np.argmax(counts, axis=-1)].astype(np.int32)

    # ---------------- conveniences ----------------

    def to_medium(self, grid: Grid, db, origin=None, supersample: int = 1):
        """Rasterize and map to a :class:`~caustica.medium.Medium` via ``db``."""
        from caustica.medium import Medium

        vol = self.rasterize(grid, origin=origin, supersample=supersample)
        return Medium.from_id_map(vol.labels.astype(np.int64), db)
