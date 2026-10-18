"""Marching cubes on the NeRF density grid with scikit-image (BSD-3-Clause) instead of the
torchmcubes C++/CUDA extension upstream TripoSR needs, so Windows needs no compiler."""

from __future__ import annotations

import numpy as np
import torch


class MarchingCubeHelper:
    points_range = (0.0, 1.0)

    def __init__(self, resolution: int) -> None:
        self.resolution = int(resolution)
        self._grid: torch.Tensor | None = None

    @property
    def grid_vertices(self) -> torch.Tensor:
        """``(R³, 3)`` positions in ``points_range``, x fastest-varying last (ij indexing),
        matching how the density volume is reshaped below."""
        if self._grid is None:
            r = self.resolution
            axis = torch.linspace(*self.points_range, r)
            x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
            self._grid = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=-1)
        return self._grid

    def __call__(self, level: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``level`` is ``density - threshold`` for every grid vertex (positive inside).
        Returns vertices in ``points_range`` and triangle indices with outward normals."""
        from skimage import measure

        r = self.resolution
        volume = level.detach().cpu().float().numpy().reshape(r, r, r)
        if volume.max() <= 0.0 or volume.min() >= 0.0:
            raise ValueError(
                "the density field never crosses the surface threshold: the model produced "
                "no object (an empty or all-background input image?)"
            )
        verts, faces, _normals, _values = measure.marching_cubes(
            volume, level=0.0, gradient_direction="descent"
        )
        verts = verts.astype(np.float32) / (r - 1.0)
        # scikit-image winds its triangles clockwise seen from outside for a positive-inside
        # volume; glTF and OBJ viewers expect counter-clockwise front faces, so flip them
        faces = np.ascontiguousarray(faces[:, ::-1]).astype(np.int64)
        return torch.from_numpy(verts), torch.from_numpy(faces)
