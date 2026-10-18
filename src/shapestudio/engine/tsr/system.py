"""The TripoSR model as one module: image → triplane scene code → mesh with vertex colours.

MIT (Tripo AI & Stability AI, 2024), vendored; see the package docstring for what changed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from shapestudio.engine.tsr.isosurface import MarchingCubeHelper
from shapestudio.engine.tsr.nerf import (
    DINOSingleImageTokenizer,
    NeRFMLP,
    Triplane1DTokenizer,
    TriplaneNeRFRenderer,
    TriplaneUpsampleNetwork,
    scale_tensor,
)
from shapestudio.engine.tsr.transformer import Transformer1D

# stabilityai/TripoSR/config.yaml (sha256 74ca708c…) as a dict. The file is still downloaded and
# hash-verified for provenance; the architecture is fixed per checkpoint, so it lives here too.
TRIPOSR_CONFIG: dict[str, Any] = {
    "cond_image_size": 512,
    "tokenizer": {"plane_size": 32, "num_channels": 1024},
    "backbone": {
        "in_channels": 1024,
        "num_attention_heads": 16,
        "attention_head_dim": 64,
        "num_layers": 16,
        "cross_attention_dim": 768,
    },
    "post_processor": {"in_channels": 1024, "out_channels": 40},
    "decoder": {"in_channels": 120, "n_neurons": 64, "n_hidden_layers": 9, "activation": "silu"},
    "renderer": {
        "radius": 0.87,
        "density_activation": "exp",
        "density_bias": -1.0,
        "num_samples_per_ray": 128,
    },
}


class TSR(nn.Module):
    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = cfg or TRIPOSR_CONFIG
        self.cfg = cfg
        self.cond_image_size = int(cfg["cond_image_size"])
        self.image_tokenizer = DINOSingleImageTokenizer()
        self.tokenizer = Triplane1DTokenizer(**cfg["tokenizer"])
        self.backbone = Transformer1D(**cfg["backbone"])
        self.post_processor = TriplaneUpsampleNetwork(**cfg["post_processor"])
        self.decoder = NeRFMLP(**cfg["decoder"])
        self.renderer = TriplaneNeRFRenderer(**cfg["renderer"])
        self.isosurface_helper: MarchingCubeHelper | None = None

    # -- loading ---------------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, weight_path: Path, cfg: dict[str, Any] | None = None) -> TSR:
        """Load ``model.ckpt``. Missing or unexpected tensors are an error with the names
        listed, never a silent partial load."""
        model = cls(cfg)
        state = torch.load(str(weight_path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
        except RuntimeError as e:  # shape mismatches
            raise RuntimeError(
                f"model.ckpt does not match the vendored TripoSR architecture: {e}"
            ) from e
        if missing or unexpected:
            raise RuntimeError(
                "model.ckpt does not match the vendored TripoSR architecture: "
                f"missing {sorted(missing)[:8]}{'…' if len(missing) > 8 else ''}, "
                f"unexpected {sorted(unexpected)[:8]}{'…' if len(unexpected) > 8 else ''}"
            )
        model.eval()
        return model

    # -- forward ---------------------------------------------------------------------------

    @staticmethod
    def preprocess(image: np.ndarray, size: int) -> torch.Tensor:
        """``(H, W, 3)`` uint8 or float 0-1 → ``(1, 1, 3, size, size)`` float in 0-1."""
        if image.dtype == np.uint8:
            t = torch.from_numpy(image.astype(np.float32) / 255.0)
        else:
            t = torch.from_numpy(np.asarray(image, dtype=np.float32))
        t = t[None].permute(0, 3, 1, 2)  # 1 C H W
        t = F.interpolate(t, (size, size), mode="bilinear", align_corners=False, antialias=True)
        return t[:, None]  # B Nv C H W

    @torch.no_grad()
    def scene_codes(self, image: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """One network pass: ``(1, 3, 40, 64, 64)`` triplane for the image."""
        rgb = self.preprocess(image, self.cond_image_size).to(device)
        tokens_img = self.image_tokenizer(rgb)  # B Nv C Nt
        b = tokens_img.shape[0]
        tokens_img = tokens_img.permute(0, 1, 3, 2).reshape(b, -1, tokens_img.shape[2])  # B Nt C
        tokens = self.tokenizer(b)
        tokens = self.backbone(tokens, encoder_hidden_states=tokens_img)
        return self.post_processor(self.tokenizer.detokenize(tokens))

    forward = scene_codes

    # -- mesh --------------------------------------------------------------------------------

    def set_marching_cubes_resolution(self, resolution: int) -> None:
        if self.isosurface_helper is None or self.isosurface_helper.resolution != resolution:
            self.isosurface_helper = MarchingCubeHelper(resolution)

    @torch.no_grad()
    def extract_mesh(
        self,
        scene_code: torch.Tensor,
        resolution: int = 256,
        threshold: float = 25.0,
        check: Callable[[], None] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``scene_code`` (3, C, H, W) → vertices (N,3) in (-radius, radius), faces (M,3),
        vertex colours (N,3) in 0-1."""
        self.set_marching_cubes_resolution(resolution)
        helper = self.isosurface_helper
        assert helper is not None
        radius = self.renderer.radius
        grid = scale_tensor(helper.grid_vertices, helper.points_range, (-radius, radius))
        if on_stage:
            on_stage("density")
        density = self.renderer.query_triplane(
            self.decoder, grid.to(scene_code.device), scene_code, check=check
        )["density_act"]
        if on_stage:
            on_stage("marching")
        v_pos, faces = helper(density.cpu() - threshold)
        v_pos = scale_tensor(v_pos, helper.points_range, (-radius, radius))
        if on_stage:
            on_stage("colour")
        color = self.renderer.query_triplane(
            self.decoder, v_pos.to(scene_code.device), scene_code, check=check
        )["color"]
        return (
            v_pos.cpu().numpy().astype(np.float32),
            faces.cpu().numpy(),
            color.cpu().numpy().astype(np.float32),
        )

    # -- NeRF renders (optional; the studio previews the mesh itself) -------------------

    @torch.no_grad()
    def render(
        self,
        scene_code: torch.Tensor,
        n_views: int = 6,
        elevation_deg: float = 0.0,
        camera_distance: float = 1.9,
        fovy_deg: float = 40.0,
        size: int = 256,
    ) -> list[np.ndarray]:
        rays_o, rays_d = spherical_cameras(n_views, elevation_deg, camera_distance, fovy_deg, size)
        out = []
        for i in range(n_views):
            img = self.renderer(
                self.decoder,
                scene_code,
                rays_o[i].to(scene_code.device),
                rays_d[i].to(scene_code.device),
            )
            out.append((img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8))
        return out


def spherical_cameras(
    n_views: int, elevation_deg: float, distance: float, fovy_deg: float, size: int
):
    azimuth = torch.linspace(0, 360.0, n_views + 1)[:n_views] * math.pi / 180
    elevation = torch.full_like(azimuth, elevation_deg * math.pi / 180)
    pos = torch.stack(
        [
            distance * torch.cos(elevation) * torch.cos(azimuth),
            distance * torch.cos(elevation) * torch.sin(azimuth),
            distance * torch.sin(elevation),
        ],
        dim=-1,
    )
    center = torch.zeros_like(pos)
    up = torch.tensor([0.0, 0.0, 1.0])[None].repeat(n_views, 1)
    lookat = F.normalize(center - pos, dim=-1)
    right = F.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
    up = F.normalize(torch.cross(right, lookat, dim=-1), dim=-1)
    c2w3x4 = torch.cat([torch.stack([right, up, -lookat], dim=-1), pos[:, :, None]], dim=-1)
    focal = 0.5 * size / math.tan(0.5 * fovy_deg * math.pi / 180)
    i, j = torch.meshgrid(
        torch.arange(size, dtype=torch.float32) + 0.5,
        torch.arange(size, dtype=torch.float32) + 0.5,
        indexing="xy",
    )
    dirs = torch.stack([(i - size / 2) / focal, -(j - size / 2) / focal, -torch.ones_like(i)], -1)
    dirs = dirs[None].repeat(n_views, 1, 1, 1)
    rays_d = (dirs[:, :, :, None, :] * c2w3x4[:, None, None, :3, :3]).sum(-1)
    rays_d = F.normalize(rays_d, dim=-1)
    rays_o = c2w3x4[:, None, None, :3, 3].expand(rays_d.shape)
    return rays_o, rays_d
