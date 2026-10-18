"""TripoSR's tokenizers, triplane upsampler, NeRF MLP and renderer (MIT, Tripo AI & Stability
AI, 2024), without einops/omegaconf. Parameter names are unchanged."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from shapestudio.engine.tsr.vit import DINO_VITB16, ViTModel


class DINOSingleImageTokenizer(nn.Module):
    """Images ``(B, 1, C, H, W)`` in 0-1 → tokens ``(B, 1, 768, N)``."""

    def __init__(self) -> None:
        super().__init__()
        self.model = ViTModel(DINO_VITB16)
        self.register_buffer(
            "image_mean",
            torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, n_views = images.shape[:2]
        images = (images - self.image_mean) / self.image_std
        flat = images.reshape(b * n_views, *images.shape[2:])
        last_hidden, _pooled = self.model(flat, interpolate_pos_encoding=True)
        local = last_hidden.permute(0, 2, 1)  # (B*N, C, Nt)
        return local.reshape(b, n_views, *local.shape[1:])


class Triplane1DTokenizer(nn.Module):
    def __init__(self, plane_size: int, num_channels: int) -> None:
        super().__init__()
        self.plane_size = plane_size
        self.num_channels = num_channels
        self.embeddings = nn.Parameter(
            torch.randn((3, num_channels, plane_size, plane_size), dtype=torch.float32)
            / math.sqrt(num_channels)
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        e = self.embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        # B Np Ct Hp Wp -> B Ct (Np Hp Wp)
        return e.permute(0, 2, 1, 3, 4).reshape(batch_size, self.num_channels, -1)

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        b, ct, nt = tokens.shape
        assert nt == self.plane_size**2 * 3 and ct == self.num_channels
        x = tokens.reshape(b, ct, 3, self.plane_size, self.plane_size)
        return x.permute(0, 2, 1, 3, 4)  # B Np Ct Hp Wp


class TriplaneUpsampleNetwork(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, triplanes: torch.Tensor) -> torch.Tensor:
        b, np_, ci, hp, wp = triplanes.shape
        up = self.upsample(triplanes.reshape(b * np_, ci, hp, wp))
        return up.reshape(b, np_, *up.shape[1:])


class NeRFMLP(nn.Module):
    def __init__(
        self, in_channels: int, n_neurons: int, n_hidden_layers: int, activation: str = "silu"
    ) -> None:
        super().__init__()
        act = {"relu": nn.ReLU, "silu": nn.SiLU}[activation]
        layers: list[nn.Module] = [nn.Linear(in_channels, n_neurons), act(inplace=True)]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), act(inplace=True)]
        layers.append(nn.Linear(n_neurons, 4))  # density 1 + features 3
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shape = x.shape[:-1]
        out = self.layers(x.reshape(-1, x.shape[-1])).reshape(*shape, -1)
        return {"density": out[..., 0:1], "features": out[..., 1:4]}


def scale_tensor(x: torch.Tensor, inp: tuple[float, float], tgt: tuple[float, float]):
    x = (x - inp[0]) / (inp[1] - inp[0])
    return x * (tgt[1] - tgt[0]) + tgt[0]


def rays_intersect_bbox(rays_o, rays_d, radius: float, near: float = 0.0, valid_thresh=0.01):
    shape = rays_o.shape[:-1]
    rays_o, rays_d = rays_o.view(-1, 3), rays_d.view(-1, 3)
    d = torch.where(rays_d.abs() < 1e-6, torch.full_like(rays_d, 1e-6), rays_d)
    r = (1.0 - 1.0e-3) * radius
    inter0 = (r - rays_o) / d
    inter1 = (-r - rays_o) / d
    t_near = torch.minimum(inter0, inter1).amax(dim=-1).clamp_min(near)
    t_far = torch.maximum(inter0, inter1).amin(dim=-1)
    valid = t_far - t_near > valid_thresh
    t_near = torch.where(valid, t_near, torch.zeros_like(t_near))
    t_far = torch.where(valid, t_far, torch.zeros_like(t_far))
    return t_near.view(*shape, 1), t_far.view(*shape, 1), valid.view(*shape)


class TriplaneNeRFRenderer(nn.Module):
    def __init__(
        self,
        radius: float,
        density_activation: str = "exp",
        density_bias: float = -1.0,
        num_samples_per_ray: int = 128,
    ) -> None:
        super().__init__()
        self.radius = radius
        self.density_activation = density_activation
        self.density_bias = density_bias
        self.num_samples_per_ray = num_samples_per_ray
        self.chunk_size = 0

    def set_chunk_size(self, chunk_size: int) -> None:
        self.chunk_size = max(0, int(chunk_size))

    def _density_act(self, x: torch.Tensor) -> torch.Tensor:
        if self.density_activation == "exp":
            return torch.exp(x)
        if self.density_activation == "softplus":
            return F.softplus(x)
        if self.density_activation == "trunc_exp":
            return torch.exp(x.clamp(max=15.0))
        raise ValueError(self.density_activation)

    def query_triplane(
        self,
        decoder: nn.Module,
        positions: torch.Tensor,
        triplane: torch.Tensor,
        check: Callable[[], None] | None = None,
    ) -> dict[str, torch.Tensor]:
        """``positions`` in (-radius, radius); ``triplane`` (3, C, H, W). ``check`` is called
        between chunks so a long extraction can be cancelled."""
        shape = positions.shape[:-1]
        positions = positions.view(-1, 3)
        positions = scale_tensor(positions, (-self.radius, self.radius), (-1, 1))

        def _chunk(x: torch.Tensor) -> dict[str, torch.Tensor]:
            idx = torch.stack((x[..., [0, 1]], x[..., [0, 2]], x[..., [1, 2]]), dim=-3)  # 3 N 2
            out = F.grid_sample(
                triplane, idx.unsqueeze(1), align_corners=False, mode="bilinear"
            )  # 3 C 1 N
            feat = out.squeeze(2).permute(2, 0, 1).reshape(x.shape[0], -1)  # N (3 C)
            return decoder(feat)

        if self.chunk_size > 0 and positions.shape[0] > self.chunk_size:
            parts: dict[str, list[torch.Tensor]] = {"density": [], "features": []}
            for i in range(0, positions.shape[0], self.chunk_size):
                if check is not None:
                    check()
                o = _chunk(positions[i : i + self.chunk_size])
                for k in parts:
                    parts[k].append(o[k])
            net_out = {k: torch.cat(v, dim=0) for k, v in parts.items()}
        else:
            net_out = _chunk(positions)
        net_out["density_act"] = self._density_act(net_out["density"] + self.density_bias)
        net_out["color"] = torch.sigmoid(net_out["features"])
        return {k: v.view(*shape, -1) for k, v in net_out.items()}

    def forward(self, decoder: nn.Module, triplane: torch.Tensor, rays_o, rays_d) -> torch.Tensor:
        rays_shape = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)
        n_rays = rays_o.shape[0]
        t_near, t_far, valid = rays_intersect_bbox(rays_o, rays_d, self.radius)
        t_near, t_far = t_near[valid], t_far[valid]
        t_vals = torch.linspace(0, 1, self.num_samples_per_ray + 1, device=triplane.device)
        t_mid = (t_vals[:-1] + t_vals[1:]) / 2.0
        z_vals = t_near * (1 - t_mid[None]) + t_far * t_mid[None]
        xyz = rays_o[valid][:, None, :] + z_vals[..., None] * rays_d[valid][..., None, :]
        out = self.query_triplane(decoder, xyz, triplane)
        eps = 1e-10
        deltas = t_vals[1:] - t_vals[:-1]
        alpha = 1 - torch.exp(-deltas * out["density_act"][..., 0])
        accum = torch.cat(
            [torch.ones_like(alpha[:, :1]), torch.cumprod(1 - alpha[:, :-1] + eps, dim=-1)],
            dim=-1,
        )
        weights = alpha * accum
        rgb_valid = (weights[..., None] * out["color"]).sum(dim=-2)
        opacity_valid = weights.sum(dim=-1)
        rgb = torch.zeros(n_rays, 3, dtype=rgb_valid.dtype, device=rgb_valid.device)
        opacity = torch.zeros(n_rays, dtype=opacity_valid.dtype, device=opacity_valid.device)
        rgb[valid] = rgb_valid
        opacity[valid] = opacity_valid
        rgb = rgb + (1 - opacity[..., None])
        return rgb.view(*rays_shape, 3)
