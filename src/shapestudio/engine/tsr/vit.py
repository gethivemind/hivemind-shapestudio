"""DINO ViT-B/16 as TripoSR's image tokenizer, without the ``transformers`` package.

Parameter names follow Hugging Face's ``ViTModel`` exactly (``embeddings.cls_token``,
``encoder.layer.N.attention.attention.query.weight``, ``layernorm.weight``, ``pooler.dense``…)
so the tensors inside ``stabilityai/TripoSR/model.ckpt`` load without renaming. The position
embedding interpolation reproduces the DINO recipe that transformers 4.35 (TripoSR's pinned
version) used: bicubic with a ``+0.1`` scale nudge.

Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors. Apache-2.0.
Architecture: Facebook AI Research (DINO), Hugging Face (ViTModel), both Apache-2.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ViTConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    layer_norm_eps: float = 1e-12
    qkv_bias: bool = True


DINO_VITB16 = ViTConfig()


class PatchEmbeddings(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            cfg.num_channels, cfg.hidden_size, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class Embeddings(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        n = (cfg.image_size // cfg.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.patch_embeddings = PatchEmbeddings(cfg)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n + 1, cfg.hidden_size))

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int):
        npatch = embeddings.shape[1] - 1
        n = self.position_embeddings.shape[1] - 1
        if npatch == n and height == width:
            return self.position_embeddings
        class_pos = self.position_embeddings[:, 0]
        patch_pos = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.cfg.patch_size + 0.1
        w0 = width // self.cfg.patch_size + 0.1
        side = int(math.sqrt(n))
        patch_pos = F.interpolate(
            patch_pos.reshape(1, side, side, dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / side, w0 / side),
            mode="bicubic",
            align_corners=False,
        )
        if int(h0) != patch_pos.shape[-2] or int(w0) != patch_pos.shape[-1]:
            raise RuntimeError("position embedding interpolation produced an unexpected grid")
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos.unsqueeze(0), patch_pos), dim=1)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool) -> torch.Tensor:
        b, _c, h, w = pixel_values.shape
        x = self.patch_embeddings(pixel_values)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1)
        if interpolate_pos_encoding:
            x = x + self.interpolate_pos_encoding(x, h, w)
        else:
            x = x + self.position_embeddings
        return x


class SelfAttention(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.qkv_bias)
        self.key = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.qkv_bias)
        self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.query(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(b, n, self.num_heads * self.head_dim)


class SelfOutput(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)


class Attention(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.attention = SelfAttention(cfg)
        self.output = SelfOutput(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(x))


class Intermediate(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.dense(x))


class Output(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.dense(x) + residual


class Layer(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.attention = Attention(cfg)
        self.intermediate = Intermediate(cfg)
        self.output = Output(cfg)
        self.layernorm_before = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(self.layernorm_before(x)) + x
        return self.output(self.intermediate(self.layernorm_after(x)), x)


class Encoder(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList([Layer(cfg) for _ in range(cfg.num_hidden_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            x = layer(x)
        return x


class Pooler(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.dense(x[:, 0]))


class ViTModel(nn.Module):
    """Returns ``(last_hidden_state, pooler_output)`` like the Hugging Face model."""

    def __init__(self, cfg: ViTConfig = DINO_VITB16) -> None:
        super().__init__()
        self.config = cfg
        self.embeddings = Embeddings(cfg)
        self.encoder = Encoder(cfg)
        self.layernorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.pooler = Pooler(cfg)

    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embeddings(pixel_values, interpolate_pos_encoding)
        x = self.encoder(x)
        x = self.layernorm(x)
        return x, self.pooler(x)
