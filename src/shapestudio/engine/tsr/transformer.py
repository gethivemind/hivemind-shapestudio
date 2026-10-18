"""TripoSR's triplane transformer: a 1D transformer with self- and cross-attention blocks.

Derived from Hugging Face diffusers (Apache-2.0) as modified by Tripo AI and Stability AI
(MIT, 2024); trimmed to the code paths the published checkpoint uses (layer norm, GEGLU,
scaled-dot-product attention, no masks, no dropout). Parameter names are unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = False,
        out_bias: bool = True,
    ) -> None:
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        kv_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.to_out = nn.ModuleList(
            [nn.Linear(self.inner_dim, query_dim, bias=out_bias), nn.Dropout(0.0)]
        )

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None
    ) -> torch.Tensor:
        b = hidden_states.shape[0]
        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        q = self.to_q(hidden_states)
        k = self.to_k(context)
        v = self.to_v(context)
        head_dim = self.inner_dim // self.heads
        q = q.view(b, -1, self.heads, head_dim).transpose(1, 2)
        k = k.view(b, -1, self.heads, head_dim).transpose(1, 2)
        v = v.view(b, -1, self.heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, -1, self.inner_dim).to(q.dtype)
        return self.to_out[1](self.to_out[0](out))


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.ModuleList([GEGLU(dim, inner), nn.Dropout(0.0), nn.Linear(inner, dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.net:
            x = m(x)
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, None, num_attention_heads, attention_head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(dim, cross_attention_dim, num_attention_heads, attention_head_dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), encoder_hidden_states) + x
        return self.ff(self.norm3(x)) + x


class Transformer1D(nn.Module):
    """Input ``(B, C, N)`` tokens; output the same shape with a residual connection."""

    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        cross_attention_dim: int,
        norm_num_groups: int = 32,
    ) -> None:
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.norm = nn.GroupNorm(norm_num_groups, in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim, num_attention_heads, attention_head_dim, cross_attention_dim
                )
                for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
        batch, _, seq_len = hidden_states.shape
        residual = hidden_states
        x = self.norm(hidden_states)
        inner_dim = x.shape[1]
        x = x.permute(0, 2, 1).reshape(batch, seq_len, inner_dim)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, encoder_hidden_states)
        x = self.proj_out(x)
        x = x.reshape(batch, seq_len, inner_dim).permute(0, 2, 1).contiguous()
        return x + residual
