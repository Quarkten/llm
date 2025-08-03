# Copyright 2025
# Apache-2.0 License

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to q and k.

    Shapes:
      q, k: [B, S, H, D] or [B, H, S, D] depending on calling site; here we assume [B, S, H, D].
      cos, sin: [S, rotary_dim] broadcastable to q[..., :rotary_dim]
    rotary_dim: number of head dims to rotate (typically int(D * rotary_pct))

    Returns rotated q, k.
    """
    q1, q2 = q[..., :rotary_dim], q[..., rotary_dim:]
    k1, k2 = k[..., :rotary_dim], k[..., rotary_dim:]

    # Interleave last dimension into pairs for rotation
    q1_2 = q1.float().reshape(*q1.shape[:-1], rotary_dim // 2, 2)
    k1_2 = k1.float().reshape(*k1.shape[:-1], rotary_dim // 2, 2)

    # cos/sin shape align: [S, rotary_dim] -> [1, S, 1, rotary_dim/2, 1]
    cos_ = cos[:, : rotary_dim].reshape(1, cos.shape[0], 1, rotary_dim // 2, 2)
    sin_ = sin[:, : rotary_dim].reshape(1, sin.shape[0], 1, rotary_dim // 2, 2)

    # Rotation: (x, y) -> (x*cos - y*sin, x*sin + y*cos)
    xq, yq = q1_2[..., 0:1], q1_2[..., 1:2]
    xk, yk = k1_2[..., 0:1], k1_2[..., 1:2]

    qx = xq * cos_[..., 0:1] - yq * sin_[..., 0:1]
    qy = xq * sin_[..., 0:1] + yq * cos_[..., 0:1]
    kx = xk * cos_[..., 0:1] - yk * sin_[..., 0:1]
    ky = xk * sin_[..., 0:1] + yk * cos_[..., 0:1]

    q_rot = torch.cat([qx, qy], dim=-1).reshape(*q1.shape)
    k_rot = torch.cat([kx, ky], dim=-1).reshape(*k1.shape)

    q_out = torch.cat([q_rot.type_as(q), q2], dim=-1)
    k_out = torch.cat([k_rot.type_as(k), k2], dim=-1)
    return q_out, k_out


class RoPECache(nn.Module):
    """
    Precomputes and caches cos/sin for rotary embeddings with NTK scaling and optional position interpolation.

    Args:
      dim: head dimension
      base: rope base (default 10000)
      ntk_factor: multiplicative scaling for base (>=1 increases context)
      max_seq_len: maximum sequence length for cache
      interpolation_factor: >=1.0, scales effective position index i' = floor(i / interpolation_factor)
      device/dtype: placement

    Note: This module is lightweight and can be re-initialized when max_seq_len grows.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        ntk_factor: float = 1.0,
        max_seq_len: int = 2048,
        interpolation_factor: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"
        assert ntk_factor > 0, "ntk_factor must be > 0"
        assert interpolation_factor >= 1.0, "interpolation_factor must be >= 1.0"

        base_eff = base * ntk_factor
        inv_freq = 1.0 / (base_eff ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.max_seq_len = max_seq_len
        self.interpolation_factor = interpolation_factor
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32

        cos, sin = self._build_cache(self.max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _build_cache(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Effective positions with interpolation
        pos = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        pos_eff = torch.floor(pos / self.interpolation_factor)
        freqs = torch.einsum("i,j->ij", pos_eff, self.inv_freq)  # [S, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [S, D]
        cos = torch.cos(emb).to(self.dtype)
        sin = torch.sin(emb).to(self.dtype)
        return cos, sin

    def maybe_extend(self, new_len: int):
        if new_len > self.max_seq_len:
            self.max_seq_len = int(new_len * 1.25)  # grow with slack
            cos, sin = self._build_cache(self.max_seq_len)
            self.cos = cos.to(self.device, self.dtype)
            self.sin = sin.to(self.device, self.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, rotary_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k: [B, S, H, D]
        seq_len = q.shape[1]
        self.maybe_extend(seq_len)
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        return apply_rope(q, k, cos, sin, rotary_dim)