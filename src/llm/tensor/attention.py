# Copyright 2025
# Apache-2.0 License

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RoPECache


def _causal_mask(S: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((S, S), float("-inf"), device=device), diagonal=1)


def _swa_mask(S: int, window: int, device: torch.device) -> torch.Tensor:
    # Allow attention only within local window [i-window+1, i]
    mask = torch.full((S, S), float("-inf"), device=device)
    for i in range(S):
        start = max(0, i - window + 1)
        mask[i, start : i + 1] = 0.0
    return mask


class GQASelfAttention(nn.Module):
    """
    Multi-head self-attention with Grouped-Query Attention (GQA) and optional SWA mask.
    Fallback matmul implementation; hooks provided for future kernel swaps.

    Shapes (using batch-first):
      x: [B, S, D]
      returns: [B, S, D]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        max_seq_len: int,
        rotary_pct: float = 1.0,
        rope_base: float = 10000.0,
        rope_ntk_factor: float = 1.0,
        rope_interpolation_factor: float = 1.0,
        use_swa: bool = False,
        swa_window: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.head_dim = d_model // n_heads
        self.kv_groups = n_heads // self.n_kv_heads

        self.rotary_dim = int(self.head_dim * rotary_pct)
        self.use_rope = self.rotary_dim > 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        self.use_swa = use_swa
        self.swa_window = swa_window

        if self.use_rope:
            self.rope = RoPECache(
                dim=self.rotary_dim,
                base=rope_base,
                ntk_factor=rope_ntk_factor,
                max_seq_len=max_seq_len,
                interpolation_factor=rope_interpolation_factor,
            )
        else:
            self.rope = None

        # TODO: hooks for FlashAttention/Triton kernels
        self.kernel = "naive"

    def _shape_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = q.shape
        q = q.view(B, S, self.n_heads, self.head_dim)  # [B,S,H,Dh]
        k = k.view(B, S, self.n_kv_heads, self.head_dim)
        v = v.view(B, S, self.n_kv_heads, self.head_dim)

        if self.use_rope and self.rotary_dim > 0 and self.rope is not None:
            q[..., : self.rotary_dim], k[..., : self.rotary_dim] = self.rope(
                q[..., : self.rotary_dim].contiguous(),
                k[..., : self.rotary_dim].contiguous(),
                self.rotary_dim,
            )

        # Expand K/V to groups to match n_heads
        if self.kv_groups > 1:
            k = k.unsqueeze(2).expand(B, S, self.kv_groups, self.n_kv_heads, self.head_dim).reshape(B, S, self.n_heads, self.head_dim)
            v = v.unsqueeze(2).expand(B, S, self.kv_groups, self.n_kv_heads, self.head_dim).reshape(B, S, self.n_heads, self.head_dim)

        return q, k, v

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, layer_index: Optional[int] = None, global_every: Optional[int] = None) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q, k, v = self._shape_qkv(q, k, v)  # [B,S,H,Dh] each

        # Compute attention scores
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.einsum("bshd,bt hd->bhst", q, k) * scale  # [B,H,S,T] ; here T=S

        # Build mask
        mask = _causal_mask(S, device=x.device)  # [S,S]
        if self.use_swa:
            mask = torch.maximum(mask, _swa_mask(S, self.swa_window, device=x.device))
            if global_every is not None and layer_index is not None and global_every > 0:
                # Every N layers, allow full attention by zeroing mask (override)
                if (layer_index + 1) % global_every == 0:
                    mask = torch.zeros_like(mask)

        if attn_mask is not None:
            mask = mask + attn_mask  # broadcast [S,S] or [B,1,S,S] ready

        attn_scores = attn_scores + mask  # broadcast over batch and heads

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum("bhst,bthd->bshd", attn, v)  # [B,S,H,Dh]
        out = out.reshape(B, S, D)
        return self.o_proj(out)