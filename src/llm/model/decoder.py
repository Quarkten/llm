# Copyright 2025
# Apache-2.0 License

from __future__ import annotations

from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..tensor.attention import GQASelfAttention


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_mult: float = 4.0):
        super().__init__()
        hidden = int(d_model * hidden_mult)
        # Use gated projection: two linear layers share input
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = GQASelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            max_seq_len=cfg.max_seq_len,
            rotary_pct=cfg.rotary_pct,
            rope_base=cfg.rope_base,
            rope_ntk_factor=cfg.rope_ntk_factor,
            rope_interpolation_factor=cfg.rope_interpolation_factor,
            use_swa=cfg.use_swa,
            swa_window=cfg.swa_window,
            dropout=cfg.dropout,
        )
        self.mlp_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, hidden_mult=cfg.mlp_ratio)
        self.dropout = nn.Dropout(cfg.dropout)
        self.swa_global_every = cfg.swa_global_every

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x)
        h = self.attn(h, layer_index=self.layer_idx, global_every=self.swa_global_every)
        x = x + self.dropout(h)

        h = self.mlp_norm(x)
        h = self.mlp(h)
        x = x + self.dropout(h)
        return x


class DecoderOnlyLM(nn.Module):
    """
    Minimal decoder-only transformer with GQA, RoPE, SWA, SwiGLU MLP, RMSNorm.

    Supports activation checkpointing via torch.utils.checkpoint if enabled in config.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=cfg.init_std)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=cfg.init_std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, S]
        returns logits: [B, S, V]
        """
        B, S = input_ids.shape
        x = self.embed(input_ids)  # [B,S,D]

        if self.cfg.activation_checkpointing:
            def _block_fn(block, x):
                return block(x)

            for blk in self.blocks:
                x = torch.utils.checkpoint.checkpoint(_block_fn, blk, x, use_reentrant=False)
        else:
            for blk in self.blocks:
                x = blk(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits


def cross_entropy_loss_with_zloss(logits: torch.Tensor, targets: torch.Tensor, z_loss: float = 0.0) -> torch.Tensor:
    """
    Standard next-token cross entropy plus z-loss regularization on logsumexp.
    Shapes:
      logits: [B,S,V], targets: [B,S]
    """
    vocab = logits.size(-1)
    loss = F.cross_entropy(logits.view(-1, vocab), targets.view(-1), reduction="mean")
    if z_loss != 0.0:
        lse = torch.logsumexp(logits, dim=-1)  # [B,S]
        loss = loss + z_loss * (lse ** 2).mean()
    return loss