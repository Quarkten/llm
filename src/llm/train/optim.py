# Copyright 2025
# Apache-2.0 License

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler


def build_optimizer(
    params: Iterable[torch.nn.Parameter],
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    # Separate out weight-decay vs no-decay parameters (bias and LayerNorm/RMSNorm weights usually no-decay)
    decay, no_decay = [], []
    for p in params:
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    opt = AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    return opt


class CosineWithWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, min_lr: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * float(step) / float(max(1, self.warmup_steps))
            else:
                t = (step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
                t = min(max(t, 0.0), 1.0)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * t))
            lrs.append(lr)
        return lrs