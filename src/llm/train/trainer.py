# Copyright 2025
# Apache-2.0 License

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.nn.utils import clip_grad_norm_

from ..config import ModelConfig, TrainConfig
from ..model.decoder import DecoderOnlyLM, cross_entropy_loss_with_zloss
from .dataloader import make_synthetic_loader, make_webdataset_loader, make_parquet_loader
from .optim import build_optimizer, CosineWithWarmup


def set_seed(seed: int, deterministic: bool = True):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


@dataclass
class TrainState:
    step: int = 0
    best_loss: float = float("inf")


def get_autocast_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def build_dataloader(cfg: TrainConfig, steps: int):
    if cfg.dataset_type == "synthetic":
        vocab_size = cfg.vocab_size or 32000
        return make_synthetic_loader(
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            vocab_size=vocab_size,
            steps=steps,
            num_workers=cfg.num_workers,
            pack_documents=cfg.pack_documents,
            language_sampling=cfg.language_sampling,
        )
    if cfg.dataset_type == "webdataset":
        return make_webdataset_loader()  # pragma: no cover
    if cfg.dataset_type == "parquet":
        return make_parquet_loader()  # pragma: no cover
    raise ValueError(f"Unknown dataset_type: {cfg.dataset_type}")


def save_checkpoint(output_dir: str, model: DecoderOnlyLM, optimizer: torch.optim.Optimizer, state: TrainState):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(output_dir) / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state": state.__dict__,
        },
        ckpt_path,
    )


def train_one_step(
    model: DecoderOnlyLM,
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    precision: str,
    scaler: torch.cuda.amp.GradScaler | None,
    z_loss: float,
    grad_clip: float,
    grad_accum_steps: int,
    optimizer: torch.optim.Optimizer,
) -> float:
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    autocast_dtype = get_autocast_dtype(precision)
    use_amp = precision in ("bf16", "fp16") and device.type == "cuda"
    loss_value = 0.0

    if use_amp:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            logits = model(x)
            loss = cross_entropy_loss_with_zloss(logits, y, z_loss=z_loss) / grad_accum_steps
        if precision == "fp16":
            assert scaler is not None
            scaler.scale(loss).backward()
        else:
            loss.backward()
    else:
        logits = model(x)
        loss = cross_entropy_loss_with_zloss(logits, y, z_loss=z_loss) / grad_accum_steps
        loss.backward()

    loss_value = float(loss.detach().item())
    return loss_value


def maybe_step_optimizer(
    model: DecoderOnlyLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    precision: str,
    grad_clip: float,
):
    if precision == "fp16" and scaler is not None:
        scaler.unscale_(optimizer)
        if grad_clip is not None and grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        if grad_clip is not None and grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def run_training(model_cfg: ModelConfig, train_cfg: TrainConfig):
    device = torch.device(train_cfg.device)
    set_seed(train_cfg.seed, train_cfg.deterministic)

    model = DecoderOnlyLM(model_cfg).to(device)
    optimizer = build_optimizer(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, betas=train_cfg.betas, eps=train_cfg.eps)

    if train_cfg.scheduler == "cosine":
        scheduler = CosineWithWarmup(optimizer, warmup_steps=train_cfg.warmup_steps, max_steps=train_cfg.max_steps, min_lr=train_cfg.min_lr)
    else:
        scheduler = None

    scaler = None
    if train_cfg.precision == "fp16" and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    data_iter = build_dataloader(train_cfg, steps=train_cfg.max_steps)
    state = TrainState(step=0)

    optimizer.zero_grad(set_to_none=True)

    t0 = time.time()
    tokens_accum = 0
    for step in range(1, train_cfg.max_steps + 1):
        step_loss_accum = 0.0
        for micro in range(train_cfg.grad_accum_steps):
            batch = next(iter([next(data_iter)]))  # obtain current batch without consuming multiple
            step_loss_accum += train_one_step(
                model=model,
                batch=batch,
                device=device,
                precision=train_cfg.precision,
                scaler=scaler,
                z_loss=train_cfg.z_loss,
                grad_clip=train_cfg.grad_clip,
                grad_accum_steps=train_cfg.grad_accum_steps,
                optimizer=optimizer,
            )
        maybe_step_optimizer(model, optimizer, scaler, train_cfg.precision, train_cfg.grad_clip)
        if scheduler is not None:
            scheduler.step()

        state.step = step
        tokens_accum += train_cfg.batch_size * train_cfg.seq_len

        if step % train_cfg.log_every_steps == 0 or step == 1:
            elapsed = time.time() - t0
            tps = tokens_accum / max(1e-6, elapsed)
            print(f"[step {step:5d}] loss={step_loss_accum:.4f} tokens/s={tps:.1f}")

        if step % train_cfg.save_every_steps == 0 or step == train_cfg.max_steps:
            save_checkpoint(train_cfg.output_dir, model, optimizer, state)

    print("Training finished.")