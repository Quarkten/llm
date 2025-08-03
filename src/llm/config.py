# Copyright 2025
# Apache-2.0 License

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union

import yaml
from pydantic import BaseModel, Field, PositiveInt, validator


def _default_device() -> str:
    # torch import is optional here to keep config importable without torch
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


class ModelConfig(BaseModel):
    # Architecture
    vocab_size: PositiveInt = 32000
    d_model: PositiveInt = 512
    n_layers: PositiveInt = 4
    n_heads: PositiveInt = 8
    n_kv_heads: Optional[PositiveInt] = None  # if None, equals n_heads (no GQA)
    rope_base: float = 10000.0
    rope_ntk_factor: float = 1.0
    rope_interpolation_factor: float = 1.0  # >=1.0, position interpolation scaling
    max_seq_len: PositiveInt = 512
    rotary_pct: float = 1.0  # fraction of dims using RoPE
    mlp_ratio: float = 4.0
    activation: Literal["swiglu"] = "swiglu"
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_word_embeddings: bool = True

    # SWA (sliding window attention)
    use_swa: bool = True
    swa_window: PositiveInt = 256
    swa_global_every: PositiveInt = 4  # every N layers allow global attention override

    # Checkpointing
    activation_checkpointing: bool = False

    # Init
    init_std: float = 0.02

    @validator("n_kv_heads", always=True)
    def _validate_gqa(cls, v, values):
        n_heads = values.get("n_heads")
        if v is None:
            return n_heads
        if n_heads % v != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
        return v

    @validator("rotary_pct")
    def _rotary_pct_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("rotary_pct must be within [0,1]")
        return v

    @validator("rope_interpolation_factor")
    def _interp_ge_one(cls, v):
        if v < 1.0:
            raise ValueError("rope_interpolation_factor must be >= 1.0")
        return v


class TrainConfig(BaseModel):
    # IO
    output_dir: str = "checkpoints"
    save_every_steps: PositiveInt = 50
    log_every_steps: PositiveInt = 10

    # Optim
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: PositiveInt = 100
    max_steps: PositiveInt = 200
    min_lr: float = 3e-5
    scheduler: Literal["cosine", "none"] = "cosine"
    z_loss: float = 0.0

    # Data
    batch_size: PositiveInt = 2  # per step effective batch = batch_size * grad_accum
    grad_accum_steps: PositiveInt = 4
    seq_len: PositiveInt = 512
    vocab_size: Optional[int] = None  # optional override for toy synthetic loader

    # AMP and precision
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    grad_clip: float = 1.0

    # Device / Seed
    device: str = Field(default_factory=_default_device)
    seed: int = 42
    deterministic: bool = True

    # Dataloader stubs
    dataset_type: Literal["synthetic", "webdataset", "parquet"] = "synthetic"
    num_workers: int = 2
    pack_documents: bool = False
    language_sampling: bool = False


class SFTConfig(BaseModel):
    base_ckpt: Optional[str] = None
    learning_rate: float = 1e-5
    max_steps: PositiveInt = 200
    dataset_type: Literal["synthetic", "jsonl", "parquet"] = "synthetic"


class DPOConfig(BaseModel):
    base_ckpt: Optional[str] = None
    learning_rate: float = 5e-6
    beta: float = 0.1
    max_steps: PositiveInt = 200


class InferenceConfig(BaseModel):
    ckpt_path: Optional[str] = None
    device: str = Field(default_factory=_default_device)
    dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    quant: Optional[Literal["awq", "gptq"]] = None
    kv_cache_quant: Optional[Literal["int8", "nf4", "none"]] = None
    max_seq_len: PositiveInt = 2048
    top_p: float = 0.95
    top_k: int = 0
    temperature: float = 0.8


class RAGConfig(BaseModel):
    retriever: Literal["none", "qdrant", "pgvector"] = "none"
    retriever_url: Optional[str] = None
    k: PositiveInt = 4
    reranker: Optional[str] = None  # placeholder hook name


class RootConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: Optional[TrainConfig] = None
    sft: Optional[SFTConfig] = None
    dpo: Optional[DPOConfig] = None
    infer: Optional[InferenceConfig] = None
    rag: RAGConfig = Field(default_factory=RAGConfig)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "RootConfig":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yml", ".yaml"):
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.model_validate(data)


def load_config_file(path: Union[str, Path]) -> RootConfig:
    return RootConfig.from_file(path)