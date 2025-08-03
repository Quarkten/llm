# Copyright 2025
# Apache-2.0 License

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class QuantConfig:
    method: Optional[str] = None  # "awq" | "gptq" | None
    kv_cache_quant: Optional[str] = None  # "int8" | "nf4" | "none" | None


def is_rocm_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def apply_fake_quant_linear(module: nn.Module, method: Optional[str]) -> nn.Module:
    """
    Placeholder that returns the module unchanged.
    TODO: integrate AWQ/GPTQ libraries when ROCm support is available.
    """
    # Intentionally no-op for now
    return module


def allocate_kv_cache(dtype: torch.dtype, kv_cache_quant: Optional[str]):
    """
    Allocate KV cache factory with optional quantization toggles.
    Currently a simple passthrough; returns dtype for use by attention cache.
    """
    if kv_cache_quant in (None, "none"):
        return dtype
    # Placeholder handling; real impl would allocate quantized buffers
    if kv_cache_quant == "int8":
        return torch.int8
    if kv_cache_quant == "nf4":
        # No native dtype; placeholder maps to float16 storage with fake dequant
        return torch.float16
    return dtype