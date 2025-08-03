"""
VRAM memory planner for Windows-only targets.

Estimates VRAM usage for:
- Model weights (by quant/precision)
- KV cache (context length, n_layers, n_kv_heads, head_dim, micro-batch)
- Activations / overhead heuristic
and suggests adjustments when near the RX 6800 16GB limit.

This is an estimator; real usage depends on backend/runtime implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


BYTES_PER_DTYPE = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "awq4": 0.5,   # approx weight-only 4-bit
    "gguf_q4_k_m": 0.5,  # heuristic
    "gguf_q5_k_m": 0.625,  # heuristic
}


@dataclass
class KvSpec:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    context_len: int
    micro_batch: int = 1
    dtype_bytes: float = 2.0  # fp16 KV typical on GPU


def estimate_weights_bytes(n_params: float, precision: str) -> float:
    """
    Estimate model weights memory in bytes.
    n_params: number of parameters (e.g., 7e9 for 7B)
    precision: key in BYTES_PER_DTYPE
    """
    bpp = BYTES_PER_DTYPE.get(precision.lower(), 2.0)
    return n_params * bpp


def estimate_kv_bytes(kv: KvSpec) -> float:
    """
    KV cache bytes for both K and V:
    micro_batch * n_layers * n_kv_heads * context_len * head_dim * dtype_bytes * 2
    """
    return (
        kv.micro_batch
        * kv.n_layers
        * kv.n_kv_heads
        * kv.context_len
        * kv.head_dim
        * kv.dtype_bytes
        * 2.0
    )


def estimate_overhead_bytes(weights_bytes: float) -> float:
    """
    Heuristic overhead: optimizer buffers, staging, activations peak, graph, etc.
    Use 10% of weights as a baseline plus 512MB.
    """
    return weights_bytes * 0.10 + 512 * 1024**2


def plan_vram(
    n_params: float,
    precision: str,
    kv: KvSpec,
    vram_limit_bytes: float = 16 * 1024**3,  # 16GB RX 6800 target
    overhead_bytes: Optional[float] = None,
) -> Dict[str, float]:
    w = estimate_weights_bytes(n_params, precision)
    k = estimate_kv_bytes(kv)
    o = estimate_overhead_bytes(w) if overhead_bytes is None else overhead_bytes
    total = w + k + o
    headroom = vram_limit_bytes - total
    return {
        "weights_gb": w / 1024**3,
        "kv_gb": k / 1024**3,
        "overhead_gb": o / 1024**3,
        "total_gb": total / 1024**3,
        "limit_gb": vram_limit_bytes / 1024**3,
        "headroom_gb": headroom / 1024**3,
    }


def suggest_adjustments(plan: Dict[str, float]) -> Dict[str, str]:
    """
    Provide coarse suggestions if headroom is low or negative.
    """
    headroom = plan["headroom_gb"]
    rec: Dict[str, str] = {}
    if headroom < 0.0:
        rec["status"] = "exceeds_vram"
        rec["suggest"] = (
            "Reduce quantization precision (e.g., AWQ4/DirectML int8), lower micro-batch, "
            "reduce context length, or use Q4_K_M over Q5_K_M."
        )
    elif headroom < 1.0:
        rec["status"] = "tight_vram"
        rec["suggest"] = (
            "Tight headroom: consider smaller KV (shorter context), fewer concurrent requests, or lower precision."
        )
    else:
        rec["status"] = "ok"
        rec["suggest"] = "Configuration appears reasonable for 16GB."
    return rec


def quick_qwen_7b_defaults(
    precision: str = "awq4",
    context_len: int = 4096,
    micro_batch: int = 1,
    n_layers: int = 32,
    n_kv_heads: int = 28,  # Qwen2.5 heads vary by model; using typical 7B KV heads
    head_dim: int = 128,
    kv_dtype_bytes: float = 2.0,
) -> Dict[str, float]:
    """
    Convenience wrapper for Qwen2.5-7B estimates.
    """
    kv = KvSpec(
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        context_len=context_len,
        micro_batch=micro_batch,
        dtype_bytes=kv_dtype_bytes,
    )
    plan = plan_vram(7e9, precision, kv)
    plan.update(suggest_adjustments(plan))
    return plan