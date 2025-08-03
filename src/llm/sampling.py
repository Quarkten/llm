"""
Sampling utilities for LLM logits (Windows-only context).

Implements:
- greedy_select(logits_row)
- nucleus_sampling(logits_row, top_p=0.9, temperature=0.7, top_k=None)
Designed to run on CPU overlapping with GPU decode for ORT/MLC paths.

Inputs are expected as 1D numpy arrays of shape [vocab_size] (float32/float16).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _softmax_stable(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        # avoid divide by zero; treat as greedy
        y = np.zeros_like(x, dtype=np.float32)
        y[np.argmax(x)] = 1.0
        return y
    x = x.astype(np.float32) / float(temperature)
    m = np.max(x)
    e = np.exp(x - m)
    return e / np.sum(e)


def greedy_select(logits_row: np.ndarray) -> int:
    """
    Return argmax token id.
    """
    return int(np.argmax(logits_row))


def nucleus_sampling(
    logits_row: np.ndarray,
    top_p: float = 0.9,
    temperature: float = 0.7,
    top_k: Optional[int] = None,
    min_tokens_to_keep: int = 1,
) -> int:
    """
    Nucleus (top-p) sampling with optional top-k and temperature.
    """
    # Apply temperature and softmax
    probs = _softmax_stable(logits_row, temperature=temperature)

    # Optionally restrict to top-k
    if top_k is not None and top_k > 0:
        idx = np.argpartition(-probs, top_k)[:top_k]
        mask = np.zeros_like(probs, dtype=bool)
        mask[idx] = True
        probs = probs * mask
        s = probs.sum()
        if s > 0:
            probs = probs / s

    # Sort by probability
    sorted_idx = np.argsort(-probs)
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, top_p)
    cutoff = max(cutoff, min_tokens_to_keep - 1)
    keep_idx = sorted_idx[: cutoff + 1]
    keep_probs = probs[keep_idx]
    keep_probs = keep_probs / keep_probs.sum()

    # Sample
    choice = np.random.choice(len(keep_idx), p=keep_probs)
    return int(keep_idx[choice])