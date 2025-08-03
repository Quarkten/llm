"""
Thin abstraction for handling past_key_values (Windows-only).

This module focuses on bookkeeping for ONNX Runtime decoder-with-past graphs.
Backends like MLC and llama.cpp should use their native KV cache mechanisms.

Concepts:
- KvState: container for past K/V tensors per layer
- KvBookkeeper: tracks sequence lengths and helps slice/append states
- Paged-like emulation metadata to support chunking at app level (no true paging)

All arrays here are represented as numpy ndarrays for portability; real GPU tensors
should be managed by backend-specific IO-binding logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class KvState:
    """
    Shapes are backend/export dependent. A common pattern for decoder-only with past:
      K: [B, num_heads, head_dim, T]
      V: [B, num_heads, T, head_dim]
    This class does not enforce shapes; it stores arbitrary arrays per layer.
    """
    keys: List[np.ndarray] = field(default_factory=list)
    values: List[np.ndarray] = field(default_factory=list)

    def num_layers(self) -> int:
        return len(self.keys)

    def append_token(self, k_list: List[np.ndarray], v_list: List[np.ndarray]) -> None:
        if len(k_list) != len(v_list):
            raise ValueError("k_list and v_list length mismatch")
        if not self.keys:
            self.keys = [k.copy() for k in k_list]
            self.values = [v.copy() for v in v_list]
            return
        if len(k_list) != len(self.keys):
            raise ValueError("Layer count mismatch")
        # Simple concat on time axis based on heuristic axes [-1] or [-2].
        for i, (k_new, v_new) in enumerate(zip(k_list, v_list)):
            k_old = self.keys[i]
            v_old = self.values[i]
            try:
                # Try concatenation along last axis for K, second-to-last for V commonly.
                self.keys[i] = np.concatenate([k_old, k_new], axis=-1)
                self.values[i] = np.concatenate([v_old, v_new], axis=-2)
            except Exception:
                # Fallback: concat last axis for both
                self.keys[i] = np.concatenate([k_old, k_new], axis=-1)
                self.values[i] = np.concatenate([v_old, v_new], axis=-1)

    def slice_context(self, start: int, end: Optional[int] = None) -> "KvState":
        """
        Return a shallow-sliced KvState in the time dimension.
        """
        end = None if end is None else int(end)

        def _slice(arr: np.ndarray) -> np.ndarray:
            try:
                return arr[..., start:end]
            except Exception:
                # If axes differ, try best-effort on last axis
                return arr[..., start:end]

        sliced_k = [_slice(k) for k in self.keys]
        sliced_v = [_slice(v) for v in self.values]
        return KvState(keys=sliced_k, values=sliced_v)


@dataclass
class PagedMeta:
    chunk_size: int
    chunks: List[Tuple[int, int]]


class KvBookkeeper:
    """
    Tracks lengths and paged-like chunks.
    """

    def __init__(self, total_len_hint: int = 0, chunk_size: int = 1024):
        self.toks: int = 0
        self.chunk_size = chunk_size
        self.paged = PagedMeta(chunk_size=chunk_size, chunks=[])
        if total_len_hint > 0:
            for s in range(0, total_len_hint, chunk_size):
                self.paged.chunks.append((s, min(total_len_hint, s + chunk_size)))

    def observe_tokens(self, n: int = 1) -> None:
        self.toks += n
        # Extend chunks if needed
        while self.paged.chunks and self.toks > self.paged.chunks[-1][1]:
            start = self.paged.chunks[-1][1]
            self.paged.chunks.append((start, start + self.chunk_size))

    def reset(self) -> None:
        self.toks = 0
        self.paged = PagedMeta(chunk_size=self.chunk_size, chunks=[])

    def current_chunk_index(self) -> int:
        for i, (s, e) in enumerate(self.paged.chunks):
            if s <= self.toks < e:
                return i
        return max(0, len(self.paged.chunks) - 1)