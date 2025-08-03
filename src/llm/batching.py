"""
Micro-batching utilities for LLM prompts (Windows-only).

- pad_and_stack: pad variable-length token lists to max length and build attention masks
- make_batches: split a list of tokenized prompts into micro-batches
- ORT-friendly outputs: int64 arrays for input_ids and attention_mask

These utilities are backend-agnostic, but tailored for ONNX Runtime inputs.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np


def pad_and_stack(
    token_seqs: List[List[int]],
    pad_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pad to max length. Returns (input_ids, attention_mask, lengths)
      input_ids: [B, T] int64
      attention_mask: [B, T] int64 with 1 for real tokens, 0 for padding
      lengths: [B] int32 actual lengths
    """
    if not token_seqs:
        raise ValueError("Empty token_seqs")
    max_len = max(len(x) for x in token_seqs)
    bsz = len(token_seqs)

    ids = np.full((bsz, max_len), pad_id, dtype=np.int64)
    mask = np.zeros((bsz, max_len), dtype=np.int64)
    lens = np.zeros((bsz,), dtype=np.int32)

    for i, seq in enumerate(token_seqs):
        L = len(seq)
        ids[i, :L] = np.array(seq, dtype=np.int64)
        mask[i, :L] = 1
        lens[i] = L

    return ids, mask, lens


def make_batches(token_seqs: List[List[int]], micro_batch: int) -> Iterable[List[List[int]]]:
    """
    Yield chunks of size micro_batch.
    """
    for i in range(0, len(token_seqs), micro_batch):
        yield token_seqs[i : i + micro_batch]