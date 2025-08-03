# Copyright 2025
# Apache-2.0 License

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class SyntheticTextDataset(Dataset):
    """
    Minimal synthetic dataset for smoke tests.
    Generates random token sequences [seq_len] from vocab_size.
    """
    num_samples: int
    seq_len: int
    vocab_size: int
    seed: int = 42

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Create a random sequence; ensure targets exist by shifting later in collate
        g = torch.Generator()
        g.manual_seed(self.seed + idx)
        return torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long, generator=g)


def pack_documents_stub(seqs: List[torch.Tensor], seq_len: int) -> List[torch.Tensor]:
    """
    Placeholder for document packing. For now, just truncate/pad to seq_len.
    """
    out: List[torch.Tensor] = []
    for s in seqs:
        if s.numel() >= seq_len:
            out.append(s[:seq_len])
        else:
            pad = torch.zeros(seq_len - s.numel(), dtype=s.dtype)
            out.append(torch.cat([s, pad], dim=0))
    return out


def language_aware_sampling_hook_stub(batch: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Placeholder for language-aware sampling. No-op.
    """
    return batch


def collate_next_token(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function producing (input_ids, labels) with next-token shift.
    """
    x = torch.stack(batch, dim=0)  # [B,S]
    labels = x.clone()
    labels[:, :-1] = x[:, 1:]
    # last label could be ignored (set to -100) but we'll keep next-token shift simple
    return x, labels


def make_synthetic_loader(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    steps: int,
    num_workers: int = 0,
    pack_documents: bool = False,
    language_sampling: bool = False,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Build a small finite iterable over random sequences for smoke tests.
    """
    dataset = SyntheticTextDataset(num_samples=max(steps * batch_size, 64), seq_len=seq_len, vocab_size=vocab_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=lambda b: collate_next_token(b))
    it = iter(loader)

    produced = 0
    while produced < steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)

        if pack_documents:
            xs = pack_documents_stub([t for t in x], seq_len)
            ys = pack_documents_stub([t for t in y], seq_len)
            x = torch.stack(xs, dim=0)
            y = torch.stack(ys, dim=0)

        if language_sampling:
            xs = language_aware_sampling_hook_stub([t for t in x])
            x = torch.stack(xs, dim=0)

        yield x, y
        produced += 1


# Placeholders for future implementations
def make_webdataset_loader(*args, **kwargs):
    raise NotImplementedError("WebDataset loader not implemented. TODO: integrate webdataset and shard sampling.")


def make_parquet_loader(*args, **kwargs):
    raise NotImplementedError("Parquet loader not implemented. TODO: integrate Arrow/Parquet and packing.")