"""
Simple metrics utilities for Windows-only pipelines.

Provides:
- MovingAverage: track moving average of values (e.g., tok/s)
- Timer: context manager for durations
- LlmMetrics: TTFT, tokens/sec, end-to-end latency helpers
- VisionMetrics: images/sec
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional
from collections import deque


class MovingAverage:
    def __init__(self, window: int = 50):
        self.window = window
        self.buf: Deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self.buf.append(value)

    def avg(self) -> float:
        if not self.buf:
            return 0.0
        return sum(self.buf) / len(self.buf)

    def to_dict(self) -> Dict[str, float]:
        return {"avg": self.avg(), "count": float(len(self.buf))}


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.t1 = time.perf_counter()
        self.dt = self.t1 - self.t0


@dataclass
class LlmMetrics:
    ttft: Optional[float] = None
    token_times: List[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None

    def mark_ttft(self) -> None:
        self.ttft = time.perf_counter() - self.start_time

    def mark_token(self) -> None:
        self.token_times.append(time.perf_counter())

    def mark_end(self) -> None:
        self.end_time = time.perf_counter()

    def tokens_per_sec(self) -> float:
        if len(self.token_times) < 2:
            return 0.0
        # Compute rate based on last N tokens
        t0 = self.token_times[0]
        t1 = self.token_times[-1]
        dt = max(1e-6, t1 - t0)
        n = len(self.token_times) - 1
        return n / dt

    def e2e_latency(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def to_dict(self) -> Dict:
        return {
            "ttft": self.ttft,
            "tokens_per_sec": self.tokens_per_sec(),
            "e2e_latency": self.e2e_latency(),
            "n_tokens": len(self.token_times),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class VisionMetrics:
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    n_images: int = 0

    def add_batch(self, batch_size: int) -> None:
        self.n_images += batch_size

    def mark_end(self) -> None:
        self.end_time = time.perf_counter()

    def images_per_sec(self) -> float:
        if self.end_time is None:
            return 0.0
        dt = max(1e-6, self.end_time - self.start_time)
        return self.n_images / dt

    def to_dict(self) -> Dict:
        return {"images": self.n_images, "images_per_sec": self.images_per_sec()}