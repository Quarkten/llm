"""
CPU fallback backend for Windows using oneDNN-capable stacks.

Provides minimal greedy generation via:
- Preferred: llama.cpp CPU (if available) using subprocess with CPU-only flags.
- Alternate: ONNX Runtime CPUExecutionProvider for small decoder-only ONNX as a stub.

Exposes env-based thread pinning (OMP_NUM_THREADS, KMP_AFFINITY) reporting.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Generator, Optional, Tuple


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _log(msg: str) -> None:
    print(f"[CPU-oneDNN] {msg}")


def _llama_exe() -> Optional[str]:
    for name in ["llama-cli.exe", "main.exe", "llama.exe", "llama.cpp.exe"]:
        p = shutil.which(name)
        if p:
            return p
    return None


@dataclass
class CpuThreadConfig:
    omp_num_threads: Optional[str]
    kmp_affinity: Optional[str]
    mkl_num_threads: Optional[str]
    note: str = "Threads configured via env. AVX2/oneDNN expected on Intel i5-8400."


def print_thread_config() -> CpuThreadConfig:
    cfg = CpuThreadConfig(
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        kmp_affinity=os.environ.get("KMP_AFFINITY"),
        mkl_num_threads=os.environ.get("MKL_NUM_THREADS"),
    )
    _log(f"OMP_NUM_THREADS={cfg.omp_num_threads} KMP_AFFINITY={cfg.kmp_affinity} MKL_NUM_THREADS={cfg.mkl_num_threads}")
    return cfg


class CpuOneDnnLLM:
    """
    Minimal CPU greedy decode using llama.cpp CPU or ORT CPU EP as stub.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.mode: str = "llama_subprocess"  # or "onnx_cpu"
        self.gguf_path: Optional[str] = None
        self.n_ctx: int = 2048

    def load_llama(self, gguf_path: str, n_ctx: int = 2048) -> None:
        if not _is_windows():
            raise RuntimeError("Windows-only backend")
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(gguf_path)
        exe = _llama_exe()
        if not exe:
            raise RuntimeError("llama.cpp executable not found in PATH for CPU fallback")
        self.gguf_path = gguf_path
        self.n_ctx = n_ctx
        self.mode = "llama_subprocess"
        self.enabled = True
        print_thread_config()
        _log(f"Loaded llama.cpp CPU: ctx={n_ctx}")

    def generate_stream(self, prompt: str, max_new_tokens: int) -> Generator[Tuple[str, Dict[str, float]], None, None]:
        if not self.enabled:
            raise RuntimeError("CpuOneDnnLLM not loaded")
        if self.mode != "llama_subprocess":
            raise RuntimeError("Only llama subprocess mode implemented for CPU fallback")

        exe = _llama_exe()
        assert exe and self.gguf_path
        cmd = [
            exe,
            "-m", self.gguf_path,
            "-p", prompt,
            "-n", str(max_new_tokens),
            "--ctx-size", str(self.n_ctx),
            "--no-mmap",  # encourage explicit allocations on CPU
        ]
        _log("Command: " + " ".join([f'"{c}"' if " " in c else c for c in cmd]))
        t0 = time.perf_counter()
        ttft: Optional[float] = None

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
        try:
            if not proc.stdout:
                raise RuntimeError("No stdout from llama subprocess")
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - t0
                    yield line, {"ttft": ttft, "toks": 1}
                else:
                    yield line, {"toks": 1}
        finally:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                pass