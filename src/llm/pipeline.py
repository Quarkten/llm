"""
LLM pipeline orchestrator for Windows-only backends.

Runtime selection priority:
  1) MLC Vulkan (AWQ 4-bit)
  2) ONNX Runtime DirectML (int8 weight-only if available)
  3) llama.cpp Vulkan (GGUF Q4_K_M/Q5_K_M)
  4) CPU oneDNN fallback

Features:
- Backend self-checks and graceful skipping when deps/artifacts unavailable
- Tokenization via Hugging Face (QwenTokenizer)
- Streaming generation with TTFT and tokens/sec metrics
- Batch/micro-batch helpers for ORT (future extension)
- Environment-driven behavior as documented in Windows setup scripts

This module provides functional stubs for streaming even when external runtimes
are not installed; those paths will log warnings and raise RuntimeError on use.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

from ..utils.metrics import LlmMetrics
from .tokenizer import QwenTokenizer

# Optional imports guarded at call sites
# Backends (Windows-only)
try:
    from ..runtime.windows.mlc_vulkan_backend import MlcVulkanLLM  # type: ignore
except Exception:
    MlcVulkanLLM = None  # type: ignore

try:
    from ..runtime.windows.onnx_dml_backend import OnnxDmlLLM  # type: ignore
except Exception:
    OnnxDmlLLM = None  # type: ignore

try:
    from ..runtime.windows.llama_vulkan_backend import LlamaVulkanLLM  # type: ignore
except Exception:
    LlamaVulkanLLM = None  # type: ignore

try:
    from ..runtime.windows.cpu_onednn_backend import CpuOneDnnLLM  # type: ignore
except Exception:
    CpuOneDnnLLM = None  # type: ignore


def _log(msg: str) -> None:
    print(f"[LlmPipeline] {msg}")


@dataclass
class LlmConfig:
    # Common
    model_name: str = "Qwen2.5-7B-Instruct"
    tokenizer_path: Optional[str] = None
    # MLC
    mlc_artifact_dir: Optional[str] = None
    mlc_micro_batch: int = 1
    # ONNX DML
    onnx_model_dir: Optional[str] = None  # contains decoder.onnx / decoder_with_past.onnx
    # llama.cpp Vulkan
    gguf_path: Optional[str] = None
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    quant_preset: str = "Q4_K_M"
    # CPU fallback
    cpu_gguf_path: Optional[str] = None


class LlmPipeline:
    def __init__(self):
        self.tokenizer = QwenTokenizer()
        self.backend_name: Optional[str] = None
        self.backend: Any = None
        self.metrics: Optional[LlmMetrics] = None

    def select_backend(self, preferred: Optional[str] = None, config: Optional[LlmConfig] = None) -> str:
        """
        Select and instantiate backend by priority. Returns name string.
        preferred: optional explicit name in {"mlc", "onnx_dml", "llama_vulkan", "cpu"}
        """
        cfg = config or LlmConfig()
        order = ["mlc", "onnx_dml", "llama_vulkan", "cpu"]
        if preferred in order:
            order = [preferred] + [x for x in order if x != preferred]

        # Try MLC Vulkan
        if "mlc" in order and self.backend is None:
            if MlcVulkanLLM is not None and cfg.mlc_artifact_dir:
                try:
                    b = MlcVulkanLLM()
                    b.load(cfg.mlc_artifact_dir, micro_batch=cfg.mlc_micro_batch)
                    self.backend = b
                    self.backend_name = "mlc"
                    _log("Selected backend: MLC Vulkan")
                    return self.backend_name
                except Exception as e:
                    _log(f"MLC Vulkan unavailable: {e}")

        # Try ONNX DML
        if "onnx_dml" in order and self.backend is None:
            if OnnxDmlLLM is not None and cfg.onnx_model_dir:
                try:
                    b = OnnxDmlLLM()
                    b.load(cfg.onnx_model_dir)
                    self.backend = b
                    self.backend_name = "onnx_dml"
                    _log("Selected backend: ONNX Runtime DirectML")
                    return self.backend_name
                except Exception as e:
                    _log(f"ONNX DML unavailable: {e}")

        # Try llama.cpp Vulkan
        if "llama_vulkan" in order and self.backend is None:
            if LlamaVulkanLLM is not None and cfg.gguf_path:
                try:
                    b = LlamaVulkanLLM()
                    b.load(cfg.gguf_path, n_ctx=cfg.n_ctx, n_gpu_layers=cfg.n_gpu_layers, quant_preset=cfg.quant_preset)
                    self.backend = b
                    self.backend_name = "llama_vulkan"
                    _log("Selected backend: llama.cpp Vulkan")
                    return self.backend_name
                except Exception as e:
                    _log(f"llama.cpp Vulkan unavailable: {e}")

        # CPU fallback
        if "cpu" in order and self.backend is None:
            if CpuOneDnnLLM is not None and (cfg.cpu_gguf_path or cfg.gguf_path):
                try:
                    b = CpuOneDnnLLM()
                    b.load_llama(cfg.cpu_gguf_path or cfg.gguf_path or "", n_ctx=cfg.n_ctx)
                    self.backend = b
                    self.backend_name = "cpu"
                    _log("Selected backend: CPU oneDNN fallback")
                    return self.backend_name
                except Exception as e:
                    _log(f"CPU fallback unavailable: {e}")

        raise RuntimeError("No available backend. Check installations and config paths.")

    def load_model(self, config: LlmConfig) -> str:
        """
        Load tokenizer and select backend according to priority.
        Returns the backend name.
        """
        # Ensure tokenizer is ready
        if config.tokenizer_path:
            self.tokenizer = QwenTokenizer(tokenizer_path=config.tokenizer_path)
        self.tokenizer.load()
        return self.select_backend(preferred=None, config=config)

    def _encode_prompt(self, prompt_or_messages: Union[str, List[Dict[str, str]]]) -> List[int]:
        if isinstance(prompt_or_messages, str):
            text = prompt_or_messages
        else:
            text = self.tokenizer.apply_chat_template(prompt_or_messages, add_generation_prompt=True)
        return self.tokenizer.encode(text, add_bos=True, add_eos=False)

    def stream(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 128,
        sampling: Union[str, Dict[str, float]] = "greedy",
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """
        Stream decoded text chunks from the selected backend. Yields (text_chunk, info).
        info dict includes timing metrics such as ttft on first token.
        """
        if not self.backend:
            raise RuntimeError("Backend not selected. Call load_model(config) first.")
        self.metrics = LlmMetrics()

        # MLC path expects token ids
        if self.backend_name == "mlc":
            ids = self._encode_prompt(prompt_or_messages)
            first = True
            for tok_id, info in self.backend.generate_stream(ids, max_new_tokens=max_new_tokens, sampling=sampling):
                if first and "ttft" in info:
                    self.metrics.ttft = info["ttft"]
                    first = False
                # In real impl, map id->text incrementally; here stub empty or a placeholder char.
                chunk = "" if tok_id == 0 else ""
                self.metrics.mark_token()
                yield chunk, {"backend": "mlc", **info}
            self.metrics.mark_end()
            return

        # ONNX DML path expects prepared inputs; we feed tokens directly as stub.
        if self.backend_name == "onnx_dml":
            ids = self._encode_prompt(prompt_or_messages)
            inputs = self.backend.prepare(ids, past_kv=None)
            first = True
            for tok_id, _past in self.backend.generate_stream(inputs, max_new_tokens=max_new_tokens, sampling=sampling):
                if first and self.metrics.ttft is None:
                    self.metrics.mark_ttft()
                    first = False
                chunk = "" if tok_id == 0 else ""
                self.metrics.mark_token()
                yield chunk, {"backend": "onnx_dml"}
            self.metrics.mark_end()
            return

        # llama.cpp Vulkan path streams text directly (via stdout parsing)
        if self.backend_name == "llama_vulkan":
            first = True
            for text_chunk, info in self.backend.generate_stream(
                prompt=self._encode_or_passthrough(prompt_or_messages),
                max_new_tokens=max_new_tokens,
                sampling=sampling,
            ):
                if first and "ttft" in info:
                    self.metrics.ttft = info["ttft"]
                    first = False
                self.metrics.mark_token()
                yield text_chunk, {"backend": "llama_vulkan", **info}
            self.metrics.mark_end()
            return

        # CPU fallback (llama.cpp CPU via subprocess) path streams text chunks
        if self.backend_name == "cpu":
            first = True
            for text_chunk, info in self.backend.generate_stream(
                prompt=self._encode_or_passthrough(prompt_or_messages),
                max_new_tokens=max_new_tokens,
            ):
                if first and "ttft" in info:
                    self.metrics.ttft = info["ttft"]
                    first = False
                self.metrics.mark_token()
                yield text_chunk, {"backend": "cpu", **info}
            self.metrics.mark_end()
            return

        raise RuntimeError(f"Unsupported backend: {self.backend_name}")

    def _encode_or_passthrough(self, prompt_or_messages: Union[str, List[Dict[str, str]]]) -> str:
        """
        llama.cpp subprocess CLIs expect raw text prompt. If user passed messages list,
        render to text via chat template.
        """
        if isinstance(prompt_or_messages, str):
            return prompt_or_messages
        return self.tokenizer.apply_chat_template(prompt_or_messages, add_generation_prompt=True)

    def metrics_summary(self) -> Dict[str, Any]:
        if not self.metrics:
            return {}
        return self.metrics.to_dict()