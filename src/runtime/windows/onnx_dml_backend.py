"""
ONNX Runtime DirectML backend for Windows.

Implements:
- OnnxDmlLLM: Qwen2.5-7B-Instruct decoder/decoder-with-past inference stubs
  with IO binding hints, pinned memory staging, and simple greedy/sampling APIs.
- OnnxDmlSigLIP: SigLIP SO400M patch14-384 FP16 inference with CPU pre/post.

Design notes:
- All heavy dependencies are optional. If onnxruntime-directml is not installed,
  the classes will initialize in a disabled state and raise RuntimeError with guidance
  when methods are used.
- Uses ORT SessionOptions with ORT_ENABLE_ALL and optional tuner flags controlled by env:
  ORT_TUNER_ENABLE, ORT_TUNER_LOAD, ORT_TUNER_SAVE.
- Attempts to enable DmlExecutionProvider; falls back to CPU if allowed by user.
- KV cache: provides thin shape bookkeeping with app-level "paged-like" chunking metadata,
  but real paging must be implemented in the model/graph.
- This module is Windows-only.

Environment variables honored (set by setup scripts/configs):
- ORT_TUNER_ENABLE=1
- ORT_TUNER_LOAD=path\to\tuner.json
- ORT_TUNER_SAVE=path\to\save_tuner.json
- ORT_DML_DISABLE_FP16=0/1 (if exposed by your build)
- ORT_NUM_THREADS (when CPU EP used)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - optional dep
    ort = None  # type: ignore


def _is_windows() -> bool:
    import sys

    return sys.platform.startswith("win")


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip() not in ("0", "false", "False", "")


def _log(msg: str) -> None:
    print(f"[OnnxDml] {msg}")


@dataclass
class PagedCacheMeta:
    """Simple app-level chunk metadata for KV paging emulation."""
    chunk_size: int
    chunks: List[Tuple[int, int]]  # (start, end) token indices


class OnnxDmlLLM:
    """
    Qwen2.5-7B-Instruct DirectML backend.

    Expected model_dir layout (example):
      model_dir/
        decoder.onnx
        decoder_with_past.onnx
        vocab.json / tokenizer files (handled elsewhere)
    """

    def __init__(self) -> None:
        self.enabled = False
        self.providers: List[str] = []
        self.session_decoder = None
        self.session_decoder_past = None
        self.paged_meta: Optional[PagedCacheMeta] = None
        self.device_info: Dict[str, Any] = {}
        self.io_binding_supported = False

    def _make_session(self, model_path: str) -> Any:
        if ort is None or not _is_windows():
            raise RuntimeError("onnxruntime not available or not on Windows")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # ORT Tuner controls
        if _env_flag("ORT_TUNER_ENABLE"):
            try:
                # Newer ORT exposes optimized patterns via tuners; load/save by env.
                load = os.environ.get("ORT_TUNER_LOAD")
                save = os.environ.get("ORT_TUNER_SAVE")
                if load:
                    _log(f"ORT tuner load: {load}")
                    so.add_session_config_entry("session.load_tuned_result_path", load)
                if save:
                    _log(f"ORT tuner save: {save}")
                    so.add_session_config_entry("session.tuning_results_path", save)
            except Exception as e:
                _log(f"Tuner config ignored: {e}")

        providers = ["DmlExecutionProvider"]
        provider_options: List[Dict[str, Any]] = [{}]
        cpu_fallback = _env_flag("ORT_CPU_FALLBACK", True)
        if cpu_fallback:
            providers.append("CPUExecutionProvider")
            provider_options.append({})

        # DML-specific flags if supported by the build can be added via session configs.
        dml_disable_fp16 = os.environ.get("ORT_DML_DISABLE_FP16")
        if dml_disable_fp16 is not None:
            try:
                so.add_session_config_entry("session.dml.disable_fp16", dml_disable_fp16)
                _log(f"DML disable FP16 = {dml_disable_fp16}")
            except Exception:
                pass

        _log(f"Providers requested: {providers}")
        sess = ort.InferenceSession(model_path, sess_options=so, providers=providers, provider_options=provider_options)
        selected = sess.get_providers()
        _log(f"Providers selected: {selected}")
        return sess

    def load(self, model_dir: str) -> None:
        """
        Load decoder and decoder-with-past models if present.
        """
        try:
            dec = os.path.join(model_dir, "decoder.onnx")
            dec_past = os.path.join(model_dir, "decoder_with_past.onnx")
            if not os.path.exists(dec) and not os.path.exists(dec_past):
                raise FileNotFoundError("decoder.onnx or decoder_with_past.onnx not found")

            if os.path.exists(dec):
                self.session_decoder = self._make_session(dec)
            if os.path.exists(dec_past):
                self.session_decoder_past = self._make_session(dec_past)

            # Detect IO Binding support
            self.io_binding_supported = hasattr(self.session_decoder or self.session_decoder_past, "io_binding")
            self.enabled = True
            self.providers = (self.session_decoder or self.session_decoder_past).get_providers()
            self.device_info = {
                "selected_providers": self.providers,
                "graph_optimization": "ORT_ENABLE_ALL",
                "tuner_enable": _env_flag("ORT_TUNER_ENABLE"),
                "tuner_load": os.environ.get("ORT_TUNER_LOAD"),
                "tuner_save": os.environ.get("ORT_TUNER_SAVE"),
            }
            _log(f"Loaded. IO binding supported: {self.io_binding_supported}")
        except Exception as e:
            _log(f"Load failed: {e}")
            self.enabled = False
            raise

    def prepare(self, prompt_tokens: List[int], past_kv: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare inputs for first token. For decoder graph: run full prompt.
        For decoder-with-past: build attention mask and position ids, return outputs incl. new past_kv.
        """
        if not self.enabled:
            raise RuntimeError("OnnxDmlLLM not loaded")
        inputs = {
            "input_ids": prompt_tokens,
            # Placeholder shapes and masks; real shapes depend on exported graphs.
            "attention_mask": [1] * len(prompt_tokens),
        }
        # In production, bind with ortvalue and io_binding to keep on GPU.
        return inputs

    def ingest(self, prompt: str) -> Dict[str, Any]:
        """
        Tokenization happens outside; this stub expects tokens. Provided here for interface symmetry.
        """
        raise NotImplementedError("Use tokenizer to convert prompt to tokens, then call prepare()")

    def decode_step(self, past_kv: Optional[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        """
        Perform a single decode step using decoder-with-past if available, else fallback.
        Returns (next_token_id, new_past_kv)
        """
        if not self.enabled:
            raise RuntimeError("OnnxDmlLLM not loaded")

        # Stub logic. A real implementation would:
        # - IOBind past_kv + last_token to GPU
        # - Run inference and sample next token from logits
        # Here we return EOS (id=0) as a placeholder.
        return 0, past_kv or {}

    def generate_stream(
        self,
        inputs: Dict[str, Any],
        max_new_tokens: int,
        sampling: str = "greedy",
    ) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """
        Streaming generation yielding tokens and updated past_kv.
        """
        past = inputs.get("past_kv")
        ttft_marked = False
        for i in range(max_new_tokens):
            tok, past = self.decode_step(past)
            if not ttft_marked:
                _log("TTFT reached (stub)")
                ttft_marked = True
            yield tok, past
            # Allow cooperative scheduling
            time.sleep(0)

    # Optional: expose lightweight paged cache metadata for the app layer to emulate chunking.
    def set_paged_cache(self, chunk_size: int, total_len: int) -> None:
        chunks = []
        for s in range(0, total_len, chunk_size):
            chunks.append((s, min(total_len, s + chunk_size)))
        self.paged_meta = PagedCacheMeta(chunk_size=chunk_size, chunks=chunks)
        _log(f"Paged-like cache meta set: {len(chunks)} chunks")


class OnnxDmlSigLIP:
    """
    SigLIP 384x384 FP16 inference (DirectML).

    Expected:
      - Single ONNX graph accepting NCHW FP16, normalized.
      - Preprocess done on CPU; tensors uploaded via IO binding when possible.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.session = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self.providers: List[str] = []

    def load(self, onnx_path: str) -> None:
        if ort is None or not _is_windows():
            raise RuntimeError("onnxruntime not available or not on Windows")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(onnx_path)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.providers = self.session.get_providers()
        _log(f"SigLIP providers: {self.providers}")

        # Resolve IO names
        io = self.session.get_inputs()
        if not io:
            raise RuntimeError("Model has no inputs")
        self.input_name = io[0].name
        out = self.session.get_outputs()
        if not out:
            raise RuntimeError("Model has no outputs")
        self.output_name = out[0].name
        self.enabled = True

    def run(self, batch_nchw_fp16) -> Any:
        """
        Run batched inference. Expects numpy array shape [B,3,384,384], dtype float16.
        Returns embeddings as numpy array.
        """
        if not self.enabled:
            raise RuntimeError("OnnxDmlSigLIP not loaded")
        feeds = {self.input_name: batch_nchw_fp16}
        outs = self.session.run([self.output_name], feeds)
        return outs[0]