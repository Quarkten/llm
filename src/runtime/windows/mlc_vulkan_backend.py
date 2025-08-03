"""
MLC LLM Vulkan backend (Windows-only) for Qwen2.5-7B (AWQ 4-bit).

This wraps the optional MLC Python API to load precompiled TVM/Vulkan artifacts and
exposes a streaming generation API with TTFT and tokens/sec metrics.

Functional stub: if mlc_llm is unavailable or artifacts missing, methods will
raise with actionable messages. Logging prints device name and shader cache path
when available.

Expected artifact_dir layout (example, produced by MLC build pipeline):
  artifact_dir/
    params
    mod.so / mod.dll (Windows)
    vm_model.json
    vocab files external to this module (handled by tokenizer)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional, Tuple


def _is_windows() -> bool:
    import sys

    return sys.platform.startswith("win")


def _log(msg: str) -> None:
    print(f"[MLC-Vulkan] {msg}")


@dataclass
class MlcDeviceInfo:
    device_name: str = "Vulkan Device"
    shader_cache_path: Optional[str] = None


class MlcVulkanLLM:
    def __init__(self) -> None:
        self.enabled = False
        self.vm = None
        self.device_info = MlcDeviceInfo()
        self.kv_cfg: Dict[str, Any] = {}
        self.micro_batch: int = 1

    def _import_mlc(self):
        try:
            import mlc_llm  # type: ignore
            return mlc_llm
        except Exception as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "mlc_llm package not found. Install MLC runtime for Windows with Vulkan support."
            ) from e

    def load(self, artifact_dir: str, micro_batch: int = 1, kv_cfg: Optional[Dict[str, Any]] = None) -> None:
        if not _is_windows():
            raise RuntimeError("Windows-only backend")

        mlc_llm = self._import_mlc()
        if not os.path.isdir(artifact_dir):
            raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")

        # Device / shader cache info (APIs may vary by version; guarded)
        try:
            get = getattr(mlc_llm, "get_device", None)
            if callable(get):
                dev = get("vulkan")
                self.device_info.device_name = str(dev)
            cache_fn = getattr(mlc_llm, "shader_cache_path", None)
            if callable(cache_fn):
                self.device_info.shader_cache_path = cache_fn()
        except Exception:
            pass

        _log(f"Device: {self.device_info.device_name}")
        if self.device_info.shader_cache_path:
            _log(f"Shader cache: {self.device_info.shader_cache_path}")

        # Load precompiled module
        # API shape differs across versions; below is a schematic stub.
        try:
            create = getattr(mlc_llm, "create", None)
            if callable(create):
                self.vm = create(artifact_dir, device="vulkan", dtype="float16")
            else:
                # Fallback hypothetical API
                self.vm = mlc_llm.MLCEngine(artifact_dir, device="vulkan", dtype="float16")
        except Exception as e:
            raise RuntimeError(f"Failed to load MLC artifacts from {artifact_dir}: {e}")

        self.kv_cfg = kv_cfg or {}
        self.micro_batch = micro_batch
        self.enabled = True
        _log("Loaded MLC Vulkan artifacts (AWQ 4-bit expected)")

    def generate_stream(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        sampling: Dict[str, Any] | str = "greedy",
    ) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """
        Yield (token_id, info). 'info' contains timing markers for TTFT and throughput.

        This is a functional stub: if the actual vm API provides token callback streaming,
        wire it here; otherwise we simulate timings.
        """
        if not self.enabled:
            raise RuntimeError("MlcVulkanLLM not loaded")

        t0 = time.perf_counter()
        ttft_marked = False

        # Real usage would pass prompt_ids and configs into vm.generate with a token callback.
        # Simulate streaming for now.
        for i in range(max_new_tokens):
            # After first decode, mark TTFT
            if not ttft_marked:
                ttft = time.perf_counter() - t0
                info = {"ttft": ttft, "toks": i + 1}
                ttft_marked = True
            else:
                info = {"toks": i + 1}
            # Placeholder token id (0 = eos)
            tok = 0
            yield tok, info
            time.sleep(0)  # cooperative yield

    def report_device(self) -> MlcDeviceInfo:
        return self.device_info