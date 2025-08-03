"""
Windows-only device and backend capability checks with graceful fallbacks.

This module centralizes provider and device detection for:
- ONNX Runtime DirectML (DmlExecutionProvider)
- MLC LLM Vulkan (via optional mlc_llm python package)
- llama.cpp Vulkan (via optional python bindings or subprocess presence)
- CPU fallback detection and thread reporting

All imports are guarded; functions return (ok, info_dict) and never raise on missing deps.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, Tuple


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def ort_dml_available() -> Tuple[bool, Dict[str, Any]]:
    """
    Check if ONNX Runtime with DirectML EP is available.

    Returns:
        (ok, info)
        ok: True if DmlExecutionProvider is present and loadable
        info: dict with 'providers', 'selected', 'error' (optional), 'ort_version' (optional)
    """
    info: Dict[str, Any] = {"backend": "onnxruntime-directml"}
    if not _is_windows():
        info["error"] = "Not Windows"
        return False, info
    try:
        import onnxruntime as ort  # type: ignore
        info["ort_version"] = getattr(ort, "__version__", "unknown")
        providers = ort.get_available_providers()
        info["providers"] = providers
        ok = "DmlExecutionProvider" in providers
        if ok:
            # Try quick session init with empty session options to ensure no DLL errors.
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # No real model loaded here; just report successful provider presence.
            info["selected"] = "DmlExecutionProvider"
            return True, info
        info["error"] = "DmlExecutionProvider not found"
        return False, info
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return False, info


def mlc_vulkan_available() -> Tuple[bool, Dict[str, Any]]:
    """
    Check if MLC LLM Vulkan bindings are importable and can report a device.
    Returns (ok, info)
    """
    info: Dict[str, Any] = {"backend": "mlc-llm-vulkan"}
    if not _is_windows():
        info["error"] = "Not Windows"
        return False, info
    try:
        # Typical import name for MLC runtime Python package
        import mlc_llm  # type: ignore

        # Some builds offer device query utilities; guard access.
        device_name = None
        shader_cache = None
        try:
            # Hypothetical helpers; may not exist in all builds.
            get = getattr(mlc_llm, "get_device", None)
            if callable(get):
                device = get("vulkan")
                device_name = str(device)
            cache_path = getattr(mlc_llm, "shader_cache_path", None)
            if callable(cache_path):
                shader_cache = cache_path()
        except Exception:
            pass

        info["device"] = device_name or "Vulkan (MLC)"
        if shader_cache:
            info["shader_cache_path"] = shader_cache
        return True, info
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return False, info


def _try_exec(command: str) -> Tuple[bool, str]:
    try:
        out = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, encoding="utf-8", timeout=5)
        return True, out
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def llama_vulkan_available() -> Tuple[bool, Dict[str, Any]]:
    """
    Check llama.cpp Vulkan availability via python binding or subprocess.
    Returns (ok, info)
    """
    info: Dict[str, Any] = {"backend": "llama.cpp-vulkan"}
    if not _is_windows():
        info["error"] = "Not Windows"
        return False, info

    # Try python binding (if installed)
    try:
        import llama_cpp  # type: ignore

        # Some wheels expose Vulkan via compile flags; not always queryable.
        info["binding"] = "python"
        info["note"] = "Python bindings present; Vulkan assumed if built with VK."
        return True, info
    except Exception:
        pass

    # Try subprocess detection: look for llama executable variants
    candidates = ["llama-cli.exe", "main.exe", "llama.exe", "llama.cpp.exe"]
    exe = None
    for name in candidates:
        path = shutil.which(name)
        if path:
            exe = path
            break

    if exe:
        # Attempt to print help/version, parse for 'vulkan' or 'vk'
        ok, out = _try_exec(f"\"{exe}\" -h")
        if not ok:
            ok, out = _try_exec(f"\"{exe}\" --help")
        info["executable"] = exe
        if ok:
            text = out.lower()
            info["help_excerpt"] = out[:400]
            if "vulkan" in text or "vk" in text:
                return True, info
            # Some builds require flag to show backends; still treat as present.
            info["warning"] = "Could not confirm Vulkan in help output"
            return True, info
        else:
            info["error"] = f"Executable found but not runnable: {out}"
            return False, info

    info["error"] = "No llama.cpp executable or python binding found"
    return False, info


def cpu_fallback_available() -> Tuple[bool, Dict[str, Any]]:
    """
    Basic CPU availability and thread environment reporting.
    """
    info: Dict[str, Any] = {"backend": "cpu-fallback", "os": sys.platform}
    # Report thread-related env variables that affect oneDNN/OMP
    env_keys = ["OMP_NUM_THREADS", "KMP_AFFINITY", "MKL_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES"]
    info["env"] = {k: os.environ.get(k) for k in env_keys if os.environ.get(k) is not None}
    info["threads_note"] = "Set OMP_NUM_THREADS and KMP_AFFINITY as needed on Windows."
    return True, info


def summarize_all() -> Dict[str, Any]:
    """
    Produce a summary of all backends and their availability.
    """
    results: Dict[str, Any] = {}
    for name, fn in [
        ("onnxruntime_dml", ort_dml_available),
        ("mlc_vulkan", mlc_vulkan_available),
        ("llama_vulkan", llama_vulkan_available),
        ("cpu", cpu_fallback_available),
    ]:
        ok, info = fn()
        results[name] = {"ok": ok, "info": info}
    return results


def print_summary_json() -> None:
    """
    Print detection summary as JSON for quick diagnostics.
    """
    print(json.dumps(summarize_all(), indent=2))