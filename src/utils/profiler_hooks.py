"""
ONNX Runtime profiling hooks (Windows-only) and placeholders for external GPU profilers.

- ort_profile_session: helper to create an ORT InferenceSession with profiling enabled,
  and save the resulting profile JSON on session close.
- rgp_notes: guidance strings for using Radeon GPU Profiler (RGP) with DirectML/Vulkan.

All imports are guarded so module can be imported without ORT installed.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional


def ort_profile_session(
    model_path: str,
    providers: Optional[list] = None,
    provider_options: Optional[list] = None,
    profile_dir: str = "logs/profiles",
    session_options_overrides: Optional[Dict[str, Any]] = None,
):
    """
    Create an onnxruntime.InferenceSession with profiling enabled.
    Returns (session, profile_path). If ORT is unavailable, returns (None, None).

    The profile JSON will be written when session.end_profiling() is called. Caller
    should ensure to call end_profiling() after finishing inference.

    Args:
        model_path: Path to ONNX model
        providers: Provider list (e.g., ["DmlExecutionProvider", "CPUExecutionProvider"])
        provider_options: Options for each provider
        profile_dir: Directory for profile JSON outputs
        session_options_overrides: Dict to override default session options fields
    """
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        print("[profiler_hooks] onnxruntime not available; profiling disabled")
        return None, None

    os.makedirs(profile_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    profile_path = os.path.join(profile_dir, f"ort_profile_{ts}.json")

    so = ort.SessionOptions()
    # Enable full graph optimizations by default
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Enable profiling
    so.enable_profiling = True
    so.profile_file_prefix = os.path.splitext(profile_path)[0]

    if session_options_overrides:
        for k, v in session_options_overrides.items():
            setattr(so, k, v)

    try:
        if providers is not None:
            sess = ort.InferenceSession(model_path, sess_options=so, providers=providers, provider_options=provider_options)
        else:
            sess = ort.InferenceSession(model_path, sess_options=so)
        # Note: ORT writes the actual profile file when end_profiling is called; it returns the path.
        return sess, profile_path
    except Exception as e:
        print(f"[profiler_hooks] Failed to create ORT session with profiling: {e}")
        return None, None


def rgp_notes() -> str:
    """
    Return guidance for using Radeon GPU Profiler (RGP) on Windows with DirectML/Vulkan.
    This is documentation only; actual integration requires external tooling.
    """
    return (
        "Radeon GPU Profiler (RGP) usage notes (Windows):\n"
        "- For DirectML workloads, ensure the app is using the D3D12 backend; RGP can capture GPU traces from D3D12.\n"
        "- For Vulkan workloads (MLC or llama.cpp Vulkan), enable Vulkan validation and ensure drivers are up to date.\n"
        "- Launch the target process, then use RGP to trigger a capture during steady-state decode for best signal.\n"
        "- Keep shader caches warm; first-run compiles can distort timing.\n"
        "- Combine with ONNX Runtime JSON profiler to correlate CPU-side operator scheduling."
    )