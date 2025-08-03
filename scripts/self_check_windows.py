#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Windows self-check script:
- Prints system/platform info and GPU/driver summary
- Probes availability of ONNX Runtime DirectML, MLC Vulkan, llama.cpp Vulkan, and CPU fallback
- Runs a tiny inference for each available backend (best-effort, non-fatal if artifacts missing)
- Writes a JSON summary to logs/runs/self_check_<timestamp>.json

Usage:
  python scripts/self_check_windows.py --model-config configs/model.qwen2_7b_instruct.yaml --runtime-config configs/runtime.windows.yaml --max-new 16 --prompt "Hello"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import datetime
import platform
from typing import Any, Dict, Optional

# Optional YAML loader
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# Guarded imports for project-local utilities/backends
_device_check = None
_metrics = None
onnx_backend_mod = None
mlc_backend_mod = None
llama_backend_mod = None
cpu_backend_mod = None

def guarded_imports():
    global _device_check, _metrics, onnx_backend_mod, mlc_backend_mod, llama_backend_mod, cpu_backend_mod

    # Utilities
    try:
        from src.utils import device_check as dc  # noqa: F401
        _device_check = dc
    except Exception as e:
        _device_check = None
        print(f"[WARN] Could not import src.utils.device_check: {e}")

    try:
        from src.utils import metrics as mt  # noqa: F401
        _metrics = mt
    except Exception as e:
        _metrics = None
        print(f"[WARN] Could not import src.utils.metrics: {e}")

    # Backends
    try:
        from src.runtime.windows import onnx_dml_backend as odb  # noqa: F401
        onnx_backend_mod = odb
    except Exception as e:
        onnx_backend_mod = None
        print(f"[INFO] ONNX DML backend not importable: {e}")

    try:
        from src.runtime.windows import mlc_vulkan_backend as mvb  # noqa: F401
        mlc_backend_mod = mvb
    except Exception as e:
        mlc_backend_mod = None
        print(f"[INFO] MLC Vulkan backend not importable: {e}")

    try:
        from src.runtime.windows import llama_vulkan_backend as lvb  # noqa: F401
        llama_backend_mod = lvb
    except Exception as e:
        llama_backend_mod = None
        print(f"[INFO] llama.cpp Vulkan backend not importable: {e}")

    try:
        from src.runtime.windows import cpu_onednn_backend as cob  # noqa: F401
        cpu_backend_mod = cob
    except Exception as e:
        cpu_backend_mod = None
        print(f"[INFO] CPU oneDNN backend not importable: {e}")


def ensure_dirs():
    os.makedirs("logs", exist_ok=True)
    os.makedirs(os.path.join("logs", "runs"), exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    if not yaml:
        print("[WARN] PyYAML not installed; returning empty dict for config load.")
        return {}
    if not os.path.isfile(path):
        print(f"[WARN] YAML file missing: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARN] Failed to parse YAML {path}: {e}")
            return {}


def system_summary() -> Dict[str, Any]:
    py_ver = sys.version.replace("\n", " ")
    plat = platform.platform()
    info: Dict[str, Any] = {
        "python": py_ver,
        "platform": plat,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }

    # GPU/driver summary from device_check if available
    if _device_check and hasattr(_device_check, "summarize_device"):
        try:
            dev = _device_check.summarize_device()
            info["device_summary"] = dev
        except Exception as e:
            info["device_summary_error"] = str(e)
    else:
        info["device_summary_error"] = "device_check not available"

    return info


def try_mlc_vulkan(prompt: str, max_new: int, model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort tiny generation with MLC Vulkan."""
    out: Dict[str, Any] = {
        "backend": "mlc_vulkan",
        "status": "skipped",
        "device": None,
        "ttft_ms": None,
        "tok_s": None,
        "error": None,
    }
    if mlc_backend_mod is None:
        out["status"] = "skipped"
        out["error"] = "Module not importable"
        return out

    # Expect the backend to provide an interface; keep guarded
    try:
        # Heuristics: look for an init/load and a generate-like API
        # The specific keys come from configs/runtime.windows.yaml and model yaml
        model_path = model_cfg.get("mlc", {}).get("artifact_path") or model_cfg.get("mlc_artifact_path")
        if not model_path or not os.path.exists(model_path):
            out["status"] = "skipped"
            out["error"] = f"MLC artifact missing: {model_path}"
            return out

        start_init = time.perf_counter()
        # Hypothetical API surfaces; adjust to your backend's real API as needed
        # Try common entry points
        if hasattr(mlc_backend_mod, "load_model"):
            engine = mlc_backend_mod.load_model(model_path=model_path, runtime_cfg=runtime_cfg)
        elif hasattr(mlc_backend_mod, "MLCEngine"):
            engine = mlc_backend_mod.MLCEngine(model_path=model_path, runtime_cfg=runtime_cfg)  # type: ignore
            if hasattr(engine, "load") and callable(getattr(engine, "load")):
                engine.load()
        else:
            out["status"] = "skipped"
            out["error"] = "MLC backend missing known entry points (load_model/MLCEngine)"
            return out

        device_name = None
        if hasattr(engine, "device_name"):
            try:
                device_name = engine.device_name  # type: ignore
            except Exception:
                device_name = None
        out["device"] = device_name

        ttft_ms = None
        gen_tokens = 0
        t0 = time.perf_counter()
        first_token_time: Optional[float] = None
        # Try to generate small number of tokens
        wanted = max(8, min(max_new, 16))
        text_accum = []
        if hasattr(engine, "generate_stream"):
            for i, tok in enumerate(engine.generate_stream(prompt, max_new_tokens=wanted)):  # type: ignore
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                gen_tokens += 1
                text_accum.append(str(tok))
        elif hasattr(engine, "generate"):
            # returns full text; cannot stream TTFT precisely—approximate
            res = engine.generate(prompt, max_new_tokens=wanted)  # type: ignore
            gen_tokens = wanted
            first_token_time = time.perf_counter()  # approximate as end
        else:
            out["status"] = "skipped"
            out["error"] = "MLC engine lacks generate/generate_stream"
            return out

        total = time.perf_counter() - t0
        if first_token_time is not None:
            ttft_ms = (first_token_time - t0) * 1000.0
        tok_s = (gen_tokens / total) if total > 0 and gen_tokens > 0 else None

        out["status"] = "ok"
        out["ttft_ms"] = round(ttft_ms, 3) if ttft_ms is not None else None
        out["tok_s"] = round(tok_s, 3) if tok_s is not None else None
        return out
    except Exception as e:
        out["status"] = "error"
        out["error"] = str(e)
        return out


def try_onnx_dml(prompt: str, max_new: int, model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort ONNX Runtime DML tiny inference."""
    out: Dict[str, Any] = {
        "backend": "onnx_dml",
        "status": "skipped",
        "providers": None,
        "dml_enabled": False,
        "ttft_ms": None,
        "tok_s": None,
        "error": None,
    }
    if onnx_backend_mod is None:
        out["error"] = "Module not importable"
        return out

    try:
        # Check available providers
        providers = None
        dml_enabled = False
        try:
            import onnxruntime as ort  # type: ignore
            providers = ort.get_available_providers()
            dml_enabled = "DmlExecutionProvider" in (providers or [])
        except Exception as e:
            providers = None
            dml_enabled = False
            out["status"] = "skipped"
            out["providers"] = providers
            out["dml_enabled"] = dml_enabled
            out["error"] = f"onnxruntime not available: {e}"
            return out

        out["providers"] = providers
        out["dml_enabled"] = dml_enabled

        if not dml_enabled:
            out["status"] = "skipped"
            out["error"] = "DmlExecutionProvider not available"
            return out

        # Try to find session paths from model config
        # Expect something like encoder/decoder or a single model path
        model_paths = model_cfg.get("onnx", {}) or {}
        encoder_path = model_paths.get("encoder_path")
        decoder_path = model_paths.get("decoder_path")
        single_path = model_paths.get("model_path")
        if not any([encoder_path, decoder_path, single_path]):
            out["status"] = "skipped"
            out["error"] = "No ONNX paths in model config"
            return out

        def path_ok(p: Optional[str]) -> bool:
            return bool(p and os.path.exists(p))

        if single_path and not path_ok(single_path):
            out["status"] = "skipped"
            out["error"] = f"ONNX file missing: {single_path}"
            return out

        if (encoder_path and not path_ok(encoder_path)) or (decoder_path and not path_ok(decoder_path)):
            out["status"] = "skipped"
            out["error"] = f"ONNX encoder/decoder path missing: {encoder_path}, {decoder_path}"
            return out

        # Build minimal session(s)
        import onnxruntime as ort  # type: ignore
        sess_options = ort.SessionOptions()
        # Optional DirectML tuners / optim flags from runtime config
        dml_tune = runtime_cfg.get("onnx", {}).get("dml_tune", {})
        if dml_tune:
            # Non-fatal if option not recognized
            for k, v in dml_tune.items():
                try:
                    setattr(sess_options, k, v)
                except Exception:
                    pass

        providers_req = ["DmlExecutionProvider"]
        if single_path:
            t0 = time.perf_counter()
            sess = ort.InferenceSession(single_path, sess_options, providers=providers_req)
            # Try one run; without tokenizer/tensors, just check metadata or a dummy io-binding path
            # If shapes not known, we won't fabricate inputs here. We measure TTFT as session init.
            ttft_ms = (time.perf_counter() - t0) * 1000.0
            out["ttft_ms"] = round(ttft_ms, 3)
            out["status"] = "ok"
            # If your backend exposes an ingest/generate helper, try it guarded
            if hasattr(onnx_backend_mod, "tiny_generate"):
                try:
                    t1 = time.perf_counter()
                    gen_tokens = onnx_backend_mod.tiny_generate(sess, prompt=prompt, max_new=min(max_new, 8))  # type: ignore
                    total = time.perf_counter() - t1
                    if total > 0 and isinstance(gen_tokens, int) and gen_tokens > 0:
                        out["tok_s"] = round(gen_tokens / total, 3)
                except Exception as e:
                    out["error"] = f"tiny_generate failed: {e}"
            return out
        else:
            # Encoder-decoder style; init sessions
            t0 = time.perf_counter()
            enc_sess = ort.InferenceSession(encoder_path, sess_options, providers=providers_req)  # type: ignore
            dec_sess = ort.InferenceSession(decoder_path, sess_options, providers=providers_req)  # type: ignore
            ttft_ms = (time.perf_counter() - t0) * 1000.0
            out["ttft_ms"] = round(ttft_ms, 3)
            out["status"] = "ok"
            if hasattr(onnx_backend_mod, "tiny_generate_seq2seq"):
                try:
                    t1 = time.perf_counter()
                    gen_tokens = onnx_backend_mod.tiny_generate_seq2seq(
                        enc_sess, dec_sess, prompt=prompt, max_new=min(max_new, 8)  # type: ignore
                    )
                    total = time.perf_counter() - t1
                    if total > 0 and isinstance(gen_tokens, int) and gen_tokens > 0:
                        out["tok_s"] = round(gen_tokens / total, 3)
                except Exception as e:
                    out["error"] = f"tiny_generate_seq2seq failed: {e}"
            return out
    except Exception as e:
        out["status"] = "error"
        out["error"] = str(e)
        return out


def try_llama_vulkan(prompt: str, max_new: int, model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort llama.cpp Vulkan tiny inference via Python binding or subprocess if available."""
    out: Dict[str, Any] = {
        "backend": "llama_vulkan",
        "status": "skipped",
        "device": None,
        "ttft_ms": None,
        "tok_s": None,
        "error": None,
    }
    if llama_backend_mod is None:
        out["error"] = "Module not importable"
        return out

    try:
        gguf_path = model_cfg.get("llama", {}).get("gguf_path") or model_cfg.get("gguf_path")
        if not gguf_path or not os.path.exists(gguf_path):
            out["status"] = "skipped"
            out["error"] = f"GGUF missing: {gguf_path}"
            return out

        # Prefer Python binding if backend exposes one, else subprocess path
        # Respect env hint for F16 path
        os.environ.setdefault("GGML_VK_F16", "1")

        # Hypothetical direct Python API
        if hasattr(llama_backend_mod, "LlamaVulkan"):
            t0 = time.perf_counter()
            llm = llama_backend_mod.LlamaVulkan(model_path=gguf_path, runtime_cfg=runtime_cfg)  # type: ignore
            # Capture device string if available
            dev = None
            if hasattr(llm, "device"):
                try:
                    dev = llm.device  # type: ignore
                except Exception:
                    dev = None
            out["device"] = dev

            first_token_time: Optional[float] = None
            gen_tokens = 0
            wanted = max(8, min(max_new, 16))
            if hasattr(llm, "generate_stream"):
                for i, tok in enumerate(llm.generate_stream(prompt=prompt, max_new_tokens=wanted)):  # type: ignore
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    gen_tokens += 1
            elif hasattr(llm, "generate"):
                # Non-stream approximate
                _ = llm.generate(prompt=prompt, max_new_tokens=wanted)  # type: ignore
                first_token_time = time.perf_counter()
                gen_tokens = wanted
            else:
                out["status"] = "skipped"
                out["error"] = "llama backend lacks generate/generate_stream"
                return out

            total = time.perf_counter() - t0
            ttft_ms = (first_token_time - t0) * 1000.0 if first_token_time else None
            tok_s = (gen_tokens / total) if total > 0 and gen_tokens > 0 else None

            out["status"] = "ok"
            out["ttft_ms"] = round(ttft_ms, 3) if ttft_ms is not None else None
            out["tok_s"] = round(tok_s, 3) if tok_s is not None else None
            return out

        # Fallback: subprocess using llama.cpp cli if backend exposes helper
        if hasattr(llama_backend_mod, "run_cli_generate"):
            t0 = time.perf_counter()
            res = llama_backend_mod.run_cli_generate(  # type: ignore
                model_path=gguf_path, prompt=prompt, max_new_tokens=max(8, min(max_new, 16))
            )
            # Expect res to contain timing/device fields if implemented
            out["status"] = "ok"
            out["device"] = res.get("device")
            if "ttft_ms" in res:
                out["ttft_ms"] = res["ttft_ms"]
            if "tok_s" in res:
                out["tok_s"] = res["tok_s"]
            # If not provided, approximate
            if out["ttft_ms"] is None:
                out["ttft_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            return out

        out["status"] = "skipped"
        out["error"] = "No Python binding or CLI helper available in llama backend"
        return out
    except Exception as e:
        out["status"] = "error"
        out["error"] = str(e)
        return out


def try_cpu_fallback(prompt: str, max_new: int, model_cfg: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort CPU minimal generation using GGUF or ONNX CPU EP, depending on availability."""
    out: Dict[str, Any] = {
        "backend": "cpu_fallback",
        "status": "skipped",
        "threads_env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "KMP_AFFINITY": os.environ.get("KMP_AFFINITY"),
            "KMP_BLOCKTIME": os.environ.get("KMP_BLOCKTIME"),
        },
        "ttft_ms": None,
        "tok_s": None,
        "error": None,
    }
    if cpu_backend_mod is None:
        out["error"] = "Module not importable"
        return out

    try:
        # Try GGUF path first through a hypothetical simple CPU runner exposed by backend
        gguf_path = model_cfg.get("cpu", {}).get("gguf_path") or model_cfg.get("gguf_path")
        if gguf_path and os.path.exists(gguf_path) and hasattr(cpu_backend_mod, "tiny_generate_gguf"):
            t0 = time.perf_counter()
            first_token_time: Optional[float] = None
            gen_tokens = cpu_backend_mod.tiny_generate_gguf(  # type: ignore
                model_path=gguf_path, prompt=prompt, max_new=min(max_new, 8),
                on_first_token=lambda: (globals().update(_ftt={"t": time.perf_counter()}))  # hacky callback
            )
            # read callback time if set
            ft = globals().get("_ftt", {}).get("t", None)
            total = time.perf_counter() - t0
            ttft_ms = (ft - t0) * 1000.0 if ft else None
            tok_s = (gen_tokens / total) if total > 0 and isinstance(gen_tokens, int) and gen_tokens > 0 else None
            out["status"] = "ok"
            out["ttft_ms"] = round(ttft_ms, 3) if ttft_ms is not None else None
            out["tok_s"] = round(tok_s, 3) if tok_s is not None else None
            return out

        # Else try ONNX CPU EP one step using same ONNX assets if present
        onnx_paths = model_cfg.get("onnx", {}) or {}
        single_path = onnx_paths.get("model_path")
        encoder_path = onnx_paths.get("encoder_path")
        decoder_path = onnx_paths.get("decoder_path")

        def path_ok(p: Optional[str]) -> bool:
            return bool(p and os.path.exists(p))

        import importlib
        try:
            ort = importlib.import_module("onnxruntime")
        except Exception:
            ort = None

        if ort:
            sess_options = getattr(ort, "SessionOptions", lambda: None)()
            providers_req = ["CPUExecutionProvider"]
            if single_path and path_ok(single_path):
                t0 = time.perf_counter()
                _ = ort.InferenceSession(single_path, sess_options, providers=providers_req)  # type: ignore
                ttft_ms = (time.perf_counter() - t0) * 1000.0
                out["status"] = "ok"
                out["ttft_ms"] = round(ttft_ms, 3)
                return out
            if encoder_path and decoder_path and path_ok(encoder_path) and path_ok(decoder_path):
                t0 = time.perf_counter()
                _ = ort.InferenceSession(encoder_path, sess_options, providers=providers_req)  # type: ignore
                _ = ort.InferenceSession(decoder_path, sess_options, providers=providers_req)  # type: ignore
                ttft_ms = (time.perf_counter() - t0) * 1000.0
                out["status"] = "ok"
                out["ttft_ms"] = round(ttft_ms, 3)
                return out

        out["status"] = "skipped"
        out["error"] = "No CPU-ready assets found (GGUF or ONNX); or onnxruntime CPU EP unavailable"
        return out
    except Exception as e:
        out["status"] = "error"
        out["error"] = str(e)
        return out


def parse_args():
    p = argparse.ArgumentParser(description="Windows self-check for LLM backends")
    p.add_argument("--model-config", type=str, default="configs/model.qwen2_7b_instruct.yaml", help="Path to model YAML")
    p.add_argument("--runtime-config", type=str, default="configs/runtime.windows.yaml", help="Path to runtime YAML")
    p.add_argument("--max-new", type=int, default=16, help="Max new tokens for generation backends")
    p.add_argument("--prompt", type=str, default="Hello", help="Prompt text")
    p.add_argument("--json-out", type=str, default="", help="Optional override for JSON output path")
    return p.parse_args()


def main():
    ensure_dirs()
    guarded_imports()

    args = parse_args()
    model_cfg = load_yaml(args.model_config)
    runtime_cfg = load_yaml(args.runtime_config)

    print("=== System Summary ===")
    sys_info = system_summary()
    print(json.dumps(sys_info, indent=2))

    results = {
        "system": sys_info,
        "backends": [],
    }

    # ONNX DML providers quick print (even if not running inference)
    if onnx_backend_mod is None:
        print("[ONNX DML] module not importable")
    else:
        try:
            import onnxruntime as ort  # type: ignore
            providers = ort.get_available_providers()
            print(f"[ONNX DML] Available providers: {providers}")
            print(f"[ONNX DML] DmlExecutionProvider enabled: {'DmlExecutionProvider' in providers}")
        except Exception as e:
            print(f"[ONNX DML] Could not query providers: {e}")

    # MLC Vulkan presence
    if mlc_backend_mod is None:
        print("[MLC Vulkan] not importable")
    else:
        dev_str = None
        try:
            if hasattr(mlc_backend_mod, "select_device_name"):
                dev_str = mlc_backend_mod.select_device_name()  # type: ignore
        except Exception as e:
            dev_str = f"query_error: {e}"
        print(f"[MLC Vulkan] present; selected device: {dev_str}")

    # llama.cpp Vulkan presence
    if llama_backend_mod is None:
        print("[llama.cpp Vulkan] not importable")
    else:
        dev_str = None
        try:
            if hasattr(llama_backend_mod, "detect_device_string"):
                dev_str = llama_backend_mod.detect_device_string()  # type: ignore
        except Exception as e:
            dev_str = f"query_error: {e}"
        print(f"[llama.cpp Vulkan] present; device: {dev_str}")

    # CPU fallback readiness
    if cpu_backend_mod is None:
        print("[CPU fallback] not importable")
    else:
        print("[CPU fallback] present")
        print(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
        print(f"  MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")
        print(f"  KMP_AFFINITY={os.environ.get('KMP_AFFINITY')}")
        print(f"  KMP_BLOCKTIME={os.environ.get('KMP_BLOCKTIME')}")

    print("\n=== Tiny Inference Checks (best-effort) ===")

    # MLC Vulkan
    mlc_res = try_mlc_vulkan(args.prompt, args.max_new, model_cfg, runtime_cfg)
    print(f"[MLC Vulkan] status={mlc_res.get('status')}, device={mlc_res.get('device')}, ttft_ms={mlc_res.get('ttft_ms')}, tok_s={mlc_res.get('tok_s')}, error={mlc_res.get('error')}")
    results["backends"].append(mlc_res)

    # ONNX DML
    onnx_res = try_onnx_dml(args.prompt, args.max_new, model_cfg, runtime_cfg)
    print(f"[ONNX DML] status={onnx_res.get('status')}, dml_enabled={onnx_res.get('dml_enabled')}, ttft_ms={onnx_res.get('ttft_ms')}, tok_s={onnx_res.get('tok_s')}, error={onnx_res.get('error')}")
    results["backends"].append(onnx_res)

    # llama.cpp Vulkan
    llama_res = try_llama_vulkan(args.prompt, args.max_new, model_cfg, runtime_cfg)
    print(f"[llama.cpp Vulkan] status={llama_res.get('status')}, device={llama_res.get('device')}, ttft_ms={llama_res.get('ttft_ms')}, tok_s={llama_res.get('tok_s')}, error={llama_res.get('error')}")
    results["backends"].append(llama_res)

    # CPU fallback
    cpu_res = try_cpu_fallback(args.prompt, args.max_new, model_cfg, runtime_cfg)
    print(f"[CPU fallback] status={cpu_res.get('status')}, ttft_ms={cpu_res.get('ttft_ms')}, tok_s={cpu_res.get('tok_s')}, error={cpu_res.get('error')}")
    results["backends"].append(cpu_res)

    # Write JSON summary
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = args.json_out.strip() or os.path.join("logs", "runs", f"self_check_{ts}.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary written to: {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to write JSON summary to {out_path}: {e}")

    # Exit code non-fatal; success even if some backends skipped
    return 0


if __name__ == "__main__":
    sys.exit(main())