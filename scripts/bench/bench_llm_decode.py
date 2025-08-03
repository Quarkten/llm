"""
Windows-only LLM decode benchmark.

Measures TTFT and steady-state tokens/sec for streaming decode across available backends:
- MLC Vulkan (AWQ 4-bit)
- ONNX Runtime DirectML
- llama.cpp Vulkan
- CPU fallback

Prompt lengths: [512, 1024, 2048]
Batch sizes: [1, 4]  (batch>1 is advisory; some backends simulate or iterate)

Outputs:
- Console table lines
- JSON log saved to logs/runs/ with timestamp, including env flags

Usage (PowerShell):
  python scripts/bench/bench_llm_decode.py --config configs/model.qwen2_7b_instruct.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

# Local imports
try:
    from src.llm.pipeline import LlmPipeline, LlmConfig
    from src.llm.tokenizer import QwenTokenizer
except Exception:
    # If running from project root, adjust path
    import sys
    sys.path.append(".")
    from src.llm.pipeline import LlmPipeline, LlmConfig  # type: ignore
    from src.llm.tokenizer import QwenTokenizer  # type: ignore


def ensure_dirs():
    os.makedirs("logs/runs", exist_ok=True)


def make_prompt(tokenizer: QwenTokenizer, n_tokens: int) -> str:
    # Create a synthetic prompt of approx n_tokens using repeated phrase
    text = "You are a helpful assistant. " * (n_tokens // 6 + 1)
    # Trim roughly by encoding length
    ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    while len(ids) > n_tokens:
        ids = ids[:-1]
    return tokenizer.decode(ids, skip_special_tokens=True)


def run_backend_once(pipeline: LlmPipeline, prompt: str, max_new_tokens: int = 64) -> Tuple[float, float]:
    """
    Return (ttft, tok_s)
    """
    ttft = None
    n = 0
    t_last = None
    for chunk, info in pipeline.stream(prompt, max_new_tokens=max_new_tokens, sampling="greedy"):
        if ttft is None and "ttft" in info:
            ttft = info["ttft"]
        n += 1
        t_last = time.perf_counter()
    metrics = pipeline.metrics_summary()
    ttft = ttft if ttft is not None else metrics.get("ttft", None)
    tok_s = metrics.get("tokens_per_sec", 0.0)
    if ttft is None:
        ttft = -1.0
    return float(ttft), float(tok_s)


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("pyyaml not installed. Install with: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_config(yaml_cfg: Dict[str, Any]) -> LlmConfig:
    return LlmConfig(
        tokenizer_path=yaml_cfg.get("tokenizer_path"),
        mlc_artifact_dir=(yaml_cfg.get("mlc") or {}).get("artifact_dir"),
        mlc_micro_batch=(yaml_cfg.get("mlc") or {}).get("micro_batch", 1),
        onnx_model_dir=(yaml_cfg.get("onnx_dml") or {}).get("model_dir"),
        gguf_path=(yaml_cfg.get("llama_vulkan") or {}).get("gguf_path"),
        n_ctx=(yaml_cfg.get("llama_vulkan") or {}).get("n_ctx", 4096),
        n_gpu_layers=(yaml_cfg.get("llama_vulkan") or {}).get("n_gpu_layers", -1),
        quant_preset=(yaml_cfg.get("llama_vulkan") or {}).get("quant_preset", "Q4_K_M"),
        cpu_gguf_path=(yaml_cfg.get("cpu") or {}).get("gguf_path"),
    )


def try_backend(pipeline: LlmPipeline, cfg: LlmConfig, name: str) -> bool:
    try:
        pipeline.backend = None
        pipeline.backend_name = None
        selected = pipeline.select_backend(preferred=name, config=cfg)
        return selected == name
    except Exception as e:
        print(f"[bench] Skip {name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model.qwen2_7b_instruct.yaml")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    ensure_dirs()

    cfg_yaml = load_yaml(args.config)
    cfg = build_config(cfg_yaml)

    pipeline = LlmPipeline()
    # Ensure tokenizer is ready
    pipeline.load_model(cfg)

    tokenizer = pipeline.tokenizer

    prompt_lengths = [512, 1024, 2048]
    batch_sizes = [1, 4]

    results: List[Dict[str, Any]] = []
    backends = ["mlc", "onnx_dml", "llama_vulkan", "cpu"]

    print("backend,prompt_len,batch,ttft_ms,tok_s")
    for be in backends:
        if not try_backend(pipeline, cfg, be):
            continue
        for L in prompt_lengths:
            prompt = make_prompt(tokenizer, L)
            for B in batch_sizes:
                # For B>1, just run sequentially as a placeholder; real batching depends on backend
                ttft_list = []
                tok_s_list = []
                for i in range(B):
                    t_ms, tok_s = run_backend_once(pipeline, prompt, max_new_tokens=args.max_new_tokens)
                    ttft_list.append(t_ms * 1000.0 if t_ms > 0 else -1.0)
                    tok_s_list.append(tok_s)
                ttft_avg = sum(ttft_list) / len(ttft_list)
                tok_s_avg = sum(tok_s_list) / len(tok_s_list)
                print(f"{be},{L},{B},{ttft_avg:.2f},{tok_s_avg:.2f}")
                results.append({
                    "backend": be,
                    "prompt_len": L,
                    "batch": B,
                    "ttft_ms": ttft_avg,
                    "tok_s": tok_s_avg,
                })

    # Save JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("logs", "runs", f"bench_llm_decode_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "env": {k: os.environ.get(k) for k in ["ORT_TUNER_ENABLE", "ORT_TUNER_LOAD", "ORT_TUNER_SAVE", "OMP_NUM_THREADS", "KMP_AFFINITY"]},
            "results": results,
        }, f, indent=2)
    print(f"[bench] Saved: {out_path}")


if __name__ == "__main__":
    main()