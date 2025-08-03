"""
Windows-only LLM prompt ingestion benchmark.

Measures:
- First token latency (TTFT) sensitivity to prompt length
- Prompt ingestion throughput (approx tokens/s until first token)

Prompt lengths: [512, 1024, 2048]
Backends attempted in priority order or individually.

Outputs:
- Console CSV lines
- JSON log in logs/runs/

Usage:
  python scripts/bench/bench_llm_ingest.py --config configs/model.qwen2_7b_instruct.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

# Local imports with fallback sys.path handling
try:
    from src.llm.pipeline import LlmPipeline, LlmConfig
    from src.llm.tokenizer import QwenTokenizer
except Exception:
    import sys
    sys.path.append(".")
    from src.llm.pipeline import LlmPipeline, LlmConfig  # type: ignore
    from src.llm.tokenizer import QwenTokenizer  # type: ignore


def ensure_dirs():
    os.makedirs("logs/runs", exist_ok=True)


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


def make_prompt(tokenizer: QwenTokenizer, n_tokens: int) -> str:
    text = "You are a helpful assistant. " * (n_tokens // 6 + 1)
    ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    while len(ids) > n_tokens:
        ids = ids[:-1]
    return tokenizer.decode(ids, skip_special_tokens=True)


def try_backend(pipeline: LlmPipeline, cfg: LlmConfig, name: str) -> bool:
    try:
        pipeline.backend = None
        pipeline.backend_name = None
        selected = pipeline.select_backend(preferred=name, config=cfg)
        return selected == name
    except Exception as e:
        print(f"[bench] Skip {name}: {e}")
        return False


def measure_ingest(pipeline: LlmPipeline, prompt: str, max_new_tokens: int = 1) -> Dict[str, float]:
    """
    Run until first token emitted. Report:
      - ttft_ms
      - ingest_tok_s ~= prompt_tokens / ttft
    """
    t0 = time.perf_counter()
    ttft = None
    for chunk, info in pipeline.stream(prompt, max_new_tokens=max_new_tokens, sampling="greedy"):
        if "ttft" in info:
            ttft = info["ttft"]
            break
    if ttft is None:
        # fallback to pipeline metrics if present
        m = pipeline.metrics_summary()
        ttft = m.get("ttft", -1.0)
    # Rough ingest throughput calculation
    tok = len(pipeline.tokenizer.encode(prompt, add_bos=True, add_eos=False))
    ingest_tok_s = (tok / ttft) if ttft and ttft > 0 else 0.0
    return {"ttft_ms": ttft * 1000.0 if ttft and ttft > 0 else -1.0, "ingest_tok_s": ingest_tok_s}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model.qwen2_7b_instruct.yaml")
    args = parser.parse_args()

    ensure_dirs()

    cfg_yaml = load_yaml(args.config)
    cfg = build_config(cfg_yaml)

    pipeline = LlmPipeline()
    pipeline.load_model(cfg)
    tokenizer = pipeline.tokenizer

    prompt_lengths = [512, 1024, 2048]
    backends = ["mlc", "onnx_dml", "llama_vulkan", "cpu"]

    results: List[Dict[str, Any]] = []

    print("backend,prompt_len,ttft_ms,ingest_tok_s")
    for be in backends:
        if not try_backend(pipeline, cfg, be):
            continue
        for L in prompt_lengths:
            prompt = make_prompt(tokenizer, L)
            r = measure_ingest(pipeline, prompt, max_new_tokens=1)
            print(f"{be},{L},{r['ttft_ms']:.2f},{r['ingest_tok_s']:.2f}")
            results.append({"backend": be, "prompt_len": L, **r})

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("logs", "runs", f"bench_llm_ingest_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "env": {
                    k: os.environ.get(k)
                    for k in ["ORT_TUNER_ENABLE", "ORT_TUNER_LOAD", "ORT_TUNER_SAVE", "OMP_NUM_THREADS", "KMP_AFFINITY"]
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"[bench] Saved: {out_path}")


if __name__ == "__main__":
    main()