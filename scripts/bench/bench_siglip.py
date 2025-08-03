"""
Windows-only SigLIP ONNX DirectML benchmark.

Measures images/sec at 384x384 for batch sizes [1, 8], logs backend/provider info,
and saves JSON results to logs/runs/.

Usage (PowerShell):
  python scripts/bench/bench_siglip.py --config configs/vision.siglip_384.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

# Local imports with fallback path
try:
    from src.vision.siglip_infer import SigLipInfer
except Exception:
    import sys
    sys.path.append(".")
    from src.vision.siglip_infer import SigLipInfer  # type: ignore


def ensure_dirs():
    os.makedirs("logs/runs", exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("pyyaml not installed. Install with: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_dummy_images(n: int):
    # Generate n dummy PIL images (gray gradient) without touching disk
    try:
        from PIL import Image  # type: ignore
        import numpy as np
    except Exception as e:
        raise RuntimeError("Pillow and numpy are required. pip install pillow numpy") from e
    imgs = []
    for i in range(n):
        arr = (np.linspace(0, 255, 384 * 384, dtype=np.uint8).reshape(384, 384))
        arr_rgb = np.stack([arr, arr, arr], axis=-1)
        imgs.append(Image.fromarray(arr_rgb, mode="RGB"))
    return imgs


def run_once(infer: SigLipInfer, batch_size: int, warmup: int = 1, iters: int = 10) -> float:
    """
    Return images/sec for steady-state (excluding warmup).
    """
    imgs = make_dummy_images(batch_size)

    # Warmup
    for _ in range(warmup):
        _ = infer.run(imgs)

    # Timed iterations
    t0 = time.perf_counter()
    n = 0
    for _ in range(iters):
        _ = infer.run(imgs)
        n += batch_size
    t1 = time.perf_counter()
    dt = max(1e-6, t1 - t0)
    return n / dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vision.siglip_384.yaml")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    ensure_dirs()

    cfg = load_yaml(args.config)
    onnx_path = cfg.get("onnx_path")
    if not onnx_path:
        raise RuntimeError("onnx_path missing in config")

    infer = SigLipInfer(onnx_path=onnx_path)
    batch_sizes = [1, 8]

    results: List[Dict[str, Any]] = []
    print("batch,images_per_sec")
    for b in batch_sizes:
        ips = run_once(infer, batch_size=b, iters=args.iters)
        print(f"{b},{ips:.2f}")
        results.append({"batch": b, "images_per_sec": float(ips)})

    # Save JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("logs", "runs", f"bench_siglip_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "env": {"provider": "DmlExecutionProvider"},
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"[bench] Saved: {out_path}")


if __name__ == "__main__":
    main()