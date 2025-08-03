"""
SigLIP SO400M patch14-384 ONNX inference via DirectML on Windows.

- Loads ONNX model through OnnxDmlSigLIP backend
- Preprocesses PIL images to 384x384, normalization to FP16 NCHW
- Batched inference with pinned-like staging (numpy arrays) and returns embeddings

Dependencies are optional; if pillow or numpy are missing, raises actionable errors.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

import numpy as np


def _log(msg: str) -> None:
    print(f"[SigLIP] {msg}")


try:
    from ..runtime.windows.onnx_dml_backend import OnnxDmlSigLIP  # type: ignore
except Exception:
    OnnxDmlSigLIP = None  # type: ignore


class SigLipInfer:
    def __init__(self, onnx_path: str) -> None:
        if OnnxDmlSigLIP is None:
            raise RuntimeError("OnnxDmlSigLIP backend unavailable. Install onnxruntime-directml.")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(onnx_path)
        self.backend = OnnxDmlSigLIP()
        self.backend.load(onnx_path)
        _log(f"Loaded SigLIP ONNX with providers: {self.backend.providers}")

    @staticmethod
    def _preprocess_pil(img) -> np.ndarray:
        """
        Convert PIL image to normalized NCHW FP16 384x384.
        Mean/Std from SigLIP references (approx CLIP-like): mean=[0.5]*3, std=[0.5]*3
        """
        try:
            from PIL import Image  # type: ignore
        except Exception as e:
            raise RuntimeError("Pillow not installed. Install with: pip install pillow") from e

        if not isinstance(img, Image.Image):
            raise TypeError("Input must be a PIL.Image.Image")

        img = img.convert("RGB").resize((384, 384), Image.BICUBIC)
        arr = np.asarray(img).astype(np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        arr = (arr - mean) / std  # HWC
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return arr.astype(np.float16)

    def run(self, images: Iterable) -> np.ndarray:
        """
        images: iterable of PIL.Image.Image
        Returns embeddings as numpy array [B, D]
        """
        batch = [self._preprocess_pil(img) for img in images]
        if not batch:
            raise ValueError("Empty image batch")
        x = np.stack(batch, axis=0)  # [B,3,384,384], fp16
        return self.backend.run(x)