# Copyright 2025
# Apache-2.0 License

from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import InferenceConfig, ModelConfig
from ..model.decoder import DecoderOnlyLM
from .quant import QuantConfig, apply_fake_quant_linear

import torch


app = FastAPI(title="GLM-4.5 Minimal Inference Server", version="0.1.0")


class ToolArgument(BaseModel):
    name: str
    type: Literal["string", "number", "integer", "boolean", "object", "array"]
    description: Optional[str] = None
    required: bool = False


class ToolSchema(BaseModel):
    name: str = Field(..., description="Tool function name")
    description: Optional[str] = None
    arguments: List[ToolArgument] = Field(default_factory=list)


class ToolCallRequest(BaseModel):
    tool: ToolSchema
    input: dict = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.95


class GenerateResponse(BaseModel):
    outputs: List[str]


@app.on_event("startup")
def _startup():
    # Lazy init lightweight model config for placeholder inference
    global _MODEL, _TOK
    model_cfg = ModelConfig(
        vocab_size=32000,
        d_model=512,
        n_layers=4,
        n_heads=8,
        n_kv_heads=8,
        max_seq_len=512,
        use_swa=True,
        swa_window=256,
        swa_global_every=4,
        rope_base=10000.0,
        rope_ntk_factor=1.0,
        rope_interpolation_factor=1.0,
    )
    _MODEL = DecoderOnlyLM(model_cfg).eval()
    # TODO: load real checkpoint if provided via env/config; apply quant stubs
    for m in _MODEL.modules():
        if isinstance(m, torch.nn.Linear):
            apply_fake_quant_linear(m, method=None)


@app.post("/v1/tool_call")
def tool_call(req: ToolCallRequest):
    # Validate that provided input keys match required arguments
    required = {a.name for a in req.tool.arguments if a.required}
    missing = [k for k in required if k not in req.input]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required arguments: {missing}")
    # Type checks placeholder (not strict JSON Schema implementation)
    for arg in req.tool.arguments:
        if arg.name in req.input:
            val = req.input[arg.name]
            t = arg.type
            ok = True
            if t == "string":
                ok = isinstance(val, str)
            elif t == "number":
                ok = isinstance(val, (float, int))
            elif t == "integer":
                ok = isinstance(val, int) and not isinstance(val, bool)
            elif t == "boolean":
                ok = isinstance(val, bool)
            elif t == "object":
                ok = isinstance(val, dict)
            elif t == "array":
                ok = isinstance(val, list)
            if not ok:
                raise HTTPException(status_code=400, detail=f"Argument '{arg.name}' has invalid type, expected {t}")
    return {"status": "ok", "tool": req.tool.name, "validated": True}


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # This is a stub; a real implementation would tokenize and autoregress.
    # For now, echo prompts with a placeholder continuation.
    outs = []
    for p in req.prompts:
        outs.append(p + " [generated text]")
    return GenerateResponse(outputs=outs)