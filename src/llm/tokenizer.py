"""
Hugging Face tokenizer wrapper for Qwen2.5-7B-Instruct (Windows-only context).

- Loads fast tokenizer with normalization options.
- Provides encode/decode utilities suitable for streaming generation.
- Defers heavy imports until use; graceful error messages if not installed.

Configuration can be provided via:
- Explicit tokenizer_path argument
- Environment variable QWEN_TOKENIZER_DIR
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple


def _log(msg: str) -> None:
    print(f"[Tokenizer] {msg}")


class QwenTokenizer:
    def __init__(self, tokenizer_path: Optional[str] = None, trust_remote_code: bool = True) -> None:
        self.tokenizer = None
        self.trust_remote_code = trust_remote_code
        self.path = tokenizer_path or os.environ.get("QWEN_TOKENIZER_DIR") or "Qwen/Qwen2.5-7B-Instruct"
        self.loaded = False

    def load(self) -> None:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Hugging Face transformers not installed. Install with: pip install transformers"
            ) from e

        _log(f"Loading tokenizer from: {self.path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, use_fast=True, trust_remote_code=self.trust_remote_code)
        self.loaded = True

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        if not self.loaded:
            self.load()
        assert self.tokenizer is not None
        out = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if add_bos and hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id is not None:
            out = [self.tokenizer.bos_token_id] + out
        if add_eos and hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            out = out + [self.tokenizer.eos_token_id]
        return out

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if not self.loaded:
            self.load()
        assert self.tokenizer is not None
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def eos_id(self) -> Optional[int]:
        if not self.loaded:
            self.load()
        assert self.tokenizer is not None
        return getattr(self.tokenizer, "eos_token_id", None)

    def bos_id(self) -> Optional[int]:
        if not self.loaded:
            self.load()
        assert self.tokenizer is not None
        return getattr(self.tokenizer, "bos_token_id", None)

    def apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """
        For instruct/chat usage, defer to tokenizer chat template if available.
        """
        if not self.loaded:
            self.load()
        assert self.tokenizer is not None
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        # Fallback simple concat
        text = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            text += f"{role}: {content}\n"
        if add_generation_prompt:
            text += "assistant: "
        return text