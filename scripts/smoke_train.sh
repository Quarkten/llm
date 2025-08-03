#!/usr/bin/env bash
# Apache-2.0
# Smoke training on Linux/WSL. For Windows, use scripts/smoke_train.bat.

set -euo pipefail

# Optionally create venv and install in editable mode
if [ "${CREATE_VENV:-0}" = "1" ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e .
fi

python -m src.llm.train.launch --config configs/train_13b.yaml