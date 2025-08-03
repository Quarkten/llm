@echo off
REM Apache-2.0
REM Smoke training on Windows. For ROCm, prefer running via WSL. This uses CPU fallback by default.

SETLOCAL ENABLEDELAYEDEXPANSION

IF "%CREATE_VENV%"=="1" (
  python -m venv .venv
  CALL .venv\Scripts\activate.bat
  pip install -e .
)

python -m src.llm.train.launch --config configs\train_13b.yaml
ENDLOCAL