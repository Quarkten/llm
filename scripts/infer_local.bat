@echo off
REM Apache-2.0
REM Start the FastAPI inference server on Windows (CPU by default).
REM For ROCm-enabled training/inference, prefer WSL2 and run:
REM   uvicorn src.llm.infer.server:app --host 0.0.0.0 --port 8000 --reload

SETLOCAL ENABLEDELAYEDEXPANSION

IF "%CREATE_VENV%"=="1" (
  python -m venv .venv
  CALL .venv\Scripts\activate.bat
  pip install -e .
)

python -m uvicorn src.llm.infer.server:app --host 0.0.0.0 --port 8000 --reload
ENDLOCAL