Windows-Only Guide: i5-8400 + AMD RX 6800 (RDNA2, 16GB)

Audience and Hardware
- OS: Windows 11 only (no Linux/WSL).
- CPU: Intel Core i5-8400 (6C/6T).
- GPU: AMD Radeon RX 6800 (RDNA2) with 16 GB VRAM.
- Goal: Local LLM and vision inference with Vulkan and DirectML backends, plus CPU fallback.

Quick Start (TL;DR)
1) Create and activate Python environment
- Open Windows Terminal (PowerShell) in the project root.
- Optional: upgrade pip.
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

2) Install one runtime stack (choose ONE)
- ONNX + DirectML (recommended for stability on AMD):
PowerShell: [scripts.setup.windows_onnx_dml.ps1](scripts/setup/windows_onnx_dml.ps1:1)

- MLC + Vulkan (fast AWQ on AMD):
PowerShell: [scripts.setup.windows_mlc_vulkan.ps1](scripts/setup/windows_mlc_vulkan.ps1:1)

- llama.cpp + Vulkan (GGUF path + solid perf):
PowerShell: [scripts.setup.windows_llama_vulkan.ps1](scripts/setup/windows_llama_vulkan.ps1:1)

- CPU fallback (reference / troubleshooting):
PowerShell: [scripts.setup.windows_cpu_fallback.ps1](scripts/setup/windows_cpu_fallback.ps1:1)

3) Place model artifacts
Create folders under checkpoints/ and download models as shown in “Models and checkpoints layout” below.

4) Run self-check
python [scripts.self_check_windows](scripts/self_check_windows.py:1) --model-config [configs.model.qwen2_7b_instruct.yaml](configs/model.qwen2_7b_instruct.yaml:1) --runtime-config [configs.runtime.windows.yaml](configs/runtime.windows.yaml:1) --max-new 16 --prompt "Hello"

5) Run a single benchmark and a single generation
- LLM decode benchmark:
python [scripts.bench.bench_llm_decode](scripts/bench/bench_llm_decode.py:1) --config [configs.model.qwen2_7b_instruct.yaml](configs/model.qwen2_7b_instruct.yaml:1)

- Single generation example (run self-check above or see “Inference pipelines” for programmatic usage)

Installation and Setup (Windows-only)
Run one of the setup scripts to install dependencies, runtime libraries, and CLI tooling:
- ONNX + DirectML:
  - [scripts.setup.windows_onnx_dml.ps1](scripts/setup/windows_onnx_dml.ps1:1)
- MLC + Vulkan:
  - [scripts.setup.windows_mlc_vulkan.ps1](scripts/setup/windows_mlc_vulkan.ps1:1)
- llama.cpp + Vulkan:
  - [scripts.setup.windows_llama_vulkan.ps1](scripts/setup/windows_llama_vulkan.ps1:1)
- CPU fallback:
  - [scripts.setup.windows_cpu_fallback.ps1](scripts/setup/windows_cpu_fallback.ps1:1)

Prerequisites
- AMD Adrenalin driver (latest WHQL) for RX 6800.
- Vulkan runtime/ICD present (installed via Adrenalin).
- Microsoft Visual C++ Redistributable (x64).

Out-of-scope
- ROCm on Windows is not supported in this repository.

Key environment variables (defaults shown if not set)
- ONNX Runtime / DirectML:
  - ORT_TUNER_ENABLE=1 (enable perf tuner)
  - ORT_TUNER_ENABLE_SAVE=1, ORT_TUNER_ENABLE_LOAD=1 (cache and reuse)
  - ORT_ENABLE_ALL=1 (telemetry/profiling enabling switch if applicable in your build)
- Vulkan / llama.cpp:
  - GGML_VK_F16=1 (prefer FP16 if supported to reduce bandwidth)
- CPU backends:
  - OMP_NUM_THREADS=6 (set to number of physical cores on i5-8400)
  - MKL_NUM_THREADS=6
  - KMP_AFFINITY=granularity=fine,compact,1,0

Set in current session (example):
$env:ORT_TUNER_ENABLE="1"; $env:ORT_TUNER_ENABLE_SAVE="1"; $env:ORT_TUNER_ENABLE_LOAD="1"; $env:ORT_ENABLE_ALL="1"
$env:GGML_VK_F16="1"
$env:OMP_NUM_THREADS="6"; $env:MKL_NUM_THREADS="6"; $env:KMP_AFFINITY="granularity=fine,compact,1,0"

Models and Checkpoints Layout
We target Qwen2.5-7B-Instruct for LLM and SigLIP SO400M 384 FP16 for vision.

LLM: Qwen2.5-7B-Instruct (choose one or more format(s))
- MLC Vulkan AWQ 4-bit:
  - Path: checkpoints/mlc/qwen2_7b_awq4/
  - Contents: MLC model artifacts produced by MLC compilation/packer.
- ONNX (DirectML) decoder + decoder_with_past:
  - Path: checkpoints/onnx/qwen2_7b/
  - Files: decoder.onnx, decoder_with_past.onnx, tokenizer.json (and any auxiliary files your converter produced).
- llama.cpp GGUF Q4_K_M or Q5_K_M:
  - Path: checkpoints/gguf/qwen2_7b/
  - Files: qwen2.5-7b-instruct.Q4_K_M.gguf (or Q5_K_M), tokenizer.*

Vision: SigLIP SO400M 384 FP16 ONNX
- Path: checkpoints/onnx/siglip/siglip_so400m_patch14_384_fp16.onnx

Example tree:
checkpoints/
  mlc/
    qwen2_7b_awq4/
      params.json
      weights/
      shaders/
  onnx/
    qwen2_7b/
      decoder.onnx
      decoder_with_past.onnx
      tokenizer.json
    siglip/
      siglip_so400m_patch14_384_fp16.onnx
  gguf/
    qwen2_7b/
      qwen2.5-7b-instruct.Q4_K_M.gguf
      tokenizer.model

Configs to edit:
- LLM model config: [configs.model.qwen2_7b_instruct.yaml](configs/model.qwen2_7b_instruct.yaml:1)
- Vision config: [configs.vision.siglip_384.yaml](configs/vision.siglip_384.yaml:1)
- Windows runtime config: [configs.runtime.windows.yaml](configs/runtime.windows.yaml:1)

Update these paths inside the configs to point to your local checkpoints.

Running Self-Check and Device Summary
Run:
python [scripts.self_check_windows](scripts/self_check_windows.py:1) --model-config [configs.model.qwen2_7b_instruct.yaml](configs/model.qwen2_7b_instruct.yaml:1) --runtime-config [configs.runtime.windows.yaml](configs/runtime.windows.yaml:1) --max-new 16 --prompt "Hello"

What it does
- Validates model files and tokenizer availability.
- Enumerates available backends on Windows:
  - MLC Vulkan, ONNX DirectML, llama.cpp Vulkan, CPU oneDNN.
- Prints device details and selects the preferred backend based on config priority.
- Produces a JSON report at logs/runs/self_check_*.json with backend statuses and timings.

Interpretation
- backend_status.ok=true indicates usable.
- If a preferred backend fails, check Troubleshooting and try another stack or CPU fallback.

Inference Pipelines and How to Run
LLM streaming with LlmPipeline
- Core pipeline: [src.llm.pipeline](src/llm/pipeline.py:1)
- Backends (Windows):
  - MLC Vulkan: [src.runtime.windows.mlc_vulkan_backend](src/runtime/windows/mlc_vulkan_backend.py:1)
  - ONNX DirectML: [src.runtime.windows.onnx_dml_backend](src/runtime/windows/onnx_dml_backend.py:1)
  - llama.cpp Vulkan: [src.runtime.windows.llama_vulkan_backend](src/runtime/windows/llama_vulkan_backend.py:1)
  - CPU oneDNN: [src.runtime.windows.cpu_onednn_backend](src/runtime/windows/cpu_onednn_backend.py:1)

Preferred backend selection
- The runtime config [configs.runtime.windows.yaml](configs/runtime.windows.yaml:1) controls priority and enabled flags. The pipeline will attempt backends in configured priority, falling back if initialization fails.

Single prompt example (programmatic)
Python snippet to stream a single prompt using the configured backend:

from src.llm.pipeline import LlmPipeline  # [python.from src.llm.pipeline import LlmPipeline](src/llm/pipeline.py:1)

pipe = LlmPipeline(
    model_config_path="configs/model.qwen2_7b_instruct.yaml",
    runtime_config_path="configs/runtime.windows.yaml",
)

for token_text in pipe.generate_stream("You are a helpful assistant.", max_new_tokens=64):
    print(token_text, end="", flush=True)
print()

Alternatively, run the self-check command (previous section) with a prompt to confirm streaming works.

Vision embedding example
Try SigLIP benchmark as a quick embedding smoke test:
python [scripts.bench.bench_siglip](scripts/bench/bench_siglip.py:1) --config [configs.vision.siglip_384.yaml](configs/vision.siglip_384.yaml:1) --just-one

Note: If --just-one is not implemented in your version, run the full benchmark without it as a smoke test:
python [scripts.bench.bench_siglip](scripts/bench/bench_siglip.py:1) --config [configs.vision.siglip_384.yaml](configs/vision.siglip_384.yaml:1)

Benchmarks and Expected Performance
Run benchmarks
- LLM ingest (prompt processing throughput):
python [scripts.bench.bench_llm_ingest](scripts/bench/bench_llm_ingest.py:1) --config [configs.model.qwen2_7b_instruct.yaml](configs/model.qwen2_7b_instruct.yaml:1)

- LLM decode (token generation throughput):
python [scripts.bench.bench_llm_decode](scripts/bench/bench_llm_decode.py:1) --config [configs.model.qwen2_7b_instruct.yaml](configs/model.qwen2_7b_instruct.yaml:1)

- SigLIP (vision embeddings):
python [scripts.bench.bench_siglip](scripts/bench/bench_siglip.py:1) --config [configs.vision.siglip_384.yaml](configs/vision.siglip_384.yaml:1)

Performance targets on RX 6800 (steady-state decode, approximate)
- MLC Vulkan AWQ 4-bit: 30–80 tok/s; TTFT <300 ms (short prompt), <1 s (~2k tokens prompt)
- ONNX DirectML INT8: 15–40 tok/s; TTFT <600 ms (short), <1.5 s (~2k)
- llama.cpp Vulkan Q4/Q5: 20–60 tok/s
- CPU fallback: 2–6 tok/s (reference only)
- SigLIP 384 FP16: ≥200 img/s, batch=8 (±20%)

Maximum Performance Checklist (Concise)
Use this in the given order; details mirrored from [docs.windows_max_performance_checklist.md](docs/windows_max_performance_checklist.md:1).

1) Update AMD Adrenalin driver (clean install preferred). Reboot.
2) Pick and run one setup script (ONNX DML or MLC Vulkan or llama.cpp Vulkan). Re-open a fresh terminal.
3) Export tuner/fast-path env vars:
   - ONNX DML: ORT_TUNER_ENABLE=1; ORT_TUNER_ENABLE_SAVE=1; ORT_TUNER_ENABLE_LOAD=1
   - Vulkan: GGML_VK_F16=1 (for llama.cpp); ensure driver Vulkan ICD present.
4) Ensure IO-binding or dedicated IO paths are enabled by the chosen backend (ONNX DML scripts cover this).
5) Warm shader caches (run a tiny generation once, then rerun).
6) CPU vars when using CPU path:
   - OMP_NUM_THREADS=6; MKL_NUM_THREADS=6; KMP_AFFINITY=granularity=fine,compact,1,0
7) Run self-check to verify backend selection:
   - python scripts/self_check_windows.py ...
8) Run benchmarks and compare to targets. If underperforming:
   - Reboot after driver updates; clear tuner cache; verify power plan (High performance); close background GPU apps.

Memory / OOM Guidance (Windows, RX 6800 16GB)
Key sizes for Qwen2.5-7B:
- FP16 weights: ~14 GB (not feasible on 16 GB with KV cache)
- INT8 weights: ~7–8 GB
- 4-bit (AWQ/GGUF): ~3.5–4.5 GB

KV cache growth (typical per token, per head dimension, depends on backend layout):
- Expect VRAM usage to rise with context length; long prompts need reduced batch/micro-batch.

Practical defaults by backend (starting points)
- MLC Vulkan AWQ 4-bit:
  - context_length: 4096–8192 (start 4096)
  - micro_batch: 1–2
- ONNX DML INT8:
  - context_length: 4096
  - micro_batch: 1
- llama.cpp Vulkan Q4/Q5:
  - context_length: 4096–8192 (start 4096)
  - n_batch: 512–1024 for ingest; adjust if OOM
- CPU fallback:
  - Large contexts okay but slow; micro_batch as needed

OOM playbook
1) Reduce max context (e.g., 8192 → 4096 → 2048).
2) Reduce micro-batch / n_batch for prompt ingestion.
3) Prefer lower quant: GGUF Q4_K_M instead of Q5_K_M; AWQ 4-bit instead of 8-bit.
4) Close background GPU apps; reboot GPU driver if degraded.
5) For ONNX DML, ensure IO-binding is active; verify ORT tuner cache exists.
6) Use mem planner to estimate headroom and KV policy:
   - [src.utils.mem_planner](src/utils/mem_planner.py:1)

Troubleshooting (Windows)
- “Provider not found” (ONNX DML):
  - Ensure Windows 11, latest DirectX, Visual C++ Redistributable, and Adrenalin driver.
  - Re-run [scripts.setup.windows_onnx_dml.ps1](scripts/setup/windows_onnx_dml.ps1:1) in an elevated PowerShell.
- Vulkan ICD issues:
  - Check GPU is RX 6800; install Adrenalin; verify vulkaninfo (if present).
  - Delete shader caches if corrupt; re-run once to warm caches.
- DirectML tuner cache permissions:
  - Run terminal as user with write permissions to your project; verify logs/runs and %LOCALAPPDATA% cache locations.
- Shader cache path (MLC / Vulkan / llama.cpp):
  - Clear the GPU shader cache in Adrenalin; re-run a short generation to warm.
- CPU threads and affinity weirdness:
  - Set OMP/MKL vars explicitly as above; reboot after driver updates.

License and Model Ownership Notes
- This repository’s code and pipelines are provided under the project’s license (see pyproject.toml / repository metadata).
- Default model weights are community models (e.g., Qwen2.5-7B-Instruct and SigLIP). You are responsible for complying with their licenses.
- To use your own fine-tuned or quantized weights:
  - Place them under checkpoints/ in the same layout.
  - Update [configs.model.qwen2_7b_instruct.yaml](configs/model.qwen2_7b_instruct.yaml:1) and/or [configs.vision.siglip_384.yaml](configs/vision.siglip_384.yaml:1) with your paths.
  - Ensure tokenizer compatibility and quantization format match the selected backend.

Appendix: File Map
- Backends (Windows):
  - [src.runtime.windows.mlc_vulkan_backend](src/runtime/windows/mlc_vulkan_backend.py:1)
  - [src.runtime.windows.onnx_dml_backend](src/runtime/windows/onnx_dml_backend.py:1)
  - [src.runtime.windows.llama_vulkan_backend](src/runtime/windows/llama_vulkan_backend.py:1)
  - [src.runtime.windows.cpu_onednn_backend](src/runtime/windows/cpu_onednn_backend.py:1)
- Pipelines and utils:
  - [src.llm.pipeline](src/llm/pipeline.py:1)
  - [src.utils.mem_planner](src/utils/mem_planner.py:1)
  - [src.utils.device_check](src/utils/device_check.py:1)
- Configs:
  - [configs.model.qwen2_7b_instruct.yaml](configs/model.qwen2_7b_instruct.yaml:1)
  - [configs.vision.siglip_384.yaml](configs/vision.siglip_384.yaml:1)
  - [configs.runtime.windows.yaml](configs/runtime.windows.yaml:1)
- Benchmarks:
  - [scripts.bench.bench_llm_ingest](scripts/bench/bench_llm_ingest.py:1)
  - [scripts.bench.bench_llm_decode](scripts/bench/bench_llm_decode.py:1)
  - [scripts.bench.bench_siglip](scripts/bench/bench_siglip.py:1)
- Setup scripts:
  - [scripts.setup.windows_onnx_dml.ps1](scripts/setup/windows_onnx_dml.ps1:1)
  - [scripts.setup.windows_mlc_vulkan.ps1](scripts/setup/windows_mlc_vulkan.ps1:1)
  - [scripts.setup.windows_llama_vulkan.ps1](scripts/setup/windows_llama_vulkan.ps1:1)
  - [scripts.setup.windows_cpu_fallback.ps1](scripts/setup/windows_cpu_fallback.ps1:1)
- Self-check:
  - [scripts.self_check_windows](scripts/self_check_windows.py:1)
- Performance checklist:
  - [docs.windows_max_performance_checklist.md](docs/windows_max_performance_checklist.md:1)

Where to Start (Immediate Commands)
1) Create env and install one stack:
PowerShell: [scripts.setup.windows_onnx_dml.ps1](scripts/setup/windows_onnx_dml.ps1:1)

2) Run self-check and one benchmark:
python [scripts.self_check_windows](scripts/self_check_windows.py:1) --model-config [configs.model.qwen2_7b_instruct.yaml](configs/model.qwen2_7b_instruct.yaml:1) --runtime-config [configs.runtime.windows.yaml](configs/runtime.windows.yaml:1) --max-new 16 --prompt "Hello"
python [scripts.bench.bench_llm_decode](scripts/bench/bench_llm_decode.py:1) --config [configs.model.qwen2_7b_instruct.yaml](configs/model.qwen2_7b_instruct.yaml:1)