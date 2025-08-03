# Windows Maximum Performance Checklist (Intel i5-8400 + AMD RX 6800)

Goal: Achieve consistent max performance across MLC Vulkan (AWQ 4-bit), ONNX Runtime DirectML, llama.cpp Vulkan, and CPU fallback on Windows. Keep steps short and actionable.

1) Verify GPU Driver + Vulkan ICD
- Update AMD Adrenalin to latest stable (RX 6800).
- Check Vulkan runtime and ICD:
  - Press Win+R, run: `cmd`
  - In cmd:
    - `where vulkaninfo` (from Vulkan SDK or AMD driver)
    - Run `vulkaninfo > %TEMP%\vulkaninfo.txt` and open the file to confirm AMD ICD is active and device shows “AMD Radeon RX 6800”.
- Optional: if Vulkan SDK installed, ensure VULKAN_SDK is set.

2) Run Setup Scripts
- Open PowerShell as Administrator and run:
  - `.\scripts\setup\windows_onnx_dml.ps1` ([scripts/setup/windows_onnx_dml.ps1](scripts/setup/windows_onnx_dml.ps1:1))
  - `.\scripts\setup\windows_mlc_vulkan.ps1` ([scripts/setup/windows_mlc_vulkan.ps1](scripts/setup/windows_mlc_vulkan.ps1:1))
  - `.\scripts\setup\windows_llama_vulkan.ps1` ([scripts/setup/windows_llama_vulkan.ps1](scripts/setup/windows_llama_vulkan.ps1:1))
  - `.\scripts\setup\windows_cpu_fallback.ps1` ([scripts/setup/windows_cpu_fallback.ps1](scripts/setup/windows_cpu_fallback.ps1:1))
- Close and reopen your terminal to pick up any env changes.

3) Configure Runtime + Model
- Use runtime config to select backends and priority:
  - Edit [configs/runtime.windows.yaml](configs/runtime.windows.yaml:1) to ensure DirectML and Vulkan backends are enabled as desired (and CPU fallback last).
- Ensure model artifact paths are set in [configs/model.qwen2_7b_instruct.yaml](configs/model.qwen2_7b_instruct.yaml:1) for ONNX/MLC/GGUF as available.

4) Optimize ONNX Runtime (DirectML)
- Enable ORT tuning flags in runtime config (SessionOptions) and IO-binding hint if supported by your pipeline.
- Typical flags to consider (apply via SessionOptions in code or runtime config):
  - graph_optimization_level = ORT_ENABLE_ALL
  - intra_op_num_threads = (6 recommended for i5-8400; experiment)
- Validate providers and DmlExecutionProvider availability via self-check (below).

5) Optimize llama.cpp Vulkan
- Set environment and warm shaders:
  - `set GGML_VK_F16=1`
  - First run will compile shaders; do a throwaway short run to warm caches.

6) Optimize MLC Vulkan
- Ensure Vulkan layers are clean (no debug layers). First run compiles pipelines; do a warm-up tiny generation once to prime caches.
- Prefer selecting the discrete AMD GPU if multiple adapters are present.

7) Optimize CPU Fallback
- Set threads for best throughput on i5-8400 (6C/6T):
  - `set OMP_NUM_THREADS=6`
  - `set MKL_NUM_THREADS=6`
  - Optional Intel OpenMP tweaks:
    - `set KMP_AFFINITY=granularity=fine,compact,1,0`
    - `set KMP_BLOCKTIME=0`

8) Run Self-Check
- Command:
  - `python scripts\self_check_windows.py --model-config configs\model.qwen2_7b_instruct.yaml --runtime-config configs\runtime.windows.yaml --max-new 16 --prompt "Hello"`
- Confirms:
  - System and GPU summary ([src/utils/device_check.py](src/utils/device_check.py:1))
  - ORT providers + DmlExecutionProvider
  - MLC Vulkan device
  - llama.cpp Vulkan device
  - CPU fallback env threads
- JSON summary written under `logs\runs\self_check_<timestamp>.json`.

9) Run Benchmarks (compare against your established baselines)
Run in this order after warm-ups:
- Ingest throughput:
  - `python scripts\bench\bench_llm_ingest.py` ([scripts/bench/bench_llm_ingest.py](scripts/bench/bench_llm_ingest.py:1))
- Decode throughput:
  - `python scripts\bench\bench_llm_decode.py` ([scripts/bench/bench_llm_decode.py](scripts/bench/bench_llm_decode.py:1))
- Vision baseline (DirectML/SigLIP):
  - `python scripts\bench\bench_siglip.py` ([scripts/bench/bench_siglip.py](scripts/bench/bench_siglip.py:1))
Tips:
- Run each twice; use the second run for numbers (cached shaders/graphs).
- Ensure background apps are closed; set Windows Power mode to “Best performance”.

10) Quick Remediation If Below Target
- ONNX DML slow or CPU fallback used:
  - Re-run [scripts/setup/windows_onnx_dml.ps1](scripts/setup/windows_onnx_dml.ps1:1).
  - Check `onnxruntime.get_available_providers()` shows `DmlExecutionProvider`.
  - Reduce intra_op_num_threads to avoid oversubscription (try 6).
- Vulkan backends slow:
  - Verify `vulkaninfo` shows RX 6800.
  - Set `GGML_VK_F16=1` for llama.cpp.
  - Warm caches with a short first run.
  - Ensure no remote/virtual display drivers active. Prefer main monitor on the RX 6800.
- CPU fallback slow:
  - Set `OMP_NUM_THREADS=6` and `MKL_NUM_THREADS=6`.
  - Confirm power plan = High performance. Disable CPU throttling.
- Model paths incorrect:
  - Fix paths in [configs/model.qwen2_7b_instruct.yaml](configs/model.qwen2_7b_instruct.yaml:1).
- Backend priority not as expected:
  - Adjust [configs/runtime.windows.yaml](configs/runtime.windows.yaml:1) to select preferred providers first.

Reference Files
- Setup: [scripts/setup/windows_onnx_dml.ps1](scripts/setup/windows_onnx_dml.ps1:1), [scripts/setup/windows_mlc_vulkan.ps1](scripts/setup/windows_mlc_vulkan.ps1:1), [scripts/setup/windows_llama_vulkan.ps1](scripts/setup/windows_llama_vulkan.ps1:1), [scripts/setup/windows_cpu_fallback.ps1](scripts/setup/windows_cpu_fallback.ps1:1)
- Benchmarks: [scripts/bench/bench_llm_ingest.py](scripts/bench/bench_llm_ingest.py:1), [scripts/bench/bench_llm_decode.py](scripts/bench/bench_llm_decode.py:1), [scripts/bench/bench_siglip.py](scripts/bench/bench_siglip.py:1)
- Utilities: [src/utils/device_check.py](src/utils/device_check.py:1)
- Configs: [configs/runtime.windows.yaml](configs/runtime.windows.yaml:1), [configs/model.qwen2_7b_instruct.yaml](configs/model.qwen2_7b_instruct.yaml:1)