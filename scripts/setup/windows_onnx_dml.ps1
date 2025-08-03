# Windows setup for ONNX Runtime DirectML
# Sets environment variables for the current PowerShell session and echoes config.

param(
  [string]$TunerLoad = "",
  [string]$TunerSave = "logs/tuner/qwen2_7b_tuner.json",
  [string]$NumThreads = "8",
  [switch]$EnableTuner = $false
)

Write-Host "[setup] Configuring ONNX Runtime DirectML environment..."

$env:ORT_TUNER_ENABLE = $(if ($EnableTuner) {"1"} else {"0"})
$env:ORT_TUNER_LOAD = $TunerLoad
$env:ORT_TUNER_SAVE = $TunerSave
$env:ORT_DML_DISABLE_FP16 = "0"
$env:ORT_CPU_FALLBACK = "1"
$env:ORT_NUM_THREADS = $NumThreads

# Ensure logs directory exists
$newDir = "logs\tuner"
if (-not (Test-Path $newDir)) { New-Item -ItemType Directory -Force -Path $newDir | Out-Null }

Write-Host "[setup] ORT_TUNER_ENABLE=$env:ORT_TUNER_ENABLE"
Write-Host "[setup] ORT_TUNER_LOAD=$env:ORT_TUNER_LOAD"
Write-Host "[setup] ORT_TUNER_SAVE=$env:ORT_TUNER_SAVE"
Write-Host "[setup] ORT_DML_DISABLE_FP16=$env:ORT_DML_DISABLE_FP16"
Write-Host "[setup] ORT_CPU_FALLBACK=$env:ORT_CPU_FALLBACK"
Write-Host "[setup] ORT_NUM_THREADS=$env:ORT_NUM_THREADS"