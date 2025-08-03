# Windows setup for CPU fallback (oneDNN/AVX2 via llama.cpp CPU or ORT CPU EP)
# Sets thread pinning env vars and echoes effective configuration.

param(
  [string]$OmpNumThreads = "8",
  [string]$KmpAffinity = "granularity=fine,compact,1,0",
  [string]$MklNumThreads = "",
  [switch]$EchoOnly = $false
)

Write-Host "[setup] Configuring CPU fallback threading..."

if (-not $EchoOnly) {
  $env:OMP_NUM_THREADS = $OmpNumThreads
  $env:KMP_AFFINITY = $KmpAffinity
  if ($MklNumThreads -ne "") { $env:MKL_NUM_THREADS = $MklNumThreads }
}

Write-Host "[setup] OMP_NUM_THREADS=$env:OMP_NUM_THREADS"
Write-Host "[setup] KMP_AFFINITY=$env:KMP_AFFINITY"
Write-Host "[setup] MKL_NUM_THREADS=$env:MKL_NUM_THREADS"
Write-Host "[setup] Note: Ensure your CPU supports AVX2 and that binaries are compiled accordingly."