# Windows setup for MLC LLM Vulkan
# Sets environment variables related to shader cache and echoes device info if available.

param(
  [string]$ShaderCacheDir = "$PWD\logs\mlc_shader_cache"
)

Write-Host "[setup] Configuring MLC Vulkan environment..."

# Create shader cache directory if not exists
if (-not (Test-Path $ShaderCacheDir)) { New-Item -ItemType Directory -Force -Path $ShaderCacheDir | Out-Null }
$env:MLC_SHADER_CACHE_DIR = $ShaderCacheDir

Write-Host "[setup] MLC_SHADER_CACHE_DIR=$env:MLC_SHADER_CACHE_DIR"
Write-Host "[setup] Note: Device detection will occur at runtime via Python API if installed."