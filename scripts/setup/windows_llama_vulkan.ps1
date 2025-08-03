# Windows setup for llama.cpp Vulkan
# Sets optional environment variables and echoes notes.
# Ensure a Vulkan-enabled llama.cpp executable is in PATH.

param(
  [string]$ExtraFlags = ""
)

Write-Host "[setup] Configuring llama.cpp Vulkan environment..."
Write-Host "[setup] Ensure your llama.cpp build includes Vulkan backend and targets AMD RX 6800."
Write-Host "[setup] Extra CLI flags (if any): $ExtraFlags"