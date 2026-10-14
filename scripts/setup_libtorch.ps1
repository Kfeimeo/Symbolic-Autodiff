# Downloads a CPU-only LibTorch build (Release, MSVC) into third_party\libtorch.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\setup_libtorch.ps1 [-Version 2.14.1]
param([string]$Version = "2.14.1")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $Root "third_party"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
if (Test-Path (Join-Path $Dest "libtorch")) {
  Write-Host "third_party\libtorch already exists"
  exit 0
}
$Url = "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-$Version%2Bcpu.zip"
$Zip = Join-Path $Dest "libtorch.zip"
Write-Host "downloading $Url"
Invoke-WebRequest -Uri $Url -OutFile $Zip
Expand-Archive -Path $Zip -DestinationPath $Dest
Remove-Item $Zip
Write-Host "installed LibTorch $(Get-Content (Join-Path $Dest 'libtorch\build-version'))"
