# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Exports the Visual Studio developer environment (vcvarsall.bat) for later steps of a
# GitHub Actions job: clang-cl and clang++ on the MSVC ABI need its include and library
# paths and its linker.
#
#   pwsh .github/scripts/msvc-env.ps1 x64|arm64
param([Parameter(Mandatory = $true)][string]$Arch)
$ErrorActionPreference = 'Stop'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products * -property installationPath
if (-not $vs) { throw 'msvc-env: no Visual Studio installation found' }
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvarsall.bat'
$lines = cmd /c "`"$vcvars`" $Arch >nul && set"
if ($LASTEXITCODE -ne 0) { throw "msvc-env: vcvarsall.bat $Arch failed" }
foreach ($line in $lines) {
  if ($line -match '^([^=]+)=(.*)$') {
    Add-Content -Path $env:GITHUB_ENV -Value "$($Matches[1])=$($Matches[2])" -Encoding utf8
  }
}
Write-Output "msvc-env: $vs ($Arch)"
