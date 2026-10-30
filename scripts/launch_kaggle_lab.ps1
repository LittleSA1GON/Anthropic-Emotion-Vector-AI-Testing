$ErrorActionPreference = 'Stop'
$labRoot = Split-Path -Parent $PSScriptRoot
Set-Location $labRoot
& (Join-Path $labRoot 'run.bat') @args
