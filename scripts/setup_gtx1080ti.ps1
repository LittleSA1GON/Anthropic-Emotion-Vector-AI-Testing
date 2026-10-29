# PowerShell local setup for the GTX 1080 Ti (Pascal sm_61). Requires Python 3.12 and compatible NVIDIA driver.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
py -3.12 -m venv .venv-gtx
$python = Join-Path $root ".venv-gtx\Scripts\python.exe"
& $python -m pip install --upgrade pip
# PyTorch 2.14.1 CUDA 12.6 retains legacy Pascal support. Do not use CUDA 13 wheels on GTX1080Ti.
& $python -m pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126
& $python -m pip install -r requirements-hf.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); print(torch.ones(2, device='cuda') + 1)"
if ($LASTEXITCODE -ne 0) { throw 'Real CUDA tensor test failed; check the NVIDIA driver and wheel architecture' }
& $python -m mm_emotion.cli doctor
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'CPU test suite failed' }
Write-Host "Setup complete. Run the HF FP16 pilot from README. IBM MIA does not support GTX 1080 Ti."
