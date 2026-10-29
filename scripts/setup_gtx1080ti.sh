#!/usr/bin/env bash
# Linux / WSL2 GTX 1080 Ti setup. CUDA 12.6 PyTorch binary retains Pascal SM61 support.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -m venv .venv-gtx
.venv-gtx/bin/python -m pip install --upgrade pip
.venv-gtx/bin/python -m pip install 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cu126
.venv-gtx/bin/python -m pip install -r requirements-hf.txt
.venv-gtx/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); print(torch.ones(2, device='cuda') + 1)"
.venv-gtx/bin/python -m mm_emotion.cli doctor
.venv-gtx/bin/python -m pytest -q
printf 'GTX 1080 Ti HF environment ready. MIA does not run on Pascal.\n'
