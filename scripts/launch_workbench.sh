#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m streamlit run app_emotion.py --server.address 127.0.0.1
