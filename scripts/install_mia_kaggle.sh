#!/usr/bin/env bash
# Use dedicated venv to avoid breaking an already running Kaggle Transformers kernel.
set -euo pipefail
SUITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIA_SOURCE="${MIA_SOURCE:-${SUITE_ROOT}/external/vLLM-Hook}"
MIA_ENV="${MIA_ENV:-${SUITE_ROOT}/.venv-mia}"
MIA_REF="${MIA_REF:-main}"
python3 - <<'PY'
import sys
if sys.version_info < (3,11) or sys.version_info >= (3,15):
    raise SystemExit("IBM MIA needs Python 3.11-3.14 in this environment")
PY
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No nvidia-smi; must run on a compatible NVIDIA GPU" >&2
  exit 1
fi
nvidia-smi
mkdir -p "$(dirname "$MIA_SOURCE")"
if [ ! -d "$MIA_SOURCE/.git" ]; then
  git clone https://github.com/IBM/vLLM-Hook.git "$MIA_SOURCE"
fi
(
  cd "$MIA_SOURCE"
  git fetch origin "$MIA_REF"
  git checkout FETCH_HEAD
  echo "MIA_COMMIT=$(git rev-parse HEAD)"
  git rev-parse HEAD > "$SUITE_ROOT/MIA_COMMIT_PIN.txt"
)
python3 -m venv "$MIA_ENV"
"$MIA_ENV/bin/python" -m pip install --upgrade pip
(
  cd "$MIA_SOURCE"
  "$MIA_ENV/bin/python" -m pip install -r requirement.txt
  "$MIA_ENV/bin/python" -m pip uninstall -y torchcodec
  "$MIA_ENV/bin/python" -m pip install -e . --no-deps
  "$MIA_ENV/bin/python" -m pip install -r "$SUITE_ROOT/requirements-core.txt"
)
echo "MIA environment created: $MIA_ENV"
echo "Check actual import with: $MIA_ENV/bin/python -c 'from mia import MiaLLM; print(\"MIA import OK\")'"
echo "Then launch a separate process: $MIA_ENV/bin/python -m mm_emotion.cli run --backend mia ..."
