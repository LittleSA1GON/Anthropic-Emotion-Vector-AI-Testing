#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python launch.py "$@"
fi
exec python3 launch.py "$@"
