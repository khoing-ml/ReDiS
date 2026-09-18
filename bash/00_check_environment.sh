#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

UV_BIN="${UV_BIN:-uv}"
if [[ ! -x .venv/bin/python ]]; then
  "$UV_BIN" venv --python 3.12 --seed .venv
fi

if [[ "${REDIS_INSTALL_QUANT:-1}" == "1" ]]; then
  "$UV_BIN" sync --extra test --extra quantization
else
  "$UV_BIN" sync --extra test
fi

.venv/bin/python scripts/check_environment.py
