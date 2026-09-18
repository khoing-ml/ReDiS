#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -x .venv/bin/python ]]; then
  python -m venv --system-site-packages .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --upgrade -r requirements-colab.txt
.venv/bin/python -m pip install --no-deps --editable .

.venv/bin/python scripts/write_colab_configs.py
.venv/bin/python scripts/check_environment.py
.venv/bin/pytest

echo
echo "Colab setup complete. Runtime configs are under .runtime/."

