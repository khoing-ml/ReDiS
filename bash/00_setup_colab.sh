#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

COLAB_PYTHON="${COLAB_PYTHON:-$(command -v python)}"
if [[ -z "$COLAB_PYTHON" || ! -x "$COLAB_PYTHON" ]]; then
  echo "Could not find the Colab Python interpreter." >&2
  exit 2
fi

# Colab images may not provide a working ensurepip for `python -m venv`.
# Install into the disposable runtime instead and record the exact interpreter
# so all later bash scripts use the same environment.
"$COLAB_PYTHON" -m pip install --upgrade pip
"$COLAB_PYTHON" -m pip install --upgrade -r requirements-colab.txt
"$COLAB_PYTHON" -m pip install --no-deps --editable .

mkdir -p .runtime
printf '%s\n' "$COLAB_PYTHON" > .runtime/python_path

"$COLAB_PYTHON" scripts/write_colab_configs.py
"$COLAB_PYTHON" scripts/check_environment.py
"$COLAB_PYTHON" -m pytest

echo
echo "Colab setup complete. Runtime configs are under .runtime/."
