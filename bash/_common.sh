#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f .runtime/python_path ]]; then
  REDIS_PYTHON="$(<.runtime/python_path)"
  if [[ ! -x "$REDIS_PYTHON" ]]; then
    echo "Saved Python interpreter is unavailable: $REDIS_PYTHON" >&2
    exit 2
  fi
elif [[ -x .venv/bin/python ]]; then
  REDIS_PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
  echo "Environment is not set up. Run bash/00_check_environment.sh locally or bash/00_setup_colab.sh on Colab." >&2
  exit 2
fi
export REDIS_PYTHON

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_XET_HIGH_PERFORMANCE=1

DEFAULT_CONFIG="configs/flux2_klein_4b_low_vram.yaml"
CONFIG_PATH="${1:-$DEFAULT_CONFIG}"
