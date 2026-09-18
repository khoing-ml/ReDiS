#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run: bash bash/00_check_environment.sh" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_XET_HIGH_PERFORMANCE=1

DEFAULT_CONFIG="configs/flux2_klein_4b_low_vram.yaml"
CONFIG_PATH="${1:-$DEFAULT_CONFIG}"
