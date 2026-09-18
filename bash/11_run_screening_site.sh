#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

SITE="${1:-A}"
CONFIG_PATH="${2:-.runtime/colab_screening.yaml}"

if [[ ! "$SITE" =~ ^[ABC]$ ]]; then
  echo "Usage: bash bash/11_run_screening_site.sh [A|B|C] [config]" >&2
  exit 2
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH. Run bash/00_setup_colab.sh first." >&2
  exit 2
fi

"$REDIS_PYTHON" scripts/run_screening.py --config "$CONFIG_PATH" --site "$SITE"
