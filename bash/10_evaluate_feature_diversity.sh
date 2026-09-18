#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

if [[ -z "${1:-}" ]]; then
  echo "Usage: bash bash/10_evaluate_feature_diversity.sh outputs/<method>/<run>" >&2
  exit 2
fi

"$REDIS_PYTHON" scripts/evaluate_feature_diversity.py --run-dir "$1"

