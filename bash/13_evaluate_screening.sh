#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

if [[ -z "${1:-}" ]]; then
  echo "Usage: bash bash/13_evaluate_screening.sh outputs/screening_site_A/<run>" >&2
  exit 2
fi

"$REDIS_PYTHON" scripts/evaluate_screening.py --run-dir "$1"
