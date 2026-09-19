#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

if [[ -z "${1:-}" ]]; then
  echo "Usage: bash bash/17_evaluate_sampler_rewards.sh outputs/sampler_ablation/<run>" >&2
  exit 2
fi

"$REDIS_PYTHON" scripts/evaluate_sampler_rewards.py \
  --run-dir "$1" \
  --metrics pickscore
"$REDIS_PYTHON" scripts/summarize_sampler_ablation.py --run-dir "$1"
