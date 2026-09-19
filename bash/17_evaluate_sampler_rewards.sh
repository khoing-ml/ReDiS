#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

RUN_DIR="${1:-latest}"
if [[ "$RUN_DIR" == "latest" || "$RUN_DIR" == *"RUN_TIMESTAMP"* ]]; then
  RUN_DIR="$(find outputs/sampler_ablation -mindepth 2 -maxdepth 2 -type f -name ablation.json -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "$RUN_DIR" ]]; then
    echo "No completed sampler ablation found. Generation must finish before evaluation." >&2
    exit 2
  fi
fi

"$REDIS_PYTHON" scripts/evaluate_sampler_rewards.py \
  --run-dir "$RUN_DIR" \
  --metrics pickscore
"$REDIS_PYTHON" scripts/summarize_sampler_ablation.py --run-dir "$RUN_DIR"
