#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

if [[ -z "${1:-}" ]]; then
  echo "Usage: bash bash/16_evaluate_sampler_ablation.sh outputs/sampler_ablation/<run>" >&2
  exit 2
fi

RUN_DIR="$1"
if [[ ! -d "$RUN_DIR/images" ]]; then
  echo "Missing ablation image directory: $RUN_DIR/images" >&2
  exit 2
fi

for CONDITION_DIR in "$RUN_DIR"/images/*; do
  [[ -d "$CONDITION_DIR" ]] || continue
  "$REDIS_PYTHON" scripts/evaluate_feature_diversity.py --run-dir "$CONDITION_DIR"
done

"$REDIS_PYTHON" scripts/summarize_sampler_ablation.py --run-dir "$RUN_DIR"
