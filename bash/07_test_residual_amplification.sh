#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
CONFIG_PATH="${1:-configs/flux2_klein_4b_intervention_smoke.yaml}"
TARGET_ARGS=()
if [[ -n "${2:-}" ]]; then TARGET_ARGS=(--target-correction-norm "$2"); fi
"$REDIS_PYTHON" scripts/run_intervention.py --config "$CONFIG_PATH" --method residual_amplification "${TARGET_ARGS[@]}"
