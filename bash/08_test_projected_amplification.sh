#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
CONFIG_PATH="${1:-configs/flux2_klein_4b_intervention_smoke.yaml}"
"$REDIS_PYTHON" scripts/run_intervention.py --config "$CONFIG_PATH" --method projected_amplification
