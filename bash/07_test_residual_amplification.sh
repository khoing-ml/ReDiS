#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
CONFIG_PATH="${1:-configs/flux2_klein_4b_intervention_smoke.yaml}"
.venv/bin/python scripts/run_intervention.py --config "$CONFIG_PATH" --method residual_amplification

