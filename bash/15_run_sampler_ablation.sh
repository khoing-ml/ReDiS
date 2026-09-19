#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
CONFIG_PATH="${1:-configs/flux2_klein_4b_ccsr_debug.yaml}"
"$REDIS_PYTHON" scripts/run_sampler_ablation.py --config "$CONFIG_PATH"
