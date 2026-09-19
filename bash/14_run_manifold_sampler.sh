#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
CONFIG_PATH="${1:-configs/flux2_klein_4b_sampler_smoke.yaml}"
shift $(( $# > 0 ? 1 : 0 ))
"$REDIS_PYTHON" scripts/run_sampler.py --config "$CONFIG_PATH" "$@"

