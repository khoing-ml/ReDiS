#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
"$REDIS_PYTHON" scripts/generate_baseline.py --config "$CONFIG_PATH"
