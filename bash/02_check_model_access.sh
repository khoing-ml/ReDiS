#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
"$REDIS_PYTHON" scripts/check_model_access.py --config "$CONFIG_PATH"
