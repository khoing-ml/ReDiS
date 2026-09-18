#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
"$REDIS_PYTHON" scripts/test_wrapper_identity.py --config "$CONFIG_PATH"
