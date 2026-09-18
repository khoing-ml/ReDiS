#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
.venv/bin/python scripts/test_wrapper_identity.py --config "$CONFIG_PATH"

