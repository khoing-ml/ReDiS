#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
.venv/bin/python scripts/check_model_access.py --config "$CONFIG_PATH"

