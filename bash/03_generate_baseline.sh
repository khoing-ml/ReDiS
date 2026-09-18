#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
.venv/bin/python scripts/generate_baseline.py --config "$CONFIG_PATH"

