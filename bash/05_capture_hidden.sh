#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
CONFIG_PATH="${1:-configs/flux2_klein_4b_capture_smoke.yaml}"
.venv/bin/python scripts/capture_hidden.py --config "$CONFIG_PATH"
