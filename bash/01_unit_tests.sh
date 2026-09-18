#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
"$REDIS_PYTHON" -m pytest
