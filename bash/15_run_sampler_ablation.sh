#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
if [[ -f .runtime/colab_hardware.json && ! -f .runtime/colab_ccsr_debug.yaml ]]; then
  "$REDIS_PYTHON" scripts/write_colab_configs.py >/dev/null
fi
if [[ -n "${1:-}" ]]; then
  CONFIG_PATH="$1"
elif [[ -f .runtime/colab_ccsr_debug.yaml ]]; then
  CONFIG_PATH=.runtime/colab_ccsr_debug.yaml
else
  CONFIG_PATH=configs/flux2_klein_4b_ccsr_debug.yaml
fi
"$REDIS_PYTHON" scripts/run_sampler_ablation.py --config "$CONFIG_PATH"
