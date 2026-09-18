#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

if ! "$REDIS_PYTHON" -c 'import torchvision' >/dev/null 2>&1; then
  TORCHVISION_VERSION="$($REDIS_PYTHON - <<'PY'
import torch

versions = {
    (2, 5): "0.20.1",
    (2, 6): "0.21.0",
    (2, 7): "0.22.1",
    (2, 8): "0.23.0",
    (2, 9): "0.24.0",
}
parts = torch.__version__.split("+", 1)[0].split(".")
key = (int(parts[0]), int(parts[1]))
if key not in versions:
    raise SystemExit(f"No safe torchvision mapping for torch {torch.__version__}")
print(versions[key])
PY
)"
  "$REDIS_PYTHON" -m pip install --no-deps "torchvision==$TORCHVISION_VERSION"
fi

"$REDIS_PYTHON" -m pip install -r requirements-metrics.txt
"$REDIS_PYTHON" -m pip install --no-deps \
  dreamsim==0.2.1 \
  hpsv2==1.2.0 \
  lpips==0.1.4

read -r OPEN_CLIP_BPE HPS_BPE < <("$REDIS_PYTHON" - <<'PY'
import importlib.util
from pathlib import Path

open_clip = Path(next(iter(importlib.util.find_spec("open_clip").submodule_search_locations)))
hpsv2 = Path(next(iter(importlib.util.find_spec("hpsv2").submodule_search_locations)))
print(
    open_clip / "bpe_simple_vocab_16e6.txt.gz",
    hpsv2 / "src/open_clip/bpe_simple_vocab_16e6.txt.gz",
)
PY
)
if [[ ! -f "$HPS_BPE" ]]; then
  cp "$OPEN_CLIP_BPE" "$HPS_BPE"
fi

"$REDIS_PYTHON" - <<'PY'
import sys
import types

import dreamsim  # noqa: F401
import hpsv2  # noqa: F401
import lpips  # noqa: F401
import torch
import torchvision

turtle_stub = types.ModuleType("turtle")
turtle_stub.forward = None
sys.modules.setdefault("turtle", turtle_stub)
from hpsv2 import img_score  # noqa: E402,F401

print({"torch": torch.__version__, "torchvision": torchvision.__version__})
PY
echo "Output metric dependencies installed."
