from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def make_run_dir(kind: str) -> Path:
    root = Path(os.environ.get("REDIS_OUTPUT_ROOT", "outputs"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / kind / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_metadata(run_dir: Path, config: dict[str, Any]) -> None:
    import diffusers
    import torch
    import transformers

    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (run_dir / "environment.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

