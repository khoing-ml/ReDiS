from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    for section in ("model", "generation"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing mapping '{section}' in {config_path}")
    return config

