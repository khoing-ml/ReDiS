from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptRecord:
    id: str
    prompt: str
    source: str | None = None
    category: str | None = None


def _record(value: Any, index: int) -> PromptRecord:
    if isinstance(value, str):
        prompt = value.strip()
        data: dict[str, Any] = {}
    elif isinstance(value, dict):
        data = value
        prompt = str(data.get("prompt", "")).strip()
    else:
        raise ValueError(f"Prompt record {index} must be a string or mapping")
    if not prompt:
        raise ValueError(f"Prompt record {index} is empty")
    return PromptRecord(
        id=str(data.get("id", index)),
        prompt=prompt,
        source=str(data["source"]) if data.get("source") is not None else None,
        category=str(data["category"]) if data.get("category") is not None else None,
    )


def load_prompt_records(path: str | Path) -> list[PromptRecord]:
    """Load immutable benchmark prompts from JSONL, JSON, or plain text."""
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise ValueError(f"Prompt file does not exist: {prompt_path}")
    if prompt_path.suffix == ".jsonl":
        values = [
            json.loads(line)
            for line in prompt_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif prompt_path.suffix == ".json":
        values = json.loads(prompt_path.read_text(encoding="utf-8"))
        if isinstance(values, dict):
            values = values.get("prompts")
        if not isinstance(values, list):
            raise ValueError(f"JSON prompt file must contain a list: {prompt_path}")
    else:
        values = [line for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [_record(value, index) for index, value in enumerate(values)]
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Prompt IDs must be unique: {prompt_path}")
    if not records:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    return records
