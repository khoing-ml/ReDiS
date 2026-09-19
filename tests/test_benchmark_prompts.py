import json

import pytest

from redis.benchmarks import load_prompt_records


def test_load_jsonl_prompt_records(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps({"id": "p1", "prompt": "A cat", "category": "animals"}) + "\n",
        encoding="utf-8",
    )
    records = load_prompt_records(path)
    assert records[0].id == "p1"
    assert records[0].prompt == "A cat"
    assert records[0].category == "animals"


def test_prompt_ids_must_be_unique(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        '{"id":"same","prompt":"one"}\n{"id":"same","prompt":"two"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_prompt_records(path)
