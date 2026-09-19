from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import urllib.request
from pathlib import Path


PICKAPIC_REVISION = "c45a875adf0fe5f2896e46bbef2124f4953dcaa6"
DRAWBENCH_REVISION = "ad794cba698c253ffdb6d18eff3fc87b004c1135"
PICKAPIC_URL = (
    "https://huggingface.co/datasets/Jialuo21/pickapic-test/resolve/"
    f"{PICKAPIC_REVISION}/test.jsonl"
)
DRAWBENCH_URL = (
    "https://huggingface.co/datasets/sayakpaul/drawbench/resolve/"
    f"{DRAWBENCH_REVISION}/DrawBench%20Prompts%20-%20Sheet1.csv"
)
SUBSET_SEED = 230501569


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "ReDiS/benchmark-preparation"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def write_immutable(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"Refusing to overwrite changed immutable prompt file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def serialize_jsonl(records: list[dict[str, object]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def serialize_text(records: list[dict[str, object]]) -> str:
    return "".join(str(record["prompt"]).replace("\n", " ") + "\n" for record in records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare pinned CCSR benchmark prompt sets.")
    parser.add_argument("--output-dir", default="prompts")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    pickapic = []
    for line in fetch_text(PICKAPIC_URL).splitlines():
        source = json.loads(line)
        pickapic.append(
            {
                "id": f"pickapic_test_unique_{int(source['id']):04d}",
                "prompt": str(source["prompt"]),
                "source": "pickapic-anonymous/pickapic_v1:test_unique",
            }
        )
    if len(pickapic) != 500 or len({row["prompt"] for row in pickapic}) != 500:
        raise RuntimeError("Expected exactly 500 unique Pick-a-Pic test prompts")

    indices = list(range(len(pickapic)))
    random.Random(SUBSET_SEED).shuffle(indices)
    sets = {
        "pickscore_debug": [pickapic[index] for index in indices[:20]],
        "pickscore_eval_100": [pickapic[index] for index in indices[:100]],
        "pickscore_eval_500": pickapic,
    }

    drawbench = []
    for index, source in enumerate(csv.DictReader(io.StringIO(fetch_text(DRAWBENCH_URL)))):
        drawbench.append(
            {
                "id": f"drawbench_{index:03d}",
                "prompt": str(source["Prompts"]),
                "source": "sayakpaul/drawbench:train",
                "category": str(source["Category"]),
            }
        )
    if len(drawbench) != 200:
        raise RuntimeError("Expected exactly 200 DrawBench prompts")
    sets["drawbench"] = drawbench

    files: dict[str, dict[str, object]] = {}
    for name, records in sets.items():
        jsonl = serialize_jsonl(records)
        text = serialize_text(records)
        jsonl_path = output_dir / f"{name}.jsonl"
        text_path = output_dir / f"{name}.txt"
        write_immutable(jsonl_path, jsonl)
        write_immutable(text_path, text)
        files[jsonl_path.name] = {
            "count": len(records),
            "sha256": hashlib.sha256(jsonl.encode()).hexdigest(),
        }
        files[text_path.name] = {
            "count": len(records),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }

    metadata = {
        "subset_seed": SUBSET_SEED,
        "sources": {
            "pickapic": {
                "dataset": "Jialuo21/pickapic-test",
                "revision": PICKAPIC_REVISION,
                "upstream": "pickapic-anonymous/pickapic_v1:test_unique",
                "url": PICKAPIC_URL,
            },
            "drawbench": {
                "dataset": "sayakpaul/drawbench",
                "revision": DRAWBENCH_REVISION,
                "url": DRAWBENCH_URL,
            },
        },
        "files": files,
    }
    write_immutable(
        output_dir / "metadata.json",
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
