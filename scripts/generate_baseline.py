from __future__ import annotations

import argparse
import json
import time

import torch

from redis.config import load_config
from redis.models.flux2_klein import load_pipeline, make_generators
from redis.utils.run import make_run_dir, write_run_metadata


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
config = load_config(args.config)
generation = config["generation"]
seeds = [int(seed) for seed in generation["seeds"]]
run_dir = make_run_dir("baseline")
write_run_metadata(run_dir, config)

pipe = load_pipeline(config["model"])
prompt = str(generation["prompt"])
records = []

for seed in seeds:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    image = pipe(
        prompt=prompt,
        height=int(generation["height"]),
        width=int(generation["width"]),
        num_inference_steps=int(generation["num_inference_steps"]),
        guidance_scale=float(generation["guidance_scale"]),
        max_sequence_length=int(generation.get("max_sequence_length", 128)),
        generator=make_generators([seed])[0],
    ).images[0]
    latency = time.perf_counter() - started
    filename = f"seed_{seed}.png"
    image.save(run_dir / filename)
    records.append(
        {
            "seed": seed,
            "file": filename,
            "latency_seconds": latency,
            "peak_vram_gib": (
                torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None
            ),
        }
    )

(run_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
print(run_dir)

