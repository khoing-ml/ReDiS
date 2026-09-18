from __future__ import annotations

import argparse
import json

import torch

from redis.analysis import seed_spectrum
from redis.config import load_config
from redis.hooks import CaptureController, patch_flux2_blocks
from redis.models.flux2_klein import load_pipeline, make_generators
from redis.utils.run import make_run_dir, write_run_metadata


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
config = load_config(args.config)
generation = config["generation"]
seeds = [int(seed) for seed in generation["seeds"]]
if len(seeds) < 2:
    raise SystemExit("ERROR: capture spectrum requires at least two seeds in generation.seeds")

run_dir = make_run_dir("capture")
write_run_metadata(run_dir, config)
pipe = load_pipeline(config["model"])
capture = config.get("capture", {})
controller = CaptureController(clone_to_cpu=bool(capture.get("clone_to_cpu", True)))
family = str(capture.get("block_family", "transformer_blocks"))
layers = [int(value) for value in capture.get("layer_ids", [0])]
patch_flux2_blocks(pipe.transformer, controller, family=family, layer_ids=layers)

# One repeated prompt with multiple generators: this preserves the group constraint.
pipe(
    prompt=[str(generation["prompt"])] * len(seeds),
    height=int(generation["height"]),
    width=int(generation["width"]),
    num_inference_steps=int(generation["num_inference_steps"]),
    guidance_scale=float(generation["guidance_scale"]),
    max_sequence_length=int(generation.get("max_sequence_length", 128)),
    generator=make_generators(seeds),
    output_type="latent",
)

report: dict[str, object] = {"seeds": seeds, "sites": {}}
for (block_family, layer_id), tensors in controller.activations.items():
    site = f"{block_family}.{layer_id}"
    report["sites"][site] = []
    for invocation, hidden in enumerate(tensors):
        metrics = seed_spectrum(hidden)
        metrics["invocation"] = invocation
        metrics["shape"] = list(hidden.shape)
        report["sites"][site].append(metrics)

(run_dir / "spectra.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
print(run_dir)

