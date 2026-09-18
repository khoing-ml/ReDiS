from __future__ import annotations

import argparse
import json
import time

import torch

from redis.config import load_config
from redis.hooks import InterventionController, patch_flux2_blocks
from redis.models.flux2_klein import load_pipeline, make_generators
from redis.utils.run import make_run_dir, write_run_metadata


METHODS = (
    "gaussian_noise",
    "residual_amplification",
    "projected_amplification",
    "gram_isotropization",
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--method", required=True, choices=METHODS)
args = parser.parse_args()
config = load_config(args.config)
generation = config["generation"]
seeds = [int(seed) for seed in generation["seeds"]]
if len(seeds) < 2:
    raise SystemExit("ERROR: interventions require at least two seeds for one prompt")
if args.method == "gram_isotropization" and len(seeds) < 4:
    raise SystemExit("ERROR: Gram isotropization smoke test requires at least four seeds")

run_dir = make_run_dir(args.method)
write_run_metadata(run_dir, config)
pipe = load_pipeline(config["model"])
intervention = config.get("intervention", {})
projection = config.get("projection", {})
controller = InterventionController(
    method=args.method,
    timestep_ids=[int(value) for value in intervention.get("timestep_ids", [0])],
    gamma=float(intervention.get("gamma", 0.3)),
    sigma=float(intervention.get("sigma", 0.1)),
    projection_rank=int(projection.get("rank", 64)),
    projection_seed=int(projection.get("seed", 0)),
    beta=float(intervention.get("beta", 0.5)),
    match_rms=bool(intervention.get("rms_match", True)),
    max_relative_correction_norm=float(
        intervention.get("max_relative_correction_norm", 0.25)
    ),
)
capture = config.get("capture", {})
family = str(capture.get("block_family", "transformer_blocks"))
layer_ids = [int(value) for value in capture.get("layer_ids", [0])]
patch_flux2_blocks(pipe.transformer, controller, family=family, layer_ids=layer_ids)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
result = pipe(
    prompt=[str(generation["prompt"])] * len(seeds),
    height=int(generation["height"]),
    width=int(generation["width"]),
    num_inference_steps=int(generation["num_inference_steps"]),
    guidance_scale=float(generation["guidance_scale"]),
    max_sequence_length=int(generation.get("max_sequence_length", 128)),
    generator=make_generators(seeds),
)
latency = time.perf_counter() - started
for seed, image in zip(seeds, result.images, strict=True):
    image.save(run_dir / f"seed_{seed}.png")

report = {
    "method": args.method,
    "seeds": seeds,
    "latency_seconds": latency,
    "peak_vram_gib": (
        torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None
    ),
    "interventions": controller.logs,
}
(run_dir / "intervention.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
print(run_dir)

