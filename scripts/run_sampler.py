from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import torch

from redis.config import load_config
from redis.models.flux2_klein import load_pipeline, make_generators
from redis.sampling import MODES, NORMAL_ESTIMATORS, SamplingRefinementConfig, SamplingRefinementController
from redis.utils.run import make_run_dir, write_run_metadata


parser = argparse.ArgumentParser(
    description="Run native or consistency-non-increasing sampling on FLUX.2 Klein."
)
parser.add_argument("--config", required=True)
parser.add_argument("--mode", choices=MODES)
parser.add_argument("--normal-estimator", choices=NORMAL_ESTIMATORS)
parser.add_argument("--strength", type=float)
parser.add_argument("--target-correction-norm", type=float)
args = parser.parse_args()

config = load_config(args.config)
generation = config["generation"]
sampler_config = SamplingRefinementConfig.from_mapping(config.get("sampler"))
overrides = {}
if args.mode is not None:
    overrides["mode"] = args.mode
if args.normal_estimator is not None:
    overrides["normal_estimator"] = args.normal_estimator
if args.strength is not None:
    overrides["strength"] = args.strength
if args.target_correction_norm is not None:
    overrides["target_relative_correction_norm"] = args.target_correction_norm
if overrides:
    sampler_config = SamplingRefinementConfig.from_mapping(
        {**asdict(sampler_config), **overrides}
    )

seeds = [int(seed) for seed in generation["seeds"]]
if sampler_config.normal_estimator == "vjp" and float(generation["guidance_scale"]) > 1:
    raise SystemExit(
        "ERROR: VJP x0 normal currently requires guidance_scale <= 1 (one transformer pass per step)"
    )

resolved = dict(config)
resolved["sampler"] = asdict(sampler_config)
run_dir = make_run_dir(f"sampler_{sampler_config.mode}_{sampler_config.normal_estimator}")
write_run_metadata(run_dir, resolved)
pipe = load_pipeline(config["model"])
controller = SamplingRefinementController(
    pipe.scheduler,
    sampler_config,
    transformer=pipe.transformer,
)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
with controller:
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

images = []
for seed, image in zip(seeds, result.images, strict=True):
    filename = f"seed_{seed}.png"
    image.save(run_dir / filename)
    images.append({"seed": seed, "file": filename})

report = controller.report()
report.update(
    {
        "prompt": str(generation["prompt"]),
        "seeds": seeds,
        "images": images,
        "latency_seconds": latency,
        "peak_vram_gib": (
            torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None
        ),
    }
)
(run_dir / "sampler.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
if controller.trajectory:
    torch.save(controller.trajectory, run_dir / "trajectory.pt")
print(json.dumps(report, indent=2))
print(run_dir)
