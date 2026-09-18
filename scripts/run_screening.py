from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from redis.config import load_config
from redis.hooks import (
    ImageTokenObserver,
    InterventionController,
    patch_flux2_blocks,
    restore_flux2_blocks,
)
from redis.models.flux2_klein import load_pipeline, make_generators
from redis.utils.run import make_run_dir, write_run_metadata


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--site", required=True, choices=("A", "B", "C"))
args = parser.parse_args()

config = load_config(args.config)
generation = config["generation"]
screening = config.get("screening", {})
site = screening.get("sites", {}).get(args.site)
if not isinstance(site, dict):
    raise SystemExit(f"ERROR: screening site {args.site} is missing")
methods = [str(value) for value in screening.get("methods", [])]
required_methods = {
    "identity",
    "gaussian_noise",
    "residual_amplification",
    "full_hidden_isotropization",
    "low_frequency_isotropization",
    "token_pooled_isotropization",
}
if set(methods) != required_methods:
    raise SystemExit(
        "ERROR: screening.methods must contain exactly the six decisive-screen methods"
    )
targets = [float(value) for value in screening.get("target_relative_norms", [])]
if targets != [0.005, 0.01, 0.02]:
    raise SystemExit("ERROR: target_relative_norms must be [0.005, 0.010, 0.020]")
seeds = [int(value) for value in generation["seeds"]]
if len(seeds) != 8:
    raise SystemExit("ERROR: decisive screening requires exactly eight seeds")

prompt_path = Path(str(screening.get("prompt_manifest", "")))
prompt_data = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
prompts = [str(value) for value in prompt_data.get("prompts", [])]
if len(prompts) != 8:
    raise SystemExit("ERROR: decisive screening requires exactly eight prompts")

run_dir = make_run_dir(f"screening_site_{args.site}")
resolved = dict(config)
resolved["selected_site"] = {"name": args.site, **site}
resolved["screening_prompts"] = prompts
write_run_metadata(run_dir, resolved)
pipe = load_pipeline(config["model"])
intervention = config["intervention"]
projection = config["projection"]
records_path = run_dir / "samples.jsonl"
logs_path = run_dir / "interventions.jsonl"


def generate(prompt: str, prompt_id: int, method: str, target: float | None) -> None:
    controller: InterventionController | None = None
    if method != "identity":
        controller = InterventionController(
            method=method,
            timestep_ids=[int(site["timestep_id"])],
            gamma=float(intervention.get("gamma", 1.0)),
            sigma=float(intervention.get("sigma", 1.0)),
            projection_rank=int(projection.get("rank", 256)),
            projection_seed=int(projection.get("seed", 0)),
            beta=float(intervention.get("beta", 0.5)),
            match_rms=bool(intervention.get("rms_match", True)),
            max_relative_correction_norm=float(
                intervention.get("max_relative_correction_norm", 0.25)
            ),
            target_relative_correction_norm=target,
        )
        if site["family"] == "single_transformer_blocks":
            patch_flux2_blocks(
                pipe.transformer,
                ImageTokenObserver(controller),
                family="transformer_blocks",
                layer_ids=[0],
            )
        patch_flux2_blocks(
            pipe.transformer,
            controller,
            family=str(site["family"]),
            layer_ids=[int(site["layer_id"])],
        )

    target_tag = "identity" if target is None else f"r{target:.3f}"
    condition = method if target is None else f"{method}__{target_tag}"
    output_dir = run_dir / "images" / condition / f"prompt_{prompt_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        result = pipe(
            prompt=[prompt] * len(seeds),
            height=int(generation["height"]),
            width=int(generation["width"]),
            num_inference_steps=int(generation["num_inference_steps"]),
            guidance_scale=float(generation["guidance_scale"]),
            max_sequence_length=int(generation.get("max_sequence_length", 128)),
            generator=make_generators(seeds),
        )
    finally:
        if controller is not None:
            restore_flux2_blocks(pipe.transformer)
    latency = time.perf_counter() - started
    with records_path.open("a", encoding="utf-8") as records:
        for seed, image in zip(seeds, result.images, strict=True):
            image_path = output_dir / f"seed_{seed}.png"
            image.save(image_path)
            records.write(
                json.dumps(
                    {
                        "site": args.site,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "method": method,
                        "target_relative_norm": target,
                        "condition": condition,
                        "seed": seed,
                        "image_path": str(image_path.relative_to(run_dir)),
                        "latency_seconds_for_batch": latency,
                    }
                )
                + "\n"
            )
    if controller is not None:
        if len(controller.logs) != 1:
            raise RuntimeError(
                f"Expected one intervention event, got {len(controller.logs)}"
            )
        with logs_path.open("a", encoding="utf-8") as logs:
            logs.write(
                json.dumps(
                    {
                        "site": args.site,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "method": method,
                        "target_relative_norm": target,
                        "event": controller.logs[0],
                    }
                )
                + "\n"
            )
    print(
        json.dumps(
            {
                "site": args.site,
                "prompt": prompt_id,
                "method": method,
                "target": target,
                "seconds": round(latency, 2),
            }
        ),
        flush=True,
    )


for prompt_id, prompt in enumerate(prompts):
    generate(prompt, prompt_id, "identity", None)
    for method in methods:
        if method == "identity":
            continue
        for target in targets:
            generate(prompt, prompt_id, method, target)

summary = {
    "run_dir": str(run_dir),
    "site": {"name": args.site, **site},
    "prompts": len(prompts),
    "seeds_per_prompt": len(seeds),
    "conditions": 1 + (len(methods) - 1) * len(targets),
    "images": len(prompts) * len(seeds) * (1 + (len(methods) - 1) * len(targets)),
}
(run_dir / "screening_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
