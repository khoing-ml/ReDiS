from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from redis.benchmarks import PromptRecord, load_prompt_records
from redis.config import load_config
from redis.metrics import aggregate_sampler_steps
from redis.models.flux2_klein import load_pipeline, make_generators
from redis.sampling import SamplingRefinementConfig, SamplingRefinementController
from redis.utils.run import make_run_dir, write_run_metadata


parser = argparse.ArgumentParser(
    description="Generate matched-seed consistency-constrained sampler ablations."
)
parser.add_argument("--config", required=True)
args = parser.parse_args()

config = load_config(args.config)
generation = config["generation"]
base = SamplingRefinementConfig.from_mapping(config.get("sampler"))
ablation = config.get("sampler_ablation", {})
seeds = [int(seed) for seed in generation["seeds"]]
prompt_file = generation.get("prompt_file")
if prompt_file is not None:
    prompts = load_prompt_records(Path(prompt_file))
else:
    prompts = [PromptRecord(id="inline_0000", prompt=str(generation["prompt"]))]
max_prompts = generation.get("max_prompts")
if max_prompts is not None:
    prompts = prompts[: int(max_prompts)]

condition_specs = ablation.get("conditions")
conditions: list[tuple[str, SamplingRefinementConfig]] = []
if condition_specs is not None:
    if not isinstance(condition_specs, list) or not condition_specs:
        raise SystemExit("ERROR: sampler_ablation.conditions must be a non-empty list")
    for index, specification in enumerate(condition_specs):
        if not isinstance(specification, dict):
            raise SystemExit("ERROR: every sampler ablation condition must be a mapping")
        values = dict(specification)
        name = str(values.pop("name", f"condition_{index}"))
        condition_config = SamplingRefinementConfig.from_mapping(
            {**asdict(base), **values, "capture_trajectory": False}
        )
        conditions.append((name, condition_config))
else:
    modes = [
        str(value)
        for value in ablation.get(
            "modes", ["native", "naive", "subspace", "non_increasing"]
        )
    ]
    estimators = [
        str(value) for value in ablation.get("normal_estimators", [base.normal_estimator])
    ]
    for mode in modes:
        if mode in ("tangent", "non_increasing"):
            for estimator in estimators:
                conditions.append(
                    (
                        f"{mode}_{estimator}",
                        replace(
                            base,
                            mode=mode,
                            normal_estimator=estimator,
                            capture_trajectory=False,
                        ),
                    )
                )
        else:
            conditions.append(
                (
                    mode,
                    replace(
                        base,
                        mode=mode,
                        normal_estimator="residual_proxy",
                        capture_trajectory=False,
                    ),
                )
            )

resolved = dict(config)
resolved["sampler_ablation_conditions"] = [
    {"name": name, **asdict(condition_config)} for name, condition_config in conditions
]
pipe = load_pipeline(config["model"])
run_dir = make_run_dir("sampler_ablation")
write_run_metadata(run_dir, resolved)
records = []
manifest = [asdict(prompt) for prompt in prompts]
(run_dir / "prompt_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

for condition, sampler_config in conditions:
    if sampler_config.normal_estimator == "vjp" and float(generation["guidance_scale"]) > 1:
        raise SystemExit("ERROR: VJP normal requires guidance_scale <= 1")
    output_dir = run_dir / "images" / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    condition_started = time.perf_counter()
    prompt_runs = []
    for prompt_index, prompt_record in enumerate(prompts):
        controller = SamplingRefinementController(
            pipe.scheduler,
            sampler_config,
            transformer=pipe.transformer,
        )
        started = time.perf_counter()
        with controller:
            result = pipe(
                prompt=[prompt_record.prompt] * len(seeds),
                height=int(generation["height"]),
                width=int(generation["width"]),
                num_inference_steps=int(generation["num_inference_steps"]),
                guidance_scale=float(generation["guidance_scale"]),
                max_sequence_length=int(generation.get("max_sequence_length", 128)),
                generator=make_generators(seeds),
            )
        prompt_latency = time.perf_counter() - started
        prompt_dir = output_dir / f"prompt_{prompt_index:04d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        images = []
        for seed, image in zip(seeds, result.images, strict=True):
            filename = f"seed_{seed}.png"
            image.save(prompt_dir / filename)
            images.append({"seed": seed, "file": str(Path(prompt_dir.name) / filename)})
        run_report = controller.report()
        prompt_runs.append(
            {
                "prompt_index": prompt_index,
                "prompt_id": prompt_record.id,
                "prompt": prompt_record.prompt,
                "category": prompt_record.category,
                "seeds": seeds,
                "images": images,
                "latency_seconds": prompt_latency,
                "steps": run_report["steps"],
            }
        )
        print(
            json.dumps(
                {
                    "condition": condition,
                    "prompt_index": prompt_index,
                    "prompt_count": len(prompts),
                    "latency_seconds": prompt_latency,
                }
            ),
            flush=True,
        )
    latency = time.perf_counter() - condition_started
    all_steps = [step for prompt_run in prompt_runs for step in prompt_run["steps"]]
    condition_report = {
        "config": asdict(sampler_config),
        "prompt_runs": prompt_runs,
        "aggregate": aggregate_sampler_steps(all_steps),
        "condition": condition,
        "seeds": seeds,
        "prompt_count": len(prompts),
        "latency_seconds": latency,
        "peak_vram_gib": (
            torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None
        ),
    }
    (run_dir / f"{condition}.json").write_text(
        json.dumps(condition_report, indent=2) + "\n", encoding="utf-8"
    )
    records.append(
        {
            "condition": condition,
            "config": asdict(sampler_config),
            "latency_seconds": latency,
            "aggregate": condition_report["aggregate"],
        }
    )
    print(json.dumps(records[-1]), flush=True)

(run_dir / "ablation.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
print(run_dir)
