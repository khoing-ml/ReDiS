from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace

import torch

from redis.config import load_config
from redis.models.flux2_klein import load_pipeline, make_generators
from redis.sampling import SamplingRefinementConfig, SamplingRefinementController
from redis.utils.run import make_run_dir, write_run_metadata


parser = argparse.ArgumentParser(
    description="Generate matched-seed native/naive/subspace/tangent sampler ablations."
)
parser.add_argument("--config", required=True)
args = parser.parse_args()

config = load_config(args.config)
generation = config["generation"]
base = SamplingRefinementConfig.from_mapping(config.get("sampler"))
ablation = config.get("sampler_ablation", {})
seeds = [int(seed) for seed in generation["seeds"]]

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
        condition_config = replace(base, **values, capture_trajectory=False)
        condition_config.validate()
        conditions.append((name, condition_config))
else:
    modes = [
        str(value)
        for value in ablation.get("modes", ["native", "naive", "subspace", "tangent"])
    ]
    estimators = [
        str(value) for value in ablation.get("normal_estimators", [base.normal_estimator])
    ]
    for mode in modes:
        if mode == "tangent":
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
                        normal_estimator="proxy",
                        capture_trajectory=False,
                    ),
                )
            )

resolved = dict(config)
resolved["sampler_ablation_conditions"] = [
    {"name": name, **asdict(condition_config)} for name, condition_config in conditions
]
run_dir = make_run_dir("sampler_ablation")
write_run_metadata(run_dir, resolved)
pipe = load_pipeline(config["model"])
records = []

for condition, sampler_config in conditions:
    if sampler_config.normal_estimator == "exact" and float(generation["guidance_scale"]) > 1:
        raise SystemExit("ERROR: exact normal requires guidance_scale <= 1")
    controller = SamplingRefinementController(
        pipe.scheduler,
        sampler_config,
        transformer=pipe.transformer,
    )
    output_dir = run_dir / "images" / condition
    output_dir.mkdir(parents=True, exist_ok=True)
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
    for seed, image in zip(seeds, result.images, strict=True):
        image.save(output_dir / f"seed_{seed}.png")
    condition_report = controller.report()
    condition_report.update(
        condition=condition,
        seeds=seeds,
        latency_seconds=latency,
        peak_vram_gib=(
            torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None
        ),
    )
    (run_dir / f"{condition}.json").write_text(
        json.dumps(condition_report, indent=2) + "\n", encoding="utf-8"
    )
    records.append(
        {
            "condition": condition,
            "config": asdict(sampler_config),
            "latency_seconds": latency,
        }
    )
    print(json.dumps(records[-1]), flush=True)

(run_dir / "ablation.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
print(run_dir)
