from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image

from redis.metrics import RewardEvaluator, paired_score_summary


def consistency_drift_by_sample(report: dict[str, Any]) -> dict[tuple[str, int], float]:
    result: dict[tuple[str, int], float] = {}
    for prompt_run in report["prompt_runs"]:
        prompt_id = str(prompt_run["prompt_id"])
        seeds = [int(seed) for seed in prompt_run["seeds"]]
        values = [0.0] * len(seeds)
        observed = [False] * len(seeds)
        for step in prompt_run["steps"]:
            before = step.get("consistency_before")
            after = step.get("consistency_after")
            if not isinstance(before, list) or not isinstance(after, list):
                continue
            for index, (before_value, after_value) in enumerate(zip(before, after, strict=True)):
                values[index] += float(after_value) - float(before_value)
                observed[index] = True
        for index, seed in enumerate(seeds):
            if observed[index]:
                result[(prompt_id, seed)] = values[index]
    return result


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


parser = argparse.ArgumentParser(description="Score sampler ablations with paired reward metrics.")
parser.add_argument("--run-dir", required=True)
parser.add_argument("--metrics", nargs="+", default=["pickscore"])
parser.add_argument("--batch-size", type=int, default=8)
args = parser.parse_args()
if args.batch_size < 1:
    raise SystemExit("ERROR: --batch-size must be positive")
run_dir = Path(args.run_dir)
conditions = json.loads((run_dir / "ablation.json").read_text(encoding="utf-8"))
evaluator = RewardEvaluator()
condition_scores: dict[str, dict[str, dict[tuple[str, int], float]]] = {}
condition_reports: dict[str, dict[str, Any]] = {}

for condition_record in conditions:
    condition = str(condition_record["condition"])
    report = json.loads((run_dir / f"{condition}.json").read_text(encoding="utf-8"))
    condition_reports[condition] = report
    samples: list[tuple[tuple[str, int], Path, str]] = []
    for prompt_run in report["prompt_runs"]:
        prompt_id = str(prompt_run["prompt_id"])
        prompt = str(prompt_run["prompt"])
        for image in prompt_run["images"]:
            key = (prompt_id, int(image["seed"]))
            path = run_dir / "images" / condition / str(image["file"])
            samples.append((key, path, prompt))
    by_metric = {metric: {} for metric in args.metrics}
    for offset in range(0, len(samples), args.batch_size):
        batch = samples[offset : offset + args.batch_size]
        images = [Image.open(path).convert("RGB") for _, path, _ in batch]
        scores = evaluator.evaluate(images, [prompt for _, _, prompt in batch], args.metrics)
        for image in images:
            image.close()
        for metric, values in scores.items():
            by_metric[metric].update(
                {key: value for (key, _, _), value in zip(batch, values, strict=True)}
            )
    condition_scores[condition] = by_metric
    print(json.dumps({"condition": condition, "sample_count": len(samples)}), flush=True)

native_condition = next(
    str(record["condition"])
    for record in conditions
    if record["config"]["mode"] == "native"
)
summary: dict[str, Any] = {
    "native_condition": native_condition,
    "metrics": args.metrics,
    "models": {
        "pickscore": {
            "model": "yuvalkirstain/PickScore_v1",
            "processor": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        }
    },
    "conditions": {},
}
score_records = []
for condition_record in conditions:
    condition = str(condition_record["condition"])
    result: dict[str, Any] = {}
    drift = consistency_drift_by_sample(condition_reports[condition])
    for metric in args.metrics:
        candidate = condition_scores[condition][metric]
        baseline = condition_scores[native_condition][metric]
        keys = sorted(set(candidate) & set(baseline))
        metric_summary = paired_score_summary(
            [candidate[key] for key in keys], [baseline[key] for key in keys]
        )
        correlation_keys = [key for key in keys if key in drift]
        metric_summary["consistency_drift_quality_delta_correlation"] = pearson(
            [drift[key] for key in correlation_keys],
            [candidate[key] - baseline[key] for key in correlation_keys],
        )
        result[metric] = metric_summary
        for key in keys:
            score_records.append(
                {
                    "condition": condition,
                    "prompt_id": key[0],
                    "seed": key[1],
                    "metric": metric,
                    "score": candidate[key],
                    "native_score": baseline[key],
                    "delta_vs_native": candidate[key] - baseline[key],
                    "consistency_drift": drift.get(key),
                }
            )
    summary["conditions"][condition] = result

(run_dir / "reward_scores.json").write_text(
    json.dumps(score_records, indent=2) + "\n", encoding="utf-8"
)
(run_dir / "reward_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
