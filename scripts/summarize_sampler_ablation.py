from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


parser = argparse.ArgumentParser(
    description="Merge sampler trajectory diagnostics and image feature metrics."
)
parser.add_argument("--run-dir", required=True)
args = parser.parse_args()
run_dir = Path(args.run_dir)

ablation_path = run_dir / "ablation.json"
if not ablation_path.exists():
    raise SystemExit(f"ERROR: missing {ablation_path}")
conditions = json.loads(ablation_path.read_text(encoding="utf-8"))


def flattened_step_values(steps: list[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for step in steps:
        item = step.get(key, [])
        if isinstance(item, list):
            values.extend(float(value) for value in item)
    return values


rows: list[dict[str, object]] = []
for condition_record in conditions:
    condition = str(condition_record["condition"])
    report_path = run_dir / f"{condition}.json"
    metrics_path = run_dir / "images" / condition / "feature_metrics.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else None
    )
    steps = report["steps"]
    consistency = flattened_step_values(steps[1:], "x0_consistency_norm")
    active_steps = [step for step in steps if step.get("active")]
    corrections = flattened_step_values(active_steps, "correction_relative_norm")
    retention = flattened_step_values(active_steps, "subspace_retention")
    row: dict[str, object] = {
        "condition": condition,
        "mode": report["config"]["mode"],
        "proposal": report["config"]["proposal"],
        "normal_estimator": report["config"]["normal_estimator"],
        "target_relative_correction_norm": report["config"].get(
            "target_relative_correction_norm"
        ),
        "latency_seconds": report["latency_seconds"],
        "mean_x0_consistency_norm": mean(consistency) if consistency else None,
        "max_x0_consistency_norm": max(consistency) if consistency else None,
        "mean_correction_relative_norm": mean(corrections) if corrections else None,
        "mean_projection_retention": mean(retention) if retention else None,
    }
    if metrics is not None:
        row.update(
            dino_diversity=metrics["dino_image_diversity"]["mean"],
            clip_image_diversity=metrics["clip_image_diversity"]["mean"],
            clip_text_alignment=metrics["clip_text_image_alignment"]["mean"],
        )
    rows.append(row)

native = next((row for row in rows if row["mode"] == "native"), None)
if native is not None:
    for row in rows:
        for key in (
            "mean_x0_consistency_norm",
            "dino_diversity",
            "clip_image_diversity",
            "clip_text_alignment",
        ):
            baseline = native.get(key)
            value = row.get(key)
            row[f"{key}_delta_vs_native"] = (
                float(value) - float(baseline)
                if value is not None and baseline is not None
                else None
            )

json_path = run_dir / "sampler_ablation_summary.json"
csv_path = run_dir / "sampler_ablation_summary.csv"
json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
fieldnames = list(dict.fromkeys(key for row in rows for key in row))
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
print(json_path)
print(csv_path)
