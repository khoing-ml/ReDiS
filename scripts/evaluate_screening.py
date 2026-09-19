from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import types
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from redis.metrics import cosine_distance_summary
from redis.metrics.rewards import pooled_feature_tensor


parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument(
    "--metrics",
    nargs="+",
    default=["dino", "dreamsim", "lpips", "clip", "hpsv2"],
    choices=("dino", "dreamsim", "lpips", "clip", "hpsv2"),
)
parser.add_argument("--dino-model", default="facebook/dinov2-small")
parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
args = parser.parse_args()

run_dir = Path(args.run_dir)
records_path = run_dir / "samples.jsonl"
if not records_path.exists():
    raise SystemExit(f"ERROR: missing {records_path}")
records = [
    json.loads(line)
    for line in records_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
for record in records:
    groups[(str(record["condition"]), int(record["prompt_id"]))].append(record)
for key, group in groups.items():
    group.sort(key=lambda item: int(item["seed"]))
    if len(group) != 8:
        raise SystemExit(f"ERROR: {key} has {len(group)} images instead of 8")

rows: dict[tuple[str, int], dict[str, object]] = {}
for key, group in groups.items():
    first = group[0]
    rows[key] = {
        "condition": first["condition"],
        "method": first["method"],
        "target_relative_norm": first["target_relative_norm"],
        "prompt_id": first["prompt_id"],
        "prompt": first["prompt"],
        "image_paths": [str(item["image_path"]) for item in group],
    }

device = "cuda" if torch.cuda.is_available() else "cpu"


def _value_summary(values: list[float]) -> dict[str, object]:
    if not values:
        raise ValueError("Cannot summarize an empty metric list")
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": tensor.mean().item(),
        "median": tensor.median().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "count": len(values),
        "values": values,
    }


def images_for(group: list[dict[str, object]]) -> list[Image.Image]:
    return [
        Image.open(run_dir / str(item["image_path"])).convert("RGB")
        for item in group
    ]


def release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if "dino" in args.metrics:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(args.dino_model)
    model = AutoModel.from_pretrained(args.dino_model).to(device).eval()
    for key, group in groups.items():
        with torch.inference_mode():
            inputs = processor(images=images_for(group), return_tensors="pt").to(device)
            output = model(**inputs)
            features = output.last_hidden_state[:, 0].float().cpu()
        rows[key]["dino_pairwise_distance"] = cosine_distance_summary(features)
    del model, processor
    release()

if "clip" in args.metrics:
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    for key, group in groups.items():
        prompt = str(group[0]["prompt"])
        with torch.inference_mode():
            image_inputs = processor(
                images=images_for(group), return_tensors="pt"
            ).to(device)
            image_features = pooled_feature_tensor(
                model.get_image_features(**image_inputs)
            ).float()
            text_inputs = processor(
                text=[prompt], return_tensors="pt", padding=True
            ).to(device)
            text_features = pooled_feature_tensor(
                model.get_text_features(**text_inputs)
            ).float()
            alignment = (
                F.normalize(image_features, dim=-1)
                @ F.normalize(text_features, dim=-1).T
            ).squeeze(1)
        rows[key]["clip_t"] = {
            "mean": alignment.mean().item(),
            "median": alignment.median().item(),
            "per_image": alignment.cpu().tolist(),
        }
    del model, processor
    release()

if "lpips" in args.metrics:
    try:
        import lpips
        from torchvision.transforms.functional import pil_to_tensor
    except ImportError as exc:
        raise SystemExit(
            "ERROR: LPIPS is missing. Run bash bash/12_setup_output_metrics.sh"
        ) from exc

    model = lpips.LPIPS(net="alex").to(device).eval()
    for key, group in groups.items():
        tensors = [
            pil_to_tensor(image).float().div(127.5).sub(1).to(device)
            for image in images_for(group)
        ]
        values: list[float] = []
        with torch.inference_mode():
            for left, right in combinations(range(len(tensors)), 2):
                values.append(
                    float(model(tensors[left][None], tensors[right][None]).item())
                )
        rows[key]["lpips"] = _value_summary(values)
    del model
    release()

if "dreamsim" in args.metrics:
    try:
        from dreamsim import dreamsim
    except ImportError as exc:
        raise SystemExit(
            "ERROR: DreamSim is missing. Run bash bash/12_setup_output_metrics.sh"
        ) from exc

    model, preprocess = dreamsim(pretrained=True, device=device)
    model = model.to(device).eval()
    for key, group in groups.items():
        processed = [preprocess(image) for image in images_for(group)]
        processed = [item[None] if item.ndim == 3 else item for item in processed]
        batch = torch.cat(processed).to(device)
        with torch.inference_mode():
            embeddings = model.embed(batch).float().cpu()
        rows[key]["dreamsim_pairwise_distance"] = cosine_distance_summary(embeddings)
    del model, preprocess
    release()

if "hpsv2" in args.metrics:
    try:
        # HPSv2 1.2.0 has an unused `from turtle import forward` in its bundled
        # OpenCLIP factory. Headless Colab has no tkinter, so provide only the
        # unused symbol rather than installing a GUI stack.
        if "turtle" not in sys.modules:
            turtle_stub = types.ModuleType("turtle")
            turtle_stub.forward = None
            sys.modules["turtle"] = turtle_stub
        from huggingface_hub import hf_hub_download
        from hpsv2 import img_score as hps_score
        from hpsv2.src.open_clip import get_tokenizer
        from hpsv2.utils import hps_version_map
    except ImportError as exc:
        raise SystemExit(
            "ERROR: HPSv2 is missing. Run bash bash/12_setup_output_metrics.sh"
        ) from exc

    # hpsv2.score reloads its multi-GB checkpoint on every prompt. Load it once
    # and batch the eight same-prompt images instead.
    hps_score.initialize_model()
    model = hps_score.model_dict["model"]
    preprocess = hps_score.model_dict["preprocess_val"]
    checkpoint_path = hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    tokenizer = get_tokenizer("ViT-H-14")
    for key, group in groups.items():
        prompt = str(group[0]["prompt"])
        image_batch = torch.stack(
            [preprocess(image) for image in images_for(group)]
        ).to(device, non_blocking=True)
        text_batch = tokenizer([prompt] * len(group)).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device == "cuda"
        ):
            outputs = model(image_batch, text_batch)
            scores = torch.diagonal(
                outputs["image_features"] @ outputs["text_features"].T
            ).float().cpu().tolist()
        rows[key]["hpsv2"] = _value_summary(scores)
    del model, preprocess, checkpoint, tokenizer
    release()

metric_rows = list(rows.values())
metric_rows.sort(key=lambda row: (str(row["condition"]), int(row["prompt_id"])))
(run_dir / "output_metrics.json").write_text(
    json.dumps(metric_rows, indent=2) + "\n", encoding="utf-8"
)

metric_fields = {
    "dino": "dino_pairwise_distance",
    "dreamsim": "dreamsim_pairwise_distance",
    "lpips": "lpips",
    "clip_t": "clip_t",
    "hpsv2": "hpsv2",
}
condition_rows: list[dict[str, object]] = []
by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
for row in metric_rows:
    by_condition[str(row["condition"])].append(row)
for condition, condition_group in by_condition.items():
    aggregate: dict[str, object] = {
        "condition": condition,
        "method": condition_group[0]["method"],
        "target_relative_norm": condition_group[0]["target_relative_norm"],
        "prompt_count": len(condition_group),
    }
    for label, field in metric_fields.items():
        values = [float(row[field]["mean"]) for row in condition_group if field in row]
        aggregate[label] = sum(values) / len(values) if values else None
    condition_rows.append(aggregate)

baseline = next(row for row in condition_rows if row["condition"] == "identity")
for row in condition_rows:
    for metric in metric_fields:
        value = row[metric]
        base_value = baseline[metric]
        row[f"delta_{metric}"] = (
            float(value) - float(base_value)
            if value is not None and base_value is not None
            else None
        )
condition_rows.sort(
    key=lambda row: (
        float("-inf") if row["delta_dino"] is None else float(row["delta_dino"])
    ),
    reverse=True,
)

csv_path = run_dir / "diversity_fidelity_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(condition_rows[0]))
    writer.writeheader()
    writer.writerows(condition_rows)
(run_dir / "diversity_fidelity_summary.json").write_text(
    json.dumps(condition_rows, indent=2) + "\n", encoding="utf-8"
)

if "dino" in args.metrics and "clip" in args.metrics:
    import matplotlib.pyplot as plt

    has_hps = "hpsv2" in args.metrics
    figure, axes = plt.subplots(1, 2 if has_hps else 1, figsize=(12 if has_hps else 6, 5))
    if not isinstance(axes, (list, tuple)) and not hasattr(axes, "flat"):
        axes = [axes]
    else:
        axes = list(axes.flat)
    fidelity_metrics = ["clip_t"] + (["hpsv2"] if has_hps else [])
    for axis, fidelity in zip(axes, fidelity_metrics, strict=True):
        for row in condition_rows:
            if row["condition"] == "identity" or row[f"delta_{fidelity}"] is None:
                continue
            axis.scatter(row[f"delta_{fidelity}"], row["delta_dino"])
            axis.annotate(
                str(row["condition"]),
                (row[f"delta_{fidelity}"], row["delta_dino"]),
                fontsize=6,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel(f"Delta {fidelity.upper()}")
        axis.set_ylabel("Delta DINO diversity")
        axis.set_title(f"Diversity vs fidelity ({fidelity})")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(run_dir / "dino_diversity_vs_fidelity.png", dpi=180)

print(json.dumps(condition_rows, indent=2))
print(csv_path)
