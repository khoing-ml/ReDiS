from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

from redis.metrics import cosine_distance_summary
from redis.metrics.rewards import pooled_feature_tensor


parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument("--dino-model", default="facebook/dinov2-small")
parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
args = parser.parse_args()
run_dir = Path(args.run_dir)
files = sorted(run_dir.glob("seed_*.png"))
config_path = next(
    (
        parent / "resolved_config.yaml"
        for parent in (run_dir, *run_dir.parents)
        if (parent / "resolved_config.yaml").exists()
    ),
    None,
)
if config_path is None:
    raise SystemExit(f"ERROR: no resolved_config.yaml found at or above {run_dir}")
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
groups: list[tuple[str, str, list[Path]]] = []
if len(files) >= 2:
    groups.append(("inline_0000", str(config["generation"]["prompt"]), files))
else:
    report_path = run_dir.parents[1] / f"{run_dir.name}.json"
    if not report_path.exists():
        raise SystemExit(
            f"ERROR: need seed images directly in {run_dir} or a condition report at {report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for prompt_run in report.get("prompt_runs", []):
        prompt_dir = run_dir / f"prompt_{int(prompt_run['prompt_index']):04d}"
        prompt_files = sorted(prompt_dir.glob("seed_*.png"))
        if len(prompt_files) < 2:
            raise SystemExit(f"ERROR: need at least two seed images in {prompt_dir}")
        groups.append(
            (str(prompt_run["prompt_id"]), str(prompt_run["prompt"]), prompt_files)
        )
if not groups:
    raise SystemExit(f"ERROR: no prompt image groups found in {run_dir}")
device = "cuda" if torch.cuda.is_available() else "cpu"

dino_processor = AutoImageProcessor.from_pretrained(args.dino_model)
dino = AutoModel.from_pretrained(args.dino_model).to(device).eval()
dino_summaries = []
for prompt_id, prompt, prompt_files in groups:
    images = [Image.open(path).convert("RGB") for path in prompt_files]
    with torch.inference_mode():
        inputs = dino_processor(images=images, return_tensors="pt").to(device)
        output = dino(**inputs)
        features = output.last_hidden_state[:, 0].float().cpu()
    dino_summaries.append(
        {"prompt_id": prompt_id, "prompt": prompt, **cosine_distance_summary(features)}
    )
    for image in images:
        image.close()
del dino, inputs, output
if torch.cuda.is_available():
    torch.cuda.empty_cache()

clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
clip = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
clip_summaries = []
alignment_summaries = []
for prompt_id, prompt, prompt_files in groups:
    images = [Image.open(path).convert("RGB") for path in prompt_files]
    with torch.inference_mode():
        image_inputs = clip_processor(images=images, return_tensors="pt").to(device)
        clip_image = pooled_feature_tensor(
            clip.get_image_features(**image_inputs)
        ).float()
        text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(device)
        clip_text = pooled_feature_tensor(
            clip.get_text_features(**text_inputs)
        ).float()
        alignment = (
            F.normalize(clip_image, dim=-1) @ F.normalize(clip_text, dim=-1).T
        ).squeeze(1)
    clip_summaries.append(
        {
            "prompt_id": prompt_id,
            "prompt": prompt,
            **cosine_distance_summary(clip_image.cpu()),
        }
    )
    alignment_summaries.append(
        {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "mean": alignment.mean().item(),
            "median": alignment.median().item(),
            "per_image": alignment.cpu().tolist(),
        }
    )
    for image in images:
        image.close()

report = {
    "run_dir": str(run_dir),
    "prompt_count": len(groups),
    "dino_model": args.dino_model,
    "clip_model": args.clip_model,
    "dino_image_diversity": {
        "mean": sum(item["mean"] for item in dino_summaries) / len(dino_summaries),
        "per_prompt": dino_summaries,
    },
    "clip_image_diversity": {
        "mean": sum(item["mean"] for item in clip_summaries) / len(clip_summaries),
        "per_prompt": clip_summaries,
    },
    "clip_text_image_alignment": {
        "mean": sum(item["mean"] for item in alignment_summaries)
        / len(alignment_summaries),
        "per_prompt": alignment_summaries,
    },
}
output_path = run_dir / "feature_metrics.json"
output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
print(output_path)
