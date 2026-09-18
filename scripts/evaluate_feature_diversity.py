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


parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument("--dino-model", default="facebook/dinov2-small")
parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
args = parser.parse_args()
run_dir = Path(args.run_dir)
files = sorted(run_dir.glob("seed_*.png"))
if len(files) < 2:
    raise SystemExit(f"ERROR: need at least two seed_*.png files in {run_dir}")
config_path = run_dir / "resolved_config.yaml"
if not config_path.exists():
    raise SystemExit(f"ERROR: missing {config_path}")
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
prompt = str(config["generation"]["prompt"])
images = [Image.open(path).convert("RGB") for path in files]
device = "cuda" if torch.cuda.is_available() else "cpu"

dino_processor = AutoImageProcessor.from_pretrained(args.dino_model)
dino = AutoModel.from_pretrained(args.dino_model).to(device).eval()
with torch.inference_mode():
    inputs = dino_processor(images=images, return_tensors="pt").to(device)
    output = dino(**inputs)
    dino_features = output.last_hidden_state[:, 0].float().cpu()
del dino, inputs, output
if torch.cuda.is_available():
    torch.cuda.empty_cache()

clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
clip = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
with torch.inference_mode():
    image_inputs = clip_processor(images=images, return_tensors="pt").to(device)
    clip_image = clip.get_image_features(**image_inputs).float()
    text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    clip_text = clip.get_text_features(**text_inputs).float()
    alignment = (
        F.normalize(clip_image, dim=-1) @ F.normalize(clip_text, dim=-1).T
    ).squeeze(1)

report = {
    "run_dir": str(run_dir),
    "files": [path.name for path in files],
    "prompt": prompt,
    "dino_model": args.dino_model,
    "clip_model": args.clip_model,
    "dino_image_diversity": cosine_distance_summary(dino_features),
    "clip_image_diversity": cosine_distance_summary(clip_image.cpu()),
    "clip_text_image_alignment": {
        "mean": alignment.mean().item(),
        "median": alignment.median().item(),
        "per_image": alignment.cpu().tolist(),
    },
}
output_path = run_dir / "feature_metrics.json"
output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
print(output_path)
