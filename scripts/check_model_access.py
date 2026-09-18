from __future__ import annotations

import argparse
import json

from huggingface_hub import HfApi, hf_hub_download

from redis.config import load_config


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
config = load_config(args.config)
model_id = config["model"]["id"]
revision = config["model"].get("revision")

info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
siblings = info.siblings or []
total = sum((file.size or 0) for file in siblings)
required = {
    "model_index.json",
    "transformer/config.json",
    "text_encoder/config.json",
    "vae/config.json",
}
names = {file.rfilename for file in siblings}
missing = sorted(required - names)
if missing:
    raise SystemExit(f"ERROR: checkpoint is missing expected files: {missing}")

model_index_path = hf_hub_download(model_id, "model_index.json", revision=info.sha)
transformer_config_path = hf_hub_download(
    model_id, "transformer/config.json", revision=info.sha
)
with open(model_index_path, encoding="utf-8") as handle:
    model_index = json.load(handle)
with open(transformer_config_path, encoding="utf-8") as handle:
    transformer_config = json.load(handle)
if model_index.get("_class_name") != "Flux2KleinPipeline":
    raise SystemExit(f"ERROR: unexpected pipeline class: {model_index.get('_class_name')}")
if not model_index.get("is_distilled"):
    raise SystemExit("ERROR: configured checkpoint is not marked as distilled")

print(f"model={model_id}")
print(f"revision={info.sha}")
print(f"files={len(siblings)}")
print(f"repository_size_gib={total / 1024**3:.2f}")
print(f"pipeline_class={model_index['_class_name']}")
print(f"dual_stream_layers={transformer_config['num_layers']}")
print(f"single_stream_layers={transformer_config['num_single_layers']}")
print("access=ok")
