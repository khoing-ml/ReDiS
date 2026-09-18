from __future__ import annotations

import argparse
import json

import torch

from redis.config import load_config
from redis.hooks import IdentityController, patch_flux2_blocks, restore_flux2_blocks
from redis.models.flux2_klein import load_pipeline, make_generators
from redis.utils.run import make_run_dir, write_run_metadata


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--atol", type=float, default=0.0)
args = parser.parse_args()
config = load_config(args.config)
generation = config["generation"]
seed = int(generation["seeds"][0])
run_dir = make_run_dir("wrapper_identity")
write_run_metadata(run_dir, config)

pipe = load_pipeline(config["model"])
call_kwargs = {
    "prompt": str(generation["prompt"]),
    "height": int(generation["height"]),
    "width": int(generation["width"]),
    "num_inference_steps": int(generation["num_inference_steps"]),
    "guidance_scale": float(generation["guidance_scale"]),
    "max_sequence_length": int(generation.get("max_sequence_length", 128)),
    "output_type": "latent",
}

baseline = pipe(**call_kwargs, generator=make_generators([seed])[0]).images.detach().float().cpu()
capture = config.get("capture", {})
family = str(capture.get("block_family", "transformer_blocks"))
layers = [int(value) for value in capture.get("layer_ids", [0])]
patched = patch_flux2_blocks(pipe.transformer, IdentityController(), family=family, layer_ids=layers)
wrapped = pipe(**call_kwargs, generator=make_generators([seed])[0]).images.detach().float().cpu()
restored = restore_flux2_blocks(pipe.transformer)

absolute = (baseline - wrapped).abs()
report = {
    "family": family,
    "patched_layers": patched,
    "restored": restored,
    "shape": list(baseline.shape),
    "max_abs_error": absolute.max().item(),
    "mean_abs_error": absolute.mean().item(),
    "atol": args.atol,
    "passed": bool(torch.allclose(baseline, wrapped, atol=args.atol, rtol=0.0)),
}
(run_dir / "identity.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
print(run_dir)
if not report["passed"]:
    raise SystemExit(1)

