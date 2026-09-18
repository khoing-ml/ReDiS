from __future__ import annotations

import argparse
import json

import torch

from redis.analysis import representation_diagnostics
from redis.config import load_config
from redis.hooks import CaptureController, patch_flux2_blocks
from redis.methods import make_random_basis
from redis.models.flux2_klein import load_pipeline, make_generators
from redis.utils.run import make_run_dir, write_run_metadata


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
config = load_config(args.config)
generation = config["generation"]
seeds = [int(seed) for seed in generation["seeds"]]
if len(seeds) < 2:
    raise SystemExit("ERROR: capture spectrum requires at least two seeds in generation.seeds")

run_dir = make_run_dir("capture")
write_run_metadata(run_dir, config)
pipe = load_pipeline(config["model"])
capture = config.get("capture", {})
controller = CaptureController(clone_to_cpu=bool(capture.get("clone_to_cpu", True)))
sites = capture.get("sites")
if sites is None:
    sites = {
        str(capture.get("block_family", "transformer_blocks")): [
            int(value) for value in capture.get("layer_ids", [0])
        ]
    }
if not isinstance(sites, dict) or not sites:
    raise SystemExit("ERROR: capture.sites must map block families to layer-id lists")
patched_sites: dict[str, list[int]] = {}
for family, layer_values in sites.items():
    layers = [int(value) for value in layer_values]
    patched_sites[str(family)] = patch_flux2_blocks(
        pipe.transformer,
        controller,
        family=str(family),
        layer_ids=layers,
    )

# One repeated prompt with multiple generators: this preserves the group constraint.
pipe(
    prompt=[str(generation["prompt"])] * len(seeds),
    height=int(generation["height"]),
    width=int(generation["width"]),
    num_inference_steps=int(generation["num_inference_steps"]),
    guidance_scale=float(generation["guidance_scale"]),
    max_sequence_length=int(generation.get("max_sequence_length", 128)),
    generator=make_generators(seeds),
    output_type="latent",
)

projection = config.get("projection", {})
projection_rank = int(projection.get("rank", 256))
projection_seed = int(projection.get("seed", 0))
basis_cache: dict[tuple[int, int], torch.Tensor] = {}
report: dict[str, object] = {
    "seeds": seeds,
    "group_size": len(seeds),
    "patched_sites": patched_sites,
    "projection_rank": projection_rank,
    "projection_seed": projection_seed,
    "token_scope": "image_only",
    "sites": {},
}
summary: list[dict[str, object]] = []
for (block_family, layer_id), tensors in controller.activations.items():
    site = f"{block_family}.{layer_id}"
    report["sites"][site] = []
    for invocation, hidden in enumerate(tensors):
        rank = min(projection_rank, hidden.shape[-1])
        basis_key = (hidden.shape[-1], rank)
        if basis_key not in basis_cache:
            basis_cache[basis_key] = make_random_basis(
                hidden.shape[-1], rank, device=hidden.device, seed=projection_seed
            )
        metrics = representation_diagnostics(
            hidden, projection_basis=basis_cache[basis_key]
        )
        metrics["invocation"] = invocation
        report["sites"][site].append(metrics)
        for view_name, view_metrics in metrics["views"].items():
            if view_metrics is None:
                continue
            raw_spectrum = view_metrics["raw_spectrum"]
            normalized_spectrum = view_metrics["row_normalized_spectrum"]
            summary.append(
                {
                    "site": site,
                    "timestep_id": invocation,
                    "view": view_name,
                    "raw_effective_rank": raw_spectrum["effective_rank"],
                    "raw_normalized_effective_rank": raw_spectrum[
                        "normalized_effective_rank"
                    ],
                    "raw_top1_ratio": raw_spectrum["top1_ratio"],
                    "row_normalized_effective_rank": normalized_spectrum[
                        "effective_rank"
                    ],
                    "row_normalized_normalized_effective_rank": (
                        normalized_spectrum["normalized_effective_rank"]
                    ),
                    "row_normalized_top1_ratio": normalized_spectrum[
                        "top1_ratio"
                    ],
                    "view_rms": view_metrics["view_rms"],
                    "view_energy_over_full_residual": view_metrics[
                        "energy_over_full_residual"
                    ],
                }
            )

(run_dir / "spectra.json").write_text(json.dumps(report, indent=2) + "\n")
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
most_concentrated = sorted(
    summary, key=lambda row: float(row["raw_normalized_effective_rank"])
)[:12]
print(json.dumps({"most_concentrated": most_concentrated}, indent=2))
print(run_dir)
