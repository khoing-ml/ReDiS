from __future__ import annotations

import json
from pathlib import Path

import psutil
import torch
import yaml

from redis.utils.hardware import select_colab_profile


if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: Colab GPU is disabled. Select Runtime > Change runtime type > GPU."
    )

gpu = torch.cuda.get_device_name(0)
vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
ram_gib = psutil.virtual_memory().total / 1024**3
bf16 = torch.cuda.is_bf16_supported()
dtype = "bfloat16" if bf16 else "float16"

profile, memory_mode, quantization, resolution = select_colab_profile(vram_gib, ram_gib)

model = {
    "id": "black-forest-labs/FLUX.2-klein-4B",
    "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
    "dtype": dtype,
    "memory_mode": memory_mode,
    "quantization": quantization,
    "offload_dir": ".offload/flux2-klein-4b",
    "local_files_only": False,
}
if memory_mode == "auto":
    model["max_gpu_memory"] = f"{max(4, int(vram_gib) - 1)}GiB"
    model["max_cpu_memory"] = f"{max(5, int(ram_gib) - 6)}GiB"


def make_config(
    seeds: list[int],
    *,
    capture_sites: dict[str, list[int]] | None = None,
) -> dict[str, object]:
    return {
        "model": dict(model),
        "generation": {
            "prompt": "A red ceramic teapot on a wooden table, soft studio light",
            "seeds": seeds,
            "height": resolution,
            "width": resolution,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "max_sequence_length": 128,
        },
        "capture": {
            "sites": capture_sites or {"transformer_blocks": [0]},
            "clone_to_cpu": True,
        },
        "intervention": {
            "sites": {"transformer_blocks": [0]},
            "timestep_ids": [0],
            "gamma": 0.3,
            "sigma": 0.1,
            "beta": 0.5,
            "rms_match": True,
            "max_relative_correction_norm": 0.25,
        },
        "projection": {"rank": 256, "seed": 0},
    }


def make_screening_config() -> dict[str, object]:
    config = make_config(list(range(8)))
    config["generation"]["prompt"] = "Decisive screening uses screening.prompt_manifest"
    config["intervention"].update(
        {"gamma": 1.0, "sigma": 1.0, "rms_match": False}
    )
    config["screening"] = {
        "prompt_manifest": "configs/prompts_screening_8.yaml",
        "target_relative_norms": [0.005, 0.010, 0.020],
        "methods": [
            "identity",
            "gaussian_noise",
            "residual_amplification",
            "full_hidden_isotropization",
            "low_frequency_isotropization",
            "token_pooled_isotropization",
        ],
        "sites": {
            "A": {
                "view": "spatial_low_frequency",
                "family": "single_transformer_blocks",
                "layer_id": 19,
                "timestep_id": 0,
            },
            "B": {
                "view": "spatial_low_frequency",
                "family": "single_transformer_blocks",
                "layer_id": 4,
                "timestep_id": 2,
            },
            "C": {
                "view": "token_pooled",
                "family": "single_transformer_blocks",
                "layer_id": 14,
                "timestep_id": 3,
            },
        },
    }
    return config


def make_sampler_config() -> dict[str, object]:
    config = make_config([0, 1])
    config["sampler"] = {
        "mode": "non_increasing",
        "proposal": "velocity_difference",
        "reliability_field": "x0_consistency",
        "normal_estimator": "vjp",
        "strength": 0.2,
        "tangent_strength": 1.0,
        "history_size": 2,
        "start_step": 1,
        "end_step": None,
        "target_relative_correction_norm": None,
        "max_relative_correction_norm": 0.25,
        "random_seed": 0,
        "eps": 1e-8,
        "capture_trajectory": True,
        "finite_consistency_check": True,
        "trust_region": False,
        "trust_region_tolerance": 0.0,
        "trust_region_shrink_factor": 0.5,
        "trust_region_max_shrinks": 4,
        "strength_schedule": None,
    }
    config["sampler_ablation"] = {
        "conditions": [
            {"name": "native", "mode": "native"},
            {
                "name": "velocity_naive",
                "mode": "naive",
                "proposal": "velocity_difference",
                "target_relative_correction_norm": 0.05,
            },
            {
                "name": "velocity_subspace",
                "mode": "subspace",
                "proposal": "velocity_difference",
                "target_relative_correction_norm": 0.05,
            },
            {
                "name": "x0_naive",
                "mode": "naive",
                "proposal": "x0_difference",
                "target_relative_correction_norm": 0.05,
            },
            {
                "name": "x0_subspace",
                "mode": "subspace",
                "proposal": "x0_difference",
                "target_relative_correction_norm": 0.05,
            },
            {
                "name": "velocity_non_increasing_vjp",
                "mode": "non_increasing",
                "proposal": "velocity_difference",
                "normal_estimator": "vjp",
                "target_relative_correction_norm": 0.05,
            },
            {
                "name": "velocity_tangent_residual_proxy_legacy",
                "mode": "tangent",
                "proposal": "velocity_difference",
                "normal_estimator": "residual_proxy",
                "target_relative_correction_norm": 0.05,
            },
            {
                "name": "x0_non_increasing_vjp",
                "mode": "non_increasing",
                "proposal": "x0_difference",
                "normal_estimator": "vjp",
                "target_relative_correction_norm": 0.05,
            },
            {
                "name": "random_ambient",
                "mode": "naive",
                "proposal": "random_ambient",
                "target_relative_correction_norm": 0.05,
            },
            {
                "name": "random_subspace",
                "mode": "subspace",
                "proposal": "random_ambient",
                "target_relative_correction_norm": 0.05,
            },
            {
                "name": "random_non_increasing_vjp",
                "mode": "non_increasing",
                "proposal": "random_ambient",
                "normal_estimator": "vjp",
                "target_relative_correction_norm": 0.05,
            },
        ]
    }
    return config


output_dir = Path(".runtime")
output_dir.mkdir(exist_ok=True)
diagnosis_group_size = 16 if vram_gib >= 30 and ram_gib >= 40 else 8
diagnosis_sites = {
    "transformer_blocks": [0, 2, 4],
    "single_transformer_blocks": [0, 4, 9, 14, 19],
}
configs = {
    "colab_baseline.yaml": make_config([0]),
    "colab_capture.yaml": make_config(
        list(range(diagnosis_group_size)), capture_sites=diagnosis_sites
    ),
    "colab_intervention.yaml": make_config([0, 1, 2, 3]),
    "colab_screening.yaml": make_screening_config(),
    "colab_sampler.yaml": make_sampler_config(),
}
for filename, config in configs.items():
    (output_dir / filename).write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

report = {
    "gpu": gpu,
    "vram_gib": round(vram_gib, 2),
    "host_ram_gib": round(ram_gib, 2),
    "bf16_supported": bf16,
    "dtype": dtype,
    "profile": profile,
    "resolution": resolution,
    "diagnosis_group_size": diagnosis_group_size,
    "diagnosis_sites": diagnosis_sites,
    "configs": [str(output_dir / name) for name in configs],
}
(output_dir / "colab_hardware.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
