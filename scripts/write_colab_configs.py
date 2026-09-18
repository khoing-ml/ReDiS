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


def make_config(seeds: list[int]) -> dict[str, object]:
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
            "block_family": "transformer_blocks",
            "layer_ids": [0],
            "clone_to_cpu": True,
        },
        "intervention": {
            "timestep_ids": [0],
            "gamma": 0.3,
            "sigma": 0.1,
            "beta": 0.5,
            "rms_match": True,
            "max_relative_correction_norm": 0.25,
        },
        "projection": {"rank": 64, "seed": 0},
    }


output_dir = Path(".runtime")
output_dir.mkdir(exist_ok=True)
configs = {
    "colab_baseline.yaml": make_config([0]),
    "colab_capture.yaml": make_config([0, 1]),
    "colab_intervention.yaml": make_config([0, 1, 2, 3]),
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
    "configs": [str(output_dir / name) for name in configs],
}
(output_dir / "colab_hardware.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
