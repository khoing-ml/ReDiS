from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path

import torch


def memory_gib() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / 1024**2
    return None


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


report = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "diffusers": version("diffusers"),
    "transformers": version("transformers"),
    "accelerate": version("accelerate"),
    "bitsandbytes": version("bitsandbytes"),
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "host_ram_gib": memory_gib(),
    "disk_free_gib": shutil.disk_usage(".").free / 1024**3,
}
if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)
    report.update(
        gpu=properties.name,
        gpu_vram_gib=properties.total_memory / 1024**3,
        bf16_supported=torch.cuda.is_bf16_supported(),
    )

print(json.dumps(report, indent=2))

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable; FLUX.2 Klein tests require an NVIDIA GPU.")
if report["bitsandbytes"] is not None:
    import bitsandbytes as bnb

    sample = torch.randn(2, 8, device="cuda", dtype=torch.float16)
    layer = bnb.nn.Linear8bitLt(8, 8, has_fp16_weights=False).cuda()
    output = layer(sample)
    if not torch.isfinite(output).all():
        raise SystemExit("ERROR: bitsandbytes CUDA smoke test produced non-finite output")
    print("bitsandbytes_cuda_smoke=ok")

# Import the real pipeline, not just the top-level diffusers package. This
# catches incompatible optional packages such as a stale Colab torchao build.
from diffusers import Flux2KleinPipeline

print(f"flux2_pipeline_import={Flux2KleinPipeline.__name__}:ok")
if report.get("gpu_vram_gib", 0) < 12:
    print(
        "WARNING: less than 12 GiB VRAM; use configs/flux2_klein_4b_low_vram.yaml. "
        "Generation may still fail because host RAM is limited."
    )

if subprocess.run(["hf", "auth", "whoami"], capture_output=True).returncode != 0:
    print("WARNING: Hugging Face login not found. Run: hf auth login")
