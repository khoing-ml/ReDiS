from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _quantization_config(kind: str, dtype: torch.dtype):
    if kind == "none":
        return None
    if kind != "bitsandbytes_4bit":
        raise ValueError(f"Unsupported quantization: {kind}")

    try:
        from diffusers.quantizers import PipelineQuantizationConfig
    except ImportError as exc:
        raise RuntimeError(
            "This diffusers build lacks PipelineQuantizationConfig; run bash/00_check_environment.sh"
        ) from exc

    return PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": dtype,
        },
        components_to_quantize=["transformer", "text_encoder"],
    )


def load_pipeline(model_config: dict[str, Any]):
    from diffusers import Flux2KleinPipeline

    dtype_name = str(model_config.get("dtype", "bfloat16"))
    if dtype_name not in DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    dtype = DTYPES[dtype_name]
    memory_mode = str(model_config.get("memory_mode", "model_cpu_offload"))
    quantization = str(model_config.get("quantization", "none"))
    offload_dir = Path(model_config.get("offload_dir", ".offload/flux2-klein-4b"))
    offload_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {
        "revision": model_config.get("revision"),
        "torch_dtype": dtype,
        "local_files_only": bool(model_config.get("local_files_only", False)),
        "quantization_config": _quantization_config(quantization, dtype),
    }
    if memory_mode == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("memory_mode=auto requires CUDA")
        kwargs.update(
            device_map="auto",
            max_memory={
                0: str(model_config.get("max_gpu_memory", "5GiB")),
                "cpu": str(model_config.get("max_cpu_memory", "5GiB")),
            },
            offload_folder=str(offload_dir),
            offload_state_dict=True,
        )

    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    pipe = Flux2KleinPipeline.from_pretrained(model_config["id"], **kwargs)

    if memory_mode == "full_cuda":
        pipe.to("cuda")
    elif memory_mode == "model_cpu_offload":
        pipe.enable_model_cpu_offload()
    elif memory_mode == "sequential_cpu_offload":
        pipe.enable_sequential_cpu_offload()
    elif memory_mode != "auto":
        raise ValueError(f"Unsupported memory_mode: {memory_mode}")

    pipe.set_progress_bar_config(disable=False)
    return pipe


def make_generators(seeds: list[int], device: str = "cpu") -> list[torch.Generator]:
    return [torch.Generator(device=device).manual_seed(seed) for seed in seeds]

