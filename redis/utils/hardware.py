from __future__ import annotations


def select_colab_profile(vram_gib: float, ram_gib: float) -> tuple[str, str, str, int]:
    """Return profile, memory mode, quantization, and smoke-test resolution."""
    if vram_gib >= 20:
        return "full_cuda", "full_cuda", "none", 512
    if vram_gib >= 13 and ram_gib >= 24:
        return "cpu_offload", "model_cpu_offload", "none", 512
    return "4bit_auto_offload", "auto", "bitsandbytes_4bit", 256

