from redis.utils.hardware import select_colab_profile


def test_a100_or_l4_uses_full_cuda():
    assert select_colab_profile(40, 80) == ("full_cuda", "full_cuda", "none", 512)
    assert select_colab_profile(24, 50) == ("full_cuda", "full_cuda", "none", 512)


def test_t4_with_high_ram_uses_cpu_offload():
    assert select_colab_profile(15, 50) == (
        "cpu_offload",
        "model_cpu_offload",
        "none",
        512,
    )


def test_t4_with_low_ram_uses_quantization():
    assert select_colab_profile(15, 12) == (
        "4bit_auto_offload",
        "auto",
        "bitsandbytes_4bit",
        256,
    )


def test_small_gpu_uses_quantization_even_with_high_ram():
    assert select_colab_profile(12, 80)[0] == "4bit_auto_offload"
