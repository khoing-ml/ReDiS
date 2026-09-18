import torch

from redis.models.flux2_klein import DTYPES, make_generators


def test_flux2_loader_module_imports():
    assert DTYPES["bfloat16"] is torch.bfloat16


def test_generators_are_seeded_independently_and_reproducibly():
    first = make_generators([3, 7])
    second = make_generators([3, 7])
    first_values = [torch.randn(4, generator=generator) for generator in first]
    second_values = [torch.randn(4, generator=generator) for generator in second]
    assert torch.equal(first_values[0], second_values[0])
    assert torch.equal(first_values[1], second_values[1])
    assert not torch.equal(first_values[0], first_values[1])
