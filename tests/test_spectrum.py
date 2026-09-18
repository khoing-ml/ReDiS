import math

import pytest
import torch

from redis.analysis import seed_spectrum


def test_isotropic_seed_spectrum_has_full_centered_rank():
    # Four points forming a regular tetrahedron in the centered seed subspace.
    hidden = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
    ).reshape(4, 1, 3)
    result = seed_spectrum(hidden)
    assert result["effective_rank"] == pytest.approx(3.0, rel=1e-5)
    assert result["normalized_effective_rank"] == pytest.approx(1.0, rel=1e-5)
    assert result["top1_ratio"] == pytest.approx(1 / 3, rel=1e-5)


def test_degenerate_spectrum_is_zero():
    result = seed_spectrum(torch.ones(4, 2, 3))
    assert result["effective_rank"] == 0.0
    assert result["normalized_effective_rank"] == 0.0


def test_spectrum_rejects_single_seed():
    with pytest.raises(ValueError):
        seed_spectrum(torch.randn(1, 2, 3))

