import pytest
import torch

from redis.metrics import cosine_distance_summary


def test_cosine_distance_summary_uses_unique_pairs():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    result = cosine_distance_summary(features)
    assert result["pair_count"] == 3
    assert result["mean"] == pytest.approx(4 / 3)


def test_cosine_distance_requires_two_features():
    with pytest.raises(ValueError):
        cosine_distance_summary(torch.ones(1, 4))
