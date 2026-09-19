from PIL import Image
import pytest
from types import SimpleNamespace
import torch

from redis.metrics import RewardEvaluator, paired_score_summary
from redis.metrics.rewards import pooled_feature_tensor


def test_reward_evaluator_supports_injected_backends():
    evaluator = RewardEvaluator(
        {"fake": lambda images, prompts: [len(prompt) for prompt in prompts]}
    )
    images = [Image.new("RGB", (1, 1)), Image.new("RGB", (1, 1))]
    assert evaluator.evaluate(images, ["a", "abcd"], ["fake"]) == {
        "fake": [1.0, 4.0]
    }


def test_paired_score_summary_uses_matched_deltas():
    result = paired_score_summary(
        [2.0, 2.0, 5.0], [1.0, 3.0, 2.0], bootstrap_samples=100, seed=7
    )
    assert result["mean_paired_improvement"] == pytest.approx(1.0)
    assert result["fraction_improved"] == pytest.approx(2 / 3)
    assert result["fraction_degraded"] == pytest.approx(1 / 3)


def test_pooled_feature_tensor_supports_transformers_4_tensor():
    features = torch.randn(2, 4)
    assert pooled_feature_tensor(features) is features


def test_pooled_feature_tensor_supports_transformers_5_model_output():
    features = torch.randn(2, 4)
    output = SimpleNamespace(
        last_hidden_state=torch.randn(2, 3, 4), pooler_output=features
    )
    assert pooled_feature_tensor(output) is features
