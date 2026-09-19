from PIL import Image
import pytest

from redis.metrics import RewardEvaluator, paired_score_summary


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
