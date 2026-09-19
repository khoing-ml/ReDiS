import pytest

from redis.metrics import aggregate_sampler_steps


def test_aggregate_sampler_steps_reports_rejection_and_first_order_failure():
    steps = [
        {"active": False},
        {
            "active": True,
            "correction_relative_norm": [0.1, 0.2],
            "state_correction_relative_native_update": [0.1, 0.2],
            "constraint_retention": [0.5, 1.0],
            "consistency_before": [1.0, 1.0],
            "consistency_after": [1.2, 0.8],
            "consistency_ratio": [1.2, 0.8],
            "directional_derivative_post": [0.0, -0.1],
            "trust_region_rejected": [True, False],
            "trust_region_shrinks": [4, 0],
        },
    ]
    result = aggregate_sampler_steps(steps)
    assert result["active_step_samples"] == 2
    assert result["trust_region_rejection_rate"] == pytest.approx(0.5)
    assert result["first_order_failure_rate"] == pytest.approx(0.5)
    assert result["mean_constraint_retention"] == pytest.approx(0.75)
