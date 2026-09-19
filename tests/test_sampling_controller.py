import json
from types import SimpleNamespace

import pytest
import torch

from redis.sampling import SamplingRefinementConfig, SamplingRefinementController


class FakeScheduler:
    def __init__(self):
        self.sigmas = torch.tensor([1.0, 0.5, 0.0])
        self._step_index = None
        self.config = SimpleNamespace(stochastic_sampling=False)

    @property
    def step_index(self):
        return self._step_index

    def _init_step_index(self, timestep):
        self._step_index = 0

    def step(self, model_output, timestep, sample, return_dict=False, **kwargs):
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        self._step_index += 1
        result = sample + (sigma_next - sigma) * model_output
        return (result,) if not return_dict else SimpleNamespace(prev_sample=result)


def run_two_steps(config):
    scheduler = FakeScheduler()
    controller = SamplingRefinementController(scheduler, config)
    sample = torch.tensor([[[2.0, 1.0]]])
    first_velocity = torch.tensor([[[1.0, 0.0]]])
    second_velocity = torch.tensor([[[0.0, 1.0]]])
    original_step = scheduler.step
    with controller:
        first = scheduler.step(first_velocity, torch.tensor(1000.0), sample, return_dict=False)[0]
        second = scheduler.step(second_velocity, torch.tensor(500.0), first, return_dict=False)[0]
    return controller, second, original_step, scheduler.step


def test_native_mode_is_exact_identity_and_restores_scheduler():
    controller, result, original, restored = run_two_steps({"mode": "native"})
    expected = torch.tensor([[[1.5, 0.5]]])
    assert torch.equal(result, expected)
    assert restored == original
    assert all(not event["active"] for event in controller.logs)


def test_sampler_report_is_json_serializable_after_trajectory_capture():
    controller, _, _, _ = run_two_steps(
        {"mode": "tangent", "capture_trajectory": True}
    )
    encoded = json.dumps(controller.report())
    assert '"active": false' in encoded
    assert '"active": true' in encoded


def test_naive_velocity_difference_changes_second_step():
    _, result, _, _ = run_two_steps(
        {
            "mode": "naive",
            "strength": 0.5,
            "max_relative_correction_norm": None,
        }
    )
    # v_ref = [0,1] + .5 * ([-1,1]) = [-.5,1.5]
    assert torch.allclose(result, torch.tensor([[[1.75, 0.25]]]))


def test_tangent_mode_removes_reliability_normal_component():
    controller, _, _, _ = run_two_steps(
        {
            "mode": "tangent",
            "strength": 1.0,
            "max_relative_correction_norm": None,
        }
    )
    assert abs(controller.logs[1]["post_projection_normal_cosine"][0]) < 1e-6


def test_target_norm_matches_orientation_ablation_strength():
    controller, _, _, _ = run_two_steps(
        {
            "mode": "naive",
            "proposal": "x0_difference",
            "target_relative_correction_norm": 0.05,
            "max_relative_correction_norm": 0.25,
        }
    )
    assert controller.logs[1]["correction_relative_norm"][0] == pytest.approx(0.05)


def test_random_control_is_reproducible():
    config = {
        "mode": "naive",
        "proposal": "random_ambient",
        "random_seed": 17,
        "max_relative_correction_norm": None,
    }
    first, first_result, _, _ = run_two_steps(config)
    second, second_result, _, _ = run_two_steps(config)
    assert torch.equal(first_result, second_result)
    assert first.logs[1]["proposal_relative_norm"] == second.logs[1]["proposal_relative_norm"]


def test_config_rejects_unsupported_exact_velocity_field():
    with pytest.raises(ValueError, match="x0_consistency"):
        SamplingRefinementConfig.from_mapping(
            {"normal_estimator": "exact", "reliability_field": "velocity_consistency"}
        )


class TinyTransformer(torch.nn.Module):
    def forward(self, *, hidden_states, return_dict=False, **kwargs):
        return (hidden_states.square(),)


def test_exact_hook_computes_x0_consistency_gradient():
    scheduler = FakeScheduler()
    transformer = TinyTransformer()
    controller = SamplingRefinementController(
        scheduler,
        {
            "mode": "tangent",
            "normal_estimator": "exact",
            "strength": 0.1,
        },
        transformer=transformer,
    )
    sample = torch.tensor([[[0.5, 0.25]]])
    with torch.no_grad(), controller:
        first_prediction = transformer(hidden_states=sample, return_dict=False)[0]
        first = scheduler.step(first_prediction, torch.tensor(1000.0), sample, return_dict=False)[0]
        second_prediction = transformer(hidden_states=first, return_dict=False)[0]
        scheduler.step(second_prediction, torch.tensor(500.0), first, return_dict=False)
    assert controller.logs[1]["normal_norm"][0] > 0
    assert controller.logs[1]["exact_x0_consistency_mse"][0] > 0
