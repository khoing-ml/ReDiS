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
    controller = SamplingRefinementController(
        scheduler,
        {"normal_estimator": "residual_proxy", **config},
    )
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
        {
            "mode": "tangent",
            "normal_estimator": "residual_proxy",
            "capture_trajectory": True,
        }
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
            "normal_estimator": "residual_proxy",
            "strength": 1.0,
            "max_relative_correction_norm": None,
        }
    )
    assert abs(controller.logs[1]["post_projection_normal_cosine"][0]) < 1e-6
    assert controller.logs[1]["subspace_retention"][0] == pytest.approx(1.0)
    assert controller.logs[1]["constraint_retention"][0] == pytest.approx(0.0, abs=1e-6)
    assert abs(controller.logs[1]["pre_projection_normal_cosine"][0]) == pytest.approx(1.0)


def test_non_increasing_constraint_uses_actual_state_displacement_sign():
    controller, _, _, _ = run_two_steps(
        {
            "mode": "non_increasing",
            "normal_estimator": "residual_proxy",
            "strength": 1.0,
            "max_relative_correction_norm": None,
        }
    )
    event = controller.logs[1]
    assert event["constraint_active"] == [True]
    assert event["constraint_retention"][0] == pytest.approx(0.0, abs=1e-6)
    assert event["directional_derivative_post"][0] == pytest.approx(0.0, abs=1e-6)


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


def test_vjp_hook_computes_x0_consistency_gradient():
    scheduler = FakeScheduler()
    transformer = TinyTransformer()
    controller = SamplingRefinementController(
        scheduler,
        {
            "mode": "tangent",
            "normal_estimator": "vjp",
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
    assert controller.logs[1]["vjp_consistency_objective"][0] > 0
    previous_x0 = sample - sample.square()
    current_x0 = first - 0.5 * first.square()
    error = current_x0 - previous_x0
    expected_normal = (1 - first) * error / error.numel()
    assert controller.logs[1]["normal_norm"][0] == pytest.approx(
        float(expected_normal.norm()), rel=1e-5
    )


class ScaledTransformer(torch.nn.Module):
    def forward(self, *, hidden_states, return_dict=False, **kwargs):
        return (3 * hidden_states,)


def test_vjp_non_increasing_controller_removes_consistency_ascent():
    scheduler = FakeScheduler()
    transformer = TinyTransformer()
    controller = SamplingRefinementController(
        scheduler,
        {
            "mode": "non_increasing",
            "normal_estimator": "vjp",
            "strength": 1.0,
            "max_relative_correction_norm": None,
        },
        transformer=transformer,
    )
    sample = torch.tensor([[[0.5, 0.25]]])
    with torch.no_grad(), controller:
        first_prediction = transformer(hidden_states=sample, return_dict=False)[0]
        first = scheduler.step(first_prediction, torch.tensor(1000.0), sample, return_dict=False)[0]
        second_prediction = transformer(hidden_states=first, return_dict=False)[0]
        scheduler.step(second_prediction, torch.tensor(500.0), first, return_dict=False)
    event = controller.logs[1]
    assert event["constraint_active"] == [True]
    assert event["directional_derivative_pre"][0] > 0
    assert event["directional_derivative_post"][0] == pytest.approx(0.0, abs=1e-6)


def test_naive_vjp_logs_finite_correction_that_worsens_consistency():
    scheduler = FakeScheduler()
    transformer = TinyTransformer()
    controller = SamplingRefinementController(
        scheduler,
        {
            "mode": "naive",
            "normal_estimator": "vjp",
            "strength": 0.2,
            "max_relative_correction_norm": None,
        },
        transformer=transformer,
    )
    sample = torch.tensor([[[0.5, 0.25]]])
    with torch.no_grad(), controller:
        first_prediction = transformer(hidden_states=sample, return_dict=False)[0]
        first = scheduler.step(first_prediction, torch.tensor(1000.0), sample, return_dict=False)[0]
        second_prediction = transformer(hidden_states=first, return_dict=False)[0]
        scheduler.step(second_prediction, torch.tensor(500.0), first, return_dict=False)
    event = controller.logs[1]
    assert event["directional_derivative_pre"][0] > 0
    assert event["consistency_ratio"][0] > 1
    assert event["finite_directional_curvature"][0] > 0


def test_finite_consistency_check_measures_nonzero_descent_correction():
    scheduler = FakeScheduler()
    transformer = ScaledTransformer()
    controller = SamplingRefinementController(
        scheduler,
        {
            "mode": "non_increasing",
            "normal_estimator": "vjp",
            "strength": 1.0,
            "max_relative_correction_norm": None,
        },
        transformer=transformer,
    )
    sample = torch.tensor([[[0.5, 0.25]]])
    with torch.no_grad(), controller:
        first_prediction = transformer(hidden_states=sample, return_dict=False)[0]
        first = scheduler.step(first_prediction, torch.tensor(1000.0), sample, return_dict=False)[0]
        second_prediction = transformer(hidden_states=first, return_dict=False)[0]
        scheduler.step(second_prediction, torch.tensor(500.0), first, return_dict=False)
    event = controller.logs[1]
    assert event["constraint_active"] == [False]
    assert event["consistency_ratio"][0] == pytest.approx(0.25, rel=1e-5)
    assert event["state_correction_norm"][0] > 0
    assert event["finite_directional_curvature"][0] > 0


def test_trust_region_shrinks_until_finite_consistency_is_accepted():
    scheduler = FakeScheduler()
    transformer = TinyTransformer()
    controller = SamplingRefinementController(
        scheduler,
        {
            "mode": "non_increasing",
            "normal_estimator": "vjp",
            "trust_region": True,
            "trust_region_shrink_factor": 0.5,
            "trust_region_max_shrinks": 4,
        },
        transformer=transformer,
    )
    controller._pending_vjp_score = torch.tensor([1.0])
    controller._previous_x0 = torch.zeros(1, 1, 2)
    controller._velocities = [torch.ones(1, 1, 2)]

    def trial_score(state, sigma):
        return torch.where(
            state.flatten(1).norm(dim=1) > 0.2,
            torch.tensor(2.0),
            torch.tensor(0.5),
        )

    controller._evaluate_consistency = trial_score
    correction, diagnostics = controller._finite_consistency_diagnostics(
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, 2),
        torch.ones(1, 1, 2),
        torch.ones(1, 1, 2),
        sigma=1.0,
        next_sigma=0.5,
    )
    assert diagnostics["trust_region_scale"].tolist() == [0.25]
    assert diagnostics["trust_region_shrinks"].tolist() == [2]
    assert diagnostics["consistency_after"].tolist() == [0.5]
    assert torch.allclose(correction, torch.full_like(correction, 0.25))
