from __future__ import annotations

import types
from contextlib import nullcontext
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .geometry import (
    batched_dot,
    batched_norm,
    cap_relative_norm,
    match_relative_norm,
    non_increasing_project,
    project_onto_span,
    tangent_project,
)


MODES = ("native", "naive", "subspace", "tangent", "non_increasing")
NORMAL_ESTIMATORS = ("vjp", "residual_proxy")
NORMAL_ESTIMATOR_ALIASES = {"exact": "vjp", "proxy": "residual_proxy"}
RELIABILITY_FIELDS = ("x0_consistency", "velocity_consistency")
PROPOSALS = (
    "velocity_difference",
    "curvature",
    "x0_difference",
    "random_ambient",
    "random_velocity_orthogonal",
)


@dataclass(frozen=True)
class SamplingRefinementConfig:
    mode: str = "non_increasing"
    proposal: str = "velocity_difference"
    reliability_field: str = "x0_consistency"
    normal_estimator: str = "vjp"
    strength: float = 0.2
    tangent_strength: float = 1.0
    history_size: int = 2
    start_step: int = 1
    end_step: int | None = None
    target_relative_correction_norm: float | None = None
    max_relative_correction_norm: float | None = 0.25
    random_seed: int = 0
    eps: float = 1e-8
    capture_trajectory: bool = False
    finite_consistency_check: bool = True
    trust_region: bool = False
    trust_region_tolerance: float = 0.0
    trust_region_shrink_factor: float = 0.5
    trust_region_max_shrinks: int = 4
    strength_schedule: list[float] | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "SamplingRefinementConfig":
        normalized = dict(values or {})
        estimator = normalized.get("normal_estimator")
        if estimator in NORMAL_ESTIMATOR_ALIASES:
            normalized["normal_estimator"] = NORMAL_ESTIMATOR_ALIASES[estimator]
        config = cls(**normalized)
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.proposal not in PROPOSALS:
            raise ValueError(f"proposal must be one of {PROPOSALS}, got {self.proposal!r}")
        if self.reliability_field not in RELIABILITY_FIELDS:
            raise ValueError(
                f"reliability_field must be one of {RELIABILITY_FIELDS}, got {self.reliability_field!r}"
            )
        if self.normal_estimator not in NORMAL_ESTIMATORS:
            raise ValueError(
                f"normal_estimator must be one of {NORMAL_ESTIMATORS}, got {self.normal_estimator!r}"
            )
        if self.normal_estimator == "vjp" and self.reliability_field != "x0_consistency":
            raise ValueError("VJP normal estimation currently supports x0_consistency only")
        if self.strength < 0:
            raise ValueError("strength must be non-negative")
        if not 0 <= self.tangent_strength <= 1:
            raise ValueError("tangent_strength must be in [0, 1]")
        if self.history_size < 1:
            raise ValueError("history_size must be positive")
        if self.start_step < 0:
            raise ValueError("start_step must be non-negative")
        if self.end_step is not None and self.end_step < self.start_step:
            raise ValueError("end_step must not precede start_step")
        if self.max_relative_correction_norm is not None and self.max_relative_correction_norm <= 0:
            raise ValueError("max_relative_correction_norm must be positive or null")
        if self.target_relative_correction_norm is not None and self.target_relative_correction_norm <= 0:
            raise ValueError("target_relative_correction_norm must be positive or null")
        if (
            self.target_relative_correction_norm is not None
            and self.max_relative_correction_norm is not None
            and self.target_relative_correction_norm > self.max_relative_correction_norm
        ):
            raise ValueError("target_relative_correction_norm cannot exceed the correction cap")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.trust_region and not self.finite_consistency_check:
            raise ValueError("trust_region requires finite_consistency_check")
        if self.trust_region_tolerance < 0:
            raise ValueError("trust_region_tolerance must be non-negative")
        if not 0 < self.trust_region_shrink_factor < 1:
            raise ValueError("trust_region_shrink_factor must be in (0, 1)")
        if self.trust_region_max_shrinks < 0:
            raise ValueError("trust_region_max_shrinks must be non-negative")
        if self.strength_schedule is not None:
            if not self.strength_schedule:
                raise ValueError("strength_schedule must be non-empty or null")
            if any(value < 0 for value in self.strength_schedule):
                raise ValueError("strength_schedule values must be non-negative")


def _per_sample(values: torch.Tensor) -> list[float]:
    return values.detach().float().cpu().tolist()


def _ratios(numerator: torch.Tensor, denominator: torch.Tensor, eps: float) -> torch.Tensor:
    return batched_norm(numerator) / batched_norm(denominator).clamp_min(eps)


class SamplingRefinementController:
    """Inject consistency-constrained corrections at a FlowMatch Euler boundary.

    The controller is a context manager: it temporarily wraps ``scheduler.step``
    and, only for the VJP normal estimator, ``transformer.forward``. No
    Diffusers source file or model weight is changed.
    """

    def __init__(
        self,
        scheduler: Any,
        config: SamplingRefinementConfig | Mapping[str, Any] | None = None,
        *,
        transformer: torch.nn.Module | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.config = (
            config
            if isinstance(config, SamplingRefinementConfig)
            else SamplingRefinementConfig.from_mapping(config)
        )
        self.transformer = transformer
        if (
            self.config.normal_estimator == "vjp"
            and self.config.mode != "native"
            and transformer is None
        ):
            raise ValueError("The VJP normal estimator requires a transformer")
        if bool(getattr(getattr(scheduler, "config", None), "stochastic_sampling", False)):
            raise ValueError("Sampling refinement currently requires deterministic FlowMatch Euler sampling")

        self.logs: list[dict[str, Any]] = []
        self.trajectory: list[dict[str, Any]] = []
        self._velocities: list[torch.Tensor] = []
        self._previous_x0: torch.Tensor | None = None
        self._previous_sigma: float | None = None
        self._pending_vjp_normal: torch.Tensor | None = None
        self._pending_vjp_score: torch.Tensor | None = None
        self._last_transformer_args: tuple[Any, ...] | None = None
        self._last_transformer_kwargs: dict[str, Any] | None = None
        self._last_model_hidden: torch.Tensor | None = None
        self._trial_evaluation = False
        self._step_id = 0
        self._installed = False
        self._original_step: Any = None
        self._original_forward: Any = None
        self._scheduler_had_instance_step = "step" in vars(scheduler)
        self._scheduler_instance_step = vars(scheduler).get("step")
        self._transformer_had_instance_forward = (
            transformer is not None and "forward" in vars(transformer)
        )
        self._transformer_instance_forward = (
            vars(transformer).get("forward") if transformer is not None else None
        )

    def __enter__(self) -> "SamplingRefinementController":
        if self._installed:
            raise RuntimeError("SamplingRefinementController is already installed")
        self._original_step = self.scheduler.step
        self.scheduler.step = self.step
        if (
            self.config.normal_estimator == "vjp"
            and self.config.mode != "native"
        ):
            self._install_vjp_normal_hook()
        self._installed = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._original_step is not None:
            if self._scheduler_had_instance_step:
                self.scheduler.step = self._scheduler_instance_step
            else:
                delattr(self.scheduler, "step")
        if self._original_forward is not None and self.transformer is not None:
            if self._transformer_had_instance_forward:
                self.transformer.forward = self._transformer_instance_forward
            else:
                delattr(self.transformer, "forward")
        self._installed = False

    def _active(self) -> bool:
        return (
            self.config.mode != "native"
            and bool(self._velocities)
            and self._step_id >= self.config.start_step
            and (self.config.end_step is None or self._step_id <= self.config.end_step)
        )

    def _sigma_pair(self, timestep: Any) -> tuple[float, float]:
        if getattr(self.scheduler, "step_index", None) is None:
            self.scheduler._init_step_index(timestep)
        index = int(self.scheduler.step_index)
        return float(self.scheduler.sigmas[index]), float(self.scheduler.sigmas[index + 1])

    def _proposal(
        self,
        velocity: torch.Tensor,
        x0: torch.Tensor,
        sigma: float,
        next_sigma: float,
    ) -> torch.Tensor:
        velocity_difference = velocity.float() - self._velocities[-1].float()
        proposal = velocity_difference
        if self.config.proposal == "curvature":
            if self._previous_sigma is None:
                return torch.zeros_like(proposal)
            previous_dt = sigma - self._previous_sigma
            current_dt = next_sigma - sigma
            if abs(previous_dt) <= self.config.eps:
                return torch.zeros_like(proposal)
            proposal = proposal * (current_dt / previous_dt)
        elif self.config.proposal == "x0_difference":
            if self._previous_x0 is None:
                return torch.zeros_like(proposal)
            proposal = x0.float() - self._previous_x0.float()
        elif self.config.proposal in ("random_ambient", "random_velocity_orthogonal"):
            generator = torch.Generator(device=velocity.device).manual_seed(
                self.config.random_seed + self._step_id
            )
            proposal = torch.randn(
                velocity.shape,
                device=velocity.device,
                dtype=torch.float32,
                generator=generator,
            )
            if self.config.proposal == "random_velocity_orthogonal":
                projected, _ = project_onto_span(proposal, [velocity], eps=self.config.eps)
                proposal = proposal - projected
            target_norm = batched_norm(velocity_difference)
            proposal_norm = batched_norm(proposal).clamp_min(self.config.eps)
            scale = target_norm / proposal_norm
            scale = scale.reshape(scale.shape[0], *([1] * (proposal.ndim - 1)))
            proposal = proposal * scale
        return proposal

    def _strength(self) -> float:
        schedule = self.config.strength_schedule
        if schedule is None:
            return self.config.strength
        return float(schedule[min(self._step_id, len(schedule) - 1)])

    def _normal(self, velocity: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        if self.config.normal_estimator == "vjp":
            if self._pending_vjp_normal is None:
                raise RuntimeError(
                    "VJP reliability normal was not produced. The model may bypass transformer.forward."
                )
            return self._pending_vjp_normal.to(device=velocity.device, dtype=torch.float32)
        if self.config.reliability_field == "x0_consistency":
            if self._previous_x0 is None:
                return torch.zeros_like(x0, dtype=torch.float32)
            # Stop-gradient approximation to d ||x0(x)-x0_prev||^2 / dx.
            return x0.float() - self._previous_x0.float()
        return velocity.float() - self._velocities[-1].float()

    def _evaluate_consistency(self, state: torch.Tensor, sigma: float) -> torch.Tensor:
        if (
            self.transformer is None
            or self._last_transformer_args is None
            or self._last_transformer_kwargs is None
            or self._last_model_hidden is None
            or self._previous_x0 is None
        ):
            raise RuntimeError("Finite consistency evaluation is missing transformer context")
        token_count = state.shape[1]
        model_state = state.to(self._last_model_hidden.dtype)
        if self._last_model_hidden.shape[1] > token_count:
            model_state = torch.cat(
                [model_state, self._last_model_hidden[:, token_count:]], dim=1
            )
        kwargs = dict(self._last_transformer_kwargs)
        kwargs["hidden_states"] = model_state
        cache_context = (
            self.transformer.cache_context("cond")
            if hasattr(self.transformer, "cache_context")
            else nullcontext()
        )
        self._trial_evaluation = True
        try:
            with torch.no_grad(), cache_context:
                output = self.transformer(*self._last_transformer_args, **kwargs)
        finally:
            self._trial_evaluation = False
        prediction = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(prediction):
            raise TypeError("Finite consistency evaluation expected transformer tensor output")
        prediction = prediction[:, :token_count]
        trial_x0 = state.float() - sigma * prediction.float()
        difference = trial_x0 - self._previous_x0.to(trial_x0.device).float()
        return 0.5 * difference.square().flatten(1).mean(dim=1)

    def _finite_consistency_diagnostics(
        self,
        sample: torch.Tensor,
        correction: torch.Tensor,
        normal: torch.Tensor,
        native_velocity: torch.Tensor,
        *,
        sigma: float,
        next_sigma: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._pending_vjp_score is None:
            raise RuntimeError("Finite consistency check requires the VJP objective")
        before = self._pending_vjp_score.to(sample.device).float()
        dt = next_sigma - sigma
        scales = torch.ones(sample.shape[0], device=sample.device)
        shrink_counts = torch.zeros(
            sample.shape[0], device=sample.device, dtype=torch.int64
        )

        def scaled_correction() -> torch.Tensor:
            shape = (scales.shape[0], *([1] * (correction.ndim - 1)))
            return correction * scales.reshape(shape)

        trial_correction = scaled_correction()
        state_delta = dt * trial_correction
        after = self._evaluate_consistency(sample.float() + state_delta, sigma)
        if self.config.trust_region:
            for _ in range(self.config.trust_region_max_shrinks):
                failing = after > before + self.config.trust_region_tolerance
                if not bool(failing.any()):
                    break
                scales = torch.where(
                    failing,
                    scales * self.config.trust_region_shrink_factor,
                    scales,
                )
                shrink_counts = shrink_counts + failing.to(shrink_counts.dtype)
                trial_correction = scaled_correction()
                state_delta = dt * trial_correction
                after = self._evaluate_consistency(sample.float() + state_delta, sigma)
            rejected = after > before + self.config.trust_region_tolerance
            if bool(rejected.any()):
                scales = torch.where(rejected, torch.zeros_like(scales), scales)
                trial_correction = scaled_correction()
                state_delta = dt * trial_correction
                after = torch.where(rejected, before, after)
        else:
            rejected = torch.zeros_like(before, dtype=torch.bool)

        linear_change = batched_dot(normal, state_delta)
        delta_squared = batched_dot(state_delta, state_delta)
        curvature = (after - before - linear_change) / delta_squared.clamp_min(
            self.config.eps
        )
        curvature = torch.where(
            delta_squared > self.config.eps,
            curvature,
            torch.zeros_like(curvature),
        )
        return trial_correction, {
            "consistency_before": before,
            "consistency_after": after,
            "consistency_ratio": after / before.clamp_min(self.config.eps),
            "finite_directional_curvature": curvature,
            "applied_state_directional_derivative": linear_change,
            "trust_region_scale": scales,
            "trust_region_shrinks": shrink_counts,
            "trust_region_rejected": rejected,
            "state_correction_norm": batched_norm(state_delta),
            "state_correction_relative_native_update": _ratios(
                state_delta,
                dt * native_velocity,
                self.config.eps,
            ),
        }

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if kwargs.get("per_token_timesteps") is not None and self.config.mode != "native":
            raise ValueError("Per-token timesteps are not supported by trajectory refinement")
        sigma, next_sigma = self._sigma_pair(timestep)
        native_velocity = model_output.detach()
        native_x0 = sample.detach().float() - sigma * native_velocity.float()
        corrected_velocity = model_output
        correction = torch.zeros_like(model_output, dtype=torch.float32)
        proposal = torch.zeros_like(model_output, dtype=torch.float32)
        projected = proposal
        proposal_subspace = proposal
        normal: torch.Tensor | None = None
        normal_subspace: torch.Tensor | None = None
        constraint_active: torch.Tensor | None = None
        finite_diagnostics: dict[str, torch.Tensor] = {}
        cap_scale = torch.ones(model_output.shape[0], device=model_output.device)
        match_scale = torch.ones(model_output.shape[0], device=model_output.device)

        if self._active():
            proposal = self._proposal(native_velocity, native_x0, sigma, next_sigma)
            proposal_subspace = proposal
            directions = [native_velocity, *reversed(self._velocities)]
            directions = directions[: self.config.history_size]
            if self.config.mode == "naive":
                projected = proposal
                if self.config.normal_estimator == "vjp":
                    normal = self._normal(native_velocity, native_x0)
                    normal_subspace = (next_sigma - sigma) * normal
                    constraint_active = torch.zeros(
                        model_output.shape[0],
                        device=model_output.device,
                        dtype=torch.bool,
                    )
            elif self.config.mode == "subspace":
                projected, _ = project_onto_span(proposal, directions, eps=self.config.eps)
                proposal_subspace = projected
                if self.config.normal_estimator == "vjp":
                    normal = self._normal(native_velocity, native_x0)
                    normal_subspace, _ = project_onto_span(
                        (next_sigma - sigma) * normal,
                        directions,
                        eps=self.config.eps,
                    )
                    constraint_active = torch.zeros(
                        model_output.shape[0],
                        device=model_output.device,
                        dtype=torch.bool,
                    )
            else:
                normal = self._normal(native_velocity, native_x0)
                # The scheduler applies a velocity correction as the state
                # displacement delta_x = (sigma_next - sigma) * delta_v.
                # Constrain the derivative along that actual displacement.
                update_normal = (next_sigma - sigma) * normal
                if self.config.mode == "tangent":
                    projected, projection_diagnostics = tangent_project(
                        proposal,
                        update_normal,
                        directions,
                        tangent_strength=self.config.tangent_strength,
                        eps=self.config.eps,
                    )
                else:
                    projected, projection_diagnostics = non_increasing_project(
                        proposal,
                        update_normal,
                        directions,
                        eps=self.config.eps,
                    )
                proposal_subspace = projection_diagnostics["proposal_subspace"]
                normal_subspace = projection_diagnostics["normal_subspace"]
                constraint_active = projection_diagnostics["constraint_active"]
            correction = self._strength() * projected
            correction, match_scale = match_relative_norm(
                correction,
                native_velocity,
                self.config.target_relative_correction_norm,
                eps=self.config.eps,
            )
            correction, cap_scale = cap_relative_norm(
                correction,
                native_velocity,
                self.config.max_relative_correction_norm,
                eps=self.config.eps,
            )
            if not torch.isfinite(correction).all():
                correction = torch.zeros_like(correction)
                fallback = "non_finite_correction"
            else:
                fallback = None
            if (
                self.config.finite_consistency_check
                and self.config.normal_estimator == "vjp"
                and normal is not None
            ):
                correction, finite_diagnostics = self._finite_consistency_diagnostics(
                    sample,
                    correction,
                    normal,
                    native_velocity,
                    sigma=sigma,
                    next_sigma=next_sigma,
                )
            corrected_velocity = (native_velocity.float() + correction).to(model_output.dtype)
        else:
            fallback = "inactive"

        previous_x0 = self._previous_x0
        consistency = (
            batched_norm(native_x0 - previous_x0)
            if previous_x0 is not None
            else torch.zeros(model_output.shape[0], device=model_output.device)
        )
        log: dict[str, Any] = {
            "step_id": self._step_id,
            "timestep": float(timestep.detach().float().item()) if torch.is_tensor(timestep) else float(timestep),
            "sigma": sigma,
            "next_sigma": next_sigma,
            "active": self._active(),
            "mode": self.config.mode,
            "proposal": self.config.proposal,
            "normal_estimator": self.config.normal_estimator,
            "reliability_field": self.config.reliability_field,
            "effective_strength": self._strength(),
            "native_velocity_norm": _per_sample(batched_norm(native_velocity)),
            "proposal_relative_norm": _per_sample(_ratios(proposal, native_velocity, self.config.eps)),
            "subspace_proposal_relative_norm": _per_sample(
                _ratios(proposal_subspace, native_velocity, self.config.eps)
            ),
            "projected_relative_norm": _per_sample(_ratios(projected, native_velocity, self.config.eps)),
            "correction_relative_norm": _per_sample(_ratios(correction, native_velocity, self.config.eps)),
            "state_correction_norm": _per_sample(
                batched_norm((next_sigma - sigma) * correction)
            ),
            "state_correction_relative_native_update": _per_sample(
                _ratios(
                    (next_sigma - sigma) * correction,
                    (next_sigma - sigma) * native_velocity,
                    self.config.eps,
                )
            ),
            "subspace_retention": _per_sample(
                batched_norm(proposal_subspace)
                / batched_norm(proposal).clamp_min(self.config.eps)
            ),
            "constraint_retention": _per_sample(
                batched_norm(projected)
                / batched_norm(proposal_subspace).clamp_min(self.config.eps)
            ),
            "x0_consistency_norm": _per_sample(consistency),
            "cap_scale": _per_sample(cap_scale),
            "strength_match_scale": _per_sample(match_scale),
            "fallback": fallback,
        }
        if normal is not None:
            log["normal_norm"] = _per_sample(batched_norm(normal))
            assert normal_subspace is not None
            dt_abs = abs(next_sigma - sigma)
            projected_state_normal = normal_subspace / max(dt_abs, self.config.eps)
            log["projected_normal_norm"] = _per_sample(
                batched_norm(projected_state_normal)
            )
            log["pre_projection_normal_cosine"] = _per_sample(
                batched_dot(proposal_subspace, normal_subspace)
                / (
                    batched_norm(proposal_subspace) * batched_norm(normal_subspace)
                ).clamp_min(self.config.eps)
            )
            log["post_projection_normal_cosine"] = _per_sample(
                batched_dot(projected, normal_subspace)
                / (batched_norm(projected) * batched_norm(normal_subspace)).clamp_min(self.config.eps)
            )
            log["directional_derivative_pre"] = _per_sample(
                batched_dot(proposal_subspace, normal_subspace)
            )
            log["directional_derivative_post"] = _per_sample(
                batched_dot(projected, normal_subspace)
            )
            log["constraint_active"] = constraint_active.detach().cpu().tolist()
        for key, value in finite_diagnostics.items():
            log[key] = value.detach().cpu().tolist()
        if self._pending_vjp_score is not None:
            log["vjp_consistency_objective"] = _per_sample(self._pending_vjp_score)
        self.logs.append(log)

        if self.config.capture_trajectory:
            self.trajectory.append(
                {
                    "step_id": self._step_id,
                    "timestep": log["timestep"],
                    "sigma": sigma,
                    "state": sample.detach().float().cpu(),
                    "native_velocity": native_velocity.detach().float().cpu(),
                    "x0": native_x0.detach().float().cpu(),
                    "proposal": proposal.detach().float().cpu(),
                    "correction": correction.detach().float().cpu(),
                    "state_correction": (
                        (next_sigma - sigma) * correction
                    ).detach().float().cpu(),
                }
            )

        result = self._original_step(corrected_velocity, timestep, sample, *args, **kwargs)
        self._velocities.append(native_velocity)
        self._velocities = self._velocities[-self.config.history_size :]
        self._previous_x0 = native_x0.detach()
        self._previous_sigma = sigma
        self._pending_vjp_normal = None
        self._pending_vjp_score = None
        self._step_id += 1
        return result

    def _install_vjp_normal_hook(self) -> None:
        assert self.transformer is not None
        self._original_forward = self.transformer.forward
        controller = self

        def vjp_forward(module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
            if controller._trial_evaluation:
                return controller._original_forward(*args, **kwargs)
            if not controller._active() or controller._previous_x0 is None:
                return controller._original_forward(*args, **kwargs)
            if controller._pending_vjp_normal is not None:
                raise RuntimeError(
                    "Exact normal supports one transformer evaluation per native step; disable CFG"
                )
            if "hidden_states" not in kwargs:
                raise RuntimeError("Exact normal hook requires keyword hidden_states")
            hidden = kwargs["hidden_states"].detach().requires_grad_(True)
            controller._last_transformer_args = tuple(args)
            controller._last_transformer_kwargs = dict(kwargs)
            controller._last_model_hidden = kwargs["hidden_states"].detach()
            differentiable_kwargs = dict(kwargs)
            differentiable_kwargs["hidden_states"] = hidden
            with torch.enable_grad():
                output = controller._original_forward(*args, **differentiable_kwargs)
                prediction = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(prediction):
                    raise TypeError("Exact normal hook expected a tensor or tuple[tensor, ...]")
                token_count = controller._previous_x0.shape[1]
                prediction = prediction[:, :token_count]
                scheduler_index = controller.scheduler.step_index
                if scheduler_index is None:
                    scheduler_index = controller._step_id
                sigma = float(controller.scheduler.sigmas[int(scheduler_index)])
                current_x0 = hidden[:, :token_count].float() - sigma * prediction.float()
                difference = current_x0 - controller._previous_x0.to(current_x0.device).float()
                # S = 1/2 mean(||x0(x_k) - stopgrad(x0_ref)||^2).
                # autograd.grad computes the state-space VJP J_x0^T e.
                per_sample_score = 0.5 * difference.square().flatten(1).mean(dim=1)
                normal = torch.autograd.grad(per_sample_score.sum(), hidden, retain_graph=False)[0]
            controller._pending_vjp_normal = normal[:, :token_count].detach()
            controller._pending_vjp_score = per_sample_score.detach()
            if isinstance(output, tuple):
                return (output[0].detach(), *output[1:])
            return output.detach()

        self.transformer.forward = types.MethodType(vjp_forward, self.transformer)

    def report(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "steps": self.logs,
            "captured_trajectory": bool(self.trajectory),
        }
