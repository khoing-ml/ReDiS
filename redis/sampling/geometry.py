from __future__ import annotations

from collections.abc import Sequence

import torch


def _require_batched_tensor(value: torch.Tensor, name: str) -> None:
    if value.ndim < 2:
        raise ValueError(f"{name} must have a batch dimension, got {tuple(value.shape)}")


def batched_dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return one ambient-space inner product per batch item."""
    if left.shape != right.shape:
        raise ValueError(f"Tensor shapes differ: {tuple(left.shape)} != {tuple(right.shape)}")
    _require_batched_tensor(left, "left")
    return (left.float().flatten(1) * right.float().flatten(1)).sum(dim=1)


def batched_norm(value: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    _require_batched_tensor(value, "value")
    squared = batched_dot(value, value)
    return squared.clamp_min(eps * eps).sqrt()


def _expand_batch(values: torch.Tensor, ndim: int) -> torch.Tensor:
    return values.reshape(values.shape[0], *([1] * (ndim - 1)))


def orthonormalize(
    directions: Sequence[torch.Tensor],
    *,
    eps: float = 1e-8,
    relative_tolerance: float = 1e-6,
) -> list[torch.Tensor]:
    """Batched modified Gram-Schmidt for a short list of ambient tensors.

    A direction can be linearly dependent for one sample but independent for
    another. Such per-sample degeneracies are represented by a zero basis row.
    """
    if not directions:
        return []
    shape = directions[0].shape
    _require_batched_tensor(directions[0], "direction")
    if any(direction.shape != shape for direction in directions):
        raise ValueError("All directions must have identical shapes")

    basis: list[torch.Tensor] = []
    for direction in directions:
        value = direction.float()
        original_norm = batched_norm(value)
        for vector in basis:
            coefficient = batched_dot(value, vector)
            value = value - _expand_batch(coefficient, value.ndim) * vector
        residual_norm = batched_norm(value)
        threshold = torch.maximum(
            original_norm * relative_tolerance,
            torch.full_like(original_norm, eps),
        )
        active = residual_norm > threshold
        normalized = value / _expand_batch(residual_norm.clamp_min(eps), value.ndim)
        normalized = normalized * _expand_batch(active, value.ndim)
        basis.append(normalized)
    return basis


def project_onto_span(
    value: torch.Tensor,
    directions: Sequence[torch.Tensor],
    *,
    eps: float = 1e-8,
    relative_tolerance: float = 1e-6,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Project an ambient tensor into a per-sample trajectory span."""
    if not directions:
        return torch.zeros_like(value, dtype=torch.float32), []
    basis = orthonormalize(
        directions,
        eps=eps,
        relative_tolerance=relative_tolerance,
    )
    projected = torch.zeros_like(value, dtype=torch.float32)
    for vector in basis:
        coefficient = batched_dot(value, vector)
        projected = projected + _expand_batch(coefficient, value.ndim) * vector
    return projected, basis


def tangent_project(
    proposal: torch.Tensor,
    normal: torch.Tensor,
    directions: Sequence[torch.Tensor],
    *,
    tangent_strength: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Project into a supported span, then onto a reliability level-set tangent."""
    if not 0.0 <= tangent_strength <= 1.0:
        raise ValueError("tangent_strength must be in [0, 1]")
    proposal_subspace, basis = project_onto_span(proposal, directions, eps=eps)
    normal_subspace = torch.zeros_like(normal, dtype=torch.float32)
    for vector in basis:
        coefficient = batched_dot(normal, vector)
        normal_subspace = normal_subspace + _expand_batch(coefficient, normal.ndim) * vector

    denominator = batched_dot(normal_subspace, normal_subspace)
    coefficient = batched_dot(proposal_subspace, normal_subspace) / denominator.clamp_min(eps)
    coefficient = torch.where(denominator > eps, coefficient, torch.zeros_like(coefficient))
    removed = _expand_batch(coefficient, proposal.ndim) * normal_subspace
    safe = proposal_subspace - tangent_strength * removed
    return safe, {
        "proposal_subspace": proposal_subspace,
        "normal_subspace": normal_subspace,
        "removed": removed,
        "tangent_dot": batched_dot(safe, normal_subspace),
    }


def cap_relative_norm(
    correction: torch.Tensor,
    reference: torch.Tensor,
    maximum: float | None,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cap ||correction_i|| / ||reference_i|| independently per sample."""
    if maximum is None:
        return correction, torch.ones(correction.shape[0], device=correction.device)
    if maximum <= 0:
        raise ValueError("maximum must be positive or None")
    correction_norm = batched_norm(correction)
    reference_norm = batched_norm(reference)
    scale = (maximum * reference_norm / correction_norm.clamp_min(eps)).clamp(max=1.0)
    scale = torch.where(correction_norm > eps, scale, torch.ones_like(scale))
    return correction * _expand_batch(scale, correction.ndim), scale


def match_relative_norm(
    correction: torch.Tensor,
    reference: torch.Tensor,
    target: float | None,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match ||correction_i|| / ||reference_i|| per sample when nonzero."""
    if target is None:
        return correction, torch.ones(correction.shape[0], device=correction.device)
    if target <= 0:
        raise ValueError("target must be positive or None")
    correction_norm = batched_norm(correction)
    reference_norm = batched_norm(reference)
    scale = target * reference_norm / correction_norm.clamp_min(eps)
    usable = correction_norm > torch.maximum(
        reference_norm * eps,
        torch.full_like(correction_norm, eps),
    )
    scale = torch.where(usable, scale, torch.zeros_like(scale))
    return correction * _expand_batch(scale, correction.ndim), scale
