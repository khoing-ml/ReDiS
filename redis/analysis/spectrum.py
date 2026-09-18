from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def seed_spectrum(hidden: torch.Tensor, eps: float = 1e-12) -> dict[str, object]:
    if hidden.ndim != 3:
        raise ValueError(f"Expected [B,N,D], got {tuple(hidden.shape)}")
    batch = hidden.shape[0]
    if batch < 2:
        raise ValueError("Seed spectrum requires at least two samples")
    values = hidden.float()
    residual = values - values.mean(dim=0, keepdim=True)
    flattened = residual.flatten(1)
    gram = (flattened @ flattened.T) / flattened.shape[1]
    eigvals = torch.linalg.eigvalsh(gram).clamp_min(0)
    tolerance = max(float(eigvals.max()) * 1e-6, eps)
    active = eigvals[eigvals > tolerance]
    if active.numel() == 0:
        return {
            "effective_rank": 0.0,
            "normalized_effective_rank": 0.0,
            "stable_rank": 0.0,
            "top1_ratio": 0.0,
            "eigvals": eigvals.tolist(),
        }
    probabilities = active / active.sum().clamp_min(eps)
    effective_rank = torch.exp(-(probabilities * torch.log(probabilities + eps)).sum())
    return {
        "effective_rank": effective_rank.item(),
        "normalized_effective_rank": (effective_rank / (batch - 1)).item(),
        "stable_rank": (active.sum() / active.max().clamp_min(eps)).item(),
        "top1_ratio": (active.max() / active.sum().clamp_min(eps)).item(),
        "eigvals": eigvals.tolist(),
    }


def residual_rms(hidden: torch.Tensor) -> dict[str, object]:
    values = hidden.float()
    residual = values - values.mean(dim=0, keepdim=True)
    per_seed = residual.square().mean(dim=(1, 2)).sqrt()
    return {
        "global": residual.square().mean().sqrt().item(),
        "per_seed": per_seed.tolist(),
    }


def representation_diagnostics(
    hidden: torch.Tensor,
    *,
    projection_basis: torch.Tensor | None = None,
) -> dict[str, object]:
    """Measure complementary views of one image-token activation [B,N,D]."""
    diagnostics: dict[str, object] = {
        "shape": list(hidden.shape),
        "full_hidden_spectrum": seed_spectrum(hidden),
        "token_pooled_spectrum": seed_spectrum(hidden.mean(dim=1, keepdim=True)),
        "residual_rms": residual_rms(hidden),
    }
    if projection_basis is not None:
        projected = hidden.float() @ projection_basis.float()
        diagnostics["projected_spectrum"] = seed_spectrum(projected)

    token_count = hidden.shape[1]
    side = math.isqrt(token_count)
    if side * side == token_count and side >= 2:
        grid = hidden.float().reshape(hidden.shape[0], side, side, hidden.shape[2])
        grid = grid.permute(0, 3, 1, 2)
        low_frequency = F.avg_pool2d(grid, kernel_size=2, stride=2)
        low_frequency = low_frequency.permute(0, 2, 3, 1).flatten(1, 2)
        diagnostics["spatial_low_frequency_spectrum"] = seed_spectrum(low_frequency)
        diagnostics["spatial_grid"] = [side, side]
    else:
        diagnostics["spatial_low_frequency_spectrum"] = None
        diagnostics["spatial_grid"] = None
    return diagnostics
