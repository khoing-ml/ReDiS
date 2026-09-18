from __future__ import annotations

import torch


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

