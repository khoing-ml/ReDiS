from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def centered_residual(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3:
        raise ValueError(f"Expected [B,N,D], got {tuple(hidden.shape)}")
    values = hidden.float()
    return values - values.mean(dim=0, keepdim=True)


def low_frequency_residual(residual: torch.Tensor) -> torch.Tensor:
    """2x2 block-average a square token grid, then lift it to full resolution."""
    if residual.ndim != 3:
        raise ValueError(f"Expected [B,N,D], got {tuple(residual.shape)}")
    token_count = residual.shape[1]
    side = math.isqrt(token_count)
    if side * side != token_count or side < 2:
        raise ValueError(
            f"Low-frequency view requires a square token grid, got N={token_count}"
        )
    grid = residual.float().reshape(
        residual.shape[0], side, side, residual.shape[2]
    ).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(grid, kernel_size=2, stride=2)
    lifted = F.interpolate(pooled, size=(side, side), mode="nearest")
    return lifted.permute(0, 2, 3, 1).reshape_as(residual)


def token_pooled_residual(residual: torch.Tensor) -> torch.Tensor:
    if residual.ndim != 3:
        raise ValueError(f"Expected [B,N,D], got {tuple(residual.shape)}")
    return residual.float().mean(dim=1, keepdim=True)
