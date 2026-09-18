from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_distance_summary(features: torch.Tensor) -> dict[str, object]:
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("Expected at least two feature vectors [B,D]")
    normalized = F.normalize(features.float(), dim=-1)
    distances = 1 - normalized @ normalized.T
    indices = torch.triu_indices(features.shape[0], features.shape[0], offset=1)
    pairs = distances[indices[0], indices[1]]
    return {
        "mean": pairs.mean().item(),
        "median": pairs.median().item(),
        "min": pairs.min().item(),
        "max": pairs.max().item(),
        "pair_count": pairs.numel(),
        "pairwise": pairs.tolist(),
    }

