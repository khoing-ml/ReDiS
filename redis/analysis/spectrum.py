from __future__ import annotations

import math

import torch

from redis.views import centered_residual, low_frequency_residual, token_pooled_residual


def _spectrum_from_rows(
    rows: torch.Tensor,
    *,
    maximum_rank: int,
    eps: float,
) -> dict[str, object]:
    gram = (rows @ rows.T) / rows.shape[1]
    eigvals = torch.linalg.eigvalsh(gram).clamp_min(0)
    tolerance = max(float(eigvals.max()) * 1e-6, eps)
    active = eigvals[eigvals > tolerance]
    if active.numel() == 0:
        return {
            "effective_rank": 0.0,
            "normalized_effective_rank": 0.0,
            "stable_rank": 0.0,
            "top1_ratio": 0.0,
            "maximum_rank": maximum_rank,
            "eigvals": eigvals.tolist(),
        }
    probabilities = active / active.sum().clamp_min(eps)
    effective_rank = torch.exp(-(probabilities * torch.log(probabilities + eps)).sum())
    return {
        "effective_rank": effective_rank.item(),
        "normalized_effective_rank": (effective_rank / maximum_rank).item(),
        "stable_rank": (active.sum() / active.max().clamp_min(eps)).item(),
        "top1_ratio": (active.max() / active.sum().clamp_min(eps)).item(),
        "maximum_rank": maximum_rank,
        "eigvals": eigvals.tolist(),
    }


def seed_spectrum(hidden: torch.Tensor, eps: float = 1e-12) -> dict[str, object]:
    if hidden.ndim != 3:
        raise ValueError(f"Expected [B,N,D], got {tuple(hidden.shape)}")
    batch = hidden.shape[0]
    if batch < 2:
        raise ValueError("Seed spectrum requires at least two samples")
    flattened = centered_residual(hidden).flatten(1)
    return _spectrum_from_rows(flattened, maximum_rank=batch - 1, eps=eps)


def row_normalized_seed_spectrum(
    hidden: torch.Tensor, eps: float = 1e-12
) -> dict[str, object]:
    """Spectrum of centered seed residuals after unit Frobenius normalization."""
    if hidden.ndim != 3:
        raise ValueError(f"Expected [B,N,D], got {tuple(hidden.shape)}")
    batch = hidden.shape[0]
    if batch < 2:
        raise ValueError("Seed spectrum requires at least two samples")
    rows = centered_residual(hidden).flatten(1)
    rows = rows / rows.norm(dim=1, keepdim=True).clamp_min(eps)
    # Do not center again: this is exactly R_i / ||R_i||_F from the diagnostic.
    return _spectrum_from_rows(rows, maximum_rank=batch, eps=eps)


def residual_rms(hidden: torch.Tensor) -> dict[str, object]:
    residual = centered_residual(hidden)
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
    """Measure raw and row-normalized seed geometry for each residual view."""
    residual = centered_residual(hidden)
    full_energy = residual.square().sum().clamp_min(1e-12)

    def describe(view: torch.Tensor, *, lifted: torch.Tensor | None = None) -> dict[str, object]:
        energy_tensor = view if lifted is None else lifted
        per_seed_rms = view.square().mean(dim=(1, 2)).sqrt()
        return {
            "raw_spectrum": seed_spectrum(view),
            "row_normalized_spectrum": row_normalized_seed_spectrum(view),
            "view_rms": {
                "global": view.square().mean().sqrt().item(),
                "per_seed": per_seed_rms.tolist(),
            },
            "energy_over_full_residual": (
                energy_tensor.square().sum() / full_energy
            ).item(),
        }

    pooled = token_pooled_residual(residual)
    low_frequency: torch.Tensor | None
    try:
        low_frequency = low_frequency_residual(residual)
    except ValueError:
        low_frequency = None

    views: dict[str, object] = {
        "full_hidden": describe(residual),
        "token_pooled": describe(
            pooled, lifted=pooled.expand(-1, hidden.shape[1], -1)
        ),
        "spatial_low_frequency": (
            describe(low_frequency) if low_frequency is not None else None
        ),
    }
    if projection_basis is not None:
        projected = residual @ projection_basis.float()
        views["projected"] = describe(projected)

    diagnostics: dict[str, object] = {
        "shape": list(hidden.shape),
        "views": views,
        # Compatibility keys for existing reports and readers.
        "full_hidden_spectrum": views["full_hidden"]["raw_spectrum"],
        "token_pooled_spectrum": views["token_pooled"]["raw_spectrum"],
        "spatial_low_frequency_spectrum": (
            views["spatial_low_frequency"]["raw_spectrum"]
            if views["spatial_low_frequency"] is not None
            else None
        ),
        "projected_spectrum": (
            views["projected"]["raw_spectrum"]
            if "projected" in views
            else None
        ),
        "spatial_grid": (
            [math.isqrt(hidden.shape[1]), math.isqrt(hidden.shape[1])]
            if low_frequency is not None
            else None
        ),
        "residual_rms": residual_rms(hidden),
    }
    return diagnostics
