from __future__ import annotations

import torch

from redis.analysis import seed_spectrum


def _require_group(hidden: torch.Tensor) -> None:
    if hidden.ndim != 3:
        raise ValueError(f"Expected hidden [B,N,D], got {tuple(hidden.shape)}")
    if hidden.shape[0] < 2:
        raise ValueError("Interventions require at least two seeds for one prompt")


def residual_amplify(hidden: torch.Tensor, gamma: float) -> torch.Tensor:
    _require_group(hidden)
    values = hidden.float()
    residual = values - values.mean(dim=0, keepdim=True)
    return (values + gamma * residual).to(hidden.dtype)


def gaussian_noise(
    hidden: torch.Tensor,
    sigma: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    _require_group(hidden)
    values = hidden.float()
    residual = values - values.mean(dim=0, keepdim=True)
    residual_rms = residual.square().mean().sqrt()
    noise = torch.randn(
        values.shape,
        device=values.device,
        dtype=values.dtype,
        generator=generator,
    )
    noise = noise / noise.square().mean().sqrt().clamp_min(1e-8)
    return (values + sigma * residual_rms * noise).to(hidden.dtype)


def make_random_basis(
    channels: int,
    rank: int,
    *,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    if rank <= 0 or rank > channels:
        raise ValueError(f"Projection rank must be in [1, {channels}], got {rank}")
    generator = torch.Generator(device=device).manual_seed(seed)
    matrix = torch.randn(channels, rank, device=device, dtype=torch.float32, generator=generator)
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    return basis


def projected_amplify(hidden: torch.Tensor, basis: torch.Tensor, gamma: float) -> torch.Tensor:
    _require_group(hidden)
    values = hidden.float()
    q = basis.float()
    residual = values - values.mean(dim=0, keepdim=True)
    correction = (residual @ q) @ q.T
    return (values + gamma * correction).to(hidden.dtype)


def gram_isotropize(
    hidden: torch.Tensor,
    basis: torch.Tensor,
    *,
    beta: float,
    gamma: float,
    eps: float = 1e-8,
    relative_tolerance: float = 1e-6,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    _require_group(hidden)
    if not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    if (beta == 0 or gamma == 0) and not return_diagnostics:
        return hidden

    values = hidden.float()
    q = basis.float()
    residual = values - values.mean(dim=0, keepdim=True)
    projected = residual @ q
    flattened = projected.flatten(1)
    flattened = flattened - flattened.mean(dim=0, keepdim=True)
    gram = (flattened @ flattened.T) / flattened.shape[1]
    eigvals, eigenvectors = torch.linalg.eigh(gram)
    eigvals = eigvals.clamp_min(0)
    tolerance = torch.maximum(
        eigvals.max() * relative_tolerance,
        torch.tensor(1e-12, device=eigvals.device),
    )
    active = eigvals > tolerance
    diagnostics: dict[str, object] = {
        "projected_pre_spectrum": seed_spectrum(projected.detach().float().cpu()),
        "projected_post_spectrum": seed_spectrum(projected.detach().float().cpu()),
        "whitening_scale_min": None,
        "whitening_scale_max": None,
        "relative_delta_y_norm": 0.0,
        "active_eigenvalues": int(active.sum()),
        "energy_match": "global",
    }
    if int(active.sum()) < 2:
        return (hidden, diagnostics) if return_diagnostics else hidden

    active_values = eigvals[active]
    reference = torch.exp(torch.log(active_values + eps).mean())
    scales = torch.zeros_like(eigvals)
    scales[active] = (reference / (active_values + eps)).pow(beta / 2)
    operator = (eigenvectors * scales.unsqueeze(0)) @ eigenvectors.T
    isotropic = operator @ flattened
    isotropic = isotropic * (flattened.norm() / isotropic.norm().clamp_min(eps))
    isotropic = isotropic.reshape_as(projected)
    delta_y = isotropic - projected
    correction = delta_y @ q.T
    modified = (values + gamma * correction).to(hidden.dtype)
    projected_modified = projected + gamma * delta_y
    diagnostics.update(
        projected_post_spectrum=seed_spectrum(
            projected_modified.detach().float().cpu()
        ),
        whitening_scale_min=scales[active].min().item(),
        whitening_scale_max=scales[active].max().item(),
        relative_delta_y_norm=(
            delta_y.norm() / projected.norm().clamp_min(eps)
        ).item(),
    )
    return (modified, diagnostics) if return_diagnostics else modified


def rms_match(modified: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dims = tuple(range(1, modified.ndim))
    modified_rms = modified.float().square().mean(dim=dims, keepdim=True).sqrt()
    reference_rms = reference.float().square().mean(dim=dims, keepdim=True).sqrt()
    return (modified.float() * (reference_rms / (modified_rms + eps))).to(modified.dtype)


def cap_relative_correction(
    modified: torch.Tensor,
    reference: torch.Tensor,
    maximum: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    if maximum <= 0:
        raise ValueError("maximum correction ratio must be positive")
    correction = modified.float() - reference.float()
    ratio = correction.norm() / reference.float().norm().clamp_min(eps)
    if float(ratio) <= maximum:
        return modified
    scaled = reference.float() + correction * (maximum / ratio)
    return scaled.to(modified.dtype)


def match_relative_correction(
    modified: torch.Tensor,
    reference: torch.Tensor,
    target: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, float, float]:
    """Scale a correction to a target Frobenius norm relative to reference."""
    if target <= 0:
        raise ValueError("target correction ratio must be positive")
    correction = modified.float() - reference.float()
    raw_ratio = correction.norm() / reference.float().norm().clamp_min(eps)
    if float(raw_ratio) <= eps:
        return reference, 0.0, float(raw_ratio)
    scale = target / raw_ratio
    matched = reference.float() + correction * scale
    return matched.to(modified.dtype), float(scale), float(raw_ratio)
