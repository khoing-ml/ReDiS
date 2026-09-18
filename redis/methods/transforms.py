from __future__ import annotations

import torch

from redis.analysis import seed_spectrum
from redis.views import centered_residual, low_frequency_residual, token_pooled_residual


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


def _isotropize_residual_view(
    view: torch.Tensor,
    basis: torch.Tensor,
    *,
    beta: float,
    eps: float = 1e-8,
    relative_tolerance: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, object]]:
    if not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    q = basis.float()
    projected = view.float() @ q
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
        return torch.zeros_like(view, dtype=torch.float32), diagnostics

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
    diagnostics.update(
        projected_post_spectrum=seed_spectrum(isotropic.detach().float().cpu()),
        whitening_scale_min=scales[active].min().item(),
        whitening_scale_max=scales[active].max().item(),
        relative_delta_y_norm=(
            delta_y.norm() / projected.norm().clamp_min(eps)
        ).item(),
    )
    return correction, diagnostics


def _finish_isotropization(
    hidden: torch.Tensor,
    correction: torch.Tensor,
    diagnostics: dict[str, object],
    *,
    gamma: float,
    view: str,
    return_diagnostics: bool,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    modified = (hidden.float() + gamma * correction).to(hidden.dtype)
    diagnostics["view"] = view
    diagnostics["gamma_before_strength_match"] = gamma
    return (modified, diagnostics) if return_diagnostics else modified


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
    if (beta == 0 or gamma == 0) and not return_diagnostics:
        return hidden
    residual = centered_residual(hidden)
    correction, diagnostics = _isotropize_residual_view(
        residual,
        basis,
        beta=beta,
        eps=eps,
        relative_tolerance=relative_tolerance,
    )
    return _finish_isotropization(
        hidden,
        correction,
        diagnostics,
        gamma=gamma,
        view="full_hidden",
        return_diagnostics=return_diagnostics,
    )


def low_frequency_isotropize(
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
    if (beta == 0 or gamma == 0) and not return_diagnostics:
        return hidden
    low_frequency = low_frequency_residual(centered_residual(hidden))
    correction, diagnostics = _isotropize_residual_view(
        low_frequency,
        basis,
        beta=beta,
        eps=eps,
        relative_tolerance=relative_tolerance,
    )
    return _finish_isotropization(
        hidden,
        correction,
        diagnostics,
        gamma=gamma,
        view="spatial_low_frequency",
        return_diagnostics=return_diagnostics,
    )


def token_pooled_isotropize(
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
    if (beta == 0 or gamma == 0) and not return_diagnostics:
        return hidden
    pooled = token_pooled_residual(centered_residual(hidden))
    pooled_correction, diagnostics = _isotropize_residual_view(
        pooled,
        basis,
        beta=beta,
        eps=eps,
        relative_tolerance=relative_tolerance,
    )
    correction = pooled_correction.expand(-1, hidden.shape[1], -1)
    return _finish_isotropization(
        hidden,
        correction,
        diagnostics,
        gamma=gamma,
        view="token_pooled",
        return_diagnostics=return_diagnostics,
    )


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


def match_relative_norm(
    delta: torch.Tensor,
    hidden: torch.Tensor,
    target: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Match ||delta_i|| / ||hidden_i|| independently for every seed."""
    if target <= 0:
        raise ValueError("target correction ratio must be positive")
    if delta.shape != hidden.shape:
        raise ValueError("delta and hidden must have identical shapes")
    delta_float = delta.float()
    hidden_float = hidden.float()
    delta_norm = delta_float.flatten(1).norm(dim=1, keepdim=True)
    hidden_norm = hidden_float.flatten(1).norm(dim=1, keepdim=True)
    scale = target * hidden_norm / (delta_norm + eps)
    scale = scale.view(hidden.shape[0], *([1] * (hidden.ndim - 1)))
    return (delta_float * scale).to(delta.dtype)
