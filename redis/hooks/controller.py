from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from .output_adapter import extract_hidden, replace_hidden
from redis.analysis import residual_rms, seed_spectrum
from redis.methods import (
    gaussian_noise,
    gram_isotropize,
    make_random_basis,
    projected_amplify,
    residual_amplify,
)
from redis.methods.transforms import (
    cap_relative_correction,
    match_relative_correction,
    rms_match,
)


class IdentityController:
    def process_block_output(self, out: Any, *, family: str, layer_id: int) -> Any:
        return out


class ImageTokenObserver(IdentityController):
    def __init__(self, target: Any) -> None:
        self.target = target

    def process_block_output(self, out: Any, *, family: str, layer_id: int) -> Any:
        if family != "transformer_blocks":
            raise ValueError("ImageTokenObserver must wrap a dual-stream block")
        self.target.image_token_count = extract_hidden(out, family).shape[1]
        return out


class CaptureController(IdentityController):
    def __init__(self, *, clone_to_cpu: bool = True) -> None:
        self.clone_to_cpu = clone_to_cpu
        self.activations: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
        self.image_token_count: int | None = None

    def process_block_output(self, out: Any, *, family: str, layer_id: int) -> Any:
        hidden = extract_hidden(out, family).detach()
        if family == "transformer_blocks":
            self.image_token_count = hidden.shape[1]
        elif family == "single_transformer_blocks":
            if self.image_token_count is None:
                raise RuntimeError(
                    "Cannot isolate image tokens in a single-stream block. "
                    "Capture at least one transformer_blocks site as well."
                )
            hidden = hidden[:, -self.image_token_count :]
        if self.clone_to_cpu:
            hidden = hidden.float().cpu()
        else:
            hidden = hidden.clone()
        self.activations[(family, layer_id)].append(hidden)
        return out


class InterventionController(IdentityController):
    def __init__(
        self,
        *,
        method: str,
        timestep_ids: list[int],
        gamma: float,
        sigma: float = 0.1,
        projection_rank: int = 64,
        projection_seed: int = 0,
        beta: float = 0.5,
        match_rms: bool = True,
        max_relative_correction_norm: float = 0.25,
        target_relative_correction_norm: float | None = None,
    ) -> None:
        self.method = method
        self.timestep_ids = set(timestep_ids)
        self.gamma = gamma
        self.sigma = sigma
        self.projection_rank = projection_rank
        self.projection_seed = projection_seed
        self.beta = beta
        self.match_rms = match_rms
        self.max_relative_correction_norm = max_relative_correction_norm
        self.target_relative_correction_norm = target_relative_correction_norm
        self.invocations: dict[tuple[str, int], int] = defaultdict(int)
        self.bases: dict[tuple[str, int, int, torch.device], torch.Tensor] = {}
        self.logs: list[dict[str, Any]] = []
        self.image_token_count: int | None = None

    def _basis(self, hidden: torch.Tensor, family: str, layer_id: int) -> torch.Tensor:
        rank = min(self.projection_rank, hidden.shape[-1])
        key = (family, layer_id, rank, hidden.device)
        if key not in self.bases:
            family_offset = 0 if family == "transformer_blocks" else 100_000
            self.bases[key] = make_random_basis(
                hidden.shape[-1],
                rank,
                device=hidden.device,
                seed=self.projection_seed + family_offset + layer_id,
            )
        return self.bases[key]

    def process_block_output(self, out: Any, *, family: str, layer_id: int) -> Any:
        key = (family, layer_id)
        timestep_id = self.invocations[key]
        self.invocations[key] += 1
        if timestep_id not in self.timestep_ids:
            return out

        full_hidden = extract_hidden(out, family)
        if family == "transformer_blocks":
            self.image_token_count = full_hidden.shape[1]
            hidden = full_hidden
        else:
            if self.image_token_count is None:
                raise RuntimeError(
                    "Cannot intervene on single-stream image tokens without observing "
                    "a transformer_blocks site in the same controller."
                )
            hidden = full_hidden[:, -self.image_token_count :]
        if hidden.shape[0] < 2:
            raise ValueError("Intervention batch must contain at least two seeds for one prompt")
        method_diagnostics: dict[str, object] = {}
        if self.method == "residual_amplification":
            modified = residual_amplify(hidden, self.gamma)
        elif self.method == "gaussian_noise":
            generator = torch.Generator(device=hidden.device).manual_seed(
                self.projection_seed + 10_000 * timestep_id + layer_id
            )
            modified = gaussian_noise(hidden, self.sigma, generator=generator)
        elif self.method == "projected_amplification":
            modified = projected_amplify(hidden, self._basis(hidden, family, layer_id), self.gamma)
        elif self.method == "gram_isotropization":
            modified, method_diagnostics = gram_isotropize(
                hidden,
                self._basis(hidden, family, layer_id),
                beta=self.beta,
                gamma=self.gamma,
                return_diagnostics=True,
            )
        else:
            raise ValueError(f"Unsupported intervention method: {self.method}")

        if self.match_rms:
            modified = rms_match(modified, hidden)
        strength_match_scale = None
        raw_relative_correction_norm = (
            (modified.float() - hidden.float()).norm()
            / hidden.float().norm().clamp_min(1e-8)
        ).item()
        if self.target_relative_correction_norm is not None:
            modified, strength_match_scale, raw_relative_correction_norm = (
                match_relative_correction(
                    modified,
                    hidden,
                    self.target_relative_correction_norm,
                )
            )
        # Cap last so the tensor actually returned by the wrapper obeys the limit.
        modified = cap_relative_correction(
            modified, hidden, self.max_relative_correction_norm
        )
        if not torch.isfinite(modified).all():
            modified = hidden
            fallback = "non_finite"
        else:
            fallback = None
        relative_norm = (
            (modified.float() - hidden.float()).norm()
            / hidden.float().norm().clamp_min(1e-8)
        ).item()
        pre_residual = residual_rms(hidden.detach().float().cpu())
        post_residual = residual_rms(modified.detach().float().cpu())
        parameters = {
            "gamma": self.gamma,
            "sigma": self.sigma,
            "k": self.projection_rank,
            "beta": self.beta,
            "projection_seed": self.projection_seed,
            "rms_match": self.match_rms,
            "energy_match": "global" if self.method == "gram_isotropization" else None,
            "target_relative_correction_norm": self.target_relative_correction_norm,
            "max_relative_correction_norm": self.max_relative_correction_norm,
        }
        log_entry = {
            "family": family,
            "layer_id": layer_id,
            "timestep_id": timestep_id,
            "method": self.method,
            "parameters": parameters,
            "raw_relative_correction_norm": raw_relative_correction_norm,
            "relative_correction_norm": relative_norm,
            "strength_match_scale": strength_match_scale,
            "full_hidden_pre_spectrum": seed_spectrum(hidden.detach().float().cpu()),
            "full_hidden_post_spectrum": seed_spectrum(modified.detach().float().cpu()),
            "residual_rms_pre": pre_residual,
            "residual_rms_post": post_residual,
            "residual_energy_ratio": (
                (post_residual["global"] / pre_residual["global"]) ** 2
                if pre_residual["global"] > 0
                else None
            ),
            "fallback": fallback,
        }
        log_entry.update(method_diagnostics)
        self.logs.append(log_entry)
        if family == "single_transformer_blocks":
            modified = torch.cat(
                [full_hidden[:, : -self.image_token_count], modified], dim=1
            )
        return replace_hidden(out, modified, family)
