from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from statistics import mean, median
from typing import Any

from PIL import Image


RewardBackend = Callable[[Sequence[Image.Image], Sequence[str]], list[float]]


class RewardEvaluator:
    """Lazy, pluggable reward evaluation with one score per image/prompt pair."""

    def __init__(self, backends: dict[str, RewardBackend] | None = None) -> None:
        self._backends = dict(backends or {})

    def evaluate(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        metrics: Sequence[str],
    ) -> dict[str, list[float]]:
        if len(images) != len(prompts):
            raise ValueError("images and prompts must have equal length")
        scores: dict[str, list[float]] = {}
        for metric in metrics:
            backend = self._backends.get(metric)
            if backend is None and metric == "pickscore":
                backend = PickScoreBackend()
                self._backends[metric] = backend
            if backend is None:
                raise ValueError(
                    f"Metric {metric!r} is not enabled; available metrics: "
                    f"{sorted(set(self._backends) | {'pickscore'})}"
                )
            values = backend(images, prompts)
            if len(values) != len(images):
                raise ValueError(f"Metric {metric!r} returned the wrong number of scores")
            scores[metric] = [float(value) for value in values]
        return scores


class PickScoreBackend:
    def __init__(
        self,
        model_id: str = "yuvalkirstain/PickScore_v1",
        processor_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(processor_id)
        self.model = AutoModel.from_pretrained(model_id).eval().to(self.device)

    def __call__(
        self, images: Sequence[Image.Image], prompts: Sequence[str]
    ) -> list[float]:
        torch = self._torch
        image_inputs = self.processor(
            images=list(images), padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(self.device)
        text_inputs = self.processor(
            text=list(prompts), padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            image_features = self.model.get_image_features(**image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            scores = self.model.logit_scale.exp() * (text_features * image_features).sum(dim=-1)
        return scores.float().cpu().tolist()


def paired_score_summary(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("candidate and baseline must be non-empty and equally sized")
    deltas = [float(value) - float(reference) for value, reference in zip(candidate, baseline, strict=True)]
    generator = random.Random(seed)
    boot_means = sorted(
        mean(generator.choice(deltas) for _ in deltas) for _ in range(bootstrap_samples)
    )
    low_index = max(0, int(0.025 * bootstrap_samples))
    high_index = min(bootstrap_samples - 1, int(0.975 * bootstrap_samples))
    return {
        "count": len(deltas),
        "mean_score": mean(float(value) for value in candidate),
        "median_score": median(float(value) for value in candidate),
        "mean_paired_improvement": mean(deltas),
        "median_paired_improvement": median(deltas),
        "fraction_improved": sum(value > 0 for value in deltas) / len(deltas),
        "fraction_degraded": sum(value < 0 for value in deltas) / len(deltas),
        "mean_improvement_95ci": [boot_means[low_index], boot_means[high_index]],
        "paired_deltas": deltas,
    }
