from __future__ import annotations

import math
from statistics import mean
from typing import Any, Iterable


def _numbers(steps: Iterable[dict[str, Any]], key: str) -> list[float]:
    result: list[float] = []
    for step in steps:
        value = step.get(key)
        if isinstance(value, list):
            result.extend(float(item) for item in value)
    return [value for value in result if math.isfinite(value)]


def aggregate_sampler_steps(steps: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Aggregate the diagnostics required by the CCSR baseline experiment."""
    active = [step for step in steps if bool(step.get("active"))]
    ratios = _numbers(active, "consistency_ratio")
    rejected = _numbers(active, "trust_region_rejected")
    constraint = _numbers(active, "constraint_retention")
    corrections = _numbers(active, "correction_relative_norm")
    state_corrections = _numbers(active, "state_correction_relative_native_update")
    shrinks = _numbers(active, "trust_region_shrinks")
    first_order_feasible: list[bool] = []
    first_order_failed: list[bool] = []
    for step in active:
        derivatives = step.get("directional_derivative_post")
        after = step.get("consistency_after")
        before = step.get("consistency_before")
        if not all(isinstance(value, list) for value in (derivatives, after, before)):
            continue
        for derivative, after_value, before_value in zip(derivatives, after, before, strict=True):
            feasible = float(derivative) <= 1e-7
            first_order_feasible.append(feasible)
            first_order_failed.append(feasible and float(after_value) > float(before_value))

    def average(values: list[float]) -> float | None:
        return mean(values) if values else None

    feasible_count = sum(first_order_feasible)
    return {
        "active_step_samples": sum(
            len(step.get("correction_relative_norm", [])) for step in active
        ),
        "mean_correction_relative_norm": average(corrections),
        "mean_state_correction_relative_native_update": average(state_corrections),
        "mean_constraint_retention": average(constraint),
        "mean_consistency_ratio": average(ratios),
        "max_consistency_ratio": max(ratios) if ratios else None,
        "trust_region_rejection_rate": average(rejected),
        "mean_trust_region_shrinks": average(shrinks),
        "first_order_feasible_count": feasible_count,
        "first_order_failure_rate": (
            sum(first_order_failed) / feasible_count if feasible_count else None
        ),
    }
