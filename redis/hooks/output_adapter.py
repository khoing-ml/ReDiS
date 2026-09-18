from __future__ import annotations

from typing import Any

import torch


def extract_hidden(out: Any, block_family: str) -> torch.Tensor:
    if block_family == "transformer_blocks":
        if not isinstance(out, tuple) or len(out) != 2 or not torch.is_tensor(out[1]):
            raise TypeError("Expected Flux2 dual-stream output (encoder_hidden_states, hidden_states)")
        return out[1]
    if block_family == "single_transformer_blocks":
        if not torch.is_tensor(out):
            raise TypeError("Expected Flux2 single-stream tensor output")
        return out
    raise ValueError(f"Unknown block family: {block_family}")


def replace_hidden(out: Any, hidden: torch.Tensor, block_family: str) -> Any:
    if block_family == "transformer_blocks":
        return (out[0], hidden)
    if block_family == "single_transformer_blocks":
        return hidden
    raise ValueError(f"Unknown block family: {block_family}")

