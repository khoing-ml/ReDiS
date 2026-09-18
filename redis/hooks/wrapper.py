from __future__ import annotations

from typing import Any, Iterable

import torch


class ModifiedBlock(torch.nn.Module):
    def __init__(self, block: torch.nn.Module, family: str, layer_id: int, controller: Any):
        super().__init__()
        self.block = block
        self.family = family
        self.layer_id = layer_id
        self.controller = controller

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        out = self.block(*args, **kwargs)
        return self.controller.process_block_output(
            out, family=self.family, layer_id=self.layer_id
        )


def patch_flux2_blocks(
    transformer: torch.nn.Module,
    controller: Any,
    *,
    family: str = "transformer_blocks",
    layer_ids: Iterable[int] | None = None,
) -> list[int]:
    blocks = getattr(transformer, family, None)
    if not isinstance(blocks, torch.nn.ModuleList):
        raise TypeError(f"transformer.{family} is not a ModuleList")
    selected = list(range(len(blocks))) if layer_ids is None else sorted(set(layer_ids))
    for layer_id in selected:
        if layer_id < 0 or layer_id >= len(blocks):
            raise IndexError(f"Layer {layer_id} outside {family}[0:{len(blocks)}]")
        if isinstance(blocks[layer_id], ModifiedBlock):
            raise RuntimeError(f"{family}[{layer_id}] is already patched")
        blocks[layer_id] = ModifiedBlock(blocks[layer_id], family, layer_id, controller)
    return selected


def restore_flux2_blocks(transformer: torch.nn.Module) -> list[tuple[str, int]]:
    restored: list[tuple[str, int]] = []
    for family in ("transformer_blocks", "single_transformer_blocks"):
        blocks = getattr(transformer, family, None)
        if not isinstance(blocks, torch.nn.ModuleList):
            continue
        for layer_id, block in enumerate(blocks):
            if isinstance(block, ModifiedBlock):
                blocks[layer_id] = block.block
                restored.append((family, layer_id))
    return restored

