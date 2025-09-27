"""Utility helpers required by the slimmed DINOv3 backbone."""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn


def cat_keep_shapes(x_list: Sequence[Tensor]) -> Tuple[Tensor, List[torch.Size], List[int]]:
    """Flatten tensors for batched processing while recording their original shapes."""

    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def uncat_with_shapes(flattened: Tensor, shapes: Sequence[torch.Size], num_tokens: Sequence[int]) -> List[Tensor]:
    """Inverse of :func:`cat_keep_shapes` that restores the original tensor shapes."""

    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    shapes_adjusted = [shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes]
    return [out.reshape(shape) for out, shape in zip(outputs_splitted, shapes_adjusted)]


def named_apply(
    fn: Callable[[nn.Module, str], None],
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    """Apply ``fn`` to every submodule, mirroring :func:`torch.nn.Module.apply`."""

    if not depth_first and include_root:
        fn(module, name)
    for child_name, child_module in module.named_children():
        child_name = f"{name}.{child_name}" if name else child_name
        named_apply(fn, child_module, child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module, name)
    return module


__all__ = ["cat_keep_shapes", "uncat_with_shapes", "named_apply"]
