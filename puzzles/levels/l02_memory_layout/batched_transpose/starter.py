import os

import torch
import tilelang
from tilelang import language as T


@T.jit
def batched_transpose_impl(x: torch.Tensor):
    @T.prim_func
    def transpose(x: T.Tensor):
        # TODO: Implement the batched transpose kernel
        with T.Kerner

def batched_transpose(x: torch.Tensor) -> torch.Tensor:
    """Implement this function with a TileLang kernel."""
    raise NotImplementedError('Implement batched_transpose in starter.py')


def transpose(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2
    return batched_transpose(x.unsqueeze(0)).squeeze(0)
