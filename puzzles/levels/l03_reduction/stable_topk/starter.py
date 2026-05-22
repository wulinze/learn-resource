import os

import torch
import tilelang
from tilelang import language as T

from tile_kernels.utils import align


def topk_gate(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    """Implement this function with a TileLang kernel."""
    raise NotImplementedError('Implement stable top-k in starter.py')
