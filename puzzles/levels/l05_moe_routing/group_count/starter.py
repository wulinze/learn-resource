import os

import torch
import tilelang
from tilelang import language as T

from tile_kernels.config import get_num_sms
from tile_kernels.utils import align


def group_count(group_idx: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Implement this function with a TileLang kernel."""
    raise NotImplementedError('Implement group_count in starter.py')
