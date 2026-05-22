import os
from typing import Optional

import torch
import tilelang
from tilelang import language as T

from tile_kernels.config import get_num_sms
from tile_kernels.quant.common import *
from tile_kernels.utils import is_power_of_two


def swiglu_forward_and_per_token_cast(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    pos_to_token_topk: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    pos_to_expert: Optional[torch.Tensor] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = True,
    use_packed_ue8m0: bool = False,
    swiglu_clamp_value: Optional[float] = None,
    clamped_count: Optional[torch.Tensor] = None,
    sf_clamp_min: Optional[float] = None,
):
    """Implement this function with a TileLang kernel."""
    raise NotImplementedError('Implement fused SwiGLU + per-token cast in starter.py')
