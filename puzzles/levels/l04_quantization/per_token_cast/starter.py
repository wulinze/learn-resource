import math
import os
from typing import Optional, Union

import torch
import tilelang
from tilelang import language as T

from tile_kernels.quant.common import *
from tile_kernels.utils import align


def per_token_cast(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    round_sf: bool = True,
    use_tma_aligned_col_major_sf: bool = False,
    use_packed_ue8m0: bool = False,
):
    """Implement this function with a TileLang kernel."""
    raise NotImplementedError('Implement per_token_cast in starter.py')
