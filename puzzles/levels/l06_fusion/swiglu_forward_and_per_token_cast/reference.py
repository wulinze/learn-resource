from typing import Optional

import torch

from tile_kernels.torch import cast, swiglu_forward


def swiglu_forward_and_per_token_cast_ref(
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
    del sf_clamp_min
    out = swiglu_forward(x, pos_to_token_topk, topk_weights, swiglu_clamp_value, clamped_count)
    if pos_to_expert is not None:
        out = out.masked_fill((pos_to_expert == -1).unsqueeze(-1), 0)
    return cast(
        out,
        fmt,
        block_size=(1, num_per_channels),
        round_sf=round_sf,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )

