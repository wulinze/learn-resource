import torch

from tile_kernels.torch import cast


def per_token_cast_ref(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    round_sf: bool = True,
    use_tma_aligned_col_major_sf: bool = False,
    use_packed_ue8m0: bool = False,
):
    return cast(
        x,
        fmt,
        block_size=(1, num_per_channels),
        round_sf=round_sf,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )

