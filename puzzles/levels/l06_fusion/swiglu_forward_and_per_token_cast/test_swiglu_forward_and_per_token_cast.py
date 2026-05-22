import pytest

from puzzles.common import load_impl, require_cuda
from tile_kernels.testing.numeric import assert_equal
from tile_kernels.testing.quant import clear_unused_sf

from .reference import swiglu_forward_and_per_token_cast_ref


@pytest.mark.parametrize('dtype', ['bfloat16', 'float32'])
@pytest.mark.parametrize('num_tokens,hidden,num_per_channels', [(1, 128, 128), (17, 256, 128), (65, 256, 256)])
def test_swiglu_forward_and_per_token_cast(dtype, num_tokens, hidden, num_per_channels):
    torch = require_cuda()
    impl = load_impl(__package__)

    x = torch.randn((num_tokens, hidden * 2), dtype=getattr(torch, dtype), device='cuda')
    kwargs = dict(
        x=x,
        fmt='e4m3',
        num_per_channels=num_per_channels,
        use_tma_aligned_col_major_sf=False,
        round_sf=True,
        use_packed_ue8m0=False,
    )

    out, out_sf = impl.swiglu_forward_and_per_token_cast(**kwargs)
    ref, ref_sf = swiglu_forward_and_per_token_cast_ref(**kwargs)
    assert_equal(out, ref)
    assert_equal(out_sf, ref_sf)


def test_swiglu_forward_and_per_token_cast_with_mask_and_packed_sf():
    torch = require_cuda()
    impl = load_impl(__package__)

    num_tokens = 65
    hidden = 256
    num_per_channels = 128
    x = torch.randn((num_tokens, hidden * 2), dtype=torch.bfloat16, device='cuda')
    pos_to_expert = torch.arange(num_tokens, dtype=torch.int32, device='cuda')
    pos_to_expert[::7] = -1
    kwargs = dict(
        x=x,
        fmt='e4m3',
        num_per_channels=num_per_channels,
        pos_to_expert=pos_to_expert,
        use_tma_aligned_col_major_sf=True,
        round_sf=True,
        use_packed_ue8m0=True,
    )

    out, out_sf = impl.swiglu_forward_and_per_token_cast(**kwargs)
    ref, ref_sf = swiglu_forward_and_per_token_cast_ref(**kwargs)
    mask = (pos_to_expert == -1).unsqueeze(-1)
    out_float = out.float().masked_fill(mask, 0)
    ref_float = ref.float().masked_fill(mask, 0)
    out_sf = clear_unused_sf(out_sf.masked_fill(mask, 0), hidden, num_per_channels)
    ref_sf = clear_unused_sf(ref_sf.masked_fill(mask, 0), hidden, num_per_channels)
    assert_equal(out_float, ref_float)
    assert_equal(out_sf, ref_sf)

