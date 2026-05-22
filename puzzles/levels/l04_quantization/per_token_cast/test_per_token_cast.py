import pytest

from puzzles.common import load_impl, require_cuda
from tile_kernels.testing.numeric import assert_equal
from tile_kernels.testing.quant import clear_unused_sf

from .reference import per_token_cast_ref


@pytest.mark.parametrize('dtype', ['bfloat16', 'float32'])
@pytest.mark.parametrize('num_tokens,hidden,num_per_channels', [(1, 128, 32), (17, 256, 64), (129, 512, 128)])
def test_per_token_cast(dtype, num_tokens, hidden, num_per_channels):
    torch = require_cuda()
    impl = load_impl(__package__)

    x = torch.randn((num_tokens, hidden), dtype=getattr(torch, dtype), device='cuda')
    kwargs = dict(
        x=x,
        fmt='e4m3',
        num_per_channels=num_per_channels,
        round_sf=True,
        use_tma_aligned_col_major_sf=False,
        use_packed_ue8m0=False,
    )

    out, out_sf = impl.per_token_cast(**kwargs)
    ref, ref_sf = per_token_cast_ref(**kwargs)
    assert_equal(out, ref)
    assert_equal(out_sf, ref_sf)


def test_per_token_cast_non_contiguous_rows():
    torch = require_cuda()
    impl = load_impl(__package__)

    base = torch.randn((33, 512), dtype=torch.bfloat16, device='cuda')
    x = base[:, :256]
    assert not x.is_contiguous()

    kwargs = dict(
        x=x,
        fmt='e4m3',
        num_per_channels=64,
        round_sf=True,
        use_tma_aligned_col_major_sf=False,
        use_packed_ue8m0=False,
    )

    out, out_sf = impl.per_token_cast(**kwargs)
    ref, ref_sf = per_token_cast_ref(**kwargs)
    assert_equal(out, ref)
    assert_equal(out_sf, ref_sf)


def test_per_token_cast_packed_scale_factor_layout():
    torch = require_cuda()
    impl = load_impl(__package__)

    hidden = 256
    num_per_channels = 128
    x = torch.randn((65, hidden), dtype=torch.bfloat16, device='cuda')
    kwargs = dict(
        x=x,
        fmt='e4m3',
        num_per_channels=num_per_channels,
        round_sf=True,
        use_tma_aligned_col_major_sf=True,
        use_packed_ue8m0=True,
    )

    out, out_sf = impl.per_token_cast(**kwargs)
    ref, ref_sf = per_token_cast_ref(**kwargs)
    out_sf = clear_unused_sf(out_sf, hidden, num_per_channels)
    ref_sf = clear_unused_sf(ref_sf, hidden, num_per_channels)
    assert_equal(out, ref)
    assert_equal(out_sf, ref_sf)

