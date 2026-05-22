import pytest

from puzzles.common import load_impl, require_cuda
from tile_kernels.testing.numeric import assert_equal

from .reference import batched_transpose_ref, transpose_ref


@pytest.mark.parametrize('dtype', ['bfloat16', 'float32'])
@pytest.mark.parametrize('shape', [(1, 64, 64), (2, 128, 64), (3, 64, 128)])
def test_batched_transpose(dtype, shape):
    torch = require_cuda()
    impl = load_impl(__package__)

    dtype_obj = getattr(torch, dtype)
    x = torch.randn(shape, dtype=dtype_obj, device='cuda')

    out = impl.batched_transpose(x)
    ref = batched_transpose_ref(x)
    assert_equal(out, ref)


def test_batched_transpose_strided_input():
    torch = require_cuda()
    impl = load_impl(__package__)

    base = torch.randn((2, 128, 128 * 2), dtype=torch.bfloat16, device='cuda')
    x = base[:, :, :128]
    assert x.stride(-2) % 4 == 0
    assert not x.is_contiguous()

    out = impl.batched_transpose(x)
    ref = batched_transpose_ref(x)
    assert_equal(out, ref)


def test_transpose_2d_wrapper():
    torch = require_cuda()
    impl = load_impl(__package__)

    x = torch.randn((128, 64), dtype=torch.bfloat16, device='cuda')
    out = impl.transpose(x)
    ref = transpose_ref(x)
    assert_equal(out, ref)

