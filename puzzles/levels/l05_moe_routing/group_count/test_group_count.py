import pytest

from puzzles.common import load_impl, require_cuda
from tile_kernels.testing.numeric import assert_equal

from .reference import group_count_ref


@pytest.mark.parametrize('num_tokens,num_topk,num_groups', [(1, 2, 8), (97, 6, 72), (257, 8, 128)])
def test_group_count(num_tokens, num_topk, num_groups):
    torch = require_cuda()
    impl = load_impl(__package__)

    group_idx = torch.randint(0, num_groups, (num_tokens, num_topk), dtype=torch.int64, device='cuda')
    group_idx[group_idx % 11 == 0] = -1

    out = impl.group_count(group_idx, num_groups)
    ref = group_count_ref(group_idx, num_groups)
    assert_equal(out, ref)

