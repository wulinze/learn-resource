import pytest

from puzzles.common import load_impl, require_cuda
from tile_kernels.testing.numeric import assert_equal

from .reference import stable_topk_ref


@pytest.mark.parametrize('num_tokens', [1, 17, 129])
@pytest.mark.parametrize('num_experts,num_topk', [(16, 4), (72, 6), (128, 8)])
def test_stable_topk(num_tokens, num_experts, num_topk):
    torch = require_cuda()
    impl = load_impl(__package__)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda')
    out = impl.topk_gate(scores, num_topk)
    ref = stable_topk_ref(scores, num_topk)
    assert_equal(out, ref)


def test_stable_topk_tie_breaks_to_smaller_index():
    torch = require_cuda()
    impl = load_impl(__package__)

    scores = torch.zeros((4, 16), dtype=torch.float32, device='cuda')
    scores[:, 3] = 7.0
    scores[:, 5] = 7.0
    scores[:, 9] = 7.0

    out = impl.topk_gate(scores, 3)
    ref = stable_topk_ref(scores, 3)
    assert_equal(out, ref)

