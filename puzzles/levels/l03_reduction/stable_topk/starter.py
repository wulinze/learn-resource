import os

from tile_kernels.moe import topk_gate_kernel
import torch
import tilelang
from tilelang import language as T

from tile_kernels.utils import align

@T.jit
def topk_gate_impl(token_num, num_experts: int, num_topk: int):
    num_threads = 256
    num_aligned_experts = align(num_experts, num_threads)

    @T.prim_func
    def topk_gate_kernel(
        scores: T.Tensor[(token_num, num_experts), T.float32],
        topk_idx: T.Tensor[(token_num, num_topk), T.int32],
    ):
        scores_frag = T.alloc_frag((num_aligned_experts), T.float32)
        max_frag = T.alloc_frag((1), T.float32)
        topk_frag = T.alloc_frag((num_topk), T.int32)
        idx_reducer = T.alloc_reducer((1), T.int32, 'min', replication='all')
        idx_frag = T.alloc_frag((num_aligned_experts), T.int32)

        with T.Kernel(token_num, threads=num_threads) as pid:
            # TODO: Implement this kernel.
            # Hint: You can refer to the implementation in `tile_kernels/moe/topk_gate_kernel.py`.
            for i in T.Parallel(num_aligned_experts):
                if i < num_experts:
                    scores_frag[i] = scores[pid, i]
                else:
                    scores_frag[i] = T.min_value(T.float32)

            for i in T.Parallel(num_aligned_experts):
                idx_frag[i] = i
                
            for i in T.Unroll(num_topk):
                T.reduce_max(scores_frag, max_frag)
                T.fill(idx_reducer, T.max_value(T.int32))
                for j in T.Parallel(num_aligned_experts):
                    if scores_frag[j] == max_frag[0]:
                        idx_reducer[0] = min(idx_frag[j], idx_reducer[0])
                T.finalize_reducer(idx_reducer)
                for j in T.Parallel(num_aligned_experts):
                    if idx_frag[j] == idx_reducer[0]:
                        scores_frag[j] = T.min_value(T.float32)

                T.copy(max_frag[0], topk_frag[i])
        
            T.copy(topk_frag, topk_idx[pid, 0:num_topk])

    return topk_gate_kernel


def topk_gate(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    """Implement this function with a TileLang kernel."""
    raise NotImplementedError('Implement stable top-k in starter.py')
