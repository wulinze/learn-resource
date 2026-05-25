import os

import torch
import tilelang
from tilelang import language as T


@T.jit
def batched_transpose_impl(shape_x_mod_128, shape_y_mod_128):
    assert shape_x_mod_128 in (0, 64) and shape_y_mod_128 in (0, 64)
    shape_x = T.dynamic("shape_x")
    shape_y = T.dynamic("shape_y")
    stride_x = T.dynamic("stride_x")
    num_batches = T.dynamic("num_batches")
    
    num_thread = 256
    block_x = 128 if shape_x_mod_128 == 0 else 64
    block_y = 128 if shape_y_mod_128 == 0 else 64
    block_k = 4
    thread_per_row = block_y // block_k
    thread_per_col = num_thread // thread_per_row // block_k

    @T.prim_func
    def transpose(x: T.StridedTensor, out: T.Tensor):
        x : T.StridedTensor((num_batches, shape_x, shape_y), (shape_x * stride_x, stride_x, 1))
        out : T.Tensor((num_batches, shape_y, shape_x))
        # TODO: Implement the batched transpose kernel
        with T.Kernel(shape_x // block_x, shape_y // block_y, num_thread, thread_num=num_thread) as (pid_x, pid_y, pid_batch):
            tid = T.get_block_binding()
            row = tid // thread_per_row
            col = tid % thread_per_row

            tmp_row = T.alloc_local((block_k,), dtype=x.dtype)
            tmp = T.alloc_local((block_k, block_k), dtype=x.dtype)
            for i in T.unroll(block_x // block_k // (num_thread // thread_per_row)):
                i_ = i * (num_thread // thread_per_row) + row
                for j in T.unroll(block_k):
                    for k in T.vectorized(block_k):
                        tmp_row[k] = x[pid_batch, pid_x * block_x + i_ * block_k + j, pid_y * block_y + col * block_k + k]
                    for k in T.unroll(block_k):
                        tmp[k, j] = tmp_row[k]

            # Copy into shared memory
            for j in T.unroll(block_k):
                swizzle_j = (j + tid // (8 // dtype.bytes)) % block_k
                for k in T.vectorized(block_k):
                    out_shared[col * block_k + swizzle_j, i * block_k + k] = tmp[swizzle_j, k]

            T.sync_threads()
            # Write into output
            for i, j in T.Parallel(block_y, block_x, loop_layout=loop_layout):
                out[pid_batch, pid_y * block_y + i, pid_x * block_x + j] = out_shared[i, j]

def batched_transpose(x: torch.Tensor) -> torch.Tensor:
    """Implement this function with a TileLang kernel."""
    raise NotImplementedError('Implement batched_transpose in starter.py')


def transpose(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2
    return batched_transpose(x.unsqueeze(0)).squeeze(0)
