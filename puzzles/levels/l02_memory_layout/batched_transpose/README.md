# L02: Batched Transpose

Implement a TileLang batched transpose kernel.

## Task

Write `batched_transpose(x)` in `starter.py`.

Input:

- `x`: CUDA tensor with shape `[batch, shape_x, shape_y]`
- `x.stride(-1) == 1`
- `shape_x` and `shape_y` are divisible by 64

Output:

- contiguous tensor with shape `[batch, shape_y, shape_x]`

## Correctness

Match `torch.transpose(x, 1, 2).contiguous()` bit-for-bit.

## Production Answer

`answer.py` wraps `tile_kernels.transpose.batched_transpose`.

Run:

```bash
pytest puzzles/levels/l02_memory_layout/batched_transpose/test_batched_transpose.py
TK_PUZZLE_IMPL=starter pytest puzzles/levels/l02_memory_layout/batched_transpose/test_batched_transpose.py
```

