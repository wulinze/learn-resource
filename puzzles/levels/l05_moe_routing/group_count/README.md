# L05: Group Count

Implement expert/group index counting for MoE routing.

## Task

Write `group_count(group_idx, num_groups)` in `starter.py`.

Input:

- `group_idx`: contiguous int64 CUDA tensor with shape `[num_tokens, num_topk]`
- values are in `[-1, num_groups)`
- `-1` means padding and must not be counted

Output:

- int32 tensor with shape `[num_groups]`

## Production Answer

`answer.py` wraps `tile_kernels.moe.group_count`.

Run:

```bash
pytest puzzles/levels/l05_moe_routing/group_count/test_group_count.py
TK_PUZZLE_IMPL=starter pytest puzzles/levels/l05_moe_routing/group_count/test_group_count.py
```

