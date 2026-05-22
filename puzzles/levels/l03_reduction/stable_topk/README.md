# L03: Stable Top-K

Implement stable top-k expert selection with TileLang reductions.

## Task

Write `topk_gate(scores, num_topk)` in `starter.py`.

Input:

- `scores`: contiguous float32 CUDA tensor with shape `[num_tokens, num_experts]`
- `num_topk`: number of selected experts per token

Output:

- int64 tensor with shape `[num_tokens, num_topk]`

## Correctness

Match stable descending sort. When scores tie, return the smaller expert index.

## Production Answer

`answer.py` wraps `tile_kernels.moe.topk_gate`.

Run:

```bash
pytest puzzles/levels/l03_reduction/stable_topk/test_stable_topk.py
TK_PUZZLE_IMPL=starter pytest puzzles/levels/l03_reduction/stable_topk/test_stable_topk.py
```

