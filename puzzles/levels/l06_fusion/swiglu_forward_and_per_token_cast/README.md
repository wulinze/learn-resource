# L06: SwiGLU Forward + Per-Token Cast

Implement a fused SwiGLU forward and per-token FP8 cast.

## Task

Write `swiglu_forward_and_per_token_cast(...)` in `starter.py`.

Minimal required configuration:

- input `x`: contiguous BF16/FP32 CUDA tensor, shape `[num_tokens, hidden * 2]`
- `fmt='e4m3'`
- `num_per_channels in (128, hidden)`
- no routing weights in the first milestone

Output:

- `(out, out_sf)` where `out` is FP8 E4M3 and `out_sf` is the dequant scale factor

## Production Answer

`answer.py` wraps `tile_kernels.quant.swiglu_forward_and_per_token_cast`.

Run:

```bash
pytest puzzles/levels/l06_fusion/swiglu_forward_and_per_token_cast/test_swiglu_forward_and_per_token_cast.py
TK_PUZZLE_IMPL=starter pytest puzzles/levels/l06_fusion/swiglu_forward_and_per_token_cast/test_swiglu_forward_and_per_token_cast.py
```

