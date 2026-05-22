# L04: Per-Token Cast

Implement per-token FP8 casting with scale-factor output.

## Task

Write `per_token_cast(...)` in `starter.py`.

Minimal required configuration:

- input `x`: contiguous or row-strided BF16/FP32 CUDA tensor, shape `[num_tokens, hidden]`
- `fmt='e4m3'`
- `num_per_channels in (32, 64, 128)`
- no input scale factors

Output:

- `(out, out_sf)` where `out` is FP8 E4M3 and `out_sf` is the dequant scale factor

## Production Answer

`answer.py` wraps `tile_kernels.quant.per_token_cast`.

Run:

```bash
pytest puzzles/levels/l04_quantization/per_token_cast/test_per_token_cast.py
TK_PUZZLE_IMPL=starter pytest puzzles/levels/l04_quantization/per_token_cast/test_per_token_cast.py
```

