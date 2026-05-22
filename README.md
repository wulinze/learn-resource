# Tile Kernels

Tile Kernels is a collection of optimized GPU kernels for LLM workloads, built
with [TileLang](https://github.com/tile-ai/tilelang). The repository contains
low-level TileLang kernels, PyTorch-facing wrappers, modeling helpers,
reference implementations, correctness tests, benchmarks, and learning puzzles.

Most kernels are written for high-throughput training and inference paths. Some
have been used in internal scenarios, but this repository is still alpha
software and the APIs are being refined.

## Kernel Families

| Area | What it contains |
| --- | --- |
| Transpose | Tiled 2D and batched transpose kernels. |
| Quantization | Per-token, per-block, and per-channel FP8/FP4/E5M6 casts, scale-factor helpers, dequantization, and SwiGLU fusion. |
| MoE routing | Top-k gate kernels, grouped routing, expert mapping, fused expand/reduce, TP masking, and routing-weight normalization. |
| Engram | N-gram hash indexing, Engram gate forward/backward kernels, fused RMSNorm weighting, and weight-gradient reduction. |
| Manifold HyperConnection | mHC residual expansion, pre/post mix kernels, Sinkhorn normalization, fused inference pre-processing, and multilayer recompute. |
| Modeling | `torch.autograd.Function` wrappers that compose low-level kernels into trainable Engram and mHC layers. |
| References | PyTorch reference implementations used for validation and debugging. |

## Requirements

- Python 3.10 or higher
- PyTorch 2.10 or higher
- TileLang 0.1.9 or higher
- NVIDIA SM90 or SM100 architecture GPU
- CUDA Toolkit 13.1 or higher

## Installation

Install a local development checkout:

```bash
pip install -e ".[dev]"
```

Install the published package:

```bash
pip install tile-kernels
```

## Quick Example

```python
import torch
import tile_kernels

x = torch.randn((4096, 7168), dtype=torch.bfloat16, device="cuda")

data, scale = tile_kernels.quant.per_token_cast(
    x,
    fmt="e4m3",
    num_per_channels=128,
    round_sf=True,
)

restored = tile_kernels.quant.per_token_cast_back(
    (data, scale),
    fmt="bf16",
    num_per_channels=128,
)
```

## Testing

Run a correctness test file:

```bash
pytest tests/transpose/test_transpose.py -n 4
```

Run correctness and benchmarks for one file:

```bash
pytest tests/transpose/test_transpose.py --run-benchmark
```

Run a broader pressure test:

```bash
TK_FULL_TEST=1 pytest -n 4 --count 2
```

Run a TileLang puzzle test:

```bash
pytest puzzles/levels/l03_reduction/stable_topk/test_stable_topk.py
TK_PUZZLE_IMPL=starter pytest puzzles/levels/l03_reduction/stable_topk/test_stable_topk.py
```

## Documentation

- [TileLang function interfaces](docs/tilelang_function_interfaces.md): public
  wrappers, internal JIT factories, macros, and usage scenarios.
- [TileLang puzzle project](docs/tilelang_puzzle_project.md): learning roadmap
  and puzzle design notes.
- [Puzzles README](puzzles/README.md): how to run answer and starter
  implementations for the standalone exercises.

## Project Structure

```txt
.
├── tile_kernels/
│   ├── quant/       # FP8/FP4/E5M6 casting, scale factors, dequant, SwiGLU fusion
│   ├── moe/         # MoE gating, mapping, fused expand/reduce, TP masking
│   ├── transpose/   # 2D and batched transpose kernels
│   ├── engram/      # Engram hash, gate kernels, fused weights, grad reduction
│   ├── mhc/         # Low-level Manifold HyperConnection kernels
│   ├── modeling/    # Autograd wrappers and higher-level Engram/mHC composition
│   ├── torch/       # PyTorch reference implementations
│   ├── testing/     # Test data, numeric checks, benchmark helpers
│   ├── config.py    # Runtime kernel configuration such as SM count
│   └── utils.py     # Shared utility helpers
├── tests/
│   ├── quant/       # Quantization and fused activation tests
│   ├── moe/         # Routing and fused MoE layout tests
│   ├── transpose/   # Transpose tests and benchmarks
│   ├── engram/      # Engram hash/gate/gradient tests
│   └── mhc/         # Manifold HyperConnection tests
├── puzzles/
│   ├── common/      # Puzzle loading and CUDA helpers
│   └── levels/      # Standalone TileLang learning exercises
├── docs/            # Project notes and generated interface documentation
├── pyproject.toml   # Package metadata and dependency declarations
└── README.md
```

### Package Entry Points

- `tile_kernels.quant`: quantization, dequantization, and fused SwiGLU kernels.
- `tile_kernels.moe`: expert routing, fused mapping, dispatch, and reduction.
- `tile_kernels.transpose`: 2D and batched transpose wrappers.
- `tile_kernels.engram`: low-level Engram kernels.
- `tile_kernels.modeling`: autograd-level Engram and mHC APIs.
- `tile_kernels.torch`: PyTorch reference implementations.
- `tile_kernels.testing`: numeric checks, generators, and benchmark helpers.

## Development Notes

- Set `TK_PRINT_KERNEL_SOURCE=1` to print generated TileLang kernel source from
  most wrapper calls.
- Tests use PyTorch references to check correctness and optional benchmark
  helpers to record throughput or bandwidth.
- Puzzle starter files are intentionally learner-owned and may contain
  incomplete code.

## Acknowledgement

This project is built on [TileLang](https://github.com/tile-ai/tilelang).
Thanks and respect to the TileLang developers.

## License

This repository is released under the [MIT License](LICENSE).

## Citation

```bibtex
@misc{tilekernels,
      title={TileKernels},
      author={Xiangwen Wang, Chenhao Xu, Huanqi Cao, Rui Tian, Weilin Zhao, Kuai Yu and Chenggang Zhao},
      year={2026},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/TileKernels}},
}
```
