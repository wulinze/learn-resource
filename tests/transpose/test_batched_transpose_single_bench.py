"""Single-operator benchmark for batched transpose.

Configure cases with:

- ``TK_BT_BENCH_BATCHES=8,32``
- ``TK_BT_BENCH_SHAPES=4096x2048,8192x4096``
- ``TK_BT_BENCH_DTYPES=bf16,fp32,e4m3``

Example:

```
TK_BT_BENCH_BATCHES=8 TK_BT_BENCH_SHAPES=4096x2048 \
  pytest tests/transpose/test_batched_transpose_single_bench.py \
  --run-benchmark -m benchmark --ncu-profile \
  --ncu-kernel-name regex:batched_transpose --ncu-launch-count 1
```
"""

import os

import pytest
import torch

import tile_kernels
from tile_kernels.testing.bench import dtype_to_str, make_param_id
from tile_kernels.testing.numeric import count_bytes


_DTYPES = {
    'bf16': torch.bfloat16,
    'fp32': torch.float32,
    'e4m3': torch.float8_e4m3fn,
}

_NCU_REEXEC_ENV = 'TK_NCU_REEXEC'


def _parse_int_list(env_name: str, default: str) -> list[int]:
    values = []
    for value in os.getenv(env_name, default).split(','):
        value = value.strip()
        if value:
            values.append(int(value))
    if not values:
        raise ValueError(f'{env_name} must contain at least one integer')
    return values


def _parse_shape_list(env_name: str, default: str) -> list[tuple[int, int]]:
    shapes = []
    for value in os.getenv(env_name, default).split(','):
        value = value.strip().lower()
        if not value:
            continue
        if 'x' not in value:
            raise ValueError(f'{env_name} entries must use MxN format, got {value!r}')
        shape_x, shape_y = (int(part.strip()) for part in value.split('x', 1))
        if shape_x % 64 != 0 or shape_y % 64 != 0:
            raise ValueError(f'{env_name} entries must be divisible by 64, got {shape_x}x{shape_y}')
        shapes.append((shape_x, shape_y))
    if not shapes:
        raise ValueError(f'{env_name} must contain at least one MxN shape')
    return shapes


def _parse_dtype_list(env_name: str, default: str) -> list[str]:
    dtypes = []
    for value in os.getenv(env_name, default).split(','):
        value = value.strip().lower()
        if not value:
            continue
        if value not in _DTYPES:
            raise ValueError(f'{env_name} supports {sorted(_DTYPES)}, got {value!r}')
        dtypes.append(value)
    if not dtypes:
        raise ValueError(f'{env_name} must contain at least one dtype')
    return dtypes


def make_batched_transpose_params() -> list[dict]:
    batches = _parse_int_list('TK_BT_BENCH_BATCHES', '8,32')
    shapes = _parse_shape_list('TK_BT_BENCH_SHAPES', '4096x2048')
    dtypes = _parse_dtype_list('TK_BT_BENCH_DTYPES', 'bf16')

    return [
        {'batch': batch, 'shape_x': shape_x, 'shape_y': shape_y, 'dtype': dtype}
        for batch in batches
        for shape_x, shape_y in shapes
        for dtype in dtypes
    ]


def _make_input(params: dict) -> torch.Tensor:
    dtype = _DTYPES[params['dtype']]
    x = torch.randn(
        (params['batch'], params['shape_x'], params['shape_y']),
        dtype=torch.bfloat16,
        device='cuda',
    )
    return x.to(dtype) if dtype == torch.float8_e4m3fn else x.to(dtype)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', make_batched_transpose_params(), ids=make_param_id)
def test_batched_transpose_single_benchmark(benchmark_timer, benchmark_record, params):
    x = _make_input(params)
    out = tile_kernels.transpose.batched_transpose(x)
    num_bytes = count_bytes(x, out)

    if os.environ.get(_NCU_REEXEC_ENV) == '1':
        for _ in range(3):
            tile_kernels.transpose.batched_transpose(x)
        torch.cuda.synchronize()
        tile_kernels.transpose.batched_transpose(x)
        torch.cuda.synchronize()
        return

    t_us = benchmark_timer(lambda: tile_kernels.transpose.batched_transpose(x))

    benchmark_record(
        kernel='batched_transpose_single',
        operation='fwd',
        params={**params, 'dtype': dtype_to_str(_DTYPES[params['dtype']])},
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )
