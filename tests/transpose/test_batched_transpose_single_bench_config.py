import pytest

from tests.transpose.test_batched_transpose_single_bench import make_batched_transpose_params


def test_make_batched_transpose_params_parses_batch_and_matrix_sizes(monkeypatch):
    monkeypatch.setenv('TK_BT_BENCH_BATCHES', '1,8')
    monkeypatch.setenv('TK_BT_BENCH_SHAPES', '512x1024,4096x2048')
    monkeypatch.setenv('TK_BT_BENCH_DTYPES', 'bf16,fp32')

    params = make_batched_transpose_params()

    assert params == [
        {'batch': 1, 'shape_x': 512, 'shape_y': 1024, 'dtype': 'bf16'},
        {'batch': 1, 'shape_x': 512, 'shape_y': 1024, 'dtype': 'fp32'},
        {'batch': 1, 'shape_x': 4096, 'shape_y': 2048, 'dtype': 'bf16'},
        {'batch': 1, 'shape_x': 4096, 'shape_y': 2048, 'dtype': 'fp32'},
        {'batch': 8, 'shape_x': 512, 'shape_y': 1024, 'dtype': 'bf16'},
        {'batch': 8, 'shape_x': 512, 'shape_y': 1024, 'dtype': 'fp32'},
        {'batch': 8, 'shape_x': 4096, 'shape_y': 2048, 'dtype': 'bf16'},
        {'batch': 8, 'shape_x': 4096, 'shape_y': 2048, 'dtype': 'fp32'},
    ]


def test_make_batched_transpose_params_rejects_unaligned_shapes(monkeypatch):
    monkeypatch.setenv('TK_BT_BENCH_SHAPES', '513x1024')

    with pytest.raises(ValueError, match='divisible by 64'):
        make_batched_transpose_params()
