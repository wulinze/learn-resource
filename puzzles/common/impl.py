import importlib
import os


def quiet_tilelang() -> None:
    os.environ.setdefault('TILELANG_PRINT_ON_COMPILATION', '0')


def require_cuda():
    import pytest

    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required for TileLang puzzle tests')
    quiet_tilelang()
    return torch


def load_impl(package_name: str):
    """Load a puzzle implementation module.

    By default tests run the checked-in answer, which wraps the production
    TileKernels implementation. Use ``TK_PUZZLE_IMPL=starter`` to run a learner
    implementation from the puzzle's ``starter.py`` instead.
    """
    impl_name = os.getenv('TK_PUZZLE_IMPL', 'answer')
    if impl_name not in ('answer', 'starter'):
        raise ValueError("TK_PUZZLE_IMPL must be either 'answer' or 'starter'")
    return importlib.import_module(f'{package_name}.{impl_name}')
