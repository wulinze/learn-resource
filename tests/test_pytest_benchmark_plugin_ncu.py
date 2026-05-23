from tests.pytest_benchmark_plugin import _build_ncu_command, _make_ncu_env, _sanitize_ncu_name


def test_build_ncu_command_uses_basic_profile_and_current_python():
    cmd = _build_ncu_command(
        ncu_path='/usr/local/cuda/bin/ncu',
        python_executable='/tmp/venv/bin/python',
        pytest_args=('tests/quant/test_per_token_cast.py', '--run-benchmark', '-m', 'benchmark'),
        output_base='ncu-reports/per-token',
        ncu_set='basic',
        launch_count=1,
        launch_skip=0,
        kernel_name=None,
        kernel_name_base='demangled',
        target_processes='application-only',
        print_summary='per-kernel',
    )

    assert cmd == [
        '/usr/local/cuda/bin/ncu',
        '-f',
        '-o',
        'ncu-reports/per-token',
        '--set',
        'basic',
        '--target-processes',
        'application-only',
        '--launch-count',
        '1',
        '--kernel-name-base',
        'demangled',
        '--print-summary',
        'per-kernel',
        '/tmp/venv/bin/python',
        '-m',
        'pytest',
        'tests/quant/test_per_token_cast.py',
        '--run-benchmark',
        '-m',
        'benchmark',
    ]


def test_build_ncu_command_adds_optional_kernel_filter_and_launch_skip():
    cmd = _build_ncu_command(
        ncu_path='ncu',
        python_executable='python',
        pytest_args=('tests/transpose/test_transpose.py',),
        output_base='ncu-reports/transpose',
        ncu_set='roofline',
        launch_count=2,
        launch_skip=3,
        kernel_name='regex:transpose',
        kernel_name_base='function',
        target_processes='all',
        print_summary='none',
    )

    assert '--launch-skip' in cmd
    assert cmd[cmd.index('--launch-skip') + 1] == '3'
    assert '--kernel-name' in cmd
    assert cmd[cmd.index('--kernel-name') + 1] == 'regex:transpose'
    assert '--set' in cmd
    assert cmd[cmd.index('--set') + 1] == 'roofline'
    assert '--target-processes' in cmd
    assert cmd[cmd.index('--target-processes') + 1] == 'all'


def test_make_ncu_env_marks_child_pytest_to_avoid_recursive_reexec():
    env = _make_ncu_env({'PATH': '/bin'})

    assert env['PATH'] == '/bin'
    assert env['TK_NCU_REEXEC'] == '1'


def test_sanitize_ncu_name_keeps_report_names_filesystem_friendly():
    assert _sanitize_ncu_name('tests/quant/test_per_token_cast.py::case[a=1,b=2]') == 'tests-quant-test_per_token_cast.py-case-a-1-b-2'
    assert _sanitize_ncu_name('') == 'pytest'
