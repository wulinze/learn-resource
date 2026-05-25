# TileLang 接口速查

这份文档面向写 `puzzles/levels/*/starter.py` 和阅读 `tile_kernels/*_kernel.py` 的场景。它按 TileLang DSL 的概念组织，而不是按 `transpose`、`quant`、`moe` 等业务模块组织。

当前项目依赖 `tilelang>=0.1.9`。下面只覆盖本仓库实际使用到的接口和写法。

## 最小 Kernel 结构

TileLang kernel 通常分三层：

1. Python wrapper：接收 PyTorch tensor，检查 shape/stride/dtype，创建输出 tensor。
2. `@tilelang.jit` kernel factory：接收编译期参数，返回一个 JIT kernel。
3. `@T.prim_func`：真正的设备端 kernel 逻辑。

典型结构：

```python
import torch
import tilelang
from tilelang import language as T


@tilelang.jit
def get_kernel(hidden: int, dtype: T.dtype):
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def kernel(
        x: T.Tensor[(num_tokens, hidden), dtype],
        out: T.Tensor[(num_tokens, hidden), dtype],
    ):
        with T.Kernel(T.ceildiv(num_tokens, 128), threads=128) as pid:
            for i in T.Parallel(128):
                row = pid * 128 + i
                if row < num_tokens:
                    out[row, 0] = x[row, 0]

    return kernel


def wrapper(x: torch.Tensor) -> torch.Tensor:
    num_tokens, hidden = x.shape
    out = torch.empty_like(x)
    kernel = get_kernel(hidden, T.dtype(x.dtype))
    kernel(x, out)
    return out
```

### 编译期参数和运行期参数

`@tilelang.jit` 的 Python 参数是编译期常量，适合放 tile size、hidden、dtype、topk 等会影响代码生成的值。

`T.dynamic('name')` 是运行期 symbol，通常来自实际 tensor shape/stride。它可以写进 `T.Tensor` shape 和 `T.Kernel` grid。

```python
@tilelang.jit
def get_kernel(hidden: int):
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def kernel(x: T.Tensor[(num_tokens, hidden), T.bfloat16]):
        ...
```

## Tensor 类型标注

### `T.Tensor`

连续 tensor 用 `T.Tensor[(shape...), dtype]`。

```python
x: T.Tensor[(num_tokens, hidden), T.bfloat16]
out: T.Tensor[(num_tokens, hidden), T.float32]
```

一维 shape 可以写成：

```python
out: T.Tensor[(num_groups,), T.int32]
```

### `T.StridedTensor`

非连续 tensor 或需要显式 stride 时用 `T.StridedTensor[(shape...), (stride...), dtype]`。

```python
x: T.StridedTensor[
    (num_batches, shape_x, shape_y),
    (shape_x * stride_x, stride_x, 1),
    dtype,
]
```

常见用途：

- 支持 PyTorch 非 contiguous leading dimension。
- 支持 scale factor tensor 的动态 stride。
- 明确最后一维连续，便于 vectorized load/store。

### dtype 写法

常用 dtype：

```python
T.float32
T.float16
T.bfloat16
T.float8_e4m3fn
T.float4_e2m1fn
T.int64
T.int32
T.uint32
T.uint8
```

从 PyTorch dtype 转 TileLang dtype：

```python
T.dtype(x.dtype)
```

## Grid 和线程

### `T.Kernel`

`T.Kernel` 定义 CUDA grid 和每个 block 的线程数。

```python
with T.Kernel(grid_x, threads=128) as pid:
    ...

with T.Kernel(grid_x, grid_y, threads=256) as (pid_x, pid_y):
    ...

with T.Kernel(grid_x, grid_y, grid_z, threads=256) as (pid_x, pid_y, pid_z):
    ...
```

例如 batched transpose：

```python
with T.Kernel(shape_y // block_y, shape_x // block_x, num_batches, threads=256) as (
    pid_y, pid_x, pid_batch
):
    ...
```

这里 `pid_batch` 是 batch 维的 block id。硬件会把所有 block 调度到 SM，不需要手动指定 batch 到哪个 SM。

### `T.get_thread_binding`

获取当前 CUDA block 内的 thread id。

```python
tid = T.get_thread_binding()
row = tid // threads_per_row
col = tid % threads_per_row
```

也可写：

```python
tid = T.get_thread_binding(0)
```

### 手动全局线程 id

适合扫描型 kernel：

```python
thread_idx = T.get_thread_binding()
global_thread_idx = pid * num_threads + thread_idx

for i in T.serial(global_thread_idx, num_tokens, num_blocks * num_threads):
    ...
```

## 内存空间

### `T.alloc_local`

分配线程私有 local/register 数组。

```python
tmp = T.alloc_local((4, 4), T.float32)
idx = T.alloc_local((num_topk,), T.int32)
```

常见用途：

- 每个线程的小向量。
- 临时标量数组。
- vectorized load 的寄存器缓存。

### `T.alloc_var`

分配线程私有标量。

```python
amax = T.alloc_var(T.float32)
counter = T.alloc_var(T.int32)
```

### `T.alloc_shared`

分配 block 内共享内存。

```python
tile = T.alloc_shared((128, 132), T.bfloat16)
counts = T.alloc_shared((align(num_groups, num_threads),), T.int32)
```

常见用途：

- tiled transpose 中做 shared-memory 重排。
- block 内 reduction / histogram。
- 缓存重复读取的输入。

### `T.alloc_fragment`

分配 fragment，通常表示寄存器 tile，并可配合 `T.copy`、`T.reduce_*`、`T.reshape` 使用。

```python
x_frag = T.alloc_fragment((block_m, block_k), T.bfloat16)
amax_frag = T.alloc_fragment((block_m, num_groups), T.float32)
```

相比 `alloc_local`，fragment 更常用于矩阵/tile 级数据搬运和布局标注。

## 数据搬运

### `T.copy`

在 global/shared/fragment/local 之间复制。

```python
T.copy(x[pid_m * block_m, pid_k * block_k], x_frag)
T.copy(x_frag, out[pid_m * block_m, pid_k * block_k])
T.copy(shared_tile, frag)
```

常见参数：

```python
T.copy(src, dst, disable_tma=True)
```

`disable_tma=True` 在本项目里常用于普通 global memory copy，避免走 TMA 路径。

### 切片 copy

```python
T.copy(x[pid * block_m : (pid + 1) * block_m, 0:hidden], frag)
T.copy(out_frag, out[pid_token * block_m, pid_hidden * block_k])
```

如果目标是 fragment/shared，shape 通常由目标 buffer 决定。

## 循环

### `T.Parallel`

生成并行循环，常用于填充 tile、逐元素计算、block 内并行 reduction 前后处理。

```python
for i in T.Parallel(block_m):
    ...

for i, j in T.Parallel(block_m, block_k):
    out_frag[i, j] = x_frag[i, j] * scale
```

可绑定自定义 layout：

```python
layout = T.Fragment((block_y, block_x), forward_fn=layout_fn)

for i, j in T.Parallel(block_y, block_x, loop_layout=layout):
    ...
```

### `T.serial`

生成串行循环。

```python
for i in T.serial(n):
    ...

for i in T.serial(start, stop, step):
    ...
```

常用于：

- 一个线程遍历多个元素。
- 外层 pipeline 或 reduction stage。
- global-thread-stride 扫描。

### `T.unroll`

编译期展开循环。

```python
for j in T.unroll(4):
    ...

for ngram_idx in T.unroll(0, max_ngram_size):
    ...
```

适合小常量循环，例如 top-k、vector lane、固定 tile 子块。

### `T.vectorized`

向量化循环，要求访存连续且对齐条件合理。

```python
for k in T.vectorized(4):
    tmp[k] = x[row, col + k]
```

常见于最后一维连续的 load/store。

### `T.Pipelined`

用于多 stage pipeline。

```python
for pz in T.Pipelined(rms_group_size // hidden_block, num_stages=2):
    ...
```

本仓库主要在 mHC norm/fusion kernel 中使用。

## 同步、清零、断言

### `T.sync_threads`

block 内线程同步。

```python
T.clear(shared_counts)
T.sync_threads()
...
T.sync_threads()
```

写 shared memory 后，如果后续其他线程要读，通常需要同步。

### `T.clear`

将 fragment/shared/local buffer 清零。

```python
T.clear(out_frag)
T.clear(shared_counts)
```

### `T.assume`

告诉编译器某个条件成立，帮助优化。

```python
T.assume(shape_x % block_x == 0)
T.assume(expert_idx < num_groups)
```

它不是面向用户的运行期错误处理。wrapper 里仍应使用 Python `assert` 检查输入。

### `T.device_assert`

设备端断言，用于调试或保护非法输入。

```python
T.device_assert(-1 <= expert_idx < num_groups)
```

## 原子操作

### `T.atomic_add`

用于 histogram、计数、跨线程累加。

```python
if expert_idx >= 0:
    T.atomic_add(out_shared[expert_idx], 1)

T.atomic_add(out[i], out_shared[i])
```

常见模式是先在 shared memory 内累加，再同步后写回 global，减少 global atomic 压力。

## 数学和类型转换

### 类型转换

可以用函数式 dtype cast：

```python
x_f32 = T.float32(x)
idx_i32 = T.int32(idx)
```

也可以用：

```python
T.cast(value, T.float32)
```

### 位级 reinterpret

```python
bits = T.reinterpret(sf, T.uint32)
sf = T.reinterpret(bits << 23, T.float32)
```

常用于自定义浮点格式、scale factor 打包/解包。

### 条件表达式

```python
value = T.if_then_else(cond, a, b)
```

适合表达式级分支。普通 `if` 也可用于 TileLang 条件分支。

### 常用数学函数

```python
T.abs(x)
T.max(a, b)
T.min(a, b)
T.exp(x)
T.rsqrt(x)
T.ceildiv(a, b)
T.max_value(dtype)
```

## Reduction 和 reshape

### `T.reshape`

将 fragment 重新解释成另一个 shape，通常用于 reduction。

```python
x_reshaped = T.reshape(x_frag, [block_m, num_groups, num_per_channels])
```

### `T.reduce_absmax`

按指定维度求绝对值最大值。

```python
T.reduce_absmax(x_reshaped, amax_frag, dim=2)
```

### `T.reduce_max`

按指定维度求最大值。

```python
T.reduce_max(stage2_reshaped, sf_inv_frag, dim=-1)
```

## Layout 标注

### `T.Fragment`

定义 fragment 的 logical index 到线程/local lane 的映射。

```python
def layout_fn(i, j):
    elems = i * block_k + j
    thread = elems // vector_width % num_threads
    local = elems % vector_width
    return thread, local

layout = T.Fragment((block_m, block_k), forward_fn=layout_fn)
```

### `T.annotate_layout`

把 layout 绑定到 fragment 或 shared buffer。

```python
T.annotate_layout({
    x_frag: T.Fragment((block_m, block_k), forward_fn=layout_fn),
})
```

也可用于 shared memory swizzle：

```python
T.annotate_layout({
    x_smem: tilelang.layout.make_swizzled_layout(x_smem),
})
```

## Macro

### `@T.macro`

TileLang macro 是可在 kernel 中内联使用的小函数，适合封装重复 DSL 片段。

```python
@T.macro
def load_scale(sf_tensor: T.Tensor, row: int, col: int):
    return sf_tensor[row, col]
```

本项目常见 macro：

- scale factor 计算：`get_sf_and_inv`
- scale factor load/store：`load_sf`、`store_sf`
- 自定义格式打包/解包：E5M6、E2M1
- top-k/group routing 内部 helper

macro 里可以使用 `T.alloc_var`、`T.reinterpret`、`T.if_then_else` 等 DSL。

## Python Wrapper 约定

TileLang kernel 外层 wrapper 负责：

1. 检查 tensor rank、contiguous、stride、shape 对齐。
2. 将 PyTorch dtype 转为 TileLang dtype。
3. 分配输出 tensor。
4. 调用 JIT kernel。
5. 可选打印生成源码。

常见写法：

```python
assert x.dim() == 2
assert x.stride(-1) == 1
assert hidden % 64 == 0

kernel = get_kernel(hidden, T.dtype(x.dtype))

if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
    print(kernel.get_kernel_source())

out = torch.empty((num_tokens, hidden), dtype=x.dtype, device='cuda')
kernel(x, out)
return out
```

## Pass Config

`@tilelang.jit` 可以带 pass config。

```python
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
    },
)
def get_kernel(...):
    ...
```

本项目常见配置：

| 配置 | 本项目用途 |
| --- | --- |
| `TL_DISABLE_WARP_SPECIALIZED` | 多数 kernel 默认打开，避免 warp specialized lowering 干扰简单 kernel。 |
| `TL_ENABLE_LOWER_LDGSTG_PREDICATED` | 部分 quant kernel 使用，处理 predicated load/store lowering。 |

## 常见 Kernel 模板

### 1D 扫描 / 计数

适合 `group_count`、`aux_fi` 这类任务。

```python
num_threads = 128
num_blocks = num_sms * 2
num_tokens = T.dynamic('num_tokens')

@T.prim_func
def kernel(idx: T.Tensor[(num_tokens, num_topk), T.int64], out: T.Tensor[(num_groups,), T.int32]):
    with T.Kernel(num_blocks, threads=num_threads) as pid:
        tid = T.get_thread_binding()
        gid = pid * num_threads + tid

        shared = T.alloc_shared((align(num_groups, num_threads),), T.int32)
        T.clear(shared)
        T.sync_threads()

        for i in T.serial(gid, num_tokens, num_blocks * num_threads):
            for j in T.unroll(num_topk):
                group = T.int32(idx[i, j])
                if group >= 0:
                    T.atomic_add(shared[group], 1)

        T.sync_threads()
        for i in T.serial(tid, num_groups, num_threads):
            if shared[i] > 0:
                T.atomic_add(out[i], shared[i])
```

### 2D Tiled Elementwise / Cast

适合 quant cast、cast back、简单 transform。

```python
with T.Kernel(T.ceildiv(num_tokens, block_m), T.ceildiv(hidden, block_k), threads=num_threads) as (pid_m, pid_k):
    x_frag = T.alloc_fragment((block_m, block_k), in_dtype)
    out_frag = T.alloc_fragment((block_m, block_k), out_dtype)

    T.copy(x[pid_m * block_m, pid_k * block_k], x_frag, disable_tma=True)

    for i, j in T.Parallel(block_m, block_k):
        out_frag[i, j] = T.cast(x_frag[i, j], out_dtype)

    T.copy(out_frag, out[pid_m * block_m, pid_k * block_k], disable_tma=True)
```

### Shared Memory Transpose

适合 `batched_transpose`、`per_channel_cast_and_transpose`。

```python
with T.Kernel(shape_y // block_y, shape_x // block_x, batch, threads=256) as (pid_y, pid_x, pid_b):
    shared = T.alloc_shared((block_y, block_x + pad), dtype)
    tid = T.get_thread_binding()

    # Load contiguous rows from global into local/register, then write transposed into shared.
    ...
    T.sync_threads()

    # Read shared in transposed order and write contiguous output.
    for i, j in T.Parallel(block_y, block_x):
        out[pid_b, pid_y * block_y + i, pid_x * block_x + j] = shared[i, j]
```

## 阅读现有 Kernel 的顺序

看一个新 TileLang kernel 时，建议按这个顺序：

1. 看 wrapper 的 Python assert，确定输入 shape、stride、dtype 约束。
2. 看 `get_*_kernel(...)` 的参数，区分编译期常量和运行期动态值。
3. 看 `T.Kernel(...)`，算出每个 CUDA block 负责哪块数据。
4. 看 `alloc_*`，区分 local/shared/fragment 的职责。
5. 看 `T.copy`，找 global memory 读写边界。
6. 看 `T.Parallel` / `T.serial` / `T.unroll` / `T.vectorized`，判断并行粒度。
7. 看 `T.sync_threads` 和 atomic，确认 block 内协作方式。
8. 最后看 math 和 dtype cast。

## 写 Puzzle Starter 的建议

先写能对齐测试输入约束的最小版本：

- shape 先按测试要求处理，不要一开始支持所有 ragged case。
- Python wrapper 先做明确 `assert`。
- block size 先选固定常量，例如 64/128。
- 每个 kernel 先确认 grid 映射正确，再优化 vectorization/shared memory。
- 通过 correctness 后再加 shared memory padding、swizzle、fragment layout。

调试时可以设置：

```bash
TK_PRINT_KERNEL_SOURCE=1 pytest ...
```

如果是 puzzle starter：

```bash
TK_PUZZLE_IMPL=starter pytest puzzles/levels/<level>/<puzzle>/test_*.py
```
