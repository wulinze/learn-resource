# TileKernels Puzzle 学习项目设计

这个项目的目标不是再做一套孤立的 toy puzzles，而是把 TileLang 官方文档里的 DSL 概念，逐步映射到当前 `TileKernels` 仓库里的真实 LLM kernel。学习路径应从可验证的小算子开始，最后能读懂、修改并重写 `quant`、`moe`、`transpose`、`mhc`、`engram` 里的核心 kernel。

## 设计原则

1. 每个 puzzle 只训练一个主要概念。
2. 每个 puzzle 都有 PyTorch reference、正确性测试和可选 benchmark。
3. 每个 puzzle 都能指向当前仓库的一个真实 kernel，避免学完以后无法迁移。
4. 初期允许性能差，只要求语义正确；中后期逐步加入带宽、寄存器、shared memory、layout 和融合目标。
5. puzzle 代码和生产代码隔离，避免学习过程污染 `tile_kernels/`。

## 依赖的官方文档主线

官方文档当前推荐的学习顺序是 Language Basics、Control Flow、Software Pipeline Annotations、Instructions、Autotuning、Type System。这个项目按同一顺序组织，但把练习替换成当前仓库的真实场景。

重点文档：

- https://tilelang.com/programming_guides/language_basics.html
- https://tilelang.com/programming_guides/instructions.html
- https://tilelang.com/programming_guides/control_flow.html
- https://tilelang.com/programming_guides/autotuning.html
- https://tilelang.com/tutorials/debug_tools_for_tilelang.html
- https://tilelang.com/deeplearning_operators/gemv.html
- https://tilelang.com/deeplearning_operators/gemm.html

## 建议目录结构

```txt
puzzles/
├── README.md
├── common/
│   ├── check.py              # assert helpers, tolerance policy
│   ├── bench.py              # thin wrapper over tile_kernels.testing.bench
│   └── random.py             # deterministic CUDA input generators
├── levels/
│   ├── l00_environment/
│   │   ├── README.md
│   │   ├── starter.py
│   │   ├── reference.py
│   │   └── test_l00.py
│   ├── l01_basics/
│   ├── l02_memory_layout/
│   ├── l03_reduction/
│   ├── l04_quantization/
│   ├── l05_moe_routing/
│   ├── l06_fusion/
│   └── l07_production_reading/
└── solutions/
    └── README.md             # optional, can stay private or ignored
```

每个 puzzle 文件夹固定包含：

- `README.md`: 题目、限制、提示、对应官方文档、对应生产 kernel。
- `starter.py`: 带 `TODO` 的 TileLang kernel factory。
- `answer.py`: 包装当前仓库已有的生产实现，作为标准答案。
- `reference.py`: PyTorch reference。
- `test_*.py`: correctness tests，必要时加 benchmark marker。

命令风格沿用当前仓库：

```bash
pytest puzzles/levels/l02_memory_layout/test_l02.py
pytest puzzles/levels/l02_memory_layout/test_l02.py --run-benchmark
```

实际拆分时，测试默认运行 `answer.py`，保证仓库在学习者未实现 `starter.py` 时仍然可测。切换到待实现入口时使用：

```bash
TK_PUZZLE_IMPL=starter pytest puzzles/levels/l03_reduction/stable_topk/test_stable_topk.py
```

这个设计把“实现考察部分”和“现有答案”分开：

- `starter.py`: 学习者提交的实现，只保留题目要求的函数签名。
- `answer.py`: 当前生产 kernel 的包装，用来对照和验收。
- `reference.py`: PyTorch 语义参考，不依赖待实现 kernel。
- `test_*.py`: 单独运行的测试，不需要跑整个 `tests/` 目录。

## 关卡路线

### L00: 环境和最小 kernel

目标：确认 TileLang JIT、CUDA、PyTorch、测试框架都可用。

Puzzle:

1. `vector_add`: 一维 `A + B -> C`
2. `masked_vector_add`: `N` 不整除 block 时处理边界
3. `strided_copy`: 读非连续输入，写连续输出

训练点：

- `@tilelang.jit`
- `@T.prim_func`
- `T.Tensor`
- `T.StridedTensor`
- `T.Kernel`
- `T.Parallel`
- `T.ceildiv`

生产映射：

- `tile_kernels/transpose/batched_transpose_kernel.py` 的动态 shape 和 strided input
- `tile_kernels/quant/per_token_cast_kernel.py` 的 `token_stride`

### L01: Fragment、copy 和简单 layout

目标：从“每个线程做一个元素”过渡到 tiled kernel。

Puzzle:

1. `tile_copy_2d`: 把 `[M, N]` 按 tile 拷贝
2. `scale_bias`: `out = x * scale + bias`
3. `rowwise_affine`: 每行一组 scale/bias

训练点：

- `T.alloc_fragment`
- `T.alloc_shared`
- `T.copy`
- `T.clear` / `T.fill`
- fragment 内部循环
- global/shared/register 的职责区分

生产映射：

- `tile_kernels/quant/per_token_cast_kernel.py` 的 `x_fragment`
- `tile_kernels/quant/per_channel_cast_kernel.py`
- `tile_kernels/mhc/norm_fn_kernel.py`

### L02: Transpose 和 shared memory

目标：理解 coalesced load/store、shared memory padding、线程到元素的映射。

Puzzle:

1. `naive_transpose`: 正确但可能不快
2. `shared_transpose`: 用 shared memory 做 tile transpose
3. `batched_transpose`: 支持 batch 和 stride
4. `bank_conflict_padding`: 给 shared memory 加 padding，比较性能

训练点：

- `T.alloc_shared`
- `T.sync_threads`
- `T.vectorized`
- `T.unroll`
- `T.Fragment(..., forward_fn=...)`
- custom `loop_layout`

生产映射：

- `tile_kernels/transpose/batched_transpose_kernel.py`

验收目标：

- correctness: 与 `torch.transpose(...).contiguous()` bitwise 一致
- benchmark: 输出 GB/s，观察 padding 前后的差异

### L03: Reduction 和 Top-K

目标：从 elementwise 进入真实 routing kernel 的控制流和 reduction。

Puzzle:

1. `row_sum`: 每行求和
2. `row_max`: 每行最大值
3. `argmax_tie_break`: 最大值相同时返回更小 index
4. `stable_topk`: 重复 reduce max 得到 top-k
5. `normalize_selected`: top-k weight sum normalization

训练点：

- `T.reduce_sum`
- `T.reduce_max`
- `T.alloc_reducer`
- `T.finalize_reducer`
- `T.infinity`
- tie-break 语义
- shared memory 暂存小输出

生产映射：

- `tile_kernels/moe/topk_gate_kernel.py`
- `tile_kernels/moe/normalize_weight_kernel.py`
- `tile_kernels/moe/topk_sum_and_topk_group_idx_kernel.py`

验收目标：

- 与 `tile_kernels.torch.stable_topk` 对齐
- ties 必须稳定返回小 index

### L04: Quantization 基础

目标：理解 `amax -> scale -> cast -> scale factor storage` 的完整路径。

Puzzle:

1. `per_row_absmax`: 每 token 求 absmax
2. `per_group_absmax`: 每 token 按 hidden group 求 absmax
3. `per_token_scale`: 生成 scale factor
4. `per_token_cast_e4m3`: cast 到 FP8 E4M3
5. `cast_only`: 复用预计算 scale factor
6. `sf_only`: 只输出 scale factor

训练点：

- `T.reshape`
- `T.reduce_absmax`
- dtype conversion
- scale factor 形状推导
- strided scale factor tensor
- boundary and alignment assumptions

生产映射：

- `tile_kernels/quant/per_token_cast_kernel.py`
- `tile_kernels/quant/per_block_cast_kernel.py`
- `tile_kernels/quant/common.py`

验收目标：

- 与 `tile_kernels.torch.cast` 对齐
- 支持 `num_per_channels in (32, 64, 128)`
- 支持非连续输入 `stride(0) != hidden`

### L05: MoE Routing 小流水线

目标：把多个小 kernel 串成 MoE routing 子流程。

Puzzle:

1. `group_count`: 统计 top-k expert group
2. `mask_indices_by_tp`: 根据 TP rank 屏蔽 expert index
3. `expand_to_fused`: token 按 expert 排列到 fused buffer
4. `reduce_fused`: fused buffer reduce 回 token
5. `mini_moe_route`: top-k + normalize + expand + reduce 的最小闭环

训练点：

- scatter/gather
- atomic 或分块统计
- persistent block 的基本思想
- 多 kernel 组合
- 与 PyTorch reference 的端到端对齐

生产映射：

- `tile_kernels/moe/group_count_kernel.py`
- `tile_kernels/moe/mask_indices_by_tp_kernel.py`
- `tile_kernels/moe/expand_to_fused_kernel.py`
- `tile_kernels/moe/reduce_fused_kernel.py`

### L06: Fusion

目标：把算子融合从“少一次 launch”推进到“少一次 global memory traffic”。

Puzzle:

1. `swiglu_forward`: 单独实现 SwiGLU
2. `swiglu_forward_and_cast`: 融合 SwiGLU + per-token cast
3. `swiglu_backward`: 实现 backward reference-compatible kernel
4. `swiglu_backward_and_cast`: backward + per-token cast
5. `per_channel_cast_and_transpose`: cast 同时转置

训练点：

- fused epilogue
- intermediate fragment reuse
- 输出多个 tensor
- forward/backward consistency
- bandwidth accounting

生产映射：

- `tile_kernels/quant/swiglu_forward_and_per_token_cast_kernel.py`
- `tile_kernels/quant/swiglu_backward_and_per_token_cast_kernel.py`
- `tile_kernels/quant/per_channel_cast_and_transpose_kernel.py`
- `tile_kernels/quant/swiglu_forward_and_per_channel_cast_and_transpose_kernel.py`

### L07: Production Reading

目标：不再填 TODO，而是读懂、解释、修改当前仓库的复杂 kernel。

Puzzle:

1. `explain_top2_sum_gate`: 写出 `top2_sum_gate_kernel.py` 的数据流图
2. `simplify_top2_sum_gate`: 去掉一组高级选项，保留最小可测版本
3. `mhc_norm_microkernel`: 从 `norm_fn_kernel.py` 抽出一个小 kernel 复现
4. `engram_hash_microkernel`: 从 `engram_hash_kernel.py` 抽出 hash 逻辑复现
5. `benchmark_regression`: 给一个 kernel 加 baseline，制造并修复性能回退

训练点：

- 读生产 TileLang kernel
- 拆解复杂 control flow
- pass config 的取舍
- benchmark regression workflow
- PR 级别的 correctness + perf 证据

生产映射：

- `tile_kernels/moe/top2_sum_gate_kernel.py`
- `tile_kernels/mhc/norm_fn_kernel.py`
- `tile_kernels/engram/engram_hash_kernel.py`
- `tests/pytest_benchmark_plugin.py`

## Puzzle README 模板

```md
# L03 P04: stable_topk

## Task

Implement a TileLang kernel that returns stable top-k indices for each token.

## Inputs

- `scores`: `[num_tokens, num_experts]`, float32, contiguous
- `num_topk`: compile-time int

## Output

- `topk_idx`: `[num_tokens, num_topk]`, int64

## Correctness

- Match `tile_kernels.torch.stable_topk`
- If scores tie, smaller expert id wins

## Constraints

- One CTA per token
- Use `T.reduce_max`
- Use a reducer or equivalent logic for tie-break
- Do not call `torch.topk` inside the kernel wrapper

## Related Docs

- TileLang Instructions: reductions and diagnostics
- TileLang Control Flow

## Production Reference

- `tile_kernels/moe/topk_gate_kernel.py`
```

## Test design

每个 puzzle 的测试分三层：

1. `test_correctness_small`: 小 shape，便于 debug。
2. `test_correctness_project_shapes`: 使用当前项目的 shape generator，如 `generate_num_tokens`、`generate_hidden_sizes`。
3. `test_benchmark`: 标记为 `@pytest.mark.benchmark`，默认跳过，使用 `--run-benchmark` 运行。

推荐测试 helper 复用：

- `tile_kernels.testing.numeric.assert_equal`
- `tile_kernels.testing.numeric.count_bytes`
- `tile_kernels.testing.bench.make_param_id`
- `tile_kernels.testing.generator.generate_num_tokens`
- `tile_kernels.testing.generator.generate_hidden_sizes`

## 评分方式

每题给三个等级：

- Bronze: correctness pass。
- Silver: correctness pass + 支持项目常用 shape/dtype/stride。
- Gold: correctness pass + benchmark 达到 reference kernel 的某个比例。

示例：

```txt
L02 shared_transpose
Bronze: matches torch transpose on contiguous bf16 input
Silver: supports batched input and stride_x
Gold: >= 80% bandwidth of tile_kernels.transpose.batched_transpose on the same shape
```

## 建议首批实现的 12 个 puzzle

优先实现这些，能最快建立从 DSL 到生产代码的桥：

1. `l00_environment/vector_add`
2. `l00_environment/strided_copy`
3. `l01_basics/tile_copy_2d`
4. `l01_basics/rowwise_affine`
5. `l02_memory_layout/naive_transpose`
6. `l02_memory_layout/shared_transpose`
7. `l03_reduction/row_max`
8. `l03_reduction/stable_topk`
9. `l04_quantization/per_group_absmax`
10. `l04_quantization/per_token_cast_e4m3`
11. `l05_moe_routing/group_count`
12. `l06_fusion/swiglu_forward_and_cast`

## 和当前仓库的集成边界

建议先不把 puzzle 暴露为 `tile_kernels` package API。测试直接从 `puzzles/levels/.../starter.py` 导入，避免学习代码成为正式发布内容。

如果后续要发布成独立学习项目，可以拆成：

```txt
tilekernels-puzzle/
├── pyproject.toml
├── puzzles/
├── tests/
└── references/
```

其中 `references/` 只保留从 `TileKernels` 提取出的最小 PyTorch reference，不依赖生产 kernel。

## 推荐推进顺序

1. 先落地 `puzzles/README.md`、`common/` 和 L00-L02。
2. 每个 level 只先实现 2 到 3 个高质量题目，保证测试和说明完整。
3. L03 开始引入当前 `tile_kernels.torch` reference。
4. L04 开始引入 benchmark 和 bandwidth 目标。
5. L07 再要求读生产 kernel 并写 explanation，避免一开始就被复杂业务逻辑淹没。
