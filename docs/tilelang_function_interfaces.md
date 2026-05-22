# TileLang 相关函数接口与使用场景

本文档基于当前代码库的静态检查整理，覆盖 `tile_kernels` 中实际使用 TileLang 的生产接口，并把 `puzzles` 中的 TileLang 教学入口单独列出。项目依赖声明见 `pyproject.toml`：`tilelang>=0.1.9`。

## 约定

- `QuantTensor = tuple[torch.Tensor, torch.Tensor]`，表示 `(data, scale_factors)`。
- 公开 API 通常接收/返回 PyTorch tensor；内部 `get_*_kernel`、`_mhc_*` 函数是 `@tilelang.jit` kernel factory，用编译期参数生成 `tilelang.JITKernel`。
- 设置 `TK_PRINT_KERNEL_SOURCE=1` 时，多数 wrapper 会打印生成的 TileLang kernel 源码。
- 下表优先列公开 wrapper；内部 JIT factory 和 macro 在后文单独列。

## Transpose

| 接口 | 输入/输出 | 使用场景 |
| --- | --- | --- |
| `tile_kernels.transpose.transpose(x: torch.Tensor) -> torch.Tensor` | 输入 `(M, N)`，`M/N` 需 64 对齐；输出 `(N, M)`。支持 BF16/FP8 等 TileLang dtype。 | 2D tiled GPU 转置，测试中覆盖非连续 leading stride。 |
| `tile_kernels.transpose.batched_transpose(x: torch.Tensor) -> torch.Tensor` | 输入 `(B, M, N)`，`M/N` 需 64 对齐，`stride(-2) % 4 == 0`；输出 `(B, N, M)`。 | 专家维/批量矩阵转置，比逐个 `torch.transpose(...).contiguous()` 更适合批处理。 |

## Quantization

| 接口 | 输入/输出 | 使用场景 |
| --- | --- | --- |
| `per_token_cast(x, fmt, num_per_channels, x_block_size=None, use_tma_aligned_col_major_sf=False, round_sf=False, use_packed_ue8m0=False) -> QuantTensor` | 输入 `(num_tokens, hidden)` BF16/FP32，或带 `x_block_size` 的 `QuantTensor`；`fmt in {'e5m6','e4m3','e2m1'}`；输出 `(out, out_sf)`。 | 激活按 token 行维度分组量化，常用于 FP8/FP4 GEMM 输入准备。 |
| `per_token_cast_with_sf_only(...) -> torch.Tensor` | 同 `per_token_cast`，只返回 scale factor。 | 先预计算量化尺度，后续再 cast。 |
| `per_token_cast_with_precomputed_sf(x, fmt, num_per_channels, sf, ...) -> torch.Tensor` | 使用已有 `sf`，只返回 cast 后数据。 | 将 scale 计算和数据 cast 分离，复用尺度。 |
| `per_token_cast_to_e5m6(x, num_per_channels, ...) -> QuantTensor` | 输入 BF16/FP32，要求 `num_per_channels == hidden`；输出 packed `uint8` 数据 `(num_tokens, hidden * 3 // 2)` 和 sf。 | 自定义 E5M6 12-bit truncated-half 激活压缩。 |
| `per_block_cast(x, fmt, block_size, ...) -> QuantTensor` | 输入 2D contiguous tensor，`hidden % 64 == 0`；`block_size=(num_per_tokens,num_per_channels)`，常见 32/128。 | MXFP8/MXFP4 2D block scale 量化。 |
| `per_block_cast_with_sf_only(...) -> torch.Tensor` | 同 `per_block_cast`，只返回 sf。 | block 量化尺度预计算。 |
| `per_block_cast_with_precomputed_sf(x, fmt, block_size, sf, ...) -> torch.Tensor` | 使用已有 block sf。 | 复用 block scale 执行 cast-only。 |
| `per_block_cast_lossless(x, fmt, x_block_size, out_block_size, ...) -> QuantTensor` | 输入 E2M1 `QuantTensor`；当前只支持 lossless `e2m1 -> e4m3`。 | FP4 权重/激活转 FP8，同时改变 scale block 形状并保持数值可恢复。 |
| `per_channel_cast(x, fmt, num_per_tokens, round_sf=False) -> QuantTensor` | 输入 contiguous 2D tensor；仅 `fmt='e4m3'`、`num_per_tokens=128`。 | 按 channel 列维度 scale 的 FP8 cast。 |
| `per_channel_cast_fused(x, fmt, num_per_tokens, round_sf=False, num_per_channels=None, pos_to_token=None) -> QuantTensor` | 输入 tensor 或 `QuantTensor`；可选 `pos_to_token` 做 token gather/expand。 | 将 rescale、token expansion、per-channel FP8 cast 融合。 |
| `per_channel_cast_and_transpose(x, fmt, num_per_tokens, round_sf=False) -> QuantTensor` | 输入 BF16 `(num_tokens, hidden)`，`num_tokens % 128 == 0`、`hidden % 64 == 0`；输出数据 `(hidden, num_tokens)`。 | 权重/激活 cast 后直接转置，服务后续 GEMM 布局。 |
| `cast_back(x, fmt, x_block_size, x_special_fmt=None) -> torch.Tensor` | 输入 `QuantTensor`；`fmt in {'bf16','fp32'}`；`x_special_fmt='e5m6'` 时走 E5M6 解包。 | 调试、测试或回退路径中的反量化。 |
| `per_token_cast_back(x, fmt, num_per_channels, x_special_fmt=None) -> torch.Tensor` | `cast_back(x, fmt, (1, num_per_channels), ...)` 的便捷封装。 | per-token quant tensor 的反量化。 |
| `swiglu_forward_and_per_token_cast(x, fmt, num_per_channels, pos_to_token_topk=None, topk_weights=None, pos_to_expert=None, ..., swiglu_clamp_value=None, clamped_count=None, sf_clamp_min=None) -> QuantTensor` | 输入 `(num_expanded_tokens, hidden*2)`；输出 SwiGLU 后 `(num_expanded_tokens, hidden)` 的 FP8 `QuantTensor`。 | MoE FFN 中把 SwiGLU、路由权重乘法、无效 expert mask、per-token FP8 cast 融合为一次 kernel。 |
| `swiglu_forward_and_per_channel_cast_and_transpose(x, fmt, num_per_tokens, round_sf=False, without_transpose=False, swiglu_clamp_value=None) -> QuantTensor` | 输入 BF16 `(num_tokens, hidden*2)`；输出 `(hidden,num_tokens)` 或 `(num_tokens,hidden)`。 | SwiGLU 前向后做 per-channel FP8 cast，并可同时转置。 |
| `swiglu_backward_and_per_token_cast(x, grad_out, weight, pos_to_token_topk, token_topk_to_pos, num_per_channels, round_sf=False, swiglu_clamp_value=None) -> tuple[torch.Tensor, QuantTensor, torch.Tensor, torch.Tensor]` | 输入 FP8 `QuantTensor` 和 BF16 grad；返回 `(out, x_grad_fp8_quant, x_grad_bf16, weight_grad)`。 | SwiGLU 反向，同时产生 BF16 梯度和 FP8 梯度供后续通信/计算使用。 |
| `unpack_from_e2m1fn_x2(x, out_dtype=torch.float32) -> torch.Tensor` | 输入 packed FP4 `int8/uint8`，最后一维解包为 2 倍。 | 测试和 debug 中查看 E2M1 FP4 数值。 |

## MoE Routing

| 接口 | 输入/输出 | 使用场景 |
| --- | --- | --- |
| `topk_gate(scores, num_topk) -> torch.Tensor` | 输入 FP32 `(num_tokens, num_experts)`；输出 int64 `(num_tokens, num_topk)`。 | 基础 top-k expert 选择；tie 时取较小 index。 |
| `topk_sum_and_topk_group_idx(scores, num_topk_sum, num_topk_groups) -> torch.Tensor` | 输入 FP32 `(num_tokens, num_groups, num_experts_per_group)`；输出 group index。 | DeepSeek 类分组路由：先算每组 top1/top2 sum，再选 top groups。 |
| `top2_sum_gate(logits, bias, num_topk, num_topk_groups, num_groups, use_shared_as_routed, num_shared_experts, routed_scaling_factor, ep_rank, num_ep_ranks, tp_rank, num_tp_ranks, scoring_func, mask=None, fix_routing_mask=None, to_physical_map=None, logical_count=None, unmapped_topk_idx=None) -> tuple[torch.Tensor, torch.Tensor]` | 输入 logits/bias；输出 `(topk_idx, topk_weights)`，shape 为 `(num_tokens, num_topk + num_shared_experts)`。 | 完整 MoE gate：支持 sigmoid/sqrtsoftplus/softmax、top2-sum group routing、shared experts、EP/TP 本地 remap、固定路由。 |
| `get_fused_mapping(topk_idx, num_experts, num_expanded_tokens, alignment, force_no_sync=False) -> tuple[...]` | 输入 token-major topk index；输出 `pos_to_expert/pos_to_token/pos_to_token_topk/token_topk_to_pos/expert_start/expert_end/num_tokens_per_expert/...`。 | 将 token-topk 布局转换为 expert-major fused 布局，用于 grouped GEMM/dispatch。 |
| `expand_to_fused(x, token_topk_to_pos, pos_to_expert) -> torch.Tensor` | 输入 `(num_tokens, hidden)`；输出 `(num_expanded_tokens, hidden)`。 | 根据 mapping 展开 token activations 到 fused expert buffer。 |
| `expand_to_fused_with_sf(x, num_per_channels, token_topk_to_pos, pos_to_expert, use_tma_aligned_col_major_sf=False) -> tuple[torch.Tensor, torch.Tensor]` | 输入 `QuantTensor`，同时展开 data 和 sf。 | FP8/FP4 activation dispatch 时保持量化尺度同步展开。 |
| `reduce_fused(x, topk_weights, token_topk_to_pos, fp8_format='', sf=None, out=None) -> torch.Tensor` | 输入 expanded expert output 或 `QuantTensor`；输出 token-major `(num_tokens, hidden)`。 | 将 expert-major 输出按 top-k 权重规约回 token layout，可选输出 FP8。 |
| `normalize_weight(topk_weights) -> tuple[torch.Tensor, torch.Tensor]` | 输入 FP32 `(num_tokens, num_topk)`；输出 `(denominator, normalized_weights)`。 | 路由权重归一化，分母加 `1e-20` 防止除零。 |
| `group_count(group_idx, num_groups) -> torch.Tensor` | 输入 int64 `(num_tokens, num_topk)`；输出 int32 `(num_groups,)`。 | 统计每个 expert/group 收到的 token 数。 |
| `aux_fi(topk_idx, num_experts, num_aux_topk) -> torch.Tensor` | 输出 FP32 `(num_experts,)`。 | MoE auxiliary loss 的频率项 `f_i`。 |
| `inplace_unique_group_indices(group_indices, num_groups) -> None` | 就地修改 `(num_tokens, num_topk)`；重复 group 置为 `-1`，`num_groups <= 128`。 | group 路由去重，避免同一 token 重复选中同组。 |
| `mask_indices_by_tp(indices, n, num_ep_ranks, tp_rank, num_tp_ranks) -> torch.Tensor` | 输出与输入同 shape，非当前 TP rank 的 expert 置 `-1` 并 remap local id。 | EP/TP 混合并行下筛选本 rank 负责的 expert。 |

## Engram

| 接口 | 输入/输出 | 使用场景 |
| --- | --- | --- |
| `engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets) -> torch.Tensor` | 输入 n-gram token ids `(num_tokens,max_ngram_size)` 和每层 hash 配置；输出 int32 `(num_ngram_layers,num_tokens,(max_ngram_size-1)*num_embed_table_per_ngram)`。 | Engram n-gram embedding table 索引生成。 |
| `fused_weight(weight_hidden, weight_embed) -> torch.Tensor` | 输入 BF16 `(hc_mult, hidden_size)` 两组 RMSNorm weight；输出 FP32 fused weight。 | Engram gate 前向前先融合 `weight_hidden * weight_embed`。 |
| `engram_gate_fwd(hidden_states, k, v, weight_fused, eps, clamp_value, save_for_backward=True) -> tuple[...]` | 输入 BF16 `hidden_states/k=(num_tokens,hc_mult,hidden_size)`、`v=(num_tokens,hidden_size)`；输出 `(output,dot,gate_score,rstd_x,rstd_k)`。 | Engram gate 前向：RMSNorm dot、signed-sqrt sigmoid gate、`x + gate*v` 融合。 |
| `engram_gate_bwd(grad_out, hidden_states, k, v, weight_fused, dot, gate_score, rstd_x, rstd_k, clamp_value) -> tuple[...]` | 返回 `(grad_hidden_states, grad_k, grad_v, grad_w_partial)`。 | Engram gate 反向，产生 weight 的 partial grad。 |
| `grad_w_reduce(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed) -> None` | 就地累加 `grad_weight_hidden/grad_weight_embed`。 | 将 persistent block 的 partial weight grad 规约并乘回两组原始权重。 |
| `tile_kernels.modeling.engram.engram_gate(...)` | `torch.autograd.Function.apply` wrapper。 | 训练中直接作为可自动求导的 Engram gate 层使用；支持参数 `main_grad` 就地累加。 |

## Manifold HyperConnection (mHC)

| 接口 | 输入/输出 | 使用场景 |
| --- | --- | --- |
| `expand_to_mhc(hidden, mhc_mult, out=None) -> torch.Tensor` / `expand_from_embedding(x, mhc_mult=4)` | 输入 `(..., hidden_size)`；输出 `(..., mhc_mult, hidden_size)`。 | 将普通 embedding/residual 扩展为 mHC 多 residual head 格式。 |
| `mhc_pre_norm_fn(residual, mhc_fn, mhc_norm_weight, mhc_norm_eps, fuse_grad_acc=True, n_splits=16) -> torch.Tensor` | 输入 BF16 residual `(..., mhc_mult, hidden)` 和 FP32 fn `(mhc_mult*(mhc_mult+2), mhc_mult*hidden)`；输出 mixes。 | mHC pre 阶段的 RMSNorm + 线性投影，带 backward。 |
| `mhc_pre_split_mixes(input_mixes, mhc_scale, mhc_base, mhc_mult, mhc_post_mult_value, mhc_pre_eps) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]` | 输出 `(pre_layer_mix, post_layer_mix, comb_res_mix)`。 | 将 projection 结果拆成 pre mix、post mix 和 residual combination matrix。 |
| `sinkhorn_normalize(x, repeat=10, eps=1e-6) -> torch.Tensor` | 输入 `(..., mhc_mult, mhc_mult)`；输出同 shape。 | 对 `comb_res_mix` 做 Sinkhorn 归一化，使组合矩阵近似双随机。 |
| `mhc_pre_apply_mix(x, mix, out=None) -> torch.Tensor` | 输入 residual `(..., mhc_mult, hidden)` 和 mix `(..., mhc_mult, 1)`；输出 `(..., hidden)`。 | 用 pre mix 聚合多 residual head，得到子层输入。 |
| `mhc_post(x, residual, post_layer_mix, comb_res_mix, out=None) -> torch.Tensor` | 输入子层输出 `(..., hidden)`、旧 residual、post mix 和 comb mix；输出新 residual。 | mHC 子层后更新 residual。 |
| `mhc_post_fwd(...)` / `mhc_post_bwd(...)` | 低层前向/反向 wrapper。 | `MHCPost` autograd 使用，也可用于测试或手动融合梯度路径。 |
| `mhc_pre_big_fuse(residual, fn, mhc_scale, mhc_base, rms_eps, mhc_pre_eps, mhc_sinkhorn_eps, mhc_post_mult_value, sinkhorn_repeat, n_splits=16) -> tuple[...]` | 输出 `(post_mix, comb_mix, layer_input)`。 | 推理模式下将 pre_norm_fn、split mixes、sinkhorn、pre_apply_mix 融合，减少 kernel launch 和 IO。 |
| `mhc_pre(residual, fn, scale, base, *, norm_weight=None, norm_eps=1e-6, mhc_mult=4, post_mult_value=1.0, pre_eps=1e-6, sinkhorn_eps=1e-6, sinkhorn_repeat=10, n_splits=16) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]` | 返回 `(layer_input, (post_mix, comb_mix))`。 | mHC block 的高层 pre 接口；训练走可求导分步 kernel，推理走 `mhc_pre_big_fuse`。 |
| `mhc_head_compute_mix(input_mix, mhc_scale, mhc_base, mhc_pre_eps) -> torch.Tensor` | 输入 `(..., mhc_mult)`；输出同 shape。 | LM head 前的 mix 计算。 |
| `mhc_head(residual, fn, scale, base, *, norm_weight=None, norm_eps=1e-6, mhc_mult=4, pre_eps=1e-6, n_splits=16) -> torch.Tensor` | 输出 `(..., hidden)`。 | mHC 最终 head 输入生成。 |
| `mhc_multilayer_recompute(initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list, layer_input_list, residual_list) -> None` | 使用 list 中 tensor 的 device pointer table，就地写 `layer_input_list/residual_list`。 | 多层 mHC activation recompute，减少逐层 Python 调度和中间 IO。 |

## 内部 TileLang JIT Kernel Factory

这些函数直接使用 `@tilelang.jit`，返回 `tilelang.JITKernel`。一般不作为用户 API 暴露，而是由上面的 PyTorch wrapper 选择编译参数并调用。

| 模块 | JIT factory | 对应公开 wrapper |
| --- | --- | --- |
| `transpose/batched_transpose_kernel.py` | `get_batched_transpose_kernel(shape_x_mod_128, shape_y_mod_128, dtype)` | `transpose`, `batched_transpose` |
| `quant/per_token_cast_kernel.py` | `get_per_token_cast_kernel(hidden, token_stride, in_config, out_config, sf_only=False, cast_only=False)` | `per_token_cast*` |
| `quant/per_token_cast_to_e5m6_kernel.py` | `get_per_token_cast_to_e5m6_kernel(hidden, token_stride, in_config, out_config)` | `per_token_cast_to_e5m6` |
| `quant/per_block_cast_kernel.py` | `get_per_block_cast_kernel(hidden, in_config, out_config, sf_only=False, cast_only=False)` | `per_block_cast*` |
| `quant/per_block_cast_lossless_kernel.py` | `get_per_block_cast_lossless_kernel(hidden, token_stride, in_config, out_config)` | `per_block_cast_lossless` |
| `quant/per_channel_cast_fused_kernel.py` | `get_per_channel_cast_fused_kernel(hidden, with_expand, in_config, out_config)` | `per_channel_cast`, `per_channel_cast_fused` |
| `quant/per_channel_cast_and_transpose_kernel.py` | `get_per_channel_cast_and_transpose_kernel(hidden, in_dtype, out_config)` | `per_channel_cast_and_transpose` |
| `quant/cast_back_kernel.py` | `get_cast_back_kernel(hidden, in_config, out_dtype=T.bfloat16)` | `cast_back`, `per_token_cast_back` |
| `quant/cast_back_e5m6_kernel.py` | `get_cast_back_e5m6_kernel(hidden, in_config, out_dtype)` | `cast_back_e5m6`, `cast_back(..., x_special_fmt='e5m6')` |
| `quant/swiglu_forward_and_per_token_cast_kernel.py` | `get_swiglu_forward_and_per_token_cast_kernel(hidden, with_weight, with_pos_to_expert, use_clamp, count_clamp, in_dtype, out_config, num_sms)` | `swiglu_forward_and_per_token_cast` |
| `quant/swiglu_forward_and_per_channel_cast_and_transpose_kernel.py` | `get_swiglu_forward_and_per_channel_cast_and_transpose_kernel(hidden, without_transpose, use_clamp, in_dtype, out_config, swiglu_clamp_value)` | `swiglu_forward_and_per_channel_cast_and_transpose` |
| `quant/swiglu_backward_and_per_token_cast_kernel.py` | `get_swiglu_backward_and_per_token_cast_kernel(hidden, out_config, use_clamp)` | `swiglu_backward_and_per_token_cast` |
| `moe/topk_gate_kernel.py` | `get_topk_gate_kernel(num_experts, num_topk)` | `topk_gate` |
| `moe/topk_sum_and_topk_group_idx_kernel.py` | `get_topk_sum_and_topk_group_idx_kernel(num_groups, num_experts_per_group, num_topk_groups, num_topk_sum)` | `topk_sum_and_topk_group_idx` |
| `moe/top2_sum_gate_kernel.py` | `get_top2_sum_gate_kernel(scoring_type, num_topk, num_topk_groups, num_groups, num_routed_experts, mask_exists, fix_routing_mask_exists, unmapped_topk_idx_exists, to_physical_map_exists)` | `top2_sum_gate` |
| `moe/get_fused_mapping_kernel.py` | `get_get_fused_mapping_kernel(num_experts, num_topk, alignment, num_sms)` | `get_fused_mapping` |
| `moe/expand_to_fused_kernel.py` | `get_expand_to_fused_kernel(hidden, num_topk, num_per_channels, use_tma_aligned_col_major_sf, use_packed_ue8m0, x_dtype, sf_dtype)` | `expand_to_fused*` |
| `moe/reduce_fused_kernel.py` | `get_reduce_fused_kernel(hidden, num_topk, in_dtype, out_dtype, with_sf, with_weights, with_x_sf)` | `reduce_fused` |
| `moe/normalize_weight_kernel.py` | `get_normalize_weight_kernel(num_topk)` | `normalize_weight` |
| `moe/group_count_kernel.py` | `get_group_count_kernel(num_topk, num_groups, num_sms)` | `group_count` |
| `moe/aux_fi_kernel.py` | `get_aux_fi_kernel(num_topk, num_experts, num_sms)` | `aux_fi` |
| `moe/inplace_unique_group_indices_kernel.py` | `get_inplace_unique_group_indices_kernel(num_topk, num_groups_aligned, num_sms)` | `inplace_unique_group_indices` |
| `moe/mask_indices_by_tp_kernel.py` | `get_mask_indices_by_tp_kernel(num_topk, dtype)` | `mask_indices_by_tp` |
| `engram/engram_hash_kernel.py` | `get_engram_hash_kernel(max_ngram_size=3, num_ngram_layers=2, num_embed_table_per_ngram=8)` | `engram_hash` |
| `engram/engram_fused_weight_kernel.py` | `get_engram_fused_weight_kernel(hidden_size, hc_mult)` | `fused_weight` |
| `engram/engram_gate_kernel.py` | `get_engram_gate_fwd_kernel(...)`, `get_engram_gate_bwd_kernel(...)` | `engram_gate_fwd`, `engram_gate_bwd` |
| `engram/engram_grad_w_reduce_kernel.py` | `get_engram_grad_w_reduce_kernel(hidden_size, num_persistent_blocks, hc_mult=4)` | `grad_w_reduce` |
| `mhc/expand_kernel.py` | `expand_to_mhc_fwd_tl(hidden, mhc_mult)`, `expand_to_mhc_bwd_tl(hidden, mhc_mult)` | `expand_to_mhc` |
| `mhc/norm_fn_kernel.py` | `_mhc_fn_normw_merge_fwd/bwd`, `_mhc_pre_norm_fn_fwd_mul/fwd_norm/bwd_norm/bwd_mul` | `mhc_pre_norm_fn` |
| `mhc/pre_split_mixes_kernel.py` | `_mhc_pre_split_mixes_fwd/bwd` | `mhc_pre_split_mixes` |
| `mhc/sinkhorn_kernel.py` | `_mhc_sinkhorn_fwd/bwd` | `sinkhorn_normalize` |
| `mhc/pre_apply_mix_kernel.py` | `_mhc_pre_apply_mix_fwd/bwd` | `mhc_pre_apply_mix` |
| `mhc/post_kernel.py` | `_mhc_post_fwd/bwd` | `mhc_post` |
| `mhc/pre_big_fuse_kernel.py` | `_mhc_pre_big_fuse(...)` | `mhc_pre_big_fuse`, inference path of `mhc_pre` |
| `mhc/head_compute_mix_kernel.py` | `_mhc_head_compute_mix_fwd/bwd` | `mhc_head_compute_mix`, `mhc_head` |
| `mhc/multilayer_recompute_kernel.py` | `_mhc_multilayer_recompute_kernel(...)` | `mhc_multilayer_recompute` |

## TileLang Macro / DSL Helper

| 函数 | 类型 | 使用场景 |
| --- | --- | --- |
| `quant.common.get_sf_and_inv(amax, out_config)` | `@T.macro` | 由 quant kernels 计算 scale factor 和 inverse scale。 |
| `quant.common.load_sf(tensor, m_idx, k_idx, config)` | `@T.macro` | 兼容 row-major、TMA aligned col-major、packed UE8M0 三种 sf 布局。 |
| `quant.common.transform_sf(sf, config)` | `@T.macro` | packed UE8M0 转 FP32 scale。 |
| `quant.common.store_sf(tensor, sf, m_idx, k_idx, config)` | `@T.macro` | 按配置写回 sf。 |
| `quant.per_token_cast_to_e5m6_kernel.get_sf_and_inv_e5m6(...)` | `@T.macro` | E5M6 专用 scale 计算。 |
| `quant.per_token_cast_to_e5m6_kernel.float_to_e5m6(...)` | `@T.macro` | 8 个 FP32 值打包成 3 个 uint32。 |
| `quant.cast_back_e5m6_kernel.e5m6_to_float(...)` | `@T.macro` | E5M6 packed 数据解包。 |
| `moe.common.get_topk_group_idx(...)` | `@T.macro` | group 内 top1/top2 sum 排序，供 group routing 使用。 |
| `moe.get_fused_mapping_kernel.divide_task(...)` | `@T.macro` | 将 token-topk 扫描任务按 warp 切块。 |
| `moe.top2_sum_gate_kernel.warp_reduce_sum(x)` | `@T.macro` | warp 内求和，top2 gate softmax 用。 |
| `moe.scoring.softplus(x)` | `@T.macro` | sqrt-softplus scoring 函数。 |
| `mhc.norm_fn_kernel.round_to_tf32(x)` | Python helper | 通过 bit round 让 FP32 权重接近 TF32 计算行为。 |

## Puzzles 中的 TileLang 教学入口

`puzzles/levels/*/starter.py` 是练习模板，不是生产 API：

- `l02_memory_layout/batched_transpose/starter.py`: `batched_transpose_impl`, `batched_transpose`, `transpose`；当前 starter 有故意未完成语法。
- `l03_reduction/stable_topk/starter.py`: `topk_gate(scores, num_topk)`。
- `l04_quantization/per_token_cast/starter.py`: `per_token_cast(...)`。
- `l05_moe_routing/group_count/starter.py`: `group_count(group_idx, num_groups)`。
- `l06_fusion/swiglu_forward_and_per_token_cast/starter.py`: `swiglu_forward_and_per_token_cast(...)`。

