# B1 — B300 (SM103) environment and baseline survey

Node: 8x NVIDIA B300 SXM6 AC, base vLLM `e66bede56b` unmodified. No vLLM code was changed.
All 8 GPUs were idle (0 MiB, no other users) before every run. Host background load was ~10 (system daemons, about 9 of 256 cores).

## 1. Environment

| item | value |
|---|---|
| GPU | 8x NVIDIA B300 SXM6 AC |
| `get_device_capability()` | **(10, 3)** |
| driver | 595.112.07 |
| CUDA toolkit | 13.2 (V13.2.86) |
| torch | 2.14.0+cu132 |
| FlashInfer | 0.7.4 (cubin 0.7.4, jit-cache 0.7.4+cu130 incl. sm103a) |
| Triton | 3.8.0 |
| cuBLAS / NCCL | 13.4.0.1 / 2.30.7 |
| vLLM | 0.34.1rc1.dev61+ge66bede56 (precompiled wheel, editable) |

Setup followed `setup/B300_START.md` §2. `vllm download-kernels` fetched the sm103a jit-cache. Model config and tokenizer were downloaded by file name (`du -shL` snapshot = 29M, no weights; dummy load).

## 2. Sanity

`pytest tests/kernels/test_bf16_skinny_gemm.py -q` (GPU 0): **351 passed, 438 skipped**, 0 failed (4m55s).

## 3. SM103 coverage (read only, nothing timed)

- **`QWEN4_EXP_GEMM_PLANS_BY_CAPABILITY[(10, 3)]`**: 12 shapes, 34 (shape, M) entries. The SM100 table has 24 shapes / 68 entries.
  - **TP4 (6 shapes):**
    - GDN QKVZ (4096, 2560): M 1, 2
    - GDN/QSA out (2560, 1536): M 1, 2, 4
    - GDN B/A (24, 2560): M 1, 2, 4, 8, 16
    - QSA QKV/gate (4224, 2560): M 1, 2, 4
    - shared-expert gate/up (320, 2560): M 1, 2, 4, 8, 16
    - LM head (62080, 2560): M 1, 2
  - **TP1 (6 shapes):**
    - GDN/QSA out (2560, 6144): M 1, 2
    - GDN B/A (96, 2560): M 1, 2, 4
    - QSA QKV/gate (13952, 2560): M 1
    - shared-expert gate/up (1280, 2560): M 1, 2, 4
    - shared-expert down (2560, 640): M 1, 2
    - HC final down (320, 10240): M 1, 2, 4. This shape is replicated, so it also applies at TP4.
  - **No TP2 or TP8 entries.** (The SM100 table has TP8 shapes, e.g. (12, 2560) and (160, 2560).)
- **`SKINNY_GEMM_PLANS_BY_CAPABILITY`** (`model_executor/kernels/linear/cute_dsl/skinny_gemm_plans.py`): contains only `(12, 1)`. **There is no `(10, 3)` entry**, and no `(10, 0)` on this base either. So every unquantized BF16 projection of Qwen3.5/3.6 runs on cuBLAS on B300.
- **`FI_ALLREDUCE_FUSION_MAX_SIZE_MB[103]`** = {2: 64, 4: 64, **8: 4**, 16: 64} MB.
- **TP8 FlashInfer AR backend on this node: `mnnvl`.**
  - Log line: `Initialized FlashInfer Allreduce norm fusion workspace with backend=mnnvl`. No trtllm-fallback warning appeared, and `allreduce_rms` was in "Enabled custom fusions".
  - Run: Qwen3-235B-A22B, dummy weights, TP8, default compilation.
  - Qwen3.6-35B-A3B-FP8 itself cannot load at TP8: `Weight input_size_per_partition = 64 is not divisible by weight quantization block_k = 128`. This is a checkpoint limit, not a node problem.

## 4. Decode profile: Qwen3.6-35B-A3B-FP8, TP1, dummy weights

### How it was run
- Server: `vllm serve` on GPU 0 with `--load-format dummy --language-model-only --max-model-len 4096 --max-num-seqs 64 --no-enable-prefix-caching`.
  - `VLLM_DISABLE_COMPILE_CACHE=1`; launched from inside `~/vllm-kopt` (`vllm.__file__` printed in the log); `Directly load AOT` count = 0.
  - Defaults resolved to `CompilationMode.NONE` + `FULL_AND_PIECEWISE` CUDA graphs.
- Bench: `vllm bench serve` with random 512 in / 128 out, `--ignore-eos`.
  - TPOT: 2 unprofiled runs per concurrency, `--num-prompts 8*C`.
  - Profile: 1 run with `--num-prompts C --profile`.
- Profiler config: torch profiler, `delay_iterations=24, max_iterations=32`, so the window is **32 pure decode steps**. The LM head runs exactly 32 times and the 40 MoE layers 1,280 times.
  - `with_stack=false`, `ignore_frontend=true`.
  - Script: `b1_profile.sh`; tables come from `kernel_table.py`.

### Median TPOT (unprofiled)

| C | run 1 | run 2 | mean | output tok/s |
|---|---|---|---|---|
| 1 | 3.003 ms | 3.004 ms | **3.00 ms** | 316 |
| 8 | 4.920 ms | 4.917 ms | **4.92 ms** | 1,503 |

Under the profiler, ITL stays at 3.01 / 4.93 ms. The profiled runs' TPOT is inflated by profiler start/stop and is not used.

### cuBLAS BF16 GEMM share

Shares are of summed kernel time under the profiler, the same basis as the B200 W11 §7 numbers.

| | c1 | c8 |
|---|---|---|
| summed kernel time / step | 4,972 us | 6,870 us |
| GPU span / step | 3,647 us | 5,006 us |
| launches / step | 1,178 | 1,178 |
| **cuBLAS BF16 GEMMs** | **896 us/step = 18.0 %** | **854 us/step = 12.4 %** |
| cuBLAS BF16 launches / step | 221 | 221 |
| B200 (W11 §7) for comparison | 15.9 % (768 us, 221 launches) | 13.3 % (856 us) |

Per-kernel breakdown, per step. The layer attribution comes from call counts (30 GDN layers, 40 MoE layers, 1 LM head) and matches the B200 W11 mapping; shapes were not recorded.

| kernel | c1 us/step | calls/step | likely layer |
|---|---|---|---|
| `cublasLt::splitKreduce_kernel` | 266.2 | 70 | split-K epilogue of the router + `in_proj_ba` |
| `nvjet_sm103_tst_32x64_64x16_2x1_v_bz_splitK_TNN` | 155.0 | 30 | GDN `in_proj_ba` (64, 2048) |
| `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` | 152.0 | 1 | LM head (248320, 2048) |
| `nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 151.3 | 40 | MoE router (256, 2048) |
| `dot_kernel` + `reduce_1Block_kernel` | 100.4 + 71.4 | 40 + 40 | shared-expert gate (1, 2048) |

| kernel | c8 us/step | calls/step | likely layer |
|---|---|---|---|
| `splitKreduce_kernel` (two variants) | 213.3 + 97.7 | 80 + 30 | split-K epilogues |
| `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` | 154.8 | 1 | LM head |
| `nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 154.3 | 40 | router |
| `nvjet_sm103_tst_8x64_64x16_1x1_h_bz_splitK_TNN` | 145.4 | 40 | shared-expert gate |
| `cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8` | 88.4 | 30 | GDN `in_proj_ba` |

These are the same four projections that #69736 targets on SM100: `in_proj_ba`, router, shared-expert gate and LM head. On B300 the c1 share is higher than on B200 (18.0% vs 15.9%), and absolute cuBLAS time per step is about the same (896 vs 768 us at c1; 854 vs 856 us at c8). The step is faster overall here: TPOT 3.00 ms.

Caveat: the shared-expert gate (`dot_kernel` + `reduce_1Block_kernel`, and `nvjet 8x64` at c8) runs on the aux stream at decode (§3 rule 7), so its time is probably off the critical path. Without it, the cuBLAS BF16 share is about 14.6% at c1 and 10.3% at c8.

### Top 20 kernels

The tables are below and in `top20_c1.md` and `top20_c8.md`; the full lists are `profile_c1.kernels.json` and `profile_c8.kernels.json`. "us" is the total over 32 steps.

**c1**

| # | kernel | us | % | calls | cuBLAS BF16 |
|---|---|---|---|---|---|
| 1 | `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, 128, 1, 128, cute::tup...` | 43158.4 | 27.1 | 5120 |  |
| 2 | `bmm_E4m3_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_rM_BN_transOut_dsFp8_schedS_bN_tma_tmaSf_r...` | 12558.8 | 7.9 | 1280 |  |
| 3 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, false, false, 128>(c10::...` | 12292.8 | 7.7 | 5120 |  |
| 4 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, _...` | 8518.4 | 5.3 | 2240 | **yes** |
| 5 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_rM_BN_transOut_dsFp8_schedS_bN_rgTma_c...` | 8147.6 | 5.1 | 1280 |  |
| 6 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16g...` | 6111.5 | 3.8 | 2560 |  |
| 7 | `void moe::dev::routing::routingCustom::routingIndicesBlockKernel<moe::dev::routing::routingCustom::KernelPa...` | 5781.3 | 3.6 | 1280 |  |
| 8 | `void moe::dev::activation::activationDeepSeekKernel<moe::dev::activation::KernelParams<cutlass::float_e4m3_...` | 5400.7 | 3.4 | 1280 |  |
| 9 | `nvjet_sm103_tst_32x64_64x16_2x1_v_bz_splitK_TNN` | 4960.5 | 3.1 | 960 | **yes** |
| 10 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` | 4864.5 | 3.1 | 32 | **yes** |
| 11 | `nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 4842.4 | 3.0 | 1280 | **yes** |
| 12 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, false, true, 128>(c10::B...` | 4098.9 | 2.6 | 1280 |  |
| 13 | `void moe::dev::finalize::finalizeKernelSmallBatch<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cut...` | 4053.4 | 2.5 | 1280 |  |
| 14 | `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous ...` | 3296.6 | 2.1 | 32 |  |
| 15 | `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasG...` | 3212.2 | 2.0 | 1280 | **yes** |
| 16 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 2821.1 | 1.8 | 1280 |  |
| 17 | `_causal_conv1d_update_kernel` | 2718.6 | 1.7 | 960 |  |
| 18 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 2619.8 | 1.6 | 960 |  |
| 19 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat1...` | 2545.1 | 1.6 | 1280 |  |
| 20 | `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP64MultiCtasKvCgaVarSeqQ8Kv256StaticGroupedSwapsAbForGen` | 2359.8 | 1.5 | 320 |  |

**c8**

| # | kernel | us | % | calls | cuBLAS BF16 |
|---|---|---|---|---|---|
| 1 | `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, 128, 1, 128, cute::tup...` | 45037.5 | 20.5 | 5120 |  |
| 2 | `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x2x2x3_bN_tma_tmaS...` | 38904.0 | 17.7 | 1280 |  |
| 3 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x2x2x3_bN_rgTma_...` | 25281.6 | 11.5 | 1280 |  |
| 4 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, false, 128>(c10::B...` | 13811.0 | 6.3 | 5120 |  |
| 5 | `void moe::dev::activation::activationDeepSeekKernel<moe::dev::activation::KernelParams<cutlass::float_e4m3_...` | 9719.9 | 4.4 | 1280 |  |
| 6 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16g...` | 7344.3 | 3.3 | 2560 |  |
| 7 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 6852.1 | 3.1 | 960 |  |
| 8 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, _...` | 6825.0 | 3.1 | 2560 | **yes** |
| 9 | `void moe::dev::routing::routingCustom::routingIndicesDynBlockKernel<moe::dev::routing::routingCustom::Kerne...` | 6114.4 | 2.8 | 1280 |  |
| 10 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` | 4952.5 | 2.2 | 32 | **yes** |
| 11 | `nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 4937.7 | 2.2 | 1280 | **yes** |
| 12 | `nvjet_sm103_tst_8x64_64x16_1x1_h_bz_splitK_TNN` | 4652.2 | 2.1 | 1280 | **yes** |
| 13 | `void moe::dev::finalize::finalizeKernelSmallBatch<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cut...` | 4425.9 | 2.0 | 1280 |  |
| 14 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, 128>(c10::BF...` | 4373.6 | 2.0 | 1280 |  |
| 15 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat1...` | 3926.9 | 1.8 | 1280 |  |
| 16 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 3869.1 | 1.8 | 1280 |  |
| 17 | `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous ...` | 3491.2 | 1.6 | 32 |  |
| 18 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, __nv_bfl...` | 3125.2 | 1.4 | 960 | **yes** |
| 19 | `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP64MultiCtasKvVarSeqQ8Kv256StaticGroupedSwapsAbForGen` | 3021.7 | 1.4 | 320 |  |
| 20 | `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(cutlass_80_tensorop_s16816g...` | 2827.5 | 1.3 | 960 | **yes** |

### Other things that stand out
- **FP8 blockwise linears** (`cutlass_3x_gemm_fp8_blockwise`, 160/step): 27.1% (c1) and 20.5% (c8), the largest item, as on B200.
- **`per_token_group_quant_8bit`** (160 + 40 launches/step): 10.3% at c1 and 8.3% at c8. This is launch-count-bound, the same as on B200.
- **Attention runs `fmhaSm100f…` (trtllm-gen) on SM103**; it is 1.5% at c1.
- **Sampler `cunn_SoftMaxForward`**: one launch per step, ~103 us at c1.

## 5. Suggested next step
B2, as planned. The B300 numbers say the four #69736 projections are a slightly larger share here than on B200 at c1. The LM head (152 us, 1 launch) and the router + `in_proj_ba` with their 70 split-K reduce launches are the main targets.
