# W14 — next opportunities on Qwen3.8-Flash-Next TEP4 and Qwen3.5-397B TP4 (interim after step 1, 06:17Z)

Status: **step 1 (Flash-Next) and the step 2 profile are taken. Ranking (step 3) not done yet; first candidates are listed at the end.**

## Setup
- Code: `sm100-skinny-qwen35-moe` at 740ff0d02c85, second worktree, `PYTHONPATH` set to it; each server log shows the loaded path and that the `(10, 0)` table is present.
- This node cannot build the C++ extension (not approved), so the compiled library is older than this head. The new `gdn_prefill_conv` op is missing; the code checks for it and falls back. It is a prefill op; decode is not affected.
- B1/B6 method: torch profiler, skip 24 engine steps, record 32; one unprofiled TPOT run (512/512) per concurrency in the same server. Compile cache off.
- Flash-Next: **real weights**, TEP4 on GPUs 2, 3, 6, 7, recipe flags, no speculative decoding. 397B: TP4, dummy weights, `--max-model-len 8192`.
- Not idle: the owner's job on GPUs 0-1 and my W13b A/B on GPUs 4-5 ran at the same time.
- Numbers are from rank 0 of 4. One profiled run per point.

## Flash-Next TEP4: TPOT and busy union
| requests | TPOT p50 (ms) | busy union (ms/step) | summed kernel time (ms/step) | overlap (ms/step) | launches/step |
|---|---|---|---|---|---|
| 1 | 4.005 | 3.852 | 5.475 | 1.622 | 1572 |
| 4 | 5.313 | 5.181 | 6.731 | 1.550 | 1657 |
| 16 | 6.658 | 6.425 | 8.920 | 2.495 | 1657 |

Busy union is 96-97 % of TPOT at all three points, so the step is GPU-bound and kernel time on the critical path converts to TPOT.

## Flash-Next TEP4: time by kind

q38_tep4_c1.kernels.json: 1572 launches/step, 5481 us summed/step
| kind | launches/step | us/step | % of kernel time |
|---|---|---|---|
| hyper-connection (hc_*) | 290 | 820 | 15.0 |
| skinny GEMM (CuTe DSL) | 230 | 811 | 14.8 |
| routed MoE GEMM (bmm) | 96 | 752 | 13.7 |
| cuBLAS BF16 GEMM/GEMV | 243 | 735 | 13.4 |
| MoE routing/activation/finalize | 144 | 559 | 10.2 |
| all-reduce | 99 | 488 | 8.9 |
| torch elementwise/index/copy | 174 | 484 | 8.8 |
| attention (fmha, QSA, rope, cache) | 75 | 273 | 5.0 |
| GDN (conv, recurrent, norm, gates) | 72 | 197 | 3.6 |
| FP8 group quant | 48 | 126 | 2.3 |
| act_and_mul | 48 | 90 | 1.6 |
| norm (rms/layer) | 38 | 76 | 1.4 |
| other | 14 | 51 | 0.9 |
| sampling | 1 | 20 | 0.4 |

q38_tep4_c4.kernels.json: 1657 launches/step, 6736 us summed/step
| kind | launches/step | us/step | % of kernel time |
|---|---|---|---|
| cuBLAS BF16 GEMM/GEMV | 340 | 1173 | 17.4 |
| routed MoE GEMM (bmm) | 96 | 1095 | 16.3 |
| hyper-connection (hc_*) | 290 | 993 | 14.7 |
| skinny GEMM (CuTe DSL) | 181 | 774 | 11.5 |
| torch elementwise/index/copy | 211 | 625 | 9.3 |
| all-reduce | 99 | 611 | 9.1 |
| MoE routing/activation/finalize | 144 | 553 | 8.2 |
| attention (fmha, QSA, rope, cache) | 75 | 281 | 4.2 |
| GDN (conv, recurrent, norm, gates) | 72 | 214 | 3.2 |
| FP8 group quant | 48 | 149 | 2.2 |
| act_and_mul | 48 | 99 | 1.5 |
| norm (rms/layer) | 38 | 79 | 1.2 |
| other | 14 | 53 | 0.8 |
| sampling | 1 | 35 | 0.5 |

q38_tep4_c16.kernels.json: 1657 launches/step, 8934 us summed/step
| kind | launches/step | us/step | % of kernel time |
|---|---|---|---|
| routed MoE GEMM (bmm) | 96 | 2161 | 24.2 |
| cuBLAS BF16 GEMM/GEMV | 437 | 1906 | 21.3 |
| hyper-connection (hc_*) | 290 | 1059 | 11.9 |
| MoE routing/activation/finalize | 144 | 963 | 10.8 |
| torch elementwise/index/copy | 211 | 717 | 8.0 |
| all-reduce | 99 | 707 | 7.9 |
| skinny GEMM (CuTe DSL) | 84 | 341 | 3.8 |
| GDN (conv, recurrent, norm, gates) | 72 | 339 | 3.8 |
| attention (fmha, QSA, rope, cache) | 75 | 302 | 3.4 |
| FP8 group quant | 48 | 151 | 1.7 |
| act_and_mul | 48 | 98 | 1.1 |
| norm (rms/layer) | 38 | 87 | 1.0 |
| other | 14 | 56 | 0.6 |
| sampling | 1 | 47 | 0.5 |



## Flash-Next TEP4: top 20 kernels, 1 request (us over 32 steps)
| # | kernel | us | % | calls | cuBLAS BF16 |
|---|---|---|---|---|---|
| 1 | `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schedS_bN_tma_tmaSf_rgTm...` | 15729.2 | 9.0 | 1536 |  |
| 2 | `void flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel<(unsigned char)4, __nv_bfloat16, fals...` | 14722.5 | 8.4 | 3104 |  |
| 3 | `nvjet_sm100_tst_128x8_64x12_4x1_v_bz_TNT` | 11450.1 | 6.5 | 3104 | **yes** |
| 4 | `kernel_cutlass_kernel_vllmmodelsqwen4_expnvidiaopscute_dsl_hc_down_silu_fmaHcDownSiluFma_object_at__tensorp...` | 9823.0 | 5.6 | 3072 |  |
| 5 | `_hc_combine_norm_kernel` | 8664.9 | 4.9 | 3040 |  |
| 6 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schedS_bN_rgTma_clmp_d...` | 8340.0 | 4.8 | 1536 |  |
| 7 | `_hc_gate_mix_kernel` | 7643.9 | 4.4 | 3104 |  |
| 8 | `void moe::dev::routing::routingCustom::routingIndicesBlockKernel<moe::dev::routing::routingCustom::KernelPa...` | 7390.5 | 4.2 | 1536 |  |
| 9 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 6567.0 | 3.7 | 2688 |  |
| 10 | `void moe::dev::activation::activationDeepSeekKernel<moe::dev::activation::KernelParams<cutlass::float_e4m3_...` | 6135.8 | 3.5 | 1536 |  |
| 11 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 6124.2 | 3.5 | 1152 |  |
| 12 | `nvjet_sm100_tst_24x64_64x16_4x1_v_bz_TNN` | 5333.8 | 3.0 | 1536 | **yes** |
| 13 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 5210.9 | 3.0 | 1536 |  |
| 14 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 4702.7 | 2.7 | 1568 |  |
| 15 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 4436.5 | 2.5 | 1536 |  |
| 16 | `void moe::dev::finalize::finalizeKernelSmallBatch<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cut...` | 4351.1 | 2.5 | 1536 |  |
| 17 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, false, true, 128>(c10::B...` | 4034.8 | 2.3 | 1536 |  |
| 18 | `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous ...` | 3474.3 | 2.0 | 32 |  |
| 19 | `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasG...` | 3401.6 | 1.9 | 1536 | **yes** |
| 20 | `_causal_conv1d_update_kernel` | 3364.0 | 1.9 | 1152 |  |

Top-30 tables for c1 / c4 / c16 and all kernels: `kernels/q38_tep4_top_c*.md`, `kernels/q38_tep4_c*.kernels.json`.

## cuBLAS BF16 GEMMs still present on Flash-Next
The generic table of #69736 has no K = 2560 shape; Flash-Next uses its own table (`qwen4_exp` low-latency GEMM, SM100 section). Shapes below are not in it, or are in it only for small M. Mapping is by calls per step plus a standalone `F.linear` probe of the kernel names (`kernels/cublas_names_q38.json`).

| layer | shape (N, K) | calls/step | 1 request | 4 requests | 16 requests |
|---|---|---|---|---|---|
| hyper-connection up projection (`input_mix_weight_up`) | (10240, 320) | 97 | `nvjet 128x8_64x12_4x1`, 3.7 us, **358 us/step** | 3.8 us, 370 us/step | `nvjet 192x8_64x8_4x2`, 4.4 us, 425 us/step |
| shared-expert gate | (1, 2560) | 48 | `dot_kernel` + `reduce_1Block`, 4.0 us, **194 us/step** | 4.6 us, 220 us/step | `nvjet splitK` + `splitKreduce`, about 6.7 us, about 320 us/step |
| shared-expert `down_proj` (TEP4 shard) | (2560, 160) | 48 | `nvjet 24x64_64x16_4x1`, 3.5 us, **167 us/step** | `cutlass_80 ... relu`, 3.4 us, 164 us/step | part of `nvjet 64x8_64x16_4x2` (96 calls, 385 us/step) |
| PLE `kv_proj` | (12800, 2560) | 1 | split-K + reduce, 17 us/step | 15 + 3 us/step | 14 us/step |
| router | (512, 2560) | 48 | skinny | **cuBLAS** split-K + reduce, about 7.3 us, about 350 us/step | same, about 370 us/step |
| LM head | (62080, 2560) | 1 | skinny | **cuBLAS**, 50 us/step | 48 us/step |
| GDN `out_proj` | (2560, 1536) | 48 | skinny | skinny | **cuBLAS** (other half of `nvjet 64x8_64x16_4x2`) |
| attention / GDN input projections | (4096, 2560), (4224, 2560) | 48 | skinny | skinny | **cuBLAS** `nvjet 64x8_64x16_2x2`, 7.2 us, 346 us/step |

- cuBLAS BF16 share of kernel time: **13.4 % at 1 request (735 us/step, 243 launches), 17.4 % at 4, 21.3 % at 16 (1906 us/step, 437 launches).** Skinny launches per step fall from 230 to 181 to 84 as M leaves the table.
- Standalone times and kernel counts for M = 1 to 32: `kernels/cublas_names_q38.json`.

## FP8 quant launches on Flash-Next
48 per step, one per layer, 2.6 us (1 request) to 3.1 us: the routed-experts input quant (UE8M0 scales variant of the kernel). The shared expert and the attention projections are BF16 here, so there are no other quant launches. Small: 2.3 % of kernel time.

## Kernels that run back to back (one layer, 1 request: `layer_q38_c1.txt`)
The layer is 71.5 us and almost fully serial. The same five-kernel chain runs **twice per layer** (97 times per step):

all-reduce (4.4 us) -> `_hc_combine_norm_kernel` (3.1) -> `hc_down_silu_fma` (3.3) -> cuBLAS up projection (10240, 320) (3.7) -> `_hc_gate_mix_kernel` (2.4)

That is about 16 us per pass including the gaps, 32 us of the 71.5 us layer. Other back-to-back groups: the shared-expert gate is dot -> reduce -> sigmoid -> mul (4 launches on the aux stream); `layer_norm_fwd_kernel` -> skinny `out_proj` in the GDN path; routed add after finalize.

At 1 request the shared and routed branches end at the same time (16.9 us vs 16.6 us after the gate-up GEMM start), so shortening only the shared branch gains little here.

## Qwen3.5-397B TP4 (dummy weights): first numbers
| requests | TPOT p50 (ms) | busy union (ms/step) | summed kernel time (ms/step) | launches/step |
|---|---|---|---|---|
| 1 | 5.556 | 5.420 | 7.623 | 1590 |
| 8 | 7.839 | 7.711 | 11.175 | 1591 |

q35_tp4_c1.kernels.json: 1590 launches/step, 7759 us summed/step
| kind | launches/step | us/step | % of kernel time |
|---|---|---|---|
| block FP8 linear GEMM (cutlass) | 240 | 2149 | 27.7 |
| routed MoE GEMM (bmm) | 120 | 1455 | 18.8 |
| all-reduce | 122 | 902 | 11.6 |
| FP8 group quant | 300 | 766 | 9.9 |
| MoE routing/activation/finalize | 180 | 723 | 9.3 |
| skinny GEMM (CuTe DSL) | 166 | 528 | 6.8 |
| torch elementwise/index/copy | 193 | 471 | 6.1 |
| GDN (conv, recurrent, norm, gates) | 150 | 337 | 4.3 |
| attention (fmha, QSA, rope, cache) | 47 | 243 | 3.1 |
| act_and_mul | 60 | 114 | 1.5 |
| sampling | 1 | 34 | 0.4 |
| other | 10 | 34 | 0.4 |
| norm (rms/layer) | 1 | 2 | 0.0 |

q35_tp4_c8.kernels.json: 1591 launches/step, 11174 us summed/step
| kind | launches/step | us/step | % of kernel time |
|---|---|---|---|
| routed MoE GEMM (bmm) | 120 | 3565 | 31.9 |
| block FP8 linear GEMM (cutlass) | 240 | 2423 | 21.7 |
| MoE routing/activation/finalize | 180 | 1131 | 10.1 |
| FP8 group quant | 300 | 912 | 8.2 |
| all-reduce | 122 | 804 | 7.2 |
| torch elementwise/index/copy | 194 | 618 | 5.5 |
| skinny GEMM (CuTe DSL) | 165 | 596 | 5.3 |
| GDN (conv, recurrent, norm, gates) | 150 | 476 | 4.3 |
| attention (fmha, QSA, rope, cache) | 47 | 275 | 2.5 |
| act_and_mul | 60 | 192 | 1.7 |
| cuBLAS BF16 GEMM/GEMV | 1 | 76 | 0.7 |
| sampling | 1 | 54 | 0.5 |
| other | 10 | 49 | 0.4 |
| norm (rms/layer) | 1 | 3 | 0.0 |


- No cuBLAS BF16 GEMM is left in the layers (0 launches at 1 request; 1 per step at 8 requests, the LM head).
- The block FP8 linear GEMM (cutlass) is the largest item: 240 launches per step (4 per layer), **9.0 us each, 27.7 % of kernel time at 1 request**. FP8 group quant: 300 launches per step, 9.9 %.
- Dummy weights make the routing random, so the routed-MoE numbers at 8 requests are an upper bound on the number of distinct experts.
- Top tables: `kernels/q35_tp4_top_c*.md`.

## First candidates (not ranked yet, no PR search yet)
1. **Flash-Next hyper-connection chain** (97 per step): the cuBLAS up projection (10240, 320) is on the critical path with no plan, and it sits between two small custom kernels. Fewer launches in this chain is worth about 2.5-3.7 us x 97 per kernel removed (6-9 % of TPOT at 1 request).
2. **Flash-Next table gaps**: (10240, 320), (1, 2560), (2560, 160) at all M; router, LM head, input projections and `out_proj` at larger M. 13 % of kernel time at 1 request, 21 % at 16.
3. **397B (and Qwen3.6): the block FP8 linear GEMM at small M** is 9 us per call where a BF16 skinny GEMM of similar size is 3-5 us.

Next: step 3 (sizes, open upstream PRs, bit-identity), a standalone check of what a skinny plan gives on the three missing Flash-Next shapes, and the 397B launch map.
