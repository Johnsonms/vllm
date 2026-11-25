# W14 — next opportunities on Qwen3.8-Flash-Next TEP4 and Qwen3.5-397B TP4 (interim after step 1, first posted 06:17Z)

Status: **done (06:34Z). Ranking is in the last section.** No code was edited for W14.

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

## Correction to the interim: two of the three "table gaps" are not plan gaps
Standalone sweep of skinny configs against cuBLAS (W11 tool, CUDA graph, cold L2, one seed; `sweep/*.jsonl`):

| shape (N, K) | M=1 | M=2 | M=4 | M=8 | M=16 |
|---|---|---|---|---|---|
| hyper-connection up (10240, 320) | 1.06x | 0.97x | 0.82x | 0.60x | 0.44x |
| shared `down_proj` TEP4 (2560, 160) | no config runs | - | - | - | - |
| shared gate (1, 2560) | 2.08x | 2.16x | 1.97x | 2.23x | 1.75x |
| router (512, 2560) | 2.18x | 2.08x | 1.96x | 1.66x | 1.18x |
| GDN `out_proj` (2560, 1536) | 1.38x | 1.55x | 1.26x | 0.97x | 0.61x |
| input projection (4096, 2560) | 1.21x | 1.36x | 1.14x | 0.88x | 0.52x |

- (10240, 320) and (2560, 160) are short and wide (small K, many outputs). The skinny kernel does not help them, even with block sizes down to 10. They need a different kernel, not a table line.
- The M=1 column ran while another of my jobs touched the same GPU; a rerun is in progress and I will correct the column if it moves.
- Above M=8 the large projections are better left on cuBLAS, so the 21 % cuBLAS share at 16 requests is mostly not recoverable by plans.

## The fused up-projection + gate-mix kernel exists but is off at decode sizes
Main already has `qwen4_exp_hc_up_gate_mix` for SM100 (C++ op, present in my build). The model uses it only from M = 48 (`MIN_FUSED_UP_M`), i.e. for prefill. Standalone (`hc_up_bench.py`, CUDA graph, cold L2):

| M | cuBLAS up projection + `hc_gate_mix` (us) | up projection alone (us) | existing fused op (us) |
|---|---|---|---|
| 1 | 5.86 | 4.29 | 6.82 |
| 4 | 6.14 | 4.29 | 6.94 |
| 16 | 6.78 | 4.67 | 6.94 |
| 32 | 7.07 | 4.70 | 6.85 |
| 48 | 7.68 | 4.96 | 6.82 |

So the threshold is right for the kernel as it is (it has a flat cost of about 6.9 us), and lowering it would lose. The up projection reads 6.5 MB of weights in 4.3 us; the BF16 skinny kernels on this GPU read 20-40 MB in 5-10 us. A kernel sized for M = 1-16 has room to be 2-3 us for the pair. The fused op is already not bit-identical to the unfused pair (max difference 0.004-0.008).

## Qwen3.5-397B TP4: where the layer goes (1 request, 85.9 us per layer, dummy weights)
FP8 linear layers at TP4 (`walk_fp8.py`), with in-graph times of the quant + GEMM pair, and the BF16 skinny time for the same shape as a reference (standalone, cold L2):

| layer | weight (N, K) | quant + FP8 GEMM in the model (us) | BF16 skinny, same shape, M=1 / M=4 (us) |
|---|---|---|---|
| input projection (GDN `in_proj_qkvz` / attention `qkv_proj`) | (5120, 4096) / (4608, 4096) | 3.1 + 11.2 | 9.8 / 11.5 |
| `out_proj` | (4096, 2048) | 2.6 + 8.0 | 5.7 / 6.7 |
| shared `gate_up_proj` | (512, 4096) | 2.2 + 11.0 | 3.3 / 3.6 |
| shared `down_proj` | (4096, 256) | 2.6 + 6.1 | 2.8 / 3.4 |

- These four pairs are 44 us of kernel time per 85.9 us layer.
- The shared branch is the longer one here too: its last kernel ends about 10 us after the routed finalize (34.2 vs 24.5 us after the gate-up GEMM start). The shared `down_proj` GEMM starts 7 us after its input is ready (it waits behind the routed GEMM). So this model has the same pattern as Qwen3.6 in W12 / W13.
- FP8 weights are half the bytes of BF16, so the BF16 skinny times are an upper bound for a W8A16 skinny kernel at the same shape.

## Step 3: ranking
Gain = kernel time removed from the critical path per step / TPOT. These are estimates from profiles and standalone timings, not A/Bs.

| rank | candidate | model | expected TPOT gain | bit-identical | size |
|---|---|---|---|---|---|
| 1 | W8A16 skinny GEMM for block-FP8 linear layers on SM100, M = 1-4 | Qwen3.5-397B (also Qwen3.6-35B) | about -10 % at 1 request on 397B TP4; estimate range -8 to -15 % | no | medium-large, depends on #64980 |
| 2 | fused up-projection + gate-mix kernel for M = 1-16 | Qwen3.8-Flash-Next | about -6 % at 1 request, -4 % at 16 | no (same as the existing fused op) | medium: one new kernel + dispatch |
| 3 | Flash-Next SM100 table: (1, 2560) for all M, router (512, 2560) for M = 3-16, LM head M = 3-4 | Qwen3.8-Flash-Next | about -3 % at 4 requests, -1 % or less at 1 and 16 | no (skinny vs cuBLAS rounding, as in #69736) | small: table lines + verification |

**1. W8A16 skinny GEMM for the block-FP8 linear layers on SM100.**
- What changes: at 1-4 tokens the layer reads the BF16 activation directly and the FP8 weight with its 128x128 block scales, instead of quantizing the activation and running the cutlass block-FP8 GEMM. This removes the quant launch as well.
- Estimate for 397B TP4 at 1 request. Main-stream pairs: input projection 14.3 -> at most 9.8 us, `out_proj` 10.6 -> at most 5.7 us: 9.4 us per layer. Shared branch: 13.2 -> 3.3 and 8.7 -> 2.8 us, but it only counts until the routed branch becomes the longer one (about 10 us of lag today). Taking 9.4 us on the main stream alone: 9.4 x 60 = 564 us = **-10 %** of the 5.556 ms TPOT. If the shared branch stops being critical, part of its 10 us lag adds to that (upper end about -15 %). With half the bytes the real kernel should beat the BF16 reference, which I did not measure.
- Upstream: **#64980** (open) is this kernel for Hopper only, wired into the FlashInfer / DeepGEMM block-scale path with row-major scales; **#64983** (draft) is the routed-experts version, also SM90. Nothing for SM100. On SM100 the layer uses the cutlass block kernel with column-major scales (W12b), so the dispatch point and the scale layout differ.
- Bit-identical: no. The activation is no longer quantized to FP8.
- Size: SM100 plans for the five shapes (M = 1-4), a dispatch in the cutlass block-FP8 path, scale-layout handling. A few hundred lines on top of #64980.
- Relation to the excluded items: at M = 1-4 this makes the SiLU-mul + quant fusion (#69778) and the RMSNorm + quant fusion unnecessary, because there is no activation quant left. They still matter above 4 tokens.

**2. Flash-Next: fused up-projection + gate-mix for decode sizes.**
- What changes: a small-M variant of the existing fused op (or a second code path in it) so that the cuBLAS (10240, 320) GEMM and `_hc_gate_mix_kernel` become one launch. 97 pairs per step.
- Estimate: the pair is 3.7 + 2.4 us in the model at 1 request (5.9 us standalone). If one kernel does it in about 3.5 us, that is 2.5 us x 97 = 240 us = **-6 %** of 4.005 ms. At 16 requests the pair is 4.4 + about 2.6 us of a 6.658 ms step: about -4 %. The layer is serial at these points (busy union 96-97 % of TPOT), so the time converts.
- Upstream: #64898, #66791, #69122 (merged) made the fused kernel and set the threshold to 48 on SM100; on SM90 the Triton version is used at all M. **#66865** (open, 37k lines) is an opt-in specialized decode for this model on GB300 only; I did not check whether it replaces this chain there. Nothing open for SM100 at small M.
- Bit-identical: no, same as the existing fused op (FP32 gate before the sigmoid).
- Size: one kernel (the existing C++ op is about 570 lines) plus the dispatch line. Needs a source build, which is not approved on this node.
- Further in the same chain, larger: the all-reduce (4.4 us) is followed directly by `_hc_combine_norm_kernel` (3.1 us), also 97 times per step. Folding the combine + norm into the all-reduce epilogue would be another launch per pass, but it is a FlashInfer kernel change.

**3. Flash-Next SM100 table additions.**
- What changes: plan lines in the model's own table for the shared-expert gate (1, 2560) at M = 1-16 (about 2x), the router (512, 2560) at M = 3-16 (2.0x at 4, 1.7x at 8, 1.2x at 16), and the LM head (62080, 2560) at M = 3-4 (not swept here; the 62080-row LM head gained at M = 4 in W11).
- Estimate: at 4 requests the router pair is about 7.3 us in the model against about 3.2 us skinny: 4 us x 48 = 190 us = **-3.6 %** of 5.313 ms. At 1 request only the shared gate is on cuBLAS and the shared and routed branches end together, so the gain is small (under 1 %). At 16 requests the router gain is about 1 us x 48.
- Upstream: none for this table on SM100 (#67622 is a different model on SM120; #69736 does not touch K = 2560).
- Bit-identical: no, as for every skinny plan.
- Size: about 15 table lines, plus the 4-seed verification from W11.

**Not ranked, but cheap to test:** P (aux-stream priority, W13) on Qwen3.5-397B. The 397B trace shows the same starved shared branch as Qwen3.6. It is one argument and output-identical. I could not run the paired A/B: TP4 needs 8 GPUs for two servers and 6 are available.

## Caveats
- All estimates assume the removed time is on the critical path. For the shared branch that is only true up to its current lag.
- 397B used dummy weights: kernel times of the linear layers do not depend on the weights, but the routed-MoE part does.
- The node was shared the whole time (owner's job on GPUs 0-1, my W13b A/B on GPUs 4-5).
- The compiled library on this node is older than the Python tree (no source build). I checked that the fused hyper-connection op is present, so its absence at decode is the M >= 48 rule, not my build.

## Files
`kernels/` (tables, all kernels, cuBLAS name probes), `layer_q38_c1.txt`, `sweep/` (skinny sweeps), `prof.sh`, `kinds.py`, `cublas_names.py`, `hc_up_bench.py`, `sweep_small_k.py`, `walk_fp8.py`.
