# W5 — independent hotspot review, Qwen3.8-Flash-Next-FP8 at TEP4

Worker node, GPUs 4-7 (idle before each run; GPUs 0-1 busy with another workload), vLLM 8399da3780, dummy weights,
no MTP, `--routing balanced`. Profiles use the W3b fixes applied locally (replay split by `cudaGraphLaunch`
correlation, 50 ms flush, #62300's `profile.py` hunk). All local edits are reverted.

## Summary

- **Correctness finding, outside the task but important: prefill above 2,048 tokens is not reproducible.** On unmodified
  code, the same input from the same restored state gives a different output on the next run, by 1.4-2.5 % relative RMS.
  See the last section. This also means A/B quality checks at long prompts carry run-to-run noise.
- **Most of the step is already covered or near memory bandwidth.** I found no large uncovered item. The three
  candidates below are worth about 12 % (decode at 256 requests, uncertain), 1.7 % and about 1 %.

## Candidates (not on the covered list, not implemented)

| # | Candidate | Share of busy time per step | Hypothesis |
|---|---|---|---|
| 1 | Linear-attention decode state update: `fused_recurrent_gated_delta_rule_packed_decode_kernel` (`third_party/flash_linear_attention/ops/fused_recurrent.py`, Triton, 36 calls) | d1 1.6 %, d64 4.3 %, **d256 12.4 %** | Time scales linearly with requests (2.5 / 17 / 70 us per call), so it is bound by reading and writing the recurrent state. If the state is bf16 it moves about 200 MB per call at about 2.9 TB/s, a third of what `_hc_combine_norm_kernel` reaches; a wider value tile than `BV = 32` (4 programs per head) is the first thing to try. |
| 2 | Per-step all-gather: `tensor_model_parallel_all_gather` -> `CudaCommunicator.all_gather` (1 call) | d64 about 0.7 % (73 + 33 us), **d256 1.7 %** (192 + 153 us) | NCCL ring all-gather into a dim-0 buffer, then `movedim` + `reshape`, which copies when `dim != 0`. The code comment there already names this extra copy. Gathering straight into the final layout removes the 153 us copy. |
| 3 | MoE input FP8 quantization: `per_token_group_quant_8bit_register_kernel` (`quantization/utils/fp8_utils.py::per_token_group_quant_fp8`, 48 calls) | 1.1-2.1 % in every case; p8k 1.2 % | At 8,192 tokens it takes 26.6 us per call for about 63 MB in + out, about 2.4 TB/s. The "register" variant looks tuned for small batches; a wider variant for large `N` should approach memory bandwidth. At decode it is launch-bound (2.6 us per call), so only fusing it into the neighbour would help there. |

**How sure:**
- Candidate 1 depends on two things I did not verify: the state dtype (fp32 would put it at about 5.8 TB/s, near
  bandwidth, and remove the candidate) and that each rank holds 12 of the 48 value heads at TP4. Check these first.
- Candidate 2's numbers are measured; I did not resolve which model call issues the gather.
- Candidate 3's bandwidth estimate assumes bf16 input of 8,192 x 2,560 and fp8 output.

## Looked at and rejected

| Item | Share | Why not a candidate |
|---|---|---|
| `_hc_combine_norm_kernel` (`ops/hc.py::hc_combine_norm`, 95 calls) | 4.5-8.8 % | At 8,192 tokens it moves about 546 MB per call (reads residual and block output, writes `out` and `y`) in 86 us, about 6.3 TB/s. Already near memory bandwidth. |
| `ncclDevKernel_AllReduce_Sum_u8_RING_LL` at d1 (1 call) | 7.6 % of d1 on rank 0 | Waiting, not work: 16 us in the eager run, and 88 / 178 / 256 / 457 us across the four ranks in the graph. It is the first collective of the forward and absorbs rank skew. 18-21 us at d64 and d256. |
| `aten::add` in `fused_moe/runner/moe_runner.py::forward` (shared + routed output) | 0.5 % d256, 0.9 % p8k | Real but small; could be folded into the MoE finalize kernel. |
| `aten::copy_` under `layernorm.py::_fla_rms_norm_gated` | 0.8 % d256, 1.1 % p8k | Looks like the covered "GDN norm z copy" (#50279). |
| Elementwise ops in `qwen3_moe.py` shared-expert `forward` | 0.9 % d256, 2.2 % p8k | Covered by the shared-expert gate fusion. |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` at p8k | 13.3 % | Covered (prefill NCCL above 32 MB). |
| `_qsa_sparse_paged_decode_kernel`, `activationDeepSeekKernel`, cuBLAS split-K, sampler kernels, skinny GEMM | — | On the covered list. |

## Top 15 kernels per case (rank 0)

Times are per forward, averaged over 3 replays, from `kernels.json` with the fixed replay split. Call sites come from one
eager pass per case under `torch.profiler(with_stack=True)`: kernel -> correlation -> launch call -> innermost enclosing
vLLM Python function. The line number is the function's `def` line, not the calling line.
Event-timed medians: d1 3,974 us; d64 10,896 us; d256 17,024 us; p2k 31,116 us; p8k 100,155 us.

### d1 — 1 decode request

Profile span 5,213 us, busy 5,627 us, 1,547 activities per forward.

| # | kernel | us | % | calls | call site (file, function def line) |
|---|---|---|---|---|---|
| 1 | `cutlass…nelslinearcute_dsl_skinny_gemmCuteSkinnyGemm` | 802 | 13.3 | 230 | `model_executor/kernels/linear/cute_dsl/skinny_gemm.py(213): __call__` |
| 2 | `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x3…` | 521 | 8.7 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 3 | `ncclDevKernel_AllReduce_Sum_u8_RING_LL` | 457 | 7.6 | 1 | `distributed/device_communicators/pynccl_wrapper.py(505): ncclAllReduce` |
| 4 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceF…` | 413 | 6.9 | 97 | `distributed/device_communicators/flashinfer_all_reduce.py(527): all_reduce` |
| 5 | `nvjet_sm100_tst_128x8_64x12_4x1_v_bz_TNT` | 358 | 6.0 | 97 | `model_executor/layers/utils.py(693): cuda_unquantized_gemm` |
| 6 | `cutlass…diaopscute_dsl_hc_down_silu_fmaHcDownSiluFma` | 308 | 5.1 | 96 | `models/qwen4_exp/nvidia/ops/cute_dsl/hc_down_silu.py(147): __call__` |
| 7 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128_s8_et64x8_m64x8…` | 270 | 4.5 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 8 | `_hc_combine_norm_kernel` | 269 | 4.5 | 95 | `models/qwen4_exp/nvidia/ops/hc.py(482): hc_combine_norm` |
| 9 | `_hc_gate_mix_kernel` | 246 | 4.1 | 97 | `models/qwen4_exp/nvidia/ops/hc.py(162): hc_gate_mix` |
| 10 | `moe::dev::routing::routingCustom::routingIndicesBlock…` | 246 | 4.1 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 11 | `moe::dev::activation::activationDeepSeekKernel` | 199 | 3.3 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 12 | `at::native::vectorized_elementwise_kernel` | 169 | 2.8 | 99 | `model_executor/models/qwen3_moe.py(128): forward` |
| 13 | `nvjet_sm100_tst_24x64_64x16_4x1_v_bz_TNN` | 164 | 2.7 | 48 | `model_executor/layers/utils.py(693): cuda_unquantized_gemm` |
| 14 | `at::native::elementwise_kernel` | 150 | 2.5 | 50 | `model_executor/models/qwen3_moe.py(128): forward` |
| 15 | `moe::dev::finalize::finalizeKernelSmallBatch` | 141 | 2.4 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |

### d64 — 64 decode requests

Profile span 11,045 us, busy 14,357 us, 1,956 activities per forward.

| # | kernel | us | % | calls | call site (file, function def line) |
|---|---|---|---|---|---|
| 1 | `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x3…` | 2,781 | 19.4 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 2 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128_s8_et64x8_m64x8…` | 1,406 | 9.8 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 3 | `cublasLt::splitKreduce_kernel` | 814 | 5.7 | 277 | `model_executor/layers/utils.py(693): cuda_unquantized_gemm` |
| 4 | `flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceK…` | 789 | 5.5 | 97 | `distributed/device_communicators/flashinfer_all_reduce.py(527): all_reduce` |
| 5 | `_hc_combine_norm_kernel` | 751 | 5.2 | 95 | `models/qwen4_exp/nvidia/ops/hc.py(482): hc_combine_norm` |
| 6 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 616 | 4.3 | 36 | `third_party/flash_linear_attention/ops/fused_recurrent.py(344): fused_recurrent_gated_de` |
| 7 | `_hc_gate_mix_kernel` | 566 | 3.9 | 97 | `models/qwen4_exp/nvidia/ops/hc.py(162): hc_gate_mix` |
| 8 | `nvjet_sm100_tst_16x64_64x16_4x1_v_bz_splitK_TNN` | 540 | 3.8 | 96 | `model_executor/layers/utils.py(693): cuda_unquantized_gemm` |
| 9 | `nvjet_sm100_tst_24x64_64x16_4x1_v_bz_TNN` | 484 | 3.4 | 96 | `models/qwen4_exp/nvidia/low_latency_gemm.py(409): _qwen4_exp_low_latency_gemm` |
| 10 | `nvjet_sm100_tst_72x64_64x12_2x1_v_bz_TNN` | 456 | 3.2 | 97 | `model_executor/layers/utils.py(693): cuda_unquantized_gemm` |
| 11 | `nvjet_sm100_tst_64x32_64x16_2x2_2cta_h_bz_splitK_TNT` | 440 | 3.1 | 97 | `models/qwen4_exp/nvidia/low_latency_gemm.py(409): _qwen4_exp_low_latency_gemm` |
| 12 | `moe::dev::activation::activationDeepSeekKernel` | 409 | 2.9 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 13 | `at::native::elementwise_kernel` | 394 | 2.7 | 87 | `model_executor/models/qwen3_moe.py(128): forward` |
| 14 | `moe::dev::routing::routingCustom::routingIndicesClust…` | 376 | 2.6 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 15 | `moe::dev::finalize::finalizeKernelSmallBatch` | 358 | 2.5 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |

### d256 — 256 decode requests

Profile span 17,156 us, busy 20,225 us, 1,860 activities per forward.

| # | kernel | us | % | calls | call site (file, function def line) |
|---|---|---|---|---|---|
| 1 | `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x3…` | 3,697 | 18.3 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 2 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 2,506 | 12.4 | 36 | `third_party/flash_linear_attention/ops/fused_recurrent.py(344): fused_recurrent_gated_de` |
| 3 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128_s8_et64x8_m64x8…` | 1,839 | 9.1 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 4 | `_hc_combine_norm_kernel` | 935 | 4.6 | 95 | `models/qwen4_exp/nvidia/ops/hc.py(482): hc_combine_norm` |
| 5 | `flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceK…` | 849 | 4.2 | 97 | `distributed/device_communicators/flashinfer_all_reduce.py(527): all_reduce` |
| 6 | `nvjet_sm100_tst_64x32_64x16_2x1_2cta_v_bz_splitK_TNT` | 809 | 4.0 | 96 | `model_executor/layers/utils.py(693): cuda_unquantized_gemm` |
| 7 | `_qsa_sparse_paged_decode_kernel` | 718 | 3.5 | 12 | `models/qwen4_exp/nvidia/ops/qsa_fp8.py(301): qsa_sparse_paged_decode` |
| 8 | `moe::dev::activation::activationDeepSeekKernel` | 669 | 3.3 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 9 | `_hc_gate_mix_kernel` | 652 | 3.2 | 97 | `models/qwen4_exp/nvidia/ops/hc.py(162): hc_gate_mix` |
| 10 | `cublasLt::splitKreduce_kernel` | 608 | 3.0 | 181 | `model_executor/layers/utils.py(693): cuda_unquantized_gemm` |
| 11 | `nvjet_sm100_tst_144x128_64x8_1x2_2cta_h_bz_TNN` | 554 | 2.7 | 97 | `model_executor/layers/utils.py(693): cuda_unquantized_gemm` |
| 12 | `at::native::elementwise_kernel` | 498 | 2.5 | 87 | `model_executor/models/qwen3_moe.py(128): forward` |
| 13 | `nvjet_sm100_tst_80x64_64x16_2x2_2cta_h_bz_TNN` | 492 | 2.4 | 96 | `models/qwen4_exp/nvidia/low_latency_gemm.py(409): _qwen4_exp_low_latency_gemm` |
| 14 | `moe::dev::finalize::finalizeKernelVecLoad` | 451 | 2.2 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 15 | `moe::dev::routing::routingCustom::routingIndicesClust…` | 395 | 1.9 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |

### p2k — 2,048-token prefill

Profile span 32,983 us, busy 32,897 us, 1,799 activities per forward.

| # | kernel | us | % | calls | call site (file, function def line) |
|---|---|---|---|---|---|
| 1 | `bmm_E4m3_E4m3E4m3_Fp32_t128x64x128_s6_et64x64_m64x64x…` | 3,979 | 12.1 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 2 | `flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceK…` | 2,984 | 9.1 | 97 | `distributed/device_communicators/flashinfer_all_reduce.py(527): all_reduce` |
| 3 | `_hc_combine_norm_kernel` | 2,904 | 8.8 | 95 | `models/qwen4_exp/nvidia/ops/hc.py(482): hc_combine_norm` |
| 4 | `moe::dev::activation::activationDeepSeekKernel` | 2,596 | 7.9 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 5 | `nvjet_sm100_tst_48x128_64x9_4x1_v_bz_TNN` | 2,441 | 7.4 | 96 | `model_executor/layers/utils.py(693): cuda_unquantized_gemm` |
| 6 | `_qsa_sparse_paged_decode_kernel` | 2,416 | 7.3 | 12 | `models/qwen4_exp/nvidia/ops/qsa_fp8.py(301): qsa_sparse_paged_decode` |
| 7 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x64x128_s6_et64x64_m64…` | 1,989 | 6.0 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 8 | `hc_up_gate_mix_kernel` | 1,635 | 5.0 | 97 | `models/qwen4_exp/nvidia/ops/hc.py(285): hc_up_gate_mix` |
| 9 | `nvjet_sm100_tst_128x256_64x6_2x2_2cta_h_bz_TNT` | 1,105 | 3.4 | 36 | `models/qwen4_exp/nvidia/low_latency_gemm.py(409): _qwen4_exp_low_latency_gemm` |
| 10 | `nvjet_sm100_tst_144x128_64x8_1x2_2cta_h_bz_TNN` | 932 | 2.8 | 96 | `models/qwen4_exp/nvidia/low_latency_gemm.py(409): _qwen4_exp_low_latency_gemm` |
| 11 | `at::native::elementwise_kernel` | 921 | 2.8 | 86 | `model_executor/models/qwen3_moe.py(128): forward` |
| 12 | `moe::dev::finalize::finalizeKernelVecLoad` | 876 | 2.7 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 13 | `cutlass…ated_delta_net_cpCPDeltaRuleFixupUtcmmaSm100` | 769 | 2.3 | 36 | `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py(402): fi_chunk_gated_delta_rule` |
| 14 | `cutlass…net_cp_prefillCPDeltaRulePrefillTcgen05Sm100` | 729 | 2.2 | 36 | `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py(402): fi_chunk_gated_delta_rule` |
| 15 | `cutlass…ta_net_cpCPDeltaRuleMNPrecomputeUtcmma1Sm100` | 678 | 2.1 | 36 | `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py(402): fi_chunk_gated_delta_rule` |

### p8k — 8,192-token prefill

Profile span 101,049 us, busy 101,986 us, 1,798 activities per forward.

| # | kernel | us | % | calls | call site (file, function def line) |
|---|---|---|---|---|---|
| 1 | `_qsa_sparse_paged_decode_kernel` | 15,000 | 14.7 | 12 | `models/qwen4_exp/nvidia/ops/qsa_fp8.py(301): qsa_sparse_paged_decode` |
| 2 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 13,566 | 13.3 | 97 | `distributed/device_communicators/pynccl_wrapper.py(505): ncclAllReduce` |
| 3 | `bmm_E4m3_E4m3E4m3_Fp32_t128x64x128u2_s6_et64x64_m64x6…` | 9,377 | 9.2 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 4 | `moe::dev::activation::activationDeepSeekKernel` | 9,374 | 9.2 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 5 | `_hc_combine_norm_kernel` | 8,197 | 8.0 | 95 | `models/qwen4_exp/nvidia/ops/hc.py(482): hc_combine_norm` |
| 6 | `nvjet_sm100_tst_168x128_64x5_1x4_h_bz_TNN` | 6,505 | 6.4 | 145 | `model_executor/layers/utils.py(693): cuda_unquantized_gemm` |
| 7 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x64x128_s6_et64x64_m64…` | 4,931 | 4.8 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 8 | `hc_up_gate_mix_kernel` | 4,389 | 4.3 | 97 | `models/qwen4_exp/nvidia/ops/hc.py(285): hc_up_gate_mix` |
| 9 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3,908 | 3.8 | 37 | `models/qwen4_exp/nvidia/low_latency_gemm.py(409): _qwen4_exp_low_latency_gemm` |
| 10 | `at::native::elementwise_kernel` | 3,601 | 3.5 | 86 | `model_executor/models/qwen3_moe.py(128): forward` |
| 11 | `moe::dev::finalize::finalizeKernelVecLoad` | 2,370 | 2.3 | 48 | `model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py(526): _apply_block_scale` |
| 12 | `nvjet_sm100_tst_128x256_64x6_2x2f_2cta_h_bz_TNT` | 2,049 | 2.0 | 48 | `models/qwen4_exp/nvidia/low_latency_gemm.py(409): _qwen4_exp_low_latency_gemm` |
| 13 | `nvjet_sm100_tst_128x192_64x7_2x1_2cta_v_bz_TNT` | 1,891 | 1.9 | 60 | `models/qwen4_exp/nvidia/low_latency_gemm.py(409): _qwen4_exp_low_latency_gemm` |
| 14 | `cutlass…net_cp_prefillCPDeltaRulePrefillTcgen05Sm100` | 1,347 | 1.3 | 36 | `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py(402): fi_chunk_gated_delta_rule` |
| 15 | `per_token_group_quant_8bit_register_kernel` | 1,278 | 1.2 | 48 | `model_executor/layers/quantization/utils/fp8_utils.py(633): per_token_group_quant_fp8` |

Note on d1: the profile span (5,213 us) is well above the event-timed median (3,974 us). I did not chase this; the
rendezvous-style wait in the u8 all-reduce above accounts for part of it on rank 0.

## Correctness finding: prefill above 2,048 tokens is not reproducible

The benchmark reports `eager_repeat` (two eager runs from the same restored state), `graph_vs_eager`, and
`graph_deterministic` (two graph replays). All three should be exact, and are for every decode case.

Unmodified vLLM 8399da3780, no profiling, TEP4, dummy weights, one request per case (relative RMS of the difference;
identical on all 4 ranks):

| prompt tokens | eager repeat | graph vs eager | graph deterministic |
|---|---|---|---|
| 2,048 | 0 | 0 | yes |
| 2,176 | 0.0141 | 0.0146 | **no** |
| 2,560 | 0.0150 | 0.0140 | **no** |
| 3,072 | 0.0186 | 0.0170 | **no** |
| 4,096 | 0.0206 | 0.0207 | **no** |
| 6,144 | 0.0248 | 0.0242 | **no** |
| 8,192 | 0.0221 | 0.0240 | **no** |
| 8,192 (repeat) | 0.0224 | 0.0255 | **no** |

- The switch is between 2,048 and 2,176 tokens. Max absolute difference at 8,192 is about 0.36.
- It is in eager mode too, so CUDA graphs are not the cause.
- 2,176-3,072 used a run without `--attention-config.indexer_kv_dtype fp8`; 2,048 and 8,192 behave the same with and
  without that flag, so the flag is not the cause.
- It is not the NCCL fallback for large all-reduces: that starts above 32 MB (about 6,250 tokens), far above 2,048.

**Guess, not verified:** the sparse attention (QSA) keeps the top-k tokens, and 2,048 is one of the supported k values
in `qsa_indexer.py` (`block_topk in (512, 1024, 2048)`). Above 2,048 tokens the selection starts dropping tokens, and if
the top-k backend ("auto" prefers FlashInfer `gvr_2`) breaks ties or orders results differently between runs, the
selected set changes. Dummy weights may make ties more common than real weights do.

**Suggested next checks:** rerun 4,096 tokens with `sparse_indexer_topk_backend` set to a non-`gvr_2` backend; rerun with
real weights; compare the selected indices of two runs directly.

## Not done

- TP1 or TP8 profiles; other routing modes; MTP.
- Verifying candidate 1's state dtype and head split, and the caller of candidate 2's gather.
- Root cause of the non-reproducible prefill.
