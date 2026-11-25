| kernel | µs/step c2 | c4 | c8 | Δ c2→c4 | Δ c4→c8 | calls/step c2/c4/c8 |
|---|---|---|---|---|---|---|
| `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x2x2x3` | 0.0 | 770.3 | 1117.7 | +770.3 | +347.4 | 0/40/40 |
| `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x2` | 0.0 | 633.3 | 0.0 | +633.3 | -633.3 | 0/40/0 |
| `bmm_E4m3_E4m3E4m3_Fp32_t128x16x128_s6_et64x16_m64x16x32_c1x1x1_rM_BN_transOut_dsFp8_schedS_bN_t` | 467.4 | 0.0 | 0.0 | -467.4 | +0.0 | 40/0/0 |
| `bmm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_rM_BN_transOut_dsFp8_sched` | 405.6 | 0.0 | 0.0 | -405.6 | +0.0 | 40/0/0 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(cutlass_80_tens` | 241.0 | 88.7 | 88.3 | -152.3 | -0.4 | 70/30/30 |
| `nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 0.0 | 151.1 | 155.0 | +151.1 | +3.9 | 0/40/40 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, fal` | 0.0 | 108.1 | 213.1 | +108.1 | +104.9 | 0/40/80 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, fal` | 199.6 | 91.8 | 88.7 | -107.8 | -3.1 | 70/30/30 |
| `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP64MultiCtasKvVarSeqQ8Kv256StaticGroupedS` | 0.0 | 93.2 | 95.2 | +93.2 | +2.0 | 0/10/10 |
| `void moe::dev::activation::activationDeepSeekKernel<moe::dev::activation::KernelParams<cutlass:` | 199.5 | 291.5 | 308.0 | +92.0 | +16.5 | 40/40/40 |
| `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP64MultiCtasKvCgaVarSeqQ8Kv256StaticGroup` | 79.2 | 0.0 | 0.0 | -79.2 | +0.0 | 10/0/0 |
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 99.6 | 161.3 | 217.4 | +61.7 | +56.1 | 30/30/30 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__te` | 192.3 | 230.2 | 229.7 | +38.0 | -0.5 | 80/80/80 |
| `void moe::dev::finalize::finalizeKernelSmallBatch<moe::dev::finalize::KernelParams<cutlass::bfl` | 161.2 | 125.4 | 135.1 | -35.8 | +9.8 | 40/40/40 |
| `void reduce_1Block_kernel<float, 128, 7, cublasGemvTensorStridedBatched<float>, cublasGemvTenso` | 98.3 | 80.6 | 0.0 | -17.7 | -80.6 | 40/40/0 |
| `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, 128, 1, 12` | 1423.5 | 1407.1 | 1397.1 | -16.4 | -10.0 | 160/160/160 |
| `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, ` | 142.6 | 127.1 | 137.6 | -15.5 | +10.5 | 40/40/40 |
| `void moe::dev::routing::routingCustom::routingIndicesBlockKernel<moe::dev::routing::routingCust` | 153.6 | 166.8 | 0.0 | +13.2 | -166.8 | 40/40/0 |
| `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<__nv_bfloat16 con` | 98.3 | 107.2 | 0.0 | +9.0 | -107.2 | 40/40/0 |
| `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<` | 79.3 | 85.3 | 110.8 | +6.0 | +25.5 | 40/40/40 |

summed kernel µs/step: c2 5310, c4 5997, c8 6704
