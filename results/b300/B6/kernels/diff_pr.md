| kernel | µs/step c2 | c4 | c8 | Δ c2→c4 | Δ c4→c8 | calls/step c2/c4/c8 |
|---|---|---|---|---|---|---|
| `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x2x2x3` | 0.0 | 787.3 | 1086.3 | +787.3 | +299.0 | 0/40/40 |
| `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x2` | 0.0 | 645.3 | 0.0 | +645.3 | -645.3 | 0/40/0 |
| `bmm_E4m3_E4m3E4m3_Fp32_t128x16x128_s6_et64x16_m64x16x32_c1x1x1_rM_BN_transOut_dsFp8_schedS_bN_t` | 473.0 | 0.0 | 0.0 | -473.0 | +0.0 | 40/0/0 |
| `bmm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_rM_BN_transOut_dsFp8_sched` | 399.5 | 0.0 | 0.0 | -399.5 | +0.0 | 40/0/0 |
| `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_` | 166.0 | 0.0 | 0.0 | -166.0 | +0.0 | 80/0/0 |
| `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_` | 0.0 | 153.8 | 0.0 | +153.8 | -153.8 | 0/70/0 |
| `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` | 0.0 | 153.1 | 155.1 | +153.1 | +2.0 | 0/1/1 |
| `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_` | 140.9 | 0.0 | 0.0 | -140.9 | +0.0 | 1/0/0 |
| `void moe::dev::activation::activationDeepSeekKernel<moe::dev::activation::KernelParams<cutlass:` | 186.7 | 300.7 | 304.9 | +113.9 | +4.2 | 40/40/40 |
| `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP64MultiCtasKvVarSeqQ8Kv256StaticGroupedS` | 0.0 | 93.3 | 95.7 | +93.3 | +2.3 | 0/10/10 |
| `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_` | 0.0 | 88.5 | 0.0 | +88.5 | -88.5 | 0/40/0 |
| `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP64MultiCtasKvCgaVarSeqQ8Kv256StaticGroup` | 78.5 | 0.0 | 0.0 | -78.5 | +0.0 | 10/0/0 |
| `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_` | 77.6 | 0.0 | 0.0 | -77.6 | +0.0 | 30/0/0 |
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 99.4 | 163.2 | 217.5 | +63.8 | +54.4 | 30/30/30 |
| `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<` | 88.4 | 146.4 | 108.9 | +58.0 | -37.5 | 40/40/40 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__te` | 189.5 | 232.1 | 230.6 | +42.6 | -1.4 | 80/80/80 |
| `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, 128, 1, 12` | 1394.2 | 1421.8 | 1417.1 | +27.6 | -4.8 | 160/160/160 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::TensorIte` | 65.5 | 92.1 | 66.8 | +26.7 | -25.3 | 40/40/40 |
| `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, false,` | 409.2 | 422.4 | 436.5 | +13.2 | +14.1 | 160/160/160 |
| `void moe::dev::finalize::finalizeKernelSmallBatch<moe::dev::finalize::KernelParams<cutlass::bfl` | 159.2 | 153.1 | 136.9 | -6.2 | -16.2 | 40/40/40 |

summed kernel µs/step: c2 4807, c4 5746, c8 6240
