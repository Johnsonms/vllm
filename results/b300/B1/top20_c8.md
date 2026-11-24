dp0_pp0_tp0_dcp0_ep0_rank0.1795553006053002224.pt.trace.json.gz: 37696 launches, summed 219848 us, span 160187 us, cuBLAS BF16 27320 us = 12.43% (7072 launches)
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
