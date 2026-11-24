dp0_pp0_tp0_dcp0_ep0_rank0.1795552921994767174.pt.trace.json.gz: 37696 launches, summed 159100 us, span 116710 us, cuBLAS BF16 28683 us = 18.03% (7072 launches)
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
