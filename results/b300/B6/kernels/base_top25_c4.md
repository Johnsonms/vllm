dp0_pp0_tp0_dcp0_ep0_rank0.1795572443953611882.pt.trace.json.gz: 37696 launches, summed 191893 us, span 168113 us, cuBLAS BF16 24968 us = 13.01% (7072 launches)
| # | kernel | us | % | calls | cuBLAS BF16 |
|---|---|---|---|---|---|
| 1 | `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, 128, 1, 128, cute::tup...` | 45027.0 | 23.5 | 5120 |  |
| 2 | `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x2x2x3_bN_tma_tmaS...` | 24650.0 | 12.8 | 1280 |  |
| 3 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x2x2x3_bN_rgTm...` | 20266.2 | 10.6 | 1280 |  |
| 4 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, false, 128>(c10::B...` | 13505.7 | 7.0 | 5120 |  |
| 5 | `void moe::dev::activation::activationDeepSeekKernel<moe::dev::activation::KernelParams<cutlass::float_e4m3_...` | 9327.6 | 4.9 | 1280 |  |
| 6 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16g...` | 7367.3 | 3.8 | 2560 |  |
| 7 | `void moe::dev::routing::routingCustom::routingIndicesBlockKernel<moe::dev::routing::routingCustom::KernelPa...` | 5337.2 | 2.8 | 1280 |  |
| 8 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 5160.7 | 2.7 | 960 |  |
| 9 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` | 4885.4 | 2.5 | 32 | **yes** |
| 10 | `nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 4835.8 | 2.5 | 1280 | **yes** |
| 11 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, 128>(c10::BF...` | 4066.8 | 2.1 | 1280 |  |
| 12 | `void moe::dev::finalize::finalizeKernelSmallBatch<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cut...` | 4011.3 | 2.1 | 1280 |  |
| 13 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 3875.2 | 2.0 | 1280 |  |
| 14 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, _...` | 3460.0 | 1.8 | 1280 | **yes** |
| 15 | `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasG...` | 3432.0 | 1.8 | 1280 | **yes** |
| 16 | `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous ...` | 3432.0 | 1.8 | 32 |  |
| 17 | `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP64MultiCtasKvVarSeqQ8Kv256StaticGroupedSwapsAbForGen` | 2981.6 | 1.6 | 320 |  |
| 18 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, __nv_bfl...` | 2936.0 | 1.5 | 960 | **yes** |
| 19 | `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(cutlass_80_tensorop_s16816g...` | 2839.0 | 1.5 | 960 | **yes** |
| 20 | `_causal_conv1d_update_kernel` | 2755.3 | 1.4 | 960 |  |
| 21 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat1...` | 2728.5 | 1.4 | 1280 |  |
| 22 | `void reduce_1Block_kernel<float, 128, 7, cublasGemvTensorStridedBatched<float>, cublasGemvTensorStridedBatc...` | 2579.8 | 1.3 | 1280 | **yes** |
| 23 | `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase&):...` | 2094.5 | 1.1 | 1280 |  |
| 24 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<ch...` | 1747.9 | 0.9 | 1280 |  |
| 25 | `_fused_gdn_gated_rms_norm_kernel` | 1619.4 | 0.8 | 960 |  |
