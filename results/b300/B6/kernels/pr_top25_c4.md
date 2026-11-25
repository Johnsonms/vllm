dp0_pp0_tp0_dcp0_ep0_rank0.1795572770885117865.pt.trace.json.gz: 34176 launches, summed 183887 us, span 163394 us, cuBLAS BF16 4898 us = 2.66% (32 launches)
| # | kernel | us | % | calls | cuBLAS BF16 |
|---|---|---|---|---|---|
| 1 | `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, 128, 1, 128, cute::tup...` | 45498.3 | 24.7 | 5120 |  |
| 2 | `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x2x2x3_bN_tma_tmaS...` | 25192.7 | 13.7 | 1280 |  |
| 3 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x2x2x3_bN_rgTm...` | 20648.7 | 11.2 | 1280 |  |
| 4 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, false, 128>(c10::B...` | 13515.8 | 7.3 | 5120 |  |
| 5 | `void moe::dev::activation::activationDeepSeekKernel<moe::dev::activation::KernelParams<cutlass::float_e4m3_...` | 9621.2 | 5.2 | 1280 |  |
| 6 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16g...` | 7425.9 | 4.0 | 2560 |  |
| 7 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 5221.6 | 2.8 | 960 |  |
| 8 | `void moe::dev::routing::routingCustom::routingIndicesBlockKernel<moe::dev::routing::routingCustom::KernelPa...` | 5108.2 | 2.8 | 1280 |  |
| 9 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 4920.3 | 2.7 | 2240 |  |
| 10 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` | 4897.9 | 2.7 | 32 | **yes** |
| 11 | `void moe::dev::finalize::finalizeKernelSmallBatch<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cut...` | 4897.7 | 2.7 | 1280 |  |
| 12 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat1...` | 4685.2 | 2.5 | 1280 |  |
| 13 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 3869.1 | 2.1 | 1280 |  |
| 14 | `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous ...` | 3434.8 | 1.9 | 32 |  |
| 15 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, 128>(c10::BF...` | 3164.0 | 1.7 | 1280 |  |
| 16 | `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP64MultiCtasKvVarSeqQ8Kv256StaticGroupedSwapsAbForGen` | 2987.2 | 1.6 | 320 |  |
| 17 | `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase&):...` | 2948.7 | 1.6 | 1280 |  |
| 18 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 2831.0 | 1.5 | 1280 |  |
| 19 | `_causal_conv1d_update_kernel` | 2724.5 | 1.5 | 960 |  |
| 20 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<ch...` | 1729.6 | 0.9 | 1280 |  |
| 21 | `_fused_gdn_gated_rms_norm_kernel` | 1646.9 | 0.9 | 960 |  |
| 22 | `void flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA<1024u, 4u, false, float, int>(float*, float*,...` | 1124.8 | 0.6 | 32 |  |
| 23 | `void flashinfer::sampling::TopPSamplingFromProbKernel<1024u, (cub::_V_300600_SM_1030::BlockScanAlgorithm)2,...` | 1056.8 | 0.6 | 32 |  |
| 24 | `_fused_qk_rmsnorm_rope_gate_kernel` | 948.4 | 0.5 | 320 |  |
| 25 | `void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0>(__nv_b...` | 945.2 | 0.5 | 320 |  |
