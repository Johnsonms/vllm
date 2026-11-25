dp0_pp0_tp0_dcp0_ep0_rank0.1795572842537377774.pt.trace.json.gz: 34176 launches, summed 199695 us, span 146711 us, cuBLAS BF16 4962 us = 2.48% (32 launches)
| # | kernel | us | % | calls | cuBLAS BF16 |
|---|---|---|---|---|---|
| 1 | `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, 128, 1, 128, cute::tup...` | 45345.9 | 22.7 | 5120 |  |
| 2 | `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x2x2x3_bN_tma_tmaS...` | 34761.0 | 17.4 | 1280 |  |
| 3 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x2x2x3_bN_rgTma_...` | 23685.2 | 11.9 | 1280 |  |
| 4 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, false, 128>(c10::B...` | 13968.1 | 7.0 | 5120 |  |
| 5 | `void moe::dev::activation::activationDeepSeekKernel<moe::dev::activation::KernelParams<cutlass::float_e4m3_...` | 9756.9 | 4.9 | 1280 |  |
| 6 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16g...` | 7380.5 | 3.7 | 2560 |  |
| 7 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 6960.9 | 3.5 | 960 |  |
| 8 | `void moe::dev::routing::routingCustom::routingIndicesDynBlockKernel<moe::dev::routing::routingCustom::Kerne...` | 6413.8 | 3.2 | 1280 |  |
| 9 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 6057.5 | 3.0 | 2240 |  |
| 10 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` | 4962.0 | 2.5 | 32 | **yes** |
| 11 | `void moe::dev::finalize::finalizeKernelSmallBatch<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cut...` | 4380.0 | 2.2 | 1280 |  |
| 12 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 4052.7 | 2.0 | 1280 |  |
| 13 | `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous ...` | 3487.0 | 1.8 | 32 |  |
| 14 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat1...` | 3483.9 | 1.7 | 1280 |  |
| 15 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, 128>(c10::BF...` | 3208.4 | 1.6 | 1280 |  |
| 16 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 3062.1 | 1.5 | 1280 |  |
| 17 | `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP64MultiCtasKvVarSeqQ8Kv256StaticGroupedSwapsAbForGen` | 3061.3 | 1.5 | 320 |  |
| 18 | `_causal_conv1d_update_kernel` | 2736.1 | 1.4 | 960 |  |
| 19 | `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase&):...` | 2137.8 | 1.1 | 1280 |  |
| 20 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<ch...` | 1950.0 | 1.0 | 1280 |  |
| 21 | `_fused_gdn_gated_rms_norm_kernel` | 1635.9 | 0.8 | 960 |  |
| 22 | `void flashinfer::sampling::TopPSamplingFromProbKernel<1024u, (cub::_V_300600_SM_1030::BlockScanAlgorithm)2,...` | 1177.5 | 0.6 | 32 |  |
| 23 | `void flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA<1024u, 4u, false, float, int>(float*, float*,...` | 1125.5 | 0.6 | 32 |  |
| 24 | `_fused_qk_rmsnorm_rope_gate_kernel` | 952.9 | 0.5 | 320 |  |
| 25 | `void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0>(__nv_b...` | 943.2 | 0.5 | 320 |  |
