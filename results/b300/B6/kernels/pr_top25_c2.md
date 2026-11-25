dp0_pp0_tp0_dcp0_ep0_rank0.1795572701380995182.pt.trace.json.gz: 34176 launches, summed 153839 us, span 161934 us, cuBLAS BF16 0 us = 0.0% (0 launches)
| # | kernel | us | % | calls | cuBLAS BF16 |
|---|---|---|---|---|---|
| 1 | `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, 128, 1, 128, cute::tup...` | 44615.7 | 29.0 | 5120 |  |
| 2 | `bmm_E4m3_E4m3E4m3_Fp32_t128x16x128_s6_et64x16_m64x16x32_c1x1x1_rM_BN_transOut_dsFp8_schedS_bN_tma_tmaSf_rgT...` | 15136.5 | 9.8 | 1280 |  |
| 3 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, false, 128>(c10::B...` | 13094.9 | 8.5 | 5120 |  |
| 4 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_rM_BN_transOut_dsFp8_schedS_bN_rgTma_c...` | 12783.8 | 8.3 | 1280 |  |
| 5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16g...` | 6063.3 | 3.9 | 2560 |  |
| 6 | `void moe::dev::activation::activationDeepSeekKernel<moe::dev::activation::KernelParams<cutlass::float_e4m3_...` | 5975.0 | 3.9 | 1280 |  |
| 7 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 5312.7 | 3.5 | 2560 |  |
| 8 | `void moe::dev::routing::routingCustom::routingIndicesBlockKernel<moe::dev::routing::routingCustom::KernelPa...` | 5201.6 | 3.4 | 1280 |  |
| 9 | `void moe::dev::finalize::finalizeKernelSmallBatch<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cut...` | 5095.2 | 3.3 | 1280 |  |
| 10 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 4509.4 | 2.9 | 32 |  |
| 11 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 3890.4 | 2.5 | 1280 |  |
| 12 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, 128>(c10::BF...` | 3297.4 | 2.1 | 1280 |  |
| 13 | `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous ...` | 3292.9 | 2.1 | 32 |  |
| 14 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 3179.5 | 2.1 | 960 |  |
| 15 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat1...` | 2828.4 | 1.8 | 1280 |  |
| 16 | `_causal_conv1d_update_kernel` | 2720.9 | 1.8 | 960 |  |
| 17 | `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP64MultiCtasKvCgaVarSeqQ8Kv256StaticGroupedSwapsAbForGen` | 2512.3 | 1.6 | 320 |  |
| 18 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 2482.4 | 1.6 | 960 |  |
| 19 | `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase&):...` | 2095.1 | 1.4 | 1280 |  |
| 20 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<ch...` | 1547.9 | 1.0 | 1280 |  |
| 21 | `_fused_gdn_gated_rms_norm_kernel` | 1534.8 | 1.0 | 960 |  |
| 22 | `void flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA<1024u, 4u, false, float, int>(float*, float*,...` | 1114.4 | 0.7 | 32 |  |
| 23 | `void flashinfer::sampling::TopPSamplingFromProbKernel<1024u, (cub::_V_300600_SM_1030::BlockScanAlgorithm)2,...` | 1061.9 | 0.7 | 32 |  |
| 24 | `void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0>(__nv_b...` | 935.9 | 0.6 | 320 |  |
| 25 | `_fused_qk_rmsnorm_rope_gate_kernel` | 932.8 | 0.6 | 320 |  |
