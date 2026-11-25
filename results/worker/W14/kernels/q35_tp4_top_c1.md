dp0_pp0_tp2_dcp0_ep2_rank2.1795586182220842095.pt.trace.json.gz: 50880 launches, summed 248298 us, span 189904 us, cuBLAS BF16 0 us = 0.0% (0 launches)
| # | kernel | us | % | calls | cuBLAS BF16 |
|---|---|---|---|---|---|
| 1 | `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, 128, 1, 128, cute::tup...` | 68761.1 | 27.7 | 7680 |  |
| 2 | `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schedS_bN_tma_tmaSf_rgTm...` | 26690.4 | 10.8 | 1920 |  |
| 3 | `void flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel<(unsigned char)4, __nv_bfloat16, true...` | 23674.3 | 9.5 | 3840 |  |
| 4 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, false, false, 128>(c10::...` | 19936.1 | 8.0 | 7680 |  |
| 5 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schedS_bN_rgTma_clmp_d...` | 19884.6 | 8.0 | 1920 |  |
| 6 | `void moe::dev::routing::routingCustom::routingIndicesBlockKernel<moe::dev::routing::routingCustom::KernelPa...` | 9375.9 | 3.8 | 1920 |  |
| 7 | `void moe::dev::activation::activationDeepSeekKernel<moe::dev::activation::KernelParams<cutlass::float_e4m3_...` | 8223.9 | 3.3 | 1920 |  |
| 8 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 7798.2 | 3.1 | 3360 |  |
| 9 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 6855.5 | 2.8 | 1920 |  |
| 10 | `void moe::dev::finalize::finalizeKernelSmallBatch<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cut...` | 5533.4 | 2.2 | 1920 |  |
| 11 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 5368.3 | 2.2 | 1920 |  |
| 12 | `void flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel<(unsigned char)4, __nv_bfloat16, fals...` | 4789.0 | 1.9 | 32 |  |
| 13 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, false, true, 128>(c10::B...` | 4589.1 | 1.9 | 1920 |  |
| 14 | `_causal_conv1d_update_kernel` | 4091.5 | 1.6 | 1440 |  |
| 15 | `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP64MultiCtasKvCgaVarSeqQ8Kv256StaticGroupedSwapsAbForGen` | 3866.9 | 1.6 | 480 |  |
| 16 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat1...` | 3653.9 | 1.5 | 1920 |  |
| 17 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 3653.2 | 1.5 | 1440 |  |
| 18 | `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous ...` | 3313.7 | 1.3 | 32 |  |
| 19 | `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase&):...` | 3018.7 | 1.2 | 1920 |  |
| 20 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<ch...` | 2370.9 | 0.9 | 1920 |  |
| 21 | `_fused_gdn_gated_rms_norm_kernel` | 2262.6 | 0.9 | 1440 |  |
| 22 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 2238.3 | 0.9 | 32 |  |
| 23 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1484.9 | 0.6 | 480 |  |
| 24 | `void flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA<1024u, 4u, false, float, int>(float*, float*,...` | 1182.2 | 0.5 | 32 |  |
| 25 | `void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0>(__nv_b...` | 1140.8 | 0.5 | 480 |  |
| 26 | `void flashinfer::sampling::TopPSamplingFromProbKernel<1024u, (cub::_V_300600_SM_1000::BlockScanAlgorithm)2,...` | 1091.1 | 0.4 | 32 |  |
| 27 | `_fused_sigmoid_gate_mul_kernel` | 777.6 | 0.3 | 480 |  |
| 28 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 408.8 | 0.2 | 32 |  |
