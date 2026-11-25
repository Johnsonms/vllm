dp0_pp0_tp0_dcp0_ep0_rank0.1795586286401897344.pt.trace.json.gz: 50912 launches, summed 357580 us, span 256122 us, cuBLAS BF16 2429 us = 0.68% (32 launches)
| # | kernel | us | % | calls | cuBLAS BF16 |
|---|---|---|---|---|---|
| 1 | `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, 128, 1, 128, cute::tup...` | 77543.2 | 21.7 | 7680 |  |
| 2 | `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x2x2x3_bN_tma_tmaS...` | 71056.4 | 19.9 | 1920 |  |
| 3 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x2x2x3_bN_rgTma_...` | 43029.6 | 12.0 | 1920 |  |
| 4 | `void flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel<(unsigned char)4, __nv_bfloat16, true...` | 24712.6 | 6.9 | 3840 |  |
| 5 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, false, 128>(c10::B...` | 23818.6 | 6.7 | 7680 |  |
| 6 | `void moe::dev::activation::activationDeepSeekKernel<moe::dev::activation::KernelParams<cutlass::float_e4m3_...` | 13861.0 | 3.9 | 1920 |  |
| 7 | `void moe::dev::finalize::finalizeKernelSmallBatch<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cut...` | 11371.8 | 3.2 | 1920 |  |
| 8 | `void moe::dev::routing::routingCustom::routingIndicesDynBlockKernel<moe::dev::routing::routingCustom::Kerne...` | 10943.5 | 3.1 | 1920 |  |
| 9 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 10557.2 | 3.0 | 3360 |  |
| 10 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 8507.7 | 2.4 | 1920 |  |
| 11 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 7891.9 | 2.2 | 1440 |  |
| 12 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 7087.5 | 2.0 | 1920 |  |
| 13 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat1...` | 6152.6 | 1.7 | 1920 |  |
| 14 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, 128>(c10::BF...` | 5355.2 | 1.5 | 1920 |  |
| 15 | `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase&):...` | 4896.9 | 1.4 | 1920 |  |
| 16 | `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP64MultiCtasKvVarSeqQ8Kv256StaticGroupedSwapsAbForGen` | 4669.3 | 1.3 | 480 |  |
| 17 | `_causal_conv1d_update_kernel` | 4091.4 | 1.1 | 1440 |  |
| 18 | `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous ...` | 3470.8 | 1.0 | 32 |  |
| 19 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<ch...` | 3089.7 | 0.9 | 1920 |  |
| 20 | `_fused_gdn_gated_rms_norm_kernel` | 2465.9 | 0.7 | 1440 |  |
| 21 | `nvjet_sm100_tst_64x8_64x16_2x1_v_bz_TNT` | 2428.6 | 0.7 | 32 | **yes** |
| 22 | `void flashinfer::sampling::TopPSamplingFromProbKernel<1024u, (cub::_V_300600_SM_1000::BlockScanAlgorithm)2,...` | 1735.3 | 0.5 | 32 |  |
| 23 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1464.3 | 0.4 | 480 |  |
| 24 | `void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0>(__nv_b...` | 1337.2 | 0.4 | 480 |  |
| 25 | `void flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA<1024u, 4u, false, float, int>(float*, float*,...` | 1199.6 | 0.3 | 32 |  |
| 26 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 796.6 | 0.2 | 32 |  |
| 27 | `_fused_sigmoid_gate_mul_kernel` | 793.5 | 0.2 | 480 |  |
| 28 | `_temperature_kernel` | 520.6 | 0.1 | 32 |  |
