dp0_pp0_tp0_dcp0_ep0_rank0.1795572372886489756.pt.trace.json.gz: 37696 launches, summed 169905 us, span 168119 us, cuBLAS BF16 25247 us = 14.86% (7072 launches)
| # | kernel | us | % | calls | cuBLAS BF16 |
|---|---|---|---|---|---|
| 1 | `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, 128, 1, 128, cute::tup...` | 45553.3 | 26.8 | 5120 |  |
| 2 | `bmm_E4m3_E4m3E4m3_Fp32_t128x16x128_s6_et64x16_m64x16x32_c1x1x1_rM_BN_transOut_dsFp8_schedS_bN_tma_tmaSf_rgT...` | 14957.2 | 8.8 | 1280 |  |
| 3 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, false, 128>(c10::B...` | 13692.1 | 8.1 | 5120 |  |
| 4 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_rM_BN_transOut_dsFp8_schedS_bN_rgTma_c...` | 12979.5 | 7.6 | 1280 |  |
| 5 | `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(cutlass_80_tensorop_s16816g...` | 7711.7 | 4.5 | 2240 | **yes** |
| 6 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, __nv_bfl...` | 6386.4 | 3.8 | 2240 | **yes** |
| 7 | `void moe::dev::activation::activationDeepSeekKernel<moe::dev::activation::KernelParams<cutlass::float_e4m3_...` | 6383.4 | 3.8 | 1280 |  |
| 8 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16g...` | 6152.5 | 3.6 | 2560 |  |
| 9 | `void moe::dev::finalize::finalizeKernelSmallBatch<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cut...` | 5158.3 | 3.0 | 1280 |  |
| 10 | `void moe::dev::routing::routingCustom::routingIndicesBlockKernel<moe::dev::routing::routingCustom::KernelPa...` | 4915.5 | 2.9 | 1280 |  |
| 11 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` | 4857.2 | 2.9 | 32 | **yes** |
| 12 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, 128>(c10::BF...` | 4564.0 | 2.7 | 1280 |  |
| 13 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 3958.7 | 2.3 | 1280 |  |
| 14 | `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous ...` | 3294.1 | 1.9 | 32 |  |
| 15 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 3186.3 | 1.9 | 960 |  |
| 16 | `void reduce_1Block_kernel<float, 128, 7, cublasGemvTensorStridedBatched<float>, cublasGemvTensorStridedBatc...` | 3147.0 | 1.9 | 1280 | **yes** |
| 17 | `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasG...` | 3144.6 | 1.9 | 1280 | **yes** |
| 18 | `_causal_conv1d_update_kernel` | 2729.0 | 1.6 | 960 |  |
| 19 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat1...` | 2536.6 | 1.5 | 1280 |  |
| 20 | `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP64MultiCtasKvCgaVarSeqQ8Kv256StaticGroupedSwapsAbForGen` | 2535.5 | 1.5 | 320 |  |
| 21 | `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase&):...` | 2057.4 | 1.2 | 1280 |  |
| 22 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<ch...` | 1721.0 | 1.0 | 1280 |  |
| 23 | `_fused_gdn_gated_rms_norm_kernel` | 1528.3 | 0.9 | 960 |  |
| 24 | `void flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA<1024u, 4u, false, float, int>(float*, float*,...` | 1118.9 | 0.7 | 32 |  |
| 25 | `void flashinfer::sampling::TopPSamplingFromProbKernel<1024u, (cub::_V_300600_SM_1030::BlockScanAlgorithm)2,...` | 1063.8 | 0.6 | 32 |  |
