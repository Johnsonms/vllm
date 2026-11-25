dp0_pp0_tp0_dcp0_ep0_rank0.1795586794525135435.pt.trace.json.gz: 50304 launches, summed 175390 us, span 178456 us, cuBLAS BF16 23534 us = 13.42% (7776 launches)
| # | kernel | us | % | calls | cuBLAS BF16 |
|---|---|---|---|---|---|
| 1 | `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schedS_bN_tma_tmaSf_rgTm...` | 15729.2 | 9.0 | 1536 |  |
| 2 | `void flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel<(unsigned char)4, __nv_bfloat16, fals...` | 14722.5 | 8.4 | 3104 |  |
| 3 | `nvjet_sm100_tst_128x8_64x12_4x1_v_bz_TNT` | 11450.1 | 6.5 | 3104 | **yes** |
| 4 | `kernel_cutlass_kernel_vllmmodelsqwen4_expnvidiaopscute_dsl_hc_down_silu_fmaHcDownSiluFma_object_at__tensorp...` | 9823.0 | 5.6 | 3072 |  |
| 5 | `_hc_combine_norm_kernel` | 8664.9 | 4.9 | 3040 |  |
| 6 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schedS_bN_rgTma_clmp_d...` | 8340.0 | 4.8 | 1536 |  |
| 7 | `_hc_gate_mix_kernel` | 7643.9 | 4.4 | 3104 |  |
| 8 | `void moe::dev::routing::routingCustom::routingIndicesBlockKernel<moe::dev::routing::routingCustom::KernelPa...` | 7390.5 | 4.2 | 1536 |  |
| 9 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 6567.0 | 3.7 | 2688 |  |
| 10 | `void moe::dev::activation::activationDeepSeekKernel<moe::dev::activation::KernelParams<cutlass::float_e4m3_...` | 6135.8 | 3.5 | 1536 |  |
| 11 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 6124.2 | 3.5 | 1152 |  |
| 12 | `nvjet_sm100_tst_24x64_64x16_4x1_v_bz_TNN` | 5333.8 | 3.0 | 1536 | **yes** |
| 13 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 5210.9 | 3.0 | 1536 |  |
| 14 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 4702.7 | 2.7 | 1568 |  |
| 15 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 4436.5 | 2.5 | 1536 |  |
| 16 | `void moe::dev::finalize::finalizeKernelSmallBatch<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cut...` | 4351.1 | 2.5 | 1536 |  |
| 17 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, false, true, 128>(c10::B...` | 4034.8 | 2.3 | 1536 |  |
| 18 | `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous ...` | 3474.3 | 2.0 | 32 |  |
| 19 | `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasG...` | 3401.6 | 1.9 | 1536 | **yes** |
| 20 | `_causal_conv1d_update_kernel` | 3364.0 | 1.9 | 1152 |  |
| 21 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 2936.7 | 1.7 | 1152 |  |
| 22 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat1...` | 2883.7 | 1.6 | 1536 |  |
| 23 | `void reduce_1Block_kernel<float, 128, 7, cublasGemvTensorStridedBatched<float>, cublasGemvTensorStridedBatc...` | 2802.2 | 1.6 | 1536 | **yes** |
| 24 | `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase&):...` | 2765.2 | 1.6 | 1536 |  |
| 25 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<ch...` | 2341.4 | 1.3 | 1536 |  |
| 26 | `layer_norm_fwd_kernel` | 2182.5 | 1.2 | 1152 |  |
| 27 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 2091.1 | 1.2 | 384 |  |
| 28 | `_qsa_sparse_paged_gqa_splitk_kernel` | 1725.7 | 1.0 | 384 |  |
| 29 | `_qsa_prepare_kernel` | 1527.8 | 0.9 | 384 |  |
| 30 | `_qsa_merge_splitk_kernel` | 1450.4 | 0.8 | 384 |  |
