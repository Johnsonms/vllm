dp0_pp0_tp0_dcp0_ep0_rank0.1795586889300758898.pt.trace.json.gz: 53024 launches, summed 215563 us, span 186839 us, cuBLAS BF16 37546 us = 17.42% (10880 launches)
| # | kernel | us | % | calls | cuBLAS BF16 |
|---|---|---|---|---|---|
| 1 | `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schedS_bN_tma_tmaSf_rgTm...` | 18828.1 | 8.7 | 1536 |  |
| 2 | `void flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel<(unsigned char)4, __nv_bfloat16, fals...` | 18262.0 | 8.5 | 3104 |  |
| 3 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schedS_bN_rgTma_clmp_d...` | 16222.2 | 7.5 | 1536 |  |
| 4 | `kernel_cutlass_kernel_vllmmodelsqwen4_expnvidiaopscute_dsl_hc_down_silu_fmaHcDownSiluFma_object_at__tensorp...` | 15823.6 | 7.3 | 3072 |  |
| 5 | `nvjet_sm100_tst_128x8_64x12_4x1_v_bz_TNT` | 11846.3 | 5.5 | 3104 | **yes** |
| 6 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 8520.1 | 4.0 | 1152 |  |
| 7 | `_hc_combine_norm_kernel` | 8050.3 | 3.7 | 3040 |  |
| 8 | `void moe::dev::routing::routingCustom::routingIndicesBlockKernel<moe::dev::routing::routingCustom::KernelPa...` | 7957.4 | 3.7 | 1536 |  |
| 9 | `_hc_gate_mix_kernel` | 7789.1 | 3.6 | 3104 |  |
| 10 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 6945.2 | 3.2 | 1536 | **yes** |
| 11 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 6507.2 | 3.0 | 1536 |  |
| 12 | `void moe::dev::activation::activationDeepSeekKernel<moe::dev::activation::KernelParams<cutlass::float_e4m3_...` | 5973.9 | 2.8 | 1536 |  |
| 13 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 5425.4 | 2.5 | 1568 |  |
| 14 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x6_tn_align8>(cutlass_80_tensor...` | 5251.9 | 2.4 | 1536 | **yes** |
| 15 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, 128>(c10::BF...` | 4779.6 | 2.2 | 1536 |  |
| 16 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, _...` | 4398.7 | 2.0 | 1568 | **yes** |
| 17 | `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasG...` | 4200.5 | 1.9 | 1536 | **yes** |
| 18 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 4038.0 | 1.9 | 1536 |  |
| 19 | `void moe::dev::finalize::finalizeKernelSmallBatch<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cut...` | 3780.0 | 1.8 | 1536 |  |
| 20 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | 3591.9 | 1.7 | 1184 |  |
| 21 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 3538.7 | 1.6 | 1152 |  |
| 22 | `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous ...` | 3489.0 | 1.6 | 32 |  |
| 23 | `_causal_conv1d_update_kernel` | 3320.7 | 1.5 | 1152 |  |
| 24 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat1...` | 3165.5 | 1.5 | 1536 |  |
| 25 | `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase&):...` | 3037.4 | 1.4 | 1536 |  |
| 26 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 2854.7 | 1.3 | 1152 |  |
| 27 | `void reduce_1Block_kernel<float, 128, 7, cublasGemvTensorStridedBatched<float>, cublasGemvTensorStridedBatc...` | 2832.9 | 1.3 | 1536 | **yes** |
| 28 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 2708.7 | 1.3 | 384 |  |
| 29 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<ch...` | 2365.9 | 1.1 | 1536 |  |
| 30 | `layer_norm_fwd_kernel` | 2300.3 | 1.1 | 1152 |  |
