dp0_pp0_tp0_dcp0_ep0_rank0.1795586989572526853.pt.trace.json.gz: 53024 launches, summed 285898 us, span 218520 us, cuBLAS BF16 61006 us = 21.34% (13984 launches)
| # | kernel | us | % | calls | cuBLAS BF16 |
|---|---|---|---|---|---|
| 1 | `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x2x2x3_bN_tma_tmaS...` | 42158.5 | 14.8 | 1536 |  |
| 2 | `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x2x2x3_bN_rgTma_...` | 26981.7 | 9.4 | 1536 |  |
| 3 | `void flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel<(unsigned char)4, __nv_bfloat16, fals...` | 20206.6 | 7.1 | 3104 |  |
| 4 | `kernel_cutlass_kernel_vllmmodelsqwen4_expnvidiaopscute_dsl_hc_down_silu_mmaHcDownSiluMma_object_at__tensorp...` | 15308.3 | 5.3 | 3072 |  |
| 5 | `nvjet_sm100_tst_192x8_64x8_4x2_h_bz_TNT` | 13595.8 | 4.8 | 3104 | **yes** |
| 6 | `nvjet_sm100_tst_64x8_64x16_4x2_h_bz_TNT` | 12334.1 | 4.3 | 3072 | **yes** |
| 7 | `void moe::dev::activation::activationDeepSeekKernel<moe::dev::activation::KernelParams<cutlass::float_e4m3_...` | 12231.4 | 4.3 | 1536 |  |
| 8 | `void moe::dev::routing::routingCustom::routingIndicesDynBlockKernel<moe::dev::routing::routingCustom::Kerne...` | 11960.4 | 4.2 | 1536 |  |
| 9 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 11071.2 | 3.9 | 1536 | **yes** |
| 10 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_object_at__tensorpt...` | 10905.6 | 3.8 | 2688 |  |
| 11 | `_hc_gate_mix_kernel` | 9880.9 | 3.5 | 3104 |  |
| 12 | `_hc_combine_norm_kernel` | 8558.5 | 3.0 | 3040 |  |
| 13 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, _...` | 8391.2 | 2.9 | 3104 | **yes** |
| 14 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c1...` | 7563.5 | 2.6 | 1568 |  |
| 15 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 7421.3 | 2.6 | 1568 | **yes** |
| 16 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 7412.9 | 2.6 | 1152 |  |
| 17 | `void moe::dev::finalize::finalizeKernelSmallBatch<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, cut...` | 6623.3 | 2.3 | 1536 |  |
| 18 | `nvjet_sm100_tst_64x8_64x16_1x1_h_bz_splitK_TNT` | 6199.6 | 2.2 | 1536 | **yes** |
| 19 | `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, 128>(c10::BF...` | 4829.6 | 1.7 | 1536 |  |
| 20 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kern...` | 3636.6 | 1.3 | 1184 |  |
| 21 | `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous ...` | 3545.1 | 1.2 | 32 |  |
| 22 | `_causal_conv1d_update_kernel` | 3442.4 | 1.2 | 1152 |  |
| 23 | `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase&):...` | 3415.7 | 1.2 | 1536 |  |
| 24 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat1...` | 3131.4 | 1.1 | 1536 |  |
| 25 | `_qsa_sparse_paged_gqa_splitk_kernel` | 2737.4 | 1.0 | 384 |  |
| 26 | `layer_norm_fwd_kernel` | 2510.1 | 0.9 | 1152 |  |
| 27 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<ch...` | 2481.8 | 0.9 | 1536 |  |
| 28 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1641.3 | 0.6 | 32 |  |
| 29 | `nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_TNT` | 1543.0 | 0.5 | 32 | **yes** |
| 30 | `_qsa_prepare_kernel` | 1539.4 | 0.5 | 384 |  |
