T/traces_c4: busy-union/step median 3739 us, summed 5731 us, overlap 1991 us
P/traces_c4: busy-union/step median 3818 us, summed 6027 us, overlap 2209 us
c4: exposed total T 2365 P 2269 us/step; overlapped (>=2 running) T 1375 pr 1552
| kernel | exposed T µs/step | exposed P | Δ |
|---|---|---|---|
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:` | 112 | 2 | -110 |
| `void moe::dev::finalize::finalizeKernelSmallBatch<moe::dev::finalize::KernelParams<cutlass` | 2 | 54 | +52 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens` | 16 | 0 | -16 |
| `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_sch` | 282 | 268 | -14 |
| `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, 128, ` | 772 | 765 | -6 |
| `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_ob` | 9 | 7 | -2 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` | 161 | 162 | +1 |
| `_prepare_rope_positions_kernel` | 4 | 3 | -1 |
| `_get_num_sampled_and_rejected_kernel` | 4 | 3 | -1 |
| `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` | 153 | 153 | +1 |
T/traces_c8: busy-union/step median 4542 us, summed 6268 us, overlap 1726 us
P/traces_c8: busy-union/step median 4455 us, summed 6806 us, overlap 2351 us
c8: exposed total T 3332 P 2771 us/step; overlapped (>=2 running) T 1214 pr 1688
| kernel | exposed T µs/step | exposed P | Δ |
|---|---|---|---|
| `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x` | 531 | 360 | -171 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:` | 123 | 0 | -123 |
| `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd` | 427 | 326 | -101 |
| `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, 128, ` | 857 | 760 | -97 |
| `void moe::dev::finalize::finalizeKernelSmallBatch<moe::dev::finalize::KernelParams<cutlass` | 0 | 78 | +78 |
| `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_ob` | 86 | 9 | -77 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens` | 62 | 0 | -62 |
| `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, f` | 127 | 119 | -8 |
| `void flashinfer::sampling::TopPSamplingFromProbKernel<1024u, (cub::_V_300600_SM_1030::Bloc` | 36 | 41 | +6 |
| `_causal_conv1d_update_kernel` | 84 | 82 | -3 |
