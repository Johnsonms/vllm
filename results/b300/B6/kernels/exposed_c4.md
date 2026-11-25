c4: exposed total base 2983 pr 2379 us/step; overlapped (>=2 running) base 1229 pr 1377
| kernel | exposed base µs/step | exposed pr | Δ |
|---|---|---|---|
| `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x` | 370 | 17 | -353 |
| `void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<__nv_bfloat1` | 103 | 0 | -103 |
| `void reduce_1Block_kernel<float, 128, 7, cublasGemvTensorStridedBatched<float>, cublasGemv` | 77 | 0 | -77 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens` | 62 | 18 | -44 |
| `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, 128, ` | 809 | 771 | -38 |
| `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_ob` | 0 | 9 | +9 |
| `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_sch` | 282 | 290 | +8 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` | 156 | 162 | +6 |
| `nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 6 | 0 | -6 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:` | 118 | 113 | -5 |
