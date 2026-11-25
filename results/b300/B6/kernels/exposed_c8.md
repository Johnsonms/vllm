c8: exposed total base 3437 pr 3313 us/step; overlapped (>=2 running) base 1361 pr 1212
| kernel | exposed base µs/step | exposed pr | Δ |
|---|---|---|---|
| `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd4x` | 658 | 518 | -140 |
| `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_ob` | 0 | 85 | +85 |
| `nvjet_sm103_tst_8x64_64x16_1x1_h_bz_splitK_TNN` | 81 | 0 | -81 |
| `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, 128, ` | 786 | 857 | +71 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16` | 49 | 0 | -49 |
| `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128_s8_et64x8_m64x8x32_c1x1x1_rM_BN_transOut_dsFp8_schPd` | 437 | 421 | -16 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` | 156 | 162 | +6 |
| `nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 6 | 0 | -6 |
| `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, t` | 0 | 5 | +5 |
| `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, f` | 122 | 127 | +5 |
