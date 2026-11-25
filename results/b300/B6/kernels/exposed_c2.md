c2: exposed total base 2010 pr 2123 us/step; overlapped (>=2 running) base 1414 pr 1148
| kernel | exposed base µs/step | exposed pr | Δ |
|---|---|---|---|
| `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_ob` | 0 | 213 | +213 |
| `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` | 152 | 0 | -152 |
| `void cutlass::device_kernel<vllm::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, 128, ` | 704 | 783 | +79 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a` | 130 | 109 | -20 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::Tens` | 48 | 61 | +12 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(cutlass_80` | 9 | 0 | -9 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat1` | 37 | 33 | -5 |
| `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, f` | 120 | 116 | -4 |
| `void per_token_group_quant_8bit_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, t` | 0 | 4 | +4 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:` | 120 | 117 | -2 |
