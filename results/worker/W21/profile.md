
### Qwen3-8B-FP8 traces_c1: 564 launches/step, 3322 us kernel time/step, 3390 us wall/step (first to last kernel)
| kind | launches/step | us/step | % of kernel time |
|---|---|---|---|
| DeepGEMM fp8 gemm | 144.0 | 1796 | 54.1 |
| rms norm | 145.0 | 430 | 12.9 |
| attention (FlashInfer) | 38.0 | 343 | 10.3 |
| FP8 group quant | 144.0 | 296 | 8.9 |
| cuBLAS (LM head) | 1.0 | 167 | 5.0 |
| rope / cache write | 36.0 | 98 | 3.0 |
| sampling | 3.0 | 74 | 2.2 |
| silu_and_mul | 36.0 | 61 | 1.8 |
| torch elementwise/index/copy | 11.0 | 33 | 1.0 |
| other | 6.0 | 24 | 0.7 |

DeepGEMM by layer (by launch order; 4.00 GEMMs per layer per step):
| qkv | 36.0 | 294 | 8.8 | 8.2 us/launch |
| o | 36.0 | 270 | 8.1 | 7.5 us/launch |
| gate_up | 36.0 | 710 | 21.4 | 19.7 us/launch |
| down | 36.0 | 522 | 15.7 | 14.5 us/launch |

top kernel names:
   54.1%    144.0/step      12.5 us  [DeepGEMM fp8 gemm] void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 1
    8.9%    144.0/step       2.1 us  [FP8 group quant] void per_token_group_quant_8bit_packed_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, 
    8.6%     36.0/step       8.0 us  [attention (FlashInfer)] fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP64MultiCtasKvVarSeqQ8Kv256StaticGro
    5.0%      1.0/step     167.0 us  [cuBLAS (LM head)] kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_ob
    4.3%     36.0/step       4.0 us  [rms norm] triton_red_fused_fused_add_rms_norm_per_token_group_fp8_quant_packed_0
    3.0%     36.0/step       2.7 us  [rope / cache write] void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheD
    2.8%     35.0/step       2.7 us  [rms norm] triton_per_fused_rms_norm_rotary_embedding_split_with_sizes_view_3
    2.8%     35.0/step       2.6 us  [rms norm] triton_red_fused_fused_add_rms_norm_per_token_group_fp8_quant_packed_2
    2.7%     35.0/step       2.5 us  [rms norm] triton_per_fused_rms_norm_rotary_embedding_split_with_sizes_view_4
    2.0%      1.0/step      66.7 us  [sampling] void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::na
    1.8%     36.0/step       1.7 us  [silu_and_mul] triton_poi_fused_mul_per_token_group_fp8_quant_packed_silu_slice_1
    1.1%      1.0/step      36.9 us  [attention (FlashInfer)] void flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA<1024u, 4u, false, float, int
    0.6%      5.0/step       4.0 us  [torch elementwise/index/copy] void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:
    0.6%      1.0/step      19.6 us  [attention (FlashInfer)] void flashinfer::sampling::TopPSamplingFromProbKernel<1024u, (cub::_V_300600_SM_1000::Bloc

### Qwen3-8B-FP8 traces_c32: 564 launches/step, 3814 us kernel time/step, 3851 us wall/step (first to last kernel)
| kind | launches/step | us/step | % of kernel time |
|---|---|---|---|
| DeepGEMM fp8 gemm | 144.0 | 1883 | 49.4 |
| attention (FlashInfer) | 38.0 | 682 | 17.9 |
| rms norm | 145.0 | 412 | 10.8 |
| FP8 group quant | 144.0 | 317 | 8.3 |
| cuBLAS (LM head) | 1.0 | 204 | 5.3 |
| rope / cache write | 36.0 | 95 | 2.5 |
| sampling | 3.0 | 79 | 2.1 |
| silu_and_mul | 36.0 | 71 | 1.9 |
| torch elementwise/index/copy | 11.0 | 39 | 1.0 |
| other | 6.0 | 33 | 0.9 |

DeepGEMM by layer (by launch order; 4.00 GEMMs per layer per step):
| qkv | 36.0 | 335 | 8.8 | 9.3 us/launch |
| o | 36.0 | 285 | 7.5 | 7.9 us/launch |
| gate_up | 36.0 | 709 | 18.6 | 19.7 us/launch |
| down | 36.0 | 554 | 14.5 | 15.4 us/launch |

top kernel names:
   49.4%    144.0/step      13.1 us  [DeepGEMM fp8 gemm] void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 1
   15.9%     36.0/step      16.9 us  [attention (FlashInfer)] fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP64VarSeqQ8Kv256PersistentGroupedSwa
    8.3%    144.0/step       2.2 us  [FP8 group quant] void per_token_group_quant_8bit_packed_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, 
    5.3%      1.0/step     204.0 us  [cuBLAS (LM head)] nvjet_sm100_tst_384x32_64x4_4x1f_2cta_v_bz_TNT
    3.4%     36.0/step       3.6 us  [rms norm] triton_red_fused_fused_add_rms_norm_per_token_group_fp8_quant_packed_0
    2.5%     36.0/step       2.6 us  [rope / cache write] void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheD
    2.5%     35.0/step       2.7 us  [rms norm] triton_per_fused_rms_norm_rotary_embedding_split_with_sizes_view_3
    2.3%     35.0/step       2.5 us  [rms norm] triton_red_fused_fused_add_rms_norm_per_token_group_fp8_quant_packed_2
    2.3%     35.0/step       2.5 us  [rms norm] triton_per_fused_rms_norm_rotary_embedding_split_with_sizes_view_4
    1.9%     36.0/step       2.0 us  [silu_and_mul] triton_poi_fused_mul_per_token_group_fp8_quant_packed_silu_slice_1
    1.8%      1.0/step      70.2 us  [sampling] void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::na
    1.0%      1.0/step      39.7 us  [attention (FlashInfer)] void flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA<1024u, 4u, false, float, int
    0.9%      1.0/step      35.1 us  [attention (FlashInfer)] void flashinfer::sampling::TopPSamplingFromProbKernel<1024u, (cub::_V_300600_SM_1000::Bloc
    0.4%      2.0/step       8.3 us  [torch elementwise/index/copy] void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso

### Qwen3-8B-FP8 traces_p4096: 198 launches/step, 11520 us kernel time/step, 17627 us wall/step (first to last kernel)
| kind | launches/step | us/step | % of kernel time |
|---|---|---|---|
| DeepGEMM fp8 gemm | 49.5 | 6492 | 56.4 |
| attention (FlashInfer) | 13.1 | 1936 | 16.8 |
| rms norm | 49.8 | 1204 | 10.5 |
| FP8 group quant | 49.5 | 848 | 7.4 |
| silu_and_mul | 12.4 | 682 | 5.9 |
| rope / cache write | 12.4 | 230 | 2.0 |
| cuBLAS (LM head) | 0.3 | 62 | 0.5 |
| sampling | 1.0 | 28 | 0.2 |
| other | 4.1 | 21 | 0.2 |
| torch elementwise/index/copy | 5.5 | 16 | 0.1 |

DeepGEMM by layer (by launch order; 1.38 GEMMs per layer per step):
| qkv | 12.4 | 889 | 7.7 | 71.9 us/launch |
| o | 12.4 | 599 | 5.2 | 48.4 us/launch |
| gate_up | 12.4 | 3296 | 28.6 | 266.4 us/launch |
| down | 12.4 | 1707 | 14.8 | 138.0 us/launch |

top kernel names:
   56.4%     49.5/step     131.1 us  [DeepGEMM fp8 gemm] void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 1
   16.6%     12.4/step     154.8 us  [attention (FlashInfer)] fmhaSm100aKernel_QkvBfloat16OBfloat16H128PagedKvCausalP64VarSeqQ256Kv128PersistentContext
    7.4%     49.5/step      17.1 us  [FP8 group quant] void per_token_group_quant_8bit_packed_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, 
    5.9%     12.4/step      55.1 us  [silu_and_mul] triton_poi_fused_mul_per_token_group_fp8_quant_packed_silu_slice_1
    4.0%     12.0/step      38.8 us  [rms norm] triton_per_fused_rms_norm_rotary_embedding_split_with_sizes_view_3
    2.8%     12.0/step      27.1 us  [rms norm] triton_red_fused_fused_add_rms_norm_per_token_group_fp8_quant_packed_2
    2.1%     12.4/step      19.9 us  [rms norm] triton_red_fused_fused_add_rms_norm_per_token_group_fp8_quant_packed_0
    2.0%     12.4/step      18.6 us  [rope / cache write] void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheD
    1.2%     12.0/step      11.0 us  [rms norm] triton_per_fused_rms_norm_rotary_embedding_split_with_sizes_view_4
    0.5%      0.3/step     180.7 us  [cuBLAS (LM head)] kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_ob
    0.2%      0.3/step      73.9 us  [sampling] void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::na
    0.1%      0.3/step      41.2 us  [attention (FlashInfer)] void flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA<1024u, 4u, false, float, int
    0.1%      0.3/step      39.0 us  [rms norm] triton_per_fused_rms_norm_rotary_embedding_split_with_sizes_view_1
    0.1%      0.3/step      25.5 us  [rms norm] triton_red_fused_fused_add_rms_norm_2

### Qwen3-32B-FP8 traces_c1: 984 launches/step, 8747 us kernel time/step, 9219 us wall/step (first to last kernel)
| kind | launches/step | us/step | % of kernel time |
|---|---|---|---|
| DeepGEMM fp8 gemm | 256.0 | 5915 | 67.6 |
| rms norm | 257.0 | 1183 | 13.5 |
| attention (FlashInfer) | 66.0 | 569 | 6.5 |
| FP8 group quant | 256.0 | 484 | 5.5 |
| cuBLAS (LM head) | 1.0 | 208 | 2.4 |
| rope / cache write | 64.0 | 154 | 1.8 |
| silu_and_mul | 64.0 | 109 | 1.2 |
| sampling | 3.0 | 71 | 0.8 |
| torch elementwise/index/copy | 11.0 | 30 | 0.3 |
| other | 6.0 | 22 | 0.3 |

DeepGEMM by layer (by launch order; 4.00 GEMMs per layer per step):
| qkv | 64.0 | 733 | 8.4 | 11.5 us/launch |
| o | 64.0 | 777 | 8.9 | 12.1 us/launch |
| gate_up | 64.0 | 2595 | 29.7 | 40.6 us/launch |
| down | 64.0 | 1810 | 20.7 | 28.3 us/launch |

top kernel names:
   67.6%    256.0/step      23.1 us  [DeepGEMM fp8 gemm] void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 1
    5.9%     64.0/step       8.1 us  [attention (FlashInfer)] fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP64MultiCtasKvVarSeqQ8Kv256StaticGro
    5.5%    256.0/step       1.9 us  [FP8 group quant] void per_token_group_quant_8bit_packed_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, 
    5.1%     64.0/step       6.9 us  [rms norm] triton_red_fused_fused_add_rms_norm_per_token_group_fp8_quant_packed_0
    4.4%     63.0/step       6.2 us  [rms norm] triton_red_fused_fused_add_rms_norm_per_token_group_fp8_quant_packed_2
    2.4%      1.0/step     207.5 us  [cuBLAS (LM head)] kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_ob
    1.9%     63.0/step       2.7 us  [rms norm] triton_per_fused_rms_norm_rotary_embedding_split_with_sizes_view_3
    1.9%     63.0/step       2.6 us  [rms norm] triton_per_fused_rms_norm_rotary_embedding_split_with_sizes_view_4
    1.8%     64.0/step       2.4 us  [rope / cache write] void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheD
    1.2%     64.0/step       1.7 us  [silu_and_mul] triton_poi_fused_mul_per_token_group_fp8_quant_packed_silu_slice_1
    0.7%      1.0/step      65.2 us  [sampling] void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::na
    0.4%      1.0/step      37.0 us  [attention (FlashInfer)] void flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA<1024u, 4u, false, float, int
    0.2%      5.0/step       3.8 us  [torch elementwise/index/copy] void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native:
    0.1%      1.0/step      12.9 us  [attention (FlashInfer)] void flashinfer::sampling::TopPSamplingFromProbKernel<1024u, (cub::_V_300600_SM_1000::Bloc

### Qwen3-32B-FP8 traces_c32: 984 launches/step, 9691 us kernel time/step, 9590 us wall/step (first to last kernel)
| kind | launches/step | us/step | % of kernel time |
|---|---|---|---|
| DeepGEMM fp8 gemm | 256.0 | 6173 | 63.7 |
| attention (FlashInfer) | 66.0 | 1145 | 11.8 |
| rms norm | 257.0 | 1072 | 11.1 |
| FP8 group quant | 256.0 | 568 | 5.9 |
| cuBLAS (LM head) | 1.0 | 249 | 2.6 |
| rope / cache write | 64.0 | 166 | 1.7 |
| silu_and_mul | 64.0 | 156 | 1.6 |
| sampling | 3.0 | 79 | 0.8 |
| other | 6.0 | 47 | 0.5 |
| torch elementwise/index/copy | 11.0 | 36 | 0.4 |

DeepGEMM by layer (by launch order; 4.00 GEMMs per layer per step):
| qkv | 64.0 | 768 | 7.9 | 12.0 us/launch |
| o | 64.0 | 800 | 8.3 | 12.5 us/launch |
| gate_up | 64.0 | 2679 | 27.6 | 41.9 us/launch |
| down | 64.0 | 1927 | 19.9 | 30.1 us/launch |

top kernel names:
   63.7%    256.0/step      24.1 us  [DeepGEMM fp8 gemm] void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 1
   11.1%     64.0/step      16.7 us  [attention (FlashInfer)] fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP64VarSeqQ8Kv256PersistentGroupedSwa
    5.9%    256.0/step       2.2 us  [FP8 group quant] void per_token_group_quant_8bit_packed_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, 
    3.9%     64.0/step       5.9 us  [rms norm] triton_red_fused_fused_add_rms_norm_per_token_group_fp8_quant_packed_0
    3.6%     63.0/step       5.6 us  [rms norm] triton_red_fused_fused_add_rms_norm_per_token_group_fp8_quant_packed_2
    2.6%      1.0/step     249.3 us  [cuBLAS (LM head)] nvjet_sm100_tst_384x32_64x4_4x1f_2cta_v_bz_TNT
    1.8%     63.0/step       2.7 us  [rms norm] triton_per_fused_rms_norm_rotary_embedding_split_with_sizes_view_3
    1.7%     64.0/step       2.6 us  [rope / cache write] void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheD
    1.6%     64.0/step       2.4 us  [silu_and_mul] triton_poi_fused_mul_per_token_group_fp8_quant_packed_silu_slice_1
    1.6%     63.0/step       2.4 us  [rms norm] triton_per_fused_rms_norm_rotary_embedding_split_with_sizes_view_4
    0.7%      1.0/step      69.3 us  [sampling] void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::na
    0.4%      1.0/step      39.0 us  [attention (FlashInfer)] void flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA<1024u, 4u, false, float, int
    0.4%      1.0/step      35.0 us  [attention (FlashInfer)] void flashinfer::sampling::TopPSamplingFromProbKernel<1024u, (cub::_V_300600_SM_1000::Bloc
    0.3%      1.0/step      28.1 us  [other] _temperature_kernel

### Qwen3-32B-FP8 traces_p4096: 342 launches/step, 49062 us kernel time/step, 56056 us wall/step (first to last kernel)
| kind | launches/step | us/step | % of kernel time |
|---|---|---|---|
| DeepGEMM fp8 gemm | 88.0 | 31776 | 64.8 |
| attention (FlashInfer) | 22.7 | 6962 | 14.2 |
| rms norm | 88.3 | 3956 | 8.1 |
| FP8 group quant | 88.0 | 2974 | 6.1 |
| silu_and_mul | 22.0 | 2765 | 5.6 |
| rope / cache write | 22.0 | 477 | 1.0 |
| cuBLAS (LM head) | 0.3 | 78 | 0.2 |
| sampling | 1.0 | 35 | 0.1 |
| other | 4.1 | 22 | 0.0 |
| torch elementwise/index/copy | 5.5 | 18 | 0.0 |

DeepGEMM by layer (by launch order; 1.38 GEMMs per layer per step):
| qkv | 22.0 | 3566 | 7.3 | 162.1 us/launch |
| o | 22.0 | 2823 | 5.8 | 128.3 us/launch |
| gate_up | 22.0 | 17025 | 34.7 | 773.8 us/launch |
| down | 22.0 | 8363 | 17.0 | 380.1 us/launch |

top kernel names:
   64.8%     88.0/step     361.1 us  [DeepGEMM fp8 gemm] void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 1
   14.1%     22.0/step     315.1 us  [attention (FlashInfer)] fmhaSm100aKernel_QkvBfloat16OBfloat16H128PagedKvCausalP64VarSeqQ256Kv128PersistentContext
    6.1%     88.0/step      33.8 us  [FP8 group quant] void per_token_group_quant_8bit_packed_register_kernel<c10::BFloat16, c10::Float8_e4m3fn, 
    5.6%     22.0/step     125.7 us  [silu_and_mul] triton_poi_fused_mul_per_token_group_fp8_quant_packed_silu_slice_1
    3.7%     21.7/step      84.9 us  [rms norm] triton_per_fused_rms_norm_rotary_embedding_split_with_sizes_view_3
    2.1%     21.7/step      47.5 us  [rms norm] triton_red_fused_fused_add_rms_norm_per_token_group_fp8_quant_packed_2
    1.5%     22.0/step      33.1 us  [rms norm] triton_red_fused_fused_add_rms_norm_per_token_group_fp8_quant_packed_0
    1.0%     22.0/step      21.7 us  [rope / cache write] void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheD
    0.6%     21.7/step      13.9 us  [rms norm] triton_per_fused_rms_norm_rotary_embedding_split_with_sizes_view_4
    0.2%      0.3/step     226.1 us  [cuBLAS (LM head)] kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_skinny_gemmCuteSkinnyGemm_ob
    0.1%      0.3/step      94.3 us  [sampling] void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::na
    0.1%      0.3/step      86.6 us  [rms norm] triton_per_fused_rms_norm_rotary_embedding_split_with_sizes_view_1
    0.0%      0.3/step      52.5 us  [attention (FlashInfer)] void flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA<1024u, 4u, false, float, int
    0.0%      0.3/step      40.0 us  [rms norm] triton_red_fused_fused_add_rms_norm_2
