| shape | M | cuBLAS kernels (median µs each) | total µs |
|---|---|---|---|
| in_proj_ba (64,2048) | 1 | `nvjet_sm103_tst_32x64_64x16_2x1_v_bz_splitK_TNN` (2.75) + `splitKreduce_kernel` (1.76) | 4.51 |
| in_proj_ba (64,2048) | 2 | `cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8` (2.69) + `splitKreduce_kernel` (1.79) | 4.48 |
| in_proj_ba (64,2048) | 3 | `cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8` (2.83) + `splitKreduce_kernel` (1.82) | 4.65 |
| in_proj_ba (64,2048) | 4 | `cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8` (2.82) + `splitKreduce_kernel` (1.98) | 4.8 |
| in_proj_ba (64,2048) | 5 | `cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8` (2.94) + `splitKreduce_kernel` (2.18) | 5.12 |
| in_proj_ba (64,2048) | 6 | `cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8` (2.94) + `splitKreduce_kernel` (1.86) | 4.8 |
| in_proj_ba (64,2048) | 8 | `cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8` (2.83) + `splitKreduce_kernel` (2.03) | 4.86 |
| shared_gate (1,2048) | 1 | `dot_kernel` (1.66) + `reduce_1Block_kernel` (1.76) | 3.42 |
| shared_gate (1,2048) | 2 | `dot_kernel` (1.79) + `reduce_1Block_kernel` (1.76) | 3.55 |
| shared_gate (1,2048) | 3 | `dot_kernel` (1.97) + `reduce_1Block_kernel` (1.76) | 3.73 |
| shared_gate (1,2048) | 4 | `dot_kernel` (1.97) + `reduce_1Block_kernel` (1.89) | 3.86 |
| shared_gate (1,2048) | 5 | `dot_kernel` (2.08) + `reduce_1Block_kernel` (1.92) | 4.0 |
| shared_gate (1,2048) | 6 | `dot_kernel` (2.11) + `reduce_1Block_kernel` (1.76) | 3.87 |
| shared_gate (1,2048) | 8 | `nvjet_sm103_tst_8x64_64x16_1x1_h_bz_splitK_TNN` (2.72) + `splitKreduce_kernel` (1.9) | 4.62 |
| router (256,2048) | 1 | `nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN` (2.98) + `splitKreduce_kernel` (1.92) | 4.9 |
| router (256,2048) | 2 | `cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8` (2.78) + `splitKreduce_kernel` (1.97) | 4.75 |
| router (256,2048) | 3 | `cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8` (2.88) + `splitKreduce_kernel` (2.02) | 4.9 |
| router (256,2048) | 4 | `nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN` (3.09) + `splitKreduce_kernel` (1.98) | 5.07 |
| router (256,2048) | 5 | `nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN` (3.17) + `splitKreduce_kernel` (1.98) | 5.15 |
| router (256,2048) | 6 | `nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN` (3.17) + `splitKreduce_kernel` (1.82) | 4.99 |
| router (256,2048) | 8 | `nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN` (3.09) + `splitKreduce_kernel` (1.9) | 4.99 |
| lm_head (248320,2048) | 1 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` (150.14) | 150.14 |
| lm_head (248320,2048) | 2 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` (152.53) | 152.53 |
| lm_head (248320,2048) | 3 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` (151.87) | 151.87 |
| lm_head (248320,2048) | 4 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` (152.93) | 152.93 |
| lm_head (248320,2048) | 5 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` (154.98) | 154.98 |
| lm_head (248320,2048) | 6 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` (155.15) | 155.15 |
| lm_head (248320,2048) | 8 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` (156.05) | 156.05 |
