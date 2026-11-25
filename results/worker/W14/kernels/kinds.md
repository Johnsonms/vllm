
out/q38_tep4_c1.kernels.json: 1572 launches/step, 5481 us summed/step
| kind | launches/step | us/step | % of kernel time |
|---|---|---|---|
| hyper-connection (hc_*) | 290 | 820 | 15.0 |
| skinny GEMM (CuTe DSL) | 230 | 811 | 14.8 |
| routed MoE GEMM (bmm) | 96 | 752 | 13.7 |
| cuBLAS BF16 GEMM/GEMV | 243 | 735 | 13.4 |
| MoE routing/activation/finalize | 144 | 559 | 10.2 |
| all-reduce | 99 | 488 | 8.9 |
| torch elementwise/index/copy | 174 | 484 | 8.8 |
| attention (fmha, QSA, rope, cache) | 75 | 273 | 5.0 |
| GDN (conv, recurrent, norm, gates) | 72 | 197 | 3.6 |
| FP8 group quant | 48 | 126 | 2.3 |
| act_and_mul | 48 | 90 | 1.6 |
| norm (rms/layer) | 38 | 76 | 1.4 |
| other | 14 | 51 | 0.9 |
| sampling | 1 | 20 | 0.4 |

out/q38_tep4_c4.kernels.json: 1657 launches/step, 6736 us summed/step
| kind | launches/step | us/step | % of kernel time |
|---|---|---|---|
| cuBLAS BF16 GEMM/GEMV | 340 | 1173 | 17.4 |
| routed MoE GEMM (bmm) | 96 | 1095 | 16.3 |
| hyper-connection (hc_*) | 290 | 993 | 14.7 |
| skinny GEMM (CuTe DSL) | 181 | 774 | 11.5 |
| torch elementwise/index/copy | 211 | 625 | 9.3 |
| all-reduce | 99 | 611 | 9.1 |
| MoE routing/activation/finalize | 144 | 553 | 8.2 |
| attention (fmha, QSA, rope, cache) | 75 | 281 | 4.2 |
| GDN (conv, recurrent, norm, gates) | 72 | 214 | 3.2 |
| FP8 group quant | 48 | 149 | 2.2 |
| act_and_mul | 48 | 99 | 1.5 |
| norm (rms/layer) | 38 | 79 | 1.2 |
| other | 14 | 53 | 0.8 |
| sampling | 1 | 35 | 0.5 |

out/q38_tep4_c16.kernels.json: 1657 launches/step, 8934 us summed/step
| kind | launches/step | us/step | % of kernel time |
|---|---|---|---|
| routed MoE GEMM (bmm) | 96 | 2161 | 24.2 |
| cuBLAS BF16 GEMM/GEMV | 437 | 1906 | 21.3 |
| hyper-connection (hc_*) | 290 | 1059 | 11.9 |
| MoE routing/activation/finalize | 144 | 963 | 10.8 |
| torch elementwise/index/copy | 211 | 717 | 8.0 |
| all-reduce | 99 | 707 | 7.9 |
| skinny GEMM (CuTe DSL) | 84 | 341 | 3.8 |
| GDN (conv, recurrent, norm, gates) | 72 | 339 | 3.8 |
| attention (fmha, QSA, rope, cache) | 75 | 302 | 3.4 |
| FP8 group quant | 48 | 151 | 1.7 |
| act_and_mul | 48 | 98 | 1.1 |
| norm (rms/layer) | 38 | 87 | 1.0 |
| other | 14 | 56 | 0.6 |
| sampling | 1 | 47 | 0.5 |

out/q35_tp4_c1.kernels.json: 1590 launches/step, 7759 us summed/step
| kind | launches/step | us/step | % of kernel time |
|---|---|---|---|
| block FP8 linear GEMM (cutlass) | 240 | 2149 | 27.7 |
| routed MoE GEMM (bmm) | 120 | 1455 | 18.8 |
| all-reduce | 122 | 902 | 11.6 |
| FP8 group quant | 300 | 766 | 9.9 |
| MoE routing/activation/finalize | 180 | 723 | 9.3 |
| skinny GEMM (CuTe DSL) | 166 | 528 | 6.8 |
| torch elementwise/index/copy | 193 | 471 | 6.1 |
| GDN (conv, recurrent, norm, gates) | 150 | 337 | 4.3 |
| attention (fmha, QSA, rope, cache) | 47 | 243 | 3.1 |
| act_and_mul | 60 | 114 | 1.5 |
| sampling | 1 | 34 | 0.4 |
| other | 10 | 34 | 0.4 |
| norm (rms/layer) | 1 | 2 | 0.0 |

out/q35_tp4_c8.kernels.json: 1591 launches/step, 11174 us summed/step
| kind | launches/step | us/step | % of kernel time |
|---|---|---|---|
| routed MoE GEMM (bmm) | 120 | 3565 | 31.9 |
| block FP8 linear GEMM (cutlass) | 240 | 2423 | 21.7 |
| MoE routing/activation/finalize | 180 | 1131 | 10.1 |
| FP8 group quant | 300 | 912 | 8.2 |
| all-reduce | 122 | 804 | 7.2 |
| torch elementwise/index/copy | 194 | 618 | 5.5 |
| skinny GEMM (CuTe DSL) | 165 | 596 | 5.3 |
| GDN (conv, recurrent, norm, gates) | 150 | 476 | 4.3 |
| attention (fmha, QSA, rope, cache) | 47 | 275 | 2.5 |
| act_and_mul | 60 | 192 | 1.7 |
| cuBLAS BF16 GEMM/GEMV | 1 | 76 | 0.7 |
| sampling | 1 | 54 | 0.5 |
| other | 10 | 49 | 0.4 |
| norm (rms/layer) | 1 | 3 | 0.0 |
