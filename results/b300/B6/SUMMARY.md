# B6 — c2/c4/c8 decode profile on B300: what causes the c4 step (2026-11-25)

Qwen3.6-35B-A3B-FP8, TP1, real weights, GPU 0, idle node. B1 profiling method: torch profiler, skip 24 engine steps, record 32. B3 serve flags. Arms as in B3: `base` = e66bede56b; `pr` = 81159c69d6 table + `(10, 3)`. The server logs confirm sha1 c41c6f99d66a / `(10, 3)` False for base and `(10, 3)` True for pr; AOT loads 0.

## Answer
1. **The c2 → c4 step is not a cuBLAS kernel.** cuBLAS BF16 time per step is flat from c2 to c4 in base (25.2 vs 25.0 ms over 32 steps). The largest kernel increase in both arms is the **routed-MoE FP8 GEMMs** (FlashInfer TRT-LLM `bmm_*`): +530 µs/step, from 873 to 1403. They also switch tile at M=4: `t128x16x128_s6` → `t128x8x128u2_s8`. This growth matches the number of active experts. With 256 experts and top-8, the expected count of distinct experts is 15.5 / 30.1 / 56.7 at M = 2 / 4 / 8, and the cost per distinct expert falls smoothly (1.41 / 1.17 / 0.83 µs per layer). So the tile choice is not a bad outlier. Both arms pay this cost.
2. **What is specific to base at c4 is lost overlap between the shared-expert aux stream and the routed MoE.** Per-step GPU-busy union (time with at least one kernel running) matches the measured TPOT:

| | c2 | c4 | c8 |
|---|---|---|---|
| base: TPOT (ms) / busy union (ms) | 3.338 / 3.43 | 4.183 / 4.21 | 4.795 / 4.80 |
| pr: TPOT (ms) / busy union (ms) | 3.225 / 3.27 | 3.720 / 3.76 | 4.569 / 4.52 |
| summed kernel time base → pr (µs/step) | 5316 → 4806 (−510) | 5997 → 5746 (−251) | 6693 → 6243 (−450) |
| overlap base → pr (µs/step) | 1882 → 1535 | **1784 → 1986 (+202)** | 1897 → 1719 |
| busy union base → pr (µs/step) | −162 | **−454** | −272 |

   At c2 and c8, pr's gain is mostly kernel time saved. At c4 the skinny plans save *less* kernel time (−251 µs) but recover +202 µs of overlap, giving the large TPOT gain. From c2 to c4, overlap falls by 98 µs in base and rises by 451 µs in pr.

3. **The mechanism, from exposed time (time a kernel runs alone, summed over all 32 steps)** at c4:

| kernel | exposed base (µs/step) | exposed pr (µs/step) |
|---|---|---|
| `bmm_E4m3…t128x8x128u2_s8` (routed-MoE gate-up) | 370 | 17 |
| `dot_kernel` + `reduce_1Block_kernel` (cuBLAS GEMV, shared-expert gate (1, 2048)) | 180 | 0 |
| total exposed | 2983 | 2379 |

   The single-layer timeline (`kernels/layer_{base,pr}_c4.txt`) shows how it happens. In base, the shared expert's `act_and_mul` could start once its gate-up GEMM ends (~+2.6 µs) but starts at +13.2 µs. Its down-proj GEMM starts at +28.9 µs, about 11 µs after its input was ready. Both wait while the routed-MoE `t128x8 s8` GEMMs hold the SMs. After that, the two-kernel cuBLAS shared-gate GEMV runs on the critical path, after the MoE finalize. In pr, the router is a single 2 µs skinny kernel instead of nvjet split-K + reduce (~6.6 µs, 2 launches). That lets the shared expert's kernels start early enough to run alongside the first MoE GEMM, and the 1.7 µs skinny shared gate lands in the MoE's shadow. Each layer ends about 5 µs earlier in pr (× 40 layers ≈ 200 µs, matching the +202 µs overlap).
   At c8, the shared expert waits behind the MoE GEMMs in **both** arms (`kernels/layer_*_c8.txt`: `act_and_mul` at +22 / +27 µs). That is why pr's c8 gain is only kernel time (−3.7%).
   This is an interpretation of one profiled run per arm and C. The stream labels in a CUDA-graph replay are the graph's own streams, so I rely on exposed time and busy union rather than on per-stream sums.

4. **Does it affect other BF16 GEMMs at M=4?** Not as a slow cuBLAS kernel. Standalone, cuBLAS at M=4 is 4.8–5.1 µs for the small shapes, the same as at M=2 and M=8. The cost is structural: cuBLAS needs two launches (GEMM + split-K reduce, or GEMV dot + reduce) for these skinny shapes, and on the critical path next to a stream that fills the SMs those launches serialize. The same pattern can appear wherever a short aux-stream branch has to fit around a GEMM that saturates the SMs. The skinny plans help here by being one short kernel.

## cuBLAS kernels on SM103 (standalone `F.linear`, M = 2/4/8; full M = 1..8 in `kernels/cublas_table.md`)
| shape | M | cuBLAS kernels (median µs each) | total µs |
|---|---|---|---|
| in_proj_ba (64,2048) | 2 | `cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8` (2.69) + `splitKreduce_kernel` (1.79) | 4.48 |
| in_proj_ba (64,2048) | 4 | `cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8` (2.82) + `splitKreduce_kernel` (1.98) | 4.80 |
| in_proj_ba (64,2048) | 8 | `cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8` (2.83) + `splitKreduce_kernel` (2.03) | 4.86 |
| shared_gate (1,2048) | 2 | `dot_kernel` (1.79) + `reduce_1Block_kernel` (1.76) | 3.55 |
| shared_gate (1,2048) | 4 | `dot_kernel` (1.97) + `reduce_1Block_kernel` (1.89) | 3.86 |
| shared_gate (1,2048) | 8 | `nvjet_sm103_tst_8x64_64x16_1x1_h_bz_splitK_TNN` (2.72) + `splitKreduce_kernel` (1.90) | 4.62 |
| router (256,2048) | 2 | `cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8` (2.78) + `splitKreduce_kernel` (1.97) | 4.75 |
| router (256,2048) | 4 | `nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN` (3.09) + `splitKreduce_kernel` (1.98) | 5.07 |
| router (256,2048) | 8 | `nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN` (3.09) + `splitKreduce_kernel` (1.90) | 4.99 |
| lm_head (248320,2048) | 2 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` (152.53) | 152.53 |
| lm_head (248320,2048) | 4 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` (152.93) | 152.93 |
| lm_head (248320,2048) | 8 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` (156.05) | 156.05 |

These are eager torch-profiler times (warm L2, 50 calls). The router switches from `cutlass_80` to nvjet split-K exactly at M=4, and the shared gate switches from GEMV to nvjet split-K at M=8. In the server trace, base c4 has `nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN` 40 calls/step (router) and `splitKreduce` 70 calls/step (router + in_proj_ba). The router's own switch costs only ~0.3 µs per call standalone.

## Base kernel changes c2 → c4 → c8 (µs/step; `kernels/diff_base.md`, `diff_pr.md`)
| kernel | c2 | c4 | c8 | calls/step |
|---|---|---|---|---|
| MoE gate-up `bmm_E4m3` (t128x16 s6 at c2, t128x8 u2 s8 at c4/c8) | 467 | 770 | 1118 | 40 |
| MoE down `bmm_Bfloat16` (t128x16 u2 s6 / t128x8 u2 s8 / t128x8 s8) | 406 | 633 | 760 | 40 |
| MoE `activationDeepSeekKernel` | 200 | 292 | 308 | 40 |
| GDN `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 100 | 161 | 217 | 30 |
| attention `fmhaSm100f…` (Cga variant at c2, non-Cga at c4/c8) | 79 | 93 | 95 | 10 |
| cuBLAS BF16, all kernels | 789 | 780 | 846 | 221 |
| summed kernel time | 5316 | 5997 | 6693 | |

## Notes
- One profiled run per arm and C, 32 steps each (prompts = C, 512 in / 128 out). The unprofiled TPOT runs in the same server reproduce B3 within 0.03 ms (base 3.338 / 4.183 / 4.795; pr 3.225 / 3.720 / 4.569).
- Profiled wall time per step (~5 ms) is inflated by the profiler. I use the GPU busy union instead, which tracks TPOT within 0.1 ms.
- Possible follow-up (not done): the aux-stream starvation at c8 is common to both arms. Launching the shared expert before the router, or giving the aux stream priority, may recover ~5 µs/layer at c ≥ 4. That would be a separate code change and is not tested here.

## Files
- `kernels/{base,pr}_c{2,4,8}.kernels.json`: all kernels (µs, calls). `*_top25_c*.md`: top-25 tables. `*_tpot_c*.json`: unprofiled TPOT.
- `kernels/union.txt`, `exposed_c{2,4,8}.md`, `diff_{base,pr}.md`, `layer_{base,pr}_c{4,8}.txt`, `cublas_table.md`, `cublas_names.json`.
- `scripts/`: `b6.sh` (runner), `kernel_table.py`, `diff_kernels.py`, `union.py`, `exposed.py`, `layer_view.py`, `by_stream.py`, `timeline.py`, `cublas_names.py`.
