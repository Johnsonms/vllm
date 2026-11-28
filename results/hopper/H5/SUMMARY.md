# H5 — removing P2's permute overhead on SM90 (2026-11-28)

Qwen3-30B-A3B shapes (E=128, K=2048, I=768, k=8), random routing from `topk_softmax`. CUPTI, cold L2, CUDA graph, median of 5. **Measured on GPU 0**, which runs ~40 °C hotter and ~8% slower than the other H100s here (see STATUS), so all numbers are comparable within this table but ~8% higher than H2's `mb_h2.json`. Code stays local; pieces in `h5_bench.py`.

## Pieces (µs)
| piece | M=256 | 1024 | 4096 | 8192 |
|---|---:|---:|---:|---:|
| **default CuTe-DSL total** | **457.8** | **497.3** | **797.0** | **1332.5** |
| current P2 total (moe_permute + FC1 + FC2 + moe_unpermute) | 495.8 | 561.2 | 912.2 | 1572.8 |
| `moe_permute` (index + row copy) | 22.0 | 55.7 | 109.6 | 204.0 |
| **index build, Triton** (histogram + scan + atomic scatter → cu_seqlens, A_idx, inv) | 13.9 | 15.7 | 21.1 | 27.2 |
| row-gather copy, Triton (x[A_idx] → contiguous) | 4.1 | 14.7 | 68.1 | 156.1 |
| FC1 contiguous (pinned 128x192 pp c2x1) | 322.2 | 336.4 | 484.5 | 777.4 |
| FC1 with `A_idx` gather, pinned config | 556.9 | 529.7 | 617.8 | 953.0 |
| FC1 with `A_idx` gather, re-tuned (128x256 coop c2x1, the autotune winner at 1024/4096/8192) | 458.5 | 453.1 | 562.8 | 902.4 |
| FC2 (pinned 128x128 pp c1x2) | 155.0 | 165.2 | 235.4 | 395.5 |
| FC2 + router weight as a ColVec epilogue | 155.3 | 165.6 | 236.8 | 420.1 |
| `moe_unpermute` (weights × gather-sum) | 8.0 | 18.3 | 60.0 | 110.7 |
| combine, Triton gather-sum × weight | 6.7 | 18.8 | 60.3 | 111.6 |

## Totals vs default (default / variant; >1 = variant faster)
| variant | 256 | 1024 | 4096 | 8192 |
|---|---:|---:|---:|---:|
| current P2 | 0.923 | 0.886 | 0.874 | 0.847 |
| **Triton index + Triton gather copy + contiguous FC1 + FC2 + Triton combine (best)** | **0.939** | **0.938** | **0.965** | **0.947** |
| Triton index + FC1 `A_idx` (re-tuned) + FC2 + combine | 0.731 | 0.783 | 0.952 | 0.927 |
| Triton index + FC1 `A_idx` (pinned) + FC2 (+w in combine / in FC2) | 0.637 | 0.701 | 0.867 / 0.859 | 0.928 / 0.891 |
All variants reproduce current P2's output exactly (rel diff 0.0000; 0.0032 when the weight is applied in bf16 inside FC2).

## Findings
1. **Index build**: a 2-kernel Triton build (+ a 128-element cumsum) gives cu_seqlens, the sorted token index and the inverse map in 14–27 µs, vs `moe_permute`'s index part. Combined with a plain row-gather copy it beats `moe_permute` by 2–3x at M ≤ 1024 and ~1.3x at 8192 (the copy itself is bandwidth-bound).
2. **Gather inside FC1 does not pay on SM90.** SM90 has no TMA gather (`use_tma_gather` is SM100-only); QuACK's cp.async `A_idx` path costs +78 µs (re-tuned) to +133 µs (pinned) at M=4096 over the contiguous GEMM, i.e. more than the explicit copy (68 µs). It needs a different tile (128x256 cooperative) than the contiguous winner. At M ≤ 1024 it is far worse (+36–73%).
3. **Router weight in FC2's epilogue**: no gain; the combine is a bandwidth-bound gather-sum of k rows either way (same 60 µs at 4096), and the bf16 product adds rounding.
4. **Remaining gap**: the default's GEMM1 gathers for free inside a fused kernel; QuACK's GEMMs alone are at parity or better (contiguous FC1 ≈ default's fused FC1, FC2 ~7–10% faster per H2), but any materialised gather (~68 µs at 4096) plus the combine leaves the best variant 3.5–6% behind.

**Decision rule** (beat default by ≥5% at M≥1024): **not met** (best 0.94–0.965x). No E2E run. What would change this: an SM90 gather path in QuACK's mainloop that is as cheap as the default's fused gather (e.g. TMA-im2col-style row gather or a producer-warp gather into smem), or fusing the combine into FC2's store (scatter-add), which QuACK does not offer today.
