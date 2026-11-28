# W22: residual-add + RMSNorm folded into QuACK GEMM epilogues (Qwen3 dense BF16 shapes, B200)

2026-11-28 16:46Z. quack-kernels 0.6.5, torch 2.14.0+cu132, one B200, CUDA graph, cold L2, CUPTI median. Random BF16 weights with the Qwen3-8B / 32B shapes. No vLLM edits.

## Verdict: no-go
Folding the residual add and the RMSNorm into the GEMM epilogues **does not beat the same chain with the Inductor add+norm kernel left in place**. The go threshold (chain >= 1.10x at M >= 32) is met at one point only (8B chain 1, M = 512, 1.11x), and that gain comes from the gated GEMM, not from the fold.
- Chain 1 (o_proj -> add+norm -> gate_up -> silu-mul): fold vs today 0.98-1.11x (8B), 1.01-1.08x (32B). Fold vs "QuACK gated GEMM + Inductor add+norm": 0.96-1.01x, with one outlier (32B M = 8192, 1.12x).
- Chain 2 (down_proj -> add+norm -> qkv_proj): fold is **slower** than today: 0.88-0.98x (8B), 0.83-0.99x (32B).

## Why
The fold removes one small kernel and adds more than it removes:
- QuACK `gemm_rms` = GEMM with residual add, per-tile sum of squares and norm weight in the epilogue, **plus a second kernel** (`rms_final_reduce`: sum the per-tile partials and take rsqrt). It costs 3.0-3.9 us more than the plain QuACK GEMM at M <= 128 on 8B (4.5-7.4 us on 32B), 7-11 us at M = 512, 11-28 us at M = 2048.
- The Inductor fused add+norm kernel it replaces costs 2.2-3.2 us at M <= 128, 4.2-5.1 us at M = 512, 12-17 us at M = 2048.
- Only at M = 8192 is the epilogue cheaper than the separate kernel (28-36 us vs 42-55 us).
- Applying rstd in the next GEMM's epilogue is free at M <= 512 (`norm-gated` = `gated` within 0.3 us) and costs 5-7 % of the gate_up GEMM at M >= 2048.
- For chain 2 there is no gated gain to hide behind, and QuACK's plain BF16 GEMM is not faster than cuBLAS for down_proj / qkv_proj at these sizes (0.91-1.06x).

## QuACK limitations found
- **The row reduce is not single-kernel.** Each CTA tile writes a partial sum of squares (an M x n_tiles fp32 buffer) and a separate finalize kernel reduces it. This extra launch is the whole loss at small M.
- `gemm_norm_act` has no `concat_layout`, so the merged gate_up weight must be stored with gate / up rows interleaved.
- The library pieces exist for everything asked: `gemm_rms` (= `rms_partial_epi` / `sq_reduce_mod`) and `gemm_norm_act(rstd=..., activation="swiglu")` (= `rstd_gated_mod`). `gemm_norm_act(activation=None)` works for the qkv case. No new epilogue had to be written.

## Accuracy
The fold is slightly **more** accurate (rstd comes from the fp32 accumulator, not from BF16): error vs FP32 for the activation 0.0029 (fold) vs 0.0034 (gated) vs 0.0042 (today); residual stream 0.0017 vs 0.0022.

## Chain 1: o_proj -> add residual -> RMSNorm -> gate_up -> silu-mul (us)
base = cuBLAS + Inductor add+norm + cuBLAS + Inductor silu-mul. gated = QuACK GEMM + Inductor add+norm + QuACK gated GEMM. fold = `gemm_rms` + `gemm_norm_act`.

| M | 8B base | 8B gated | 8B fold | base/fold | gated/fold | 32B base | 32B gated | 32B fold | base/fold | gated/fold |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 60.6 | 61.9 | 61.9 | 0.98 | 1.00 | 123.7 | 117.8 | 119.5 | 1.03 | 0.99 |
| 32 | 62.0 | 62.5 | 62.2 | 1.00 | 1.00 | 125.2 | 117.1 | 119.6 | 1.05 | 0.98 |
| 64 | 62.9 | 63.3 | 62.7 | 1.00 | 1.01 | 127.8 | 117.8 | 120.8 | 1.06 | 0.98 |
| 128 | 63.3 | 62.7 | 62.1 | 1.02 | 1.01 | 123.5 | 122.0 | 122.4 | 1.01 | 1.00 |
| 512 | 101.5 | 90.2 | 91.2 | 1.11 | 0.99 | 225.5 | 216.8 | 215.8 | 1.05 | 1.00 |
| 2048 | 323.9 | 306.5 | 319.5 | 1.01 | 0.96 | 819.4 | 762.7 | 776.7 | 1.05 | 0.98 |
| 8192 | 1288.8 | 1191.0 | 1187.8 | 1.09 | 1.00 | 3256.6 | 3389.4 | 3020.2 | 1.08 | 1.12 |

## Chain 2: down_proj -> add residual -> RMSNorm -> qkv_proj (us)
quack = QuACK GEMM + Inductor add+norm + QuACK GEMM. fold = `gemm_rms` + `gemm_norm_act(activation=None)`.

| M | 8B base | 8B quack | 8B fold | base/fold | 32B base | 32B quack | 32B fold | base/fold |
|---|---|---|---|---|---|---|---|---|
| 16 | 41.4 | 45.1 | 47.0 | 0.88 | 83.0 | 81.3 | 89.5 | 0.93 |
| 32 | 42.0 | 45.6 | 47.9 | 0.88 | 81.6 | 80.0 | 89.0 | 0.92 |
| 64 | 42.8 | 46.1 | 48.6 | 0.88 | 83.7 | 82.0 | 91.3 | 0.92 |
| 128 | 46.2 | 48.1 | 50.0 | 0.93 | 87.3 | 87.0 | 90.1 | 0.97 |
| 512 | 72.6 | 74.3 | 79.4 | 0.91 | 129.2 | 141.4 | 155.1 | 0.83 |
| 2048 | 201.4 | 208.9 | 222.9 | 0.90 | 505.3 | 475.9 | 517.1 | 0.98 |
| 8192 | 803.1 | 803.4 | 821.8 | 0.98 | 1929.6 | 1982.9 | 1943.2 | 0.99 |

## Pieces, 8B (us)
| M | QuACK o_proj | + rms epilogue and finalize | extra | Inductor add+norm | QuACK gated | QuACK gated with rstd |
|---|---|---|---|---|---|---|
| 16 | 12.2 | 15.5 | 3.3 | 2.2 | 46.1 | 46.2 |
| 32 | 12.2 | 16.1 | 3.9 | 3.2 | 46.2 | 46.3 |
| 128 | 13.4 | 16.5 | 3.0 | 2.5 | 46.5 | 46.8 |
| 512 | 18.5 | 25.9 | 7.4 | 4.2 | 68.1 | 68.4 |
| 2048 | 48.4 | 59.7 | 11.3 | 12.2 | 245.2 | 257.5 |
| 8192 | 159.2 | 187.6 | 28.4 | 42.3 | 977.0 | 1047.9 |

All pieces for both models are in `bench_8b.jsonl` / `bench_32b.jsonl` (keys `p_*`).

## Notes
- In this microbench the BF16 gated GEMM alone gives 0.98-1.12x on chain 1, smaller than its share would suggest, because the o_proj GEMM and the add+norm are in the chain too.
- QuACK timings use its autotuner (best case for QuACK). One layer's worth of shapes, random activations.
- What would change the verdict: a single-kernel row reduce (one CTA owning a full row of N, or the finalize done in the consuming GEMM's prologue). That is a QuACK change.
