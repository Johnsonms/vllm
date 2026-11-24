# W11a — SM100 generic skinny plans for Qwen3.5 / Qwen3.6 MoE (FP8 checkpoints)

Node: 8x B200, vLLM main @ 8399da3780 (precompiled), no vLLM file edited.
Status: **done (sweep 18:37Z, verify and checks 18:45Z).** E2E not run here; that is the manager's part.

## Result
- **49 entries on 14 shapes, all pass the verify (median >= 1.10x in 4 seeds).** `verified.json` equals `proposed.json`.
- Small projections (GDN `in_proj_ba`, shared-expert gate, router): **1.49-3.29x** at M = 1-8, 4.4-7.3 us -> 1.9-4.8 us.
- LM heads: **1.11-1.42x** at M = 1-2, and at M = 4 for the three shards of 62,080 rows or fewer. M = 8 loses on every LM head, and M = 4 is under 1.10x on the two largest; those cells have no entry.
- **N = 1 reaches the skinny kernel on SM100 without the #64982 `layers/utils.py` change** (section 4). N = 1 is in scope.
- Weakest entries, in case the manager wants a margin: (62080, 2048) M=4 at 1.11x (min 1.10) and (248320, 2048) M=1 at 1.11x.
- Not measured: E2E, and the host cost of the custom-op dispatch at unplanned token counts (above 8), which is what #64982's eager fast path is for. Please include a concurrency above 8 in the E2E.

## 1. Unquantized layers (module walk, `load_format=dummy`)

Hooked `UnquantizedLinearMethod` and `UnquantizedEmbeddingMethod` (`process_weights_after_loading` and `apply`) in-process and
ran the model once. Local weight shapes per rank, all BF16, no bias, `use_skinny_gemm=False` at base:

| layer | class | count | Qwen3.6-35B TP1 | TP2 | TP4 | Qwen3.5-397B TP4 | TP8 |
|---|---|---|---|---|---|---|---|
| GDN `linear_attn.in_proj_ba` | MergedColumnParallelLinear | 30 / 45 | (64, 2048) | (32, 2048) | (16, 2048) | (32, 4096) | (16, 4096) |
| `mlp.shared_expert_gate` | ReplicatedLinear | 40 / 60 | (1, 2048) | same | same | (1, 4096) | same |
| `mlp.gate` (router) | GateLinear | 40 / 60 | (256, 2048) | same | same | (512, 4096) | same |
| `lm_head` | ParallelLMHead | 1 | (248320, 2048) | (124160, 2048) | (62080, 2048) | (62080, 4096) | (31040, 4096) |

- Also unquantized but not a GEMM: `linear_attn.conv1d` (3-D weight).
- The router is a `GateLinear`, but for this model its `apply` goes through `UnquantizedLinearMethod.apply` with a BF16 `(M, K)`
  input at every M (seen in the walk), i.e. the plain `F.linear` tier. So a generic plan entry takes effect for the router too.
- The LM head goes through `UnquantizedEmbeddingMethod.apply`, which also calls `maybe_use_skinny_gemm`.
- None of the 14 shapes is in `results/manager/W11_existing_shapes.json`.
- Walks run: Qwen3.6-35B at TP1, TP2, TP4; Qwen3.5-397B at TP4 and TP8 (no expert parallel).

## 2. Sweep (manager's `sweep_skinny_generic.py`, one M per GPU, CUPTI, CUDA graph, cold L2)

Raw records: `sweep.jsonl`. Proposed entries (best config where cuBLAS / best >= 1.10): `proposed.json`, **49 entries on 14 shapes**.
Times in us, cuBLAS -> best skinny config:

| (N, K) | M=1 | M=2 | M=4 | M=8 |
|---|---|---|---|---|
| (1, 2048) | 4.61 -> 1.86 (2.48x) | 4.61 -> 1.89 (2.44x) | 4.80 -> 2.05 (2.34x) | 4.86 -> 2.27 (2.14x) |
| (16, 2048) | 4.74 -> 1.95 (2.43x) | 4.96 -> 2.02 (2.46x) | 4.99 -> 2.18 (2.29x) | 4.93 -> 2.30 (2.14x) |
| (32, 2048) | 4.96 -> 2.05 (2.42x) | 5.66 -> 2.08 (2.72x) | 5.89 -> 2.21 (2.67x) | 5.70 -> 2.40 (2.37x) |
| (64, 2048) | 4.86 -> 2.11 (2.30x) | 5.15 -> 2.18 (2.37x) | 5.54 -> 2.27 (2.44x) | 5.15 -> 2.46 (2.09x) |
| (256, 2048) | 5.66 -> 2.43 (2.33x) | 6.02 -> 2.46 (2.44x) | 5.86 -> 2.59 (2.26x) | 5.57 -> 2.85 (1.96x) |
| (62080, 2048) | 50.46 -> 39.30 (1.28x) | 51.04 -> 39.42 (1.29x) | 51.20 -> 46.27 (1.11x) | 50.14 -> 63.90 (0.78x, drop) |
| (124160, 2048) | 88.48 -> 74.78 (1.18x) | 88.54 -> 73.47 (1.21x) | 89.02 -> 83.52 (1.07x, drop) | 89.17 -> 119.2 (0.75x, drop) |
| (248320, 2048) | 161.15 -> 144.10 (1.12x) | 161.57 -> 142.27 (1.14x) | 162.53 -> 156.99 (1.04x, drop) | 163.49 -> 229.5 (0.71x, drop) |
| (1, 4096) | 4.42 -> 1.98 (2.23x) | 4.58 -> 2.05 (2.24x) | 4.74 -> 2.11 (2.24x) | 4.48 -> 2.43 (1.84x) |
| (16, 4096) | 5.15 -> 2.11 (2.44x) | 7.26 -> 2.21 (3.29x) | 7.42 -> 2.30 (3.22x) | 6.85 -> 2.66 (2.58x) |
| (32, 4096) | 5.02 -> 2.18 (2.31x) | 5.70 -> 2.24 (2.54x) | 5.47 -> 2.34 (2.34x) | 5.57 -> 2.69 (2.07x) |
| (512, 4096) | 7.30 -> 3.26 (2.24x) | 7.39 -> 3.36 (2.20x) | 7.36 -> 3.68 (2.00x) | 7.30 -> 4.61 (1.58x) |
| (31040, 4096) | 57.15 -> 38.56 (1.48x) | 57.12 -> 39.52 (1.45x) | 56.90 -> 46.27 (1.23x) | 57.79 -> 61.54 (0.94x, drop) |
| (62080, 4096) | 98.85 -> 72.86 (1.36x) | 99.62 -> 75.62 (1.32x) | 100.86 -> 85.31 (1.18x) | 89.57 -> 113.7 (0.79x, drop) |

Caveats on the sweep numbers:
- One run per cell. For the small shapes the top three configs are within a few percent of each other, so the chosen config
  is partly noise; the verify (section 3) decides.
- The Qwen3.6 sweep (GPUs 0-3) overlapped the 397B module walk on GPUs 4-7 for its first minutes. The verify ran on an idle node.

## 3. Verify (4 seeds on 4 GPUs, 3 interleaved rounds each, CUPTI, CUDA graph, cold L2, idle node)

Per seed the middle of 3 rounds is taken; cells show median cuBLAS -> median plan in us (median speedup, min-max over the 4 seeds).
Raw: `verify.jsonl`; per-cell summary: `verify_summary.json`.

| layer | (N, K) | M=1 | M=2 | M=4 | M=8 |
|---|---|---|---|---|---|
| Q3.6 TP1 `in_proj_ba` | (64, 2048) | 5.06 -> 2.19 (2.31x, 2.28-2.50) | 5.60 -> 2.27 (2.45x, 2.38-2.49) | 5.54 -> 2.27 (2.44x, 2.37-2.51) | 5.76 -> 2.54 (2.26x, 2.20-2.42) |
| Q3.6 TP2 `in_proj_ba` | (32, 2048) | 4.85 -> 2.10 (2.31x, 2.26-2.38) | 5.47 -> 2.10 (2.63x, 2.53-2.71) | 5.74 -> 2.21 (2.62x, 2.55-2.65) | 5.63 -> 2.35 (2.39x, 2.30-2.53) |
| Q3.6 TP4 `in_proj_ba` | (16, 2048) | 4.77 -> 2.05 (2.37x, 2.27-2.43) | 4.91 -> 2.11 (2.34x, 2.29-2.58) | 5.28 -> 2.18 (2.44x, 2.35-2.59) | 5.44 -> 2.30 (2.35x, 2.31-2.40) |
| Q3.6 shared-expert gate (all TP) | (1, 2048) | 4.35 -> 1.87 (2.33x, 2.31-2.37) | 4.91 -> 1.95 (2.54x, 2.41-2.59) | 4.72 -> 2.05 (2.32x, 2.20-2.41) | 4.74 -> 2.24 (2.10x, 2.07-2.49) |
| Q3.6 router (all TP) | (256, 2048) | 5.44 -> 2.42 (2.26x, 2.24-2.34) | 5.81 -> 2.50 (2.33x, 2.29-2.38) | 5.63 -> 2.59 (2.19x, 2.10-2.22) | 5.50 -> 2.91 (1.90x, 1.88-1.95) |
| Q3.6 TP1 LM head | (248320, 2048) | 160.56 -> 144.61 (1.11x, 1.11-1.12) | 161.66 -> 142.11 (1.14x, 1.13-1.14) | cuBLAS (no entry) | cuBLAS (no entry) |
| Q3.6 TP2 LM head | (124160, 2048) | 88.69 -> 74.77 (1.19x, 1.19-1.19) | 88.69 -> 75.14 (1.18x, 1.18-1.19) | cuBLAS (no entry) | cuBLAS (no entry) |
| Q3.6 TP4 LM head | (62080, 2048) | 50.69 -> 39.33 (1.29x, 1.29-1.29) | 51.28 -> 39.41 (1.30x, 1.30-1.30) | 51.14 -> 46.24 (1.11x, 1.10-1.11) | cuBLAS (no entry) |
| Q3.5-397B TP4 `in_proj_ba` | (32, 4096) | 4.82 -> 2.27 (2.12x, 2.10-2.28) | 6.18 -> 2.24 (2.76x, 2.72-2.79) | 5.68 -> 2.38 (2.38x, 2.33-2.54) | 5.79 -> 2.66 (2.18x, 2.13-2.25) |
| Q3.5-397B TP8 `in_proj_ba` | (16, 4096) | 4.75 -> 2.18 (2.18x, 2.15-2.32) | 7.26 -> 2.21 (3.29x, 2.94-3.41) | 7.07 -> 2.32 (3.03x, 2.97-3.11) | 6.62 -> 2.62 (2.52x, 2.49-2.60) |
| Q3.5-397B shared-expert gate | (1, 4096) | 4.40 -> 1.98 (2.20x, 2.14-2.33) | 4.90 -> 1.97 (2.44x, 2.42-2.57) | 4.90 -> 2.16 (2.27x, 2.19-2.34) | 4.64 -> 2.56 (1.81x, 1.77-1.90) |
| Q3.5-397B router | (512, 4096) | 7.33 -> 3.33 (2.20x, 2.17-2.22) | 7.23 -> 3.36 (2.15x, 2.10-2.17) | 7.17 -> 3.63 (1.98x, 1.97-1.98) | 7.10 -> 4.78 (1.50x, 1.47-1.51) |
| Q3.5-397B TP4 LM head | (62080, 4096) | 98.32 -> 72.88 (1.35x, 1.34-1.35) | 99.76 -> 75.02 (1.33x, 1.32-1.34) | 100.03 -> 85.12 (1.18x, 1.17-1.18) | cuBLAS (no entry) |
| Q3.5-397B TP8 LM head | (31040, 4096) | 55.36 -> 38.90 (1.42x, 1.41-1.44) | 56.21 -> 40.06 (1.40x, 1.39-1.41) | 55.94 -> 46.06 (1.21x, 1.20-1.23) | cuBLAS (no entry) |

- Spread (max - min over seeds, relative to the median speedup): at most 2.0 % on the LM heads, at most 19.8 % on the small shapes. The small-shape kernels take about 2 us, where one 32 ns timer step is 1.5 %; the worst cell, (1, 2048) M=8, has one seed at 2.49x against 2.07-2.12x for the others.
- Error vs an fp32 reference (max abs error / max abs reference): at most 3.8e-03 for the plan, and **equal to the cuBLAS figure in all 196 records**.
- From the sweep, against cuBLAS BF16 output directly: the small shapes are bit-identical (max abs diff 0, or 1e-4 on the 397B router); the LM heads differ by at most 0.0156 absolute.

### Proposed `SM100_SKINNY_GEMM_PLANS` entries (same content as `verified.json`)
```python
# Q3.6 TP1 in_proj_ba
(64, 2048): {
    1: SkinnyGemmConfig(1, 128, 2, static_k=2048),
    2: SkinnyGemmConfig(2, 128, 2, static_k=2048),
    4: SkinnyGemmConfig(4, 256, 1, vector_width=4, static_k=2048),
    8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2048),
},
# Q3.6 TP2 in_proj_ba
(32, 2048): {
    1: SkinnyGemmConfig(1, 128, 1, k_unroll=2, vector_width=4, static_k=2048),
    2: SkinnyGemmConfig(2, 128, 1, k_unroll=2, vector_width=4, static_k=2048),
    4: SkinnyGemmConfig(4, 128, 1, k_unroll=2, static_k=2048),
    8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, static_k=2048),
},
# Q3.6 TP4 in_proj_ba
(16, 2048): {
    1: SkinnyGemmConfig(1, 128, 2, vector_width=4, static_k=2048),
    2: SkinnyGemmConfig(2, 128, 1, vector_width=4, static_k=2048),
    4: SkinnyGemmConfig(4, 128, 1, vector_width=4, static_k=2048),
    8: SkinnyGemmConfig(8, 128, 1, static_k=2048),
},
# Q3.6 shared-expert gate (all TP)
(1, 2048): {
    1: SkinnyGemmConfig(1, 128, 1, k_unroll=2, vector_width=4, static_k=2048),
    2: SkinnyGemmConfig(2, 64, 1, static_k=2048),
    4: SkinnyGemmConfig(4, 128, 1, k_unroll=2, vector_width=4, static_k=2048),
    8: SkinnyGemmConfig(8, 128, 1, static_k=2048),
},
# Q3.6 router (all TP)
(256, 2048): {
    1: SkinnyGemmConfig(1, 64, 1, k_unroll=2, static_k=2048),
    2: SkinnyGemmConfig(2, 128, 1, static_k=2048),
    4: SkinnyGemmConfig(4, 128, 1, static_k=2048),
    8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2048),
},
# Q3.6 TP1 LM head
(248320, 2048): {
    1: SkinnyGemmConfig(1, 256, 4),
    2: SkinnyGemmConfig(2, 32, 2, k_unroll=4),
},
# Q3.6 TP2 LM head
(124160, 2048): {
    1: SkinnyGemmConfig(1, 256, 4, k_unroll=2, vector_width=4),
    2: SkinnyGemmConfig(2, 32, 2, k_unroll=4),
},
# Q3.6 TP4 LM head
(62080, 2048): {
    1: SkinnyGemmConfig(1, 256, 4, k_unroll=2, vector_width=4),
    2: SkinnyGemmConfig(2, 128, 4),
    4: SkinnyGemmConfig(4, 32, 4),
},
# Q3.5-397B TP4 in_proj_ba
(32, 4096): {
    1: SkinnyGemmConfig(1, 256, 1, vector_width=4, static_k=4096),
    2: SkinnyGemmConfig(2, 256, 1, k_unroll=2, static_k=4096),
    4: SkinnyGemmConfig(4, 256, 1, k_unroll=2, static_k=4096),
    8: SkinnyGemmConfig(8, 256, 1, k_unroll=2, static_k=4096),
},
# Q3.5-397B TP8 in_proj_ba
(16, 4096): {
    1: SkinnyGemmConfig(1, 128, 1, static_k=4096),
    2: SkinnyGemmConfig(2, 128, 1, static_k=4096),
    4: SkinnyGemmConfig(4, 256, 1, static_k=4096),
    8: SkinnyGemmConfig(8, 256, 1, k_unroll=2, static_k=4096),
},
# Q3.5-397B shared-expert gate
(1, 4096): {
    1: SkinnyGemmConfig(1, 128, 1, vector_width=4, static_k=4096),
    2: SkinnyGemmConfig(2, 256, 1, k_unroll=4, static_k=4096),
    4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, static_k=4096),
    8: SkinnyGemmConfig(8, 128, 1, k_unroll=2, static_k=4096),
},
# Q3.5-397B router
(512, 4096): {
    1: SkinnyGemmConfig(1, 128, 1, static_k=4096),
    2: SkinnyGemmConfig(2, 256, 1, static_k=4096),
    4: SkinnyGemmConfig(4, 128, 1, static_k=4096),
    8: SkinnyGemmConfig(8, 128, 2, k_unroll=2, static_k=4096),
},
# Q3.5-397B TP4 LM head
(62080, 4096): {
    1: SkinnyGemmConfig(1, 256, 2, k_unroll=2),
    2: SkinnyGemmConfig(2, 128, 4, k_unroll=4),
    4: SkinnyGemmConfig(4, 32, 2, k_unroll=4),
},
# Q3.5-397B TP8 LM head
(31040, 4096): {
    1: SkinnyGemmConfig(1, 256, 2, k_unroll=4, vector_width=4),
    2: SkinnyGemmConfig(2, 128, 4, k_unroll=4),
    4: SkinnyGemmConfig(4, 32, 2, k_unroll=4, vector_width=4),
},
```

### Shapes and cells that should NOT get entries
- Every LM head at M = 8 (0.71-0.94x in the sweep).
- (124160, 2048) and (248320, 2048) at M = 4 (1.07x and 1.04x in the sweep).
- `linear_attn.conv1d` is unquantized but is not a 2-D GEMM.

## 4. N = 1 and #64982
- #64982 changes two things in `layers/utils.py`: it caches the plan on the layer, and in eager mode it calls the kernel directly instead of going through `torch.ops.vllm.cuda_skinny_unquantized_gemm`. Its stated reason is host time (about 5 us per custom-op call at every token count), not reachability.
- At 8399da3780 nothing in the path is specific to N: `maybe_use_skinny_gemm` only needs a 2-D BF16 weight, no bias and a plan; the op looks the plan up by `(N, K)` and M.
- **Check run:** injected an SM100 table in-process (`SKINNY_GEMM_PLANS_BY_CAPABILITY[(10, 0)]`, no file edited) with entries for (1, 2048), (64, 2048), (256, 2048) and (248320, 2048) at M = 1, 2, 4, 8, loaded Qwen3.6-35B-A3B-FP8 at TP1 with dummy weights, and logged the calls of `shape_dynamic_skinny_gemm`.
  - (1, 2048), (64, 2048) and (256, 2048) ran the skinny kernel at M = 1, 2, 4 and 8; the LM head at M = 1 and 2 (the only planned sizes it was called with).
  - The model ran to the end, and the 8 greedy tokens of two requests equal those of a run without the injected table.
  - The injected configs were a fixed valid one (`block_size=128, outputs_per_block=1, static_k=K`), not the proposed ones.
- The only N = 1 constraint is `outputs_per_block = 1`, which the sweep tool already enforces (`n % opb`).

## 5. Per-step estimate and duplicates
- Qwen3.6-35B TP1, one request: 30 `in_proj_ba` + 40 gate + 40 router + 1 LM head per step. With the verify medians that is 30 x 2.87 + 40 x 2.48 + 40 x 3.02 + 16 = **about 320 us of kernel time per decode step**. This is a sum of isolated cold-L2 timings, not a measurement of the step.
- Duplicates: no open upstream PR adds SM100 entries for these shapes. #64982 (open) is the SM90 counterpart for (1, 2048) and (64, 2048) only; it has no router or LM-head entry. #57848 (open) tunes the SM103 MoE router GEMM, a different path (`GateLinear` tiers); not read in detail.

## 6. Not done
- The optional decode profile is in section 7.
- No expert-parallel walk for the 397B model. The shapes above do not depend on it as far as the code shows (gate and router are replicated), but that was not run.

## 7. Optional: decode profile of Qwen3.6-35B-A3B-FP8, TP1, dummy weights (added 18:47Z)

`vllm bench forward --profile-dir`, cases `1*1@1024` and `8*1@1024`, `--language-model-only`, 3 profiled replays, zipf routing.
Kernel lists: `profile_d1.kernels.json`, `profile_d8.kernels.json`. Each step has 1,169 GPU activities.

**1 request:** unprofiled step median 3390 us; under the profiler, span 4642 us and summed kernel time 4822 us (the profiler inflates the times, so read the shares).

| # | kernel | us | % | calls |
|---|---|---|---|---|
| 1 | `cutlass_3x_gemm_fp8_blockwise` (FP8 linears) | 1383.3 | 28.7 | 160 |
| 2 | `per_token_group_quant_8bit_register_kernel` | 481.1 | 10.0 | 200 |
| 3 | TRT-LLM MoE `bmm_E4m3_E4m3E4m3` (FC2) | 384.2 | 8.0 | 40 |
| 4 | TRT-LLM MoE `bmm_Bfloat16_E4m3E4m3` (FC1) | 244.2 | 5.1 | 40 |
| 5 | FlashInfer fused add + RMSNorm | 186.2 | 3.9 | 80 |
| 6 | `activationDeepSeekKernel` | 185.1 | 3.8 | 40 |
| 7 | `splitKreduce_kernel` | 181.7 | 3.8 | 70 |
| 8 | `routingIndicesBlockKernel` | 168.4 | 3.5 | 40 |
| 9 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 160.4 | 3.3 | 40 |
| 10 | `nvjet_sm100_tst_192x8_64x8_2x1_v_bz_TNT` | 146.9 | 3.0 | 1 |
| 11 | `finalizeKernelSmallBatch` | 118.4 | 2.5 | 40 |
| 12 | torch elementwise BF16 mul | 112.2 | 2.3 | 40 |
| 13 | `dot_kernel` | 111.0 | 2.3 | 40 |
| 14 | `nvjet_sm100_tst_32x64_64x16_2x1_v_bz_splitK_TNN` | 109.9 | 2.3 | 30 |
| 15 | torch vectorized elementwise (sigmoid and others) | 108.7 | 2.3 | 82 |

**8 requests:** unprofiled step median 4769 us; under the profiler, span 4852 us and summed kernel time 6437 us (the profiler inflates the times, so read the shares).

| # | kernel | us | % | calls |
|---|---|---|---|---|
| 1 | `cutlass_3x_gemm_fp8_blockwise` (FP8 linears) | 1451.2 | 22.6 | 160 |
| 2 | TRT-LLM MoE `bmm_E4m3_E4m3E4m3` (FC2) | 903.4 | 14.1 | 40 |
| 3 | TRT-LLM MoE `bmm_Bfloat16_E4m3E4m3` (FC1) | 634.5 | 9.9 | 40 |
| 4 | `per_token_group_quant_8bit_register_kernel` | 562.9 | 8.8 | 200 |
| 5 | `splitKreduce_kernel` | 305.5 | 4.8 | 110 |
| 6 | `activationDeepSeekKernel` | 274.3 | 4.3 | 40 |
| 7 | FlashInfer fused add + RMSNorm | 228.8 | 3.6 | 80 |
| 8 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 222.5 | 3.5 | 30 |
| 9 | `routingIndicesDynBlockKernel` | 211.7 | 3.3 | 40 |
| 10 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 165.9 | 2.6 | 40 |
| 11 | `nvjet_sm100_tst_192x8_64x8_2x1_v_bz_TNT` | 149.7 | 2.3 | 1 |
| 12 | `nvjet_sm100_tst_8x64_64x16_1x1_h_bz_splitK_TNN` | 143.6 | 2.2 | 40 |
| 13 | `finalizeKernelSmallBatch` | 136.3 | 2.1 | 40 |
| 14 | torch vectorized elementwise (sigmoid and others) | 135.0 | 2.1 | 82 |
| 15 | torch elementwise BF16 mul | 129.7 | 2.0 | 40 |

What stands out (shares of summed kernel time under the profiler):
- **cuBLAS BF16 GEMMs are 15.9 % at 1 request (768 us in 221 launches) and 13.3 % at 8 (856 us).** These are exactly the W11a layers: router (`nvjet 32x64 4x1 splitK`, 40), `in_proj_ba` (`nvjet 32x64 2x1 splitK`, 30), shared-expert gate (`dot_kernel` + `reduce_1Block_kernel`, 40 + 40 at 1 request; `nvjet 8x64` at 8), their `splitKreduce_kernel` launches (70 at 1 request, 110 at 8) and the LM head (`nvjet 192x8`, 1). A skinny plan turns each GEMM into one launch.
- `per_token_group_quant_8bit_register_kernel`: 200 launches, 10.0 % and 8.8 %. At 1-8 tokens each launch is 2.4-2.8 us, so this is launch count (5 per layer), not the conversion cost W7c measured at thousands of tokens.
- Three kernels of similar size run 40 times each around the shared expert: the gate GEMM, a torch sigmoid and a torch BF16 multiply (112 us for the multiply alone at 1 request). I did not trace which module issues the sigmoid and the multiply, so the link to the shared-expert gate is a guess from the call counts.
- Not a GEMM, one launch each: torch softmax in the sampler (103-110 us) and the LM head (147-150 us).
