# W11a — SM100 generic skinny plans for Qwen3.5 / Qwen3.6 MoE (FP8 checkpoints)

Node: 8x B200, vLLM main @ 8399da3780 (precompiled), no vLLM file edited.
Status: **sweep posted (18:40Z); verify running.** This file is rewritten when the verify ends.

## 1. Unquantized layers (module walk, `load_format=dummy`)

Hooked `UnquantizedLinearMethod` and `UnquantizedEmbeddingMethod` (`process_weights_after_loading` and `apply`) in-process and
ran the model once. Local weight shapes per rank, all BF16, no bias, `use_skinny_gemm=False` at base:

| layer | class | count | Qwen3.6-35B TP1 | TP2 | TP4 | Qwen3.5-397B TP4 | TP8 |
|---|---|---|---|---|---|---|---|
| GDN `linear_attn.in_proj_ba` | MergedColumnParallelLinear | 30 / 45 | (64, 2048) | (32, 2048) | (16, 2048) | (32, 4096) | (16, 4096) derived |
| `mlp.shared_expert_gate` | ReplicatedLinear | 40 / 60 | (1, 2048) | same | same | (1, 4096) | same |
| `mlp.gate` (router) | GateLinear | 40 / 60 | (256, 2048) | same | same | (512, 4096) | same |
| `lm_head` | ParallelLMHead | 1 | (248320, 2048) | (124160, 2048) | (62080, 2048) | (62080, 4096) | (31040, 4096) derived |

- Also unquantized but not a GEMM: `linear_attn.conv1d` (3-D weight).
- The router is a `GateLinear`, but for this model its `apply` goes through `UnquantizedLinearMethod.apply` with a BF16 `(M, K)`
  input at every M (seen in the walk), i.e. the plain `F.linear` tier. So a generic plan entry takes effect for the router too.
- The LM head goes through `UnquantizedEmbeddingMethod.apply`, which also calls `maybe_use_skinny_gemm`.
- None of the 14 shapes is in `results/manager/W11_existing_shapes.json`.
- The 397B TP8 shapes are derived from the config for now; a TP8 walk follows.

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
  is partly noise; the verify decides.
- The Qwen3.6 sweep (GPUs 0-3) overlapped the 397B module walk on GPUs 4-7 for its first minutes. The verify runs on an idle node.

## 3. Still to come
Verify (4 seeds, 3 interleaved rounds), the N=1 reachability check for #64982, the 397B TP8 walk, duplicate search.
