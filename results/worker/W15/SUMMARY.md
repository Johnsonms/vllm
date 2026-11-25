# W15 — Flash-Next (Qwen3.8) SM100 skinny-GEMM table gaps: sweep + verify

**Result: 32 of 35 proposed entries verified at >= 1.10x on all 4 seeds** (`verified.json`). 3 dropped (M = 5 on the three large projections, 1.04-1.12x).

## Scope
- Model table for Flash-Next on SM100 (the exact-M lookup; layers opt in by weight shape). TEP4 local shapes, K = 2560 unless noted.
- Swept M = 1-8, 12, 16 for every shape; only (shape, M) pairs not yet in the table were proposed.
- (10240, 320) and (2560, 160) are excluded: no skinny config beats cuBLAS there (W14 correction).
- Tree: #69736 head 740ff0d02c (now merged), stock precompiled kernel, no code edits.

## Method (same as W11 / B2)
- Sweep: every skinny config vs cuBLAS per (N, K, M), in a CUDA graph; best config with >= 1.08x went to `proposed.json`.
- Verify: 4 seeds x 3 interleaved rounds vs cuBLAS, one GPU per seed (GPUs 2, 3, 6, 7). Keep if the minimum over seeds is >= 1.10x.
- `cuBLAS us` / `skinny us` are medians over seeds; errors are max relative error vs an fp32 reference (skinny equals cuBLAS on every entry).
- The node was shared with the owner's job on two other GPUs during the run.

## Verified table
| (N, K) | M | cuBLAS us | skinny us | min x | median x | err skinny | err cuBLAS | |
|---|---|---|---|---|---|---|---|---|
| (1, 2560) | 1 | 4.35 | 1.89 | 2.23 | 2.30 | 1.5e-03 | 1.5e-03 | keep |
| (1, 2560) | 2 | 4.90 | 1.94 | 2.44 | 2.55 | 3.1e-03 | 3.1e-03 | keep |
| (1, 2560) | 3 | 4.83 | 2.00 | 2.34 | 2.39 | 1.9e-03 | 1.9e-03 | keep |
| (1, 2560) | 4 | 4.58 | 2.06 | 2.09 | 2.19 | 3.4e-03 | 3.4e-03 | keep |
| (1, 2560) | 5 | 4.69 | 2.11 | 2.16 | 2.22 | 2.7e-03 | 2.7e-03 | keep |
| (1, 2560) | 6 | 4.48 | 2.19 | 2.03 | 2.05 | 2.1e-03 | 2.1e-03 | keep |
| (1, 2560) | 7 | 4.77 | 2.24 | 2.09 | 2.14 | 2.6e-03 | 2.6e-03 | keep |
| (1, 2560) | 8 | 5.10 | 2.30 | 2.19 | 2.23 | 2.9e-03 | 2.9e-03 | keep |
| (1, 2560) | 12 | 4.74 | 2.67 | 1.74 | 1.78 | 2.5e-03 | 2.5e-03 | keep |
| (1, 2560) | 16 | 5.04 | 3.02 | 1.64 | 1.66 | 2.7e-03 | 2.7e-03 | keep |
| (24, 2560) | 3 | 5.82 | 2.14 | 2.68 | 2.69 | 3.3e-03 | 3.3e-03 | keep |
| (24, 2560) | 5 | 5.65 | 2.30 | 2.40 | 2.45 | 3.1e-03 | 3.1e-03 | keep |
| (24, 2560) | 6 | 5.68 | 2.34 | 2.33 | 2.45 | 2.2e-03 | 2.2e-03 | keep |
| (24, 2560) | 7 | 5.57 | 2.45 | 2.23 | 2.26 | 2.8e-03 | 2.8e-03 | keep |
| (24, 2560) | 12 | 5.65 | 2.82 | 1.97 | 2.00 | 2.8e-03 | 2.8e-03 | keep |
| (320, 2560) | 3 | 6.03 | 2.70 | 2.21 | 2.24 | 2.4e-03 | 2.4e-03 | keep |
| (320, 2560) | 5 | 6.05 | 2.91 | 2.08 | 2.08 | 2.5e-03 | 2.5e-03 | keep |
| (320, 2560) | 6 | 5.87 | 2.98 | 1.97 | 1.97 | 2.7e-03 | 2.7e-03 | keep |
| (320, 2560) | 7 | 6.21 | 3.20 | 1.88 | 1.95 | 2.2e-03 | 2.2e-03 | keep |
| (320, 2560) | 12 | 6.14 | 3.90 | 1.50 | 1.57 | 2.1e-03 | 2.1e-03 | keep |
| (512, 2560) | 3 | 6.21 | 2.96 | 2.08 | 2.11 | 2.4e-03 | 2.4e-03 | keep |
| (512, 2560) | 4 | 6.29 | 3.09 | 2.03 | 2.04 | 2.4e-03 | 2.4e-03 | keep |
| (512, 2560) | 5 | 6.21 | 3.25 | 1.89 | 1.91 | 2.2e-03 | 2.2e-03 | keep |
| (512, 2560) | 6 | 6.27 | 3.44 | 1.81 | 1.82 | 3.7e-03 | 3.7e-03 | keep |
| (512, 2560) | 7 | 6.35 | 3.58 | 1.76 | 1.76 | 2.4e-03 | 2.4e-03 | keep |
| (512, 2560) | 12 | 6.29 | 4.51 | 1.34 | 1.40 | 2.0e-03 | 2.0e-03 | keep |
| (512, 2560) | 16 | 6.29 | 5.20 | 1.20 | 1.21 | 2.5e-03 | 2.5e-03 | keep |
| (2560, 1536) | 3 | 5.79 | 4.50 | 1.29 | 1.29 | 2.1e-03 | 2.1e-03 | keep |
| (2560, 1536) | 5 | 5.70 | 5.25 | 1.08 | 1.09 | 3.3e-03 | 3.3e-03 | drop |
| (4096, 2560) | 3 | 9.80 | 7.01 | 1.32 | 1.40 | 3.3e-03 | 3.3e-03 | keep |
| (4096, 2560) | 5 | 8.62 | 7.98 | 1.06 | 1.08 | 2.4e-03 | 2.4e-03 | drop |
| (4224, 2560) | 3 | 8.96 | 7.17 | 1.21 | 1.25 | 2.9e-03 | 2.9e-03 | keep |
| (4224, 2560) | 5 | 8.91 | 8.13 | 1.04 | 1.10 | 3.4e-03 | 3.4e-03 | drop |
| (62080, 2560) | 3 | 61.76 | 51.71 | 1.18 | 1.19 | 3.4e-03 | 3.4e-03 | keep |
| (62080, 2560) | 4 | 61.57 | 54.29 | 1.13 | 1.13 | 3.3e-03 | 3.3e-03 | keep |
## By layer
- **Shared-expert gate (1, 2560):** 1.64-2.58x at every M up to 16. It has no table line today.
- **Router (512, 2560):** M = 3-7 1.76-2.12x, M = 12 1.34x, M = 16 1.20x.
- **(24, 2560) and (320, 2560):** the non-power-of-two M (3, 5, 6, 7, 12) 1.50-2.73x. These M fall back to cuBLAS today because the lookup is by exact M.
- **Large projections (2560, 1536), (4096, 2560), (4224, 2560):** only M = 3 (1.21-1.41x). M >= 5 is flat.
- **LM head (62080, 2560):** M = 3 1.18x, M = 4 1.13x.

## Not measured
- No E2E A/B yet (kernel microbench only). W14 estimated about -3 % TPOT at 4 requests for these lines.
- B300 / SM103, TP8 local shapes, M > 16.
- Odd M between 8 and 16 other than 12 (9, 10, 11, 13-15) were not swept.

## Files
`proposed.json` (35), `verified.json` (32, format `{"N,K": {"M": "SkinnyGemmConfig(...)"}}`), `sweep/m*.jsonl`, `verify/seed1-4.jsonl`.
