# B2 — SM103 generic skinny plans for Qwen3.5 / Qwen3.6 MoE (port of #69736 to B300)

Node: 8x B300 (SM103), vLLM `e66bede56b` unmodified, no vLLM file edited. All 8 GPUs were idle of other users for the sweep and the verify.
Status: **done (sweep and verify).** E2E has not been run; that waits for the manager's `(10, 3)` table.

## Result
- **49 entries on 14 shapes pass the verify (median ≥ 1.10× in 4 seeds × 3 interleaved rounds).** `verified.json` equals `proposed.json`. These are **the same 49 cells as W11 on SM100**.
- Small projections (GDN `in_proj_ba`, shared-expert gate, router): **1.50–3.12×**, from 4.2–7.0 us down to 1.8–4.5 us.
- LM heads: **1.13–1.46×** at M = 1–2, and at M = 4 for the three shards of 62,080 rows or fewer.
- No entry for any LM head at M = 8 (sweep 0.75–0.96×), or for (248320, 2048) / (124160, 2048) at M = 4 (sweep 1.04× / 1.07×).
- **Weakest entries:** (248320, 2048) M=1 at 1.13× (min 1.12), and (62080, 2048) M=4 at 1.13× (min 1.13).
- **Error vs an fp32 reference** (max abs error / max abs reference): at most 3.8e-03, and **equal to the cuBLAS figure in all 196 records**.

### Shared table? The W11 SM100 configs also qualify on SM103.
- Only 8 of 49 SM103 winners are literally the SM100 config. But **every one of the 49 W11 SM100 configs passes the same ≥ 1.10× verify on SM103.**
- Each config was timed in the same interleaved rounds as the SM103 winner. The ratio sm100_us / b300_us has a median of **1.000** and a range of **0.973–1.049**:
  - Worst for SM100: (124160, 2048) M=1 is 4.9% slower than the SM103 winner, and (62080, 2048) M=1 is 3.4% slower.
  - In 12 cells the SM100 config is slightly faster (down to 0.973), and 14 cells are equal within 0.05%. This is within the noise of the sweep's single run.
- **The thinnest SM100 margin on SM103:** (62080, 2048) M=4 at 1.11× (min 1.10). Next is (248320, 2048) M=1 at 1.12×.
- **Reading:** registering `(10, 3)` to the same table as `(10, 0)` is defensible. It would give up at most ~5% of skinny-kernel time on two LM-head cells, and ≤ 2.4% elsewhere.
  - A separate SM103 table buys about 1–5% on a handful of LM-head cells. Since the LM head runs once per step, that is ≤ ~4 us of a 3,000 us step.
  - The manager decides. Both variants are listed below.

## Method
- **Shapes:** the 14 keys of `results/worker/W11/verified.json` (`shapes_small.json`, `shapes_lmhead.json`).
- **Sweep:** `results/manager/tools/sweep_skinny_generic.py`, unchanged; raw output in `sweep.jsonl`.
  - Run as 8 processes, one per GPU: {small, LM-head} × M ∈ {1, 2, 4, 8}. 180-config grid, CUPTI, CUDA graph, cold L2.
  - Keep rule: cuBLAS / best ≥ 1.10.
  - The real-weight download (network/disk only) overlapped the sweep. The sweep only nominates configs; the verify decides.
- **Verify:** `verify_sm103.py`, the manager's `verify_skinny_reference.py` generalized to three arms per cell: cuBLAS, the SM103 proposal and the W11 SM100 config.
  - Each seed runs on its own idle GPU (GPUs 0-3, seeds 0-3). Each cell does 3 rounds, with the arm order rotating each round. Per arm, the middle of the 3 rounds is kept. CUPTI, CUDA graph, cold L2, `repeat_time_ms=100`.
  - Raw: `verify.jsonl`; per-cell summary: `verify_summary.json`.
- Baseline is cuBLAS: there is no SM103 entry for any of these shapes on main.

## Sweep (one run per cell; times in us, cuBLAS -> best skinny config)

| (N, K) | M=1 | M=2 | M=4 | M=8 |
|---|---|---|---|---|
| (64, 2048) | 4.80 -> 2.05 (2.34x) | 5.06 -> 2.08 (2.43x) | 4.80 -> 2.18 (2.21x) | 5.31 -> 2.37 (2.24x) |
| (32, 2048) | 4.58 -> 1.98 (2.31x) | 5.02 -> 2.05 (2.45x) | 4.93 -> 2.11 (2.33x) | 5.12 -> 2.27 (2.25x) |
| (16, 2048) | 4.67 -> 1.89 (2.47x) | 4.99 -> 1.98 (2.52x) | 4.96 -> 2.02 (2.46x) | 5.18 -> 2.21 (2.35x) |
| (1, 2048) | 3.94 -> 1.76 (2.24x) | 4.16 -> 1.82 (2.28x) | 4.10 -> 1.86 (2.21x) | 4.77 -> 2.11 (2.26x) |
| (256, 2048) | 5.12 -> 2.30 (2.22x) | 5.31 -> 2.37 (2.24x) | 5.28 -> 2.43 (2.17x) | 5.60 -> 2.69 (2.08x) |
| (248320, 2048) | 165.67 -> 147.20 (1.13x) | 166.08 -> 144.77 (1.15x) | 166.37 -> 159.74 (1.04x, drop) | 168.03 -> 223.68 (0.75x, drop) |
| (124160, 2048) | 90.66 -> 74.88 (1.21x) | 90.72 -> 74.94 (1.21x) | 90.78 -> 84.48 (1.07x, drop) | 91.36 -> 116.19 (0.79x, drop) |
| (62080, 2048) | 52.03 -> 39.62 (1.31x) | 52.38 -> 39.52 (1.33x) | 51.55 -> 45.60 (1.13x) | 50.72 -> 62.34 (0.81x, drop) |
| (32, 4096) | 4.64 -> 2.05 (2.27x) | 5.54 -> 2.11 (2.62x) | 5.47 -> 2.24 (2.44x) | 5.47 -> 2.53 (2.16x) |
| (16, 4096) | 4.45 -> 2.02 (2.21x) | 6.82 -> 2.08 (3.28x) | 6.11 -> 2.21 (2.77x) | 6.75 -> 2.43 (2.78x) |
| (1, 4096) | 4.00 -> 1.86 (2.16x) | 4.19 -> 1.92 (2.18x) | 4.16 -> 2.02 (2.06x) | 4.70 -> 2.40 (1.96x) |
| (512, 4096) | 6.75 -> 3.20 (2.11x) | 7.07 -> 3.26 (2.17x) | 7.01 -> 3.55 (1.97x) | 6.94 -> 4.45 (1.56x) |
| (62080, 4096) | 100.54 -> 75.62 (1.33x) | 101.03 -> 76.13 (1.33x) | 101.25 -> 85.92 (1.18x) | 91.14 -> 111.94 (0.81x, drop) |
| (31040, 4096) | 56.99 -> 39.65 (1.44x) | 57.09 -> 40.45 (1.41x) | 56.96 -> 46.21 (1.23x) | 58.08 -> 60.29 (0.96x, drop) |

### Verify: SM103 proposal vs cuBLAS

Cells show median cuBLAS -> median plan in us (median speedup, min-max over the 4 seeds).

| layer | (N, K) | M=1 | M=2 | M=4 | M=8 |
|---|---|---|---|---|---|
| Q3.6 TP1 `in_proj_ba` | (64, 2048) | 4.85 -> 2.05 (2.37x, 2.33-2.41) | 5.06 -> 2.08 (2.41x, 2.40-2.46) | 4.99 -> 2.14 (2.33x, 2.27-2.43) | 5.14 -> 2.37 (2.18x, 2.04-2.22) |
| Q3.6 TP2 `in_proj_ba` | (32, 2048) | 4.74 -> 1.97 (2.41x, 2.35-2.44) | 4.86 -> 2.05 (2.38x, 2.38-2.41) | 5.01 -> 2.10 (2.39x, 2.37-2.45) | 5.12 -> 2.30 (2.22x, 2.17-2.28) |
| Q3.6 TP4 `in_proj_ba` | (16, 2048) | 4.58 -> 1.95 (2.34x, 2.33-2.37) | 4.85 -> 1.92 (2.52x, 2.40-2.58) | 4.77 -> 2.06 (2.30x, 2.29-2.33) | 4.62 -> 2.26 (2.06x, 1.99-2.07) |
| Q3.6 shared-expert gate (all TP) | (1, 2048) | 4.34 -> 1.79 (2.43x, 2.25-2.46) | 4.48 -> 1.87 (2.42x, 2.37-2.48) | 4.51 -> 2.00 (2.20x, 2.17-2.44) | 4.58 -> 2.14 (2.13x, 2.10-2.14) |
| Q3.6 router (all TP) | (256, 2048) | 5.36 -> 2.34 (2.29x, 2.18-2.30) | 5.25 -> 2.43 (2.15x, 2.13-2.28) | 5.28 -> 2.46 (2.14x, 2.08-2.18) | 5.38 -> 2.80 (1.92x, 1.91-1.98) |
| Q3.6 TP1 LM head | (248320, 2048) | 165.73 -> 147.14 (1.13x, 1.12-1.13) | 165.95 -> 144.53 (1.15x, 1.15-1.15) | cuBLAS (no entry) | cuBLAS (no entry) |
| Q3.6 TP2 LM head | (124160, 2048) | 90.58 -> 75.06 (1.21x, 1.16-1.21) | 90.67 -> 75.12 (1.21x, 1.21-1.21) | cuBLAS (no entry) | cuBLAS (no entry) |
| Q3.6 TP4 LM head | (62080, 2048) | 51.87 -> 39.55 (1.31x, 1.31-1.31) | 51.79 -> 39.66 (1.31x, 1.30-1.31) | 51.73 -> 45.66 (1.13x, 1.13-1.14) | cuBLAS (no entry) |
| Q3.5-397B TP4 `in_proj_ba` | (32, 4096) | 4.67 -> 2.10 (2.23x, 2.11-2.25) | 5.36 -> 2.18 (2.46x, 2.41-2.48) | 5.39 -> 2.24 (2.41x, 2.34-2.50) | 5.41 -> 2.59 (2.09x, 2.07-2.15) |
| Q3.5-397B TP8 `in_proj_ba` | (16, 4096) | 4.45 -> 2.08 (2.14x, 2.11-2.20) | 6.46 -> 2.06 (3.12x, 2.89-3.33) | 6.26 -> 2.24 (2.79x, 2.76-2.99) | 6.50 -> 2.54 (2.55x, 2.45-2.78) |
| Q3.5-397B shared-expert gate | (1, 4096) | 4.21 -> 1.87 (2.26x, 2.24-2.37) | 4.51 -> 1.95 (2.31x, 2.26-2.46) | 4.93 -> 2.11 (2.33x, 2.25-2.35) | 4.50 -> 2.53 (1.79x, 1.76-1.80) |
| Q3.5-397B router | (512, 4096) | 7.01 -> 3.20 (2.19x, 2.15-2.23) | 6.82 -> 3.26 (2.10x, 2.08-2.12) | 6.86 -> 3.68 (1.86x, 1.85-1.96) | 6.85 -> 4.54 (1.50x, 1.50-1.52) |
| Q3.5-397B TP4 LM head | (62080, 4096) | 100.59 -> 75.65 (1.33x, 1.31-1.34) | 100.16 -> 76.24 (1.31x, 1.31-1.33) | 101.20 -> 86.11 (1.17x, 1.17-1.18) | cuBLAS (no entry) |
| Q3.5-397B TP8 LM head | (31040, 4096) | 57.52 -> 39.63 (1.45x, 1.43-1.46) | 58.70 -> 40.14 (1.46x, 1.45-1.47) | 57.18 -> 46.78 (1.22x, 1.22-1.25) | cuBLAS (no entry) |

### W11 SM100 config on SM103

Cells show the median speedup over cuBLAS with the min-max over 4 seeds, followed by sm100_us / b300_us (above 1 means the SM103 config is faster). **bold** marks cells where the SM100 config equals the SM103 winner.

| layer | (N, K) | M=1 | M=2 | M=4 | M=8 |
|---|---|---|---|---|---|
| Q3.6 TP1 `in_proj_ba` | (64, 2048) | 2.35x (2.29-2.41); 1.005 | 2.38x (2.33-2.42); 1.024 | 2.31x (2.27-2.43); 1.009 | 2.15x (2.04-2.20); 1.000 |
| Q3.6 TP2 `in_proj_ba` | (32, 2048) | 2.40x (2.39-2.41); 1.000 | 2.41x (2.41-2.46); 0.985 | 2.37x (2.36-2.41); 1.005 | 2.22x (2.17-2.25); 1.000 |
| Q3.6 TP4 `in_proj_ba` | (16, 2048) | 2.40x (2.34-2.48); 0.985 | 2.50x (2.43-2.58); 1.000 | 2.26x (2.23-2.27); 1.024 | 2.06x (2.04-2.12); 0.982 |
| Q3.6 shared-expert gate (all TP) | (1, 2048) | **2.43x (2.25-2.46); 1.000** | 2.38x (2.37-2.42); 1.011 | 2.31x (2.06-2.40); 0.985 | 2.13x (2.07-2.19); 1.009 |
| Q3.6 router (all TP) | (256, 2048) | 2.29x (2.18-2.30); 1.000 | 2.21x (2.16-2.28); 0.975 | 2.12x (2.03-2.18); 1.016 | 1.91x (1.91-1.98); 1.007 |
| Q3.6 TP1 LM head | (248320, 2048) | 1.12x (1.12-1.12); 1.004 | **1.15x (1.15-1.15); 1.000** | no entry | no entry |
| Q3.6 TP2 LM head | (124160, 2048) | 1.15x (1.15-1.15); 1.049 | **1.21x (1.21-1.21); 0.999** | no entry | no entry |
| Q3.6 TP4 LM head | (62080, 2048) | 1.27x (1.27-1.27); 1.034 | 1.28x (1.28-1.29); 1.018 | 1.11x (1.10-1.11); 1.022 | no entry |
| Q3.5-397B TP4 `in_proj_ba` | (32, 4096) | **2.21x (2.11-2.25); 1.005** | 2.50x (2.48-2.52); 0.982 | 2.41x (2.34-2.50); 1.000 | 2.10x (2.10-2.15); 0.996 |
| Q3.5-397B TP8 `in_proj_ba` | (16, 4096) | 2.14x (2.08-2.20); 1.000 | 3.09x (2.89-3.28); 1.010 | 2.88x (2.84-3.03); 0.973 | 2.54x (2.45-2.75); 1.008 |
| Q3.5-397B shared-expert gate | (1, 4096) | 2.25x (2.22-2.33); 1.011 | 2.33x (2.30-2.46); 0.985 | 2.32x (2.25-2.37); 0.995 | 1.80x (1.78-1.82); 0.988 |
| Q3.5-397B router | (512, 4096) | 2.15x (2.13-2.17); 1.019 | 2.07x (2.04-2.10); 1.021 | **1.87x (1.85-1.96); 1.000** | 1.50x (1.48-1.51); 1.000 |
| Q3.5-397B TP4 LM head | (62080, 4096) | 1.31x (1.29-1.31); 1.017 | **1.31x (1.31-1.33); 1.000** | **1.17x (1.17-1.18); 1.000** | no entry |
| Q3.5-397B TP8 LM head | (31040, 4096) | 1.43x (1.41-1.44); 1.016 | **1.46x (1.45-1.47); 1.000** | 1.22x (1.22-1.23); 1.005 | no entry |

### Proposed SM103 entries (same content as `verified.json`)

```python
# Q3.6 TP1 in_proj_ba
(64, 2048): {
    1: SkinnyGemmConfig(1, 64, 1, vector_width=4, static_k=2048),
    2: SkinnyGemmConfig(2, 256, 1, vector_width=4, static_k=2048),
    4: SkinnyGemmConfig(4, 256, 1, k_unroll=2, vector_width=4, static_k=2048),
    8: SkinnyGemmConfig(8, 128, 1, k_unroll=2, static_k=2048),
},
# Q3.6 TP2 in_proj_ba
(32, 2048): {
    1: SkinnyGemmConfig(1, 128, 1, vector_width=4, static_k=2048),
    2: SkinnyGemmConfig(2, 64, 1, static_k=2048),
    4: SkinnyGemmConfig(4, 256, 1, vector_width=4, static_k=2048),
    8: SkinnyGemmConfig(8, 128, 1, static_k=2048),
},
# Q3.6 TP4 in_proj_ba
(16, 2048): {
    1: SkinnyGemmConfig(1, 64, 2, k_unroll=4, static_k=2048),
    2: SkinnyGemmConfig(2, 256, 1, k_unroll=2, vector_width=4, static_k=2048),
    4: SkinnyGemmConfig(4, 256, 1, vector_width=4, static_k=2048),
    8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2048),
},
# Q3.6 shared-expert gate (all TP)
(1, 2048): {
    1: SkinnyGemmConfig(1, 128, 1, k_unroll=2, vector_width=4, static_k=2048),
    2: SkinnyGemmConfig(2, 128, 1, k_unroll=2, static_k=2048),
    4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, static_k=2048),
    8: SkinnyGemmConfig(8, 128, 1, k_unroll=2, vector_width=4, static_k=2048),
},
# Q3.6 router (all TP)
(256, 2048): {
    1: SkinnyGemmConfig(1, 64, 1, k_unroll=2, vector_width=4, static_k=2048),
    2: SkinnyGemmConfig(2, 64, 1, k_unroll=2, vector_width=4, static_k=2048),
    4: SkinnyGemmConfig(4, 256, 1, vector_width=4, static_k=2048),
    8: SkinnyGemmConfig(8, 128, 1, k_unroll=2, vector_width=4, static_k=2048),
},
# Q3.6 TP1 LM head
(248320, 2048): {
    1: SkinnyGemmConfig(1, 128, 2, k_unroll=2),
    2: SkinnyGemmConfig(2, 32, 2, k_unroll=4),
},
# Q3.6 TP2 LM head
(124160, 2048): {
    1: SkinnyGemmConfig(1, 128, 2),
    2: SkinnyGemmConfig(2, 32, 2, k_unroll=4),
},
# Q3.6 TP4 LM head
(62080, 2048): {
    1: SkinnyGemmConfig(1, 128, 2, k_unroll=2),
    2: SkinnyGemmConfig(2, 64, 4, k_unroll=4, vector_width=4),
    4: SkinnyGemmConfig(4, 32, 2, k_unroll=4, vector_width=4),
},
# Q3.5-397B TP4 in_proj_ba
(32, 4096): {
    1: SkinnyGemmConfig(1, 256, 1, vector_width=4, static_k=4096),
    2: SkinnyGemmConfig(2, 128, 1, static_k=4096),
    4: SkinnyGemmConfig(4, 256, 1, static_k=4096),
    8: SkinnyGemmConfig(8, 128, 1, static_k=4096),
},
# Q3.5-397B TP8 in_proj_ba
(16, 4096): {
    1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4, static_k=4096),
    2: SkinnyGemmConfig(2, 256, 1, static_k=4096),
    4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=4, static_k=4096),
    8: SkinnyGemmConfig(8, 256, 1, k_unroll=4, static_k=4096),
},
# Q3.5-397B shared-expert gate
(1, 4096): {
    1: SkinnyGemmConfig(1, 128, 1, static_k=4096),
    2: SkinnyGemmConfig(2, 256, 1, static_k=4096),
    4: SkinnyGemmConfig(4, 128, 1, static_k=4096),
    8: SkinnyGemmConfig(8, 128, 1, static_k=4096),
},
# Q3.5-397B router
(512, 4096): {
    1: SkinnyGemmConfig(1, 256, 2, k_unroll=4, vector_width=4, static_k=4096),
    2: SkinnyGemmConfig(2, 256, 2, vector_width=4, static_k=4096),
    4: SkinnyGemmConfig(4, 128, 1, static_k=4096),
    8: SkinnyGemmConfig(8, 256, 1),
},
# Q3.5-397B TP4 LM head
(62080, 4096): {
    1: SkinnyGemmConfig(1, 128, 1, k_unroll=4),
    2: SkinnyGemmConfig(2, 128, 4, k_unroll=4),
    4: SkinnyGemmConfig(4, 32, 2, k_unroll=4),
},
# Q3.5-397B TP8 LM head
(31040, 4096): {
    1: SkinnyGemmConfig(1, 128, 1, k_unroll=4),
    2: SkinnyGemmConfig(2, 128, 4, k_unroll=4),
    4: SkinnyGemmConfig(4, 64, 2, k_unroll=4, vector_width=4),
},
```

## Files
- `sweep.jsonl`: raw sweep records.
- `proposed.json`: the sweep winners.
- `verified.json`: the verified entries (equal to `proposed.json`).
- `verify.jsonl`: raw verify records, all arms and rounds.
- `verify_summary.json`: per-cell medians, min/max, keep flags and both errors.
- `verify_sm103.py`: the verify script.
- `shapes_*.json`: the shape lists.

## Next
- Waiting for the manager's review. Real weights for Qwen3.6-35B-A3B-FP8 are already downloaded (42/42 files, 35 GB), ready for the E2E A/B.
