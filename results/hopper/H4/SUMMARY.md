# H4 — QuACK W4A16 (`gemm_w4a16`, int4awq) vs vLLM's int4 kernel on H100 (2026-11-28)

## 1. What vLLM runs today (H100, vLLM `cfbc42099a`)
- `Qwen/Qwen3-8B-AWQ` and `Qwen/Qwen3-32B-AWQ` (AWQ, 4-bit, group 128, zero points, checkpoint dtype FP16): **`MacheteLinearKernel`** via `AutoAWQMarlinLinearMethod`, in both FP16 (default) and `--dtype bfloat16`:
  `Using MacheteLinearKernel for AutoAWQMarlinLinearMethod`.
- GPTQ-Int4: no ungated `Qwen/Qwen3-{8B,32B}-GPTQ-Int4` repo exists, so it's not covered.
- Share of GPU kernel time taken by Machete (torch profiler, 32 engine steps after 24 skipped; `time_share.txt`). BF16 is within 0.6 pt of FP16:

| model | TPOT c1 (fp16 / bf16) | Machete share c1 | c32 | prefill 4096/1 |
|---|---|---:|---:|---:|
| Qwen3-8B-AWQ | 4.95 / 4.94 ms | 71.7% | 56.5% | 86.5% |
| Qwen3-32B-AWQ | 12.88 / 12.79 ms | 84.3% | 73.2% | 89.1% |

## 2. Microbench: QuACK `gemm_w4a16(wformat="int4awq")` vs `ops.machete_mm` (uint4 + zero points, g128)
CUPTI, cold L2, CUDA graph; both kernels get the same synthetic quantized weights. Machete is timed in BF16 and FP16 (`mb_h4.txt`); they're within ~2% of each other, so FP16 is shown. "auto" = QuACK's built-in `pick_w4_cfg`. "tuned" = best of a tile_m × tile_n × split_k sweep (`sweep_w4.txt`, ≤19 configs per point, 1-round median).

**Qwen3-8B shapes** — Machete FP16 µs / QuACK auto µs / QuACK tuned µs (tuned speedup vs Machete FP16)

| layer (K x N) | M=1 | M=32 | M=512 | M=2048 |
|---|---|---|---|---|
| 8b.qkv (4096x6144) | 17.8 / 15.5 / 11.7 (1.52x) | 19.0 / 19.0 / 11.7 (1.62x) | 43.8 / 42.1 / 40.1 (1.09x) | 134.5 / 132.3 / 130.7 (1.03x) |
| 8b.o (4096x4096) | 16.7 / 10.2 / 10.1 (1.64x) | 17.5 / 11.9 / 10.5 (1.67x) | 29.1 / 41.2 / 23.7 (1.23x) | 109.2 / 87.5 / 87.5 (1.25x) |
| 8b.gate_up (4096x24576) | 43.8 / 33.5 / 29.9 (1.46x) | 34.2 / 34.6 / 31.0 (1.10x) | 134.5 / 130.7 / 129.7 (1.04x) | 550.3 / 531.5 / 512.6 (1.07x) |
| 8b.down (12288x4096) | 27.6 / 20.4 / 18.8 (1.47x) | 28.6 / 21.1 / 20.0 (1.43x) | 79.9 / 114.8 / 65.3 (1.22x) | 320.4 / 259.9 / 254.5 (1.26x) |
| **sum per layer** | 106 / 80 / 71 (**1.50x**, auto 1.33x) | 99 / 87 / 73 (**1.36x**, auto 1.15x) | 287 / 329 / 259 (**1.11x**, auto 0.87x) | 1114 / 1011 / 985 (**1.13x**, auto 1.10x) |

**Qwen3-32B shapes** — Machete FP16 µs / QuACK auto µs / QuACK tuned µs (tuned speedup vs Machete FP16)

| layer (K x N) | M=1 | M=32 | M=512 | M=2048 |
|---|---|---|---|---|
| 32b.qkv (5120x10240) | 23.1 / 25.0 / 19.2 (1.20x) | 27.5 / 26.0 / 20.3 (1.35x) | 80.2 / 103.8 / 80.1 (1.00x) | 285.1 / 277.7 / 266.3 (1.07x) |
| 32b.o (8192x5120) | 24.3 / 23.9 / 17.7 (1.37x) | 24.4 / 25.0 / 19.0 (1.28x) | 81.3 / 78.6 / 62.5 (1.30x) | 230.3 / 251.3 / 238.2 (0.97x) |
| 32b.gate_up (5120x51200) | 84.7 / 75.4 / 75.4 (1.12x) | 83.0 / 79.2 / 78.6 (1.06x) | 358.1 / 374.9 / 352.7 (1.02x) | 1456.5 / 1451.5 / 1349.8 (1.08x) |
| 32b.down (25600x5120) | 53.5 / 56.1 / 43.6 (1.23x) | 52.1 / 58.3 / 47.2 (1.10x) | 241.0 / 235.7 / 187.4 (1.29x) | 718.4 / 802.9 / 708.4 (1.01x) |
| **sum per layer** | 186 / 180 / 156 (**1.19x**, auto 1.03x) | 187 / 188 / 165 (**1.13x**, auto 0.99x) | 761 / 793 / 683 (**1.11x**, auto 0.96x) | 2690 / 2783 / 2563 (**1.05x**, auto 0.97x) |

Full M grid (1, 8, 16, 32, 64, 128, 512, 2048) with the QuACK auto config is in `mb_h4.txt`. M=1/8/16 behave alike (weight-bandwidth bound).

**Error** vs an FP32 dequantized reference (BF16 activations): QuACK 0.0023, Machete BF16 0.0032, Machete FP16 0.0003. The FP16 run is more precise because FP16 keeps 3 more mantissa bits in activations and output. **QuACK supports BF16 activations only**, so using it means serving these FP16 AWQ checkpoints with `--dtype bfloat16` and casting the FP16 group scales to BF16.

## 3. Checkpoint compatibility / repack
QuACK can't consume the AWQ checkpoint as-is. It needs (N, K/2) uint8 nibbles in K order, (N, K/128) BF16 scales and uint8 zeros, then `prepare_w4_weight(..., "int4awq")` to build its WGMMA fragment blob + SF strip. Conversion from the AWQ checkpoint: unpack `qweight`/`qzeros` (int32 with the [0,2,4,6,1,3,5,7] nibble order), transpose, re-pack, scales FP16→BF16, then `prepare`.
- Time on GPU: **2.9 ms** (Qwen3-8B `gate_proj`, 4096x12288), **6.2 ms** (32B `gate_proj`, 5120x25600). That's well under 1 s per model at load.
- Correctness: QuACK output vs vLLM's own `awq_dequantize_triton` + FP32 matmul: rel. err **0.0017** on both real layers.
- Group size must be 128 (`repack_int4_awq_sf` asserts `K/128` groups); g64/g32 exist only for symmetric `int4`.

## 4. Assessment vs the go threshold (≥1.10x on ≥5% of step time)
Amdahl estimate = Machete share × (1 − 1/speedup), using **tuned** QuACK. Per-layer sum at M=1 (c1), M=32 (c32), and M=2048 as a proxy for prefill (the prefill profile is at 4096 tokens/step):

| model | c1 | c32 | prefill |
|---|---|---|---|
| Qwen3-8B-AWQ | 1.50x on 71.5% → **~24% step** | 1.36x on 57% → **~15%** | 1.13x on 87% → **~10%** |
| Qwen3-32B-AWQ | 1.19x on 84% → **~13%** | 1.13x on 74% → **~8.5%** | 1.05x on 89% → ~4% |

**Passes the gate for both models at decode and for 8B prefill.** This depends on per-shape tuning. With QuACK's built-in heuristic the 32B gain drops to ~1.0x and 8B M=512 *loses* (0.87x), so an integration needs a small per-(N, K, M-bucket) config table, like H3's SwiGLU table. Caveats: these are kernel-level projections; the E2E arm would also switch the checkpoint from FP16 to BF16 activations (accuracy should be checked with GSM8K); and the sweep picks the best of ~19 configs on one-round medians (selection bias ~1–2%).

Suggested E2E (if the manager wants it): `--dtype bfloat16` on both arms (base = Machete BF16, PR = QuACK int4awq + per-shape table), Qwen3-8B-AWQ and Qwen3-32B-AWQ, TPOT c1/c8/c32/c128 + TTFT 4096, GSM8K per arm.
