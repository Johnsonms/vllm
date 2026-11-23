# W2 (run on the manager node): Qwen3-32B TP8, dense, dummy weights, real serving (compile on), 2 alternating rounds

The fusion fires: DEBUG logs show `Maximal number of tokens used by Flashinfer Allreduce Fusion` = 102 / 1638 / 6553
for the default 1 MB / 16 MB / 64 MB caps.

| workload | default | 16 MB | 64 MB |
|---|---|---|---|
| decode TPOT c16 | 4.41 / 4.48 | -0.27% | +0.33% |
| decode TPOT c64 | 5.10 / 5.12 | -0.59% | +0.49% |
| **decode TPOT c256** | 9.36 / 9.38 ms | **-6.48%** (8.72 / 8.80) | -5.68% |
| 8k prefill TTFT c1 | 116.5 / 117.5 | +0.59% | +1.69% |
| 8k prefill TTFT c4 | 345.0 / 344.8 | -0.17% | +0.52% |

- The gain appears exactly where the decode batch exceeds 102 tokens.
- c16/c64 are already fused at the default cap.
- The 8k prefill chunk exceeds every cap, so prefill is unaffected.
- Next: real weights + GSM8K, decode c32..c384, caps default / 8 / 16 / 32 (manager node).
