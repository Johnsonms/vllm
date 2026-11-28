# Status: second Hopper node (role `hopper2`)

The node writes this file. Newest entries at the bottom.

## 2026-11-28 — H2-0 ack
- H2-0 ack, running. Node: 8x H100 80GB HBM3 (SM90), driver 610.57.04 (CUDA 13.3 UMD), system CUDA toolkit present (13.3). All GPU pairs NV18; GPUs 0-3 → NUMA 0 (CPUs 0-63), GPUs 4-7 → NUMA 1 (CPUs 64-127).
- First look: all 8 GPUs idle at 28-33 °C, ~70 W (no hot outlier at idle). Building env on `cfbc42099a`; GEMM health check follows.
- No partial H4 from the first node in `results/hopper/`, so H2-4 starts from scratch after H2-0.

## 2026-11-28 — H2-0 done (results/hopper2/H2-0/SUMMARY.md)
- Env on `cfbc42099a` matches the first node (torch 2.14.0+cu132, Triton 3.8.0, FlashInfer 0.7.4, quack-kernels 0.6.5, cutlass-dsl 4.8.0).
- All 8 H100s healthy: 30 s BF16 GEMM 679-695 TFLOPS (2.3% spread, power-capped at 700 W), idle 28-33 °C, no ECC errors. No GPU excluded.
- H2-4 ack, starting (Qwen3-8B-AWQ and Qwen3-32B-AWQ already downloaded).
