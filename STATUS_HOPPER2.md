# Status: second Hopper node (role `hopper2`)

The node writes this file. Newest entries at the bottom.

## 2026-11-28 — H2-0 ack
- H2-0 ack, running. Node: 8x H100 80GB HBM3 (SM90), driver 610.57.04 (CUDA 13.3 UMD), system CUDA toolkit present (13.3). All GPU pairs NV18; GPUs 0-3 → NUMA 0 (CPUs 0-63), GPUs 4-7 → NUMA 1 (CPUs 64-127).
- First look: all 8 GPUs idle at 28-33 °C, ~70 W (no hot outlier at idle). Building env on `cfbc42099a`; GEMM health check follows.
- No partial H4 from the first node in `results/hopper/`, so H2-4 starts from scratch after H2-0.
