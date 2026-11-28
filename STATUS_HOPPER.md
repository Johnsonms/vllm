# Status: Hopper node (role `hopper`)

The Hopper node writes this file. Newest entries at the bottom.

## 2026-11-28 — H1 ack
- H1 ack, running. Node: 8x H100 80GB HBM3 (SM90), all pairs NV18 (full NVLink mesh), driver 595.91.07 (CUDA 13.2 driver). No system CUDA toolkit `bin/` (no nvcc) on this node; relying on pip-provided CUDA libs.
- Building env on base `cfbc42099a` now; will start H2 directly after H1 is pushed.

## 2026-11-28 — H1 done (results/hopper/H1/SUMMARY.md)
- 8x H100 80GB SM90, full NV18 mesh, torch 2.14.0+cu132, Triton 3.8.0, FlashInfer 0.7.4, quack-kernels 0.6.5, cutlass-dsl 4.8.0.
- **SM90 default BF16 MoE = `FlashInferCuteDSLSm90Experts` (FlashInfer CuTe-DSL), not Triton.** It is modular, so routing is still vLLM `fused_topk` → P1 applies to the default path too. Working `--moe-backend`: triton, flashinfer_cutedsl, flashinfer_cutlass; trtllm (SM100-only) and batched_triton fail at init.
- QuACK on SM90: gemm, gemm_gated_out swiglu concat B (dense + varlen), varlen gemm all PASS (rel err 1.7e-3). `quack.topk(softmax=True)` PASS for k=8; FAIL for k=4/k=2 at M>1 (mValues stride alignment), as expected.
- Starting H2 (base = default CuTe-DSL and separately `--moe-backend triton`; PR = triton + P1 + P2).
- Node policy from the owner: if anyone else starts using a GPU here, this node steps back immediately (watcher kills my jobs and posts here).

## 2026-11-28 — H2 progress: kernels done, paired E2E running
P1/P2 rebuilt locally on `cfbc42099a` (env-gated, code local). P2 configs pinned from a one-off SM90 autotune at M=4096: **FC1 128x192, cluster 2x1, pingpong; FC2 128x128, cluster 1x2, pingpong** (both non-persistent; all SM90 winners are pingpong). These fixed configs lose ~11–14% on FC1 vs per-M tuning at M=1024/8192.
Kernels (cold L2, CUDA graph, CUPTI, median of 7; `results/hopper/H2/mb_h2.json`):
- **P1** vs `ops.topk_softmax` (E=128, k=8): M=1 1.70x, 8 1.72x, 64 1.59x, 256 1.60x, 1024 1.57x (4096 1.99x, 8192 2.67x). Ids identical, weights within 1.5e-7. The k-derived alignment fixes k=4/k=2 at M>1.
- **P2** vs **default FlashInfer CuTe-DSL** experts: 0.92x / 0.93x / 0.90x / 0.91x / 0.90x / 0.96x at M = 64 / 256 / 1024 / 2048 / 4096 / 8192 (**QuACK slower everywhere**). vs Triton `fused_experts` (no H100 BF16 config file for E=128,N=768 → defaults): 0.88x / 0.93x / 0.94x / 1.01x / 1.10x / 1.18x. Rel err vs FP32: QuACK 0.0029, CuTe-DSL 0.0029, Triton 0.0043.
- Why P2 loses (M=4096, warm, `mb_p2_breakdown.txt`): QuACK GEMMs are faster (FC1 424 vs 433 µs, FC2 228 vs 254 µs), but `moe_permute` + `moe_unpermute` + sort cost ~148 µs vs ~79 µs (FlashInfer gathers inside GEMM1; it pays only unpermute + routing). Same pattern as B14 vs TRT-LLM.
- In-server path check: P1 and P2 both fire (P2 on a 1504-token prefill); greedy output matches.
E2E now running, three paired experiments at once (2 passes x 6 rounds, GPU-swapped): GPUs 0/1 default vs triton+P1+P2; 2/3 triton vs triton+P1+P2; 4/5 (extra) **default vs default+P1**, since P1 also applies to the default path. GSM8K once per arm.
