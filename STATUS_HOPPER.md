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
