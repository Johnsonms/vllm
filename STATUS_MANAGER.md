# STATUS_MANAGER (manager-owned)

## 2026-11-23 — context for the worker
- The round's target model is Qwen/Qwen3.8-Flash-Next-FP8 on 8x B200, base vLLM main @ 8399da3780.
- The manager node is running an E2E + GSM8K A/B for a fused shared-expert sigmoid-gate kernel (Qwen MoE MLP) on GPUs 0-3. Its forward-step result: decode −0.6…−3.1%, prefill −2.0…−2.2%.
- W1/W2 (TP8 AR+RMSNorm fusion cap on B200) are delegated to the worker, so they can run in parallel with the manager's E2E.
- Rule reminder: the comms branch is the only thing either side pushes. All upstream (vllm-project) activity is done by the human owner.

## 2026-11-23 (later) — manager's own queue
- Shared-gate E2E round 1 done (main vs fused): GSM8K 0.965 / 0.964; decode TPOT c4 −0.8%, c16 −1.7%; prefill 8k TTFT
  unchanged so far. Round 2 running.
- Manager will also run W1 (TP8 fusion cap kernel) on its own node after E2E, as a cross-check of the worker's numbers.
- New manager task M1: B200 **TP4** standalone all-reduce backends, 2-128 MB (benchmark_device_communicators.py).
  Reason: Qwen3.8 TP4 8k prefill all-reduces 42 MB tensors. These exceed the 32 MB FlashInfer/symm-mem caps at
  (SM100, ws=4); `FI_MNNVL_ALLREDUCE_MAX_SIZE_MB` has no (100,4,1) entry, while B300 has (103,4,1)=80. So they fall
  back to NCCL, which takes 16.7% of prefill time.
- Node note: the FlashInfer MNNVL log prints "FABRIC handles failed ... CUDA_ERROR_NOT_PERMITTED; using POSIX fds".
  On a single node this is benign (no IMEX channel). Worker: report whether you see the same warning.

## 23:50Z
- W2 follow-up done on the manager node with real Qwen3-32B weights. The TP8 fusion cap 1 -> 16 MB gives decode TPOT -9.6/-6.5/-5.9%
  at c128/192/256, and GSM8K is unchanged. It is now local PR candidate T (raise the cap from 1 to 16 MB).
- FlashInfer 0.7.6rc1 vs 0.7.4 (Qwen3.8 TEP4 E2E): 8k-prefill TTFT -8.4/-9.2%, and GSM8K is unchanged (flashinfer #9329).

## 01:43Z — round closed
- W6 accepted: the prefill non-determinism comes from FlashInfer `gvr_2` top-k tie-break and output order in the QSA
  indexer (first divergent module `layers.3.self_attn.indexer`). The selection is always a valid top-k. It starts at
  2,056 tokens (block top-k 512), and dummy weights probably inflate the tie counts. This is not a wrong-data bug.
- Worker stood down. Channel idle. No further tasks.

## 2026-11-28: QuACK scouting round 2 (owner: "are there more QuACK kernels that beat vLLM?")
- Manager: QKV GEMM + per-head RMSNorm (QK-norm) + RoPE in QuACK's epilogue (`epilogue/head_rmsnorm.py`, `epilogue/rotary.py`) vs vLLM's cuBLAS + qk-norm/rope, Qwen3 dense BF16, B200.
- Worker W21: FP8 (block-128) GEMM / gated + quantized output vs vLLM FP8 defaults, B200.
- B300 B15: NVFP4 GEMM / gated + fp4 output vs vLLM NVFP4 defaults, SM103.
- Hopper H1-H3 (MoE / routing / SwiGLU on SM90), then H4: `gemm_w4` vs Marlin/Machete.
- Already ruled out: QuACK RMSNorm (11-27: E2E +0.3..+1.1% slower vs Inductor-native), QuACK MoE vs TRT-LLM on SM100/SM103.
- 11-28 later: H1 accepted (SM90 default MoE = FlashInfer CuTe-DSL; added "default + P1" arm to H2). Queued W22 (residual+RMSNorm folded into QuACK GEMM epilogues) and B16 (QuACK softmax/topk for the sampler). Manager does QKV GEMM + QK-norm + RoPE epilogue (`qk_rope_epi`).
