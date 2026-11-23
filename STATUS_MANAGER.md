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
