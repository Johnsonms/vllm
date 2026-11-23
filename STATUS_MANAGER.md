# STATUS_MANAGER (manager-owned)

## 2026-11-23 — context for the worker
- The round's target model is Qwen/Qwen3.8-Flash-Next-FP8 on 8x B200, base vLLM main @ 8399da3780.
- The manager node is running an E2E + GSM8K A/B for a fused shared-expert sigmoid-gate kernel (Qwen MoE MLP) on GPUs 0-3. Its forward-step result: decode −0.6…−3.1%, prefill −2.0…−2.2%.
- W1/W2 (TP8 AR+RMSNorm fusion cap on B200) are delegated to the worker, so they can run in parallel with the manager's E2E.
- Rule reminder: the comms branch is the only thing either side pushes. All upstream (vllm-project) activity is done by the human owner.
