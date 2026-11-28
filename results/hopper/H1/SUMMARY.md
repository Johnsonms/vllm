# H1 — Hopper env + SM90 default-path survey (2026-11-28)

## 1. Environment
| item | value |
|---|---|
| GPUs | 8x NVIDIA H100 80GB HBM3 (79.2 GiB visible), `get_device_capability()` = (9, 0) |
| topology | full NVLink mesh: every GPU pair `NV18` |
| driver | 595.91.07 (CUDA 13.2 driver); no system CUDA toolkit `bin/` (no nvcc), pip CUDA libs only |
| vLLM | `cfbc42099a` (0.34.1rc1.dev321), precompiled, `vllm.__file__` from the local tree |
| torch | 2.14.0+cu132 |
| Triton | 3.8.0 |
| FlashInfer | 0.7.4 (python, cubin; jit-cache 0.7.4+cu130), `vllm download-kernels` OK |
| quack-kernels | 0.6.5 |
| nvidia-cutlass-dsl | 4.8.0 (libs-cu12 / libs-cu13 4.8.0) |
| cuda-python / cupti-python | 13.5.0 / 13.2.0 |

## 2. Default BF16 MoE path on SM90 (Qwen/Qwen3-30B-A3B, TP1, no `--moe-backend`)
**Default = FlashInfer CuTe-DSL SM90 (`FlashInferCuteDSLSm90Experts`), not Triton.** Log (DEBUG):
```
[oracle/unquantized.py:375] Using FlashInfer CuTeDSL Unquantized MoE backend out of potential backends:
  ['FlashInfer CuTeDSL', 'TRITON', 'BATCHED_TRITON', 'FlashInfer TRTLLM', 'FlashInfer CUTLASS'].
[oracle/base.py:398] Using FlashInferCuteDSLSm90Experts MoE backend
  moe       |    48 | FlashInferCuteDSLSm90Experts | model.layers.[0-47].mlp.experts.routed_experts
```
Why: `_get_priority_backends` (oracle/unquantized.py) on `is_device_capability_family(90)` inserts FLASHINFER_CUTEDSL first and moves TRTLLM and CUTLASS to the back ("The other FlashInfer unquantized MoE kernels are slower than Triton there"). The class needs SM90 + `has_flashinfer_cutedsl_moe_bf16()` (true with FlashInfer 0.7.4).
- It is a **modular** experts class (`FusedMoEExpertsModular`): routing is still vLLM's `fused_topk` (`ops.topk_softmax`), so **P1 applies to the default path too**. The kernel does a gather-fused GEMM1 + activation epilogue, GEMM2 writing one row per (token, slot), then a fixed-order FP32 combine (no separate permute kernel, unlike the QuACK P2 path).
- Its tactics come from the FlashInfer autotuner at warmup: `CuteDslFusedMoE::run_moe_sm90::Swiglu`, 21 token buckets (1..8192), ~53 s at startup (graph-capture safe).
- The router GEMM (`mlp.gate`) uses vLLM's CuTe-DSL `ll_bf16` skinny kernels at small M (warmup log).

`--moe-backend` values on this node (unquantized BF16, each served + 1 greedy request):
| value | result |
|---|---|
| (none) / `flashinfer_cutedsl` | OK, `FlashInferCuteDSLSm90Experts` |
| `triton` | OK, `TritonExperts` |
| `flashinfer_cutlass` | OK, `FlashInferExperts` |
| `flashinfer_trtllm` | FAIL at init: "kernel does not support current device cuda" (SM100-only) |
| `batched_triton` | FAIL at init: "BatchedTritonExperts expects batched_experts activations but the selected prepare/finalize uses standard" (needs a batched all2all) |
Greedy text for "The capital of France is" is identical for default/cutedsl/triton over 16 tokens; flashinfer_cutlass diverges at token ~13.

## 3. QuACK smoke test on SM90 (no vLLM; `smoke.py`, `smoke_result.txt`, `topk_matrix.log`)
| test | result | rel. err vs FP32 |
|---|---|---|
| `gemm` 4096x2048x1536 | PASS | 1.66e-3 |
| `gemm_gated_out(swiglu, concat_layout='B')` dense M=1024, K=2048, I=768 | PASS | 1.66e-3 |
| same, varlen `cu_seqlens_m` (8 experts incl. empty groups, (E,K,2I) weights) + varlen `gemm` FC2 | PASS | 1.66e-3 / 1.66e-3 |
| `quack.topk(softmax=True)` BF16, k=8 (E=64/128/256, M=1..4096) | PASS, ids identical to torch.topk, values within BF16 rounding (≤1.9e-3) |
| `quack.topk(softmax=True)` BF16, **k=4 (E=64) and k=2 (E=32), M>1** | **FAIL**: `Invalid mValues.strides[0] ... expected to be divisible by 8` (M=1 passes). This is the fixed fake-tensor alignment noted in H2; irrelevant for Qwen3-30B-A3B (k=8), handled in the P1 subclass. |

Configs QuACK's autotuner picked on SM90 (all `is_dynamic_persistent=False`, the SM90 search space never uses it):
- `gemm` 4096x2048x1536: 256x192, cluster 1x2, cooperative (no pingpong), **swap_ab**.
- varlen `gemm` (FC2-like): 128x256, cluster 1x2, cooperative.
- `gemm_gated_out` dense M=1024: 128x128, cluster 2x1, **pingpong**; varlen: 128x192, cluster 1x2, **pingpong**.
(First calls compile + autotune: 5–14 s per new shape.)

## Implications for H2
- Base for H2 = FlashInfer CuTe-DSL SM90 experts (default) and, separately, `--moe-backend triton`.
- P1 (routing) can be measured on **both** default and Triton paths since the default still calls `fused_topk`.
- P2 replaces the experts, so the PR arm needs `--moe-backend triton` (+ P1 + P2).

## Node hygiene
A local watcher checks every 5 s for any GPU usage not from this session; if one appears, all hopper jobs are killed and a "stepped back" note is pushed here. The GPUs were idle (0 MiB, 0%) when work started; driver accounting is off, so there is no earlier usage history.
