# B11: SM103 perf hunt (FA4 / QuACK / decode profile)

vLLM main `97dfd07879`, 1× B300 (SM103) per run, GPUs 0–4, CUDA 13.2, torch 2.14, FlashInfer 0.7.4, quack-kernels 0.6.5, nvidia-cutlass-dsl 4.8.0.
- Qwen3.6-35B-A3B-FP8 TP1: real weights, real routing.
- Qwen3.8-Flash-Next-FP8 TP1: dummy weights with zipf routing. The 186 GB checkpoint downloads at ~3 MB/s unauthenticated.

Both arms use the same prebuilt extensions; main's csrc change since that build is DSA top-k only. Profiles come from `vllm bench forward` (CUDA graphs, KV 1024 ±50%, sampler included). Rooflines use 8 TB/s nominal HBM.

## Ranked candidates

| # | candidate | evidence | expected E2E (Qwen3.6 TP1) | change size | overlap |
|---|---|---|---|---|---|
| 1 | **Sampler: give FlashInfer `top_k_max`** so `top_k_top_p_sampling_from_logits` takes its top-k-first path. Qwen's generation config sets top_k=20. | Without it, every step runs radix mask + full 248k-vocab `torch.softmax` + top-p (168 µs at M=1). **Measured E2E with #64116 merged on main:** c1 −3.62%, c8 −3.21%, c32 −1.57% (paired, 1 pass × 4 rounds, 4/4 each, AOT 0). | **−3.6 / −3.2 / −1.6%** (measured) | ~30 lines | **#64116 (open since 10-27)** does exactly this. Supporting it is the cheapest win. #65649 (open) is the same idea for the non-fused masking path. |
| 2 | **Shared-expert gate tail fusion.** Today it is 4 kernels per layer: bf16 gate GEMV, sigmoid, broadcast mul (non-vectorized), and the add into the routed output. | 7.6 µs per layer in the trace. A one-kernel standalone prototype (dot + sigmoid + mul + add, bf16 rounding kept) takes 1.84 / 1.84 / 2.25 µs at M=1/8/32, vs the eager chain's 7.58 / 8.40 / 10.44 µs; max diff 1–2 bf16 ulp. | ≈ −230 µs per step at M=1 (**~−7%**), −260 µs at M=8 (~−6%), −330 µs at M=32 (~−5%). Upper bound from the forward. | small (one Triton kernel + Qwen3MoeMLP / combine hook) | **#43190** (open, stale since 10-02) fuses dot + sigmoid + mul in `qwen3_moe.py`, which is Qwen3.6's shared MLP, but not the add. **#60846** (SM120 opt-in, Qwen2MoeMLP). **#45717** (opt-in, NVFP4 122B). None is merged. |
| 3 | **Batch-1 MoE GEMMs far from roofline.** FlashInfer TRT-LLM-gen bmm, autotuned at startup. | At M=1 every token touches exactly 8 experts per layer: FC1 403 µs vs 84 µs roofline (**21%**), FC2 241 µs vs 42 µs (**17%**). About 10 µs to stream 16.8 MB per layer, so latency-bound (few CTAs, prologue). At M≥8 real routing is skewed, so the roofline is not computable from the trace. | ~−10% at c1 if a small-M kernel reached ~50% of roofline; small at c≥8 | large (new low-latency MoE GEMV path, or FlashInfer tile work) | No open vLLM PR targets this. #63386 (GLM-5.3 CuTe DSL latency decode) and #66865 (Qwen3.8 GB300 specialized decode) may have reusable pieces. |
| 4 | **Post-attention norm output quantized twice**, row-major for the MoE (UE8M0 scales) and for the shared-expert gate_up. Both are `per_token_group_quant_8bit_register_kernel` on the same tensor. | 2 × ~2.2 µs × 40 per step; fusing into the post-attention `gemma_fused_add_rmsnorm` would also save a norm-output round trip. | ~−3% at M=1, ~−1.5% at M=8 | medium (the #70028 pattern applied to post_attention_layernorm, emitting both scale formats) | Not covered by #69778 / #70028 (input_layernorm only) or #69812 (per-token). |
| 5 | Remaining group-quant launches before out_proj / o_proj (40 per step, ~2.4 µs each at M=1) | profile | ~−3% at M=1 | medium | Planned follow-up once #69812 lands (nq152, parked). |

**Already covered by our open PRs.** Dense FP8 `cutlass_3x_gemm_fp8_blockwise` (160 calls, 1.40 GB of weights per step) runs at **13% of roofline** at M=1 (1356 µs vs 176 µs) and is the #1 kernel: #69949 + #70205. Group quant into down_proj: #69778. Group quant into the input norm: #70028.

**Near roofline / not worth it.**
- lm_head bf16 GEMV: 148 µs vs 127 µs (86%).
- GDN recurrent decode, fp32 state: 77% at M=32 (22 µs per layer vs 16.8 µs).
- Attention decode (TRT-LLM fmha) at M=32: 47%, but only 2–3% of the step.

## Task 1: which attention path runs

- **Qwen3.6-35B-A3B-FP8.** The serve log shows `attention | 10 | FLASHINFER` (decode trtllm-gen, prefill TRTLLM, bf16 KV, sm103) and `GDN_ATTN` × 30.
  - FA4 isn't used because `vllm/platforms/cuda.py:158-165` ranks FLASHINFER ahead of FLASH_ATTN for causal attention on CC major 10. That has been by design since 2025-11; it is not a guard failure.
  - Forcing `--attention-backend FLASH_ATTN` does select FA4 (`Using FlashAttention version 4`), but the forward gets slower: M=1 +2.6%, M=8 +1.1%, M=32 +1.8%, M=8/KV 4096 +1.0%. **The default is right.**
  - Block size is 1088 under both backends (hybrid page alignment), which is not a multiple of 128. So FA4's SwapAB decode kernel and 2-CTA are ineligible, and FA4 decode runs the 128-row tile with PackGQA and split-KV.
- **Qwen3.8-Flash-Next-FP8.** It doesn't use FA or FlashInfer for full attention.
  - Backends: `QWEN4_EXP_QSA_TRITON` × 12, `QWEN4_EXP_QSA_INDEXER`, `GDN_ATTN` × 36, `PLE_SHORT_CONV_ATTN` × 1. Block size is 1600.
  - `--attention-backend FLASH_ATTN` has no effect (forward within ±1%). QSA prefill already uses FlashInfer on Blackwell (#69735).

## Task 2: FA4 pin vs upstream (details in `fa4_compare.md`)

- **The pin.** `1b32c5caf7` is the **squash-merge commit** of vllm-project/flash-attention#222 (merged 2026-11-18), not an unmerged PR head.
  - Its tree is identical to the tested head, and it is on the fork's main.
  - vLLM #68348 kept its "[Do Not Merge]" title in the commit message, but it was repointed to the merged SHA before merging. **No problem.**
  - The fork's main has one extra commit, 8bfb5666 (#206, SM90-only FP8), which doesn't affect B300.
- **Upstream since the sync base 54b044c8:** 34 `flash_attn/cute` commits. None adds an inference feature for bf16 hd256 on SM100/SM103. The pin already has split-KV, SwapAB decode, paged KV (TMA / cp.async), tensorwise FP8, SM103 tunings, hd256, pack_gqa, varlen, sinks, softcap, CLC, 2-CTA and block sparsity.
- **Worth noting:**
  - **c055bdc6** (#3148, +7/−1): hd256 PackGQA selection fix. It matters for 12:1 GQA prefill >128 tokens (Qwen3.8's 24:2 shape; upstream reports 15–28% on B200), but Qwen3.8 doesn't run FA.
  - **8e3cb052 / f4cc3b0b:** varlen calls auto-route to the persistent scheduler. Possibly helps mixed steps; unmeasured.
  - **9985a28e** (#3027): API break (`q/k/v_descale` becomes `TensorwiseScales`). vLLM's wrapper needs an update at the next sync.
  - MXFP8 attention is hd64/128 dense only: no effect.
- **Net:** with FlashInfer as the default, no FA4 sync item moves Qwen3.6 or 3.8 on B300 today.

## Task 3: QuACK vs vLLM kernels (`mb_quack.py`, `mb_quack.json`)

CUDA graphs (20 calls per replay), L2 flushed before each replay, median of 7 interleaved rounds. Baselines are the kernels vLLM runs for Qwen: FlashInfer CuTe `rmsnorm` and `gemma_fused_add_rmsnorm`, plus the vLLM C++ `rms_norm`.

| op | result |
|---|---|
| RMSNorm, H 2048/4096/5120, M 1–16 | QuACK 2.05–2.56 µs; FlashInfer 1.74–2.25; vLLM C++ 2.35–2.76. **QuACK is 5–18% slower than FlashInfer.** |
| RMSNorm, M 256–8192 | QuACK within 0–16% of FlashInfer (slower), faster than vLLM C++ |
| fused add + Gemma RMSNorm (weight_offset=1), M 1–16 | QuACK **12–22% slower** than `gemma_fused_add_rmsnorm` |
| fused add + Gemma RMSNorm, M 256–8192 | QuACK 0–43% slower (1.43× at H2048/M8192 and H4096/M4096) |
| softmax over V=248320, fp32 | **QuACK 5.8 / 5.8 / 15.7 µs vs torch 62.8 / 63.5 / 64.4 µs** at M=1/8/32 (10.8× / 10.9× / 4.1×); max diff 6e-8 |

- **Numerics.** Max error vs an fp32 reference is identical to FlashInfer's for both norms. Outputs are bit-equal to FlashInfer at M ≤ 16 and differ by 1 ulp in places at larger M. Residual outputs are bit-equal.
- **JIT.** First call per new (op, N) is 0.35–0.48 s for rmsnorm and 0.56 s for softmax; later M values reuse it.
- **CUDA graphs.** All QuACK ops captured and replayed fine.
- **QuACK allocates its outputs** (out and residual_out), whereas FlashInfer's add+norm is in place.
- **Verdict.** No case for QuACK norms; FlashInfer's CuTe norms are already better here. QuACK softmax is the only clear win. It would matter for sampling paths that still softmax the full vocab (top-p-only requests, and the fallback when #64116's top-k path doesn't apply). #54417 (open) targets the same softmax by fusing temperature into it. QuACK top-k wasn't benchmarked, since FlashInfer's radix top-k is already used in the fast path. Any QuACK integration needs the owner's OK first, per TASKS.

## Task 4: decode profile (`top15_qwen36.txt`, `top15_qwen38.txt`)

Qwen3.6 TP1 forward medians (default backend): M=1 2894 µs, M=8 4284 µs, M=32 6162 µs, M=8/KV 4096 4327 µs.

Top kernels at M=1, % of busy time:

| kernel | share | covered by |
|---|---:|---|
| CUTLASS block-FP8 dense | 30.8% | #69949 / #70205 |
| group quant (200 calls) | 10.8% | #69778 / #70028 cover 80 of 200; candidates 4–5 cover the rest |
| MoE FC1 | 9.2% | candidate 3 |
| skinny bf16 (router, gates, lm_head) | 8.7% | |
| MoE FC2 | 5.5% | candidate 3 |
| routing | 4.4% | |
| FlashInfer add+norm | 4.2% | |
| MoE activation | 3.9% | |
| finalize | 2.9% | |
| torch elementwise (sigmoid, mul, add) | 5.2% | candidate 2 |
| `cunn_SoftMaxForward` | 2.4% | candidate 1 |
| attention | 2.1% | |

Qwen3.8 at M=8/32 is dominated by MoE FP8 bmm (27–35%) and BF16 nvjet GEMMs plus cuBLASLt split-K reduce kernels (6.9% at M=8, 194 calls), from its unquantized GDN and shared-expert projections. Our open PRs don't cover these. #66865 (GB300 specialized decode) is the related upstream work.

## Not done

- No E2E for candidates 2–5.
- No real-weight Qwen3.8 run (download too slow unauthenticated).
- No TP2.
- No nvidia-cutlass-dsl 4.8.1 evaluation.
