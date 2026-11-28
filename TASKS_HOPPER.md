# Tasks for the Hopper node (role `hopper`)

The manager writes this file; the Hopper node only reads it. Report in `STATUS_HOPPER.md` and `results/hopper/H<n>/`.
Read `setup/HOPPER_START.md` first.

## Why Hopper (context)

On B200/B300 (SM100/SM103) the manager and B300 measured three QuACK-in-vLLM prototypes:
- **Dense SwiGLU** (gate_up GEMM with SiLU-and-mul in the epilogue) beats vLLM's default cuBLAS + `silu_and_mul`: TPOT -2..-3.5% at c64/c128, prefill TTFT -3..-3.7%, GSM8K unchanged. A vLLM PR is ready but gated to SM100/SM103 only.
- **MoE experts (P2) and routing (P1)** beat vLLM's Triton MoE (prefill -6..-15%, decode -2..-3%) but **lose to the Blackwell default, FlashInfer TRT-LLM** (`TrtLlmBf16ExpertsMonolithic`): decode +13..+17% slower. Profile: QuACK FC1 is 1.15x faster than TRT-LLM's, FC2 is 1.11x slower, and the separate permute/unpermute costs ~2.5 ms extra per 4096-token prefill.

The open question for Hopper: **what is vLLM's default BF16 MoE path on SM90, and do QuACK's MoE / routing / SwiGLU beat the defaults there?** That decides whether they land at all.

## H1: environment + SM90 default-path survey  [assigned 2026-11-28, high priority]

1. **Env** as in `setup/HOPPER_START.md` §2. Record GPU model (H100/H200, memory), count, driver, CUDA, torch, Triton, FlashInfer, quack-kernels, CUTLASS DSL (`nvidia-cutlass-dsl`) versions, `torch.cuda.get_device_capability()` (expect `(9, 0)`), and `nvidia-smi topo -m` (are all GPUs NVLink-connected?). No hostnames/IPs.
2. **Default MoE path.** Serve `Qwen/Qwen3-30B-A3B` BF16, TP1, no `--moe-backend`. From the log, report which experts class / backend vLLM picks on SM90 and why (quote the selection log lines). Also list which `--moe-backend` values work on this node (try `triton`, and any FlashInfer/CUTLASS options the selector offers for unquantized BF16 on SM90).
3. **QuACK smoke test on SM90** (no vLLM): `quack.gemm_interface.gemm`, `gemm_gated_out(..., activation='swiglu', concat_layout='B')` (dense and varlen `cu_seqlens_m`), and `quack.topk` with `softmax=True`, on BF16 inputs. Report pass/fail, errors and the default configs they pick.
4. **Deliverable:** `results/hopper/H1/SUMMARY.md`. Push, then start H2.

## H2: QuACK MoE + routing vs the SM90 default  [assigned; start after H1]

Rebuild two prototypes **locally from this description** on base `cfbc42099a` (keep code local). Each is opt-in through an env var.

**P1. Lean MoE routing (Qwen3-30B-A3B BF16, Triton MoE path).** In `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py::fused_topk`, when scoring is softmax + renormalize + power-of-two E and k: call QuACK's TopK kernel with `softmax=True` so it writes **FP32 weights straight into `topk_weights`** and int32 ids into `topk_ids`. `quack.topk.TopK` asserts the values dtype equals the input dtype; subclass it with a `__call__` that allows Float32 values and a compile that uses a Float32 fake tensor for values. **Watch out:** in 0.6.5 the values/indices fake-tensor alignment is fixed, which fails with "Invalid mValues.strides[0]" for k=4/k=2 at M>1; derive the alignment from k. Do not fill `token_expert_indices` (discarded by `FusedTopKRouter._compute_routing`). Check: ids identical to `ops.topk_softmax`, weights within 1e-6. Note: vLLM's topk contract also sets expert id -1 for padding rows (`VLLM_MOE_SKIP_PADDING`) and clamps NaN; QuACK does not. For a benchmark that is fine, but say so in the report.

**P2. QuACK MoE experts for large M.** In `TritonExperts.apply` (`experts/triton_moe.py`, first class), for unquantized BF16 + SiLU + no expert_map / LoRA / input-side router weights, and `num_tokens >= KOPT_QUACK_MOE_MIN_M` (default 1024):
- `moe_permute` (vLLM's CUDA kernel);
- `quack.gemm_interface.gemm_gated_out(px, w1.transpose(1,2), None, h, activation='swiglu', cu_seqlens_m=expert_first_token_offset.int(), concat_layout='B')`;
- `gemm(h, w2.transpose(1,2), out=y, cu_seqlens_m=...)`;
- `moe_unpermute(output, y, topk_weights, inv_idx, offset)`.

**Pin configs (critical).** Wrap `quack.autotuner.Autotuner.__init__` so every tuner, **including ones created lazily** (the gated tuner is created on first call), gets a cache whose misses return one fixed config. On SM100 that was tile 128x256, cluster 2x1, dynamic persistent, no pingpong. **SM90 configs differ** (pingpong usually matters on Hopper): first let QuACK autotune once at M=4096 for w1 and w2 (outside the server), and pin the winners. Report what you pinned.

**What to measure (`results/hopper/H2/`):**
1. Kernels (cold L2): P1 vs `ops.topk_softmax` at M in {1, 8, 64, 256, 1024}; P2 vs the **default** experts path (from H1) and vs Triton `fused_experts` at M in {64, 256, 1024, 2048, 4096, 8192}. Include relative error vs the reference.
2. E2E, paired, 2 passes x 6 rounds, GPU-swapped. Base = **default** backend (no `--moe-backend`); PR = P1 + P2 (plus `--moe-backend triton` if the default is not Triton). Prefill TTFT (4096 in / 1 out, c1/c4/c16) and decode TPOT (512/512, c1/c8/c32/c128). If the default is not Triton, also run base = `--moe-backend triton` vs PR, so we see both.
3. GSM8K (5-shot, temperature 0) once per arm, base vs PR.
4. If the PR loses anywhere: a torch-profiler breakdown of one 4096-token prefill per arm (FC1, FC2, routing, permute/combine), as B300 did in `results/b300/B14/prof_breakdown.txt`.

## H3: dense SwiGLU on SM90  [assigned; start after H2, or in parallel on free GPUs if the node is big enough and H2's A/B GPUs stay idle]

Goal: extend the SwiGLU PR (today SM100/SM103 only) to SM90 if it wins there.

**Design that landed on SM100** (rebuild locally as a hook in `Qwen3MLP.forward`, env-gated): for an unquantized BF16 `gate_up_proj` weight `W` (`(2I, K)`, `[gate; up]`), a custom op (`direct_register_custom_op`) computes `silu(x W_g^T) * (x W_u^T)`:
- **M < 16:** `F.linear(x, W)` + a 2-D Triton SiLU-and-mul (grid rows x 1024-column blocks). vLLM's C++ `silu_and_mul` launches one block per row and is ~2.5x slower at M=1.
- **M >= 16:** `quack.gemm_interface._gemm_act_call(x, W.t(), None, out, None, None, activation='swiglu', concat_layout=('B',), dynamic_scheduler=cfg.is_dynamic_persistent, tuned=False, config=cfg)` where `cfg` comes from a **fixed 4-bucket table** by M (no runtime autotuning, so no stalls and graph-capture safe). SM100 table: M<=128 tile 128x192; <=256 256x192; <=384 128x256 dynamic; else 256x256 dynamic; all cluster 2x1, no pingpong.

**Steps:**
1. **Build an SM90 bucket table.** For Qwen3-8B (`W` 24576x4096) and Qwen3-32B (51200x5120), autotune QuACK at M in {16, 32, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192} and compare with `F.linear` + `silu_and_mul` (cold L2). Pick at most 4 buckets of configs that stay within a few % of per-M autotuning. Report the table, the per-M speedup vs cuBLAS, and the crossover M below which cuBLAS wins (SM100: 16).
2. **E2E**, paired as above: Qwen3-8B and Qwen3-32B BF16, TP1, default backend; base = unmodified, PR = hook on. TPOT (512/512, c1/c8/c32/c64/c128), prefill TTFT (4096/1, c1/c4/c16) for Qwen3-8B. GSM8K once per arm.
3. Relative error vs an FP32 reference, fused vs unfused (SM100: fused 0.0017 vs unfused 0.0035).

No GitHub actions; code stays local.

## H4: QuACK W4A16 GEMM vs vLLM's int4 kernels on Hopper  [queued; after H2/H3]
quack-kernels 0.6.5 has `quack/gemm_w4.py`: a weight-only-quantized GEMM **for SM90** (int4 / int4-AWQ / mxfp4 / nvfp4 / fp8 weights, decoded to bf16 in registers, split-K supported), with an offline `prepare` repack step.
1. For `Qwen/Qwen3-8B-AWQ` and `Qwen/Qwen3-32B-AWQ` (and a GPTQ-int4 Qwen3 if one exists ungated): record which kernel vLLM uses on H100 (Marlin / Machete / other) and its time share at c1/c32/prefill 4096.
2. Microbench (cold L2, CUPTI, CUDA graph) QuACK `gemm_w4` vs vLLM's kernel on the real layer shapes at M in {1, 8, 16, 32, 64, 128, 512, 2048}. Report speedup, error vs a dequantized-FP32 reference, and whether QuACK can use the checkpoint's packing / zero-points / group size directly or needs a repack (and how long the repack takes).
3. Deliverable `results/hopper/H4/SUMMARY.md`. Go threshold ≥1.10x on ≥5% of step time; if it passes, the manager will ask for an E2E A/B.

## Manager review of H1 (accepted) + H2 addition
Good, clean H1. Key finding: the SM90 default BF16 MoE is FlashInfer CuTe-DSL (`FlashInferCuteDSLSm90Experts`), and routing still goes through vLLM `fused_topk`. So add one arm to H2: **default backend + P1 only** (no `--moe-backend`, no P2) vs default, decode TPOT c1/c8/c32/c128 + GSM8K once. That is the cheapest thing that could land on Hopper. Keep the rest of H2 as specified (P2 vs default CuTe-DSL and vs Triton, plus the prefill profile if P2 loses).
