# Tasks for the B300 node (role `b300`)

The manager writes this file; the B300 node only reads it. Report in `STATUS_B300.md` and `results/b300/B<n>/`.
Read `setup/B300_START.md` first.

## B1: environment and SM103 baseline survey  [assigned 2026-11-24]

1. **Set up** as in `setup/B300_START.md` §2.
   - Record: GPU model, count, driver, CUDA, torch, FlashInfer and Triton versions, and `torch.cuda.get_device_capability()` (expect `(10, 3)`).
   - Do not record hostnames or IPs.
2. **Sanity check:** `pytest tests/kernels/test_bf16_skinny_gemm.py -q` (report the pass and skip counts).
3. **SM103 coverage check.** Read and summarize, without timing anything yet:
   - `QWEN4_EXP_GEMM_PLANS_BY_CAPABILITY[(10, 3)]`: which shapes and TPs.
   - `SKINNY_GEMM_PLANS_BY_CAPABILITY`: confirm there is no `(10, 3)` entry.
   - `FI_ALLREDUCE_FUSION_MAX_SIZE_MB[103]`, plus whether this node's FlashInfer all-reduce resolves to `mnnvl` or `trtllm` at TP8. The serve log says which.
4. **Decode profile, Qwen3.6-35B-A3B-FP8, TP1, dummy weights, on the idle node.**
   - Run `vllm serve ... --load-format dummy --language-model-only` with the torch profiler (`--profiler-config '{"profiler":"torch","torch_profiler_dir":"..."}'`). Then do a short `vllm bench serve --profile` at c1 and c8: input 512, output 128, `--num-prompts C`.
   - Report the top 20 GPU kernels by time share at c1 and c8, and the median TPOT.
   - Mark which kernels are cuBLAS BF16 GEMMs (`nvjet*`, `splitKreduce`, `dot_kernel`, `reduce_1Block_kernel`, `cutlass_80_*`). On B200 these were ~16% at c1.
5. **Deliverable:** `results/b300/B1/SUMMARY.md` (and the kernel tables as JSON or CSV, each < 5 MB). Post when done.

## B2: SM103 generic skinny plans for Qwen3.5/3.6 MoE  [queued; start after the manager reviews B1]

Port #69736 to B300. Use the same 14 shapes from `results/worker/W11/verified.json` (Qwen3.6-35B-A3B TP1/2/4, Qwen3.5-397B-A17B TP4/8; GDN `in_proj_ba`, shared-expert gate, router, LM head).
- **Sweep:** M ∈ {1, 2, 4, 8}; keep ≥ 10% over cuBLAS.
- **Verify:** 4 seeds × 3 interleaved rounds, CUPTI, CUDA graph, cold L2. Keep entries whose median is ≥ 1.10×. Record the error vs an fp32 reference for both cuBLAS and the plan.
- **Deliverable:** `results/b300/B2/{SUMMARY.md, proposed.json, verified.json}`, in the same format as W11.
  - Do not edit vLLM files. The manager writes the table and asks you for the E2E A/B.

## Later candidates (not assigned yet; the manager picks after B1/B2)

- Qwen4Exp SM103 TP2/TP8 plans (the SM100 equivalent merged as #69529). The Qwen3.8-Flash-Next-FP8 checkpoint is needed.
- An SM103 TP8 AR+RMSNorm fusion cap sweep (4 MB today), as in #69543.
- Anything B300-specific that B1's profile shows.
