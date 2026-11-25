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

## Manager review of B1 (23:xxZ): accepted. Start B2 now.

Clean and complete. Useful details: the window is pure decode, launches per step are counted, attribution is by call count, the aux-stream caveat is applied to the gate, and Qwen3.6 cannot load at TP8. All of that is noted.
- **Main conclusion:** the same four BF16 projections run on cuBLAS on B300, at **18.0% of c1 kernel time** (14.6% without the off-critical-path gate). That is slightly more than on B200, where #69736 gave −5.7% TPOT at c4 and −4.7% at c8, with an unchanged GSM8K score. So B2 is the right next task.
- **Context update:** #69736 (open) creates `SM100_SKINNY_GEMM_PLANS` and registers `(10, 0)` only. On B300 it does nothing, which is why your base has no `(10, 0)` entry either.

### B2: as specified above, with these details
- **Shapes:** the 14 keys of `results/worker/W11/verified.json`. Sweep M ∈ {1, 2, 4, 8} with `results/manager/tools/sweep_skinny_generic.py`, one process per idle GPU.
- **Keep rule:** ≥ 10% over cuBLAS in the sweep, then a 4-seed × 3-round CUPTI verify (CUDA graph, cold L2); keep entries whose median is ≥ 1.10×.
- **Per entry, report:** the median speedup, the min-max over seeds, and the error vs fp32 for both cuBLAS and the plan.
- **Expect different winners from SM100.** The Qwen4Exp tables already differ between SM100 and SM103, so do not copy the SM100 configs. Do report, per shape, whether the SM100 config from W11 would also have qualified on SM103. That tells the manager whether a shared table is defensible.
- **Deliverable:** `results/b300/B2/{SUMMARY.md, proposed.json, verified.json}`, in the W11 format. Post after the sweep and again after the verify.
- **After B2 is accepted:** the manager will write a `(10, 3)` table on top of #69736's branch and ask you for the E2E serving A/B on B300. Plan for real weights then: Qwen3.6-35B-A3B-FP8 is about 35 GB. You can start the download now; fetch files by name or one pattern per call.

## Manager review of B2 (23:xxZ): accepted. Decision: share the table. → B3

Excellent: 49/49 verified, error equal to cuBLAS, and the three-arm verify answers the shared-table question directly. The SM100 configs give up at most ~5% of skinny time on two LM-head cells (≤ ~4 µs per step). **Decision: register `(10, 3)` to the same `SM100_SKINNY_GEMM_PLANS` as `(10, 0)`.** No separate SM103 table.

### B3: E2E serving A/B on B300 for the shared table
- **Code.** The table is already public, in #69736's branch `sm100-skinny-qwen35-moe` on `https://github.com/Johnsonms/vllm.git`, commit `81159c69d6`, on top of `e66bede56b`.
  - Fetch that commit into your tree.
  - Make one local edit: in `SKINNY_GEMM_PLANS_BY_CAPABILITY`, add `(10, 3): SM100_SKINNY_GEMM_PLANS,` next to the `(10, 0)` line.
  - Do not commit it anywhere public; describe it in words in your report.
- **Arms:** `base` = `e66bede56b`; `pr` = `81159c69d6` + that one-line edit. In each server log, print the loaded `skinny_gemm_plans.py` path and `(10, 3) in SKINNY_GEMM_PLANS_BY_CAPABILITY`.
- **Run:** Qwen3.6-35B-A3B-FP8, TP1, **real weights**.
  - `vllm serve ... --language-model-only --max-num-seqs 256 --max-model-len 8192 --gpu-memory-utilization 0.9 --no-enable-prefix-caching`
  - `VLLM_DISABLE_COMPILE_CACHE=1`; the AOT-load count must be 0.
  - `vllm bench serve` random 512/512, `--ignore-eos`, c = 1, 2, 4, 8, `--num-prompts max(4c, 16)`, `--num-warmups c`.
  - Order base, pr, base, pr, on an idle node.
- **GSM8K** in every arm: `tests/evals/gsm8k/gsm8k_eval.py --use-chat-completions --chat-template-kwargs '{"enable_thinking": false}' --max-tokens 1024 --temperature 0 --max-concurrency 64`.
- **Optional if time allows:** the same A/B at TP2 (2 GPUs), c1 and c8 only.
- **Deliverable:** `results/b300/B3/SUMMARY.md`, with per-run TPOT, the mean Δ, GSM8K, and the code-check lines.

## Manager review of B3 TP1 (23:5xZ): accepted. Very clear gain → B4 (verify-only)

−2.4 / −3.2 / **−11.5%** / −3.7% TPOT at c1/2/4/8 and GSM8K unchanged, with set 1 agreeing. That is better than B200 (−5.7% at c4). The c4 cuBLAS observation is worth a line in the PR; no need to profile it now. Let the optional TP2 A/B finish.

**Manager side:** commit `77b0ac244d` registers `(10, 3)` to the same `SM100_SKINNY_GEMM_PLANS`. It sits on top of #69736's commit, is local only, and waits for the owner's OK to push.

**A consequence to check:** the same table also receives the dense-Qwen3 entries (#69743, open) and the Qwen3-MoE attention/router/LM-head entries (next PR), which were verified on B200 only. With `(10, 3)` registered they apply on B300 too.

### B4: verify the B200 configs of those two sets on SM103 (no sweep)
- **Data:** `results/manager/B4/dense_qwen3_sm100.json` (41 shapes / 91 entries) and `results/manager/B4/qwen3_moe_sm100.json` (15 shapes / 49 entries). Format: `{"N,K": {"M": "SkinnyGemmConfig(...)"}}`.
- **Run:** your `verify_sm103.py` with two arms per cell, cuBLAS vs the given config. Use 4 seeds × 3 rounds, rotating the order, with CUPTI, CUDA graph and cold L2, one seed per idle GPU.
- **Report per cell:** median speedup, min-max over seeds, and the error vs fp32 for both arms. List every cell below 1.10×, and separately every cell below 1.00× (a regression on B300).
- **Deliverable:** `results/b300/B4/{SUMMARY.md, verify.jsonl, verify_summary.json}`.
- **Decision rule (manager):**
  - All cells ≥ 1.10×: the shared table stands as is.
  - Any cell < 1.00×: the manager either drops that entry or splits off an SM103 table. Report which shapes are affected.

## Manager update (11-25): B4 is still first; then B5 and B6, in that order

The owner reports this node idle. **Resume polling** (`setup/B300_START.md`, every ~10 min) and work the queue below. Post an ack first.

Context since B4 was assigned: #69736 now registers `(10, 3)` (pushed with your B2/B3 numbers in the PR text); Qwen3 MoE opened as #69773 (B200 only, no `(10, 3)`). The force-push of #69743 that brings the dense entries to `(10, 3)` waits for **B4**.

### B4 (unchanged, top priority)
As specified above. Report as soon as the verify is done; the decision rule is above.

### B5: SM103 TP8 all-reduce + RMSNorm fusion cap (B300 counterpart of #69543)
`FI_ALLREDUCE_FUSION_MAX_SIZE_MB[103][8]` is 4 MB (409 tokens at hidden 5120). On B200, raising the TP8 cap from 1 to 16 MB gave −9.6 / −6.5 / −5.9% TPOT at concurrency 128 / 192 / 256 on Qwen3-32B (#69543). Check whether SM103 also gains above 4 MB.
- **Model:** `Qwen/Qwen3-32B`, real weights (~65 GB; download by name or one pattern per call). TP8, all 8 GPUs, idle node.
- **Server:** `vllm serve Qwen/Qwen3-32B -tp 8 --max-num-seqs 512 --max-model-len 8192 --gpu-memory-utilization 0.9 --no-enable-prefix-caching`, `VLLM_DISABLE_COMPILE_CACHE=1` (AOT-load count 0).
- **Arms:** default (4 MB) vs `--compilation-config '{"pass_config":{"fi_allreduce_fusion_max_size_mb": X}}'` for X = 8, 16, 32. No code edits. Order: default, 8, 16, 32, then again (two rounds).
- **Bench:** random 512/512, `--ignore-eos`, concurrency 32, 96, 128, 192, 256, 384 (`--num-prompts 2c`, `--num-warmups` 8); median TPOT. Plus 1k-prefill TTFT at c4 as a control.
- **Checks:** GSM8K (same command as B3) for default and the best X; with `VLLM_LOGGING_LEVEL=DEBUG` once per arm, report the logged `Maximal number of tokens used by Flashinfer Allreduce Fusion` and the backend (`mnnvl` / `trtllm`).
- **Deliverable:** `results/b300/B5/SUMMARY.md` with the per-run table and your pick of X (or "keep 4 MB").

### B6: c4 decode profile on B300 (follow-up to your B3 observation)
In B3, base TPOT jumped 0.84 ms from c2 to c4 but only 0.58 ms from c4 to c8, and the skinny plan removed most of it. Find out which kernel causes the c4 step on SM103; it may hit other BF16 GEMMs at M = 4 too.
- Qwen3.6-35B-A3B-FP8 TP1, real weights, B1 profiling method, at c2, c4 and c8, for `e66bede56b` (base) and base + the `(10, 3)` table (your B3 arm).
- Report the top kernels per step at each concurrency and which ones change between c2 → c4 → c8 (time per call and calls per step). If it is cuBLAS, record the kernel names (`nvjet*` variant, split-K) at M = 2/4/8 for the affected shapes.
- **Deliverable:** `results/b300/B6/SUMMARY.md` (kernel tables < 5 MB).

## Manager review of B4 (11-25): accepted. Shared table stands; continue with B5, then B6.
140 cells, 0 below 1.00×, error equal to cuBLAS: the dense-Qwen3 and Qwen3-MoE entries stay in the shared table for `(10, 3)`. The 6 cells at 1.05–1.10× are kept (still faster, stable over seeds); the PR text will say so.

## Manager review of B5 (11-25): accepted — keep 4 MB. Go to B6.
Clean negative result with the right mechanism (409 tokens already covers every decode batch to c384). The 32 MB prefill TTFT signal (−4.7%, 2–3 ms at c4 × 1k) is noted but not worth a PR on its own; no follow-up for now. Start B6.

## Manager review of B6 (11-25): accepted — excellent. → B7

The busy-union / exposed-time analysis answers the question: the c4 step is MoE expert growth (both arms), and base's extra cost is the shared expert starving on the aux stream behind the routed-MoE GEMMs. Your follow-up is the next task. The worker runs the same experiment on B200 as W13, so keep the method identical.

### B7: can the shared expert get the SMs earlier? (local edits only; nothing pushed)
Base for both arms: `e66bede56b` **plus the `(10, 3)` table line** (your B3 `pr` arm), so the skinny plans are on in both. Qwen3.6-35B-A3B-FP8, TP1, real weights, B3 serve flags, `VLLM_DISABLE_COMPILE_CACHE=1`.

1. **Does stream priority survive CUDA-graph replay?** `vllm/utils/torch_utils.py:aux_stream()` creates `torch.cuda.Stream()` with the default priority.
   - Variant P: `torch.cuda.Stream(priority=-1)` (highest).
   - First check whether the priority reaches the captured graph. Read the kernel-node launch attributes of an instantiated graph if you can. Otherwise take a profile of P vs base at c8 and see whether the shared expert's `act_and_mul` start offset (your `layer_view.py`) moves.
   - If node priority needs `cudaGraphInstantiateFlagUseNodePriority` and torch does not pass it, report that, and say where torch instantiates.
2. **Variant O, order:** enqueue the shared expert on the aux stream before the router GEMM instead of after, if the call order allows it. Read `Qwen3NextSparseMoeBlock.forward` and the runner path (`runner/shared_experts.py:maybe_forward_async`), and report where the router runs relative to `maybe_forward_async`. Only prototype if it is a small local change.
3. **Paired A/B** for whichever of P / O changes the timeline:
   - Both servers up at once on two GPUs. Alternate short bursts (5 × c prompts, 6 rounds per c, order flipped each round), then swap the GPUs and repeat. Report the mean of the two passes. This is the manager's `e2e_paired.sh` design: the worker showed sequential A/Bs can fake 6–8%.
   - Concurrency c = 1, 2, 4, 8, 16.
   - Plus GSM8K, 2 runs per arm. Output should be bit-identical for P; check greedy tokens on 4 prompts.
   - **Gotcha:** if your venv has an editable vLLM install, `vllm serve` imports *that* tree, not the cwd. Set `PYTHONPATH=<tree>` per server, and log the loaded file path and the presence of your edit in each server log.
4. **Deliverable:** `results/b300/B7/SUMMARY.md`: the priority finding, the order finding, the A/B table, and a layer timeline (c4, c8) for the winning variant vs base. Describe edits in words only.

## Manager review of B7 (11-25): accepted — not a blanket change. → B8

Clean answer, with the sign the same in every pair. P stays parked until W13 shows whether B200 follows the same pattern.

### B8: independent check of #69778 (SiLU-and-mul + block-FP8 quant fusion) on B300, with accuracy first
#69778 is public: `https://github.com/Johnsonms/vllm.git` branch `fp8-block-silu-quant-fusion`, commit `c5ef00f633` on main `fb159244e6`. Base = `fb159244e6`. On B200, TPOT was −3.8 / −2.5 / −2.4% at c1/4/8. **The open question is accuracy:** GSM8K over 10 runs per arm was 0.9636 (main) vs 0.9595 (PR). Per-question flips look like near-tie noise, but the owner wants it settled before anyone relies on it.
1. **Confirm the fusion fires on SM103.** Log the kernel class and `_input_quant_key` of one shared-expert `down_proj`. Set `PYTHONPATH` per tree.
2. **Paired A/B:** two GPUs, both servers up, bursts alternating, GPUs swapped (your B7 design). c = 1, 2, 4, 8.
3. **GSM8K, 8 runs per arm** (same command as B3), interleaved base/PR. Both servers up at once is fine. **Save per-question correctness.** `gsm8k_eval.py` only prints the total. Wrap `_score_gsm8k` to dump `[int(pred == label) ...]` per run to JSON; no vLLM edit is needed.
   - Report: per-run accuracy, means, and the right→wrong / wrong→right flip counts base vs PR.
   - Also report the same flips between base runs (the A/A noise floor).
4. **Deliverable:** `results/b300/B8/SUMMARY.md`.

## Manager note (11-25): B8 cancelled
The owner has given this node another job. Skip B8; the manager is covering #69778's accuracy question on B200.
