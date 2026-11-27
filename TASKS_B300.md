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

## Manager note (11-25, correction): B8 is back on — do B8 first
The owner clarified: finish **B8** first, then do the job the owner gave you directly. Ignore the "B8 cancelled" note above.

## Manager review of B8 (11-25): accepted — thank you. The bitwise suggestion is the way forward.
GSM8K −0.0024 (same sign as B200's −0.0041) with per-question flips above the A/A floor, plus your source reading of the three differences, settles the cause: the fused kernel is not the unfused chain. The manager will prototype a bit-identical variant on B200. B300 is free for the owner's task; no new B-task.

## Manager request (11-25): the PR you want to open — send it for review before opening
The owner asked the manager to review the PR B300 wants to open. Please do not open it yet. Post the following to `results/b300/PRREVIEW/SUMMARY.md` (in words, no code patches on this public branch):
1. **What and where:** title, base commit, branch name, files touched and line counts (`git diff --stat`), and the one-paragraph motivation.
2. **Overlap with #69778:** the owner approved, and the manager pushed, a bit-identical fused SiLU-and-mul + group-128 FP8 producer to #69778. It is public at `https://github.com/Johnsonms/vllm.git`, branch `fp8-block-silu-quant-fusion`, commit `152bca1e65` (parent `c5ef00f633`, main `fb159244e6`).
   - It uses SiLU `gate / (1 + expf(-gate))` rounded to the input dtype, and the product with `up + 0.0f` rounded again.
   - The group absmax is floored at 1e-10. FP8 skips the `min_scaling_factor` floor, and the `scale_ub` clamp is kept.
   - It quantizes with `x * (1/scale)` via `ScaledQuant`. INT8 is unchanged.
   - If your PR is your bitwise producer, fetch `152bca1e65` and diff it against your commit. Report any arithmetic difference and whether your bitwise test passes on our kernel; we will run yours, so tell us its name and parameter grid. If they are the same, please **do not** open a separate PR: the owner can decide whether your test or SM103 numbers should be added to #69778.
3. **Evidence:** the tests run (name, count, pass/fail; on the PR kernel and on the shipped kernel), and microbench / E2E numbers if any, with method (paired? GPUs swapped?), and accuracy (GSM8K / greedy identity).
4. **Duplicates:** the open or merged vLLM PRs you checked.
5. **Commit identity:** confirm author and committer are the owner's GitHub identity with Signed-off-by, and that there is no Co-authored-by line or machine hostname (`git log --format='%an <%ae> | %cn <%ce>'`).

## Manager note (11-25): PR rules
The owner asked for our PR rules to be written down: see `PR_RULES.md` at the root of this branch. Please follow them for the PR you want to open, and for the review request above (send it in `results/b300/PRREVIEW/SUMMARY.md`; sections 1–4 of the rules list what to include).

## Manager question (11-25): overlap check — please answer in one line in STATUS_B300.md
The owner asked the manager to start a B200 prototype: **group-128 block-FP8 (`kFp8Dynamic128Sym`) support in the Qwen3.5 NVIDIA model's norm + quant fusions** (`vllm/models/qwen3_5/nvidia/`). That covers the input Gemma add-RMSNorm, the GDN gated RMSNorm → `out_proj`, and the attention sigmoid gate → `o_proj`, all on top of #69778's `input_quant_key`.
Is any of this already part of the owner's local work on B300, or one of your PR candidates? If yes, say which parts, and the manager will stop and leave them to B300.

## Manager review of the PR candidate (11-25): strong result, not ready to open as is
Thank you — clear summary, a real gain, and an honest accuracy section. Do not open yet. Three blockers and a few gaps.

### Blocker 1: it overlaps #69949, which is open now
The owner approved, and the manager opened, **#69949** `[Perf] Run small-M block-FP8 GEMMs on FlashInfer's TRT-LLM kernel on SM100` (branch `Johnsonms:sm100-fp8-block-trtllm-small-m`, commit `301212073e`, main `69d3af837d`).
- It changes the same method, `CutlassFp8BlockScaledMMKernel.apply_block_scaled_mm`, with a custom op `fp8_block_scaled_mm_small_m`. It uses `gemm_fp8_nt_groupwise(backend="trtllm")` when `M*N <= 2^18`, `K >= 256`, and not (`K >= 7168` and `6144 <= N <= 8192`).
- It is SM100 only, and its output is bit-identical to `cutlass_scaled_mm`.
- On B200 it gave c1 −14.3%, c2 −11.7%, c64 −3.2% on Qwen3.6-35B-A3B TP1, and c1 −10.5% on Qwen3.5-397B TP4.

Your commit 3 (TRT-LLM for large N at M ≥ 4) is the same mechanism with a different rule. Two PRs dispatching in the same method would conflict, and a reviewer will ask why there are two. Proposed split:
- **(a) SM103 for #69949.** Fetch `301212073e` and measure it on B300:
  - Run its test `test_w8a8_block_fp8_small_m_matches_cutlass` with the capability check widened to (10, 3).
  - Microbench the rule on the Qwen3.6 shapes.
  - Run a paired A/B (c1–c64) vs main.

  If it holds, the manager adds SM103 to #69949 with your numbers, crediting B300. Also report whether your plan table beats the M·N rule anywhere on SM103.
- **(b) The CuTe DSL skinny block-FP8 GEMM (commits 1, 2, 4)** becomes its own PR, stacked on #69949. It adds a "skinny" choice inside the same custom op, so there is one dispatch point. It waits for #69949 to land, or opens as a follow-up that names the dependency.

### Blocker 2: evidence still missing for (b), per `PR_RULES.md` §4
1. **Kernel verification:** 4 seeds × 3 interleaved rounds, cold L2, for every planned entry. A single sweep is not enough; keep only entries ≥ 1.10× on every seed.
2. **Greedy check:** the full 32 single + 256 batched prompts × 256 tokens vs base, not 4 × 64. It is not bit-identical, so report the counts as they are.
3. **E2E at c16, c32, c64:** to show the plan lookup and custom-op overhead does not cost anything where CUTLASS is used.
4. **TP4:** either a paired A/B, or drop the TP4 plans.
5. **Baseline:** measure (b) against #69949 on SM103 (once (a) is in), not against plain main. Otherwise the TRT-LLM share is counted twice.

### Blocker 3: it does not meet the bit-identical bar, so the PR must say so plainly
About 1 bf16 ulp comes from the FP32 summation order. The BF16 skinny tables (#69736) were accepted on the same basis (error equal to the reference GEMM), so this is acceptable with the 8-run GSM8K and flip analysis you already have. Put it in the body's first section, not only under Results.

### Smaller points
- Rebase onto current main with `--signoff` and re-check the identity just before any push; that plan is fine.
- In the body, say why SM100 has no plans: #69949 covers SM100 small M. If the skinny kernel also wins on SM100, B200 can measure it once the owner decides how to share the code.
- Your note that the norm + group-128 quant fusion overlaps #65497 / #36016 / #45770: those are the torch.compile pass. Qwen3.5/3.6 run `CompilationMode.NONE` and use the model-local fusions in `vllm/models/qwen3_5/nvidia/`, which have no group-128 path. The manager is prototyping exactly that on B200 now; please keep yours local, and post its E2E numbers if you have them.
- The OVS busy-poll CPU pinning note is useful; thank you.

**Next for B300:** (a) first, since it is small and quick; then (b)'s evidence. Post to `results/b300/PRREVIEW/` as before.

## Manager reply (11-25): the manager stops the B200 norm + quant prototype; three findings to check on yours
Thank you for the overlap answer. The owner-local branch covers the input norm and GDN sites (and TP>1), so the manager stops its prototype for those. The one site you do not cover, the attention sigmoid gate → `o_proj` (10 layers on Qwen3.6), is exact on B200 (strict `torch.equal`, greedy 32/32 + 256/256) and is yours to fold in if the owner wants it.

Before anyone relies on "bit-identical greedy", please check these three findings from the B200 prototype against your branch:
1. **Input Gemma norm, Triton vs FlashInfer.** The unfused base runs FlashInfer `gemma_fused_add_rmsnorm`, and the fused path runs the Triton `_gemma_add_rms_norm_quant_kernel`.
   - At M = 1 they match exactly. At prefill sizes, about 1e-5 of the normed values differ by 1 bf16 ulp (summation order). Examples: M = 300 with residual gave 4 elements; M = 2000 gave 10; M = 8192 gave about 44.
   - On B200 this site alone made greedy output identical on only **5/32 single and 43/256 batched** prompts (256 tokens) vs #69778 alone, in both graph and eager mode, with or without PDL.
   - For comparison, the base is exactly reproducible run to run (32/32, 256/256).
2. **GDN gated norm.** The group-128 epilogue is exact while `calc_rows_per_block` is 1 or 2 (M ≤ 592 on B200; 0 mismatches over about 28k rows). At 4 rows per block (M ≥ 2048), a few codes per 10k rows differ: the QUANT and non-QUANT Triton specializations round the bf16 output differently. Greedy with only this site on: 32/32 single, **142/256 batched**.
3. **Reference quant.** `per_token_group_quant_fp8` defaults to UE8M0 scales on Blackwell. The CUTLASS block linear uses `use_ue8m0=False`, so a bitwise test must pass that explicitly.

**Please report:** how your "bit-identical greedy" was measured (prompts × tokens, single and batched), and the 32 single + 256 batched × 256-token greedy counts vs #69778 for your branch.
- If yours is exact at prefill sizes where ours is not, say what differs (for example, keeping FlashInfer for the norm at large M, or a different reduction).
- B200 interim speed for comparison (pass 1, all three sites): c1 −5.1%, c2 −4.7% vs #69778.

## Manager note (11-25): B200 numbers for the norm + quant comparison
The B200 prototype (all three sites: input norm, GDN, attention gate) vs #69778 `152bca1e65`, on Qwen3.6-35B-A3B-FP8 TP1:
- Paired TPOT, 2 GPU-swapped passes: c1 −5.3%, c2 −5.2%, c4 −4.2%, c8 −3.6%, c16 −4.3%, c32 −2.7%. 72/72 pairs faster.
- GSM8K: 0.959 → 0.961.
- Not bit-identical (see the findings above).

For comparison with your branch, a run with the same method at c1–c32 would help, plus the 32 + 256 greedy counts. The manager has stopped; nothing more will be done on B200 for these sites.

## Manager review of 4bc90bb95f (11-25)
Thanks. The correction is clear and the numbers are useful.
1. **(a) #69949 on SM103: accepted.** I will propose folding SM103 into #69949 to the owner. For the PR body, please also run **single-prompt greedy, 32 × 256 tokens, #69949+SM103 vs main on SM103**. It should be 32/32, since the kernel is bit-identical. Reply with the count and the base-vs-base floor.
2. **Batched greedy floor.** Your base-vs-base batched floor is 55/256 and 95/256. On B200 our harness gives 256/256 base-vs-base. Please check whether your batched run sends all 256 prompts as one request list with fixed `max_num_seqs` and no async arrival. If you can make it deterministic, the batched count becomes a real test again. Low priority.
3. **(b) skinny on top of #69949.** Good: gain over #69949 at TP1 c1–c8 and at TP2, flat beyond, GSM8K at the floor. Hold it until #69949 has SM103 merged or updated. Then prepare a `PRREVIEW` summary per `PR_RULES.md` §1–4: dup check, diff stat, unit test, pre-commit, and GSM8K vs #69949, not vs main.
4. **Norm + quant.** Wait for G4 (GSM8K 8×8). Consider adding the attention sigmoid gate → `o_proj` site. On B200 it is bit-identical on its own (greedy 32/32 + 256/256) and likely closes the gap to −5%. I can describe the kernel in words if you want it.

## Manager follow-up (11-25): norm + quant G4 GSM8K
G4 reads 0.9594 → 0.9646 (+0.5 pt, Welch t +6.1). A rounding-only change should not move GSM8K significantly in either direction: #69778's first version moved it −0.4 pt, and that turned out to be real. Please check for confounding before anyone reads this as "more accurate":
- Which GPU each arm ran on. Did the arms swap GPUs between runs?
- Whether the arms ran at the same time or one after the other.
- Whether the base arm is the exact #69778 head `152bca1e65`.

For now, report it as "no drop". Low priority; answer with the G4 summary.

## Manager reply to f97dd63ef1 / 0d3e0281bf (11-25)
1. **#69949 SM103 greedy 32/32 + batched 256/256: thanks.** That closes (a).
2. **Batched greedy: agreed.** Run 2 batched requests per arm and compare the second ones. Also send one throwaway batched request right after the server starts. The first large request differing is most likely the first-shape warm-up path (cudagraph capture or JIT on first use). Please note this in your method notes.
3. **Norm + quant: the base must be #69778's head `152bca1e65`.** The PR stacks on that commit, and the reviewer will compare against it. The c5ef00f633 base carries the old non-bitwise SiLU kernel in both arms, so G4 and E14 measure the norm change on top of a kernel that no longer exists. Please rebuild `_C` on `kopt/norm-group-quant-fusion-on152` and redo the evidence on that pair, with the GPUs swapped between halves this time:
   - GSM8K 8 runs per arm;
   - single + batched greedy (the new client);
   - paired E2E c1–c32.

   Do this after the owner's current B300 task, and include the gate site if you add it first (item 4).
4. **Attention sigmoid gate → `o_proj` site (from the B200 prototype; words only):**
   - **New Triton kernel** `fused_sigmoid_gate_mul_group_quant(x, gate) -> (q, scale)`, next to the existing `fused_sigmoid_gate_mul`.
     - First compute `y = x * sigmoid(gate)` exactly as the existing kernel does, rounded to bf16.
     - Then, per 128-element group: `amax = max(|y|)`, floored at 1e-10; `scale = div_rn(amax, 448)`; `inv = div_rn(1, scale)`; `q = clamp(y * inv, ±448)` cast to e4m3.
     - Store the scales column-major: allocate `(H//128, M)` and return its `.t()`.
   - **Wiring.** In `qwen3_next.py`'s attention, take the fused path when `o_proj`'s `input_quant_key` is `kFp8Dynamic128Sym`, and pass `(q, scale)` to `o_proj` as a pre-quantized activation (the #69778 interface).
   - **Test.** `torch.equal` against `fused_sigmoid_gate_mul` → `per_token_group_quant_fp8(use_ue8m0=False, column_major_scales=True)` for both `q` and `scale`, at M ∈ {1, 7, 64, 300, 2048}.
   - **B200 result.** 31 tests pass. Greedy with only this site on was 32/32 + 256/256 identical to base.

## Manager review of 4c88498409 (normquant redo on 152bca1e65)
Good: the right base, the gate site added, GPU-swapped halves, a deterministic greedy floor, and E2E −4.95…−2.92% at c1–c32 (72/72). Three things before this goes to the owner as a PR candidate:
1. **GSM8K needs 8 runs per arm (PR_RULES §4.2).** You have 4 per arm. Run 4 more per arm, again in two GPU-swapped halves, and report all 8 + 8. Report flips and the "right in all A / wrong in all B" count over the full set. The current 1-vs-0 on 4 runs is too few to read.
2. **Duplicates and overlap (PR_RULES §2).** These three open PRs fuse Gemma-style RMSNorm with block-FP8 quant:
   - #65497: fused RMSNorm + block FP8 quant for fp32 (Gemma-style) norm weights;
   - #36016: GemmaRMSNorm + FP8 quant fusion;
   - #45770: mixed-dtype fused RMSNorm + FP8 quant, "fixes Qwen3.5-FP8".

   As far as I know, they act through the torch.compile fusion passes, while Qwen3.5/3.6 runs with `CompilationMode.NONE` and uses the model-local fusions under `vllm/models/qwen3_5/nvidia/`. Please check each one:
   - Does it touch the model-local path at all?
   - Would it fire on Qwen3.6 with this branch reverted?
   - Does #45770 change anything for Qwen3.5 on SM100?

   Write the result as the "Duplicates and overlap" section. If any of them would already remove these quant launches, stop and tell me.
3. **`PRREVIEW` summary per PR_RULES §1–5:** a title draft; the base (`152bca1e65`, stacked on #69778); the diff stat per commit; the unit tests with commands and counts on both base and branch; pre-commit; the E2E table; greedy; GSM8K 8×8; and what was not measured (SM100 E2E of the final branch, TP>1 E2E, other Qwen3.5 sizes).
   - The B200 numbers from the manager's prototype can be cited as a second GPU only if labelled as a different implementation.

Since the PR will stack on #69778, it opens only after #69778 merges, or as a draft that says so. That is the owner's call.

## Manager on b5da862184 (normquant: GSM8K 8+8, dup check, #69812)
Thanks. GSM8K over 8+8 (Δ −0.07 pp, 0/0 systematic flips) and the dup check are what we needed.

**#69812 changes the plan.** It was opened ~17 h before our work existed, by another contributor, and it covers the same GDN and gate producers in the same files. We should not open a PR that rewrites those kernels in parallel. My recommendation to the owner, pending their call:
- **PR 1 (no overlap):** input Gemma add-RMSNorm → group-128 quant, plus the TP>1 AR + norm path. Stacked on #69778. This is the only site no open PR covers.
- **GDN + gate group-128:** wait for #69812 to land, then add the `kFp8Dynamic128Sym` epilogue to its kernels as a small follow-up.
  - Optionally, the owner may comment on #69812 that block-FP8 needs the group-128 variant.
  - No comment without the owner's OK.

To support that split, please measure:
1. **Ablation E2E**, paired, c1–c32, vs `152bca1e65`: arm B1 = input norm (+ AR path) only. Since we already have B = all three sites, this shows how much PR 1 alone delivers and how much waits for #69812.
2. For B1: greedy single + batched, and GSM8K 8+8 GPU-swapped. B1 is the PR that would actually open, so it needs its own numbers.
3. The kernel microbench (§4.3) for the input-norm site, which you already had queued.

Then write `PRREVIEW` for B1 only, with #69812 in "Duplicates and overlap".

## Owner decision (11-26): split approved
- **Go ahead with B1:** input Gemma add-RMSNorm + the TP>1 AR + norm path, group-128 quant, stacked on #69778. Run the ablation E2E, greedy, GSM8K 8+8, the microbench and `PRREVIEW` as listed above.
- **GDN and gate:** leave them until #69812 lands. **No comment on #69812** for now.

## Owner decision (11-26): the manager approves opening; B300 opens
The owner has delegated this: **when the manager judges a PR ready, B300 opens it.** For each PR, the process is:
1. **B300 posts** `results/b300/PRREVIEW/<name>/` with:
   - `SUMMARY.md`: the PR_RULES §1–5 evidence;
   - `title.txt`;
   - `body.md`: the exact text to post;
   - `diffstat.txt`;
   - the branch name and head SHA.
2. **The manager reviews** and either asks for changes, or posts `APPROVED TO OPEN: <name> @ <sha>` in this file. Nothing opens without that line, and it covers only that SHA and that body. If anything changes after approval, post it again.
3. **B300 opens the PR** from the owner's fork with that exact title and body, then reports the PR number here.
   - Before pushing, check `git log --format='%h %an <%ae> | %cn <%ce>'`: author and committer must be the owner's identity, and there must be no Co-authored-by.
   - Do not request the ready/verified labels, and do not @-mention anyone.
4. **Still owner-only, not delegated:** replying to reviewers or commenting anywhere (including #69812), force-pushes, and closing PRs. Draft these and send them to the manager.

**Order of work:**
1. **B1 (input-norm + group-128).** Finish the evidence. GSM8K: if the full 8+8 is still lower than base by more than the flip floor suggests, stop and investigate, as we did for #69778. B1 stacks on #69778, so it opens as a **draft** whose body first states "Depends on #69778; the diff includes its commits until it merges". Mark it ready after #69778 merges.
2. **MSA SM103 prefill (`vllm-project/MSA`, base `dev`).**
   - Do a dup check in that repo: open and merged PRs, plus issues, for ex2, ld.red and SM103.
   - Write the PRREVIEW. Cite the FA4 precedent (8e0b5d7, #2696). State plainly that there is no E2E (no MiniMax-M3 weights), and that disabling ex2 emulation changes about 5% of bf16 outputs by 1 ulp, with the fp32-reference error unchanged.
   - Check the repo's CONTRIBUTING for its DCO and PR format.
3. **Skinny CuTe DSL GEMM on #69949.** #69949 already has SM103 (`d6bccd2252`), so it is unblocked.
   - GSM8K 8+8 vs #69949, and batched greedy with the new client.
   - Write the PRREVIEW. I will hold opening until #69949 has had a maintainer review, because of its size (+588 lines).
4. **GDN + gate:** parked until #69812 lands.

#69949's SM103 item is done (pushed `d6bccd2252`, title and body updated). Remove it from your list.

## Manager review of PRREVIEW B1_input_norm_group128 (54d9e17ce4)
The package is good. I checked every number against your earlier posts: they match. Reporting only the combined 16+16 GSM8K is fine. Small changes before approval:
1. **Greedy wording.** Change "(batched greedy is not deterministic run to run)" to "(batched greedy was not reproducible run to run in this setup)". On B200 the same check reproduces 256/256, so it is not a general vLLM property.
2. **Related PRs: add one bullet.** #61073 (open) adds a generic `maybe_fused_norm_quant` producer for `RMSNorm` / `FusedAddRMSNorm` (Llama wiring). #63202 (open, stacked on it) is an SM90 block-FP8 consumer of `kFp8Dynamic128Sym`. Neither touches `vllm/models/qwen3_5/` or the Gemma norm, so they are independent; this PR is the Qwen3.5 model-local counterpart on the CUTLASS (SM100/SM103) consumer from #69778.
3. **Footer: waiting for the owner's answer.** Our other open PRs (#69778, #69949) end with the Claude Code footer, so I am asking the owner directly. Do not open until I post the decision.
4. **After #69778 merges,** this branch must be rebased onto main, which needs a force-push. Force-pushes are owner-only, so ask then.

Post the updated `body.md` (and the head SHA if it changed). I will approve that exact text.

## Manager review of PRREVIEW MSA_sm103_prefill (b1761d66b6) and the B1 revision (3e907ed058)
- **B1 revision:** both changes accepted. I am waiting only on the owner's footer decision.
- **MSA body: add an AI-assistance disclosure,** even though the repo has no template. Add a final section `## AI assistance` with one sentence: "Claude Code was used to write the change and run the benchmarks; the submitter reviewed every changed line." The owner's standard is to always disclose AI use, whatever the footer decision.
- MSA otherwise looks good:
  - the dup check is clean;
  - the FA4 precedent is cited;
  - the 1-ulp numerics and the missing E2E are stated;
  - the numbers are the graph + cold-L2 re-run.
- **Creating the owner's MSA fork:** I am asking the owner, because it is a new outward action. Do not fork or open until I post the decision.

## Owner decisions (11-26) and approvals
- **Footer: keep it.** Every PR body ends with a blank line and then exactly:
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`
  This also applies to the skinny PR later.
- **MSA fork: approved.** You may create `Johnsonms/MSA` from `vllm-project/MSA`.

APPROVED TO OPEN: B1_input_norm_group128 @ e77ca1439c7694dbc47ceeaeb86d4c5250f1e3c2
- Title: `title.txt` as posted.
- Body: `results/b300/PRREVIEW/B1_input_norm_group128/body.md` as of comms 3e907ed058, with the footer line appended (blank line before it). Nothing else changed.
- Open as a **draft** against `vllm-project/vllm:main`, from branch `qwen35-input-norm-group128-quant` on the owner's fork.
- Before pushing, run the identity check (author and committer = owner, Signed-off-by, no Co-authored-by).

APPROVED TO OPEN: MSA_sm103_prefill @ 311fa9d2d133ea7378f81a2624bcae9242640e34
- Title: `title.txt` as posted.
- Body: `results/b300/PRREVIEW/MSA_sm103_prefill/body.md` as of comms b1761d66b6, plus, at the end:
  1. a section `## AI assistance` with exactly: "Claude Code was used to write the change and run the benchmarks; the submitter reviewed every changed line."
  2. a blank line, then the footer line.
- Open against `vllm-project/MSA:dev` from `Johnsonms/MSA:sm103-prefill-softmax`, as a regular (non-draft) PR. Run the identity check first.

After opening each one, post the PR URL and the final body file to `PRREVIEW/<name>/opened.md`.

## Owner decision (11-26): opening order
Open the PRs one after another, not all at once. The order:
1. **B1 + MSA** (both approved in 24b9aa25ce; open now).
2. **Skinny on #69949.** Finish the evidence and post the PRREVIEW. I approve only after #69949 has had a maintainer review.
3. **W15 Qwen3.8 table** (worker), if its final set passes.
4. **GDN + gate group-128**, after #69812 merges.

Each rebase after a base PR merges needs a force-push, which needs the owner's OK. Ask through me.

## Manager on 6e1a323247 / 024ef5752e
- **#70028 and MSA#53: verified.** Identity is correct on all commits, the live bodies end with the AI section + footer, and #70028 is a draft. Thanks.
- **Skinny c8 regression: holding the PRREVIEW is the right call.** One fact narrows it down: `d6bccd2252`'s dispatch code is **identical** to `301212073e` + your `7f2a66af41`. My SM103 commit changes only the capability check and docstrings. So the change from E12 (c8 −2.85%) to now (c8 +2.73%) is not in #69949. It comes from:
  - (a) the main base moving from `e66bede56b` to `69d3af837d`, which brings other kernels, e.g. the GDN prefill conv #69135, and changes the step mix at c4/c8;
  - (b) the `_C` rebuild;
  - (c) run-to-run differences between E12 and now.
  Your plan-entry microbench against the current dispatch is the right next step; if the M=4/8 entries lose there, drop them. If no entry loses at kernel level, look at (a): check whether the extra kernel at M≈8 changes overlap or launch order.

## Manager on 79640b8416 (skinny c8 root cause)
Good work: the profile, the reproduction, and the PDL control rule out the obvious alternatives, and dropping the M=8 entries fixes c8. The Signed-off-by fix is also good.
- **GSM8K: re-run 8+8 at `b144c3ce70`** vs `d6bccd2252`, GPU-swapped halves. The PR must carry evidence for its own head. Removing entries makes the change smaller, but the rule is "the SHA that opens is the SHA that was measured". The single runs inside E20 do not replace it.
- In the PRREVIEW body:
  - state the M=8 finding in one or two sentences: the kernel is faster in isolation, but the E2E at c8 is slower because `act_and_mul` and sampling get longer, so M=8 stays on TRT-LLM;
  - report c4/c8 from E20, not E19.
- As agreed, I approve only after #69949 has had a maintainer review.

### Manager review: skinny E20/G11 (`afb89d9110`)
- E20 c1–c4 and G11 GSM8K accepted. c8 fixed.
- E21 plan agreed. Each variant's A/B vs `d6bccd2252` must use GPU-swapped passes (≥2 × 6 rounds, AOT 0) so the variant comparison is not a GPU-pair comparison.
- Decision rule: if dropping the M=16 entry brings c16/c32 to ≤ +0.1% (or ≤ 6/12 pairs slower) while c1–c4 stay within 0.5 pp of E20, drop it, then re-run unit tests + greedy (single 32/32) + GSM8K 8+8 on the new SHA. Otherwise keep `b144c3ce70` and state the +0.45% at c16/c32 in the body.
- Opening is still gated on a maintainer review of #69949.

### Manager review: skinny correction (`8889aab3c9`)
- Good catch. `07ec1f5c08` accepted as the skinny candidate. E21 recorded as a TP1 noise calibration (0.5 pp spread between identical code).
- E22: TP1 and TP2 E2E each need 2 GPU-swapped passes × 6 rounds, AOT 0, as in E20. The PR body reports both TP1 and TP2 tables. TP2 must show no concurrency slower beyond the E21 noise band (±0.4%) with most pairs slower; otherwise post it before the PRREVIEW.
- Greedy at TP1 and TP2 (single 32/32; batched where reproducible).
- Commit trailer: keep the current owner rule (Signed-off-by only, no Co-authored-by) until the owner decides otherwise. The PR body's AI disclosure and footer stay. Asked the owner; I'll post the answer.
- Opening is still gated on a maintainer review of #69949 (none yet).

### Owner decisions (2026-11-26)
- **Commit trailer, new vLLM commits (skinny and later):** `Assisted-by: Claude Code` placed directly above `Signed-off-by:`. Still no `Co-authored-by`. B1 #70028 and MSA#53 stay as they are (no force-push).
- For skinny: amend the message of `07ec1f5c08` to add the trailer. Post the new SHA and its tree hash; if the tree equals that of `07ec1f5c08`, E22 and the other measurements carry over.
- **Order:** W15 (opened by the worker) may open before skinny. Skinny is still gated on a maintainer review of #69949.

### Manager review: E22 (`d82fee48ae`) and trailer (`81ac98846b`)
- Trailer amend accepted: `d5b0ff18ce` has the same tree as `07ec1f5c08`. TP1 E22 and G12 GSM8K (Δ +0.02 pp) accepted.
- TP2 c8 +2.44% (0/12) fails the rule, so the skinny head is not a candidate. Proposal agreed: if E23(a) puts c8 inside ±0.4% and leaves c4/c16 unchanged (within 0.5 pp of E22), drop the (512, 2048) M=8 entry with a comment matching the TP1 one.
- On the new SHA:
  - TP2 c1–c32 and TP1 c1–c32: TP1 may be the short ABAB form, because the dropped entry never runs at TP1.
  - Greedy single 32/32 at TP1 and TP2.
  - Unit tests.
  - GSM8K: not required again if the tree differs from `d5b0ff18ce` only in that table line (a TP2-only entry); say so in the PRREVIEW.
- If E23(a) does not clear c8, post it before changing anything else.

### Manager review: skinny PRREVIEW (`179d602c10`)
- E24 accepted at TP1 and TP2. Nothing is slower beyond ±0.4%, and the TP2 c1 noise is stated honestly. The body is accurate: identity, trailers, no hostnames or emails, footer kept, opening as a draft.
- Points 1–5: agreed as written. The CI gap is documented, which is enough. Skip the sync check.
- One item is pending the owner. The AI section says "the submitter reviewed every changed line". That is a claim on the owner's behalf for about 590 lines, so I've asked them whether to review the diff first or change the wording. I'll post the answer.
- **Not approved yet.** Still gated on a maintainer review of #69949 (none yet). Hold `fd7eea8834`, and do not rebase or amend unless asked.

### B9: #69778 sampled-eval check (moved from worker W18; the worker has been silent since 05:26Z)
Context: on #69778, gau-nernst asked whether the bitwise-parity commit is needed. Fused kernels like DeepGEMM and FlashInfer keep FP32 and multiply by the reciprocal, and they suggested the original numerics may favour greedy decoding. We replied publicly that we'll run GSM8K with sampling and drop the second commit if v1 is within noise.
- **Arms** (Qwen3.6-35B-A3B-FP8, TP1, the same serve flags as your B8):
  - main `fb159244e6`, the PR base;
  - v1 `c5ef00f633`, fp32 product;
  - v2 `152bca1e65`, bit-identical.
  - Use your B8 builds of these if you still have them.
- **Eval:** `tests/evals/gsm8k/gsm8k_eval.py --use-chat-completions --chat-template-kwargs '{"enable_thinking": false}' --max-tokens 1024 --max-concurrency 64 --temperature 0.6` (top_p 0.95 if supported). 8 runs per arm, with distinct seeds if supported. Halves on swapped GPUs, AOT 0. GPUs 0–5 only (the owner is using 6–7).
- **Report:**
  - mean, sd and Welch t for v1 vs main and for v2 vs main;
  - the per-question analysis (questions differing by ≥ 4 of 8);
  - a short setup paragraph we can paste into the PR thread (no hostnames).
- Skinny stays on hold meanwhile. Do not post anything on GitHub.

### Owner decision (2026-11-26 ~17:55Z): open skinny now as a DRAFT, without waiting for a maintainer review of #69949
APPROVED TO OPEN: skinny @ fd7eea88349237f939b72052c61b2fb6f1045217

- Title and body as in `results/b300/PRREVIEW/skinny_on_69949/`, with one change. In `## AI assistance`, replace "the submitter reviewed every changed line." with "the submitter reviewed the change and the measurements." The owner has not reviewed the whole diff line by line, and we can edit the body later if they do.
- Open as a **draft**, base `main`, head branch `sm103-skinny-fp8-block-gemm` on the owner's fork. Keep the "Stacked on #69949" first line.
- Before pushing, check:
  - `git log --format='%an <%ae> | %cn <%ce>' d6bccd2252..fd7eea8834` shows the owner identity;
  - the trailers are `Assisted-by: Claude Code` + `Signed-off-by:`, with no Co-authored-by;
  - the pushed head SHA equals `fd7eea8834`.
- After opening, post the PR number, URL and the head SHA as GitHub shows it. Do not request reviewers or labels, and do not comment on #69949.

### B10: smoke check of #69778 after the maintainer's push (low effort, GPUs 0–5)
gau-nernst approved #69778 and pushed `1fe793b18e` and `a6db1de994`:
- `expose_input_quant_key` moves from `process_weights_after_loading` to `create_weights`;
- the contract test now covers the block kernel;
- a test no longer mutates torch global state.

CI's bot then merged main, so the PR head is now `e1df7ec5f6` on the owner's fork, branch `fp8-block-silu-quant-fusion`. Fetch it read-only; **do not push to that branch.**
- Confirm the fusion still fires with real weights at the new head. Qwen3.6-35B-A3B-FP8, TP1: the shared-expert down_proj takes the fused SiLU+group-quant path. Show it with a short profile or a counter.
- Run a short paired TPOT check at c1/c4/c8 (1 pass × 4 rounds is enough) against the new head's merge base with main.
- Run `pytest tests/fusion/test_quant_activation_contract.py tests/compile/passes/test_silu_mul_quant_manual_fusion.py -q`.
- Report one paragraph. No GitHub actions.

### B11: find vLLM performance problems on SM103, with FA4 and QuACK in focus (owner request, 2026-11-27)
Priority: below any re-test of #70028 / #70205 after a rebase (those come first when they arrive). Text only on comms; no GitHub activity.

What the manager already checked on main `97dfd07879`:
- FA: `get_flash_attn_version()` picks FA4 for any compute-capability major 10 (SM100 and SM103) when `is_fa_version_supported(4)`. The vllm-flash-attn pin is `1b32c5caf7`. The latest commit touching that cmake file is `d9162cc8fc`, titled "[Do Not Merge][Build] Point vllm-flash-attn at vllm-project/flash-attention#222 for CI" (#68348). Check whether main really builds FA from an unmerged PR commit, and whether that matters.
- QuACK: `quack-kernels==0.6.5` (the newest on PyPI) is a dependency, but vLLM imports only `quack.compile_utils.make_fake_tensor`. None of QuACK's kernels (RMSNorm / fused add+RMSNorm, softmax, top-k, GEMM with epilogues, etc.) run in vLLM.
- `nvidia-cutlass-dsl` is pinned to 4.8.0; 4.8.1 exists.

Tasks:
1. **Which attention path actually runs.** For Qwen3.6-35B-A3B-FP8 (TP1) and Qwen3.8-Flash-Next-FP8 on B300: which backend and FA version are selected, from the server log. If FA4 is not used where it could be, say why (a guard, a head size, sinks, KV dtype, CUDA graphs).
2. **FA4 features we are missing.** Compare the vllm-flash-attn pin with upstream Dao-AILab flash-attention FA4 (cute DSL): list features or SM103 tunings upstream has and vLLM's pin lacks (for example decode split-KV, paged-KV paths, FP8 KV). For each, say whether it would affect our models.
3. **QuACK kernels vs vLLM kernels, microbench on SM103.** At least RMSNorm and fused add+RMSNorm (hidden 2048 / 4096 / 5120, BF16, M = 1–16 and 256–8192), plus any QuACK op that matches a top-10 kernel in task 4. Report µs vs vLLM's current kernel, output equality / max abs diff, JIT compile time, and CUDA-graph compatibility.
4. **Decode profile.** `vllm bench forward --profile-dir` (or torch profiler) on Qwen3.6-35B-A3B-FP8 TP1 at M = 1, 8, 32. Top-15 kernels by time, with % of the memory roofline where it applies. Flag anything far from roofline that is not already covered by our open PRs (#69778, #69949, #70028, #70205, #69543) or the worker's W15.
5. **Write up** `results/b300/B11/FINDINGS.md`: ranked candidates, each with evidence, expected E2E gain, size of change, and overlap with open upstream PRs (search the vLLM PR list). Do not build a PR yet. The manager and the owner choose which to pursue, and a QuACK integration needs the owner's OK first.

Rules: same as before (no hostnames, tokens or code patches on comms; GPUs 0–5; record SHAs).

#### B11 note from the manager's B200 survey (2026-11-27), to avoid duplicate work
- Attention: on SM100/SM103 a causal decoder picks FLASHINFER (trtllm-gen) before FLASH_ATTN (`vllm/platforms/cuda.py`), so FA4 runs only for non-causal attention, the multimodal encoder, sparse MLA, or an explicit `--attention-backend`. Qwen3.6 TP1 on B200: 10 full-attention layers on FLASHINFER, about 1.5% of decode GPU time. For task 1, also compare FA4 vs FLASHINFER for those layers on SM103 (kernel time at c1/c8/c32).
- QuACK RMSNorm / fused add+RMSNorm on B200 is equal to FlashInfer's CuTe norm at decode M and not better at prefill M; Qwen3.6 already uses the FlashInfer norm. vLLM's own C `rms_norm` is 1.3–2× slower than both (e.g. H2048 M8192: 19.2 vs 9.5 µs). Re-check on SM103 only briefly.
- Sampler: the largest item found (~110 µs/step at c1 with Qwen's default top_k 20 / top_p 0.95) is already covered by open upstream PRs #64116, #54417 and #65649. Do not build a candidate for it; an SM103 number for `flashinfer_sample` with and without `top_k_max=20` is enough.
