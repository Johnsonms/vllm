# B300 PR review request (11-25)

**One PR candidate.** Nothing has been opened or pushed. Code stays local; this file describes it in words only.

## 1. What and where
- **Title (draft):** `[Perf] Skinny CuTe DSL block-FP8 GEMM and FlashInfer TRT-LLM GEMM for decode-sized token counts on SM103`
- **Local branch:** `kopt/fp8-block-skinny-gemm`, 4 commits on `e66bede56b`:
  1. `c2a7e84f9f`: CuTe DSL skinny block-FP8 GEMM and its dispatch.
  2. `f50df96b06`: plans for TP2/TP4.
  3. `44c5179e80`: large-N shapes at M ≥ 4 route to FlashInfer's TRT-LLM GEMM.
  4. `02a695f014`: mypy fix.
- **Applies to main:** cleanly on main `aec644a8f9` (checked with `git merge-tree`, commit by commit).
- **`git diff --stat`:** 5 files, +716 / −0.
  - new `vllm/model_executor/kernels/linear/cute_dsl/_skinny_fp8_block_gemm.py` (222): the kernel
  - new `vllm/model_executor/kernels/linear/cute_dsl/skinny_fp8_block_gemm.py` (143): compile cache and warmup before graph capture
  - new `vllm/model_executor/kernels/linear/cute_dsl/skinny_fp8_block_gemm_plans.py` (146): plans
  - `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py` (+84): dispatch
  - new `tests/kernels/quantization/test_skinny_fp8_block_gemm.py` (121)
- **Motivation:**
  - At decode sizes (M ≤ 16), vLLM's CUTLASS block-FP8 GEMM is latency-bound on B300. It takes 6–11 µs for every Qwen3.6-35B-A3B linear regardless of weight size, and these linears run 160 times per decode step, about 40% of a c1 step.
  - **Kernel and dispatch:**
    - The PR adds a CUDA-core skinny GEMM (CuTe DSL, modeled on the BF16 skinny GEMM). It works on block-FP8 directly: 1×128 activation groups and 128×128 weight blocks, with the scales applied per K group in FP32.
    - `CutlassFp8BlockScaledMMKernel` dispatches through one custom op, per (N, K, M) from a measured plan table: skinny GEMM, FlashInfer `gemm_fp8_nt_groupwise(backend="trtllm")` (flat in M; it wins for the large in_proj/qkv shapes from M=4), or CUTLASS.
  - **Plan table:**
    - Plans exist for SM103 only (capability (10, 3)), for the Qwen3.6-35B-A3B linear shapes at TP1/2/4.
    - An entry is kept only when it beats CUTLASS by ≥ 1.10×: 42 skinny entries and 18 TRT-LLM entries.
    - Unplanned shapes and M, other GPUs, and other output dtypes keep CUTLASS unchanged.

## 2. Overlap with #69778
- **This PR does not touch the SiLU/quant producer.** It consumes the same `QuantizedActivation` (column-major group scales) that #69778 produces, and the two stack. My full-stack run (E10 below) includes #69778.
- **My local bitwise producer (`kopt/silu-quant-bitexact`, `1a0043229f`) vs your `152bca1e65`.** I will **not** open it. Differences:
  - FP8 arithmetic is the same except the product: mine computes `silu * up`, yours `silu * (up + 0.0f)`. They differ only when `up == -0.0`, and yours is the one that matches `silu_and_mul`.
    - Confirmed by test: your test fails 12/443 on my kernel, all fp16 at magnitude 1e-4, where `up` underflows to −0. Mine writes FP8 0x80 vs 0x00 there; scales are equal everywhere.
  - The reciprocal: you use `fp8_reciprocal_scale` (guards scale == 0); I use `1/scale`. They are the same for all reachable scales.
  - INT8: you leave it unchanged; mine changed it (rounded product, dropped the `min_scaling_factor` floor), which was not needed. Yours is better.
- **Tests on your kernel** (JIT-built `152bca1e65` .cu for sm_103a, op redirected in the test process):
  - Your test file passes 443/443.
  - My test file passes 363/363. My bitwise test is `test_silu_and_mul_per_block_quant_fp8_matches_unfused_bitwise`: dtype ∈ {bf16, fp16} × group 64/128 × scale transposed or not × num_tokens ∈ {7, 4096}. Your test already covers more (tiny magnitudes, misaligned input, zero group), so I see nothing to add to #69778.

## 3. Evidence
- **Unit tests:** `pytest tests/kernels/quantization/test_skinny_fp8_block_gemm.py` covers:
  - every planned config vs CUTLASS and vs the exact product of the dequantized operands
  - every TRT-LLM entry through the dispatch op
  - the CUTLASS fallback for unplanned M

  Results:
  - PR head: **65 passed**.
  - Base `e66bede56b`: collection error (the module does not exist).
- **pre-commit:** `pre-commit run --files <5 files>` passes with no files modified, after commit 4.
- **Kernel microbench** (CUPTI, CUDA graph, cold L2; single sweep, not multi-seed or interleaved):
  - Planned entries are 1.18×–3.44× faster than CUTLASS.
  - Small shapes (shared expert, o_proj): 2.4–3.4× at M=1–2.
  - in_proj/qkv: 1.5–1.6× at M=1, and about 1.27× via TRT-LLM at M=4–16.
  - No planned entry is slower; everything else stays on CUTLASS.
- **E2E, paired A/B** (both servers at once, 6 rounds of 5×c-prompt bursts per c, order flipped each round, then GPUs swapped, mean of the 2 passes):
  - `PYTHONPATH` set per tree, with the loaded file path and HEAD logged in each server log.
  - `VLLM_DISABLE_COMPILE_CACHE=1`; "Directly load AOT" appears 0 times in all server logs.
  - Qwen3.6-35B-A3B-FP8, random 512/512, `--ignore-eos`, median TPOT.

  | run | c1 | c2 | c4 | c8 | pairs faster |
  |---|---|---|---|---|---|
  | E9: TP1, PR (commits 1–3) vs `e66bede56b` | −18.0% | −16.4% | −13.5% | −7.0% | 12/12 at every c |
  | E8: TP2, same | −17.2% | −15.5% | −11.6% | −6.5% | 12/12 at every c |
  | E10: full stack (this PR + #69778 + my other local branches) vs `fb159244e6` | −15.5% | −20.4% | −24.7% | −18.2% | 12/12 at every c |

  Higher c (16+) was not measured. At M > 16 nothing is planned, so it falls back to CUTLASS.
- **Accuracy** (not bit-identical: about 1 bf16 ulp from FP32 summation order; max error vs the exact product equals CUTLASS's):
  - GSM8K, 8 interleaved runs per arm with per-question dumps (final commits 1–3 vs `e66bede56b`): 0.9613 (sd 0.0027) → 0.9627 (sd 0.0028), Δ +0.0013.
    - Flips A→B: 17.0 per pair vs an A/A floor of 18.1 and B/B of 16.8.
    - 0 questions are right in all A runs and wrong in all B runs (or the reverse).
  - Greedy token check (4 prompts × 64 tokens): 3/4 identical. The 32-single / 256-batched check was not run.
- **Not measured:** SM100/B200 (no plans for it, so unchanged); other models; TP4 E2E (plans measured at kernel level only); c ≥ 16.

## 4. Duplicates checked
- **Searches:** `gh pr list --state all` with "skinny fp8 block gemm", "block fp8 small M decode gemm", "gemm_fp8_nt_groupwise trtllm", "flashinfer groupwise fp8 linear", "cute dsl fp8 gemv", "fp8 block scaled mm small batch SM100".
- **Related, not duplicates:**
  - #64980 (open): W8A16 skinny GEMM for block-FP8 on Hopper. Different arch and different math (dequantizes weights to BF16); it touches `skinny_gemm_plans.py` and `scaled_mm/flashinfer.py`, while this PR touches neither.
  - #69736 / #62956 (merged): BF16 skinny GEMM plans. This PR reuses their structure for block-FP8.
  - The existing `FlashInferFp8BlockScaledMMKernel` (swapAB blockscale GEMM with BF16 input) is a different kernel and is not selected here.
- **Other local branches, not proposed:**
  - sampler top-k fast path: duplicate of #64116.
  - norm + group-128 quant fusion: overlaps #65497 / #36016 / #45770.
  - shared-expert sigmoid-gate fusion: about −3% E2E; related to #60846; owner's call later.
  - MoE finalize into the next norm: no E2E gain.

## 5. Commit identity
- `git log --format='%an <%ae> | %cn <%ce>'`: author = committer = the owner's GitHub identity on all 4 commits.
- No Co-authored-by line, no hostname.
- **No Signed-off-by yet:** it will be added with `git rebase --signoff` onto current main just before opening, then the identity check re-run.

## Node note
- OVS busy-poll CPUs 182/239 are 100% busy. Python processes on this node (vLLM servers, bench clients) came up pinned to CPU 182 and never ran, which hung pass 2 of E8/E9/E10 for 9 h.
- I now run a loop that moves my own processes off those CPUs. All reported numbers come from runs that completed normally.
