**Stacked on #69949; the diff includes its 2 commits until it merges.** This PR's own change is the last commit (`fd7eea8834`, 5 files, +589/−7).

## Overview

On B300 (SM103), decode-sized block-FP8 linears of Qwen3.6-35B-A3B-FP8 run a CuTe DSL "skinny" GEMM at the token counts a measured per-shape plan lists. All other token counts keep #69949's small-M dispatch: FlashInfer's TRT-LLM GEMM for small M, otherwise CUTLASS.

## Claims

- Qwen3.6-35B-A3B-FP8 on B300, median TPOT against #69949:
  - TP=1: −9.0% at concurrency 1, −8.8% at 2, −4.4% at 4;
  - TP=2: −3.5% at 1, −7.9% at 2, −4.5% at 4, −3.8% at 16.
  - All other concurrencies up to 32 are within ±0.3%.
- Accuracy is unchanged: GSM8K is 0.9622 on the base and 0.9624 on this PR (8 runs each). The output is not bit-identical to CUTLASS. Measured against the exact product of the dequantized operands, its max error is the same as CUTLASS's; the test allows up to 2×.
- Only SM103 is affected. Other devices, and shapes or token counts not in the plan, take exactly the same path as before.

## Validation

**End to end: Qwen3.6-35B-A3B-FP8, real weights, B300.**

Method:
- Base: #69949 head `d6bccd2252`. PR: `fd7eea8834`. No server log contains "Directly load AOT".
- Server: `VLLM_DISABLE_COMPILE_CACHE=1 vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --language-model-only --max-num-seqs 256 --max-model-len 8192 --gpu-memory-utilization 0.9 --no-enable-prefix-caching`, plus `--tensor-parallel-size 2` for TP=2.
- Both servers run at the same time on separate GPUs. Bursts alternate between them, with the order flipped each round, 6 rounds per concurrency `C`:
  `vllm bench serve --backend openai --endpoint /v1/completions --dataset-name random --random-input-len 512 --random-output-len 512 --random-range-ratio 0 --ignore-eos --num-prompts 5*C --max-concurrency C --num-warmups C --seed <round>`
- Then both servers restart with the GPUs swapped and the bursts run again. "Mean" averages the two passes; "pairs faster" counts the 12 rounds.

| Concurrency | TP=1 base → PR | Mean | Pairs faster | TP=2 base → PR | Mean | Pairs faster |
|---:|---|---:|---:|---|---:|---:|
| 1 | 2.545 → 2.315 ms | **−9.03%** | 12 / 12 | 2.457 → 2.371 ms | **−3.48%** | 10 / 12 |
| 2 | 2.891 → 2.638 | **−8.75%** | 12 / 12 | 2.738 → 2.521 | **−7.89%** | 11 / 12 |
| 4 | 3.615 → 3.454 | **−4.44%** | 12 / 12 | 2.949 → 2.816 | **−4.50%** | 12 / 12 |
| 8 | 4.299 → 4.293 | −0.15% | 7 / 12 | 3.464 → 3.458 | −0.17% | 7 / 12 |
| 16 | 5.451 → 5.464 | +0.25% | 3 / 12 | 4.473 → 4.304 | **−3.78%** | 12 / 12 |
| 32 | 6.818 → 6.828 | +0.16% | 6 / 12 | 5.035 → 5.049 | +0.28% | 6 / 12 |

Median TPOT. Between two servers running identical code, this setup shows up to about 0.5 points of difference.

<details><summary>Raw per-pass output (TP=1 and TP=2)</summary>

TP=1:

A=base (d6bccd2252 Enable the small-M TRT-LLM block-FP8 GEMM path on)
B=PR (fd7eea8834 [Perf] Add a CuTe DSL skinny block-FP8 GEMM to th)

| c | pass 1: A / B ms, Δ (pairs) | pass 2: A / B ms, Δ (pairs) | **mean Δ** |
|---|---|---|---|
| 1 | 2.554 / 2.309, -9.60% (-9.88..-9.30; 6/6 neg) | 2.535 / 2.320, -8.47% (-8.69..-8.05; 6/6 neg) | **-9.03%** |
| 2 | 2.897 / 2.631, -9.18% (-9.57..-8.55; 6/6 neg) | 2.885 / 2.645, -8.32% (-8.79..-7.65; 6/6 neg) | **-8.75%** |
| 4 | 3.620 / 3.435, -5.09% (-5.43..-4.74; 6/6 neg) | 3.609 / 3.473, -3.78% (-4.20..-3.42; 6/6 neg) | **-4.44%** |
| 8 | 4.299 / 4.293, -0.14% (-0.70..+0.26; 3/6 neg) | 4.299 / 4.292, -0.16% (-0.62..+0.45; 4/6 neg) | **-0.15%** |
| 16 | 5.440 / 5.470, +0.56% (-0.88..+1.32; 1/6 neg) | 5.461 / 5.458, -0.05% (-1.19..+0.82; 2/6 neg) | **+0.25%** |
| 32 | 6.813 / 6.848, +0.52% (-0.72..+3.43; 3/6 neg) | 6.822 / 6.808, -0.20% (-1.38..+0.84; 3/6 neg) | **+0.16%** |

TP=2:

A=base (d6bccd2252 Enable the small-M TRT-LLM block-FP8 GEMM path on)
B=PR (fd7eea8834 [Perf] Add a CuTe DSL skinny block-FP8 GEMM to th)

| c | pass 1: A / B ms, Δ (pairs) | pass 2: A / B ms, Δ (pairs) | **mean Δ** |
|---|---|---|---|
| 1 | 2.438 / 2.432, -0.27% (-1.26..+0.23; 4/6 neg) | 2.475 / 2.309, -6.69% (-11.85..-3.08; 6/6 neg) | **-3.48%** |
| 2 | 2.707 / 2.577, -4.79% (-7.48..+2.78; 5/6 neg) | 2.769 / 2.465, -10.99% (-11.47..-10.09; 6/6 neg) | **-7.89%** |
| 4 | 2.936 / 2.826, -3.72% (-4.24..-3.26; 6/6 neg) | 2.962 / 2.806, -5.29% (-5.64..-4.77; 6/6 neg) | **-4.50%** |
| 8 | 3.462 / 3.462, -0.02% (-0.74..+0.42; 2/6 neg) | 3.465 / 3.454, -0.32% (-0.70..+0.08; 5/6 neg) | **-0.17%** |
| 16 | 4.460 / 4.311, -3.33% (-3.91..-2.82; 6/6 neg) | 4.486 / 4.297, -4.22% (-4.54..-3.78; 6/6 neg) | **-3.78%** |
| 32 | 5.017 / 5.047, +0.61% (-0.79..+2.01; 2/6 neg) | 5.053 / 5.050, -0.05% (-0.65..+0.73; 4/6 neg) | **+0.28%** |

</details>

**Kernel level.** Each plan entry is listed in `skinny_fp8_block_gemm_plans.py` with its minimum speedup:
- against the kernel the small-M dispatch would otherwise run (TRT-LLM or CUTLASS), and against CUTLASS;
- measured over 4 seeds × 3 interleaved rounds, CUDA graph replay, L2 flushed;
- an entry is kept only when it is ≥ 1.10× over both on every seed.

The entries range from 1.13× to 2.29× over the default kernel.

**Accuracy.**
- **GSM8K.** `tests/evals/gsm8k/gsm8k_eval.py --use-chat-completions --chat-template-kwargs '{"enable_thinking": false}' --max-tokens 1024 --temperature 0 --max-concurrency 64`, 1,319 questions, TP=1.
  - 8 runs per arm in two groups of 4, with the GPUs swapped between groups.
  - Base 0.9622 (sd 0.0021), PR 0.9624 (sd 0.0022), Δ +0.02 points (Welch t +0.17).
  - Per question: 16.8 flips per base/PR run pair, against 17.7 (base/base) and 15.7 (PR/PR).
  - No question is right in all base runs and wrong in all PR runs, or the reverse.
  - Measured on `07ec1f5c08`. It differs from this head only in the TP=2-only (512, 2048) M=8 plan entry and comments, so the code that runs at TP=1 is the same.
- **Greedy, 256 tokens, 32 single + 256 batched prompts.**
  - TP=1: 3 / 32 single and 200 / 256 batched prompts are identical to the base. The base against itself gives 32 / 32 and 162 / 256.
  - TP=2: 8 / 32 and 222 / 256, against 32 / 32 and 253 / 256.
  - The differences come from the FP32 summation order (about 1 bf16 ulp per output).

<details><summary>Raw GSM8K and greedy output</summary>

```
A: n=8 acc 0.9659 0.9613 0.9629 0.9629 0.9591 0.9629 0.9598 0.9629 | mean 0.9622 sd 0.0021
B: n=8 acc 0.9606 0.9629 0.9636 0.9629 0.9644 0.9651 0.9613 0.9583 | mean 0.9624 sd 0.0022
delta B-A +0.0002 (Welch t +0.17)
flips A->B: right->wrong 8.3, wrong->right 8.5, total 16.8 over 64 pairs
flips A->A: right->wrong 9.6, wrong->right 8.1, total 17.7 over 28 pairs
flips B->B: right->wrong 8.2, wrong->right 7.4, total 15.7 over 28 pairs
right in all A & wrong in all B: 0; reverse: 0
```
TP=1 greedy:
```
A vs A (floor):
single: 32/32 identical; divergent: first differing token at median - (min -)
batched: 162/256 identical; divergent: first differing token at median 51.5 (min 0)
A vs B:
single: 3/32 identical; divergent: first differing token at median 29 (min 1)
batched: 200/256 identical; divergent: first differing token at median 51.0 (min 3)
serve_A.log: "Directly load AOT" count 0
serve_B.log: "Directly load AOT" count 0
DONE
```
TP=2 greedy:
```
A vs A (floor):
single: 32/32 identical; divergent: first differing token at median - (min -)
batched: 253/256 identical; divergent: first differing token at median 16 (min 1)
A vs B:
single: 8/32 identical; divergent: first differing token at median 37.0 (min 1)
batched: 222/256 identical; divergent: first differing token at median 50.0 (min 1)
serve_A.log: "Directly load AOT" count 0
serve_B.log: "Directly load AOT" count 0
DONE
```
</details>

**Tests** (on `fd7eea8834`):
```
pytest -q tests/kernels/quantization/test_skinny_fp8_block_gemm.py
26 passed, 35 warnings in 30.45s

pytest -q tests/kernels/quantization/test_block_fp8.py tests/kernels/quantization/test_skinny_fp8_block_gemm.py
1 failed, 394 passed, 456 skipped, 40 warnings in 320.37s (0:05:20)
# base d6bccd2252, test_block_fp8.py only:
1 failed, 368 passed, 456 skipped, 22 warnings in 307.25s (0:05:07)
```
- The one failure is the same on base and PR: `test_w8a8_block_fp8_torch_scaled_mm_matmul`, `CUBLAS_STATUS_NOT_SUPPORTED` from `cublasLtMatmulAlgoGetHeuristic`.
- `test_skinny_fp8_block_gemm_matches_cutlass` checks every plan entry against CUTLASS and against the exact product.
- `test_small_m_dispatch_uses_skinny_plan` checks that the dispatch op takes the skinny GEMM for planned token counts, and stays bit-identical to CUTLASS for the others.
- `pre-commit run --files <changed files>` passes with no changes.

All on B300 (SM103), CUDA 13.2, torch 2.14.

## Details

**Why.** At decode token counts these linears are latency-bound:
- The TRT-LLM GEMM takes a flat ~6–11 µs per call on B300, whatever N and K are.
- For the narrower Qwen3.6-35B-A3B shapes, a CUDA-core GEMM that streams the weight once is much faster: shared-expert gate_up / down, GDN `out_proj` / attention `o_proj`, and the TP=2 `in_proj_qkvz` / `qkv_proj`.

**Kernel** (`cute_dsl/_skinny_fp8_block_gemm.py`):
- Each CTA owns 1–2 output columns. Every thread streams 16 FP8 values of K per tile, and scales each K group's partial sum by the activation group scale times the 128×128 weight-block scale.
- The CTA reduces in FP32.
- Weights are prefetched before the PDL wait.
- K is static per config; N is dynamic.
- It reuses `ShapeDynamicSkinnyGemm`'s availability check, PDL setting and stream, and the CuTe DSL warmup registry: planned configs compile before CUDA graph capture.

**Dispatch** (`scaled_mm/cutlass.py`):
- `_fp8_block_scaled_mm_small_m` looks up the plan for `(N, K)` (cached per shape) and takes the skinny GEMM when the token count is listed. Otherwise #69949's TRT-LLM / CUTLASS choice is unchanged.
- The choice is made per call, as in #69949, so every CUDA graph size gets its own kernel.

**Plans:**
- SM103 only (`(10, 3)`), Qwen3.6-35B-A3B at TP=1 and TP=2: 7 shapes, 2–4 token counts each.
- Shapes where TRT-LLM wins were dropped: TP=1 `in_proj_qkvz` (12288, 2048), TP=1 `qkv_proj` (9216, 2048), and all TP=4 shapes.
- SM100 has no plan, since it has not been swept, so it keeps #69949's behaviour.

**Tradeoff found end to end.** The shared-expert gate_up shapes (1024, 2048) at TP=1 and (512, 2048) at TP=2 win in isolation at 8 tokens (1.38× / 1.70×), but made decode at batch 8 2–3% slower end to end. They have no 8-token entry.
- A torch profile at TP=1, batch 8, showed the skinny GEMMs faster than TRT-LLM. But `act_and_mul` went from 2.54 to 4.52 µs and sampling (`_gumbel_sample`) from 36 to 117 µs.
- Disabling PDL made it worse. We did not find the exact cause.
- Removing the entry: TP=2 batch 8 −2.50% (12 / 12) against keeping it.

**Limitations:**
- Not measured: SM100 / B200 (no plan there), other models, TP ≥ 4.
- No CPU/GPU synchronization is added (a dict lookup per call). `VLLM_GPU_SYNC_CHECK=error` was not run.
- **CI.** The new tests need SM100+ and the CuTe DSL. The `kernels/quantization` Buildkite step runs on L4, so they skip there. The plan path itself only runs on SM103, which CI does not have. The tests above were run on B300.
- **Complexity.** About 500 new lines outside tests (222 kernel, 143 wrapper, 93 plans, 43 dispatch). All of it is behind the SM103 plan lookup.

## AI assistance

Claude Code was used to write the kernel, plans and tests and to run the benchmarks; the submitter reviewed the change and the measurements.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
