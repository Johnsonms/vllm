# W17 PRREVIEW — shared-expert gate skinny plan ("W15" PR), for manager review (2026-11-27 19:50Z)

- **Commit:** `383e366bcb1d4bf55e6c63d24c8aeacd6abfed96`, tree `3e82101d8d898145da2e32691d41e7dc563b4b31`, one commit on main `361f362033`. Local only; nothing pushed.
- **Identity:** author and committer = owner identity (checked with the `%an <%ae> | %cn <%ce>` log line), `Assisted-by: Claude Code` directly above `Signed-off-by`, no Co-authored-by.
- **Diffstat** (corrected; the first version was taken against `origin/main` after it had moved to 97dfd07879, so it also listed upstream's own changes):
  - `git rev-list --count 361f362033..383e366bcb` -> `1`
  - `git show --stat --format= 383e366bcb` ->

        tests/kernels/test_bf16_skinny_gemm.py           |  8 ++++++++
        vllm/models/qwen4_exp/nvidia/low_latency_gemm.py | 15 +++++++++++++++
        2 files changed, 23 insertions(+)

  - `git log -1 --format='%an <%ae> | %cn <%ce>' 383e366bcb` -> `Johnsonms <owner> | Johnsonms <owner>`
  - Parent of the commit is 361f362033. Upstream main is now 97dfd07879; it does not touch either of the 2 files since 361f362033, and a trial merge of the commit into it is clean.

- **What the diff is, in words:** 11 lines plus a 2-line comment for key (1, 2560) at the end of the Flash-Next SM100 table (the entries in `entries.json`, written in the table's short form), and one 6-line test that asserts the key's M list and that each config's row count equals its M. The existing table-driven SM100 test picks up the 11 new cases by itself.

## Evidence

**Unit tests.** `pytest tests/kernels/test_bf16_skinny_gemm.py -q` on one B200: PR 711 passed / 471 skipped; base 699 passed / 471 skipped. The 12 extra are the 11 new table cases plus the new test; the new test cannot pass on the base (the key does not exist). `pre-commit run --files` on the 2 files: all passed, no file modified.

**Kernel microbench** (W15 / W16 method: CUDA graph, 4 seeds x 3 interleaved rounds vs cuBLAS, min-max over seeds):

| M | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speedup | 2.23-2.34 | 2.44-2.58 | 2.34-2.49 | 2.09-2.27 | 2.16-2.25 | 2.03-2.09 | 2.19-2.52 | 1.84-2.06 | 1.74-1.78 | 1.70-1.79 | 1.64-1.76 |

cuBLAS 4.1-5.0 us per call, skinny 1.9-2.9 us. Max relative error vs an fp32 reference equals cuBLAS on every entry. No entry is slower.

**End to end on this SHA** (`ab/`). Qwen3.8-Flash-Next-FP8, real weights, TEP4 on 4 B200, recipe flags, no speculative decoding, random 512 in / 512 out, no prefix cache, compile cache off, "Directly load AOT" 0 times in all 6 logs, loaded file path and HEAD logged per server. Two TEP4 servers do not fit on the free GPUs, so the A/B is interleaved restarts A B A B A B on the same 4 GPUs: 3 starts per arm; after each start a discarded warm-up, then 4 bursts of 5 x c prompts per concurrency. Per start = median over bursts of median TPOT; delta = mean of per-start medians.

| requests | base per start (ms) | base spread | PR per start (ms) | PR spread | delta | PR starts faster than every base start |
|---|---|---|---|---|---|---|
| 1 | 3.984 3.995 3.991 | 0.28 % | 3.977 3.978 3.993 | 0.40 % | -0.19 % | 2 of 3 |
| 2 | 4.484 4.493 4.466 | 0.59 % | 4.465 4.468 4.470 | 0.10 % | -0.29 % | 1 of 3 |
| 4 | 5.316 5.320 5.304 | 0.29 % | 5.241 5.236 5.247 | 0.21 % | **-1.35 %** | 3 of 3 |
| 8 | 6.080 6.091 6.078 | 0.22 % | 5.977 5.983 5.986 | 0.14 % | **-1.67 %** | 3 of 3 |
| 16 | 6.623 6.637 6.624 | 0.21 % | 6.571 6.562 6.587 | 0.38 % | **-0.82 %** | 3 of 3 |
| 32 | 8.038 8.046 8.022 | 0.30 % | 8.030 8.021 8.025 | 0.12 % | -0.13 % | 1 of 3 |

- **The gain is real at 4, 8 and 16 requests and not shown at 1, 2 and 32.**
- **It is about half of the W16 number.** W16 (main 0fbba2aa71, same 11 entries) had -0.7 / -2.3 / -3.0 / -0.9 % at 2 / 4 / 8 / 16. The base TPOT is the same within 0.8 % on both mains, so the candidate got slower relative to W16, not the base faster. I have not found the cause. Main moved 12 files of the Flash-Next model code between the two SHAs. The body below uses only the numbers of this SHA.

**Greedy, 32 single prompts x 256 tokens:** identical, 0 of 32 differ in all 9 base-vs-PR start pairs (base vs base 0, PR vs PR 0). This differs from W16 on the older main, where 2 of 32 differed. The kernel is still a different summation order than cuBLAS, so I do not claim bit-identical output in the body.

**Batched greedy:** not used. On TEP4 the 256-prompts-in-one-request output is not reproducible between two starts of the same tree (W16).

**GSM8K, 5-shot, 1319 questions, temperature 0, 64 concurrent, 8 runs per arm** (3 + 3 + 2 over the three starts; per-question dumps in `ab/*/gsm_perq_*.json`):
- Correct — base: 1273 1277 1281 1276 1274 1277 1277 1271 (mean 0.9672, sd 3.1 questions). PR: 1276 1275 1272 1280 1270 1276 1277 1278 (mean 0.9670, sd 3.2).
- Per-question flips between two runs: base vs base 7-19 (median 12); PR vs PR 7-18 (median 13); base vs PR 6-17 (median 12).
- No question differs by 4 or more of 8 between the arms (largest difference 3 of 8).

**Not measured:** TP8 / TEP8; B300 / SM103; speculative decoding on; prefill; any base that includes #69907. The compiled extension on this node is from an older wheel: main changed two CUDA sources since (the DSA top-k and one HC kernel file); both arms use the same binary.

## Duplicate check
Searched open and merged PRs for "skinny gemm", "Qwen3.8 skinny", "Flash-Next GEMM", "SM100 small GEMM", "low_latency_gemm". No PR adds (1, 2560) to the SM100 table. Related:
- #69736 (merged): the SM100 table this extends.
- #57848 (open): SM103 router lines in the same file; different table, different shape.
- #66865 (open): GB300 specialized decode; SM103 only.
- #69907 (open, no review): makes FlashInfer cute-dsl the default small-batch BF16 GEMM in the generic unquantized linear method. Today the gate is not in the SM100 table, so after #69907 the gate would run on that kernel in the base. This PR would then move it to the skinny kernel at M <= 16 and to torch linear at 17-32. **If #69907 merges first, the A/B must be repeated against it.**

## Title
`[Perf] SM100 skinny GEMM plan for the Qwen3.8-Flash-Next shared-expert gate`

## Body

### Overview
The shared-expert gate of Qwen3.8-Flash-Next is a BF16 linear with a single output row, local weight shape `(N, K) = (1, 2560)`. It runs 48 times per decode step on each rank. On SM100 this shape is not in the model's low-latency GEMM table, so it runs on the standard linear backend (cuBLAS), at 4-5 us per call.

This PR adds measured CuTe DSL skinny GEMM configurations for that shape at `M = 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16`. These are the decode CUDA graph sizes up to 16. It reuses the existing kernel and dispatch guards; only the table grows.

Not affected: batches above 16 rows, other shapes, other GPUs (the SM103, SM90 and SM121 tables are unchanged), batch-invariant mode.

### Results
End to end, Qwen3.8-Flash-Next-FP8, TEP4 on 4x B200, no speculative decoding, random 512 in / 512 out.
- Method: the two arms are restarted alternately on the same 4 GPUs (A B A B A B, 3 starts per arm). After each start: one discarded warm-up, then 4 bursts of 5 x c prompts per concurrency. The number per start is the median over bursts of the median TPOT.
- Compile cache off; main at `361f362033`.

| concurrent requests | main TPOT (ms), 3 starts | PR TPOT (ms), 3 starts | change |
|---|---|---|---|
| 1 | 3.984 3.995 3.991 | 3.977 3.978 3.993 | -0.2 % |
| 2 | 4.484 4.493 4.466 | 4.465 4.468 4.470 | -0.3 % |
| 4 | 5.316 5.320 5.304 | 5.241 5.236 5.247 | -1.4 % |
| 8 | 6.080 6.091 6.078 | 5.977 5.983 5.986 | -1.7 % |
| 16 | 6.623 6.637 6.624 | 6.571 6.562 6.587 | -0.8 % |
| 32 | 8.038 8.046 8.022 | 8.030 8.021 8.025 | -0.1 % |

At 4, 8 and 16 requests every PR start is faster than every main start. At 1, 2 and 32 the difference is inside the start-to-start spread.

Kernel level, skinny vs cuBLAS in a CUDA graph (4 seeds, 3 interleaved rounds): 1.64-2.58x over the 11 sizes; no size is slower.

Accuracy. The kernel sums in a different order than cuBLAS, so the output is not guaranteed bit-identical.
- Greedy, 32 single prompts x 256 tokens: identical to main.
- GSM8K (5-shot, 1319 questions, temperature 0), 8 runs per arm: 0.9672 on main, 0.9670 with the PR. Two runs of the same arm differ on 7-19 questions; a main run and a PR run differ on 6-17. No question differs by 4 or more of 8 runs between the arms.

Entries that were tried and left out:
- The router `(512, 2560)` at M = 16 wins at kernel level (1.2x) and loses end to end (+3.7 % TPOT at 16 requests).
- Extra M for the already-listed projections and the LM head showed no end-to-end effect.
- M = 7, 9, 11, 13 and 15 are not decode CUDA graph sizes.

Not measured: TP8, SM103, speculative decoding, prefill.

### Testing
- `pytest tests/kernels/test_bf16_skinny_gemm.py` on a B200: 711 passed, 471 skipped (main: 699 passed, 471 skipped). The existing table-driven SM100 test covers the 11 new entries; one new test pins the list of sizes.
- `pre-commit run --files` on the changed files passes.

### Related PRs
- #69736 added the SM100 table this extends.
- #57848 tunes the SM103 router lines in the same file; no overlap.
- #69907 would make FlashInfer cute-dsl the default small-batch BF16 GEMM for layers on the generic linear method. Layers in this table do not use that method, so with both PRs the gate uses the skinny kernel up to 16 rows. The two have not been compared on this shape.

### AI assistance
Claude Code was used to run the sweeps and benchmarks and to draft the change and this description. The submitter reviewed the change and the measurements.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
