# W16 — end-to-end check of the W15 Flash-Next SM100 skinny table entries

## Interim (2026-11-26 00:36Z): gain at 1-8 requests, **slower at 16**; ablation running

**Arms.** Base = upstream main 0fbba2aa71 (contains #69736). Candidate = base + the 32 `verified.json` entries in the Flash-Next SM100 table (local commit 38cc50f415, not pushed). Qwen3.8-Flash-Next-FP8, real weights, TEP4 on GPUs 2-5, W14 recipe flags, no speculative decoding, compile cache off (AOT count 0 in all 6 logs), each server log records the loaded table path and HEAD (`ab/*/checks.txt`).

**Method.** Interleaved restarts A B A B A B (3 starts per arm). After each start: one discarded 16-way warm-up, then 4 bursts per concurrency (5 x c prompts, random 512 in / 512 out, ignore-eos, no prefix cache). Per start = median over the 4 bursts of the median TPOT. Delta = mean of the per-start medians.

| requests | base per start (ms) | base spread | candidate per start (ms) | cand spread | delta |
|---|---|---|---|---|---|
| 1 | 4.014 4.011 4.004 | 0.25 % | 3.974 3.968 3.968 | 0.16 % | **-1.00 %** |
| 2 | 4.531 4.525 4.506 | 0.55 % | 4.506 4.467 4.464 | 0.96 % | -0.92 % |
| 4 | 5.333 5.323 5.324 | 0.18 % | 5.211 5.144 5.153 | 1.30 % | **-2.96 %** |
| 8 | 6.106 6.061 6.070 | 0.74 % | 5.972 5.988 5.986 | 0.27 % | **-1.59 %** |
| 16 | 6.790 6.667 6.659 | 1.95 % | 6.902 6.884 6.885 | 0.26 % | **+2.76 %** |
| 32 | 8.196 8.097 8.114 | 1.21 % | 8.184 8.089 8.114 | 1.18 % | -0.08 % |

- 1, 4 and 8 requests: the gain exceeds the base start-to-start spread. 2 requests: the gain is about the size of the spread.
- **16 requests is slower, beyond the spread** (all 3 candidate starts are above all 3 base starts). 32 is flat, as expected.
- So the layer-group ablation is running now (groups a-e, 2 rounds of base + one arm per group, c = 1, 4, 8, 16).

## Which entries fire (`hits_rank0.txt`)
One counting start of the candidate with a local logging line in the table lookup (not in the measured arms). The lookup runs at graph capture, so this lists every (N, K, M) per captured size.
- **Captured sizes are 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, ...** So:
  - **M = 7 never occurs** for the in-graph layers; those 4 entries are dead and should be dropped.
  - **M = 10 and 14 do occur** and W15 did not sweep them.
- Calls per forward per rank, any M: (1, 2560) 48, (512, 2560) 48, (320, 2560) 48, (2560, 1536) 48, (24, 2560) 36, (4096, 2560) 36, (4224, 2560) 12. The LM head runs outside the graph at the real batch size.
- At the benchmark concurrencies the new entries that fire all the time are: the gate (1, 2560) at every c; the router at M = 4 and 16; the LM head at M = 4. The M = 3, 5, 6, 12 entries fire only while the number of running requests passes through those sizes.

## Greedy check (256 tokens)
- **32 single prompts: candidate differs from base on 2 of 32.** Base vs base (3 starts): 0 of 32. Candidate vs candidate: 0 of 32. So the entries are not output-identical to cuBLAS, as expected for a different summation order.
- **256 prompts in one request (first one after start discarded): the base-vs-base floor is 189-251 of 256 differing**, candidate-vs-candidate 0-243, base-vs-candidate 2-249. The batched output is not reproducible between starts of the same tree, so this check cannot separate the arms.
- Because the single prompts are not identical, GSM8K (8 runs per arm, per-question dumps) is owed. I will run it on the final entry set after the ablation, not on the 32-entry set that regresses at 16.

## Dup check
No open PR adds these entries. Related: #69907 (FlashInfer cute-dsl as the default small-batch BF16 GEMM in the generic linear path; could overlap for these layers), #57848 (SM103 router lines in the same table), #66865 (GB300 specialized decode), #58662 (TGV for (5120, 6144) at TP1).
