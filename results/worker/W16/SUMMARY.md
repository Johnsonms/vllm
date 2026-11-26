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

## Ablation by layer group (2026-11-26 03:15Z)
One arm per group (only that group's new entries on top of base), 2 rounds of [base, a, b, c, d, e], 4 bursts per concurrency per start. Cells: the two per-start medians (ms) and the delta of their mean vs base.

| requests | base | (a) gate (1, 2560) | (b) router (512, 2560) | (c) (24, 2560) + (320, 2560) | (d) large projections M3 | (e) LM head |
|---|---|---|---|---|---|---|
| 1 | 4.015 3.986 | 3.985 3.971 (**-0.58 %**) | 3.998 3.995 (-0.11 %) | 3.993 3.999 (-0.12 %) | 3.988 3.994 (-0.25 %) | 3.992 3.995 (-0.18 %) |
| 4 | 5.334 5.318 | 5.247 5.244 (**-1.50 %**) | 5.316 5.320 (-0.14 %) | 5.319 5.318 (-0.13 %) | 5.322 5.317 (-0.13 %) | 5.312 5.322 (-0.17 %) |
| 8 | 6.063 6.094 | 5.987 5.981 (**-1.55 %**) | 6.096 6.097 (+0.29 %) | 6.100 6.098 (+0.34 %) | 6.100 6.104 (+0.38 %) | 6.094 6.101 (+0.32 %) |
| 16 | 6.629 6.658 | 6.599 6.565 (**-0.93 %**) | 6.893 6.883 (**+3.68 %**) | 6.643 6.632 (-0.09 %) | 6.633 6.619 (-0.27 %) | 6.648 6.646 (+0.05 %) |

- **The gate (1, 2560) carries the gain** at every concurrency, including 16.
- **The router causes the 16-request loss** (its M = 16 line, the same loss #69736 part 5 saw) and gives nothing at 4 (its M = 4 line).
- Groups c, d, e are inside the base spread (base moved 0.5-0.7 % between its two starts). At 1 request they add no entry that fires, so their -0.1...-0.25 % there is the noise level.
- The 32-entry candidate gave -3.0 % at 4 requests, the gate alone -1.5 %. The groups do not add up to the difference one by one; I did not chase it.

## Final entry set (`final_entries.json`): gate only, 11 entries
- (1, 2560) at M = 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16. M = 7 dropped (never fires).
- **M = 10 and 14 swept and verified** with the W15 method (`sweep_m10_14/`): M = 10 1.84-2.06x, M = 14 1.70-1.79x over 4 seeds; error equal to cuBLAS.
- Dropped: router (all new M), (24, 2560) / (320, 2560) odd M, large projections M3, LM head M3 / M4.
- Final A/B (3 starts per arm, c1-c32), single-prompt greedy and GSM8K 8 + 8 are running on this set (local commit ec909d402c).

## #69907 interaction
#69907 changes the default backend inside the generic unquantized linear method (FlashInfer cute-dsl for M <= 32). The Flash-Next table replaces the layer's method with its own, which calls the table op directly, so layers whose shape is in the table are not routed by #69907. But today the gate (1, 2560) is **not** in the SM100 table, so after #69907 it would run on FlashInfer cute-dsl in the base. Adding the gate lines would then move it from that kernel to the skinny kernel at M <= 16, and to plain torch linear at M = 17-32. So if #69907 merges first, this A/B must be repeated against it; I have not measured that.
