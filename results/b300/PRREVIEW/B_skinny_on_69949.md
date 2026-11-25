# (b) skinny block-FP8 GEMM on top of #69949 — evidence (B300 / SM103)

Candidate: owner-local commit `6a11c99a04` on top of #69949 `301212073e` + SM103 enable `7f2a66af41`. 22 plan entries; TP1 in_proj (N=12288) and qkv (N=9216) entries and all TP4 entries dropped because TRT-LLM wins there (see `a_rule_vs_plan_microbench.md`).

## Paired E2E vs #69949 (Qwen3.6-35B-A3B-FP8, 2 GPU-swapped passes × 6 rounds, AOT count 0 in all logs)
| arm | c1 | c2 | c4 | c8 | c16 | c32 | c64 |
|---|---|---|---|---|---|---|---|
| E12 TP1 | −10.14% | −8.66% | −8.45% | −2.85% | +0.04% | +0.00% | +0.01% |
| E13 TP2 | −6.41% | −8.35% | −4.38% | −1.46% | −4.05% | +0.04% | −0.34% |

- E12: c1–c8 12/12 pairs faster each; c16–c64 pair ranges straddle 0 (flat, no regression).
- E13: c1–c16 12/12 pairs faster each; c32/c64 flat.
- For reference, E11 (#69949 + SM103 enable vs main `e66bede56b`, TP1): −9.96 / −8.67 / −5.36 / −4.36 / −3.98 / −3.34 / −2.08% at c1–c64, 84/84 pairs faster.

## Correctness
- Not bit-identical (different accumulation order than CUTLASS).
- Greedy, 32 single + 256 batched × 256 tokens, vs #69949: single 4/32 identical (base-vs-base 32/32); batched 41/256 (base-vs-base floor only 55/256 — batched greedy is not deterministic here, so only the single set is a bit-identity test).
- GSM8K 8×8 (G3, skinny+TRT-LLM vs main): 0.9613 → 0.9627 (+0.0013, Welch t +0.97); flips A→B 17.0 per pair vs floors A/A 18.1, B/B 16.8; 0 questions right in all A and wrong in all B.
- Unit test `test_small_m_dispatch_uses_skinny_plan` (M ∈ {1,2,3,4,16,33}).
