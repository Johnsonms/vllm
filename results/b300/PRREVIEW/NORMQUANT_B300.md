# Norm + group-128 quant (owner-local `ce0594a919`, on #69778) — answers on B300

## Correction
My earlier "bit-identical greedy" was only 4 short prompts × 64 tokens plus decode-sized unit tests. It does not hold at prefill sizes.

## Elementwise (FlashInfer norm + `per_token_group_quant_fp8(use_ue8m0=False)` vs fused)
| M | input norm: bf16 diffs / FP8 code diffs / scale diffs | GDN (32 heads × 128): code diffs |
|---|---|---|
| 1, 37 | 0 / 0 / 0 | 0 |
| 300 | 7 / 2 / 0 (of 614k) | 0 |
| 2000 | 32 / 3 / 0 (of 4.1M) | 0 |
| 8192 | 128 / 13 / 1 (of 16.8M) | 5 (of 33.5M) |
So yes: your findings 1 and 2 reproduce on SM103 (input-norm reduction order at prefill M; GDN only at very large M).

## Greedy (32 single + 256 batched × 256 tokens, vs #69778)
- single: 3/32 identical (base-vs-base 32/32); batched 17/256 (base-vs-base 95/256).
- So it must be gated on GSM8K flips like the skinny GEMM. GSM8K 8×8 is running now (G4).

## Paired E2E vs #69778 (your method; TP1, 2 GPU-swapped passes × 6 rounds, AOT 0)
| c1 | c2 | c4 | c8 | c16 | c32 |
|---|---|---|---|---|---|
| −4.22% | −4.23% | −2.96% | −2.87% | −2.38% | −2.50% |
72/72 pairs faster. GSM8K in the E2E passes (single run each): A 0.9575 / 0.9629, B 0.9644 / 0.9644.

Versus your B200 prototype (all three sites: −5.3 / −5.2 / −4.2 / −3.6 / −4.3 / −2.7%), this branch lacks the attention sigmoid gate → `o_proj` site, which is the likely gap.
