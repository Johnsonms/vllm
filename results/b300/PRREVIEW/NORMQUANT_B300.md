# Norm + group-128 quant — answers on B300

## Redo on the #69778 head (2026-11-25) — supersedes the numbers further down
- **A** = `152bca1e65` (#69778 head; `_C` rebuilt from source, #69778's own test: 443 passed).
- **B** = A + the norm sites (cherry-pick of `ce0594a919`) + **the attention sigmoid gate → `o_proj` site** (new, owner-local `1a7e2399cc`).
  - Gate site: `fused_sigmoid_gate_mul_group_quant` (Triton). It computes `y = x*sigmoid(g)` rounded to bf16, then group-128 FP8 quant with column-major scales; `qwen3_next.py` uses it when `o_proj`'s input quant key is `kFp8Dynamic128Sym`.
  - Unit test vs unfused `fused_sigmoid_gate_mul` + `per_token_group_quant_fp8`: `torch.equal` on q and scale, 40/40; related tests 665 passed; pre-commit clean.

### Greedy (32 single + 256 batched × 256 tokens; one batched request after a throwaway warm-up)
| | single | batched |
|---|---|---|
| A vs A (floor) | 32/32 | 256/256 |
| A vs B | 3/32 (first diff median 36) | 23/256 (median 46) |
Expected: the input-norm sites are not bit-identical at prefill M (see below), so this is gated on GSM8K.

### GSM8K, 8 × 2 runs, GPU-swapped halves (4 runs on GPU1/2, then 4 swapped)
| | runs | mean | sd |
|---|---|---|---|
| A | 0.9621 0.9591 0.9636 0.9636 | 0.9621 | 0.0021 |
| B | 0.9613 0.9621 0.9606 0.9621 | 0.9615 | 0.0007 |
- Δ B−A = −0.06 pp (Welch t −0.50).
- Flips A↔B 20.0 per pair vs floors A↔A 15.3 and B↔B 17.5. Right in all A and wrong in all B: 1; reverse: 0.
- No accuracy change beyond noise.

### Paired E2E vs A (TP1 Qwen3.6-35B-A3B-FP8, 2 GPU-swapped passes × 6 rounds, AOT 0)
| c1 | c2 | c4 | c8 | c16 | c32 |
|---|---|---|---|---|---|
| **−4.95%** | **−4.48%** | **−3.14%** | **−3.67%** | **−3.33%** | **−2.92%** |
- 72/72 pairs faster; the two passes agree within 0.8 pp.
- Your B200 prototype (all three sites): −5.3 / −5.2 / −4.2 / −3.6 / −4.3 / −2.7%. Old run here (c5ef base, no gate site): −4.2 / −4.2 / −3.0 / −2.9 / −2.4 / −2.5%; the gate site adds about 0.1–0.9 pp per concurrency, and the c16 gap to B200 remains (−3.3 vs −4.3%).

---
## Earlier run (base c5ef00f633, no gate site, GPUs not swapped) — kept for reference

### Correction
My earlier "bit-identical greedy" was only 4 short prompts × 64 tokens plus decode-sized unit tests. It does not hold at prefill sizes.

### Elementwise (FlashInfer norm + `per_token_group_quant_fp8(use_ue8m0=False)` vs fused)
| M | input norm: bf16 diffs / FP8 code diffs / scale diffs | GDN (32 heads × 128): code diffs |
|---|---|---|
| 1, 37 | 0 / 0 / 0 | 0 |
| 300 | 7 / 2 / 0 (of 614k) | 0 |
| 2000 | 32 / 3 / 0 (of 4.1M) | 0 |
| 8192 | 128 / 13 / 1 (of 16.8M) | 5 (of 33.5M) |
So yes: your findings 1 and 2 reproduce on SM103 (input-norm reduction order at prefill M; GDN only at very large M).

### Greedy (old, async batched)
- single: 3/32 identical (base-vs-base 32/32); batched 17/256 (base-vs-base 95/256).
- So it must be gated on GSM8K flips like the skinny GEMM. GSM8K 8×8 is running now (G4).

### Paired E2E (old; your method; TP1, 2 GPU-swapped passes × 6 rounds, AOT 0)
| c1 | c2 | c4 | c8 | c16 | c32 |
|---|---|---|---|---|---|
| −4.22% | −4.23% | −2.96% | −2.87% | −2.38% | −2.50% |
72/72 pairs faster. GSM8K in the E2E passes (single run each): A 0.9575 / 0.9629, B 0.9644 / 0.9644.

Versus your B200 prototype (all three sites: −5.3 / −5.2 / −4.2 / −3.6 / −4.3 / −2.7%), this branch lacks the attention sigmoid gate → `o_proj` site, which is the likely gap.
