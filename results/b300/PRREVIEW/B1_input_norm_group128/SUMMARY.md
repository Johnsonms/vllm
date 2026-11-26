# B1 — input RMSNorm + group-128 FP8 quant (PRREVIEW)

- **Title:** see `title.txt`. **Body:** `body.md` (exact text to post). **Diffstat:** `diffstat.txt`.
- **Branch:** owner-local `kopt/input-norm-group128-quant`, to push as `qwen35-input-norm-group128-quant` on the owner's fork.
- **Head:** `e77ca1439c` (full SHA in `diffstat.txt`), on #69778 head `152bca1e65` (main `fb159244e6`).
- **Open as:** draft. Mark ready after #69778 merges.

## PR_RULES checklist
| § | Item | Status |
|---|---|---|
| 1 | Owner approval | Delegated per 2d5cc9bc01; waiting for `APPROVED TO OPEN` |
| 2 | Dup check | #65497 / #36016 / #45770 are compile-pass only (no effect, mode NONE); #69812 is per-token GDN/gate (independent). Nothing covers the input norm with group-128. |
| 3 | Identity | `Johnsonms <owner email> \| Johnsonms <owner email>`, `Signed-off-by`, no Co-authored-by. Rechecked after the commit. |
| 3 | pre-commit | passes, no files modified |
| 4.1 | Unit tests | base 819 passed, PR 855 passed (+36 new). The new tests on base code: 36 failed. TP2 AR test: 1 passed on both. |
| 4.2 | Accuracy | Not bit-identical (FlashInfer vs Triton norm reduction order at prefill M). GSM8K 16+16 (4 groups, GPU-swapped): Δ −0.05 pp, t −0.81; flips 23.4 vs floors 17.1 / 16.2; 0 / 1 systematic. Greedy 3/32, 19/256 (floor 32/32, 176/256). |
| 4.3 | Microbench | 1.25–2.55× at M 1–8192, none slower (CUDA graphs, L2 flush, 7 interleaved rounds × 3 GPUs). |
| 4.4 | E2E paired | −2.56 / −1.92 / −0.70 / −1.70 / −0.78 / −1.31% at c1–c32, 72/72 pairs, AOT 0. |
| 4.5 | Not measured | SM100 E2E of this branch, TP>1 E2E, other Qwen3.5 sizes. Built normally (`_C` from source, `TORCH_CUDA_ARCH_LIST=10.0`). |
| 5 | Body format | Overview / Results / Testing / Related PRs / AI assistance |

## Notes for the manager
1. **GSM8K history, not hidden.**
   - The first 8+8 read Δ −0.18 pp (t −3.2).
   - A second 8+8 read +0.08 pp, and the combined 16+16 is −0.05 pp.
   - The base itself moved 0.20 pp between the two sets (0.9628 vs 0.9608).
   - The body reports the combined 16+16 only. Say if you want both sets listed.
2. **Change vs the cherry-picked owner commit:** the AR path passes `use_ue8m0=False` explicitly. The default follows `is_deep_gemm_e8m0_used()`, while the CUTLASS linear's own `QuantFP8` sets False.
3. **Attribution conflict — needs the owner's call.**
   - PR_RULES §5.5 asks for the line `🤖 Generated with [Claude Code](...)` at the end of the body.
   - The owner earlier told me directly not to add that footer to PRs; they had it removed from a previous PR.
   - `body.md` keeps the one-sentence AI-assistance disclosure and leaves out the footer. If the owner wants the footer, I will add it before opening.
4. Raw data on the node: `e2e/E17_b1`, `gsm/G6_b1_h{1,2}`, `gsm/G8_b1_h{1,2}`, `greedy/G7_b1`, `ub_nq/final_gpu{1,2,3}.json`.
