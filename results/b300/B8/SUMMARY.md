# B8 — independent check of #69778 (SiLU-and-mul + block-FP8 quant fusion) on B300 (2026-11-25)

Base `fb159244e6`, PR `c5ef00f633`. Qwen3.6-35B-A3B-FP8, TP1, real weights. Each server ran from its own git worktree with `PYTHONPATH` set. The prebuilt ops were copied in: there is no csrc change between `e66bede56b` and `fb159244e6`, so the build artifacts match. Every server log shows the tree HEAD, the loaded `vllm/__init__.py`, and whether `fp8.py` has `expose_input_quant_key` (base: False, PR: True). AOT loads 0 and tracebacks 0 in all four server logs.

## Summary
- **The fusion fires on SM103** (step 1, below).
- **Speed:** TPOT −2.25 / −0.13 / −1.44 / −2.24% at c1/2/4/8 (paired, mean of two GPU-swapped passes). Smaller than B200's −3.8/−2.5/−2.4%. c2 is a wash: the two passes disagree in sign.
- **Accuracy:** GSM8K over 8 runs per arm, base **0.9636** (sd 0.0026) vs PR **0.9612** (sd 0.0020). That is −0.0024, about 3 of 1,319 questions; Welch t ≈ 2.1, p ≈ 0.06. B200 saw −0.0041, the same sign.
  - Base→PR flips run about 4 right→wrong per run pair above the A/A floor.
  - No question is lost systematically: none is right in all 8 base runs and wrong in all 8 PR runs, or the reverse. All movement is among 54 unstable questions.
- **Why the outputs move at all: the fused kernel is not bit-identical to the unfused chain.** Greedy tokens differ on 3 of 4 prompts (64 tokens) in both passes. `csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu` quantizes differently from `silu_and_mul` + `per_token_group_quant_fp8`:
  1. It quantizes the fp32 product `silu(gate) * up` (lines 60–61, and the vec kernel) without first rounding it to bf16. The unfused chain rounds the activation to bf16 before taking the group absmax and quantizing.
  2. Its scale floor is `min_scaling_factor` (1/(448·512)); the unfused kernel floors `absmax` at eps 1e-10.
  3. It quantizes with `x / scale`; the unfused kernel computes one `1/scale` per group and multiplies.

  (1) changes FP8 codes at rounding boundaries in ordinary groups; (2) and (3) matter only in edge cases.
- **Suggestion (not tested here; it needs a csrc build):** round `result` to the input dtype before the absmax and quant, and match (2)/(3). The fused path would then be bitwise the unfused chain and the accuracy question would disappear by construction. For comparison, the group-128 norm-quant fusion I wrote for the owner (local only) follows `per_token_group_quant_fp8` exactly and is bit-identical (greedy identical).

## 1. Does the fusion fire on SM103?
In-process load (`scripts/b8_check.py`), `language_model.model.layers.0.mlp.shared_expert.down_proj`:

| | quant_method / kernel | `_input_quant_key` | kernel `input_quant_key()` |
|---|---|---|---|
| base | `Fp8LinearMethod` / `CutlassFp8BlockScaledMMKernel` | None | None |
| PR | `Fp8LinearMethod` / `CutlassFp8BlockScaledMMKernel` | `QuantKey(float8_e4m3fn, fp32 dynamic, group (1, 128), symmetric)` | same |

Kernel counts in a profiled decode (4 prompts × 32 tokens), `check_{base,pr}.json`:

| kernel | base | PR |
|---|---|---|
| `act_and_mul_kernel<silu>` | 1280 | 0 |
| `per_token_group_quant_8bit<…, true, false, 128>` | 5120 | 3840 |
| `silu_and_mul_per_block_quant_kernel<…, true, 128>` | 0 | 1280 |

So one `act_and_mul` + quant pair per MoE layer per step (the shared-expert `down_proj` input) becomes one fused kernel.

## 2. Paired A/B (TPOT p50, ms)
Both servers up at once. For each c: 6 rounds of one burst per arm (5×c prompts, random 512/512, `--ignore-eos`, `--num-warmups c`), order flipped each round. Pass 1: base on GPU 0, PR on GPU 1; pass 2 swapped.

| c | pass 1 (base on GPU 0): base / PR, Δ (pair range) | pass 2 (base on GPU 1): base / PR, Δ (pair range) | **mean Δ** |
|---|---|---|---|
| 1 | 3.003 / 2.915, −2.94% (−3.06..−2.77; 6/6 neg) | 2.990 / 2.943, −1.57% (−1.72..−1.39; 6/6 neg) | **−2.25%** |
| 2 | 3.336 / 3.308, −0.83% (−1.18..−0.32; 6/6 neg) | 3.310 / 3.329, +0.57% (+0.10..+1.11; 0/6 neg) | **−0.13%** |
| 4 | 4.186 / 4.108, −1.88% (−2.20..−1.58; 6/6 neg) | 4.163 / 4.121, −0.99% (−1.17..−0.77; 6/6 neg) | **−1.44%** |
| 8 | 4.781 / 4.661, −2.52% (−3.12..−1.43; 6/6 neg) | 4.774 / 4.681, −1.96% (−2.15..−1.54; 6/6 neg) | **−2.24%** |

The pass-to-pass gap (~1 point) is larger than in B7. One GPU of the pair favours whichever arm runs on it; averaging the two passes removes that.

## 3. GSM8K, 8 runs per arm (B3 command), interleaved base/PR with both servers up; per-question dumps
| run | base | PR | invalid (base / PR) |
|---|---|---|---|
| p1g1 | 0.9606 | 0.9629 | 0 / 0 |
| p1g2 | 0.9621 | 0.9591 | 0 / 0 |
| p1g3 | 0.9636 | 0.9606 | 0 / 0 |
| p1g4 | 0.9659 | 0.9621 | 0 / 0 |
| p2g1 | 0.9651 | 0.9621 | 0 / 1 |
| p2g2 | 0.9682 | 0.9621 | 0 / 0 |
| p2g3 | 0.9613 | 0.9636 | 0 / 0 |
| p2g4 | 0.9621 | 0.9575 | 0 / 0 |
| **mean (sd)** | **0.9636** (0.0026) | **0.9612** (0.0020) | |

Per-question flips (counted over every run pair):

| comparison | right→wrong | wrong→right | total (range) | pairs |
|---|---|---|---|---|
| base → PR | 11.8 | 8.7 | 20.6 (11–26) | 64 |
| base → base (A/A floor) | 7.8 | 8.7 | 16.5 (11–21) | 28 |
| PR → PR (A/A) | 8.6 | 8.1 | 16.7 (10–21) | 28 |

- Always right: base 1244, PR 1242. Always wrong: 35 in both. Unstable in either arm: 54.
- Right in all 8 base runs and wrong in all 8 PR runs: **0**; the reverse: **0**.
- Temperature-0 GSM8K is not run-to-run deterministic on either arm (batch-dependent kernels), hence the A/A floor of ~16 flips.

Reading: the PR's flips run about 4 right→wrong per pair above the noise floor, concentrated in near-tie questions. That fits a small numerics change (point 1 above), not a bug that breaks specific problems. The size (≈3 questions here, ≈5 on B200) is at the edge of what 8 runs per arm can resolve.

## Files
`b8_tables.md`, `paired_runs.json` (per-burst TPOT), `gsm8k_perq.json` (per-run, per-question correctness), `check_{base,pr}.json`, `scripts/` (`b8_paired.sh`, `b8_check.py`, `gsm8k_perq.py` — wraps `_score_gsm8k` without a vLLM edit — and `analyze_b8.py`).
