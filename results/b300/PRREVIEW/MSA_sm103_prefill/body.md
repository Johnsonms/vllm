## Changes

SM103 (B300 / GB300) has two softmax-relevant differences from SM100 that the Blackwell sparse prefill (`blackwell_prefill/atten_fwd_sm100.py`, the MiniMax-M3 MSA prefill path) does not use yet:

- **MUFU.EX2 at twice the SM100 rate.** The causal kernel emulates part of `exp2` with an FFMA polynomial (`ex2_emu_freq = 16`) to offload MUFU. On SM103 that no longer pays off. FlashAttention-4 turns the emulation off on SM103 for the same reason (Dao-AILab/flash-attention 8e0b5d7, `select_ex2_emu`).
- **`tcgen05.ld.red`.** The TMEM load of S can return each row's max along with the values, which removes the per-row `fmax` reduction. FlashAttention-4 uses it in its SM100-family forward (Dao-AILab/flash-attention #2696, `use_ldred_rowmax`).

This PR, on the SM103 family only (`sm_103f`, excluding SM107 / Rubin, which I could not test):
- sets `ex2_emu_freq = 0`, so every element uses hardware `exp2`;
- loads S with `LdRed32x32bOp(MAX)`. The hardware row max is used only when no mask changed the fragment: a full KV block and, for causal, no masked token in the group. Otherwise the kernel falls back to the software max. The code follows the existing Rubin ld.red path in `sm100/fwd/atten_fwd.py`.

SM100 and SM107 code paths are unchanged. One file, +53/−7.

## Correctness

- **Tests.** `pytest -q python/fmha_sm100/cute/test_sparse_atten.py` on SM103 (B300): 1173 passed, 143 skipped, the same as `dev`.
- **ld.red.** The ld.red max is exact.
- **Hardware exp2.** It rounds differently from the polynomial. About 5% of the bf16 outputs move by one ulp (max abs difference 0.0039–0.0078 on the cases below).
- **Reference.** Against the fp32 PyTorch reference (`_reference_varlen_forward`, upcast inputs), the max abs and mean abs errors of this PR and of `dev` are equal within 1e-8.

## Performance

**Setup.**
- 1× B300 (SM103), MiniMax-M3 prefill shapes: topk 16, `blk_kv` 128, dim 128, causal, bf16, paged KV (page 128, shuffled page table), inputs from the test's `_build_paged_inputs`.
- Called through `sparse_atten_func` with a schedule and page table, as vLLM's MiniMax-M3 backend does. The harness asserts that `blackwell_prefill.run_pagekv` ran.
- One CUDA-graph replay per sample, with L2 flushed before each replay, median of 20. Three rounds alternate `dev` and this PR on the same GPU.

| case | `dev` ms | PR ms | Δ (range over rounds) |
|---|---:|---:|---|
| b1, sq = skv = 4096, hkv 4, 16 q/kv | 0.506 | 0.455 | −10.1% (−10.2..−10.1) |
| b1, sq = skv = 8192, hkv 4, 16 q/kv | 1.040 | 0.967 | −7.1% (−7.1..−6.9) |
| b1, sq = skv = 16384, hkv 4, 16 q/kv, sink | 2.028 | 1.888 | −6.9% (−6.9..−6.8) |
| b1, sq = skv = 32768, hkv 4, 16 q/kv, sink | 4.205 | 3.893 | −7.4% (−7.7..−6.7) |
| b4, sq 4096, skv 16384, hkv 4, 16 q/kv | 1.524 | 1.393 | −8.6% (−8.7..−8.5) |
| b1, sq = skv = 8192, hkv 8, 8 q/kv | 1.029 | 0.897 | −12.9% (−12.9..−12.5) |

**Ablation.** This was measured earlier with back-to-back CUDA-event timing, not graphs, so the absolute numbers are a little higher:

| change | Δ vs `dev` |
|---|---|
| both changes | −8.6..−14.2% |
| hardware `exp2` only | −8.2..−12.5% |
| ld.red only | −6.4..−9.6% |

The two do not add up, because the softmax warp group is the bottleneck and either change relieves most of it.

**Not measured.**
- End to end: I do not have MiniMax-M3 weights.
- SM100 (code path unchanged) and SM107 (excluded).
