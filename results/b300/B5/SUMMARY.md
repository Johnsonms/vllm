# B5 — SM103 TP8 all-reduce + RMSNorm fusion cap, Qwen3-32B (2026-11-25)

**Recommendation: keep 4 MB.** On B300, raising the TP8 cap to 8, 16 or 32 MB gives no decode gain at any concurrency from 32 to 384. Every TPOT difference at c32–c256 is within ±2% and changes sign from arm to arm. GSM8K is unchanged.

**Why there is no gain:** on SM103 the default cap is already 4 MB, which is 409 tokens at hidden size 5120. Every pure-decode batch up to c384 is therefore already fused. On B200 the old cap was 1 MB (102 tokens), so #69543 moved batches of c128 and above from unfused to fused. Here there is nothing left to move; a bigger cap only affects steps that include a prefill chunk.

## Checks (one DEBUG launch per arm, untimed)
| arm | `Maximal number of tokens used by Flashinfer Allreduce Fusion` | backend | AOT loads | tracebacks |
|---|---|---|---|---|
| default (4 MB) | 409 | mnnvl | 0 | 0 |
| 8 MB | 819 | mnnvl | 0 | 0 |
| 16 MB | 1638 | mnnvl | 0 | 0 |
| 32 MB | 3276 | mnnvl | 0 | 0 |

`Enabled custom fusions: allreduce_rms` appears in every arm. The cap is set through `--compilation-config '{"pass_config":{"fi_allreduce_fusion_max_size_mb": X}}'`, with no code edits. `max_num_batched_tokens` = 16384, so it does not limit any arm.

## Per-run median TPOT (ms), random 512/512, `--ignore-eos`, `--num-prompts 2c`, `--num-warmups 8`
| run | c32 | c96 | c128 | c192 | c256 | c384 | TTFT c4, 1k prefill (ms) | GSM8K | AOT loads |
|---|---|---|---|---|---|---|---|---|---|
| r1_default | 4.39 | 5.45 | 5.76 | 7.54 | 8.77 | 13.50 | 55.8 | 0.941 | 0 |
| r1_8 | 4.39 | 5.50 | 5.88 | 7.46 | 8.87 | 13.88 | 55.9 | 0.945 | 0 |
| r1_16 | 4.47 | 5.33 | 5.82 | 7.52 | 8.48 | 13.83 | 54.7 | 0.944 | 0 |
| r1_32 | 4.36 | 5.39 | 5.88 | 7.45 | 8.55 | 14.47 | 53.4 | 0.944 | 0 |
| r2_default | 4.37 | 5.43 | 5.87 | 7.57 | 8.83 | 13.70 | 55.8 | - | 0 |
| r2_8 | 4.38 | 5.39 | 5.85 | 7.56 | 9.04 | 13.57 | 55.7 | - | 0 |
| r2_16 | 4.39 | 5.47 | 5.92 | 7.49 | 9.42 | 14.50 | 55.2 | - | 0 |
| r2_32 | 4.45 | 5.28 | 5.88 | 7.51 | 9.09 | 11.40 | 52.9 | - | 0 |

## Mean of 2 rounds, Δ vs default
| arm | c32 | c96 | c128 | c192 | c256 | c384 | TTFT c4 1k |
|---|---|---|---|---|---|---|---|
| 8 MB | +0.2% | +0.2% | +0.9% | −0.6% | +1.8% | +0.9% | +0.0% |
| 16 MB | +1.0% | −0.7% | +1.0% | −0.6% | +1.7% | +4.1% | −1.6% |
| 32 MB | +0.6% | −1.9% | +1.2% | −1.0% | +0.2% | −4.9% | −4.7% |

## Notes
- **c384 is noisy.** Each run is 768 prompts, only about 16 s. Within one arm the two rounds differ by up to 3 ms: 32 MB gave 14.47 and then 11.40, and 16 MB gave 13.83 and then 14.50. The −4.9% for 32 MB comes entirely from the single 11.40 run (its round 1 was the slowest of all runs), so I don't count it as a gain. A c384 call would need longer runs; I can do that if wanted.
- **The one consistent signal is in the prefill control.** TTFT for 1k-token prefill at c4 is −4.7% with 32 MB in both rounds (53.4 and 52.9 vs 55.8 and 55.8). This fits the mechanism: c4 × 1024 = 4096 prefill tokens, so only 32 MB (3276 tokens) fuses most of those steps. It is a 2–3 ms TTFT effect at small concurrency and does not show up in decode. If the manager cares about prefill-heavy TP8 traffic, a prefill-focused A/B of 4 vs 32 MB would be the follow-up. It is not part of this pick.
- **GSM8K** in round 1, every arm: 0.941 / 0.945 / 0.944 / 0.944. No failed requests and no tracebacks in any run.
- **Setup:** node idle at every check (other-user GPU check before each launch and each bench). Tree e66bede56b, clean; launched from inside the tree; `VLLM_DISABLE_COMPILE_CACHE=1`. Order: default, 8, 16, 32, then the same again.

## Files
`b5.sh` (runner), `analyze_b5.py`, `tables.md`, and `runs/r{1,2}_{arm}/` with the per-concurrency bench JSONs, `ttft_c4.json` and GSM8K summary JSONs.
