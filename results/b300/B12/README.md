# B12 — #70028 re-test at b8b5ff3926 (after rebase onto main 269fca48a1)

Qwen3.6-35B-A3B-FP8, TP1, real weights, B300 (SM103). Arms: A = base `269fca48a1`, B = PR `b8b5ff3926`.
`git patch-id --stable` of `b8b5ff3926` equals the B1 commit `e77ca1439c`.

- pytest (`tests/models/qwen3_5/test_gemma_add_rms_norm_quant.py tests/fusion/test_quant_activation_contract.py`): base 108 passed, PR 144 passed (+36 new group-128 tests), 0 failed.
- Fusion (eager, 4 prompts x 16 tokens, `count_norm.py`): PR calls `gemma_add_rms_norm_quant` with key `kFp8Dynamic128Sym` 640 times (40 layers x 16 steps). The CUTLASS block-FP8 linear gets a pre-quantized input for (12288, 2048) GDN in_proj_qkvz 480x, (9216, 2048) attn qkv_proj 160x and (2048, 512) shared down_proj 640x. Base: only the (2048, 512) 640x from #69778. Same greedy text.
- Paired TPOT (median; 1 pass x 4 rounds, 512/512, --ignore-eos, both servers up at once, GPUs 0/1; "Directly load AOT" = 0 in both logs):

| c | base -> PR (ms) | delta | pairs faster |
|---:|---|---:|---:|
| 1 | 2.891 -> 2.761 | -4.49% | 4/4 |
| 4 | 3.721 -> 3.621 | -2.68% | 4/4 |
| 8 | 4.511 -> 4.391 | -2.65% | 4/4 |
| 32 | 6.876 -> 6.853 | -0.33% | 3/4 |

- GSM8K 1+1 (greedy, 1319 q): base 0.9583, PR 0.9636; 23 per-question flips (8 base-only right, 15 PR-only right), in line with the B1 run-to-run noise (~17 flips base/base, sd 0.002).
- Greedy check (4 prompts, 64 tokens): 3/4 identical, 1 diverges at token 7 (expected: norm reduction order differs, as in B1).
