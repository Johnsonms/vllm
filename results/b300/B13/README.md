# B13 — QuACK prototypes on SM103 (B300)

Base: vLLM main `14e9c30d86`, quack-kernels 0.6.5. P1/P2/P3 rebuilt locally from the TASKS description (code kept local), each behind an env flag; one tree serves both arms.
Builds: SM103 needed no workaround. The pinned P2 config (128x256, cluster 2x1, dynamic persistent, no swap_ab/pingpong/TMA gather, `device_capacity` 10) is valid on SM103; the FP32-values TopK subclass compiles and runs as is.

## Kernels (cold L2, CUDA graph, median of 7 rounds; `mb_p*.json`)
- P1 vs `ops.topk_softmax` (E=128, k=8): ids identical at every M; weights within 1.2e-7. M=1 1.91x, M=8-256 1.45-1.66x, 512 1.77x, 1024 2.09x, 8192 3.70x.
- P2 vs `fused_experts` (Triton; no B300 tuned config file present), Qwen3-30B-A3B shapes: M=64 1.06x, 256 1.08x, 1024 1.14x, 2048 1.30x, 4096 1.45x, 8192 1.58x. Rel. error vs FP32: QuACK 0.0029, Triton 0.0043 (QuACK vs Triton 0.0051).
- P3 vs `F.linear` + `silu_and_mul` (Qwen3-8B gate_up): M=1/8 1.05x (F.linear + 2-D Triton SiLU-mul), 16/32 1.06x, 128 1.20x, 512 1.23x, 2048 (untuned) 1.15x. Rel. error vs FP32: QuACK 0.0017, cuBLAS+C++ 0.0035.

## Paths fire in the server code (`path_counts.txt`, eager, 1 prompt of ~1500 tokens + 1 short, 8 tokens)
P1 topk 384x (48 layers x 8 steps); P2 experts 48x (prefill only, M=1506 >= 1024); P3 `gemm_gated_out` 36x on the prefill (M=1506), 2-D SiLU-mul 252x on decode (M<16).

## Paired E2E (2 passes x 6 rounds, GPUs swapped in pass 2; mean of the two pass deltas; pairs where the flag arm is better)
All three experiments ran at the same time on GPU pairs 0/1, 2/3, 4/5. 512/512 random for TPOT, 4096/1 for TTFT, `--ignore-eos`, 5 x c prompts per burst.

**P1, Qwen3-30B-A3B BF16, `--moe-backend triton`, median TPOT**
| c | pass 1 | pass 2 | mean | pairs |
|---:|---|---|---:|---:|
| 1 | 3.338 -> 3.199 ms (-4.18%) | 3.317 -> 3.244 (-2.19%) | **-3.18%** | 12/12 |
| 8 | 5.842 -> 5.785 (-0.96%) | 5.880 -> 5.684 (-3.33%) | **-2.15%** | 6/12 |
| 32 | 8.582 -> 8.360 (-2.59%) | 8.538 -> 8.533 (-0.06%) | **-1.32%** | 9/12 |
| 64 | 10.821 -> 10.707 (-1.06%) | 10.802 -> 10.683 (-1.10%) | **-1.08%** | 8/12 |

**P2, same model/backend, median TTFT (4096 in, 1 out)**
| c | pass 1 | pass 2 | mean | pairs |
|---:|---|---|---:|---:|
| 1 | 61.88 -> 57.21 ms (-7.55%) | 61.35 -> 55.40 (-9.70%) | **-8.62%** | 12/12 |
| 4 | 166.3 -> 146.5 (-11.93%) | 168.8 -> 145.4 (-13.85%) | **-12.89%** | 12/12 |
| 16 | 606.1 -> 523.3 (-13.65%) | 616.9 -> 514.8 (-16.56%) | **-15.11%** | 12/12 |

**P3, Qwen3-8B BF16, default backend, median TPOT** (QuACK disk cache pre-tuned at M=16..512 step 8 before the servers started)
| c | pass 1 | pass 2 | mean | pairs |
|---:|---|---|---:|---:|
| 1 | 3.818 -> 3.859 (+1.06%) | 3.856 -> 3.806 (-1.29%) | -0.11% | 6/12 |
| 8 | 3.953 -> 3.959 (+0.15%) | 3.962 -> 3.951 (-0.26%) | -0.06% | 6/12 |
| 32 | 4.582 -> 4.595 (+0.30%) | 4.555 -> 4.589 (+0.74%) | **+0.52%** | 1/12 |
| 128 | 7.249 -> 7.009 (-3.31%) | 7.174 -> 7.018 (-2.18%) | **-2.75%** | 12/12 |

## Accuracy
GSM8K (1319 q, greedy, chat, thinking off, once per arm). Per-question flips as (base-only right / flag-only right). The two Qwen3-30B-A3B base arms (P1 A, P2 A: same code, flags off) differ on 14 questions (0.9310 vs 0.9280), so ~0.3 pt is run-to-run noise here.
- P1: 0.9310 -> 0.9280 (11 / 7). Greedy 3/4 prompts identical over 64 tokens, 1 diverges at token 60.
- P2: 0.9280 -> 0.9212 (15 / 6), -0.68 pt, 21 flips. Larger than the base/base gap of 14; one run per arm, so not conclusive. P2 only engages on prefill batches >= 1024 tokens (GSM8K at concurrency 64 batches prompts). Greedy 4/4 identical (short prompts, below the 1024 threshold).
- P3: 0.9022 -> 0.9052 (8 / 12). Greedy 4/4 identical.

## Notes
- SM103 vs B200: P1 and P2 carry over (P2 TTFT gains are larger here than B200's -6.0/-8.1/-7.6%). P3 matches B200 at c1/c8 (flat) and c128 (-2.75% vs -2.9%); c32 is +0.5% here (B200 -0.95%).
- P1 does not apply the MoE padding mask (`VLLM_MOE_SKIP_PADDING`, default on): padded CUDA-graph rows get real expert ids instead of -1. Base masks them. Kept as described; worth a note if this goes further.
