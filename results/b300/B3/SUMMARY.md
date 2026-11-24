# B3 — E2E serving A/B on B300 for the shared SM100 skinny table at `(10, 3)`

Node: 8x B300 (SM103). Qwen3.6-35B-A3B-FP8, **real weights**, `--language-model-only`.
All runs were on an idle node; no other GPU users appeared at any check.
Status: **done (TP1 c1/c2/c4/c8, plus the optional TP2 c1/c8).**

## Result
- **TP1: TPOT −2.4% (c1), −3.2% (c2), −11.5% (c4), −3.7% (c8).** Clean A/B/A/B; the per-run spread within an arm is ≤ 0.03 ms.
- **TP2: TPOT −3.8% (c1), −5.1% (c8).**
  - The two c1 base runs differ by 3% (2.989 / 2.899 ms), so c1 at TP2 is −2.3% to −5.2% depending on the base run.
- **GSM8K unchanged** (1,319 questions, 0 invalid responses):
  - TP1: base 0.964 / 0.958, pr 0.964 / 0.964.
  - TP2: base 0.961 / 0.958, pr 0.959 / 0.962.
- For comparison, #69736 on B200 gave −5.7% at c4 and −4.7% at c8.

## Arms and code check
- `base` = `e66bede56b`, unmodified.
- `pr` = `81159c69d6` (#69736, which only adds `SM100_SKINNY_GEMM_PLANS` in `skinny_gemm_plans.py`) **plus one local line**: `(10, 3)` was added to `SKINNY_GEMM_PLANS_BY_CAPABILITY`, mapped to the same `SM100_SKINNY_GEMM_PLANS` as `(10, 0)`.
  - That edit was not committed or pushed anywhere.
- The arms differ only in that one file. Both run in one tree: the arm's version of the file is copied in before the server starts, and the tree was restored to base afterwards (`git status` clean).
- Each server log starts with these lines. Paths are relative to the vLLM tree; the server is launched from inside it.

```
base: CHECK plans file vllm/model_executor/kernels/linear/cute_dsl/skinny_gemm_plans.py sha1 c41c6f99d66a
      CHECK (10, 3) in SKINNY_GEMM_PLANS_BY_CAPABILITY: False
      CHECK keys [(12, 1)]
      CHECK plan(256, 2048) on this GPU: False
pr:   CHECK plans file vllm/model_executor/kernels/linear/cute_dsl/skinny_gemm_plans.py sha1 9d820fa9816e
      CHECK (10, 3) in SKINNY_GEMM_PLANS_BY_CAPABILITY: True
      CHECK keys [(10, 0), (10, 3), (12, 1)]
      CHECK plan(256, 2048) on this GPU: True
```

- In every run, the same lines appeared for its arm, `Directly load AOT` = 0, and the Traceback count = 0.

## Method
- **Server:**
  - Command: `vllm serve Qwen/Qwen3.6-35B-A3B-FP8 -tp {1,2} --language-model-only --max-num-seqs 256 --max-model-len 8192 --gpu-memory-utilization 0.9 --no-enable-prefix-caching`.
  - Environment: `VLLM_DISABLE_COMPILE_CACHE=1`.
  - GPUs: GPU 0 for TP1, GPUs 0-1 for TP2.
- **Bench:** `vllm bench serve` with random 512 in / 512 out, `--ignore-eos`, `--num-prompts max(4c, 16)`, `--num-warmups c`, `--seed 0`. The table reports median TPOT.
- **GSM8K:** `tests/evals/gsm8k/gsm8k_eval.py --use-chat-completions --chat-template-kwargs '{"enable_thinking": false}' --max-tokens 1024 --temperature 0 --max-concurrency 64`, run on the same server after the benches.
- **Order:** base, pr, base, pr; one server start per run.
- **Step-back check:** before every server start and every bench, the script checked for other users on the GPUs. It never triggered.
- **Script:** `b3_ab.sh`. Compact per-run numbers are in `runs.json`.

## Per-run TPOT and GSM8K

#### TP1 set 2 (main)

| run | c1 TPOT (ms) | c2 TPOT (ms) | c4 TPOT (ms) | c8 TPOT (ms) | GSM8K |
|---|---|---|---|---|---|
| r1_base | 3.002 | 3.339 | 4.179 | 4.750 | 0.964 |
| r2_pr | 2.929 | 3.236 | 3.704 | 4.597 | 0.964 |
| r3_base | 2.998 | 3.339 | 4.185 | 4.775 | 0.958 |
| r4_pr | 2.928 | 3.226 | 3.702 | 4.571 | 0.964 |
| **base mean** | 3.000 | 3.339 | 4.182 | 4.762 | 0.961 |
| **pr mean** | 2.928 | 3.231 | 3.703 | 4.584 | 0.964 |
| **Δ pr vs base** | **-2.4%** | **-3.2%** | **-11.5%** | **-3.7%** | |

#### TP1 set 1 (supplementary; see note)

| run | c1 TPOT (ms) | c2 TPOT (ms) | c4 TPOT (ms) | c8 TPOT (ms) | GSM8K |
|---|---|---|---|---|---|
| r1_base | 2.999 | 3.339 | 4.182 | 4.790 | 0.963 |
| r3_base | 2.999 | 3.339 | 4.175 | 4.743 | 0.962 |
| r4_pr | 2.929 | 3.227 | 3.726 | 4.554 | 0.961 |
| **base mean** | 2.999 | 3.339 | 4.178 | 4.767 | 0.962 |
| **pr mean** | 2.929 | 3.227 | 3.726 | 4.554 | 0.961 |
| **Δ pr vs base** | **-2.3%** | **-3.3%** | **-10.8%** | **-4.5%** | |

#### TP2 (optional task)

| run | c1 TPOT (ms) | c8 TPOT (ms) | GSM8K |
|---|---|---|---|
| r1_base | 2.989 | 3.977 | 0.961 |
| r2_pr | 2.832 | 3.766 | 0.959 |
| r3_base | 2.899 | 3.956 | 0.958 |
| r4_pr | 2.833 | 3.760 | 0.962 |
| **base mean** | 2.944 | 3.966 | 0.959 |
| **pr mean** | 2.833 | 3.763 | 0.961 |
| **Δ pr vs base** | **-3.8%** | **-5.1%** | |

**Note on set 1:** my first TP1 set had a script bug. The server-cleanup function reused the run-counter variable, so round 4 (pr) overwrote round 2 (pr).
- The three surviving runs (base r1, base r3, pr r4) are valid, with correct code checks. They agree with set 2 to within 0.8 points.
- Set 2 is a full clean rerun after the fix.

## Observation: c4 at TP1
- The pr gain at c4 (−11.5%, reproduced in both sets: −10.8% and −11.5%) is much larger than at c2 or c8.
- Base TPOT rises 0.84 ms from c2 to c4 but only 0.58 ms from c4 to c8. With the plan, the steps are 0.47 and 0.88 ms.
- So base looks unusually slow at batch 4 on SM103, and the skinny plan bypasses it.
- In isolation, cuBLAS is not slow at M=4 for these shapes (B2 sweep: router 5.28 us at M=4 vs 5.25 at M=2; LM head 166.4 vs 166.1 us). The extra time must come from how cuBLAS behaves inside the served graph at batch 4, or from another kernel.
- **Not profiled.** A c4 decode profile of both arms (B1 method, about 15 min) would show which kernel it is. This could matter beyond these shapes, for other BF16 GEMMs at M=4 on SM103. Proposed as a follow-up if useful.

## Files
- `SUMMARY.md`: this report.
- `runs.json`: per-run TPOT, ITL, TTFT, throughput and GSM8K for all 12 runs.
- `b3_ab.sh`: the A/B script, with the counter fix applied.
- `analyze.py`: builds the tables above.
