# B9: #69778 sampled GSM8K, main vs v1 vs v2 (2026-11-26)

**Setup.** Qwen3.6-35B-A3B-FP8, TP=1, 1× B300 per server. The three servers run at the same time. Serve flags: `--language-model-only --max-num-seqs 256 --max-model-len 8192 --gpu-memory-utilization 0.9 --no-enable-prefix-caching`, with `VLLM_DISABLE_COMPILE_CACHE=1`.

**Arms:**
- main `fb159244e6`;
- v1 `c5ef00f633` (fp32 product);
- v2 `152bca1e65` (bit-identical to main under greedy).

**Eval.** `gsm8k_eval.py --use-chat-completions --chat-template-kwargs '{"enable_thinking": false}' --max-tokens 1024 --max-concurrency 64 --temperature 0.6 --top-p 0.95 --seed S`, 1,319 questions.
- 8 seeds (101–108), the same set for every arm. The run order rotates per seed.
- Seeds 101–104 ran with the arms on GPUs 0/1/2, and seeds 105–108 on GPUs 4/5/3, so each arm used two GPUs.
- AOT 0, no tracebacks, 0 invalid answers.

## Accuracy
| arm | mean | sd | Δ vs main | Welch t | paired-by-seed t |
|---|---:|---:|---:|---:|---:|
| main | 0.9594 | 0.0027 | | | |
| v1 | 0.9581 | 0.0033 | −0.13 pp | −0.87 | −1.37 |
| v2 | 0.9567 | 0.0024 | −0.27 pp | −2.13 | −2.25 |

v2 vs v1: −0.14 pp, Welch t −0.98. Per-seed values are in `analysis.txt`.

## Per question
Each arm has 8 runs, so each question gets a correct count from 0 to 8 per arm.

| pair | questions whose correct counts differ by ≥ 4 | null (labels permuted), mean / 95th pct |
|---|---:|---:|
| v1 vs main | 0 | 2.4 / 5 |
| v2 vs main | 1 (better in v2) | 3.0 / 7 |

No question is systematically worse in either PR version.

## Reading
- **v1 is within noise of main:** −0.13 pp, t −0.87, and no question shifts.
- **The v2–main gap is the noise scale here.** v2 computes the same numbers as main, yet it reads −0.27 pp (t −2.13).
  - Same-seed outputs of main and v2 have the same length on only 656–693 of 1,319 questions. Batch composition at concurrency 64 changes the trajectories even with identical math.
  - So a ~0.3 pp, |t| ~ 2 gap over 8 runs can arise with no numerical change at all.
  - v1's gap is half that.
- **On the reviewer's hypothesis:** we see no sign that the original (unfused) numerics favour greedy decoding. Under sampling, v1 is not worse than v2 (+0.14 pp, t +0.98).
- Same-seed output lengths match for main/v1 on 549–593 questions, against 656–693 for main/v2. v1 does change trajectories somewhat more, as expected from the different rounding, but not accuracy.

## Paste-ready setup paragraph
> Sampled GSM8K on Qwen3.6-35B-A3B-FP8, TP=1, one B300 per server, the three servers running at the same time. Command: `tests/evals/gsm8k/gsm8k_eval.py --use-chat-completions --chat-template-kwargs '{"enable_thinking": false}' --max-tokens 1024 --max-concurrency 64 --temperature 0.6 --top-p 0.95 --seed S`, 1,319 questions. Each arm used the same 8 seeds, with the GPU assignment rotated halfway. main `fb159244e6`: 0.9594 (sd 0.0027). v1 `c5ef00f633` (fp32 product): 0.9581 (sd 0.0033), −0.13 pp, Welch t −0.87. v2 `152bca1e65` (bit-identical): 0.9567 (sd 0.0024), −0.27 pp, t −2.13. v2 computes the same numbers as main, so its gap shows the run-to-run noise from batch composition under sampling. No question's correct count (out of 8) differs by 4 or more between v1 and main; a label-permutation null gives 2.4 on average. So v1 is within noise of main, and we see no evidence that the original numerics favour greedy decoding.
