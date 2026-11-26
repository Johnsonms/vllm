A=~/kt/p49v2 (d6bccd2252 Enable the small-M TRT-LLM block-FP8 GEMM path on)
B=~/kt/sk_nom8 (d5b0ff18ce [Perf] Add a CuTe DSL skinny block-FP8 GEMM to th)

| c | pass 1: A / B ms, Δ (pairs) | pass 2: A / B ms, Δ (pairs) | **mean Δ** |
|---|---|---|---|
| 4 | 2.945 / 2.833, -3.83% (-4.63..-2.73; 6/6 neg) | 2.955 / 2.811, -4.86% (-5.21..-4.54; 6/6 neg) | **-4.35%** |
| 8 | 3.458 / 3.464, +0.18% (-0.25..+0.40; 1/6 neg) | 3.470 / 3.458, -0.33% (-0.77..-0.03; 6/6 neg) | **-0.08%** |
| 16 | 4.492 / 4.303, -4.20% (-4.83..-3.97; 6/6 neg) | 4.484 / 4.300, -4.12% (-4.65..-3.66; 6/6 neg) | **-4.16%** |

pass 1: GSM8K A None B None; greedy tokens identical on 1/4 prompts

pass 2: GSM8K A None B None; greedy tokens identical on 1/4 prompts
