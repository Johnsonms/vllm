A=~/kt/sk49v2 (d5b0ff18ce [Perf] Add a CuTe DSL skinny block-FP8 GEMM to th)
B=~/kt/sk_nom8 (d5b0ff18ce [Perf] Add a CuTe DSL skinny block-FP8 GEMM to th)

| c | pass 1: A / B ms, Δ (pairs) | pass 2: A / B ms, Δ (pairs) | **mean Δ** |
|---|---|---|---|
| 4 | 2.801 / 2.822, +0.74% (+0.23..+1.37; 0/6 neg) | 2.811 / 2.802, -0.33% (-0.68..+0.20; 5/6 neg) | **+0.21%** |
| 8 | 3.547 / 3.463, -2.36% (-2.52..-2.15; 6/6 neg) | 3.542 / 3.449, -2.64% (-2.89..-2.18; 6/6 neg) | **-2.50%** |
| 16 | 4.295 / 4.305, +0.25% (-0.97..+1.46; 2/6 neg) | 4.283 / 4.296, +0.31% (-0.13..+0.89; 1/6 neg) | **+0.28%** |

pass 1: GSM8K A None B None; greedy tokens identical on 4/4 prompts

pass 2: GSM8K A None B None; greedy tokens identical on 4/4 prompts
