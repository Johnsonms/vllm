| c | pass 1 (base on GPU 0): base / PR ms, Δ (pairs) | pass 2 (base on GPU 1): base / PR ms, Δ (pairs) | **mean Δ** |
|---|---|---|---|
| 1 | 3.003 / 2.915, -2.94% (-3.06..-2.77; 6/6 neg) | 2.990 / 2.943, -1.57% (-1.72..-1.39; 6/6 neg) | **-2.25%** |
| 2 | 3.336 / 3.308, -0.83% (-1.18..-0.32; 6/6 neg) | 3.310 / 3.329, +0.57% (+0.10..+1.11; 0/6 neg) | **-0.13%** |
| 4 | 4.186 / 4.108, -1.88% (-2.20..-1.58; 6/6 neg) | 4.163 / 4.121, -0.99% (-1.17..-0.77; 6/6 neg) | **-1.44%** |
| 8 | 4.781 / 4.661, -2.52% (-3.12..-1.43; 6/6 neg) | 4.774 / 4.681, -1.96% (-2.15..-1.54; 6/6 neg) | **-2.24%** |

| run | base acc | PR acc | base invalid | PR invalid |
|---|---|---|---|---|
| p1g1 | 0.9606 | 0.9629 | 0 | 0 |
| p1g2 | 0.9621 | 0.9591 | 0 | 0 |
| p1g3 | 0.9636 | 0.9606 | 0 | 0 |
| p1g4 | 0.9659 | 0.9621 | 0 | 0 |
| p2g1 | 0.9651 | 0.9621 | 0 | 1 |
| p2g2 | 0.9682 | 0.9621 | 0 | 0 |
| p2g3 | 0.9613 | 0.9636 | 0 | 0 |
| p2g4 | 0.9621 | 0.9575 | 0 | 0 |
| **mean (sd)** | **0.9636** (0.0026) | **0.9612** (0.0020) | | |

flips base→PR (all 64 run pairs): right→wrong 11.8, wrong→right 8.7, total 20.6 (range 11–26), over 64 pairs
flips base→base (A/A noise floor): right→wrong 7.8, wrong→right 8.7, total 16.5 (range 11–21), over 28 pairs
flips PR→PR (A/A): right→wrong 8.6, wrong→right 8.1, total 16.7 (range 10–21), over 28 pairs
questions always right: base 1244, PR 1242; always wrong: base 35, PR 35; unstable (mixed in either arm): 54
questions right in all base runs but wrong in all PR runs: 0; reverse: 0
