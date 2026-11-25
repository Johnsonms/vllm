| run | c32 | c96 | c128 | c192 | c256 | c384 | TTFT c4 1k (ms) | GSM8K | AOT loads |
|---|---|---|---|---|---|---|---|---|---|
| r1_default | 4.39 | 5.45 | 5.76 | 7.54 | 8.77 | 13.50 | 55.8 | 0.941 | 0 |
| r1_8 | 4.39 | 5.50 | 5.88 | 7.46 | 8.87 | 13.88 | 55.9 | 0.945 | 0 |
| r1_16 | 4.47 | 5.33 | 5.82 | 7.52 | 8.48 | 13.83 | 54.7 | 0.944 | 0 |
| r1_32 | 4.36 | 5.39 | 5.88 | 7.45 | 8.55 | 14.47 | 53.4 | 0.944 | 0 |
| r2_default | 4.37 | 5.43 | 5.87 | 7.57 | 8.83 | 13.70 | 55.8 | - | 0 |
| r2_8 | 4.38 | 5.39 | 5.85 | 7.56 | 9.04 | 13.57 | 55.7 | - | 0 |
| r2_16 | 4.39 | 5.47 | 5.92 | 7.49 | 9.42 | 14.50 | 55.2 | - | 0 |
| r2_32 | 4.45 | 5.28 | 5.88 | 7.51 | 9.09 | 11.40 | 52.9 | - | 0 |

| arm (mean of 2 rounds) | c32 | c96 | c128 | c192 | c256 | c384 | TTFT c4 |
|---|---|---|---|---|---|---|---|
| default | 4.38 | 5.44 | 5.82 | 7.56 | 8.80 | 13.60 | 55.8 |
| 8 | 4.39 | 5.45 | 5.87 | 7.51 | 8.96 | 13.72 | 55.8 |
| 16 | 4.43 | 5.40 | 5.87 | 7.51 | 8.95 | 14.16 | 54.9 |
| 32 | 4.41 | 5.33 | 5.88 | 7.48 | 8.82 | 12.94 | 53.1 |

| Δ TPOT vs default | c32 | c96 | c128 | c192 | c256 | c384 | TTFT c4 |
|---|---|---|---|---|---|---|---|
| 8 MB | +0.2% | +0.2% | +0.9% | -0.6% | +1.8% | +0.9% | +0.0% |
| 16 MB | +1.0% | -0.7% | +1.0% | -0.6% | +1.7% | +4.1% | -1.6% |
| 32 MB | +0.6% | -1.9% | +1.2% | -1.0% | +0.2% | -4.9% | -4.7% |

probe_16: ['Flashinfer max size: 16 MB,Maximal number of tokens used by Flashinfer Allreduce Fusion: 1638'] ['workspace with backend=mnnvl'] AOT=0 tracebacks=0

probe_32: ['Flashinfer max size: 32 MB,Maximal number of tokens used by Flashinfer Allreduce Fusion: 3276'] ['workspace with backend=mnnvl'] AOT=0 tracebacks=0

probe_8: ['Flashinfer max size: 8 MB,Maximal number of tokens used by Flashinfer Allreduce Fusion: 819'] ['workspace with backend=mnnvl'] AOT=0 tracebacks=0

probe_default: ['Flashinfer max size: 4 MB,Maximal number of tokens used by Flashinfer Allreduce Fusion: 409'] ['workspace with backend=mnnvl'] AOT=0 tracebacks=0
