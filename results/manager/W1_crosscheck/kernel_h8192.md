# FlashInfer Fused Collective Operations Benchmark Results

**World Size:** 8  
**Hidden Dimension:** 8192  
**Warmup Iterations:** 5  
**Benchmark Trials:** 50  
**Quantization Modes:** fp8,none  

---

## Configuration: num_tokens=32, dtype=bfloat16, with residual
**Input Size:** 0.50 MB

| Operation                                                   |   Time (ms) | Speedup   |
|:------------------------------------------------------------|------------:|:----------|
| Standard Allreduce  Native Rms Norm                         |       0.03  | 1.02x     |
| Standard Allreduce  Custom Rms Norm                         |       0.03  | 1.02x     |
| Standard Allreduce Rmsnorm Native Compiled                  |       0.03  | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           |       0.013 | 2.38x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           |       0.039 | 0.79x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            |       0.013 | 2.43x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            |       0.008 | 3.73x     |
| Standard Allreduce Native Rms Norm Native Quant Fp8         |       0.031 | 0.98x     |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         |       0.032 | 0.96x     |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         |       0.031 | 0.97x     |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         |       0.032 | 0.96x     |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        |       0.031 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot |       0.013 | 2.47x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot |       0.039 | 0.80x     |

## Configuration: num_tokens=64, dtype=bfloat16, with residual
**Input Size:** 1.00 MB

| Operation                                                   |   Time (ms) | Speedup   |
|:------------------------------------------------------------|------------:|:----------|
| Standard Allreduce  Native Rms Norm                         |       0.032 | 1.00x     |
| Standard Allreduce  Custom Rms Norm                         |       0.032 | 1.00x     |
| Standard Allreduce Rmsnorm Native Compiled                  |       0.032 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           |       0.019 | 1.69x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           |       0.039 | 0.82x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            |       0.019 | 1.70x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            |       0.01  | 3.34x     |
| Standard Allreduce Native Rms Norm Native Quant Fp8         |       0.034 | 0.94x     |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         |       0.034 | 0.93x     |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         |       0.034 | 0.95x     |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         |       0.034 | 0.94x     |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        |       0.034 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot |       0.019 | 1.77x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot |       0.04  | 0.85x     |

## Configuration: num_tokens=96, dtype=bfloat16, with residual
**Input Size:** 1.50 MB

| Operation                                                   |   Time (ms) | Speedup   |
|:------------------------------------------------------------|------------:|:----------|
| Standard Allreduce  Native Rms Norm                         |       0.035 | 1.00x     |
| Standard Allreduce  Custom Rms Norm                         |       0.035 | 1.00x     |
| Standard Allreduce Rmsnorm Native Compiled                  |       0.035 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           |       0.028 | 1.25x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           |       0.04  | 0.88x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            |       0.025 | 1.40x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            |       0.01  | 3.38x     |
| Standard Allreduce Native Rms Norm Native Quant Fp8         |       0.037 | 0.95x     |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         |       0.037 | 0.95x     |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         |       0.037 | 0.95x     |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         |       0.037 | 0.94x     |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        |       0.037 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot |       0.028 | 1.33x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot |       0.04  | 0.92x     |

## Configuration: num_tokens=128, dtype=bfloat16, with residual
**Input Size:** 2.00 MB

| Operation                                                   |   Time (ms) | Speedup   |
|:------------------------------------------------------------|------------:|:----------|
| Standard Allreduce  Native Rms Norm                         |       0.039 | 1.27x     |
| Standard Allreduce  Custom Rms Norm                         |       0.043 | 1.14x     |
| Standard Allreduce Rmsnorm Native Compiled                  |       0.049 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           |       0.034 | 1.43x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           |       0.047 | 1.05x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            |       0.032 | 1.53x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            |       0.012 | 4.25x     |
| Standard Allreduce Native Rms Norm Native Quant Fp8         |       0.041 | 1.20x     |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         |       0.04  | 1.21x     |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         |       0.04  | 1.21x     |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         |       0.04  | 1.22x     |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        |       0.04  | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot |       0.034 | 1.18x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot |       0.046 | 0.87x     |

## Configuration: num_tokens=192, dtype=bfloat16, with residual
**Input Size:** 3.00 MB

| Operation                                                   |   Time (ms) | Speedup   |
|:------------------------------------------------------------|------------:|:----------|
| Standard Allreduce  Native Rms Norm                         |       0.051 | 1.00x     |
| Standard Allreduce  Custom Rms Norm                         |       0.051 | 1.00x     |
| Standard Allreduce Rmsnorm Native Compiled                  |       0.051 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           |       0.049 | 1.03x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           |       0.044 | 1.14x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            |       0.049 | 1.03x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            |       0.014 | 3.71x     |
| Standard Allreduce Native Rms Norm Native Quant Fp8         |       0.054 | 0.95x     |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         |       0.053 | 0.96x     |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         |       0.053 | 0.95x     |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         |       0.053 | 0.95x     |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        |       0.053 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot |       0.05  | 1.06x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot |       0.045 | 1.18x     |

## Configuration: num_tokens=256, dtype=bfloat16, with residual
**Input Size:** 4.00 MB

| Operation                                                   |   Time (ms) | Speedup   |
|:------------------------------------------------------------|------------:|:----------|
| Standard Allreduce  Native Rms Norm                         |       0.058 | 1.00x     |
| Standard Allreduce  Custom Rms Norm                         |       0.058 | 1.00x     |
| Standard Allreduce Rmsnorm Native Compiled                  |       0.058 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           |       0.066 | 0.87x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           |       0.047 | 1.22x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            |       0.063 | 0.91x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            |       0.016 | 3.50x     |
| Standard Allreduce Native Rms Norm Native Quant Fp8         |       0.061 | 0.94x     |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         |       0.06  | 0.96x     |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         |       0.061 | 0.94x     |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         |       0.06  | 0.96x     |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        |       0.061 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot |       0.066 | 0.92x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot |       0.048 | 1.26x     |

## Configuration: num_tokens=384, dtype=bfloat16, with residual
**Input Size:** 6.00 MB

| Operation                                                   |   Time (ms) | Speedup   |
|:------------------------------------------------------------|------------:|:----------|
| Standard Allreduce  Native Rms Norm                         |       0.078 | 1.00x     |
| Standard Allreduce  Custom Rms Norm                         |       0.078 | 1.00x     |
| Standard Allreduce Rmsnorm Native Compiled                  |       0.078 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           |       0.099 | 0.79x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           |       0.057 | 1.38x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            |       0.087 | 0.89x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            |       0.024 | 3.26x     |
| Standard Allreduce Native Rms Norm Native Quant Fp8         |       0.083 | 0.94x     |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         |       0.081 | 0.96x     |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         |       0.082 | 0.95x     |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         |       0.081 | 0.96x     |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        |       0.081 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot |       0.098 | 0.83x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot |       0.057 | 1.42x     |

## Configuration: num_tokens=512, dtype=bfloat16, with residual
**Input Size:** 8.00 MB

| Operation                                                   |   Time (ms) | Speedup   |
|:------------------------------------------------------------|------------:|:----------|
| Standard Allreduce  Native Rms Norm                         |       0.097 | 1.02x     |
| Standard Allreduce  Custom Rms Norm                         |       0.097 | 1.02x     |
| Standard Allreduce Rmsnorm Native Compiled                  |       0.098 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           |       0.128 | 0.77x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           |       0.063 | 1.57x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            |       0.112 | 0.88x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            |       0.03  | 3.23x     |
| Standard Allreduce Native Rms Norm Native Quant Fp8         |       0.102 | 0.96x     |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         |       0.1   | 0.98x     |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         |       0.102 | 0.96x     |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         |       0.1   | 0.98x     |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        |       0.1   | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot |       0.127 | 0.79x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot |       0.063 | 1.60x     |

## Configuration: num_tokens=768, dtype=bfloat16, with residual
**Input Size:** 12.00 MB

| Operation                                                   | Time (ms)   | Speedup   |
|:------------------------------------------------------------|:------------|:----------|
| Standard Allreduce  Native Rms Norm                         | 0.135       | N/A       |
| Standard Allreduce  Custom Rms Norm                         | 0.136       | N/A       |
| Standard Allreduce Rmsnorm Native Compiled                  | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           | 0.191       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           | 0.073       | N/A       |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            | 0.159       | N/A       |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            | 0.044       | N/A       |
| Standard Allreduce Native Rms Norm Native Quant Fp8         | 0.143       | N/A       |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         | 0.139       | N/A       |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         | 0.143       | N/A       |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         | 0.139       | N/A       |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot | 0.192       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot | 0.094       | N/A       |

## Configuration: num_tokens=1024, dtype=bfloat16, with residual
**Input Size:** 16.00 MB

| Operation                                                   | Time (ms)   | Speedup   |
|:------------------------------------------------------------|:------------|:----------|
| Standard Allreduce  Native Rms Norm                         | 0.174       | N/A       |
| Standard Allreduce  Custom Rms Norm                         | 0.175       | N/A       |
| Standard Allreduce Rmsnorm Native Compiled                  | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           | 0.251       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           | 0.092       | N/A       |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            | 0.207       | N/A       |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            | 0.057       | N/A       |
| Standard Allreduce Native Rms Norm Native Quant Fp8         | 0.184       | N/A       |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         | 0.179       | N/A       |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         | 0.184       | N/A       |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         | 0.179       | N/A       |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot | 0.251       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot | 0.110       | N/A       |

## Configuration: num_tokens=1536, dtype=bfloat16, with residual
**Input Size:** 24.00 MB

| Operation                                                   | Time (ms)   | Speedup   |
|:------------------------------------------------------------|:------------|:----------|
| Standard Allreduce  Native Rms Norm                         | 0.265       | N/A       |
| Standard Allreduce  Custom Rms Norm                         | 0.264       | N/A       |
| Standard Allreduce Rmsnorm Native Compiled                  | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           | 0.376       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           | 0.136       | N/A       |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            | FAILED      | FAILED    |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            | 0.084       | N/A       |
| Standard Allreduce Native Rms Norm Native Quant Fp8         | 0.277       | N/A       |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         | 0.269       | N/A       |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         | 0.277       | N/A       |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         | 0.269       | N/A       |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot | 0.375       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot | 0.161       | N/A       |

## Configuration: num_tokens=2048, dtype=bfloat16, with residual
**Input Size:** 32.00 MB

| Operation                                                   | Time (ms)   | Speedup   |
|:------------------------------------------------------------|:------------|:----------|
| Standard Allreduce  Native Rms Norm                         | 0.362       | N/A       |
| Standard Allreduce  Custom Rms Norm                         | 0.360       | N/A       |
| Standard Allreduce Rmsnorm Native Compiled                  | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           | 0.504       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           | 0.158       | N/A       |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            | FAILED      | FAILED    |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            | 0.111       | N/A       |
| Standard Allreduce Native Rms Norm Native Quant Fp8         | 0.378       | N/A       |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         | 0.366       | N/A       |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         | 0.377       | N/A       |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         | 0.368       | N/A       |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot | 0.501       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot | 0.210       | N/A       |

## Configuration: num_tokens=4096, dtype=bfloat16, with residual
**Input Size:** 64.00 MB

| Operation                                                   | Time (ms)   | Speedup   |
|:------------------------------------------------------------|:------------|:----------|
| Standard Allreduce  Native Rms Norm                         | 0.752       | N/A       |
| Standard Allreduce  Custom Rms Norm                         | 0.754       | N/A       |
| Standard Allreduce Rmsnorm Native Compiled                  | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           | 1.004       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           | 0.296       | N/A       |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            | FAILED      | FAILED    |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            | 0.224       | N/A       |
| Standard Allreduce Native Rms Norm Native Quant Fp8         | 0.799       | N/A       |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         | 0.766       | N/A       |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         | 0.799       | N/A       |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         | 0.766       | N/A       |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot | 1.002       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot | 0.382       | N/A       |
