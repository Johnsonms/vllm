# FlashInfer Fused Collective Operations Benchmark Results

**World Size:** 8  
**Hidden Dimension:** 5120  
**Warmup Iterations:** 5  
**Benchmark Trials:** 50  
**Quantization Modes:** fp8,none  

---

## Configuration: num_tokens=32, dtype=bfloat16, with residual
**Input Size:** 0.31 MB

| Operation                                                   |   Time (ms) | Speedup   |
|:------------------------------------------------------------|------------:|:----------|
| Standard Allreduce  Native Rms Norm                         |       0.04  | 0.99x     |
| Standard Allreduce  Custom Rms Norm                         |       0.039 | 1.00x     |
| Standard Allreduce Rmsnorm Native Compiled                  |       0.039 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           |       0.01  | 3.87x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           |       0.039 | 1.02x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            |       0.01  | 3.93x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            |       0.008 | 5.09x     |
| Standard Allreduce Native Rms Norm Native Quant Fp8         |       0.041 | 0.97x     |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         |       0.042 | 0.95x     |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         |       0.041 | 0.97x     |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         |       0.041 | 0.95x     |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        |       0.041 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot |       0.01  | 3.99x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot |       0.039 | 1.04x     |

## Configuration: num_tokens=64, dtype=bfloat16, with residual
**Input Size:** 0.62 MB

| Operation                                                   |   Time (ms) | Speedup   |
|:------------------------------------------------------------|------------:|:----------|
| Standard Allreduce  Native Rms Norm                         |       0.04  | 1.00x     |
| Standard Allreduce  Custom Rms Norm                         |       0.04  | 1.00x     |
| Standard Allreduce Rmsnorm Native Compiled                  |       0.04  | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           |       0.014 | 2.88x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           |       0.039 | 1.04x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            |       0.015 | 2.64x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            |       0.008 | 4.76x     |
| Standard Allreduce Native Rms Norm Native Quant Fp8         |       0.042 | 0.97x     |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         |       0.042 | 0.96x     |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         |       0.042 | 0.97x     |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         |       0.042 | 0.96x     |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        |       0.042 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot |       0.014 | 2.97x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot |       0.039 | 1.06x     |

## Configuration: num_tokens=96, dtype=bfloat16, with residual
**Input Size:** 0.94 MB

| Operation                                                   |   Time (ms) | Speedup   |
|:------------------------------------------------------------|------------:|:----------|
| Standard Allreduce  Native Rms Norm                         |       0.042 | 1.00x     |
| Standard Allreduce  Custom Rms Norm                         |       0.042 | 1.00x     |
| Standard Allreduce Rmsnorm Native Compiled                  |       0.042 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           |       0.018 | 2.35x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           |       0.039 | 1.08x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            |       0.018 | 2.38x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            |       0.009 | 4.89x     |
| Standard Allreduce Native Rms Norm Native Quant Fp8         |       0.044 | 0.96x     |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         |       0.044 | 0.96x     |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         |       0.044 | 0.96x     |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         |       0.044 | 0.96x     |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        |       0.044 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot |       0.018 | 2.47x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot |       0.04  | 1.11x     |

## Configuration: num_tokens=128, dtype=bfloat16, with residual
**Input Size:** 1.25 MB

| Operation                                                   |   Time (ms) | Speedup   |
|:------------------------------------------------------------|------------:|:----------|
| Standard Allreduce  Native Rms Norm                         |       0.044 | 1.00x     |
| Standard Allreduce  Custom Rms Norm                         |       0.044 | 1.00x     |
| Standard Allreduce Rmsnorm Native Compiled                  |       0.044 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           |       0.021 | 2.07x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           |       0.039 | 1.12x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            |       0.021 | 2.06x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            |       0.009 | 4.63x     |
| Standard Allreduce Native Rms Norm Native Quant Fp8         |       0.046 | 0.96x     |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         |       0.046 | 0.96x     |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         |       0.046 | 0.96x     |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         |       0.046 | 0.95x     |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        |       0.046 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot |       0.021 | 2.13x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot |       0.04  | 1.14x     |

## Configuration: num_tokens=192, dtype=bfloat16, with residual
**Input Size:** 1.88 MB

| Operation                                                   |   Time (ms) | Speedup   |
|:------------------------------------------------------------|------------:|:----------|
| Standard Allreduce  Native Rms Norm                         |       0.048 | 1.00x     |
| Standard Allreduce  Custom Rms Norm                         |       0.048 | 1.00x     |
| Standard Allreduce Rmsnorm Native Compiled                  |       0.048 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           |       0.032 | 1.53x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           |       0.041 | 1.18x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            |       0.031 | 1.56x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            |       0.011 | 4.34x     |
| Standard Allreduce Native Rms Norm Native Quant Fp8         |       0.05  | 0.95x     |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         |       0.05  | 0.96x     |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         |       0.05  | 0.96x     |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         |       0.05  | 0.96x     |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        |       0.05  | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot |       0.031 | 1.60x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot |       0.041 | 1.22x     |

## Configuration: num_tokens=256, dtype=bfloat16, with residual
**Input Size:** 2.50 MB

| Operation                                                   |   Time (ms) | Speedup   |
|:------------------------------------------------------------|------------:|:----------|
| Standard Allreduce  Native Rms Norm                         |       0.055 | 3.27x     |
| Standard Allreduce  Custom Rms Norm                         |       0.055 | 3.27x     |
| Standard Allreduce Rmsnorm Native Compiled                  |       0.179 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           |       0.039 | 4.60x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           |       0.042 | 4.23x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            |       0.039 | 4.60x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            |       0.013 | 14.17x    |
| Standard Allreduce Native Rms Norm Native Quant Fp8         |       0.057 | 3.12x     |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         |       0.057 | 3.14x     |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         |       0.057 | 3.13x     |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         |       0.057 | 3.14x     |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        |       0.057 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot |       0.039 | 1.47x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot |       0.043 | 1.33x     |

## Configuration: num_tokens=384, dtype=bfloat16, with residual
**Input Size:** 3.75 MB

| Operation                                                   |   Time (ms) | Speedup   |
|:------------------------------------------------------------|------------:|:----------|
| Standard Allreduce  Native Rms Norm                         |       0.066 | 1.00x     |
| Standard Allreduce  Custom Rms Norm                         |       0.066 | 1.00x     |
| Standard Allreduce Rmsnorm Native Compiled                  |       0.066 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           |       0.059 | 1.11x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           |       0.047 | 1.39x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            |       0.053 | 1.24x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            |       0.016 | 4.22x     |
| Standard Allreduce Native Rms Norm Native Quant Fp8         |       0.069 | 0.95x     |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         |       0.07  | 0.95x     |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         |       0.069 | 0.95x     |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         |       0.068 | 0.96x     |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        |       0.069 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot |       0.057 | 1.21x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot |       0.048 | 1.44x     |

## Configuration: num_tokens=512, dtype=bfloat16, with residual
**Input Size:** 5.00 MB

| Operation                                                   |   Time (ms) | Speedup   |
|:------------------------------------------------------------|------------:|:----------|
| Standard Allreduce  Native Rms Norm                         |       0.077 | 1.00x     |
| Standard Allreduce  Custom Rms Norm                         |       0.077 | 1.00x     |
| Standard Allreduce Rmsnorm Native Compiled                  |       0.077 | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           |       0.076 | 1.02x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           |       0.051 | 1.51x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            |       0.068 | 1.14x     |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            |       0.02  | 3.86x     |
| Standard Allreduce Native Rms Norm Native Quant Fp8         |       0.081 | 0.95x     |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         |       0.08  | 0.97x     |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         |       0.081 | 0.95x     |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         |       0.08  | 0.97x     |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        |       0.08  | baseline  |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot |       0.076 | 1.06x     |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot |       0.052 | 1.56x     |

## Configuration: num_tokens=768, dtype=bfloat16, with residual
**Input Size:** 7.50 MB

| Operation                                                   | Time (ms)   | Speedup   |
|:------------------------------------------------------------|:------------|:----------|
| Standard Allreduce  Native Rms Norm                         | 0.101       | N/A       |
| Standard Allreduce  Custom Rms Norm                         | 0.101       | N/A       |
| Standard Allreduce Rmsnorm Native Compiled                  | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           | 0.113       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           | 0.266       | N/A       |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            | 0.099       | N/A       |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            | 0.028       | N/A       |
| Standard Allreduce Native Rms Norm Native Quant Fp8         | 0.106       | N/A       |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         | 0.104       | N/A       |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         | 0.106       | N/A       |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         | 0.104       | N/A       |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot | 0.114       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot | 0.054       | N/A       |

## Configuration: num_tokens=1024, dtype=bfloat16, with residual
**Input Size:** 10.00 MB

| Operation                                                   | Time (ms)   | Speedup   |
|:------------------------------------------------------------|:------------|:----------|
| Standard Allreduce  Native Rms Norm                         | 0.123       | N/A       |
| Standard Allreduce  Custom Rms Norm                         | 0.123       | N/A       |
| Standard Allreduce Rmsnorm Native Compiled                  | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           | 0.148       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           | 0.061       | N/A       |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            | 0.128       | N/A       |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            | 0.036       | N/A       |
| Standard Allreduce Native Rms Norm Native Quant Fp8         | 0.130       | N/A       |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         | 0.127       | N/A       |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         | 0.130       | N/A       |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         | 0.127       | N/A       |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot | 0.148       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot | 0.062       | N/A       |

## Configuration: num_tokens=1536, dtype=bfloat16, with residual
**Input Size:** 15.00 MB

| Operation                                                   | Time (ms)   | Speedup   |
|:------------------------------------------------------------|:------------|:----------|
| Standard Allreduce  Native Rms Norm                         | 0.171       | N/A       |
| Standard Allreduce  Custom Rms Norm                         | 0.171       | N/A       |
| Standard Allreduce Rmsnorm Native Compiled                  | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           | 0.221       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           | 0.089       | N/A       |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            | 0.187       | N/A       |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            | 0.052       | N/A       |
| Standard Allreduce Native Rms Norm Native Quant Fp8         | 0.182       | N/A       |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         | 0.193       | N/A       |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         | 0.180       | N/A       |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         | 0.175       | N/A       |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot | 0.221       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot | 0.091       | N/A       |

## Configuration: num_tokens=2048, dtype=bfloat16, with residual
**Input Size:** 20.00 MB

| Operation                                                   | Time (ms)   | Speedup   |
|:------------------------------------------------------------|:------------|:----------|
| Standard Allreduce  Native Rms Norm                         | 0.224       | N/A       |
| Standard Allreduce  Custom Rms Norm                         | 0.224       | N/A       |
| Standard Allreduce Rmsnorm Native Compiled                  | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           | 0.293       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           | 0.144       | N/A       |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            | 0.245       | N/A       |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            | 0.068       | N/A       |
| Standard Allreduce Native Rms Norm Native Quant Fp8         | 0.237       | N/A       |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         | 0.229       | N/A       |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         | 0.235       | N/A       |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         | 0.229       | N/A       |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot | 0.293       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot | 0.112       | N/A       |

## Configuration: num_tokens=4096, dtype=bfloat16, with residual
**Input Size:** 40.00 MB

| Operation                                                   | Time (ms)   | Speedup   |
|:------------------------------------------------------------|:------------|:----------|
| Standard Allreduce  Native Rms Norm                         | 0.465       | N/A       |
| Standard Allreduce  Custom Rms Norm                         | 0.467       | N/A       |
| Standard Allreduce Rmsnorm Native Compiled                  | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Oneshot           | 0.585       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Twoshot           | 0.208       | N/A       |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Oneshot            | FAILED      | FAILED    |
| Flashinfer Mnnvl Fused Allreduce Rmsnorm Twoshot            | 0.138       | N/A       |
| Standard Allreduce Native Rms Norm Native Quant Fp8         | 0.488       | N/A       |
| Standard Allreduce Native Rms Norm Custom Quant Fp8         | 0.474       | N/A       |
| Standard Allreduce Custom Rms Norm Native Quant Fp8         | 0.488       | N/A       |
| Standard Allreduce Custom Rms Norm Custom Quant Fp8         | 0.474       | N/A       |
| Standard Allreduce Rmsnorm Fp8 Quant Native Compiled        | FAILED      | FAILED    |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Oneshot | 0.582       | N/A       |
| Flashinfer Trtllm Fused Allreduce Rmsnorm Fp8 Quant Twoshot | 0.207       | N/A       |
