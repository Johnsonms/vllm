# W21: QuACK FP8 kernels vs vLLM's FP8 defaults on B200 (kernel scouting, no vLLM edits)

2026-11-28 15:58Z. vLLM main 361f362033 (clean), quack-kernels 0.6.5, torch 2.14.0+cu132, one B200 per run (TP1), real weights (layer 10).

## Verdict
- **QuACK cannot read these checkpoints as they are.** It has no 128-block fp32-scale FP8 format. Its only FP8 block format is MXFP8 (1x32, UE8M0 scales). All QuACK numbers below use weights **re-quantized to MXFP8** from the dequantized checkpoint.
- **Correction (2026-11-28 16:10Z): the gated GEMM with SwiGLU and FP8-quantized output does exist**, as the epilogue mod `quack.epilogue.library.swiglu_quant_mod`. My first version said it did not, because the `gemm_act` interface rejects a quantized output; I had not looked at the mod library. It is correct with a hand-pinned config and **wrong with every `tile_n=192` config**, which the autotuner picks often (details in "Bug").
- **Best result: gate_up as one kernel (MXFP8 in, SwiGLU, MXFP8 out with scales) vs DeepGEMM + silu-mul + quant kernel: 1.12-1.44x at M >= 8 on Qwen3-8B, 1.15-1.37x at M >= 32 on Qwen3-32B.** See "Addendum".
- **What passes the go threshold:** the gate_up **gated GEMM (MXFP8 in, SwiGLU epilogue, BF16 out)** vs DeepGEMM + silu-mul: **1.10-1.31x at M >= 32 on Qwen3-8B, 1.22-1.29x at M >= 512 on Qwen3-32B**. That is prefill and high-concurrency decode. gate_up + silu is 34.5 % (8B) and 40.3 % (32B) of prefill kernel time.
- **What does not:** decode at c1-c8. At M <= 8 QuACK is **slower** than DeepGEMM (0.83-0.99x) on every layer. down_proj is within 1.16x everywhere.

## Ranked table
Share = % of GPU kernel time per step from the serve profile (c1 / c32 / prefill 4096).

| # | op | M range | speedup vs vLLM | share 8B | share 32B | format caveat | go? |
|---|---|---|---|---|---|---|---|
| 0 | **gate_up gated GEMM with MXFP8 quantized output** (`swiglu_quant_mod`, config pinned) vs DeepGEMM + Inductor silu-mul + packed quant kernel | 8B: 8-8192; 32B: 32-8192 | 8B 1.12 / 1.28 / 1.39 / 1.38 / 1.44 / 1.39 (M 8/32/128/512/2048/8192); 32B 1.15 / 1.25 / 1.37 / 1.34 / 1.28 (32/128/512/2048/8192); M=1: 1.01 / 0.86 | about 25 / 23 / 36 (gate_up + silu + 1 of 4 quants) | about 32 / 31 / 42 | as row 1, plus down_proj must take the 1x32 scales; needs the tile_n=192 bug avoided | **yes, from c8 (8B) / c32 (32B) up, and prefill** |
| 1 | gate_up gated GEMM (SwiGLU epilogue, BF16 out) vs DeepGEMM + Inductor silu-mul | 8B: 32-8192; 32B: 512-8192 | 8B 1.10 / 1.22 / 1.19 / 1.31 / 1.24 (M 32/128/512/2048/8192); 32B 1.25 / 1.29 / 1.22 (512/2048/8192) | 23.2 / 20.5 / 34.5 | 30.9 / 29.2 / 40.3 | MXFP8 re-quantized weights, 1x32 activation scales, interleaved gate/up rows | **yes, large M only** |
| 2 | plain GEMM, qkv and o | 32-512 | 8B 1.31-1.38; 32B 1.08-1.27 | 16.9 / 16.3 / 12.9 | 17.3 / 16.2 / 13.1 | same | 8B yes (two ops together), 32B borderline |
| 3 | plain GEMM, down | 32-512 | 8B 1.09-1.16; 32B 0.97-1.06 | 15.7 / 14.5 / 14.8 | 20.7 / 19.9 / 17.0 | same | no |
| 4 | any GEMM at M <= 8 | 1-8 | 0.83-1.11 (M=1: 0.83-0.92 everywhere) | all decode at c <= 8 | | same | **no** |
| 5 | plain GEMM with MXFP8 quantized output | all | kernel cost = BF16-out cost (+0-2 %), so it would remove the next quant kernel | quant is 8.9 / 8.3 / 7.4 | 5.5 / 5.9 / 6.1 | wrong results with tile_n=192; in a dense model only gate_up feeds a quantized GEMM directly, so row 0 is the use | covered by row 0 |
| 6 | activation quantization (BF16 -> FP8 + scales) | 1-32 | QuACK has no kernel; its compiled-torch quantizer is 0.2-0.6x at M <= 32, parity from M = 128 | same as row 5 | | 1x32 scales in QuACK's blocked layout | no |

Rough size of row 0: 8B decode at c32 (M=32) (1 - 1/1.28) x 23 % = about 5 % of kernel time, prefill 4096 (1 - 1/1.42) x 36 % = about 11 %; 32B c32 about 4 %, prefill about 10 %. Not measured end to end.
Rough size of row 1 at prefill 4096: 8B (1 - 1/1.27) x 34.5 % = about 7 % of prefill kernel time; 32B (1 - 1/1.25) x 40.3 % = about 8 %. Not measured end to end.

## Step 1: what vLLM runs (profile, `profile.md`)
Both models: all 4 linears per layer use **`DeepGemmFp8BlockScaledMMKernel`** (kernel `deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl`), with UE8M0 scales (vLLM already rounds the checkpoint's 128x128 fp32 weight scales to powers of two at load on SM100). Activation quant is the packed 1x128 kernel `per_token_group_quant_8bit_packed_register_kernel`, **one separate launch before every GEMM** (4 per layer; not fused with silu-mul or RMSNorm). silu-mul is an Inductor (Triton) kernel. No CUTLASS, no FlashInfer GEMM. LM head = BF16 CuTe skinny GEMM (c1) / cuBLAS.

| % of kernel time | 8B c1 | 8B c32 | 8B prefill 4096 | 32B c1 | 32B c32 | 32B prefill 4096 |
|---|---|---|---|---|---|---|
| DeepGEMM total | 54.1 | 49.4 | 56.4 | 67.6 | 63.7 | 64.8 |
| - gate_up | 21.4 | 18.6 | 28.6 | 29.7 | 27.6 | 34.7 |
| - down | 15.7 | 14.5 | 14.8 | 20.7 | 19.9 | 17.0 |
| - qkv | 8.8 | 8.8 | 7.7 | 8.4 | 7.9 | 7.3 |
| - o | 8.1 | 7.5 | 5.2 | 8.9 | 8.3 | 5.8 |
| FP8 group quant | 8.9 | 8.3 | 7.4 | 5.5 | 5.9 | 6.1 |
| silu-mul | 1.8 | 1.9 | 5.9 | 1.2 | 1.6 | 5.6 |
| RMSNorm (Inductor) | 12.9 | 10.8 | 10.5 | 13.5 | 11.1 | 8.1 |
| attention | 10.3 | 17.9 | 16.8 | 6.5 | 11.8 | 14.2 |
| kernel time per step (us) | 3322 | 3814 | 11520 | 8747 | 9691 | 49062 |

Unprofiled: 8B TPOT 3.46 ms (c1), 3.87 ms (c32), TTFT 52.3 ms at 4096 tokens. 32B TPOT 8.99 / 9.73 ms, TTFT 163.4 ms.
The prefill trace holds 32 engine iterations of which about 11 are prefill steps, so its "per step" numbers are low by about 3x; the percentages are not affected.

## Step 2: formats (`probe.py`, `probe2.py`)
| call | result |
|---|---|
| checkpoint scales as is (128x128 fp32 weight, 1x128 activation) | **not supported**. `blockscaled/operand.py` lists "fp8_e4m3_1x128" and 2D 128x128 grids as formats that are not registered yet |
| plain e4m3 x e4m3 without scales | compiles for some configs only, and is not useful for these checkpoints |
| MXFP8 x MXFP8 -> BF16 | works |
| BF16 activations x MXFP8 weights (weight-only) | rejected: both operands must carry scales |
| MXFP8 x MXFP8 -> MXFP8 quantized output (`gemm`, out_dtype) | works with the default config; **wrong with some tuned configs** |
| gated (SwiGLU) MXFP8 x MXFP8 -> BF16 | works, but only with interleaved gate/up rows (`concat_layout` is rejected for blockscaled), so the merged weight must be re-laid out at load |
| gated -> MXFP8 quantized output via `gemm_act(postact_dtype=...)` | NotImplementedError ("only supported by quack.gemm for now") |
| gated -> MXFP8 quantized output via `swiglu_quant_mod(...)` (`probe3.py`) | **works** with the default or a pinned config (MXFP8 or BF16 inputs); needs 6-D scale tensors; wrong with tile_n=192 |

Since vLLM already uses power-of-two scales here, moving to MXFP8 changes the scale granularity (1x32 instead of 1x128 / 128x128), not the kind of rounding. With random activations the error vs FP32 is the same for both arms (0.0376-0.0383 plain, 0.053-0.054 after SwiGLU).

## Step 3: microbench (`bench.py`, tables in `bench_tables.md`, raw `bench_*.jsonl`)
CUDA graph, cold L2, CUPTI, median. vLLM arm = the same functions the DeepGEMM kernel class calls. Speedup = vLLM / QuACK.

Plain GEMM (prequantized inputs -> BF16):

| M | 8B qkv | 8B o | 8B gate_up | 8B down | 32B qkv | 32B o | 32B gate_up | 32B down |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.89 | 0.86 | 0.89 | 0.91 | 0.92 | 0.83 | 0.88 | 0.89 |
| 8 | 1.11 | 1.07 | 0.97 | 0.96 | 0.99 | 1.01 | 0.89 | 0.93 |
| 32 | 1.32 | 1.31 | 1.07 | 1.09 | 1.12 | 1.13 | 0.98 | 0.97 |
| 128 | 1.38 | 1.32 | 1.16 | 1.16 | 1.25 | 1.27 | 1.02 | 1.06 |
| 512 | 1.32 | 1.34 | 1.06 | 1.12 | 1.08 | 1.12 | 1.06 | 0.97 |
| 2048 | 1.06 | 1.12 | 1.00 | 1.01 | 0.98 | 1.05 | 1.00 | 0.99 |
| 8192 | 1.06 | 0.94 | 1.08 | 1.00 | 1.10 | 1.03 | 1.09 | 1.04 |

gate_up chain, us (vLLM = DeepGEMM + Inductor silu-mul; QuACK = one gated GEMM):

| M | 8B vLLM | 8B QuACK | 8B speedup | 32B vLLM | 32B QuACK | 32B speedup |
|---|---|---|---|---|---|---|
| 1 | 29.2 | 32.0 | 0.91 | 58.9 | 64.4 | 0.91 |
| 8 | 30.4 | 30.9 | 0.98 | 59.0 | 66.9 | 0.88 |
| 32 | 30.0 | 27.2 | 1.10 | 61.2 | 60.1 | 1.02 |
| 128 | 33.5 | 27.6 | 1.22 | 65.2 | 60.5 | 1.08 |
| 512 | 52.2 | 43.7 | 1.19 | 115.4 | 92.6 | 1.25 |
| 2048 | 162.4 | 123.8 | 1.31 | 415.0 | 321.3 | 1.29 |
| 8192 | 656.2 | 528.3 | 1.24 | 1623.8 | 1328.5 | 1.22 |

With the quant of h added on both sides (vLLM's kernel for both): 8B 0.92 / 0.98 / 1.10 / 1.18 / 1.16 / 1.25 / 1.43, 32B 0.90 / 0.90 / 0.99 / 1.07 / 1.22 / 1.22 / 1.20.

## Addendum: gate_up with quantized output, measured (`bench2.py`, `bench2_*.jsonl`)
vLLM chain as in serving: DeepGEMM + Inductor silu-mul + packed 1x128 quant of h (3 kernels). QuACK: `swiglu_quant_mod`, one kernel that writes h as MXFP8 values plus 1x32 scales. For each M I timed 16 pinned configs (tile_m 128 / 256 with 2-CTA, tile_n 64 / 128 / 192 / 256, static / dynamic scheduler) and took the fastest **correct** one.

| M | 8B vLLM us | 8B QuACK us | 8B speedup | 8B config | 32B vLLM us | 32B QuACK us | 32B speedup | 32B config |
|---|---|---|---|---|---|---|---|---|
| 1 | 30.3 | 30.1 | 1.01 | 128x256 | 59.3 | 69.3 | 0.86 | 128x256 |
| 8 | 34.8 | 30.9 | 1.12 | 128x256 | 63.6 | 64.3 | 0.99 | 256x256 2cta dyn |
| 32 | 35.5 | 27.7 | 1.28 | 128x256 | 65.5 | 57.0 | 1.15 | 128x128 dyn |
| 128 | 38.9 | 27.9 | 1.39 | 128x256 dyn | 71.8 | 57.2 | 1.25 | 128x128 |
| 512 | 60.0 | 43.5 | 1.38 | 256x256 2cta dyn | 125.5 | 91.4 | 1.37 | 256x256 2cta dyn |
| 2048 | 180.7 | 125.9 | 1.44 | 256x256 2cta dyn | 425.3 | 316.5 | 1.34 | 256x256 2cta dyn |
| 8192 | 725.4 | 522.8 | 1.39 | 256x256 2cta | 1716.2 | 1341.6 | 1.28 | 256x256 2cta |

Error of h vs FP32: QuACK 0.059-0.060 **after** its output quantization; vLLM 0.053-0.054 **before** its quant kernel (I did not measure vLLM's h after quantization, so the two numbers are not the same stage).

## Bug in QuACK 0.6.5 (for the owner; I opened nothing)
**Any GEMM with an MXFP8 quantized output is wrong when `tile_n=192`.** In the addendum sweep (`swiglu_quant_mod`, 14 points, 16 configs each) all 4 `tile_n=192` configs are wrong at every point and all 12 others are correct at every point. The autotuner often picks 192, so:
`quack.gemm(A, B, out_dtype=MXFP8_E4M3)` with `tuned=True` returns wrong or non-finite values for some shapes: relative error 0.22-0.48, inf or nan, and different from run to run. 19 of 56 bench points are affected. `tuned=False` is correct on all 14 cases tested (0.046 MXFP8 in, 0.027 BF16 in). The failing case I traced picked `GemmConfig(tile_m=256, tile_n=192, cluster_m=2)`; a passing one picked `tile_n=128`. Likely cause (not verified in the source): 192 is not a multiple of the 128-column scale-factor block. Repro: `qout_check2.py`, `qout_check3.py` (M=512, K=4096, N=6144; set QUACK_PRINT_AUTOTUNING=1 to see the config). BF16 inputs with quantized output fail the same way (nan), so it is the output-scale epilogue, not the MXFP8 inputs.

## Limits of this measurement
- Cold-L2 microbench numbers are higher than in serving (gate_up at M=1: 27.4 us here, 19.7 us in the c1 trace). Both arms are measured the same way.
- Random activations, one layer per model, TP1. No end-to-end run: that needs vLLM edits (weight re-quantization at load, an MXFP8 1x32 activation quant kernel, the gated call), which phase 1 excludes.
- QuACK timings are with its autotuner on (best case for QuACK).
- The first version of the vLLM silu baseline used the `_C.silu_and_mul` custom op; serving uses an Inductor kernel that is 1.2-3.4x faster. The tables use the Inductor timing (`silu_supp.py`).
