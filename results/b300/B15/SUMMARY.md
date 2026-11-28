# B15 — QuACK NVFP4 kernels vs vLLM's NVFP4 defaults on SM103 (B300), phase 1 (kernels only)

Setup: vLLM main `14e9c30d86`, quack-kernels 0.6.5, FlashInfer 0.7.4, 1x B300. Checkpoint: **nvidia/Qwen3-32B-NVFP4** (ungated, ModelOpt NVFP4 W4A4, group 16, static input scales; FP8 KV). No vLLM edits; bench code local.

## Ranked table (go threshold: >= 1.10x at an M range carrying >= 5% of step time)
| # | op | M range | speedup vs vLLM default | format caveat | share of step time (Qwen3-32B, serve) | verdict |
|---|---|---|---|---|---|---|
| 1 | gate_up GEMM + SwiGLU (QuACK gated GEMM, bf16 postact) + vLLM fp4 quant, vs cute-dsl GEMM + fused `silu_mul_cvt_fp16_to_fp4` | M >= 512 | **1.18-1.24x** (1.03x at 128; 0.83-0.91x at M <= 32) | gate_up weight rows must be **interleaved** once offline (blockscaled QuACK rejects `concat_layout`); bit-compatible NVFP4 otherwise | prefill 4096: ~41% (gate_up GEMM ~32% + silu+quant 9.4%) | **GO for prefill / M >= 512 only** (est. ~6-7% of prefill GPU time); decode no |
| 2 | o_proj plain NVFP4 GEMM | M 32-128 | 1.10-1.27x (0.86-0.95x at M <= 8, ~1.0x at M >= 512) | none (consumes vLLM tensors as is) | c32 decode ~10% | borderline: 2 points, not a range; needs per-M dispatch |
| 3 | qkv_proj plain | M = 128 | 1.30x (1.01-1.07x at 8-32, 0.85x at 1) | none | c32 ~8% | no (single point) |
| 4 | down_proj plain | M = 128 | 1.11x (**0.41-0.64x at M <= 32**, 0.89x at 8192) | none | c32 ~13% | no |
| 5 | gate_up plain | all | 0.84-1.00x | none | ~30% | no |
| 6 | gate_up gated GEMM with **NVFP4-quantized postact**, one kernel (B15b: `swiglu_quant_mod` + alpha, configs pinned) | M >= 512 | **1.27-1.39x** vs cute-dsl GEMM + fused silu_mul_fp4 (1.03x at 128; 0.80x at M <= 32) | interleaved gate_up rows; down_proj consumes the output as is | prefill ~41% | **GO for prefill / M >= 512** — supersedes row 1 (see B15b) |

## 1. What vLLM runs (serve profile, torch profiler, default config: torch.compile + CUDA graphs)
- All 256 linears (qkv/o/gate_up/down x 64 layers): **`FlashInferCuteDslNvFp4LinearKernel`** (FlashInfer cute-dsl `Sm100BlockScaledPersistentDenseGemmKernel`; `Sm100BlockScaledSplitKGemmKernel` for down_proj at small M). lm_head stays BF16 (`nvjet`).
- Activation quant: `fuse_act_quant` is on → SiLU-mul + NVFP4 quant fused (`silu_mul_cvt_fp16_to_fp4`). `fuse_norm_quant` is off, but Inductor fuses fused_add_rms_norm + `scaled_fp4_quant` into Triton kernels for qkv and gate_up inputs. The o_proj input (attention output) uses standalone `cvt_fp16_to_fp4`.
- GPU kernel time shares (`serve_q32_shares.txt`; c1 = 1 prompt 512 in / 32 out, c32 = 32 such, pf4k = 4 x 4096 in / 1 out):

| category | c1 | c32 | prefill 4096 |
|---|---:|---:|---:|
| NVFP4 GEMM (cute-dsl) | 56.5% | 59.4% | 57.3% |
| fp4 act quant (norm+quant Triton, standalone o_proj quant) | 16.7% | 12.0% | 8.8% |
| fused silu_mul + fp4 quant | 4.3% | 7.9% | 9.4% |
| attention | 11.7% | 11.8% | 17.6% |
| norms (unfused part) | 5.1% | 6.1% | 6.3% |
| lm_head / other | 5.7% | 2.7% | 0.6% |

Per-op GEMM split estimated from the microbench times x call counts: gate_up ~29-32%, down ~11-15%, o ~5-10%, qkv ~6-8% of step time.

## 2. Format compatibility
**QuACK consumes vLLM's NVFP4 tensors as is.** vLLM's packed e2m1 weights (uint8, low nibble = even K) view as `torch.float4_e2m1fn_x2`. Its `swizzle_blockscale` / `scaled_fp4_quant(is_sf_swizzled_layout=True)` SF layout (pad to 128 rows x 4 cols, permute (rm, rk, 32, 4, 4)) is byte-identical to QuACK's blocked SF atom, so `sf.view(rm, rk, 32, 4, 4)` works with no copy. The two per-tensor scales go in `alpha` = 1/(g_a * g_w), as in vLLM. Plain GEMM outputs are **bit-identical** to the cute-dsl GEMM (relative difference 0.0 at every shape and M). Only repack needed: the gate_up row interleave for the gated kernel.

## 3. Microbench (cold L2, CUDA graph of 5-20 calls, median of 7 interleaved rounds; vLLM us / QuACK us = speedup; `mb_plain.json`, `mb_gated.json`)
Timing uses CUDA events around graph replays after a 256 MB L2 flush, not CUPTI per kernel. QuACK autotuned per shape (default `tuned=True`).

**Plain NVFP4 GEMM** (same pre-quantized input; activation quant excluded from both):

| op (N x K) | M=1 | M=8 | M=32 | M=128 | M=512 | M=2048 | M=8192 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qkv_proj (10240 x 5120) | 7.0 / 8.2 = **0.85x** | 7.5 / 7.0 = **1.07x** | 7.6 / 7.5 = **1.01x** | 8.8 / 6.8 = **1.30x** | 13.7 / 13.1 = **1.05x** | 45.1 / 44.2 = **1.02x** | 143.8 / 145.0 = **0.99x** |
| o_proj (5120 x 8192) | 7.5 / 8.7 = **0.86x** | 7.5 / 7.9 = **0.95x** | 9.2 / 7.3 = **1.27x** | 8.2 / 7.5 = **1.10x** | 11.2 / 10.9 = **1.02x** | 31.1 / 31.9 = **0.97x** | 111.4 / 112.3 = **0.99x** |
| gate_up_proj (51200 x 5120) | 25.6 / 30.3 = **0.84x** | 25.6 / 28.8 = **0.89x** | 25.5 / 29.2 = **0.87x** | 26.9 / 29.0 = **0.93x** | 56.2 / 61.6 = **0.91x** | 181.9 / 182.3 = **1.00x** | 806.7 / 817.4 = **0.99x** |
| down_proj (5120 x 25600) | 9.6 / 23.4 = **0.41x** | 9.6 / 18.5 = **0.52x** | 11.5 / 17.8 = **0.64x** | 20.3 / 18.3 = **1.11x** | 29.1 / 29.5 = **0.99x** | 87.6 / 88.4 = **0.99x** | 359.6 / 401.8 = **0.89x** |

| M | vLLM GEMM + silu_and_mul + fp4 quant | vLLM GEMM + fused silu_mul_fp4 (serve default) | QuACK gated (bf16 postact) + fp4 quant | vs fused |
|---:|---:|---:|---:|---:|
| 1 | 33.8 | 28.5 | 34.3 | **0.83x** |
| 8 | 34.1 | 28.6 | 31.5 | **0.91x** |
| 32 | 34.2 | 28.9 | 32.2 | **0.90x** |
| 128 | 36.4 | 31.5 | 30.5 | **1.03x** |
| 512 | 87.4 | 73.8 | 62.7 | **1.18x** |
| 2048 | 291.6 | 251.5 | 202.3 | **1.24x** |
| 8192 | 1221.6 | 982.2 | 816.2 | **1.20x** |

Error vs FP32 reference of the dequantized operands: 0.00165 relative for both (identical outputs).

**gate_up + SwiGLU + NVFP4 output** (N = 2 x 25600, K = 5120): see the second table above. Error vs FP32 reference (FP32 SwiGLU of dequantized operands, output dequantized): vLLM 0.0903-0.0905, QuACK fallback 0.0903-0.0904 (this is the fp4 output rounding itself). QuACK vs vLLM codes: 98.8% identical. The rest differ by one code, because QuACK applies alpha + SiLU in fp32 before rounding to bf16, while vLLM rounds the GEMM output to bf16 first.

## Notes / what breaks on SM103
- ~~Gated GEMM with NVFP4-quantized postact does not compile~~ **Wrong, corrected in B15b below: it works with pinned configs; my failure came from autotuned configs.** Original note: in quack-kernels 0.6.5 (my alpha-carrying copy of `library.gated_quant_mod`: `TileStore("postact", gated=True, quant=BlockScaleFactorStore(..., output="postact"))`). All 48 SM100 configs fail in `setup_epi_tensor -> make_smem_layout_epi` (`assert major_mode_size_bits % inter_num_contiguous_bits == 0`), apparently because the 4-bit postact epilogue tile is too narrow for the smem atom. The stock `gated_quant_mod` also has no `alpha`, and NVFP4 needs alpha applied before SiLU. Fallback measured: bf16 postact + vLLM `scaled_fp4_quant`.
- Blockscaled QuACK rejects `concat_layout` (`"blockscaled GEMM does not support concat_layout"`) → gate/up rows must be interleaved.
- QuACK small-M is weak for long K (down_proj K = 25600: 0.41x at M = 1). The cute-dsl kernel uses split-K there; QuACK's autotuner has no split-K for blockscaled.
- Not measured: Qwen3-8B-NVFP4 microbench (only a forward profile, not used), MoE NVFP4, E2E.

## B15b (2026-11-28): gated GEMM + SwiGLU + NVFP4 output in one QuACK kernel
Correction to row 6 / the note above: **it works.** The B15 failure was from autotuning, not the epilogue. Every config the autotuner tried failed in the smem-layout assert. Pinned configs compile and are correct: 12 configs (tile_m 128 c1 / 256 c2 x tile_n 64/128/256 x dynamic on/off; 192 not tried because of the B200-found bug), 252 runs in all, 0 wrong and 0 failed.
- Kernels: stock `quack.epilogue.library.swiglu_quant_mod`, and the same mod plus a pre-activation `alpha`. NVFP4 needs alpha = 1/(g_a*g_w) applied before SiLU, and the stock mod has no alpha. With unit global scales (alpha = 1) the two give the same results and timings.
- Output global scale: `sfd_norm_const` = down_proj's input global scale (vLLM's `input_global_scale_inv` convention). The SF is written straight into vLLM's swizzled layout. Weight: gate_up rows interleaved once offline (the codes are unchanged).
- Table (real global scales, alpha variant; CUPTI via `bench_gpu_time_with_cupti`, CUDA graph, cold L2; vLLM = cute-dsl GEMM + `silu_and_mul_nvfp4_quant`, the serve default; `b15b.jsonl`):

| M | vLLM cute-dsl GEMM + fused silu_mul_fp4, µs | QuACK one kernel, µs (best pinned config) | speedup | fp4 codes = vLLM | SF bytes = vLLM | err vs FP32 (vLLM / QuACK) | down_proj err on QuACK input / vLLM input |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 32.0 | 40.1 (128x128 c1 dyn1) | **0.80x** | 98.8% | 100.0% | 0.0912 / 0.0911 | 0.0903 / 0.0900 |
| 8 | 32.0 | 40.1 (128x128 c1 dyn1) | **0.80x** | 98.9% | 99.8% | 0.0901 / 0.0900 | 0.0908 / 0.0906 |
| 32 | 32.5 | 39.9 (128x128 c1 dyn1) | **0.81x** | 98.9% | 99.3% | 0.0904 / 0.0903 | 0.0905 / 0.0904 |
| 128 | 35.6 | 34.7 (128x128 c1 dyn0) | **1.03x** | 98.8% | 97.1% | 0.0906 / 0.0905 | 0.0902 / 0.0903 |
| 512 | 68.8 | 54.3 (256x256 c2 dyn1) | **1.27x** | 98.9% | 97.1% | 0.0904 / 0.0904 | 0.0904 / 0.0904 |
| 2048 | 218.8 | 157.5 (256x256 c2 dyn1) | **1.39x** | 98.7% | 96.6% | 0.0903 / 0.0904 | 0.0903 / 0.0903 |
| 8192 | 859.9 | 666.6 (256x256 c2 dyn0) | **1.29x** | 98.8% | 97.1% | 0.0905 / 0.0904 | 0.0904 / 0.0905 |

configs tried: 252, wrong or failed: 0

- **Code-level match:** 98.7-98.9% of fp4 codes and 96.6-100% of SF bytes are identical to vLLM's fused path. The rest differ by one step: QuACK applies alpha and SiLU on the fp32 accumulator, while vLLM rounds the GEMM output to bf16 first. Error vs FP32 is the same for both (it is the fp4 output rounding).
- **down_proj consumes QuACK's output as is:** the codes plus SF (viewed as vLLM's 2-D swizzled scale) go straight into the cute-dsl down GEMM, with the same error as on vLLM's own output.
- Verdict: **GO at M >= 512 (prefill), 1.27-1.39x** on ~41% of prefill GPU time (gate_up + silu_quant), so roughly 9-11% of prefill GPU time. Needs an M-dispatch (cute-dsl at M <= 128, where QuACK is 0.80-1.03x) and pinned configs (no autotune; avoid tile_n 192).
