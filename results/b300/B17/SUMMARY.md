# B17 — end-to-end NVFP4 gate_up prototype (QuACK one-kernel gate_up + SwiGLU + NVFP4 out), Qwen3-32B-NVFP4, B300

**Result: prefill TTFT −9.6 / −9.8 / −11.4 % (c1/c4/c16, 12/12 each); decode TPOT flat at c8/c32 (−0.2/−0.3 %) and −5.6 % at c128 (12/12). GSM8K unchanged (0.9399 → 0.9416, 4 runs per arm). Cost: +8.85 GiB for a second, interleaved gate_up copy.**

## Prototype (local code, env `VLLM_B17=1`, base `14e9c30d86`, quack-kernels 0.6.5)
- `Qwen3MLP.forward` on NVFP4 (both linears on `FlashInferCuteDslNvFp4LinearKernel`) does four steps:
  1. `gate_up_proj.quant_method.kernel.prepare_input(x)` (the same NVFP4 input quant, so the Inductor norm+quant fusion is kept).
  2. One custom op that writes the NVFP4 h (codes + swizzled SF).
  3. Wrap h as a `QuantizedActivation`.
  4. `down_proj(h)`.
- In the op:
  - **M ≥ 128:** QuACK gated GEMM, an alpha-carrying copy of `library.gated_quant_mod("swiglu")`. The epilogue applies `acc * alpha` (alpha = gate_up `layer.alpha` = 1/(g_x·g_w)), then SwiGLU, then NVFP4 with `sfd_norm_const` = **down_proj `input_global_scale_inv`**. Configs are pinned: 128x128 cluster 1 for M < 512, 256x256 cluster 2 dynamic for M ≥ 512; tile_n 192 is never used.
  - **M < 128:** vLLM's own kernels: cute-dsl GEMM + `silu_and_mul_nvfp4_quant`.
- **SF layout:** QuACK writes its blocked `(1, rm, rk, 32, 4, 4)` scales straight into the `(Mpad128, I/16)` swizzled buffer from `create_fp4_output_tensors`. That buffer is byte-identical to vLLM's 128x4 swizzle (helpers checked against `swizzle_blockscale`). down_proj's `prepare_input` accepts it as is; no repack or copy.
- **Weights:** gate/up rows interleaved once (codes unchanged, SF unswizzled → interleaved → reswizzled), cached on first call (during the profile run, before graph capture). The original copy stays for the M < 128 branch.
- Checks:
  - Op level: codes 98.9 % and SF 98.6 % identical to vLLM's branch at M = 128/300/512/2048/4096. The rest are one-step roundings: QuACK applies alpha and SiLU in fp32, while vLLM rounds the GEMM output to bf16 first.
  - The kernel is bit-deterministic over 20 reruns.
  - The path fires 64x per prefill in eager mode, and works under torch.compile + FULL_AND_PIECEWISE CUDA graphs (prefill graphs up to 8192 tokens include it).
- **Threshold, warm-L2 re-check** (manager note 16:56Z): 4-layer CUDA graph, distinct weights, no flush (`b17_warm.jsonl`): M=32 0.88x, 64 1.08x, **128 1.15x**, 256 1.10x, 512 1.33x, 1024 1.25x, 2048 1.44x, 8192 1.22x. So 128 is a valid cut (64 would also be positive).

## Paired E2E (2 passes x 6 rounds, GPU-swapped; both servers up at once; "Directly load AOT" = 0)
| | c | pass 1 | pass 2 | mean | B17 faster |
|---|---:|---|---|---:|---:|
| TTFT 4096 in / 1 out | 1 | 93.70 → 85.24 ms (−9.03 %) | 96.08 → 86.24 (−10.24 %) | **−9.63 %** | 12/12 |
| | 4 | 317.2 → 281.1 (−11.37 %) | 312.4 → 286.6 (−8.26 %) | **−9.82 %** | 12/12 |
| | 16 | 1306.6 → 1145.4 (−12.34 %) | 1294.3 → 1158.8 (−10.47 %) | **−11.40 %** | 12/12 |
| TPOT 512/512 | 8 | 6.058 → 6.037 (−0.35 %) | 6.059 → 6.051 (−0.13 %) | −0.24 % | 9/12 |
| | 32 | 6.851 → 6.813 (−0.56 %) | 6.815 → 6.808 (−0.10 %) | −0.33 % | 7/12 |
| | 128 | 10.067 → 9.554 (−5.10 %) | 10.171 → 9.544 (−6.16 %) | **−5.63 %** | 12/12 |
c8/c32 decode stay below M = 128 (vLLM kernels), so flat is expected. Their small gain comes from prefill chunks of the mixed batches.

## GSM8K (1319 q, greedy, chat, thinking off; 4 runs per arm, both servers up at once, GPUs swapped after 2)
- Base: 0.9401 / 0.9401 / 0.9393 / 0.9401, mean 0.9399 (sd 0.0004). B17: 0.9454 / 0.9416 / 0.9416 / 0.9378, mean 0.9416 (sd 0.0031).
- Per-question flips: within base 0.5, **within B17 35**, between arms 46.6. 5 questions are right in all base runs and wrong in all B17 runs; 14 the other way.
- B17 is less run-to-run stable, but not because of a nondeterministic kernel (bit-identical reruns). The numerics now depend on batch composition: a token's MLP takes the QuACK or the vLLM branch depending on whether its step's M is ≥ 128, and the two branches differ in ~1 % of fp4 codes. Base uses one kernel path at every M.

## Memory
The interleaved gate_up copy is 64 x (51200 x 2560 B codes + SF) = **8.85 GiB** (KV cache 204.84 → 195.99 GiB at gpu_memory_utilization 0.85, same 19.52 GiB weights load). Dropping the vLLM copy would need cute-dsl on the interleaved layout plus a de-interleaving silu_mul_quant, or QuACK for every M (0.80-0.88x at M ≤ 32).

## What a real landing needs
**From QuACK:**
1. An `alpha` (pre-activation scale) on `gated_quant_mod`. NVFP4 needs it; the stock mod has none.
2. Autotuning that skips configs that cannot compile and the wrong tile_n=192 quantized output. Until then: pinned per-M configs.
3. `concat_layout` for blockscaled operands, or a documented interleave helper, so the [gate; up] checkpoint layout can be used without a second copy.

**From vLLM:**
1. A linear-kernel hook for a "gated producer" in the NVFP4 path. Today it is model-local in `Qwen3MLP`; the same shape as #69778 / #70028 (producer writes a `QuantizedActivation` that the consumer reads), keyed on `kNvfp4Dynamic`.
2. Interleave at `process_weights_after_loading` (one copy if the small-M path can also use it).
3. An M dispatch (≥ 128) that stays CUDA-graph safe.
4. A batch-invariance note or flag, since numerics switch with M.
5. A test that compares codes and SF with `silu_and_mul_nvfp4_quant`, as the op check above does.
