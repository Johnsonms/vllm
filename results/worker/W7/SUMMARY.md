# W7 — single-GPU kernel microbenchmarks (B200)

Worker node, one idle B200 per run (GPU 4, 5 or 6; checked idle before each run), vLLM 8399da3780, driver 580.159.04.
Timing: `flashinfer.testing.bench_gpu_time_with_cupti`, CUDA graph, cold L2, tensors passed through `input_args`;
medians. No vLLM file was edited for W7a or W7c. Benchmark scripts are kept on the node, not in this branch.

## W7a — GDN packed-decode kernel: launch config on B200

**Answer: yes, a modest one. `BV = 4` beats the current `BV = 8` at every measured shape where the launcher picks 8,
by 5-21 % (about 10 % from B·HV ≥ 768). `num_warps` and `num_stages` should stay as they are. A B200 branch is a
one-line change, but the end-to-end gain is about 1 % at 256 requests.**

### Method

- Real launcher `fused_recurrent_gated_delta_rule_packed_decode`, called as the model calls it (`use_qk_l2norm_in_kernel`
  on, scale = 128^-0.5). Only `BV`, `num_warps`, `num_stages` and the matching grid dimension are overridden, by wrapping
  the Triton kernel object at launch. The launcher file is not edited.
- Shapes: K = V = 128, fp32 state, bf16 q/k/v/a/b and output, H = HV / 3, as in Qwen3.8-Flash-Next.
  State pool of max(B, 512) + 1 slots; the B requests use random distinct slots.
- Sweep: B in {1 … 512} x HV in {6, 12, 24} x BV in {4, 8, 16, 32} x warps in {1, 2, 4} x stages in {2, 3}: 30 shapes,
  24 configs each, no launch errors.
- Bytes for the TB/s column: state read + write (B·HV·128·128·4·2) plus q/k/v and output.
- Noise: the current config timed twice per shape differs by at most 1.1 %.

### Results

| HV (TP) | B | current (BV, warps, stages) | current us | current TB/s | best config | best us | speedup | BV=4, w=1, s=3 us | speedup | second GPU, contiguous slots |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 (TP8) | 1 | (4, 1, 3) | 2.88 | 0.27 | (4, 1, 3) | 2.88 | 1.00x | 2.88 | 1.00x | same config |
| 6 (TP8) | 2 | (4, 1, 3) | 2.91 | 0.54 | (4, 1, 2) | 2.91 | 1.00x | 2.91 | 1.00x | same config |
| 6 (TP8) | 4 | (4, 1, 3) | 3.36 | 0.94 | (4, 1, 2) | 3.36 | 1.00x | 3.36 | 1.00x | same config |
| 6 (TP8) | 8 | (4, 1, 3) | 3.94 | 1.61 | (4, 1, 2) | 3.90 | 1.01x | 3.94 | 1.00x | same config |
| 6 (TP8) | 16 | (8, 1, 3) | 6.34 | 2.00 | (4, 1, 3) | 5.22 | 1.21x | 5.22 | 1.21x | 1.15x |
| 6 (TP8) | 32 | (8, 1, 3) | 8.00 | 3.16 | (4, 1, 2) | 7.42 | 1.08x | 7.42 | 1.08x | 1.06x |
| 6 (TP8) | 64 | (8, 1, 3) | 13.41 | 3.77 | (4, 1, 2) | 11.46 | 1.17x | 11.46 | 1.17x | 1.17x |
| 6 (TP8) | 128 | (8, 1, 3) | 21.57 | 4.69 | (4, 1, 2) | 19.81 | 1.09x | 19.81 | 1.09x | 1.11x |
| 6 (TP8) | 256 | (8, 1, 3) | 39.07 | 5.18 | (4, 1, 2) | 35.58 | 1.10x | 35.62 | 1.10x | 1.10x |
| 6 (TP8) | 512 | (8, 1, 3) | 73.57 | 5.50 | (4, 1, 2) | 68.19 | 1.08x | 68.22 | 1.08x | 1.09x |
| 12 (TP4) | 1 | (4, 1, 3) | 2.98 | 0.53 | (4, 1, 2) | 2.98 | 1.00x | 2.98 | 1.00x | same config |
| 12 (TP4) | 2 | (4, 1, 3) | 3.17 | 1.00 | (4, 1, 2) | 3.14 | 1.01x | 3.16 | 1.00x | same config |
| 12 (TP4) | 4 | (4, 1, 3) | 4.13 | 1.53 | (4, 1, 2) | 4.16 | 0.99x | 4.16 | 0.99x | same config |
| 12 (TP4) | 8 | (8, 1, 3) | 6.40 | 1.98 | (4, 1, 2) | 5.31 | 1.20x | 5.31 | 1.20x | 1.17x |
| 12 (TP4) | 16 | (8, 1, 3) | 7.97 | 3.17 | (4, 1, 2) | 7.58 | 1.05x | 7.58 | 1.05x | 1.09x |
| 12 (TP4) | 32 | (8, 1, 3) | 13.38 | 3.78 | (4, 1, 2) | 11.52 | 1.16x | 11.52 | 1.16x | 1.17x |
| 12 (TP4) | 64 | (8, 1, 3) | 21.54 | 4.70 | (4, 1, 2) | 19.68 | 1.09x | 19.68 | 1.09x | 1.10x |
| 12 (TP4) | 128 | (8, 1, 3) | 38.88 | 5.21 | (4, 1, 2) | 35.07 | 1.11x | 35.07 | 1.11x | 1.10x |
| 12 (TP4) | 256 | (8, 1, 3) | 73.70 | 5.49 | (4, 1, 2) | 66.88 | 1.10x | 66.88 | 1.10x | 1.10x |
| 12 (TP4) | 512 | (8, 1, 3) | 141.09 | 5.74 | (4, 1, 2) | 129.66 | 1.09x | 129.70 | 1.09x | 1.10x |
| 24 (TP2) | 1 | (4, 1, 3) | 3.30 | 0.96 | (4, 1, 3) | 3.26 | 1.01x | 3.26 | 1.01x | same config |
| 24 (TP2) | 2 | (4, 1, 3) | 3.90 | 1.62 | (4, 1, 2) | 3.90 | 1.00x | 3.94 | 0.99x | same config |
| 24 (TP2) | 4 | (8, 1, 3) | 6.27 | 2.02 | (4, 2, 2) | 5.34 | 1.17x | 5.41 | 1.16x | 1.13x |
| 24 (TP2) | 8 | (8, 1, 3) | 8.05 | 3.14 | (4, 1, 2) | 7.55 | 1.07x | 7.58 | 1.06x | 1.08x |
| 24 (TP2) | 16 | (8, 1, 3) | 13.38 | 3.78 | (4, 1, 2) | 11.49 | 1.16x | 11.52 | 1.16x | 1.15x |
| 24 (TP2) | 32 | (8, 1, 3) | 21.54 | 4.70 | (4, 1, 3) | 19.17 | 1.12x | 19.17 | 1.12x | 1.11x |
| 24 (TP2) | 64 | (8, 1, 3) | 38.81 | 5.21 | (4, 1, 2) | 34.46 | 1.13x | 34.46 | 1.13x | 1.10x |
| 24 (TP2) | 128 | (8, 1, 3) | 73.31 | 5.52 | (4, 1, 2) | 65.95 | 1.11x | 65.95 | 1.11x | 1.11x |
| 24 (TP2) | 256 | (8, 1, 3) | 140.96 | 5.74 | (4, 1, 2) | 127.30 | 1.11x | 127.36 | 1.11x | 1.11x |
| 24 (TP2) | 512 | (8, 1, 3) | 276.74 | 5.85 | (4, 1, 2) | 252.45 | 1.10x | 252.54 | 1.10x | 1.10x |

- **BV = 4, warps = 1 is the best or within 1 % of the best at all 30 shapes.** Against the current choice its speedup
  ranges from 0.99x to 1.21x, geometric mean 1.08x over all shapes (the shapes that already use BV = 4 count as 1.00x).
- **stages 2 vs 3 makes no difference** (within 0.1 us everywhere).
- **More warps or bigger tiles lose:** (BV 4, warps 2, stages 3) has a geometric mean of 0.92x and is as low as 0.83x;
  BV 16 and BV 32 are 10-25 % slower than the current config at large B and about 2x slower at B = 1.
- **Confirmed on a second GPU**, with both contiguous and scattered state slots: same ranking, speedups within about
  0.03 of the table (last column).
- **Correctness:** for BV = 4 vs BV = 8 with warps = 1, output and final state are bit-identical (max abs diff 0) at
  every shape. Across all 24 configs the largest difference is 6.1e-5 (configs with more warps).
- **Bandwidth:** the current config reaches 5.2-5.9 TB/s from B·HV ≥ 1,536; BV = 4 lifts that to about 5.7-6.4 TB/s.
  So the kernel is already near memory bandwidth and there is no larger gain left in launch parameters.

### What it is worth

- The current rule is `BV = 4 if B*HV <= 64 else 8` (fp32 state, SM100 family). On B200 the simplest better rule is
  **`BV = 4` for an fp32 state at every size**.
- Kernel gain at common TP4 decode sizes (HV = 12): B = 8 1.21x, 16 1.05x, 32 1.16x, 64 1.09x, 128 1.11x, 256 1.10x.
- Estimated step gain at TEP4: the kernel is 12.4 % of busy time at 256 requests and 4.3 % at 64 (W5), so about
  **1.1 % at 256 requests and 0.4 % at 64**. Not measured end to end.
- Whether this also holds on GB300 (SM103) is unknown: the current values were tuned there, and I have no GB300.
  A branch keyed on SM100 exactly (B200) would leave GB300 untouched.

### Duplicate search

- **#61355 (merged)** "Tune Triton GDN decode tiles on SM100-family GPUs" introduced the current rule. Its tables cover
  B = 1-64 on GB300 only; B > 64 was not measured there.
- **#66865 (open)** "Qwen3.8-Flash-Next specialized decode on GB300": opt-in and GB300-only; it replaces the stock
  decode path there, so it does not touch this launcher's default on B200.
- **#64846 (open)** "Launch the Qwen GDN decode and Triton fused-MoE kernels with PDL": same kernel, different change
  (launch attribute); would need a rebase against a BV change but does not overlap in content.
- **#68576, #61823, #63864, #60400**: the same kind of tile tuning for SM80, SM89, SM90 and ROCm. None covers B200.
- I found no open PR that retunes this kernel for B200 or for B > 64 on SM100.

## W7c — MoE input FP8 quantization kernel (`per_token_group_quant_8bit_register_kernel`)

**Answer: no ≥1.5x by selection or Python-side launch config. The kernel vLLM picks is already the fastest of the
variants that produce the fp32 scales the TRT-LLM MoE needs. But an existing sibling kernel does the same per-element
work 1.5-1.8x faster at large N, which shows the gap is in how the fp32-scale kernel is launched or indexed, not in
memory bandwidth. Closing it needs a small C++ change and a rebuild, which I did not do.**

### How the kernel is picked

- `fused_moe/utils.py::_fp8_quantize` -> `fp8_utils.py::per_token_group_quant_fp8` ->
  `torch.ops._C.per_token_group_fp8_quant` whenever the input is contiguous on CUDA. The Triton kernel in the same
  Python function is only a fallback for non-contiguous input.
- Inside the C++ op (`csrc/quantization/w8a8/fp8/per_token_group_quant.cu`), `CanUseRegisterPath` chooses the
  **register kernel** when group size is 128, the input is 16-bit, and both buffers are contiguous and 16-byte aligned.
  Otherwise it uses the older **shared-memory kernel**. Launch: 256 threads per block, 8 threads per group, 1-D grid.
  Nothing here is reachable from Python or from config.
- A third kernel, the **packed register kernel** (`per_token_group_fp8_quant_packed`), exists for DeepGEMM. It writes
  UE8M0 scales packed four per int32 in a TMA-aligned layout. The TRT-LLM MoE takes fp32 scales (`a1q_scale` passed to
  `flashinfer.fused_moe.trtllm_fp8_block_scale_moe`), so it cannot use this output, and power-of-two scales would also
  change the numerics.

### Measurements (N x 2,560 bf16 -> fp8, group 128; microseconds, 3 runs each agreeing to 0.1 %)

| N | A: C++ register, row-major scales (what runs today) | A': register, column-major scales | B: C++ shared-memory kernel | C: Triton fallback (warps 1 / 2 / 4) | D: packed register kernel (other scale layout) | D vs A |
|---|---|---|---|---|---|---|
| 1 | 2.82 | 2.82 | 2.69 | 1.95 / 1.95 / 1.95 | 2.18 | 1.29x |
| 64 | 3.07 | 3.26 | 3.07 | 2.40 / 2.50 / 2.59 | 2.43 | 1.26x |
| 256 | 3.55 | 3.71 | 3.81 | 4.45 / 4.61 / 5.06 | 2.69 | 1.32x |
| 2,048 | 8.77 | 9.09 | 11.65 | 22.9 / 23.1 / 25.6 | 5.63 | 1.56x |
| 8,192 | 25.6 | 27.2 | 38.3 | 86.1 / 86.3 / 95.1 | 14.6 | **1.76x** |

- B was reached by offsetting the output buffer by 8 bytes, which makes `CanUseRegisterPath` false; the kernel name was
  checked with the profiler. B's output and scales are bit-identical to A's.
- The Triton fallback is 1.3-1.4x **faster** than the C++ kernels at N = 1 and 64 (about 0.7-0.9 us per call), and
  3.4x slower at 8,192. Its scales are not bit-identical to the C++ kernels', and its quantized bytes differ from N = 64 up.
- At N = 8,192, A moves 63.6 MB in 25.6 us, about 2.5 TB/s. A plain bf16 copy of the same input takes 15.8 us (about
  5.3 TB/s), and D takes 14.6 us. So A is not limited by memory bandwidth.

### Where the 1.76x comes from (partly tested)

- **Not the scale math:** A with `use_ue8m0=True` takes 25.6 us at 8,192 and 8.83 us at 2,048, the same as with fp32
  scales. Odd row counts make no difference either (8,190 rows: 25.4 us).
- **What differs between A and D:** D launches 128-thread blocks (16 groups) on a 2-D grid with 32-bit index
  arithmetic and writes a quarter as many scale words; A launches 256-thread blocks on a 1-D grid with 64-bit group
  indices. Which of these matters was **not tested**, because every one of them is a C++ change.

### Proposal (not implemented)

Give the fp32-scale register kernel the packed kernel's launch geometry and 32-bit indexing, keeping its fp32 scale
output. If it then matches D, the MoE input quant drops from about 1.2 % to about 0.7 % of an 8k prefill step, and by
about a quarter at decode sizes. First test is a one-constant change (`REG_BLOCK_THREADS` 256 -> 128) and a rebuild.
At N ≤ 64 the Triton fallback is faster still, but it is not bit-compatible, so I would not route to it.

### Duplicate search

- **#61641 and #60927 (merged), #61343 (merged)** already optimised the per-element conversion in these kernels.
- **#62514 (open)** "Single-pass register-resident rms_norm_per_block_quant kernel": a different, fused kernel.
- **#64906 (open)** keeps fp32 activation scales on the Triton FP8 experts with a fused SiLU+quant path; it concerns the
  Triton experts, not the TRT-LLM path used here.
- I found no open PR that changes the launch geometry of the fp32-scale register kernel. The TRT-LLM experts code has a
  `TODO: fuse into the quant kernel` comment, so a fused path may be planned.

## W7b — the copy after the per-step logits all-gather

**Answer: the change is already proposed upstream. #65436 (open) "Gather vocab-parallel logits into their final layout
in one pass" does exactly this, but only for models that opt in (MiniMax-M3 / M3.1). Qwen3.8 (`Qwen4Exp`) is not opted
in. The remaining work is to opt it in once #65436 lands. My single-GPU numbers put the saving at about 150 us per
step at 256 requests (about 0.8 % of a TEP4 decode step) and about 25 us at 64.**

### Call site

`qwen3_vl.py::compute_logits` -> `models/qwen4_exp/nvidia/model.py::compute_logits` -> `LogitsProcessor.forward` ->
`_get_logits` -> `_gather_logits` (`model_executor/layers/logits_processor.py`, around line 122) ->
`tensor_model_parallel_all_gather(logits)` with the default `dim=-1` -> `CudaCommunicator.all_gather`.

- What is gathered: the vocab-parallel LM head's logits, one `[rows, 62,080]` bf16 shard per rank
  (vocab 248,320 / TP4), once per step.
- NCCL fills a rank-major buffer `[4 x rows, 62,080]`. Then `reshape(4, rows, 62080).movedim(0, 1).reshape(rows, 248320)`
  has to copy, because the target is vocab-major. That is the `aten::copy_` seen in W5.
- **The next consumer copies everything again:** `v1/worker/gpu/sample/sampler.py::apply_sampling_params` does
  `torch.empty_like(logits, dtype=torch.float32).copy_(logits)` whenever any request needs logits processing. In W5
  this was the 175 us `unrolled_elementwise_kernel` at 256 requests.

So each sampled step moves the logits three times: gather, layout copy, FP32 copy.

### Cost of the copies (one B200, real shapes, 4 shards x rows x 62,080 bf16; microseconds)

| rows | bf16 MB | layout copy (today) | FP32 cast (today) | both, as today | one fused pass (layout + cast) | saving | layout only, as 4 slice copies |
|---|---|---|---|---|---|---|---|
| 1 | 0.5 | 2.1 | 4.7 | 5.1 | 4.7 | 0.4 | 8.3 |
| 16 | 7.6 | 16.7 | 14.3 | 26.5 | 16.4 | 10 | 19.7 |
| 64 | 30.3 | 49.3 | 46.2 | 83.2 | 58.4 | 25 | 49.1 |
| 128 | 60.6 | 102 | 89 | 189 | 113 | 77 | 86 |
| 256 | 121 | 198 | 175 | 371 | 223 | **148** | 156 |
| 512 | 243 | 389 | 346 | 734 | 444 | 290 | 299 |

- Cold-L2 numbers. In the W5 model profile the layout copy was 153 us at 256 rows and 33 us at 64, and the cast 175 us
  at 256, so the in-model layout copy is somewhat cheaper than here and the saving is at most what the table shows.
- "One fused pass" is `out_fp32.view(rows, 4, 62080).copy_(gathered.view(4, rows, 62080).movedim(0, 1))`: plain PyTorch,
  bit-identical to today's two steps.
- When no request needs logits processing the sampler skips the FP32 copy. Then only the layout copy remains, and doing
  it as four slice copies is about 20 % cheaper from 128 rows up (156 vs 198 us at 256), slower below 64.
- Two runs agree within 3 %.

### What to change (local sketch only, not applied)

Either of these, in order of preference:

1. **Wait for #65436 and opt `Qwen4Exp` in.** It gathers along rows (no copy), then writes the final `[rows, vocab]`
   layout with one Triton kernel that also casts to the dtype the sampler asks for through the forward context. That
   removes both of today's extra passes where the sampler wants FP32.
2. **If #65436 stalls:** a small local version for the default path. Let `_gather_logits` return the rank-major buffer
   plus its shape, and let the sampler's existing FP32 copy do the layout change in the same pass (the fused expression
   above). No new kernel; saves the layout copy when the FP32 copy happens anyway.

### Estimated saving at TEP4 (not measured end to end)

- 256 requests: up to 148 us of a 17.0 ms step, about 0.8 %.
- 64 requests: about 25 us of a 10.9 ms step, about 0.2 %.
- Larger with MTP verification, where more rows go through the head per step; not measured.

### Duplicate search

- **#65436 (open, not draft, last updated 2026-11-18)**: the same idea, opt-in, MiniMax-M3 only. This is the duplicate.
- **#66639 (open, draft)** changes what the sampler does after the logits exist; it does not remove these copies.
- **#63890 (merged)** uses AITER's all-gather on ROCm; unrelated to CUDA.

## W7c addendum — the gap is the element conversion, and a bit-identical 2x is available at large N

**This corrects my W7c guess. Launch geometry makes no difference. The fp32-scale register kernel still converts each
element with the software `c10::Float8_e4m3fn` cast, while the packed kernel uses the hardware two-values-per-`cvt`
path. Giving the register kernel the fp32 form of that conversion makes it 2.07x faster at 8,192 tokens with
bit-identical output. It is a 6-line C++ change, so it needs a build; I tested it in a standalone program.**

### Method

- A standalone nvcc program (`-arch=sm_100a`, CUDA 13.1) that includes the two kernels **verbatim**, cut from
  `csrc/quantization/w8a8/fp8/per_token_group_quant.cu` at 8399da3780 by line range, with the c10 bf16/fp8 headers from
  the installed torch. No vLLM file was edited and nothing was rebuilt in the vLLM tree.
- Two local variants of the register kernel that differ only in the quantize loop.
- Timing: 400-2,000 back-to-back launches between CUDA events, rotating over 4 buffer sets (about 250 MB at 8,192
  tokens). This includes launch overhead and is not the CUPTI cold-L2 method; the unmodified kernel reads 26.6 us here
  against 25.6 us with CUPTI, so the two agree at large N. Below about 256 tokens this timer floors at about 4 us and
  cannot resolve differences.
- Same result on two GPUs (5 and 6).

### Results (microseconds per launch)

| N | register kernel as is, 256 threads/block | 128 | 64 | 512 | **variant A: fp32 math, `__nv_cvt_float2_to_fp8x2`** | variant B: bf16 math + bf16x2 cvt (as the packed kernel) | packed kernel (verbatim) |
|---|---|---|---|---|---|---|---|
| 256 | 4.10 | 4.10 | 4.10 | 4.10 | 4.05 | 4.05 | 4.06 |
| 2,048 | 8.33 | 8.61 | 10.24 | 10.24 | **6.10 (1.37x)** | 4.10 | 6.14 |
| 8,192 | 26.62 | 26.62 | 26.62 | 26.62 | **12.85 (2.07x)** | 12.33 | 14.35 |

- **Block size 64 / 128 / 256 / 512 changes nothing** at 8,192 tokens and only hurts at 2,048. Launch geometry is not it.
- **Variant A is bit-identical to the current kernel**: 0 differing output bytes out of 20,971,520 at 8,192 tokens, and
  identical scales. The test data includes values scaled by 1000, exact zeros, an all-zero group, groups scaled by
  1e-4, values near ±448, ±inf, NaN, 3e38 and 1e-30.
- **Variant B is not bit-identical** with fp32 scales (about 3 % of bytes differ), because it multiplies in bf16. That
  explains why the merged two-per-`cvt` work could use it only where the scale is a power of two.
- Variant A is also faster than the packed kernel (12.85 vs 14.35 us).

### The change (described; not applied to the vLLM tree)

In `per_token_group_quant_8bit_register_kernel`, replace the per-element loop that clamps, casts through `DST_DTYPE(q)`
and packs one byte at a time with a loop over pairs, under the existing `VLLM_FP8_CVT_X2` gate:

```cpp
for (int i = 0; i < VEC_SIZE; i += 2) {
  const float s0 = fminf(fmaxf(float(regs[i]) * inv_y, min_8bit), max_8bit);
  const float s1 = fminf(fmaxf(float(regs[i + 1]) * inv_y, min_8bit), max_8bit);
  packed[i >> 2] |= uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(s0, s1), __NV_SATFINITE, __NV_E4M3)) << ((i & 3) * 8);
}
```

### What it is worth

- Kernel: 2.07x at 8,192 tokens, 1.37x at 2,048; no measurable change at decode sizes, where the kernel is
  launch-bound.
- Step, estimated from W5 shares, not measured end to end: 8k prefill about 1.2 % -> about 0.6 % (a 0.6 % gain);
  2k prefill about 0.3 %.
- int8 output and fp16 input take the same template; I tested bf16 -> e4m3 only. The int8 path must keep the old loop.

### Duplicate search

- **#61641 (merged 2026-11-02)** "Convert two FP8 values per cvt in the FP8 quant kernels" introduced
  `__nv_cvt_float2_to_fp8x2` and the `VLLM_FP8_CVT_X2` gate for the shared conversion helpers and the packed kernel. It
  states that the software cast and the hardware `cvt` agree on every non-NaN fp32 input, which matches my result.
- **#55330 (merged)** added the register kernel. Its quantize loop was not converted: on upstream `main` today the
  register kernel is byte-for-byte the same as at 8399da3780, and the file has a single `VLLM_FP8_CVT_X2` use, in the
  packed kernel.
- I found no open PR that applies the two-per-`cvt` conversion to the fp32-scale register kernel.

## Extra (worker's own, negative result): launch parameters of the hyper-connection Triton kernels

`_hc_combine_norm_kernel`, `_hc_gate_mix_kernel` and `_hc_silu_kernel` (`models/qwen4_exp/nvidia/ops/hc.py`) together
take about 9-14 % of a TEP4 step (W5) and are launched with a fixed `BLOCK_SIZE = 512` and Triton's default
`num_warps`. I swept them the same way as W7a (real launchers, kernel object wrapped at launch, CUPTI, CUDA graph,
cold L2, one idle B200): `num_warps` in {1, 2, 4, 8} x `BLOCK_SIZE` in {256, 512, 1024, 4096} (silu: warps only), for
N in {1, 8, 64, 256, 2,048, 8,192} with the real shapes (4 streams x 2,560, bf16).

| kernel | N = 1 | 8 | 64 | 256 | 2,048 | 8,192 | best config vs default |
|---|---|---|---|---|---|---|---|
| `_hc_combine_norm_kernel`, default us | 3.01 | 2.98 | 3.33 | 5.06 | 23.9 | 84.8 | at most 1.03x |
| `_hc_gate_mix_kernel`, default us | 2.02 | 2.21 | 2.91 | 4.74 | 18.8 | 64.5 | at most 1.03x |
| `_hc_silu_kernel`, default us | 1.76 | 1.73 | 2.02 | 2.02 | 3.46 | 9.82 | at most 1.02x |

**Nothing to gain: the defaults are within 3 % of the best config at every size**, which is at the level of the
measurement noise (about 1 %). No change proposed. (For `_hc_combine_norm_kernel`, other `BLOCK_SIZE` values also change
the result slightly, up to 3e-2 max abs, because the sum of squares is reduced in a different order.)

A related audit of the C++ kernels found one more scalar FP8 cast of the same kind as in the W7c addendum: the
shared-memory group-quant kernel (`QuantizeGroup`, same file, line 91). It is only used when the register path cannot
be, so it matters less.
