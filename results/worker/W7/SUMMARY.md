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
