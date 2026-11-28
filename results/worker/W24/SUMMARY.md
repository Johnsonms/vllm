# W24: decode gap of the MXFP8 MLP prototype, and the GSM8K sign

2026-11-28 18:43Z. Qwen3-8B-FP8, TP1, B200, same tree and prototype as W23 (threshold 16).

## 1. Decode gap: explained. The vLLM arm of my kernel benches was too slow.
**In every kernel bench of W21 and W23 the vLLM side ran a slower activation-quant path than serving does.** I built `QuantFP8(...)` in a bare script; there it did not take the packed-kernel branch. Per quant + GEMM pair (`benchbias.py`, torch profiler):

| | kernels per call |
|---|---|
| serving, and the corrected bench (`benchbias2.py`) | `per_token_group_quant_8bit_packed_register_kernel` 2.1 us + DeepGEMM |
| my W21 / W23 benches | two Inductor quant kernels 2.6 + 1.9 us + DeepGEMM `pack_fp32_into_ue8m0` 2.9 us + DeepGEMM |

That is +5.3 us of kernel time and two extra launches per GEMM on the vLLM side, about +10.6 us per MLP. It is a constant, so it matters at small M and is small at prefill sizes.

**Corrected warm sweep** (`multilayer2.py`, `multi2.jsonl`: whole MLP, 2 layers in one CUDA graph, no L2 flush, best QuACK config per M, us per layer):

| M | vLLM (corrected) | prototype | speedup | W23 table said |
|---|---|---|---|---|
| 8 | 39.5 | 52.3 | 0.76 | 0.89 |
| 16 | 41.6 | 44.5 | 0.94 | 1.13 |
| 32 | 42.0 | 42.0 | 1.00 | 1.23 |
| 64 | 42.9 | 42.6 | 1.01 | 1.27 |
| 128 | 47.1 | 42.1 | 1.12 | 1.38 |
| 256 | 56.6 | 46.1 | 1.23 | 1.44 |
| 512 | 81.3 | 62.9 | 1.29 | 1.19 |
| 1024 | 133.1 | 121.3 | 1.10 | 1.20 |
| 2048 | 274.5 | 209.4 | 1.31 | 1.28 |
| 4096 | 579.1 | 415.1 | 1.40 | 1.17 |

The large-M rows moved by more than the fix explains (run-to-run and config choice), so read them as "about 1.1-1.4x from M = 512". **The decode expectation is now about 0 % at c32 and small at c128, which is what E2E showed.**

**Serve profiles, base vs prototype on the same GPU, per step (us):**

| | c32 base | c32 proto | c128 base | c128 proto |
|---|---|---|---|---|
| kernel launches | 564 | 492 | 564 | 492 |
| sum of kernel time | 3818 | 3924 | 5433 | 5427 |
| MLP kernels per layer | 41.5 | 44.9 | 44.9 | 45.1 |
| - DeepGEMM gate_up + down | 35.0 | - | 37.0 | - |
| - packed quant x2 | 4.5 | - | 5.1 | - |
| - silu-mul | 1.9 | - | 2.8 | - |
| - QuACK gated + quant-out | - | 23.1 | - | 23.2 |
| - QuACK down GEMM | - | 18.7 | - | 19.0 |
| - FlashInfer mxfp8 quant | - | 3.1 | - | 2.9 |
| unprofiled TPOT ms (3 bursts, one server at a time) | 4.134 | 4.169 | 6.623 | 6.312 |

- In serving the prototype's MLP kernels are **8 % slower at c32 and equal at c128**. No copy or layout op was added: the prototype path is exactly 3 kernels per layer, attention / norm / LM head are unchanged within 1 %.
- The c128 TPOT gain (-4.7 % here, -2.8 % paired) therefore does **not** come from kernel time. The prototype has 72 fewer launches per step and the gaps between kernels shrink by 137 us per step at c128. I did not verify the mechanism further. At c32 this single-server run shows +0.8 %; the paired run showed -1.4 %. Treat c32 as no change.
- QuACK's kernels at M = 32-128 are flat at 23 and 19 us; DeepGEMM is at 17-19 us per GEMM. QuACK only wins here by removing the two quant kernels and the silu kernel.

**Consequences**
- **Threshold should be about 128-256, not 16.** The prototype is a prefill and high-concurrency optimisation: -4..-8 % TTFT at 4096 tokens is real (E2E, 35 of 36 rounds).
- **W21 needs the same correction.** Its vLLM arm had the same +5.3 us per GEMM. The M <= 128 speedups there are overstated (the gate_up chain "1.12-1.44x from M = 8" is closer to no gain below M = 128); M >= 512 is affected by a few percent only. I added a warning at the top of the W21 and W23 summaries. A full re-run of the W21 tables with the packed quant is about 30 minutes if you want it.
- B300 / Hopper nodes that copied my `bench.py` pattern (building `QuantFP8` outside a server) should check which quant kernels their baseline launches.

## 2. GSM8K sign: not the weight re-quantization
- **The prototype's weights are the weights vLLM already runs.** On SM100 vLLM rounds the checkpoint's 128x128 scales to powers of two at load (`requant_weight_ue8m0_inplace`). Against the checkpoint values that changes the weights by **2.66 %** relative (all 108 MLP matrices, 2.64-2.68 %). The prototype's MXFP8 weights have the same 2.66 % error, and compared with vLLM's re-quantized weights they are equal in 99.98-99.99 % of elements, relative difference 2.5e-7 to 6e-7 (`werr.py`, `werr.txt`).
- So the two arms differ only in the activations: scale granularity (1x32 instead of 1x128, finer), h quantized from the fp32 accumulator instead of from BF16, and summation order.
- **Greedy, 64 single prompts, 128 tokens, base vs prototype forced on for all M** (`greedy.py`): each arm is deterministic (64 of 64 identical on repeat), and base vs prototype is identical on **5 of 64**; the median first differing token is 13. The outputs diverge early, as for any change in FP8 rounding.
- Conclusion: nothing in the numerics says the prototype is less accurate (kernel-level error vs FP32 is equal, weights are equal). The -0.4 points (5-6 questions of 1,319) is a deterministic difference between two equally valid roundings, inside one run's sampling error. GSM8K cannot separate these arms; a larger or different eval would be needed to claim "no loss". I did not run one.
- Side note for the landing plan: vLLM's own power-of-two scale rounding on SM100 already moves the weights 2.66 % away from the checkpoint. That is existing behaviour, not ours.
