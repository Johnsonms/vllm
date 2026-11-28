# B16 — QuACK softmax / top-k for the sampler on SM103 (B300)

Setup: vLLM `14e9c30d86`, quack-kernels 0.6.5, FlashInfer 0.7.4, 1x B300, vocab 151936 (Qwen3), Qwen3 defaults top_k 20 / top_p 0.95 / temperature 0.6. Timing: `bench_gpu_time_with_cupti` (CUDA graph, cold L2), median. Data: `mb16.jsonl`, `fix_chain.jsonl`.

## Verdict
- QuACK softmax is **6-13x faster than torch softmax at batch <= 16** (fp32), but **not usable as is** in the sampler. It returns **NaN on top-k-masked logits** (any fill <= -1e10, including the -inf that `top_k_mask_logits` writes). It also returns **all-NaN for bf16/fp16 input at vocab ~151-152K** (fp32 is fine; 4K-131K and 248320 are fine in every dtype). These are two QuACK bugs worth reporting upstream (owner's call).
- With a workaround (replace -inf by -1e6, one extra elementwise kernel), the whole top-k/top-p chain is **1.51x / 1.42x / 1.36x at B = 1 / 4 / 16**, 1.08x at 64, **0.85x at 256**. Seeded top-p tokens match the current path in 100% of draws (probs are not bit-exact: max |Δ| ≤ 2.7e-7).
- **But FlashInfer's own `top_k_max` fast path** (one top-k kernel, no full-vocab softmax) is already **1.30x at B = 1** and **1.49-2.94x at B >= 4**. Open PRs #64116 / #65649 go after it, so the QuACK route only wins at B = 1-ish and is dominated elsewhere. **Recommendation: no QuACK sampler candidate; push the existing top_k_max PRs instead.**
- **QuACK top-k cannot run at this vocab:** `TopK` asserts N is a power of two and N <= 4096.

## 1. Where the softmax is and why it is slow at batch 1
- vLLM's `flashinfer_sample` (`vllm/v1/sample/ops/topk_topp_sampler.py:335`) calls `flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, k, p)` with a per-row `k` tensor and no `top_k_max`. FlashInfer then skips its fast path and runs `top_k_mask_logits` → **`torch.softmax(masked_logits)`** → `top_p_sampling_from_probs` (`flashinfer/sampling.py`, `filter_apply_order == "top_k_first"`). Top-k only and top-p only call `logits.softmax(..., dtype=float32)` in vLLM directly.
- `torch.softmax` on the last dim launches `cunn_SoftMaxForward` with **one thread block per row**. At batch 1, one SM reduces all 151936 elements (two passes plus a write), about 49 µs regardless of batch up to 64. QuACK splits a row across a thread-block cluster (up to 16 CTAs with DSMEM reduction), giving 4.6 µs at B = 1.

## 2. Microbench (fp32 logits; vLLM's sampler logits are fp32)
| B | torch softmax fp32 (µs) | QuACK softmax fp32 | speedup | bf16 in: torch / QuACK | FI top_k_mask_logits | FI top_p_from_probs | temperature div |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 49.28 | 4.58 | **10.76x** | 45.44 / 3.49 (**NaN output**) | 37.92 | 38.11 | 2.05 |
| 4 | 50.08 | 5.06 | **9.9x** | 46.53 / 3.71 (**NaN output**) | 41.18 | 66.08 | 2.91 |
| 16 | 51.33 | 8.03 | **6.39x** | 52.22 / 6.11 (**NaN output**) | 42.11 | 67.55 | 5.38 |
| 64 | 54.63 | 18.4 | **2.97x** | 71.17 / 14.72 (**NaN output**) | 80.54 | 86.24 | 13.82 |
| 256 | 112.96 | 55.26 | **2.04x** | 211.55 / 45.54 (**NaN output**) | 231.43 | 135.2 | 48.16 |

**Whole sampler chain** (fp32, k = 20, p = 0.95):

| B | current (`top_k_top_p_sampling_from_logits`) | mask -> finite fill -> QuACK softmax -> top-p | speedup | FlashInfer `top_k_max=20` fast path | speedup | seeded top-p tokens equal (QuACK chain vs current probs) |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 139.39 | 92.38 | 1.51x | 106.88 | 1.30x | 200/200 |
| 4 | 162.4 | 114.02 | 1.42x | 109.18 | 1.49x | 800/800 |
| 16 | 170.94 | 125.6 | 1.36x | 109.25 | 1.56x | 3200/3200 |
| 64 | 226.85 | 210.98 | 1.08x | 119.81 | 1.89x | 12800/12800 |
| 256 | 514.72 | 603.91 | 0.85x | 175.3 | 2.94x | 51200/51200 |

## 3. Exactness
- QuACK softmax fp32 vs torch: not bit-exact (max |Δ| 5e-9 to 5e-7), argmax identical (greedy unaffected).
- Seeded top-p (`top_p_sampling_from_probs`, same generator seed, 200 seeds per batch row) from QuACK probs (finite-fill workaround) vs torch probs: **identical tokens in every draw** at every batch.
- Without the workaround: NaN probs, so 0% agreement.

## 4. Fusing temperature + softmax (+ top-k)
QuACK 0.6.5 `softmax_fwd` has no scale/temperature argument (and its top-k cannot handle this vocab), so it cannot be one QuACK kernel without changing QuACK. The temperature divide is 2-14 µs at B <= 64 (48 µs at 256). #54417 (open) already fuses temperature into the sampler softmax.

## 5. Open vLLM PRs on this path (checked 2026-11-28; read only)
#64116 (shared top_k as an int to FlashInfer, MRV2), #65649 (FlashInfer top-k selection for top-k/top-p masking), #54417 (fuse temperature into the sampler softmax), #51526 (sample in log space, skip the full-vocab softmax, native path), #66792 (exact Triton top-k/top-p, ~3x), #66630 (CuTe DSL top-k/top-p filter), #66521 (CUDA top-k/top-p masking kernel for Hopper/Blackwell), #67750 (fused in-register top-k/top-p for seeded requests), #60993 (FlashInfer Cake sampling). This area is crowded; a QuACK-softmax PR would overlap #54417 / #51526 / #65649.

No GitHub actions; code local.
